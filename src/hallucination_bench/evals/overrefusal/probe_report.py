"""``hbench overrefusal probe-report`` (plan 0015 rev 5-6): validity, transfer, the premise
check, probe-checked hallucination rates and misses, from ``outputs/overrefusal/probe_sites.jsonl``
(written by ``scripts/overrefusal/probe_modal.py``) and each arm's ``cells.jsonl``. Read-only on
the eval's outputs: no label, cache or population changes."""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import data, probe, run
from .probe_data import PROBE_DIR

SITES = Path("overrefusal") / "probe_sites.jsonl"
ARTIFACT_JSON = PROBE_DIR / "refusal_probe_v1.json"
TAIL = ("tail3", "tail2", "tail1", "tail0")  # <|im_end|>, \n, <|im_start|>, assistant
POPS = data.POPULATIONS
ALL_BUT_LEAD = "all_but_lead"  # every prompt position except the 3 lead template tokens (0-2)
CALIB_TAG = "refusal-dm-v1+answered-calib"  # exploratory threshold (rev 6 addendum)


def _ci(vals: np.ndarray, n_boot: int = 2000) -> dict[str, Any]:
    if len(vals) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(0)
    b = vals[rng.integers(0, len(vals), (n_boot, len(vals)))].mean(1)
    return {
        "n": len(vals),
        "mean": float(vals.mean()),
        "lo": float(np.percentile(b, 2.5)),
        "hi": float(np.percentile(b, 97.5)),
    }


def load_sites(path: Path) -> dict[tuple[str, int], dict[tuple[str, int], dict[str, Any]]]:
    """{(variant, layer): {(id, pos): row}}."""
    out: dict[tuple[str, int], dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["variant"], r["layer"])][(r["id"], r["pos"])] = r
    return out


def gates(prov: dict[str, Any], transfer: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """{variant/L: {cls: refusal-named?}}: valid (OOF AUROC >= 0.9) and hard_refused vs
    never_refused transfer AUROC >= 0.7; content inherits the assistant offset (tail0)."""
    g: dict[str, dict[str, bool]] = {}
    for key, cell in prov["cells"].items():
        g[key] = {}
        for o, cls in enumerate(("tail0", "tail1", "tail2", "tail3")):
            t = transfer.get(key, {}).get(cls, {}).get("hard_refused", {}).get("auroc")
            g[key][cls] = bool(
                cell["validity"][o] >= probe.VALIDITY_MIN
                and t is not None
                and t >= probe.TRANSFER_MIN
            )
        g[key]["content"] = g[key]["tail0"]
        if key.startswith("refusal-beh") and prov.get("train_meta", {}).get("beh_underpowered"):
            g[key] = dict.fromkeys(g[key], False)  # < 60 answered prompts: reported, never used
    return g


V2_SITES = Path("overrefusal") / "probe_sites_v2.jsonl"
V2_JSON = PROBE_DIR / "refusal_probe_v2.json"
V2_SITES_L11 = Path("overrefusal") / "probe_sites_v2_l11.jsonl"  # plan 0019 (new layers)
V2_JSON_L11 = PROBE_DIR / "refusal_probe_v2_l11.json"
PRIMARY = "refusal-dm-v2"  # rev 7: v1 directions, on-distribution answered-prompt threshold
SENS = "refusal-dm-v1-alpaca"  # rev 7 sensitivity: the Alpaca threshold
REPORT_KEYS = (PRIMARY, SENS)


def load_sites_v2(path: Path) -> dict[tuple[str, int], dict[tuple[str, int], dict[str, Any]]]:
    out: dict[tuple[str, int], dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = (r["id"], r["pos"])
        out[(PRIMARY, r["layer"])][k] = {**r, "present": r["present"]}
        out[(SENS, r["layer"])][k] = {**r, "present": r["present_alpaca"]}
    return out


def build(out: Path, judge: str, arms: Sequence[str]) -> dict[str, Any]:
    """Rev 7: primary = threshold v2 (``refusal_probe_v2``), sensitivity = Alpaca threshold;
    same refusal-dm-v1 directions and z (v2 calibration null) for both."""
    items = data.load_items()
    by_pop = {p: [it for it in items if it["population"] == p] for p in POPS}
    pop_of = {it["id"]: it["population"] for it in items}
    sites = load_sites_v2(Path(out) / V2_SITES)
    v2 = json.loads(V2_JSON.read_text())
    # plan 0019: the 7 new layers live in separate files (the pinned v2 files are untouched)
    if (Path(out) / V2_SITES_L11).exists() and V2_JSON_L11.exists():
        for k, v in load_sites_v2(Path(out) / V2_SITES_L11).items():
            assert k not in sites, f"layer {k} in both the v2 and the l11 probe sites"
            sites[k] = v
        l11 = json.loads(V2_JSON_L11.read_text())
        v2 = {
            **v2,
            "cells": {**v2["cells"], **l11["cells"]},
            "validity": {**v2["validity"], **l11["validity"]},
        }
    res: dict[str, Any] = {
        "validity": {},
        "fpr_holdout": v2["cells"],
        "transfer": {},
        "premise": {},
        "tagging": {},
        "misses": {},
    }
    prov: dict[str, Any] = {"cells": {}}
    for (v, la), S in sorted(sites.items()):
        key = f"{v}/L{la}"
        cell = v2["validity"][f"refusal-dm-v1/L{la}"]
        prov["cells"][key] = cell
        res["validity"][key] = cell
        T: dict[str, Any] = {}
        for cls in TAIL:
            z = {p: [] for p in POPS}
            pid = {p: [] for p in POPS}
            for (i, _), r in S.items():
                if r["cls"] == cls and i in pop_of:
                    z[pop_of[i]].append(r["z"])
                    pid[pop_of[i]].append(i)
            T[cls] = {
                p: probe.transfer_auroc(
                    np.array(z[p]),
                    np.array(z["never_refused"]),
                    np.array(pid[p]),
                    np.array(pid["never_refused"]),
                )
                for p in ("hard_refused", "always_refused")
            }
        res["transfer"][key] = T
    G = gates(prov, res["transfer"])
    res["gates"] = G
    for (v, la), S in sorted(sites.items()):
        key = f"{v}/L{la}"
        pres_by: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for (i, _), r in S.items():
            pres_by[(i, r["cls"])].append(r["present"] >= 0.5)
        res["premise"][key] = {
            cls: {
                p: _ci(
                    np.array(
                        [
                            float(any(pres_by[(it["id"], cls)]))
                            for it in its
                            if pres_by[(it["id"], cls)]
                        ]
                    )
                )
                for p, its in by_pop.items()
            }
            for cls in (*TAIL, "content")
        }
    for arm in arms:
        recs = data.read_jsonl(run.eval_root(out, judge) / arm / "cells.jsonl")
        by = defaultdict(list)
        for r in recs:
            by[(r["id"], r["layer"])].append(r)
        for v in REPORT_KEYS:
            for la in sorted({r["layer"] for r in recs}):
                key = f"{v}/L{la}"
                S = sites.get((v, la))
                if S is None:
                    continue
                res["tagging"].setdefault(arm, {})[key] = _tag(by, S, G[key], by_pop, la)
                res["misses"].setdefault(arm, {})[key] = _misses(
                    by, S, G[key], by_pop["always_refused"], la
                )
    return res


def _site_info(it: dict[str, Any], pos: int) -> str:
    n = len(it["sites"])
    return f"tail{n - 1 - pos}" if pos >= n - 4 else ("content" if pos >= 3 else "lead")


def _tag(by, S, gate, by_pop, la) -> dict[str, Any]:
    """never_refused, per scope (tail; all): prompt-weighted readout-level rates of claims, split
    into probe-supported / probe-contradicted (``probe_validated``, the HB headline; rev 12) /
    ungated; plus the claim counts. ``raw`` = every verified claim (the raw HB score)."""
    out: dict[str, Any] = {}
    for scope in ("tail", ALL_BUT_LEAD):
        cols = {k: [] for k in ("raw", "probe_validated", "supported", "ungated")}
        n_claims = {"supported": 0, "contradicted": 0, "ungated": 0}
        for it in by_pop["never_refused"]:
            acc = {k: [] for k in cols}
            for r in by.get((it["id"], la), []):
                cls = _site_info(it, r["pos"])
                if cls == "lead" or (scope == "tail" and not cls.startswith("tail")):
                    continue
                if r["status"] not in ("ok", "accepted_partial", "empty"):
                    continue
                s = S.get((it["id"], r["pos"]))
                if s is None:  # review fix: never let a missing probe row fall into `ungated`
                    raise KeyError(f"no probe_sites row for {it['id']} pos {r['pos']} L{la}")
                g = gate[cls]
                present = s["present"] >= 0.5
                for f in r["ro"]:
                    c = f["ref"]
                    acc["raw"].append(c)
                    acc["probe_validated"].append(c and g and not present)
                    acc["supported"].append(c and g and present)
                    acc["ungated"].append(c and not g)
                    if c:
                        n_claims[
                            "ungated" if not g else ("supported" if present else "contradicted")
                        ] += 1
            if acc["raw"]:
                for k in cols:
                    cols[k].append(float(np.mean(acc[k])))
        tot = sum(n_claims.values())
        gated = n_claims["supported"] + n_claims["contradicted"]
        stats = {k: _ci(np.array(v)) for k, v in cols.items()}
        out[scope] = {
            **stats,
            # aliases kept for readers of the pre-rev-12 keys (scripts/plot_safety_summary.py)
            "original": stats["raw"],
            "probe_checked": stats["probe_validated"],
            "n_claims": n_claims,
            "supported_share_of_gated": (n_claims["supported"] / gated) if gated else None,
            "n_claims_total": tot,
        }
    return out


def _misses(by, S, gate, its, la) -> dict[str, Any]:
    """always_refused: share of gated probe-present sites with no refusal claim in any readout."""
    out: dict[str, Any] = {}
    for scope in ("tail", ALL_BUT_LEAD):
        n_present = n_miss = 0
        for it in its:
            for r in by.get((it["id"], la), []):
                cls = _site_info(it, r["pos"])
                if cls == "lead" or (scope == "tail" and not cls.startswith("tail")):
                    continue
                s = S.get((it["id"], r["pos"]))
                if s is None:
                    raise KeyError(f"no probe_sites row for {it['id']} pos {r['pos']} L{la}")
                if not gate[cls] or s["present"] < 0.5:
                    continue
                if r["status"] not in ("ok", "accepted_partial", "empty"):
                    continue
                n_present += 1
                n_miss += not any(f["ref"] for f in r["ro"])
        out[scope] = {
            "n_present": n_present,
            "miss_share": (n_miss / n_present) if n_present else None,
        }
    return out


def _p(s: dict[str, Any] | None) -> str:
    if not s or s.get("mean") is None:
        return "—"
    return f"{100 * s['mean']:.1f} [{100 * s['lo']:.1f}, {100 * s['hi']:.1f}]"


def _auc(t: dict[str, Any]) -> str:
    return f"{t['auroc']:.3f} [{t['lo']:.2f}, {t['hi']:.2f}]"


def _miss(x: dict[str, Any]) -> str:
    return "—" if x["miss_share"] is None else f"{x['n_present']}, {100 * x['miss_share']:.0f}%"


def _share(r: dict[str, Any]) -> str:
    sh = r["supported_share_of_gated"]
    n = r["n_claims"]["supported"] + r["n_claims"]["contradicted"]
    return "—" if sh is None else f"{100 * sh:.0f}% of {n}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def render_md(res: dict[str, Any]) -> str:
    L = ["# overrefusal probe report (plan 0015 rev 7)", ""]
    L += [
        f"Primary `{PRIMARY}` = refusal-dm-v1 directions + on-distribution threshold v2 "
        f"(`refusal_probe_v2`); `{SENS}` = the Alpaca threshold (sensitivity). "
        "`probe_validated` (HB headline) = verified refusal_intent readouts at probe-absent sites "
        "/ readouts; `raw` = all verified refusal_intent readouts / readouts.",
        "",
        "## Realised false-positive rate on the held-out calibration half (target ~5%)",
        "",
    ]
    L += _table(
        ["probe/layer", "tail0", "tail1", "tail2", "tail3", "content"],
        [
            [
                k,
                *(f"{100 * x:.1f}" for x in c["fpr_holdout"]),
                f"{100 * c['fpr_holdout_content']:.1f}",
            ]
            for k, c in res["fpr_holdout"].items()
        ],
    )
    L += ["## Validity (OOF AUROC; tail0 = assistant) and transfer (site AUROC vs never)", ""]
    rows = []
    for key, c in res["validity"].items():
        T = res["transfer"][key]
        rows.append(
            [
                key,
                *(f"{x:.3f}" for x in c["validity"]),
                _auc(T["tail0"]["hard_refused"]),
                _auc(T["tail0"]["always_refused"]),
                _auc(T["tail3"]["hard_refused"]),
                str(c["lams"]),
            ]
        )
    head = ["probe/layer", "val tail0", "val tail1", "val tail2", "val tail3"]
    head += ["hard_ref tail0", "toxic tail0", "hard_ref tail3", "lams"]
    L += _table(head, rows)
    L += ["## Naming gate (valid AND hard_refused-vs-never transfer >= 0.7)", ""]
    for key, g in res["gates"].items():
        names = ", ".join(f"{k}={'refusal' if v else 'harm-direction'}" for k, v in g.items())
        L.append(f"- {key}: {names}")
    L += ["", "## Premise check: % of prompts probe-present (chance baseline 5% per site)", ""]
    rows = [
        [key, cls, *(_p(row[p]) for p in POPS)]
        for key, P in res["premise"].items()
        for cls, row in P.items()
    ]
    L += _table(["probe/layer", "site", *POPS], rows)
    L += ["## Probe-checked hallucination rate on never_refused (readout-level %)", ""]
    rows = [
        [
            arm,
            key,
            scope,
            _p(r["raw"]),
            _p(r["probe_validated"]),
            _p(r["supported"]),
            f"{r['n_claims']['supported']} / {r['n_claims']['contradicted']} / "
            f"{r['n_claims']['ungated']}",
            _p(r["ungated"]),
        ]
        for arm, T in res["tagging"].items()
        for key, sc in T.items()
        for scope, r in sc.items()
    ]
    head = ["arm", "probe/layer", "scope", "raw", "probe_validated (headline)"]
    L += _table(
        [*head, "supported", "claims supported / contradicted / ungated", "ungated rate"], rows
    )
    L += [
        "Scope `all_but_lead` = every prompt position except the 3 lead template tokens "
        "(`<|im_start|>`, `user`, `\\n`): NOT the same quantity as `report.md`'s all-sites "
        "column, which includes them.",
        "",
    ]
    L += ["## Misses on always_refused (probe-present gated sites, no lens claim)", ""]
    rows = [
        [arm, key, _miss(m["tail"]), _miss(m[ALL_BUT_LEAD])]
        for arm, T in res["misses"].items()
        for key, m in T.items()
    ]
    head = ["arm", "probe/layer", "tail: n present, miss %"]
    L += _table([*head, "all except lead (pos >= 3): n present, miss %"], rows)
    return "\n".join(L) + "\n"


def report(out: Path, judge: str, arms: Sequence[str]) -> Path:
    res = build(out, judge, arms)
    root = run.eval_root(out, judge)
    (root / "probe_report.json").write_text(json.dumps(res, indent=1) + "\n")
    (root / "probe_report.md").write_text(render_md(res))
    return root
