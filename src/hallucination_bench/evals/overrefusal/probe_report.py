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
from .probe_data import PROBE_DIR, VARIANTS

SITES = Path("overrefusal") / "probe_sites.jsonl"
ARTIFACT_JSON = PROBE_DIR / "refusal_probe_v1.json"
TAIL = ("tail3", "tail2", "tail1", "tail0")  # <|im_end|>, \n, <|im_start|>, assistant
POPS = data.POPULATIONS
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


def build(out: Path, judge: str, arms: Sequence[str]) -> dict[str, Any]:
    items = data.load_items()
    by_pop = {p: [it for it in items if it["population"] == p] for p in POPS}
    pop_of = {it["id"]: it["population"] for it in items}
    sites = load_sites(Path(out) / SITES)
    prov = json.loads(ARTIFACT_JSON.read_text())
    res: dict[str, Any] = {
        "validity": {},
        "transfer": {},
        "premise": {},
        "tagging": {},
        "misses": {},
    }
    for (v, la), S in sorted(sites.items()):
        key = f"{v}/L{la}"
        res["validity"][key] = prov["cells"][key]
        # transfer (site z; tail offsets)
        T: dict[str, Any] = {}
        for cls in TAIL:
            z = {p: [] for p in POPS}
            pid = {p: [] for p in POPS}
            for (i, _), r in S.items():
                if r["cls"] == cls:
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
    # exploratory (not pre-registered): dm-v1 with the threshold re-calibrated on prompts Qwen
    # answered (80k 0/10); same directions, same gate, presence column swapped
    cal_path = Path(out) / "overrefusal" / "probe_sites_calib.jsonl"
    if cal_path.exists():
        for line in cal_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            base = sites.get((r["variant"], r["layer"]), {}).get((r["id"], r["pos"]))
            if base is None:
                continue
            ck = (CALIB_TAG, r["layer"])
            sites.setdefault(ck, {})[(r["id"], r["pos"])] = {**base, "present": r["present_calib"]}
        for la in {k[1] for k in sites if k[0] == CALIB_TAG}:
            G[f"{CALIB_TAG}/L{la}"] = G[f"refusal-dm-v1/L{la}"]
    for (v, la), S in sorted(sites.items()):
        key = f"{v}/L{la}"
        P: dict[str, Any] = {}
        for cls in (*TAIL, "content"):
            P[cls] = {}
            for p, its in by_pop.items():
                vals = []
                for it in its:
                    pres = [
                        r["present"] >= 0.5
                        for (i, _), r in S.items()
                        if i == it["id"] and r["cls"] == cls
                    ]
                    if pres:
                        vals.append(float(any(pres)))
                P[cls][p] = _ci(np.array(vals))
        res["premise"][key] = P
    for arm in arms:
        recs = data.read_jsonl(run.eval_root(out, judge) / arm / "cells.jsonl")
        by = defaultdict(list)
        for r in recs:
            by[(r["id"], r["layer"])].append(r)
        for v in (*VARIANTS, CALIB_TAG):
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
    into probe-supported / unsupported / ungated; plus the supported share of claims."""
    out: dict[str, Any] = {}
    for scope in ("tail", "all"):
        cols = {k: [] for k in ("original", "unsupported", "supported", "ungated")}
        n_claims = {"supported": 0, "unsupported": 0, "ungated": 0}
        for it in by_pop["never_refused"]:
            acc = {k: [] for k in cols}
            for r in by.get((it["id"], la), []):
                cls = _site_info(it, r["pos"])
                if cls == "lead" or (scope == "tail" and not cls.startswith("tail")):
                    continue
                if r["status"] not in ("ok", "accepted_partial", "empty"):
                    continue
                s = S.get((it["id"], r["pos"]))
                g = bool(s) and gate[cls]
                present = bool(s) and s["present"] >= 0.5
                for f in r["ro"]:
                    c = f["ref"]
                    acc["original"].append(c)
                    acc["unsupported"].append(c and g and not present)
                    acc["supported"].append(c and g and present)
                    acc["ungated"].append(c and not g)
                    if c:
                        n_claims[
                            "ungated" if not g else ("supported" if present else "unsupported")
                        ] += 1
            if acc["original"]:
                for k in cols:
                    cols[k].append(float(np.mean(acc[k])))
        tot = sum(n_claims.values())
        gated = n_claims["supported"] + n_claims["unsupported"]
        out[scope] = {
            **{k: _ci(np.array(v)) for k, v in cols.items()},
            "n_claims": n_claims,
            "supported_share_of_gated": (n_claims["supported"] / gated) if gated else None,
            "n_claims_total": tot,
        }
    return out


def _misses(by, S, gate, its, la) -> dict[str, Any]:
    """always_refused: share of gated probe-present sites with no refusal claim in any readout."""
    out: dict[str, Any] = {}
    for scope in ("tail", "all"):
        n_present = n_miss = 0
        for it in its:
            for r in by.get((it["id"], la), []):
                cls = _site_info(it, r["pos"])
                if cls == "lead" or (scope == "tail" and not cls.startswith("tail")):
                    continue
                s = S.get((it["id"], r["pos"]))
                if not s or not gate[cls] or s["present"] < 0.5:
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
    n = r["n_claims"]["supported"] + r["n_claims"]["unsupported"]
    return "—" if sh is None else f"{100 * sh:.0f}% of {n}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def render_md(res: dict[str, Any]) -> str:
    L = ["# overrefusal probe report (plan 0015 rev 5-6)", ""]
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
            _p(r["original"]),
            _p(r["unsupported"]),
            _p(r["supported"]),
            _p(r["ungated"]),
            _share(r),
        ]
        for arm, T in res["tagging"].items()
        for key, sc in T.items()
        for scope, r in sc.items()
    ]
    head = ["arm", "probe/layer", "scope", "original", "unsupported (probe absent)", "supported"]
    L += _table([*head, "ungated", "supported share of gated claims"], rows)
    L += ["## Misses on always_refused (probe-present gated sites, no lens claim)", ""]
    rows = [
        [arm, key, _miss(m["tail"]), _miss(m["all"])]
        for arm, T in res["misses"].items()
        for key, m in T.items()
    ]
    L += _table(["arm", "probe/layer", "tail: n present, miss %", "all: n present, miss %"], rows)
    return "\n".join(L) + "\n"


def report(out: Path, judge: str, arms: Sequence[str]) -> Path:
    res = build(out, judge, arms)
    root = run.eval_root(out, judge)
    (root / "probe_report.json").write_text(json.dumps(res, indent=1) + "\n")
    (root / "probe_report.md").write_text(render_md(res))
    return root
