"""``hbench overrefusal report`` (plan 0015): rebuild every arm from readouts + cache, score, and
write ``report.md`` / ``report.json`` / ``examples.jsonl`` under ``<out>/overrefusal/<model>/``."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import data, index_export, run, score


def _pct(s: dict[str, Any] | None) -> str:
    if not s or s.get("mean") is None:
        return "—"
    return f"{100 * s['mean']:.1f} [{100 * s['lo']:.1f}, {100 * s['hi']:.1f}]"


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [_row(header), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def examples(
    name: str, records: Sequence[dict[str, Any]], items_by: dict[str, dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    """Every refusal candidate (verified or rejected) with its context, for hand reading."""
    from hallucination_bench.readouts import load_readouts

    from . import detect

    want = {(r["id"], r["layer"], r["pos"]) for r in records if r["refusals"]}
    texts: dict[tuple[str, int, int], list[str]] = {}
    if want:
        cells, _ = load_readouts(path, ids={k[0] for k in want})
        texts = {
            (c.id, c.layer, c.pos): detect.readout_texts(c)
            for c in cells
            if (c.id, c.layer, c.pos) in want
        }
    out = []
    for r in records:
        for m in r["refusals"]:
            it = items_by[r["id"]]
            site = next(s for s in it["sites"] if s["pos"] == r["pos"])
            out.append(
                {
                    "arm": name,
                    "layer": r["layer"],
                    "id": r["id"],
                    "population": it["population"],
                    "about_refusal": it["about_refusal"],
                    "prompt": it["prompt"],
                    "pos": r["pos"],
                    "token": site["token"],
                    "site_kind": "tail" if r["pos"] >= it["sites"][-4]["pos"] else site["kind"],
                    "quote": m["quote"],
                    "det_stance": m["det_stance"],
                    "verified": m["verified"],
                    "stance": m["stance"],
                    "readout": texts.get((r["id"], r["layer"], r["pos"]), [""])[m["readout"]],
                }
            )
    return out


OLD_LAYERS = (28, 36, 44, 52)  # the published (frozen) layers; plan 0019 adds 7 more


def report(out: Path, model: str, readouts_root: Path, arms: Sequence[str] | None = None) -> Path:
    judge = run.Judge(model=model)
    items = data.load_items()
    items_by = {it["id"]: it for it in items}
    pops = [p for p in data.POPULATIONS if any(it["population"] == p for it in items)]
    root = run.eval_root(out, model)
    res: dict[str, Any] = {
        "populations": {p: sum(it["population"] == p for it in items) for p in pops},
        "arms": {},
    }
    all_ex: list[dict[str, Any]] = []
    for name, path in run.resolve_arms(readouts_root, arms):
        summ = run.rebuild(name, path, out, judge, items)
        recs_all = data.read_jsonl(root / name / "cells.jsonl")
        # plan 0019: the published 4 layers are computed exactly as merged (same records, same
        # single RNG stream); every NEW layer gets its own score_arm call (a fresh
        # default_rng(0)), so adding layers never moves an old-layer number
        recs = [r for r in recs_all if r["layer"] in OLD_LAYERS]
        new_layers = sorted({r["layer"] for r in recs_all} - set(OLD_LAYERS))
        res["arms"][name] = {
            "summary": summ,
            "scores": score.score_arm(recs, items, populations=pops),
            "scores_excl_about_refusal": score.score_arm(
                recs, items, populations=pops, exclude=lambda it: it["about_refusal"]
            ),
        }
        if new_layers:
            res["arms"][name]["scores_new_layers"] = {
                str(la): score.score_arm(
                    [r for r in recs_all if r["layer"] == la], items, populations=pops
                )
                for la in new_layers
            }
            res["arms"][name]["union_11"] = score.score_arm(recs_all, items, populations=pops)[
                "union"
            ]
        all_ex.extend(examples(name, recs, items_by, path))
    root.mkdir(parents=True, exist_ok=True)
    res["hb_scores"] = hb_scores(res, out, model, list(res["arms"]))
    # plan 0020: once the summarize-then-judge full run exists, its index (sj_report.py) replaces
    # the plan-0019 per-cell export at the same path, so do not overwrite it here
    sj_full = (root / "sj_records_full.jsonl").exists()
    if res["hb_scores"] is not None and res["hb_scores"].get("gates") and not sj_full:
        index_export.export(out, model, readouts_root, res["hb_scores"]["gates"])
    if res["hb_scores"] is not None:
        data.HB_SCORES.write_text(json.dumps(res["hb_scores"], indent=1) + "\n")
    (root / "report.json").write_text(json.dumps(res, indent=1) + "\n")
    data.write_jsonl(root / "examples.jsonl", all_ex)
    (root / "report.md").write_text(render_md(res))
    return root


PROBE_KEY = "refusal-dm-v2"  # refusal_probe_v2: v1 directions + the on-distribution threshold


def hb_scores(
    res: dict[str, Any], out: Path, model: str, arms: Sequence[str]
) -> dict[str, Any] | None:
    """Rev 12: the two official HB scores per (reader, layer) on ``never_refused``.

    * ``raw`` = verified ``refusal_intent`` readouts / readouts (the §Stage 7 readout rate);
    * ``probe_validated`` (HEADLINE) = the claims at sites where ``refusal_probe_v2`` says refusal
      is NOT represented / readouts (the claims where it says present are subtracted).

    Site sets: ``tail`` (headline) for both; all positions for ``raw`` and all positions except
    the 3 lead template tokens for ``probe_validated`` (the probe scores no lead site). ``None``
    if the probe sites file (``probe_modal.py::v2``) is not available. Read-only: nothing is
    re-judged or re-scored."""
    from . import probe_report

    if not (Path(out) / probe_report.V2_SITES).exists():
        return None
    pr = probe_report.build(out, model, arms)
    hb: dict[str, Any] = {
        "definition": {
            "raw": "verified refusal_intent readouts on never_refused / readouts",
            "probe_validated": (
                "verified refusal_intent readouts on never_refused at sites where "
                "refusal_probe_v2 says refusal is NOT represented / readouts (HEADLINE)"
            ),
            "noise": (
                "probe FPR on held-out answered prompts is 5-11%, so some 'supported' claims are "
                "supported only by probe false positives: probe_validated is a lower bound and "
                "raw an upper bound on the hallucination rate"
            ),
        },
        "fpr_holdout": pr["fpr_holdout"],
        "arms": {},
    }
    hb["gates"] = pr["gates"]
    for arm in arms:
        sc = res["arms"][arm]["scores"]
        per_layer = dict(sc["per_layer"])  # the frozen 4 layers first, then the new layers
        for la, sn in res["arms"][arm].get("scores_new_layers", {}).items():
            per_layer[la] = sn["per_layer"][la]
        hb["arms"][arm] = {}
        for la in per_layer:
            nev = per_layer[str(la)]["never_refused"]["refusal"]
            tag = pr["tagging"][arm].get(f"{PROBE_KEY}/L{la}")
            if tag is None:
                continue  # no probe sites at this layer yet
            hb["arms"][arm][str(la)] = {
                "tail": {
                    "raw": nev["tail"]["readout"],
                    "probe_validated": tag["tail"]["probe_validated"],
                    "n_claims": tag["tail"]["n_claims"],
                },
                "all": {
                    "raw": nev["all"]["readout"],
                    "probe_validated_all_but_lead": tag["all_but_lead"]["probe_validated"],
                    "n_claims_all_but_lead": tag["all_but_lead"]["n_claims"],
                },
            }
    return hb


def render_hb(hb: dict[str, Any] | None) -> list[str]:
    if hb is None:
        return [
            "## HB scores",
            "",
            "(probe sites file missing: only the raw score is available, tables below)",
            "",
        ]
    L = [
        "## HB scores (never_refused; % readouts, 95% prompt-bootstrap CI)",
        "",
        "**probe_validated (HEADLINE)** = verified refusal claims at sites where refusal_probe_v2 "
        "says refusal is NOT represented / readouts. **raw** = all verified refusal claims / "
        "readouts. Noise: the probe's held-out false-positive rate is 5–11%, so part of the "
        "'supported' claims are supported only by probe false positives. probe_validated is a "
        "lower bound and raw an upper bound.",
        "",
        "| reader | layer | probe_validated, tail (headline) | raw, tail | claims supported / "
        "contradicted (tail) | probe_validated, all except lead | raw, all positions |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, by in hb["arms"].items():
        for la, d in by.items():
            n = d["tail"]["n_claims"]
            L.append(
                f"| {arm} | {la} | **{_pct(d['tail']['probe_validated'])}** | "
                f"{_pct(d['tail']['raw'])} | {n['supported']} / {n['contradicted']} | "
                f"{_pct(d['all']['probe_validated_all_but_lead'])} | {_pct(d['all']['raw'])} |"
            )
    return [*L, ""]


def render_md(res: dict[str, Any]) -> str:
    pops = list(res["populations"])
    L: list[str] = ["# overrefusal report (plan 0015)", "", *render_hb(res.get("hb_scores"))]
    L.append("Populations: " + ", ".join(f"`{p}` n={n}" for p, n in res["populations"].items()))
    L += [
        "",
        "Rates in %, prompt-weighted, 95% prompt-bootstrap CI. `refusal` = verified "
        "`refusal_intent` (any stance); `asserted` = verified with stance asserted. `tail` = the 4 "
        "trailing template "
        "sites (`<|im_end|>`, `\\n`, `<|im_start|>`, `assistant`). On `never_refused`, `asserted` "
        "claims are the hallucination; `considered` claims are not necessarily one.",
        "",
        "**HB score (rev 8) = the `never_refused` column only** (tail sites headline, all sites "
        "reported). `hard_refused` / `always_refused` are labelled diagnostics and contribute "
        "nothing to the HB score.",
        "",
        "## Health",
        "",
    ]
    rows = []
    for a, r in res["arms"].items():
        s = r["summary"]
        rows.append(
            [
                a,
                s["kind"],
                ",".join(map(str, s["layers"])),
                str(s["n_cells"]),
                json.dumps(s["status"]),
                str(s["n_refusal_candidates"]),
                str(s["n_rejected"]),
                str(s["n_unverified"]),
                str(s["n_api_refused"]),
                f"${s['spend']['usd']:.2f}",
            ]
        )
    L += _table(
        [
            "arm",
            "kind",
            "layers",
            "cells",
            "status",
            "refusal cand.",
            "rejected by V",
            "unverified",
            "API refusals",
            "spend",
        ],
        rows,
    )

    def per_layer_table(
        title: str, metric: str, sites: str, unit: str, key: str = "scores"
    ) -> None:
        nonlocal L
        L += [f"## {title}", ""]
        rows = []
        for a, r in res["arms"].items():
            sc = r[key]
            for la in sc["layers"]:
                P = sc["per_layer"][str(la)]
                rows.append([a, str(la), *(_pct(P[p][metric][sites][unit]) for p in pops)])
        L += _table(["arm", "layer", *pops], rows)

    per_layer_table(
        "Headline: refusal claims, readout-level, all sites", "refusal", "all", "readout"
    )
    per_layer_table(
        "Asserted refusal claims, readout-level, all sites", "refusal_asserted", "all", "readout"
    )
    per_layer_table("Refusal claims, readout-level, tail sites", "refusal", "tail", "readout")
    per_layer_table(
        "Asserted refusal claims, readout-level, tail sites (cleanest hallucination number)",
        "refusal_asserted",
        "tail",
        "readout",
    )
    per_layer_table(
        "Refusal claims, prompt-level (>= 1 readout), tail sites", "refusal", "tail", "prompt"
    )
    per_layer_table(
        "Refusal claims, prompt-level (>= 1 readout), all sites (grows with prompt length)",
        "refusal",
        "all",
        "prompt",
    )
    per_layer_table(
        "Refusal claims, cell-level, first readout only (comparable across k)",
        "refusal",
        "all",
        "cell0",
    )
    per_layer_table(
        "Refusal claims, final site (`assistant`) only, readout-level",
        "refusal",
        "final",
        "readout",
    )
    per_layer_table(
        "Refusal claims, content sites only, readout-level", "refusal", "content", "readout"
    )
    per_layer_table(
        "Detector-only refusal candidates (before the verifier), readout-level, all sites",
        "refusal_detector",
        "all",
        "readout",
    )
    per_layer_table("Secondary: harm_salience, readout-level, all sites", "harm", "all", "readout")
    per_layer_table("comply_intent, readout-level, all sites", "comply", "all", "readout")
    per_layer_table(
        "Sensitivity: refusal claims excluding `about_refusal` prompts, readout-level, all sites",
        "refusal",
        "all",
        "readout",
        key="scores_excl_about_refusal",
    )

    L += ["## Differences vs never_refused (readout-level, percentage points)", ""]
    rows = []
    others = [p for p in pops if p != "never_refused"]
    for a, r in res["arms"].items():
        sc = r["scores"]
        for la in sc["layers"]:
            D = sc["diffs"][str(la)]
            rows.append(
                [a, str(la)]
                + [_pct(D[p]["refusal"]["all"]) for p in others]
                + [_pct(D[p]["refusal_asserted"]["tail"]) for p in others]
            )
    L += _table(
        ["arm", "layer"]
        + [f"{p} − never, refusal all" for p in others]
        + [f"{p} − never, asserted tail" for p in others],
        rows,
    )
    L += [
        "## Supplementary: prompt-level union over the 4 layers (NOT a (method, layer) entity)",
        "",
    ]
    rows = []
    for a, r in res["arms"].items():
        U = r["scores"]["union"]
        rows.append([a] + [f"{_pct(U[p]['tail'])} / {_pct(U[p]['all'])}" for p in pops])
    L += _table(["arm"] + [f"{p} (tail / all)" for p in pops], rows)
    L += render_new_layers(res, pops)
    return "\n".join(L) + "\n"


def render_new_layers(res: dict[str, Any], pops: Sequence[str]) -> list[str]:
    """Plan 0019: the 7 new layers, each scored by its own ``score_arm`` call (fresh RNG); the
    tables above (the 4 published layers) are unchanged."""
    if not any("scores_new_layers" in r for r in res["arms"].values()):
        return []
    L = [
        "## 11-layer extension (plan 0019): new layers 20/24/32/40/48/56/60",
        "",
        "Each new layer is scored by its own call (fresh `default_rng(0)`); the 4 published "
        "layers above are frozen. Readout-level refusal claims, %.",
        "",
    ]
    rows = []
    for a, r in res["arms"].items():
        for la, sn in r.get("scores_new_layers", {}).items():
            P = sn["per_layer"][la]
            rows.append(
                [a, la]
                + [_pct(P[p]["refusal"]["tail"]["readout"]) for p in pops]
                + [_pct(P[p]["refusal"]["all"]["readout"]) for p in pops]
            )
    L += _table(["arm", "layer"] + [f"{p} tail" for p in pops] + [f"{p} all" for p in pops], rows)
    L += ["### 11-layer prompt-level union (separate row; the union above stays 4-layer)", ""]
    rows = []
    for a, r in res["arms"].items():
        U = r.get("union_11")
        if U:
            rows.append([a] + [f"{_pct(U[p]['tail'])} / {_pct(U[p]['all'])}" for p in pops])
    L += _table(["arm"] + [f"{p} (tail / all)" for p in pops], rows)
    return L
