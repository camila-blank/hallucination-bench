"""``hbench overrefusal report`` (plan 0015): rebuild every arm from readouts + cache, score, and
write ``report.md`` / ``report.json`` / ``examples.jsonl`` under ``<out>/overrefusal/<model>/``."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import data, run, score


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
        recs = data.read_jsonl(root / name / "cells.jsonl")
        res["arms"][name] = {
            "summary": summ,
            "scores": score.score_arm(recs, items, populations=pops),
            "scores_excl_about_refusal": score.score_arm(
                recs, items, populations=pops, exclude=lambda it: it["about_refusal"]
            ),
        }
        all_ex.extend(examples(name, recs, items_by, path))
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(res, indent=1) + "\n")
    data.write_jsonl(root / "examples.jsonl", all_ex)
    (root / "report.md").write_text(render_md(res))
    return root


def render_md(res: dict[str, Any]) -> str:
    pops = list(res["populations"])
    L: list[str] = ["# overrefusal report (plan 0015)", ""]
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
    return "\n".join(L) + "\n"
