"""Summarise the probe-recipe sweep (``probe_sweep_modal.py``) -> results.json + summary.md.

    uv run python scripts/sibling_probe/probe_sweep_summary.py [--dir outputs/sibling/probe_sweep]

Also checks V0 against the pipeline run (``outputs/sibling/probes/validity.jsonl`` and the
transfer AUROC recomputed by ``score.transfer_auroc`` from ``eval_scores.jsonl.gz``).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe.probe_io import (
    PROBES_DIR,
    iter_eval_scores,
    load_validity,
)
from hallucination_bench.evals.sibling_probe.score import transfer_auroc

VARIANTS = ("V0", "V1", "V2", "V3", "V4")
NAMES = {
    "V0": "current (text-mean DiffMean, raw)",
    "V1": "per-position DiffMean",
    "V2": "shrinkage-whitened DiffMean",
    "V3": "per-position + whitened",
    "V4": "logistic regression (ceiling ref)",
}


def nanmed(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def nanmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def share(x, t):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x >= t)) if x.size else float("nan")


def agg(pcs: list[dict]) -> dict:
    out = {}
    for k in ("auroc", "auroc_req", "auroc_tail"):
        v = [p[k] for p in pcs]
        out[k] = {
            "median": nanmed(v),
            "min": float(np.nanmin(v)),
            "ge85": share(v, 0.85),
            "ge70": share(v, 0.70),
        }
    for k in ("transfer", "transfer_offmean", "transfer_req", "transfer_tail"):
        v = [p[k] for p in pcs]
        out[k] = {"median": nanmed(v), "ge70": share(v, 0.70), "n": int(np.isfinite(v).sum())}
    for k in ("sens", "eval_fpr", "fpr_heldout", "fpr_neutral"):
        out[k] = nanmean([p[k] for p in pcs])
    return out


def f2(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.2f}"


def pct(x):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/sibling/probe_sweep")
    a = ap.parse_args()
    d = Path(a.dir)
    data = {}
    for p in sorted(d.glob("*.json")):
        if p.name in ("config.json", "results.json"):
            continue
        j = json.loads(p.read_text())
        data[j["category"]] = j
    cats = list(data)
    layers = data[cats[0]]["layers"]

    res: dict = {"per_cat_layer": {}, "per_cat_best": {}, "overall": {}, "hyper": {}, "check": {}}
    for v in VARIANTS:
        pool_all = []
        for cat in cats:
            for L in layers:
                r = data[cat]["results"][str(L)].get(v)
                if r is None:
                    continue
                pcs = list(r["per_concept"].values())
                pool_all += pcs
                res["per_cat_layer"].setdefault(v, {}).setdefault(cat, {})[str(L)] = agg(pcs)
                res["hyper"].setdefault(v, {}).setdefault(cat, {})[str(L)] = r["hyper"]
            pl = res["per_cat_layer"].get(v, {}).get(cat, {})
            if pl:
                bl = max(pl, key=lambda L: pl[L]["auroc"]["median"])
                res["per_cat_best"].setdefault(v, {})[cat] = {"layer": int(bl), **pl[bl]}
        if pool_all:
            res["overall"][v] = agg(pool_all)
            # per-offset AUROC medians (pooled over categories/layers/concepts)
            by_off = defaultdict(list)
            for p in pool_all:
                for o, x in p["auroc_by_offset"].items():
                    by_off[int(o)].append(x)
            res["overall"][v]["auroc_by_offset_median"] = {
                o: nanmed(x) for o, x in sorted(by_off.items())
            }
        # L44 slice
        p44 = [
            p
            for cat in cats
            for p in data[cat]["results"]["44"].get(v, {}).get("per_concept", {}).values()
        ]
        if p44:
            res["overall"].setdefault(v, {})["L44"] = agg(p44)

    # V0 check vs the pipeline run
    try:
        val = load_validity(
            (PROBES_DIR / "v0" if (PROBES_DIR / "v0").is_dir() else PROBES_DIR) / "validity.jsonl"
        )
        diffs = [
            abs(
                data[cat]["results"][str(L)]["V0"]["per_concept"][c]["auroc"]
                - val[(cat, c, L)]["auroc"]
            )
            for cat in cats
            for L in layers
            for c in data[cat]["concepts"]
        ]
        items = {it["id"]: it for it in B.load_items()}
        scores = {
            (r["id"], int(r["layer"]), int(r["pos"]), r["concept"]): (r["s"],)
            for r in iter_eval_scores(
                (PROBES_DIR / "v0" if (PROBES_DIR / "v0").is_dir() else PROBES_DIR)
                / "eval_scores.jsonl.gz"
            )
        }
        tr = transfer_auroc(scores, items, {c: data[c]["concepts"] for c in cats})
        tdiffs = [
            abs(data[cat]["results"][str(L)]["V0"]["per_concept"][c]["transfer"] - tr[(cat, c, L)])
            for cat in cats
            for L in layers
            for c in data[cat]["concepts"]
            if np.isfinite(tr[(cat, c, L)])
        ]
        res["check"] = {
            "V0_validity_max_abs_diff": float(max(diffs)),
            "V0_transfer_max_abs_diff": float(max(tdiffs)),
            "n": len(diffs),
        }
    except Exception as e:  # check is best-effort
        res["check"] = {"error": f"{type(e).__name__}: {e}"}

    (d / "results.json").write_text(json.dumps(res, indent=1))

    # ------------------------------------------------------------------ markdown
    M = ["# Probe-recipe sweep (plan 0003 diagnostics)", ""]
    M.append(
        "Same category-level folds as the pipeline (`probes.assign_folds`), layers "
        + ", ".join(map(str, layers))
        + ". Position-level OOF AUROC = c's positions vs the category's other concepts' positions. "
        "Transfer = eval-site AUROC of the fold-averaged score, c-gold items' cells vs other-gold "
        "items' cells (gold-named items not positives), pooled over cells. Calibration at the fold's "
        "tau_cat (95th pct over fold k's other-concept OOF positions): FPR on held-out other-concept "
        "positions (text split-half inside the fold), on neutral texts, and at eval cells of "
        "other-gold items (target 5%). Sensitivity = gold probe detection at its own items' cells."
    )
    M.append("")
    M.append(f"V0 check vs pipeline: `{json.dumps(res['check'])}`")
    M.append("")
    M.append("## Headline (all categories, all layers, every (concept, layer))")
    M.append("")
    M.append(
        "| variant | OOF AUROC median | ≥0.85 | ≥0.70 | req. median | tail median | transfer median | transfer ≥0.70 | FPR held-out | FPR neutral | FPR eval | gold sens. |"
    )
    M.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        o = res["overall"].get(v)
        if not o or "auroc" not in o:
            continue
        M.append(
            f"| {v} {NAMES[v]} | {f2(o['auroc']['median'])} | {pct(o['auroc']['ge85'])} | {pct(o['auroc']['ge70'])} | "
            f"{f2(o['auroc_req']['median'])} | {f2(o['auroc_tail']['median'])} | {f2(o['transfer']['median'])} | "
            f"{pct(o['transfer']['ge70'])} | {pct(o['fpr_heldout'])} | {pct(o['fpr_neutral'])} | {pct(o['eval_fpr'])} | {pct(o['sens'])} |"
        )
    M.append("")
    M.append("## Median OOF AUROC per category at the best layer (layer in brackets)")
    M.append("")
    M.append("| variant | " + " | ".join(cats) + " |")
    M.append("|---|" + "---|" * len(cats))
    for v in VARIANTS:
        b = res["per_cat_best"].get(v)
        if not b:
            continue
        M.append(
            f"| {v} | "
            + " | ".join(f"{f2(b[c]['auroc']['median'])} (L{b[c]['layer']})" for c in cats)
            + " |"
        )
    M.append("")
    for title, key in (
        ("Median OOF AUROC at L44", ("auroc", "median")),
        ("Min OOF AUROC over concepts at L44", ("auroc", "min")),
        ("Share of concepts ≥0.85 at L44", ("auroc", "ge85")),
        ("Median OOF AUROC at L44, request-sentence offsets (≥9)", ("auroc_req", "median")),
        ("Median OOF AUROC at L44, template-tail offsets (0..8)", ("auroc_tail", "median")),
        ("Median transfer AUROC at L44 (pooled cells)", ("transfer", "median")),
        ("Median transfer AUROC at L44 (mean over offsets)", ("transfer_offmean", "median")),
        ("Gold sensitivity at tau_cat, L44", ("sens", None)),
        ("FPR at eval cells (other-gold items), tau_cat, L44", ("eval_fpr", None)),
        ("FPR on neutral texts, tau_cat, L44", ("fpr_neutral", None)),
        ("FPR held-out other-concept positions, tau_cat, L44", ("fpr_heldout", None)),
    ):
        M.append(f"### {title}")
        M.append("")
        M.append("| variant | " + " | ".join(cats) + " |")
        M.append("|---|" + "---|" * len(cats))
        for v in VARIANTS:
            pl = res["per_cat_layer"].get(v)
            if not pl:
                continue
            cells = []
            for c in cats:
                x = pl[c]["44"][key[0]]
                x = x[key[1]] if key[1] else x
                cells.append(pct(x) if key[1] in ("ge85", None) else f2(x))
            M.append(f"| {v} | " + " | ".join(cells) + " |")
        M.append("")
    M.append("## Per category × layer (median OOF AUROC / share ≥0.85 / transfer median)")
    M.append("")
    for cat in cats:
        M.append(f"### {cat}")
        M.append("")
        M.append("| variant | " + " | ".join(f"L{L}" for L in layers) + " |")
        M.append("|---|" + "---|" * len(layers))
        for v in VARIANTS:
            pl = res["per_cat_layer"].get(v, {}).get(cat)
            if not pl:
                continue
            M.append(
                f"| {v} | "
                + " | ".join(
                    f"{f2(pl[str(L)]['auroc']['median'])} / {pct(pl[str(L)]['auroc']['ge85'])} / {f2(pl[str(L)]['transfer']['median'])}"
                    for L in layers
                )
                + " |"
            )
        M.append("")
    M.append("## Per-offset median OOF AUROC (pooled over categories, layers, concepts)")
    M.append("")
    offs = sorted(
        {o for v in VARIANTS for o in res["overall"].get(v, {}).get("auroc_by_offset_median", {})}
    )
    M.append("| variant | " + " | ".join(str(o) for o in offs) + " |")
    M.append("|---|" + "---|" * len(offs))
    for v in VARIANTS:
        bo = res["overall"].get(v, {}).get("auroc_by_offset_median")
        if bo:
            M.append(f"| {v} | " + " | ".join(f2(bo.get(o)) for o in offs) + " |")
    M.append("")
    M.append("## Chosen hyperparameters (per outer fold)")
    M.append("")
    for v in ("V2", "V3", "V4"):
        h = res["hyper"].get(v, {})
        flat = [x for cat in h.values() for hs in cat.values() for x in hs]
        vals, cnt = np.unique(flat, return_counts=True) if flat else ([], [])
        M.append(f"- {v}: " + ", ".join(f"{x:g}×{n}" for x, n in zip(vals, cnt, strict=True)))
    (d / "summary.md").write_text("\n".join(M) + "\n")
    print((d / "summary.md").read_text()[:6000])


if __name__ == "__main__":
    main()
