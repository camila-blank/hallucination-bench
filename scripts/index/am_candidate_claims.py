"""Exploratory: the agentic_misalignment index family re-scored on candidate claims only.

The index's AM value (``hallucination_bench.index``) is the unweighted share
``(false_claim + substitution_category + detail) / n_claims`` over EVERY claim segment of every
mapped cell. Plan 0008 maps a cell only because the detector located a misalignment mention in
it, but the map covers the whole readout, so most claims are about other content. Here a
**candidate claim** is a claim segment (``taxonomy_v2.is_claim``) whose ``[start, end)`` overlaps
a merged mention span of the same readout of the same cell (``min(end) - max(start) > 0``, the
rule of ``score._overlaps`` / ``mention_verdict``):

* (a) ``all``: the cell's located mentions (the map population: every detector mention, any
  actor / stance / item arm; J-lens: the ``:Dsum`` mentions, located in the summary the map
  segments too);
* (b) ``headline``: only the Stage-H-verified headline mentions (``headline`` in the arm's
  ``mentions.jsonl``, keyed ``(cell, readout, start, end)``).

Mention spans of one readout are merged with ``score.merge_intervals`` first (the counting unit);
overlap with a merged span is overlap with at least one of its mentions. The item population and
bootstrap are the index's: items registered in ``claim_rows`` order (first cell with segments),
items with no candidate claims kept as zero rows, ``claim_metric`` (2,000 draws, fresh
``default_rng(0)`` per row). Nothing is re-judged; nothing in the index changes.

Run (dev box; one arm streamed at a time)::

    PYTHONPATH=src python scripts/index/am_candidate_claims.py \
        --index /workspace/hb-index-v4-res/outputs/index/claude-opus-5-5/index.json \
        --out outputs/index/am_candidate/results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from hallucination_bench import index as ix
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.agentic_misalignment.score import merge_intervals

AM = ix.AM
VARIANTS = ("base", "all", "headline")
HALL = ("false_claim", "substitution_category", "detail")


def headline_spans(path: Path) -> set[tuple[str, int, int, int]]:
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if '"headline": true' not in line:
                continue
            r = json.loads(line)
            if r["headline"]:
                out.add((r["key"], r["readout"], r["start"], r["end"]))
    return out


def _hits(seg: dict[str, Any], spans: dict[int, list[tuple[int, int]]]) -> bool:
    return any(
        min(seg["end"], b) - max(seg["start"], a) > 0 for a, b in spans.get(seg["readout"], ())
    )


def arm_rows(
    cells: Path, headline: set[tuple[str, int, int, int]]
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Variant -> layer -> (n_items, N_COLS) matrices, items in ``claim_rows`` order."""
    acc: dict[str, dict[int, dict[str, np.ndarray]]] = {v: defaultdict(dict) for v in VARIANTS}
    stats: dict[str, Any] = {"n_cells": 0, "headline_seen": set()}
    with open(cells, encoding="utf-8") as fh:
        for line in fh:
            if '"segments"' not in line:
                continue
            rec = json.loads(line)
            segs = ix._segments(rec, AM)
            if not segs:
                continue
            stats["n_cells"] += 1
            by_r: dict[str, dict[int, list[tuple[int, int]]]] = {"all": {}, "headline": {}}
            for m in rec.get("mentions") or []:
                iv = (m["start"], m["end"])
                by_r["all"].setdefault(m["readout"], []).append(iv)
                if (rec["key"], m["readout"], *iv) in headline:
                    by_r["headline"].setdefault(m["readout"], []).append(iv)
                    stats["headline_seen"].add((rec["key"], m["readout"], *iv))
            merged = {v: {r: merge_intervals(ivs) for r, ivs in d.items()} for v, d in by_r.items()}
            vecs = {
                v: acc[v][rec["layer"]].setdefault(rec["id"], np.zeros(ix.N_COLS)) for v in VARIANTS
            }
            for s in segs:
                lf = tx.JUDGE_TO_LEAF.get(s["leaf"], s["leaf"])
                if not tx.is_claim(lf):
                    continue
                inside = {"base": True, **{v: _hits(s, merged[v]) for v in ("all", "headline")}}
                for v in VARIANTS:
                    if not inside[v]:
                        continue
                    vecs[v][ix.C_CLAIMS] += 1
                    if lf in ix.MAIN_LEAF_WEIGHTS:
                        vecs[v][ix._LEAF_COL[lf]] += 1
                        vecs[v][ix.C_WEIGHTED] += ix.MAIN_LEAF_WEIGHTS[lf]
            for v in ("base",):  # readout / segment columns, as claim_rows (base only)
                vecs[v][ix.C_READOUTS] += len({s["readout"] for s in segs})
                vecs[v][ix.C_SEGMENTS] += len(segs)
    out = {
        v: {L: np.array(list(items.values())) for L, items in sorted(acc[v].items())}
        for v in VARIANTS
    }
    return out, stats


def n_hall(mat: np.ndarray) -> int:
    t = mat.sum(0)
    return int(t[ix.C_FC] + t[ix.C_CAT] + t[ix.C_DET])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--index", type=Path, required=True, help="the hindex-v4 index.json baseline")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-crosscheck", action="store_true", help="skip the claim_rows re-read")
    p.add_argument("--print-only", action="store_true", help="re-print the tables of --out")
    a = p.parse_args()
    if a.print_only:
        print_md(json.loads(a.out.read_text(encoding="utf-8"))["rows"])
        return
    index = json.loads(a.index.read_text(encoding="utf-8"))
    assert index["version"] == "hindex-v4", index["version"]
    roots = [Path(r) for r in index["sources"][AM]]
    dirs = ix._arm_dirs(AM, roots)
    used = sorted({r["families"][AM]["arm"] for r in index["rows"] if AM in r["families"]})
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in used:
        d = dirs[arm]
        hl = headline_spans(d / "mentions.jsonl")
        rows, stats = arm_rows(d / ix.CELLS, hl)
        # headline mentions of mentions.jsonl outside mapped cells (map failed) are not on the
        # index either: count them
        assert stats["headline_seen"] <= hl, arm
        n_hl_unmapped = len(hl - stats["headline_seen"])
        if not a.no_crosscheck:
            ref = ix.claim_rows(d / ix.CELLS, AM)
            assert ref.keys() == rows["base"].keys(), arm
            for L in ref:
                assert np.array_equal(ref[L], rows["base"][L]), (arm, L)
        results[arm] = {}
        for L, base in rows["base"].items():
            rec: dict[str, Any] = {
                "n_items": len(base),
                "n_hall": {},
                "n_hl_unmapped_arm": n_hl_unmapped,
            }
            for v in VARIANTS:
                mat = rows[v][L]
                assert len(mat) == len(base), (arm, L, v)
                rec[v] = ix.claim_metric(mat, AM)
                rec["n_hall"][v] = n_hall(mat)
            results[arm][L] = rec
        print(
            f"{arm}: {stats['n_cells']} mapped cells, {len(hl)} headline mention spans "
            f"({n_hl_unmapped} in unmapped cells)",
            flush=True,
        )
        del rows

    # reproduce the index's AM value exactly, then the macros with the AM value swapped
    table = []
    for r in index["rows"]:
        if AM not in r["families"]:
            continue
        cur = r["families"][AM]
        res = results[cur["arm"]][r["layer"]]
        mine = res["base"]
        for k in ("v", "ci", "n", "n_items"):
            assert mine[k] == cur[k], (r["method"], r["layer"], k, mine[k], cur[k])
        row: dict[str, Any] = {
            "method": r["method"],
            "arm": cur["arm"],
            "layer": r["layer"],
            "am": {v: res[v] for v in VARIANTS},
            "n_hall": res["n_hall"],
            "macros": {},
        }
        for key, fams in (
            ("full", ix.FAMILIES),
            ("matched", ix.MATCHED),
            ("full_v3", ix.FAMILIES_V3),
            ("matched_v3", ix.MATCHED_V3),
        ):
            if key not in r:
                continue
            row["macros"][key] = {"base": r[key]}
            for v in ("all", "headline"):
                alt = res[v]
                if alt is None:
                    row["macros"][key][v] = None
                    continue
                per = {**r["families"], AM: {**cur, "v": alt["v"], "ci": alt["ci"]}}
                row["macros"][key][v] = ix.weighted_macro(per, fams)
            m = ix.weighted_macro(r["families"], fams)
            assert abs(m["v"] - r[key]["v"]) < 1e-12, (r["method"], r["layer"], key)
        table.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"baseline": str(a.index), "rows": table}, indent=1) + "\n")
    print(f"reproduced {len(table)} index AM rows exactly; wrote {a.out}")
    print_md(table)


def _p(r: dict[str, Any] | None) -> str:
    if r is None:
        return "—"
    return f"{100 * r['v']:.1f} [{100 * r['ci'][0]:.1f}, {100 * r['ci'][1]:.1f}]"


def print_md(table: list[dict[str, Any]]) -> None:
    print(
        "\n| method | layer | current | cand (a) all | cand (b) headline | n claims | n cand (a) "
        "| n cand (b) | hall. outside (a) | hall. outside (b) |"
    )
    print("|---|---:|---|---|---|---:|---:|---:|---:|---:|")
    for r in table:
        am, nh = r["am"], r["n_hall"]
        out = [
            f"{100 * (1 - nh[v] / nh['base']):.0f}%" if nh["base"] else "—" for v in VARIANTS[1:]
        ]
        n = [str(am[v]["n"]) if am[v] else "0" for v in VARIANTS]
        cells = [r["method"], str(r["layer"]), *(_p(am[v]) for v in VARIANTS), *n, *out]
        print("| " + " | ".join(cells) + " |")
    print("\n| method | layer | macro | current | with (a) | Δ (a) | with (b) | Δ (b) |")
    print("|---|---:|---|---|---|---:|---|---:|")
    for r in table:
        for key, m in r["macros"].items():
            if key.endswith("_v3") and "matched" in r["macros"]:
                continue  # the v4 macros where they exist, else the v3 ones (V4, mlnla64)
            b = m["base"]["v"]
            cells = [r["method"], str(r["layer"]), key, _p(m["base"])]
            for v in ("all", "headline"):
                cells += [_p(m[v]), f"{100 * (m[v]['v'] - b):+.1f}" if m[v] else "—"]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
