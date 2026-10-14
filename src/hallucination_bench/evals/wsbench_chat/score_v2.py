"""Claim-map (taxonomy v2) metrics: character-weighted shares, readout-level rates,
comparisons with v5c / verify, health; item-bootstrap 95% CIs (plan 0002, §Metrics).

The headline is the **verdict layer** (plan rev 13, ``taxonomy_v2.VERDICTS``): verdict shares
(character-weighted) and readout-level verdict rates; leaves and groups are the breakdown.

Which readouts count:

* a cell enters once its ``:M`` stage is settled: ``ok`` or ``failed`` (a failed cell's units
  are all ``failed``). Pending cells (not yet mapped) are left out and counted in health;
* **shares** (headline) are per readout, then averaged over readouts (each non-empty readout
  weighs the same; failed readouts are in, under the ``failed`` row). The 6 verdicts plus
  ``failed`` sum to 1; so do the 8 groups plus ``unknown_incorrect`` plus ``failed``, and the
  22 leaf rows. Segment-count shares are pooled over segments;
* **readout-level rates** are over readouts of ``ok`` cells only (a failed readout has no
  labels), with two denominators: all of them (``all``) and the v5c-specific ones
  (``specific``, plan 0001's ``hallucination_rate`` denominator);
* the **v5c comparison** is over the readouts that have both a v5c span verdict and an ``ok``
  map (``n_cmp``): in single-call runs that is every cell whose span result is cached, else the
  seeded 10% item subsample (``run.in_v5c_subsample``).

**Rev 21** (plan 0009): :class:`AccumulatorV2` takes an optional ``sibling_labels`` map
(segment key ``(cell key, readout idx, start, end)`` -> the sibling-gate unit's ``final``
status). :func:`slim` applies ``taxonomy_v2.effective_leaf`` to each ``ok`` segment's leaf and
alt view once, at load time, so gated siblings score as ``substitution_sibling_gated`` (the
fourth hallucination component); the agreement spans stay raw. ``None`` (no labels file) is
exact rev-20 behaviour.

Every ratio is a ratio of per-item sums with the item-level percentile bootstrap of plan 0001
(:func:`score.bootstrap_indices`, 2,000 draws, seed 0).
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.judges.v5c import SPECIFIC_CLASSES
from hallucination_bench.units import locate

from . import score

N_BOOT = score.N_BOOT
# rev 21: the scoring-only gated sibling row (taxonomy's ALL_LEAVES is untouched)
LEAF_ROWS: tuple[str, ...] = (*tx.ALL_LEAVES, tx.SIBLING_GATED)
GROUP_ROWS: tuple[str, ...] = tx.ROWS  # 8 groups + unknown_incorrect + failed
VERDICT_ROWS: tuple[str, ...] = tx.VERDICT_ROWS  # 6 verdicts + failed
HALLUCINATION_LEAVES = frozenset(
    {*tx.VERDICTS[tx.V_HALLUCINATION], tx.SIBLING_GATED, tx.UNKNOWN_INCORRECT}
)
# rev 21: four components (the gated sibling is scoring-only)
HALLUCINATION_COMPONENTS: tuple[str, ...] = (
    "false_claim",
    "detail",
    "substitution_category",
    tx.SIBLING_GATED,
)
# the judge-side rev-20 components: hedged_origin tests the raw judge leaf against these
REV20_HALLUCINATION_COMPONENTS: tuple[str, ...] = ("detail", "false_claim", "substitution_category")
assert set(REV20_HALLUCINATION_COMPONENTS) == set(tx.VERDICTS[tx.V_HALLUCINATION])

# readout-level verdict rates (headline): the readout has >= 1 segment with that verdict
VERDICT_FLAGS: tuple[str, ...] = (
    *(f"verdict_{v}" for v in tx.VERDICTS),
    "verdict_junk_only",  # every segment is junk (rev 18: non-claims and off-topic claims)
)

FLAGS: tuple[str, ...] = (
    *VERDICT_FLAGS,
    "hallucination",  # detail | false_claim | unknown_incorrect (hallucination_rate_v2)
    "hallucination_excl_unknown",  # sensitivity: unknown_incorrect not counted
    "hallucination_alt",  # sensitivity: disputed segments re-scored with their alt_leaf
    "false_claim",
    "cruxy_false_claim",
    "detail",
    "substitution_category",  # rev 20: the third hallucination component
    tx.SIBLING_GATED,  # rev 21: the fourth (scoring-only)
    "co_activation",
    "thinky",
    "correct",
    "nonclaim_or_junk_only",
)

# readout group for the v5c transition table: the first group present, in this order
TRANSITION_PRIORITY: tuple[str, ...] = (
    tx.HALLUCINATION,
    tx.UNKNOWN_INCORRECT,
    tx.CO_ACTIVATION,
    tx.SUPPLEMENTARY,
    tx.THINKY,
    tx.UNSETTLED,
    tx.CORRECT,
    tx.JUNK,
    tx.FAILED,
)

# verify status -> the v2 leaves that agree with it (plan 0002, §Inputs; mapped through the
# rev 13 renames and the rev 16 merges: the three correct leaves -> correct, the thinky and
# supplementary-unsettled leaves -> unverifiable, the non-claim leaves -> non_claim)
AGREEMENT: dict[str, frozenset[str]] = {
    "true": frozenset(tx.CORRECT_LEAVES),
    "unverifiable": frozenset({*tx.SUPPLEMENTARY_LEAVES, *tx.THINKY_LEAVES, "false_claim"}),
    "disputed": frozenset({*tx.INCORRECT_LEAVES, "supplementary_false", tx.UNKNOWN_INCORRECT}),
    "off_topic": frozenset({"ambiguous_topic", *tx.JUNK_LEAVES}),
}

HEALTH_V2: tuple[str, ...] = (
    "n_cells",
    "n_map_ok",
    "n_map_failed",
    "n_map_pending",
    "n_readouts",
    *cm.PARSE_COUNTS,
    "n_unlocated_established",
    # rev 19: hedged segments re-scored unverifiable, by the judge's original leaf
    "n_hedged_from_hallucination",
    "n_hedged_from_substitution",
    "n_hedged_from_correct",
    "n_hedged_from_other",
)


def hedged_origin(seg: dict[str, Any]) -> str | None:
    """The judge's leaf family of a rev-19 hedged remap (``None`` if not remapped)."""
    if not seg.get("hedged") or seg.get("leaf") != "unverifiable":
        return None
    judge = seg.get("raw_leaf")  # same criterion as n_hedged_remapped (a quote-check re-label
    if judge == "unverifiable":  # to unverifiable comes first and is not a hedged remap)
        return None
    if judge in REV20_HALLUCINATION_COMPONENTS:  # the raw judge leaf (rev 21: unchanged)
        return "hallucination"
    if judge in tx.SUBSTITUTION_LEAVES:  # substitution_sibling
        return "substitution"
    if judge in ("correct", "supplementary_true"):
        return "correct"
    return "other"


# ---------------------------------------------------------------- slim records

SegT = tuple[str, str, int, bool, str | None]  # (leaf, alt-view leaf, chars, cruxy, reason)
SegKey = tuple[str, int, int, int]  # (cell key, readout idx, start, end): the sibling-label join
JUNK_REASON_ROWS: tuple[str, ...] = (*tx.JUNK_REASONS, "unspecified")


@dataclass
class SlimReadoutV2:
    v5c_cls: str | None  # None: no v5c span verdict for this cell (not in the comparison)
    ok: bool  # the cell's map call succeeded
    segs: list[SegT]


@dataclass
class SlimCellV2:
    item: str
    layer: int
    map_status: str  # ok | failed | pending | none
    readouts: list[SlimReadoutV2]
    health: dict[str, int]
    agreement: list[tuple[str, str | None]]  # (verify status, majority v2 leaf | None)


def majority_leaf(spans: Iterable[tuple[int, int, str]], start: int, end: int) -> str | None:
    """The leaf covering most characters of ``[start, end)`` among ``(start, end, leaf)``
    spans (ties: the first in reading order); ``None`` if nothing overlaps."""
    cover: Counter = Counter()
    order: list[str] = []
    for a, b, lf in spans:
        ov = min(end, b) - max(start, a)
        if ov > 0:
            if lf not in cover:
                order.append(lf)
            cover[lf] += ov
    if not cover:
        return None
    best = max(cover.values())
    return next(lf for lf in order if cover[lf] == best)


def seg_key(rec: dict[str, Any], seg: dict[str, Any]) -> SegKey:
    return (rec["key"], int(seg["readout"]), int(seg["start"]), int(seg["end"]))


def scored_leaves(
    rec: dict[str, Any], seg: dict[str, Any], labels: dict[SegKey, str | None] | None
) -> tuple[str, str]:
    """(leaf, alt-view leaf) of one segment under rev 21. Only ``ok`` segments join their unit's
    status (disputed raw siblings form no unit and stay raw, alt view included)."""
    alt = cm.alt_leaf_view(seg)
    if not labels or seg.get("status") != "ok":
        return seg["leaf"], alt
    st = labels.get(seg_key(rec, seg))
    return tx.effective_leaf(seg["leaf"], st), tx.effective_leaf(alt, st)


def slim(rec: dict[str, Any], sibling_labels: dict[SegKey, str | None] | None = None) -> SlimCellV2:
    h = dict.fromkeys(HEALTH_V2, 0)
    st = (rec.get("stages") or {}).get("map") or "none"
    m = rec.get("map")
    if m is None:
        return SlimCellV2(rec["id"], rec["layer"], "none", [], h, [])
    h["n_cells"] = 1
    h[f"n_map_{st}"] = 1 if st in ("ok", "failed", "pending") else 0
    if st == "pending":
        return SlimCellV2(rec["id"], rec["layer"], st, [], h, [])
    for k, v in (m.get("counts") or {}).items():
        if k in h:
            h[k] += int(v)
    by_r: dict[int, list[dict[str, Any]]] = {}
    for s in m["segments"]:
        by_r.setdefault(s["readout"], []).append(s)
        if (o := hedged_origin(s)) is not None:
            h[f"n_hedged_from_{o}"] += 1
    readouts: list[SlimReadoutV2] = []
    agreement: list[tuple[str, str | None]] = []
    verdict = {v["idx"]: v for v in rec.get("verdict") or []}
    for idx in range(len(rec.get("samples") or [])):
        segs = by_r.get(idx, [])
        if not segs:
            continue
        v = verdict.get(idx)
        h["n_readouts"] += 1
        readouts.append(
            SlimReadoutV2(
                v["class"] if v is not None else None,
                st == "ok",
                [
                    (
                        *scored_leaves(rec, s, sibling_labels),
                        s["end"] - s["start"],
                        bool(s["cruxy"]),
                        s.get("reason"),
                    )
                    for s in segs
                ],
            )
        )
        quotes = v.get("quotes") if v is not None else None
        if st != "ok" or not quotes:
            continue
        text = rec["samples"][idx]
        for status in AGREEMENT:
            for q in quotes.get(status) or []:
                loc = locate(q, text)
                spans = [(s["start"], s["end"], s["leaf"]) for s in segs]
                agreement.append((status, majority_leaf(spans, *loc) if loc else None))
    return SlimCellV2(rec["id"], rec["layer"], st, readouts, h, agreement)


# ---------------------------------------------------------------- per-readout stats


def readout_group(leaves: Iterable[str]) -> str:
    groups = {tx.group_of(lf) for lf in leaves}
    return next((g for g in TRANSITION_PRIORITY if g in groups), tx.FAILED)


def readout_flags(r: SlimReadoutV2) -> dict[str, bool]:
    leaves = [s[0] for s in r.segs]
    groups = {tx.group_of(lf) for lf in leaves}
    verdicts = {tx.verdict_of(lf) for lf in leaves}
    return {
        **{f"verdict_{v}": v in verdicts for v in tx.VERDICTS},
        "verdict_junk_only": verdicts == {tx.V_JUNK},
        "hallucination": any(lf in HALLUCINATION_LEAVES for lf in leaves),
        "hallucination_excl_unknown": any(lf in HALLUCINATION_COMPONENTS for lf in leaves),
        "hallucination_alt": any(s[1] in HALLUCINATION_LEAVES for s in r.segs),
        "false_claim": "false_claim" in leaves,
        "cruxy_false_claim": any(s[0] == "false_claim" and s[3] for s in r.segs),
        "detail": "detail" in leaves,
        "substitution_category": "substitution_category" in leaves,
        tx.SIBLING_GATED: tx.SIBLING_GATED in leaves,
        "co_activation": tx.CO_ACTIVATION in groups,
        "thinky": tx.THINKY in groups,
        "correct": tx.CORRECT in groups,
        "nonclaim_or_junk_only": groups <= {tx.JUNK},
    }


def _columns() -> list[str]:
    cols = ["n_share_readouts", "n_segments", "n_rate_all", "n_rate_specific", "n_v5c_hal"]
    cols += [
        "n_cmp",
        "n_cmp_v5c_hal",
        "n_cmp_v2_hal",
        "n_cmp_v2_fc",
        "n_cmp_v2_det",
        "n_cmp_v2_cat",
        "n_cmp_v2_gated",
    ]
    cols += [f"cv_{v}" for v in VERDICT_ROWS] + [f"sv_{v}" for v in VERDICT_ROWS]
    cols += [f"cj_{r}" for r in JUNK_REASON_ROWS]
    cols += [f"cs_{lf}" for lf in LEAF_ROWS] + [f"cg_{g}" for g in GROUP_ROWS]
    cols += [f"acg_{g}" for g in GROUP_ROWS]
    cols += [f"sc_{lf}" for lf in LEAF_ROWS] + [f"sg_{g}" for g in GROUP_ROWS]
    cols += [f"ra_{f}" for f in FLAGS] + [f"rs_{f}" for f in FLAGS]
    return cols


COLUMNS: tuple[str, ...] = tuple(_columns())
_COL = {c: i for i, c in enumerate(COLUMNS)}


def readout_vector(r: SlimReadoutV2) -> np.ndarray:
    v = np.zeros(len(COLUMNS))
    total = sum(s[2] for s in r.segs)
    if total > 0:
        v[_COL["n_share_readouts"]] = 1
        for lf, alt, ch, _c, reason in r.segs:
            v[_COL[f"cs_{lf}"]] += ch / total
            v[_COL[f"cg_{tx.group_of(lf)}"]] += ch / total
            v[_COL[f"cv_{tx.verdict_of(lf)}"]] += ch / total
            if tx.group_of(lf) == tx.JUNK:
                why = reason if reason in tx.JUNK_REASONS else "unspecified"
                v[_COL[f"cj_{why}"]] += ch / total
            v[_COL[f"acg_{tx.group_of(alt)}"]] += ch / total
    for lf, _alt, _ch, _c, _r in r.segs:
        v[_COL["n_segments"]] += 1
        v[_COL[f"sc_{lf}"]] += 1
        v[_COL[f"sg_{tx.group_of(lf)}"]] += 1
        v[_COL[f"sv_{tx.verdict_of(lf)}"]] += 1
    if r.ok:
        spec = r.v5c_cls in SPECIFIC_CLASSES
        v[_COL["n_rate_all"]] = 1
        v[_COL["n_rate_specific"]] = spec
        v[_COL["n_v5c_hal"]] = spec and r.v5c_cls == "hallucinated"
        flags = readout_flags(r)
        for f, on in flags.items():
            v[_COL[f"ra_{f}"]] = on
            v[_COL[f"rs_{f}"]] = on and spec
        if r.v5c_cls is not None:
            v[_COL["n_cmp"]] = 1
            v[_COL["n_cmp_v5c_hal"]] = r.v5c_cls == "hallucinated"
            v[_COL["n_cmp_v2_hal"]] = flags[f"verdict_{tx.V_HALLUCINATION}"]
            v[_COL["n_cmp_v2_fc"]] = flags["false_claim"]
            v[_COL["n_cmp_v2_det"]] = flags["detail"]
            v[_COL["n_cmp_v2_cat"]] = flags["substitution_category"]
            v[_COL["n_cmp_v2_gated"]] = flags[tx.SIBLING_GATED]
    return v


# ---------------------------------------------------------------- ratios + bootstrap


@lru_cache(maxsize=8)
def _boot(n_items: int, n_boot: int) -> np.ndarray:
    return score.bootstrap_indices(n_items, n_boot)


def ratios(
    cells: Sequence[SlimCellV2], *, n_boot: int = N_BOOT
) -> tuple[dict[str, float], np.ndarray | None, int]:
    """Per-item column sums -> (totals, bootstrap draws of every column sum, n_items)."""
    per_item: dict[str, np.ndarray] = {}
    for c in cells:
        if c.map_status in ("none", "pending"):
            continue
        acc = per_item.setdefault(c.item, np.zeros(len(COLUMNS)))
        for r in c.readouts:
            acc += readout_vector(r)
    mat = np.array(list(per_item.values())) if per_item else np.zeros((0, len(COLUMNS)))
    totals = {c: float(mat[:, i].sum()) if len(mat) else 0.0 for i, c in enumerate(COLUMNS)}
    draws = mat[_boot(len(mat), n_boot)].sum(axis=1) if len(mat) else None
    return totals, draws, len(mat)


def _rate(totals: dict[str, float], draws: Any, num: str, den: str) -> dict[str, Any]:
    d = totals[den]
    ci = score._ci(draws[:, _COL[num]], draws[:, _COL[den]]) if draws is not None else None
    return {"value": totals[num] / d if d else None, "n": int(d), "ci95": ci}


def block(cells: Sequence[SlimCellV2], *, n_boot: int = N_BOOT) -> dict[str, Any]:
    totals, draws, n_items = ratios(cells, n_boot=n_boot)
    health = dict.fromkeys(HEALTH_V2, 0)
    for c in cells:
        for k, v in c.health.items():
            health[k] += v

    def r(num: str, den: str) -> dict[str, Any]:
        return _rate(totals, draws, num, den)

    return {
        "n_items": n_items,
        "n_share_readouts": int(totals["n_share_readouts"]),
        "n_segments": int(totals["n_segments"]),
        "n_rate_readouts": {
            "all": int(totals["n_rate_all"]),
            "specific": int(totals["n_rate_specific"]),
        },
        "char_share_verdict": {v: r(f"cv_{v}", "n_share_readouts") for v in VERDICT_ROWS},
        "seg_share_verdict": {v: r(f"sv_{v}", "n_segments") for v in VERDICT_ROWS},
        "char_share_group": {g: r(f"cg_{g}", "n_share_readouts") for g in GROUP_ROWS},
        # rev 18: the junk share split by its reason tag (sums to the junk share)
        "char_share_junk_reason": {x: r(f"cj_{x}", "n_share_readouts") for x in JUNK_REASON_ROWS},
        "char_share_leaf": {lf: r(f"cs_{lf}", "n_share_readouts") for lf in LEAF_ROWS},
        "char_share_group_alt": {g: r(f"acg_{g}", "n_share_readouts") for g in GROUP_ROWS},
        "seg_share_group": {g: r(f"sg_{g}", "n_segments") for g in GROUP_ROWS},
        "seg_share_leaf": {lf: r(f"sc_{lf}", "n_segments") for lf in LEAF_ROWS},
        "rates": {
            f: {"all": r(f"ra_{f}", "n_rate_all"), "specific": r(f"rs_{f}", "n_rate_specific")}
            for f in FLAGS
        },
        "v5c_hallucination_rate_same_readouts": r("n_v5c_hal", "n_rate_specific"),
        "v5c_comparison": {
            "n_compared": int(totals["n_cmp"]),
            "n_readouts": int(totals["n_rate_all"]),
            "coverage": (totals["n_cmp"] / totals["n_rate_all"]) if totals["n_rate_all"] else None,
            # over every compared readout (any v5c class) ...
            "v5c_rate": r("n_cmp_v5c_hal", "n_cmp"),
            "v2_rate": r("n_cmp_v2_hal", "n_cmp"),
            "v2_false_claim_rate": r("n_cmp_v2_fc", "n_cmp"),
            "v2_detail_rate": r("n_cmp_v2_det", "n_cmp"),
            "v2_substitution_category_rate": r("n_cmp_v2_cat", "n_cmp"),
            "v2_substitution_sibling_gated_rate": r("n_cmp_v2_gated", "n_cmp"),
            # ... and over the v5c-specific ones (plan 0001's hallucination_rate denominator)
            "v5c_rate_specific": r("n_v5c_hal", "n_rate_specific"),
            "v2_rate_specific": r(f"rs_verdict_{tx.V_HALLUCINATION}", "n_rate_specific"),
        },
        "health": health,
        "transition": transition(cells),
        "agreement": agreement(cells),
    }


def transition(cells: Iterable[SlimCellV2]) -> dict[str, dict[str, int]]:
    """v5c readout class -> v2 readout group (first present in ``TRANSITION_PRIORITY``)."""
    t: dict[str, Counter] = {}
    for c in cells:
        for r in c.readouts:
            if r.ok and r.v5c_cls is not None:
                t.setdefault(r.v5c_cls, Counter())[readout_group(s[0] for s in r.segs)] += 1
    return {k: dict(v) for k, v in sorted(t.items())}


def agreement(cells: Iterable[SlimCellV2]) -> dict[str, dict[str, Any]]:
    """Verify status vs the v2 leaf covering most of the quote (failed / unlocated excluded)."""
    out: dict[str, dict[str, Any]] = {
        s: {"n": 0, "n_unlocated": 0, "n_agree": 0, "leaves": Counter()} for s in AGREEMENT
    }
    for c in cells:
        for status, lf in c.agreement:
            row = out[status]
            if lf is None:
                row["n_unlocated"] += 1
                continue
            if lf == tx.FAILED:
                continue
            row["n"] += 1
            row["n_agree"] += lf in AGREEMENT[status]
            row["leaves"][lf] += 1
    for row in out.values():
        row["agree_rate"] = row["n_agree"] / row["n"] if row["n"] else None
        row["leaves"] = dict(row["leaves"].most_common())
    return out


# ---------------------------------------------------------------- accumulate + summarise


@dataclass
class AccumulatorV2:
    """``sibling_labels`` (rev 21): segment key -> the unit's ``final`` (``None`` while rung 2 is
    unfinished); ``None`` = no labels file = rev-20 scoring."""

    sibling_labels: dict[SegKey, str | None] | None = None
    cells: list[SlimCellV2] = field(default_factory=list)

    def add(self, rec: dict[str, Any]) -> None:
        c = slim(rec, self.sibling_labels)
        if c.map_status != "none" or rec["status"] == "judged":
            self.cells.append(c)


def summarize(
    acc: AccumulatorV2, *, config: dict[str, Any], spend: dict[str, Any], n_boot: int = N_BOOT
) -> dict[str, Any]:
    overall = block(acc.cells, n_boot=n_boot)
    settled = sorted({c.layer for c in acc.cells if c.map_status in ("ok", "failed")})
    by_layer = {
        str(la): block([c for c in acc.cells if c.layer == la], n_boot=n_boot) for la in settled
    }
    h = overall["health"]
    n_settled = h["n_map_ok"] + h["n_map_failed"]
    return {
        "config": config,
        "health": {
            **h,
            "failed_cell_rate": h["n_map_failed"] / n_settled if n_settled else None,
            "unlocated_subspan_rate": (
                h["n_unlocated_subspans"] / h["n_splits"] if h["n_splits"] else None
            ),
            "missing_probe_target_rate": (
                h["n_missing_probe_targets"] / h["n_probe_candidates"]
                if h["n_probe_candidates"]
                else None
            ),
        },
        "overall": overall,
        "by_layer": by_layer,
        "spend_map": spend,
    }
