"""Claim-map (taxonomy v2) metrics: character-weighted shares, readout-level rates,
comparisons with v5c / verify, health; item-bootstrap 95% CIs (plan 0002, §Metrics).

Which readouts count:

* a cell enters once its ``:M`` stage is settled: ``ok`` or ``failed`` (a failed cell's units
  are all ``failed``). Pending cells (not yet mapped) are left out and counted in health;
* **shares** (headline) are per readout, then averaged over readouts (each non-empty readout
  weighs the same; failed readouts are in, under the ``failed`` row). The 7 groups plus
  ``unknown_incorrect`` plus ``failed`` sum to 1, and so do the 21 leaf rows. Segment-count
  shares are pooled over segments;
* **readout-level rates** are over readouts of ``ok`` cells only (a failed readout has no
  labels), with two denominators: all of them (``all``) and the v5c-specific ones
  (``specific``, plan 0001's ``hallucination_rate`` denominator).

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
LEAF_ROWS: tuple[str, ...] = tx.ALL_LEAVES  # 19 leaves + unknown_incorrect + failed
GROUP_ROWS: tuple[str, ...] = tx.ROWS  # 7 groups + unknown_incorrect + failed
HALLUCINATION_LEAVES = frozenset({"detail", "false_claim", tx.UNKNOWN_INCORRECT})

FLAGS: tuple[str, ...] = (
    "hallucination",  # detail | false_claim | unknown_incorrect (hallucination_rate_v2)
    "hallucination_excl_unknown",  # sensitivity: unknown_incorrect not counted
    "hallucination_alt",  # sensitivity: disputed segments re-scored with their alt_leaf
    "false_claim",
    "cruxy_false_claim",
    "detail",
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
    tx.THINKY,
    tx.UNSETTLED,
    tx.CORRECT,
    tx.JUNK,
    tx.NON_CLAIM,
    tx.FAILED,
)

# verify status -> the v2 leaves that agree with it (plan 0002, §Inputs)
AGREEMENT: dict[str, frozenset[str]] = {
    "true": frozenset({"restates_prefix", "stated_after", "entailed"}),
    "unverifiable": frozenset({"world_true", *tx.THINKY_LEAVES, "false_claim"}),
    "disputed": frozenset({*tx.INCORRECT_LEAVES, "possible_belief", tx.UNKNOWN_INCORRECT}),
    "off_topic": frozenset({"off_topic", "ambiguous_topic", *tx.NON_CLAIM_LEAVES}),
}

HEALTH_V2: tuple[str, ...] = (
    "n_cells",
    "n_map_ok",
    "n_map_failed",
    "n_map_pending",
    "n_readouts",
    *cm.PARSE_COUNTS,
    "n_unlocated_established",
)


# ---------------------------------------------------------------- slim records

SegT = tuple[str, str, int, bool]  # (leaf, alt-view leaf, chars, cruxy)


@dataclass
class SlimReadoutV2:
    v5c_cls: str
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


def slim(rec: dict[str, Any]) -> SlimCellV2:
    h = dict.fromkeys(HEALTH_V2, 0)
    st = (rec.get("stages") or {}).get("map") or "none"
    m = rec.get("map")
    if rec["status"] != "judged" or m is None:
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
    readouts: list[SlimReadoutV2] = []
    agreement: list[tuple[str, str | None]] = []
    for v in rec.get("verdict") or []:
        segs = by_r.get(v["idx"], [])
        if not segs:
            continue
        h["n_readouts"] += 1
        readouts.append(
            SlimReadoutV2(
                v["class"],
                st == "ok",
                [
                    (s["leaf"], cm.alt_leaf_view(s), s["end"] - s["start"], bool(s["cruxy"]))
                    for s in segs
                ],
            )
        )
        quotes = v.get("quotes")
        if st != "ok" or not quotes:
            continue
        text = (rec.get("samples") or [""])[v["idx"]]
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
    return {
        "hallucination": any(lf in HALLUCINATION_LEAVES for lf in leaves),
        "hallucination_excl_unknown": any(lf in ("detail", "false_claim") for lf in leaves),
        "hallucination_alt": any(s[1] in HALLUCINATION_LEAVES for s in r.segs),
        "false_claim": "false_claim" in leaves,
        "cruxy_false_claim": any(s[0] == "false_claim" and s[3] for s in r.segs),
        "detail": "detail" in leaves,
        "co_activation": tx.CO_ACTIVATION in groups,
        "thinky": tx.THINKY in groups,
        "correct": tx.CORRECT in groups,
        "nonclaim_or_junk_only": groups <= {tx.NON_CLAIM, tx.JUNK},
    }


def _columns() -> list[str]:
    cols = ["n_share_readouts", "n_segments", "n_rate_all", "n_rate_specific", "n_v5c_hal"]
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
        for lf, alt, ch, _c in r.segs:
            v[_COL[f"cs_{lf}"]] += ch / total
            v[_COL[f"cg_{tx.group_of(lf)}"]] += ch / total
            v[_COL[f"acg_{tx.group_of(alt)}"]] += ch / total
    for lf, _alt, _ch, _c in r.segs:
        v[_COL["n_segments"]] += 1
        v[_COL[f"sc_{lf}"]] += 1
        v[_COL[f"sg_{tx.group_of(lf)}"]] += 1
    if r.ok:
        spec = r.v5c_cls in SPECIFIC_CLASSES
        v[_COL["n_rate_all"]] = 1
        v[_COL["n_rate_specific"]] = spec
        v[_COL["n_v5c_hal"]] = spec and r.v5c_cls == "hallucinated"
        for f, on in readout_flags(r).items():
            v[_COL[f"ra_{f}"]] = on
            v[_COL[f"rs_{f}"]] = on and spec
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
        "char_share_group": {g: r(f"cg_{g}", "n_share_readouts") for g in GROUP_ROWS},
        "char_share_leaf": {lf: r(f"cs_{lf}", "n_share_readouts") for lf in LEAF_ROWS},
        "char_share_group_alt": {g: r(f"acg_{g}", "n_share_readouts") for g in GROUP_ROWS},
        "seg_share_group": {g: r(f"sg_{g}", "n_segments") for g in GROUP_ROWS},
        "seg_share_leaf": {lf: r(f"sc_{lf}", "n_segments") for lf in LEAF_ROWS},
        "rates": {
            f: {"all": r(f"ra_{f}", "n_rate_all"), "specific": r(f"rs_{f}", "n_rate_specific")}
            for f in FLAGS
        },
        "v5c_hallucination_rate_same_readouts": r("n_v5c_hal", "n_rate_specific"),
        "health": health,
        "transition": transition(cells),
        "agreement": agreement(cells),
    }


def transition(cells: Iterable[SlimCellV2]) -> dict[str, dict[str, int]]:
    """v5c readout class -> v2 readout group (first present in ``TRANSITION_PRIORITY``)."""
    t: dict[str, Counter] = {}
    for c in cells:
        for r in c.readouts:
            if r.ok:
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
    cells: list[SlimCellV2] = field(default_factory=list)

    def add(self, rec: dict[str, Any]) -> None:
        c = slim(rec)
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
        },
        "overall": overall,
        "by_layer": by_layer,
        "spend_map": spend,
    }
