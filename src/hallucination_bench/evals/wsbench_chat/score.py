"""wsbench_chat metrics: the v5c headline plus the taxonomy metrics, with item-bootstrap CIs.

Denominators (plan 0001, "Metrics"):

* readout-level rates over **specific** readouts (class in ``SPECIFIC_CLASSES``), as in v5c;
  revoked readouts count (the headline is unrevoked);
* unsupported / disputed / strict / lenient rates over **T** = verify-tallied specific readouts,
  always reported beside the coverage T / specific.

Every rate gets an item-level percentile bootstrap 95% CI of the ratio of per-item sums (2,000
draws, seed 0, resampling items). The resample indices are drawn exactly as the source's
``bootstrap_rate_ci`` draws them (``random.Random(0).randrange`` over items in first-seen order),
so the headline CI equals the source's; the same draws are reused for every other rate.

Decomposition identities (tested): ``concept_only + detail_only + both + unclassified_only =
hallucination_rate`` and the same over T for the strict and lenient views.
"""

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hallucination_bench import taxonomy as tx
from hallucination_bench.judges.v5c import SPECIFIC_CLASSES

from .run import STAGES

N_BOOT = 2000
UNKNOWN = tx.UNKNOWN
UKINDS: tuple[str, ...] = (*tx.UNSUPPORTED_KINDS, UNKNOWN)
BUCKETS: tuple[str, ...] = ("concept_only", "detail_only", "both", "unclassified_only")

# (name, numerator count, denominator count)
RATES: tuple[tuple[str, str, str], ...] = (
    ("hallucination_rate", "n_hallucinated", "n_specific"),
    ("hallucination_rate_revoked", "n_hallucinated_unrevoked", "n_specific"),
    ("off_topic_rate", "n_off_topic", "n_specific"),
    ("assert_share", "n_specific", "n_readouts"),
    *((f"{b}_rate", f"c_{b}", "n_specific") for b in BUCKETS),
    ("concept_rate", "c_concept", "n_specific"),
    ("detail_rate", "c_detail", "n_specific"),
    ("major_rate", "n_major", "n_specific"),
    ("mean_max_severity", "sum_max_severity", "n_max_severity"),
    ("verify_coverage", "n_tallied", "n_specific"),
    ("unsupported_rate", "n_unsupported", "n_tallied"),
    ("unsupported_concept_rate", "n_unsupported_concept", "n_tallied"),
    ("unsupported_detail_rate", "n_unsupported_detail", "n_tallied"),
    *((f"unsupported_{k}_rate", f"n_unsupported_{k}", "n_tallied") for k in UKINDS),
    ("unsupported_rate_lenient", "n_unsupported_lenient", "n_tallied"),
    ("disputed_rate", "n_disputed", "n_tallied"),
    ("strict_hallucination_rate", "n_strict", "n_tallied"),
    *((f"strict_{b}_rate", f"s_{b}", "n_tallied") for b in BUCKETS),
    ("strict_concept_rate", "s_concept", "n_tallied"),
    ("strict_detail_rate", "s_detail", "n_tallied"),
    ("lenient_hallucination_rate", "n_lenient", "n_tallied"),
    *((f"lenient_{b}_rate", f"l_{b}", "n_tallied") for b in BUCKETS),
    ("lenient_concept_rate", "l_concept", "n_tallied"),
    ("lenient_detail_rate", "l_detail", "n_tallied"),
)

READOUT_COUNTS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            "n_readouts",
            "n_specific",
            "n_hallucinated",
            "n_hallucinated_unrevoked",
            "n_off_topic",
            "n_revoked",
            *(f"c_{b}" for b in BUCKETS),
            "c_concept",
            "c_detail",
            "n_major",
            "sum_max_severity",
            "n_max_severity",
            "n_hallucinated_all_unknown",
            "n_tallied",
            "n_unsupported",
            "n_unsupported_concept",
            "n_unsupported_detail",
            *(f"n_unsupported_{k}" for k in UKINDS),
            "n_unsupported_lenient",
            "n_disputed",
            "n_strict",
            *(f"s_{b}" for b in BUCKETS),
            "s_concept",
            "s_detail",
            "n_lenient",
            *(f"l_{b}" for b in BUCKETS),
            "l_concept",
            "l_detail",
        ]
    )
)

HEALTH_COUNTS: tuple[str, ...] = (
    "n_cells",
    "n_empty_cells",
    "n_pending_cells",
    "n_judged_cells",
    "n_short_cells",
    *(f"n_entered_{s}" for s in STAGES),
    *(f"n_failed_{s}" for s in STAGES),
    *(f"n_pending_{s}" for s in STAGES),
    "n_unverified_spans",
    "n_unverified_claim_quotes",
    "n_cells_claims_unjudged",
    "n_offtopic_overlap_quotes",
    "n_spans",
    "n_unclassified",
    "n_extra_ids",
    "n_duplicate_ids",
    "n_missing_ids",
    "n_bad_labels",
    "n_kind_ignored",
    # source stage-2 claim counts (readouts with a tally)
    "n_readouts_claims_judged",
    "n_claims_false",
    "n_claims_true",
    "n_claims_unverifiable",
    "n_claims_disputed",
    "n_claims_off_topic",
)

SpanT = tuple[str, str, str, int | None, str | None]  # grounding, type, target, severity, ukind


def bucket(types: Iterable[str]) -> str | None:
    """concept_only / detail_only / both / unclassified_only for a readout's span types (None
    when it has no span). ``unknown`` spans beside classified ones are ignored."""
    ts = list(types)
    if not ts:
        return None
    c, d = "concept" in ts, "detail" in ts
    if c and d:
        return "both"
    if c:
        return "concept_only"
    if d:
        return "detail_only"
    return "unclassified_only"


@dataclass
class SlimReadout:
    cls: str
    revoked: bool
    tallied: bool
    spans: list[SpanT]


@dataclass
class SlimCell:
    item: str
    layer: int
    status: str
    readouts: list[SlimReadout]
    health: dict[str, int]
    claims: list[dict[str, int]]


def slim(rec: dict[str, Any], k: int) -> SlimCell:
    """A ``cells.jsonl`` record -> what scoring needs (no text)."""
    h = dict.fromkeys(HEALTH_COUNTS, 0)
    h["n_cells"] = 1
    st = rec["status"]
    stages = rec.get("stages") or {}
    h["n_empty_cells"] = st == "empty"
    h["n_pending_cells"] = st == "pending"
    h["n_judged_cells"] = st == "judged"
    if st != "empty" and rec.get("samples") is not None and rec["n_samples"] < k:
        h["n_short_cells"] = 1
    for s in STAGES:
        v = stages.get(s)
        if v in ("ok", "failed", "pending"):
            h[f"n_entered_{s}"] = v != "pending"
            h[f"n_failed_{s}"] = v == "failed"
            h[f"n_pending_{s}"] = v == "pending"
    h["n_offtopic_overlap_quotes"] = rec.get("n_offtopic_overlap_quotes") or 0
    for k2, v2 in (rec.get("classify_counts") or {}).items():
        if k2 in h and k2 != "n_unclassified":
            h[k2] = v2
    readouts: list[SlimReadout] = []
    claims: list[dict[str, int]] = []
    by_r: dict[int, list[SpanT]] = defaultdict(list)
    for sp in rec.get("spans") or []:
        by_r[sp["readout_idx"]].append(
            (sp["grounding"], sp["type"], sp["target"], sp["severity"], sp["unsupported_kind"])
        )
        h["n_spans"] += 1
        h["n_unclassified"] += sp["type"] == UNKNOWN
    verdict = rec.get("verdict")
    if verdict is not None:
        if any(v["claims"] is None for v in verdict):
            h["n_cells_claims_unjudged"] = 1
        for v in verdict:
            h["n_unverified_spans"] += v["n_unverified"]
            if v["claims"] is not None:
                claims.append(v["claims"])
            readouts.append(
                SlimReadout(v["class"], bool(v["revoked"]), v["claims"] is not None, by_r[v["idx"]])
            )
    for c in claims:
        h["n_readouts_claims_judged"] += 1
        h["n_unverified_claim_quotes"] += c["n_unverified"]
        for key in ("false", "true", "unverifiable", "disputed", "off_topic"):
            h[f"n_claims_{key}"] += c[key]
    return SlimCell(
        rec["id"], rec["layer"], st, readouts, {k3: int(v3) for k3, v3 in h.items()}, claims
    )


def readout_counts(cell: SlimCell) -> dict[str, int]:
    c = dict.fromkeys(READOUT_COUNTS, 0)
    for r in cell.readouts:
        c["n_readouts"] += 1
        if r.cls not in SPECIFIC_CLASSES:
            continue
        c["n_specific"] += 1
        contra = [s for s in r.spans if s[0] == "contradicted"]
        if r.cls == "hallucinated":
            c["n_hallucinated"] += 1
            c["n_revoked"] += r.revoked
            c["n_hallucinated_unrevoked"] += not r.revoked
            b = bucket(s[1] for s in contra) or "unclassified_only"
            c[f"c_{b}"] += 1
            c["c_concept"] += b in ("concept_only", "both")
            c["c_detail"] += b in ("detail_only", "both")
            sevs = [s[3] for s in contra if s[1] != UNKNOWN and s[3] is not None]
            if sevs:
                c["sum_max_severity"] += max(sevs)
                c["n_max_severity"] += 1
            else:
                c["n_hallucinated_all_unknown"] += 1
        elif r.cls == "off_topic":
            c["n_off_topic"] += 1
        if any(s[3] == 3 and s[1] != UNKNOWN for s in contra):
            c["n_major"] += 1
        if not r.tallied:
            continue
        c["n_tallied"] += 1
        unsup = [s for s in r.spans if s[0] == "unsupported"]
        disp = [s for s in r.spans if s[0] == "disputed"]
        c["n_unsupported"] += bool(unsup)
        c["n_unsupported_concept"] += any(s[1] == "concept" for s in unsup)
        c["n_unsupported_detail"] += any(s[1] == "detail" for s in unsup)
        for k in UKINDS:
            c[f"n_unsupported_{k}"] += any((s[4] or UNKNOWN) == k for s in unsup)
        lenient_unsup = [s for s in unsup if s[4] not in tx.LENIENT_DROPPED_KINDS]
        c["n_unsupported_lenient"] += bool(lenient_unsup)
        c["n_disputed"] += bool(disp)
        for prefix, total, spans in (
            ("s", "n_strict", contra + disp + unsup),
            ("l", "n_lenient", contra + disp + lenient_unsup),
        ):
            b = bucket(s[1] for s in spans)
            if b is None:
                continue
            c[total] += 1
            c[f"{prefix}_{b}"] += 1
            c[f"{prefix}_concept"] += b in ("concept_only", "both")
            c[f"{prefix}_detail"] += b in ("detail_only", "both")
    return c


def span_distribution(cells: Iterable[SlimCell]) -> dict[str, Any]:
    """Counts and shares by grounding x type x target x severity x unsupported_kind, over the
    spans that enter the rates (contradicted spans of hallucinated readouts; disputed and
    unsupported spans of tallied specific readouts)."""
    joint: Counter = Counter()
    for cell in cells:
        for r in cell.readouts:
            if r.cls not in SPECIFIC_CLASSES:
                continue
            for s in r.spans:
                if s[0] != "contradicted" and not r.tallied:
                    continue
                joint[s] += 1
    marg: dict[str, dict[str, dict[str, Any]]] = {}
    for g in ("contradicted", "disputed", "unsupported"):
        sub = {s: n for s, n in joint.items() if s[0] == g}
        tot = sum(sub.values())
        axes: dict[str, Any] = {"n": tot}
        for ai, axis in ((1, "type"), (2, "target"), (3, "severity"), (4, "unsupported_kind")):
            cnt: Counter = Counter()
            for s, n in sub.items():
                cnt[str(s[ai])] += n
            axes[axis] = {
                k: {"n": n, "share": n / tot if tot else None} for k, n in sorted(cnt.items())
            }
        marg[g] = axes
    return {
        "joint": [
            {
                "grounding": s[0],
                "type": s[1],
                "target": s[2],
                "severity": s[3],
                "unsupported_kind": s[4],
                "n": n,
            }
            for s, n in sorted(joint.items(), key=lambda x: (-x[1], str(x[0])))
        ],
        "by_grounding": marg,
    }


# ---------------------------------------------------------------- bootstrap


def bootstrap_indices(n_items: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n_items) item indices, drawn exactly as the source's ``bootstrap_rate_ci``."""
    rng = random.Random(seed)
    out = np.empty((n_boot, n_items), dtype=np.int64)
    for b in range(n_boot):
        for i in range(n_items):
            out[b, i] = rng.randrange(n_items)
    return out


def _ci(num: np.ndarray, den: np.ndarray) -> list[float] | None:
    ok = den > 0
    if not ok.any():
        return None
    draws = np.sort(num[ok] / den[ok])
    n = len(draws)
    return [float(draws[int(0.025 * (n - 1))]), float(draws[int(0.975 * (n - 1))])]


def block(cells: Sequence[SlimCell], *, n_boot: int = N_BOOT) -> dict[str, Any]:
    """Counts, rates and bootstrap CIs for one group of cells."""
    per_item: dict[str, dict[str, int]] = {}
    health = dict.fromkeys(HEALTH_COUNTS, 0)
    for cell in cells:
        for k, v in cell.health.items():
            health[k] += v
        if cell.status == "empty":
            continue
        # Items whose cells all failed / are pending stay in the resample as (0, 0) rows, as in
        # the source bootstrap.
        acc = per_item.setdefault(cell.item, dict.fromkeys(READOUT_COUNTS, 0))
        if cell.status != "judged":
            continue
        rc = readout_counts(cell)
        for k, v in rc.items():
            acc[k] += v
    names = list(READOUT_COUNTS)
    mat = np.array([[d[k] for k in names] for d in per_item.values()], dtype=np.float64)
    if mat.size == 0:
        mat = np.zeros((0, len(names)))
    totals = {k: int(mat[:, i].sum()) if len(mat) else 0 for i, k in enumerate(names)}
    rates: dict[str, float | None] = {}
    cis: dict[str, list[float] | None] = {}
    if len(mat):
        idx = bootstrap_indices(len(mat), n_boot)
        sums = mat[idx].sum(axis=1)  # (n_boot, n_counts)
    col = {k: i for i, k in enumerate(names)}
    for name, num, den in RATES:
        d = totals[den]
        rates[name] = totals[num] / d if d else None
        cis[name] = _ci(sums[:, col[num]], sums[:, col[den]]) if len(mat) else None
    return {
        "n_items": len(per_item),
        "counts": totals,
        "health": health,
        "rates": rates,
        "ci95": cis,
    }


# ---------------------------------------------------------------- accumulate + summarise


@dataclass
class Accumulator:
    k: int
    cells: list[SlimCell] = field(default_factory=list)

    def add(self, rec: dict[str, Any]) -> None:
        self.cells.append(slim(rec, self.k))


def failed_rates(health: dict[str, int]) -> dict[str, float | None]:
    return {
        s: (health[f"n_failed_{s}"] / health[f"n_entered_{s}"])
        if health[f"n_entered_{s}"]
        else None
        for s in STAGES
    }


def summarize(
    acc: Accumulator,
    *,
    config: dict[str, Any],
    n_missing: int,
    n_expected: int,
    skipped_rows: dict[str, int],
    spend: dict[str, dict[str, Any]],
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    overall = block(acc.cells, n_boot=n_boot)
    by_layer: dict[str, Any] = {}
    for layer in sorted({c.layer for c in acc.cells}):
        by_layer[str(layer)] = block([c for c in acc.cells if c.layer == layer], n_boot=n_boot)
    dist = span_distribution(c for c in acc.cells if c.status == "judged")
    unsup = dist["by_grounding"]["unsupported"]["unsupported_kind"]
    health = {
        "n_expected_cells": n_expected,
        "n_missing": n_missing,
        "skipped_rows": skipped_rows,
        **overall["health"],
        "failed_rate_by_stage": failed_rates(overall["health"]),
    }
    tot_usd = sum(float(v.get("usd") or 0) for v in spend.values())
    return {
        "config": config,
        "health": health,
        "overall": overall,
        "by_layer": by_layer,
        "unsupported_kind_span_share": {k: v["share"] for k, v in unsup.items()},
        "spans": dist,
        "spend": {**spend, "total_usd": tot_usd},
    }
