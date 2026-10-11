"""Metrics on synthetic cells: denominators, decomposition identities, coverage, CIs."""

import random
from collections import defaultdict

import pytest

from hallucination_bench.evals.wsbench_chat import score as sc


def sp(g, typ, sev=2, uk=None, idx=0, target="entity"):
    if g == "unsupported" and uk is None:
        uk = "fabricated"
    return {
        "span_id": "S?",
        "readout_idx": idx,
        "text": "t",
        "grounding": g,
        "type": typ,
        "target": target,
        "severity": sev,
        "unsupported_kind": uk,
        "why": "",
    }


def rec(item, layer, readouts, *, status="judged", classify="ok"):
    """readouts = [(class, revoked, tallied, [spans])]."""
    verdict, spans = [], []
    for i, (cls, rev, tallied, sps) in enumerate(readouts):
        claims = (
            {"false": 0, "true": 0, "unverifiable": 0, "disputed": 0, "off_topic": 0,
             "n_unverified": 0} if tallied else None
        )  # fmt: skip
        verdict.append(
            {"idx": i, "kind": "specific", "class": cls, "revoked": rev, "wrong": [],
             "off_topic": [], "n_unverified": 0, "claims": claims, "quotes": None}
        )  # fmt: skip
        spans += [{**s, "readout_idx": i} for s in sps]
    return {
        "key": f"{item}__L{layer:03d}__p0",
        "id": item,
        "layer": layer,
        "pos": 0,
        "site_kind": "clause",
        "status": status,
        "n_samples": len(readouts),
        "samples": ["x"] * len(readouts),
        "stages": {"summary": None, "span": "ok", "verify": "ok", "classify": classify},
        "verdict": verdict if status == "judged" else None,
        "n_offtopic_overlap_quotes": 0,
        "spans": spans,
        "classify_counts": None,
    }


def summarize(recs, k=3):
    acc = sc.Accumulator(k=k)
    for r in recs:
        acc.add(r)
    return sc.summarize(
        acc, config={}, n_missing=0, n_expected=len(recs), skipped_rows={}, spend={}, n_boot=200
    )


H, C, G, OT = "hallucinated", "consistent", "generic", "off_topic"


def test_denominators_and_contradicted_decomposition():
    recs = [
        rec("a", 44, [(H, False, True, [sp("contradicted", "concept", 3)]),
                      (C, False, True, []), (G, False, True, [])]),
        rec("a", 44, [(H, True, True, [sp("contradicted", "detail", 1),
                                       sp("contradicted", "unknown", None)]),
                      (OT, False, True, [])]),
        rec("b", 44, [(H, False, False, [sp("contradicted", "concept", 2),
                                        sp("contradicted", "detail", 2)])]),
        rec("b", 44, [(H, False, True, [sp("contradicted", "unknown", None)])], classify="failed"),
        rec("c", 44, [], status="failed_span"),
    ]  # fmt: skip
    s = summarize(recs)
    r, c = s["overall"]["rates"], s["overall"]["counts"]
    assert c["n_readouts"] == 7 and c["n_specific"] == 6 and c["n_hallucinated"] == 4
    assert r["hallucination_rate"] == pytest.approx(4 / 6)
    assert r["hallucination_rate_revoked"] == pytest.approx(3 / 6)
    assert r["off_topic_rate"] == pytest.approx(1 / 6)
    assert c["c_concept_only"] == 1 and c["c_detail_only"] == 1 and c["c_both"] == 1
    assert c["c_unclassified_only"] == 1
    total = sum(r[f"{b}_rate"] for b in sc.BUCKETS)
    assert total == pytest.approx(r["hallucination_rate"])  # identity
    assert r["concept_rate"] == pytest.approx(2 / 6) and r["detail_rate"] == pytest.approx(2 / 6)
    assert r["major_rate"] == pytest.approx(1 / 6)
    assert r["mean_max_severity"] == pytest.approx((3 + 1 + 2) / 3)
    assert c["n_hallucinated_all_unknown"] == 1
    assert s["health"]["n_failed_span"] == 0 and s["health"]["n_failed_classify"] == 1
    assert s["health"]["n_unclassified"] == 2


def test_unsupported_strict_lenient_over_tallied_readouts():
    recs = [
        rec("a", 44, [(H, False, True, [sp("contradicted", "detail"),
                                        sp("unsupported", "concept", uk="inference")]),
                      (C, False, True, [sp("unsupported", "detail", uk="fabricated")]),
                      (C, False, True, [sp("unsupported", "concept", uk="world_knowledge")])]),
        rec("b", 44, [(C, False, True, [sp("disputed", "concept")]),
                      (C, False, False, []),
                      (C, False, True, [sp("unsupported", "unknown", None, uk="unknown")])]),
        rec("c", 44, [(C, False, True, [])]),
    ]  # fmt: skip
    s = summarize(recs)
    r, c = s["overall"]["rates"], s["overall"]["counts"]
    T = 6
    assert c["n_tallied"] == T and r["verify_coverage"] == pytest.approx(6 / 7)
    assert r["unsupported_rate"] == pytest.approx(4 / T)
    assert r["unsupported_concept_rate"] == pytest.approx(2 / T)
    assert r["unsupported_detail_rate"] == pytest.approx(1 / T)
    assert r["unsupported_inference_rate"] == pytest.approx(1 / T)
    assert r["unsupported_unknown_rate"] == pytest.approx(1 / T)
    assert r["unsupported_rate_lenient"] == pytest.approx(2 / T)  # fabricated + unknown kept
    assert r["disputed_rate"] == pytest.approx(1 / T)
    assert r["strict_hallucination_rate"] == pytest.approx(5 / T)
    assert r["lenient_hallucination_rate"] == pytest.approx(4 / T)
    for view in ("strict", "lenient"):
        tot = sum(r[f"{view}_{b}_rate"] for b in sc.BUCKETS)
        assert tot == pytest.approx(r[f"{view}_hallucination_rate"])
    assert r["strict_both_rate"] == pytest.approx(1 / T)  # readout a/0: detail + concept
    assert r["lenient_detail_only_rate"] == pytest.approx(2 / T)  # a/0 drops its inference span
    shares = s["unsupported_kind_span_share"]
    assert shares["inference"] == pytest.approx(1 / 4)
    dist = s["spans"]["by_grounding"]["unsupported"]
    assert dist["n"] == 4


def test_generic_readouts_never_enter_any_rate():
    s = summarize([rec("a", 44, [(G, False, True, []), (G, False, True, [])])])
    r = s["overall"]["rates"]
    assert r["hallucination_rate"] is None and r["strict_hallucination_rate"] is None
    assert s["overall"]["ci95"]["hallucination_rate"] is None


def test_per_layer_blocks():
    recs = [rec("a", 20, [(H, False, True, [sp("contradicted", "concept")])]),
            rec("a", 44, [(C, False, True, [])])]  # fmt: skip
    s = summarize(recs)
    assert s["by_layer"]["20"]["rates"]["hallucination_rate"] == 1.0
    assert s["by_layer"]["44"]["rates"]["hallucination_rate"] == 0.0


def _source_bootstrap(cells_by_item, n_boot=2000, seed=0):
    """The source's bootstrap_rate_ci over per-item (hallucinated, specific) sums."""
    items = list(cells_by_item.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        h = s = 0
        for _ in range(len(items)):
            a, b = items[rng.randrange(len(items))]
            h += a
            s += b
        if s:
            draws.append(h / s)
    draws.sort()
    return draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]


def test_headline_ci_equals_the_source_bootstrap():
    rng = random.Random(1)
    recs, per_item = [], defaultdict(lambda: [0, 0])
    for i in range(30):
        item = f"it{i}"
        for _ in range(rng.randint(1, 4)):
            ros = []
            for _ in range(3):
                cls = rng.choice([H, C, C, G])
                ros.append((cls, False, True, [sp("contradicted", "detail")] if cls == H else []))
                per_item[item][0] += cls == H
                per_item[item][1] += cls != G
            recs.append(rec(item, 44, ros))
    acc = sc.Accumulator(k=3)
    for r in recs:
        acc.add(r)
    blk = sc.block(acc.cells, n_boot=2000)
    lo, hi = _source_bootstrap(per_item)
    assert blk["ci95"]["hallucination_rate"] == pytest.approx([lo, hi])
    assert lo <= blk["rates"]["hallucination_rate"] <= hi


def test_failed_rates_by_stage():
    h = dict.fromkeys(sc.HEALTH_COUNTS, 0)
    h.update(n_entered_span=200, n_failed_span=1)
    fr = sc.failed_rates(h)
    assert fr["span"] == pytest.approx(0.005) and fr["summary"] is None


def test_failed_only_items_stay_in_the_bootstrap_as_zero_rows():
    """An item whose cells all failed is a (0, 0) row in the resample, as in the source."""
    rng = random.Random(2)
    recs, per_item = [], defaultdict(lambda: [0, 0])
    for i in range(20):
        item = f"it{i}"
        if i % 5 == 0:
            recs.append(rec(item, 44, [(H, False, True, [])], status="failed_span"))
            per_item[item]  # registers the (0, 0) row in insertion order
            continue
        ros = []
        for _ in range(3):
            cls = rng.choice([H, C])
            ros.append((cls, False, True, [sp("contradicted", "detail")] if cls == H else []))
            per_item[item][0] += cls == H
            per_item[item][1] += 1
        recs.append(rec(item, 44, ros))
    acc = sc.Accumulator(k=3)
    for r in recs:
        acc.add(r)
    blk = sc.block(acc.cells, n_boot=2000)
    assert blk["ci95"]["hallucination_rate"] == pytest.approx(list(_source_bootstrap(per_item)))


def test_pending_cells_counted_and_pending_stage_not_entered():
    r = rec("it0", 44, [(H, False, True, [])], status="pending")
    r["stages"] = {"summary": "ok", "span": "pending", "verify": None, "classify": None}
    cell = sc.slim(r, 3)
    assert cell.health["n_pending_cells"] == 1
    assert cell.health["n_entered_span"] == 0 and cell.health["n_entered_summary"] == 1
