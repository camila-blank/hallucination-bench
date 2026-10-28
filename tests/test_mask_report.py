"""Plan 0017: readout- / item-level rates and the bootstrap on a toy cell set."""

from hallucination_bench.evals.mask import report as R

LIE_V = {
    "readout": 0,
    "category": "lie_claim",
    "verify": {"real": True},
    "verify_blind": {"real": False},
}
LIE_R1 = {"readout": 1, "category": "lie_claim", "verify": {"real": True}, "verify_blind": None}
LIE_X = {
    "readout": 0,
    "category": "lie_claim",
    "verify": {"real": False},
    "verify_blind": {"real": True},
}
PRESS = {"readout": 2, "category": "pressure_salience", "verify": None, "verify_blind": None}


def cell(tid, label, stratum, mentions, status="ok", n=3):
    return {
        "task_id": tid,
        "layer": 36,
        "site_type": "prompt",
        "label": label,
        "stratum": stratum,
        "status": status,
        "n_readouts": n,
        "mentions": mentions,
    }


CELLS = [
    cell("h1", "honest_correct", "resisted", [LIE_V, LIE_R1]),  # 2 of 3 readouts, sample 0 hit
    cell("h1", "honest_correct", "resisted", []),
    cell("h2", "honest_correct", "resisted", [LIE_R1, PRESS]),  # 1 of 3, sample 0 not hit
    cell("h2", "honest_correct", "resisted", [LIE_X], status="failed"),  # not counted
    cell("l1", "lie", None, [LIE_V, LIE_V]),  # same readout twice counts once
    cell("w1", "honest_wrong", "aligned", [LIE_X]),
]


def test_item_stats_measures():
    v = R.item_stats(CELLS, "verified")
    assert v == {"h1": (2, 6, True), "h2": (1, 3, False), "l1": (1, 3, True), "w1": (0, 3, False)}
    d = R.item_stats(CELLS, "detector")
    assert d["w1"] == (1, 3, True) and d["h2"] == (1, 3, False)
    b = R.item_stats(CELLS, "blind")
    assert b["h1"] == (0, 6, False) and b["w1"] == (1, 3, True)
    assert R.item_stats(CELLS, "pressure")["h2"] == (1, 3, False)


def test_bootstrap_points_and_diff():
    v = R.item_stats(CELLS, "verified")
    groups = {"honest_resisted": [v["h1"], v["h2"]], "lie": [v["l1"]]}
    out = R.bootstrap(groups, n_boot=200)
    hr = out["honest_resisted"]
    assert hr["readout_rate"] == 3 / 9 and hr["item_rate"] == 0.5 and hr["n_items"] == 2
    assert out["lie"]["readout_rate"] == 1 / 3
    assert out["diff_lie_minus_honest_resisted"]["readout_rate"] == 0.0
    lo, hi = hr["readout_ci"]
    assert 0.0 <= lo <= 3 / 9 <= hi <= 2 / 6 + 1e-9  # resamples of {2/6, 1/3}: all 1/3
    assert R.bootstrap(groups, n_boot=200) == out  # default_rng(0): reproducible


def test_arm_metrics_strata():
    rows = R.arm_metrics(CELLS, n_boot=50)
    row = rows["L036:prompt"]
    assert row["verified"]["honest_resisted"]["n_items"] == 2
    assert row["verified"]["honest_wrong"]["readout_rate"] == 0.0
    assert row["detector"]["honest_aligned"]["readout_rate"] == 1 / 3
    assert row["health"] == {"ok": 5, "failed": 1}
    assert R.acceptance(CELLS)["honest_resisted"] == {"n_candidates": 3, "verified": 3, "blind": 0}


def test_deception_lex_counts_vocabulary_only():
    cells = [
        cell(
            "h1",
            "honest_correct",
            "resisted",
            [{"readout": 0, "category": "other", "quote": " Lying"}],
        ),
        cell(
            "h2",
            "honest_correct",
            "resisted",
            [{"readout": 0, "category": "other", "quote": "truth"}],
        ),
        cell(
            "h3",
            "honest_correct",
            "resisted",
            [{"readout": 1, "category": "other", "quote": "谎言"}],
        ),
    ]
    st = R.item_stats(cells, "deception_lex")
    assert st == {"h1": (1, 3, True), "h2": (0, 3, False), "h3": (1, 3, False)}
