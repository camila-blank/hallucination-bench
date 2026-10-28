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


def _g(label, stratum, matched, not_matched, fm=0, fn=0, arch="known_facts"):
    z = {"by_order": {}, "raw": []}
    return {
        "task_id": f"{label}{matched}{fm}",
        "label": label,
        "stratum": stratum,
        "archetype": arch,
        "ctx": {
            **z,
            "n_matched": matched,
            "n_not_matched": not_matched,
            "n_unparsed": 10 - matched - not_matched,
        },
        "fresh": {**z, "n_matched": fm, "n_not_matched": fn, "n_unparsed": 10 - fm - fn},
    }


def test_gate_pass_rule_and_frame_choice():
    lie9 = _g("lie", None, 1, 9)
    hon8 = _g("honest_correct", "resisted", 8, 1)
    assert R.gate_pass(lie9, "ctx", 9) and not R.gate_pass(lie9, "ctx", 10)
    assert R.gate_pass(hon8, "ctx", 8) and not R.gate_pass(hon8, "ctx", 9)
    assert R.gate_pass(_g("evasive", None, 10, 0), "ctx", 9) is None
    # carryover: ctx says "matched" on lies, fresh does not -> fresh
    gate = [_g("lie", None, 9, 1, fm=0, fn=10), _g("lie", None, 8, 2, fm=1, fn=9)]
    assert R.choose_frame(gate)[0] == "fresh"
    assert R.choose_frame([lie9])[0] == "ctx"
    surv = R.gate_survival([lie9, hon8])
    assert surv["lie"]["all"]["ctx@9"] == 1 and surv["honest_resisted"]["all"]["ctx@9"] == 0


def test_probe_tables_toy(monkeypatch):
    from hallucination_bench.evals.mask import bank

    sites = {
        "h": [{"pos": 1, "site_type": "prompt"}, {"pos": 5, "site_type": "response"}],
        "l": [{"pos": 1, "site_type": "prompt"}, {"pos": 5, "site_type": "response"}],
    }
    monkeypatch.setattr(bank, "load_sites", lambda: sites)
    gate = [_g("honest_correct", "resisted", 10, 0), _g("lie", None, 0, 10)]
    gate[0]["task_id"], gate[1]["task_id"] = "h", "l"
    labels = {"h": {"archetype": "known_facts"}, "l": {"archetype": "known_facts"}}
    t1 = {"tau": 1.0, "tau_pos": -0.5}
    summary = {
        "layers": {
            "36": {
                "loao:known_facts": {"loao_prompt": t1, "loao_response": t1},
                "cf": {"cf_prompt": {"tau_pos": 1.0}, "cf_response": {"tau_pos": 1.0}},
                "ip": {},
            }
        }
    }

    def sc(h_p, l_p, h_r, l_r):
        return {
            "prompt": {"h": [[0], [h_p]], "l": [[0], [l_p]]},
            "response": {"h": [[1], [h_r]], "l": [[1], [l_r]]},
        }

    scores = {
        "layers": {
            "36": {
                "loao:known_facts": {"scores": sc(0.0, 3.0, 0.0, 3.0)},
                "cf": {"scores": sc(0.0, 3.0, 2.0, 0.0)},
                "ip": {"scores": sc(0.0, 1.0, 0.0, 1.0)},
            }
        }
    }
    claim = {"readout": 0, "category": "lie_claim", "verify": {"real": True}, "verify_blind": None}

    def c(t, lab, st, pos, ms):
        return {
            **cell(t, lab, "resisted" if lab != "lie" else None, ms),
            "pos": pos,
            "site_type": st,
        }

    cells = [c("h", "honest_correct", "response", 5, [claim]), c("l", "lie", "response", 5, [])]
    out = R.probe_tables(summary, scores, gate, "ctx", labels, {"arm": cells})
    r = out["layers"]["36"]
    assert r["prompt"]["transfer_loao"] == 1.0 and r["response"]["transfer_cf"] == 0.0
    assert (
        r["response"]["premise_sites_hr_loao"] == 0.0
        and r["response"]["premise_items_lie_loao"] == 1.0
    )
    d = out["arms"]["arm"]["L036:response"]
    # loao: the honest site is not present -> the claim is probe-contradicted
    assert d["loao_hr_contradicted"] == 1 and d["loao_hr_supported"] == 0
    assert d["loao_rate_probe_checked"] == 1 / 3 and d["loao_rate_supported"] == 0.0
    # cf: the honest site is present -> supported; the lie site is not present -> no miss counted
    assert d["cf_hr_supported"] == 1 and d["cf_rate_probe_checked"] == 0.0
    assert d["loao_miss_rate"] == 1.0 and d["cf_miss_rate"] is None
    # OOF-threshold sensitivity (tau_pos = -0.5): the honest site (score 0.0) is now present
    assert d["loao_oof_hr_supported"] == 1 and d["loao_oof_rate_probe_checked"] == 0.0


def test_failed_bounds():
    gate = [_g("honest_correct", "resisted", 10, 0)]
    gate[0]["task_id"] = "h"
    claim = {"readout": 0, "category": "lie_claim", "verify": {"real": True}, "verify_blind": None}
    cells = [
        cell("h", "honest_correct", "resisted", [claim]),
        cell("h", "honest_correct", "resisted", [], status="failed"),
    ]
    b = R.failed_bounds(cells, gate, "ctx")["L036:prompt"]
    assert (
        b["rate"] == 1 / 3
        and b["lower"] == 1 / 6
        and b["upper"] == 4 / 6
        and b["n_failed_cells"] == 1
    )
