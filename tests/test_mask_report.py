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


def test_probe_tables_toy():
    sites = {
        "h": [
            {"pos": 1, "site_type": "prompt", "token": "assistant"},
            {"pos": 5, "site_type": "response", "token": "x"},
        ],
        "l": [
            {"pos": 1, "site_type": "prompt", "token": "assistant"},
            {"pos": 5, "site_type": "response", "token": "x"},
        ],
    }
    gate = [_g("honest_correct", "resisted", 10, 0), _g("lie", None, 0, 10)]
    gate[0]["task_id"], gate[1]["task_id"] = "h", "l"
    summ = {"auroc_text": 1.0, "tau_pos": 1.0, "tau_text": 1.0}
    probe = {
        "layers": {
            "36": {
                "ip_end": summ,
                "ip_resp": summ,
                "cf_prompt": summ,
                "cf_response": summ,
                "ip_scores": {"h": [0.0, 2.0], "l": [3.0, 3.0]},
                "cf_scores": {"h": [0.0, 0.0], "l": [3.0, 0.0]},
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
    out = R.probe_tables(probe, sites, gate, "ctx", {"arm": cells})
    r = out["layers"]["36"]
    assert r["prompt"]["transfer_ip"] == 1.0 and r["prompt"]["premise_hr_ip"] == 0.0
    assert r["response"]["premise_hr_ip"] == 1.0 and r["response"]["premise_lie_cf"] == 0.0
    d = out["arms"]["arm"]["L036:response"]
    assert d["hr_hits_supported"] == 1 and d["hr_hits_supported_cf"] == 0
    assert d["rate_original"] == 1 / 3 and d["rate_cf_checked"] == 0.0
    assert d["miss_rate"] == 1.0  # the lie site is probe-present and carries no claim
