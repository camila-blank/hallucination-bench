"""Plan 0002 rev 13: the supplementary branch, the verdict layer, probe targets, single-call
judging (no established spans, no verify), the v5c comparison and the pre-gate label."""

import json

import pytest
from test_claimmap_v2 import V5C, MapFake, _cells, _rec, _run, e, prompt_quote, sp, unit_lines

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import report_v2
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import bank, score_v2
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.units import split_cell

# ------------------------------------------------------------------ verdicts


def test_verdict_mapping_covers_every_leaf_exactly_once():
    mapped = [lf for leaves in tx.VERDICTS.values() for lf in leaves]
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == {*tx.LEAF_NAMES, tx.UNKNOWN_INCORRECT}
    assert set(tx.VERDICTS) == {
        "hallucination",
        "probe_candidate",
        "thinky",  # rev 19
        "correct",
        "unresolved",
        "junk",
    }
    assert tx.VERDICTS["hallucination"] == ("detail", "false_claim")
    assert set(tx.VERDICTS["junk"]) == set(tx.NON_CLAIM_LEAVES) and len(tx.NON_CLAIM_LEAVES) == 1
    assert tx.verdict_of("supplementary_true") == "correct"
    assert tx.verdict_of("supplementary_false") == "probe_candidate"
    assert tx.verdict_of("unverifiable") == "unresolved"
    assert tx.verdict_of(tx.UNKNOWN_INCORRECT) == "unresolved"
    assert tx.verdict_of(tx.FAILED) == tx.FAILED
    assert tx.VERDICT_ROWS == (*tx.VERDICTS, tx.FAILED)


def test_groups_follow_option_a():
    # rev 16: one correct leaf; supplementary_unsettled folded into unverifiable (world tag)
    assert tx.CORRECT_LEAVES == ("correct",)
    assert tx.SUPPLEMENTARY_LEAVES == ("supplementary_true", "supplementary_false")
    assert set(tx.THINKY_LEAVES) == {"model_state", "unverifiable"}
    assert "world_true" not in tx.LEAVES and "possible_belief" not in tx.LEAVES
    assert tx.LEAF_RENAMES["world_true"] == "supplementary_true"
    assert tx.LEAF_RENAMES["possible_belief"] == "supplementary_false"


# ------------------------------------------------------------------ prompt


@pytest.mark.parametrize("system", [cm.SYSTEM, cm.SYSTEM_V5C])
def test_supplementary_routing_is_in_the_prompt(system):
    assert tx.SETTLEMENT_RULE in system
    flat = system.replace("**", "")
    assert "The test is settlement, not subject." in flat
    assert '"Picasso painted Guernica" when the text only discusses the Blue Period' in flat
    assert "is supplementary if the conversation is silent on it" in flat
    for lf in tx.SUPPLEMENTARY_LEAVES:
        assert f"- `{lf}`: {tx.LEAVES[lf].definition}" in system
    assert "### supplementary claim" in system
    assert "`unverifiable` with `world = true` if world knowledge" in system  # rev 16
    assert "world_true" not in system and "possible_belief" not in system
    assert "three top branches" in system


def test_single_call_prompt_has_no_established_flags():
    units, _ = split_cell(["Picasso painted it. Fine."], [["Picasso"]])
    assert units[0].established
    system, user = cm.render_prompt("Who painted it?", "Leonardo⟦READ⟧ did.", units)
    assert system == cm.SYSTEM
    assert "ESTABLISHED" not in system and "ESTABLISHED" not in user
    assert "established" not in system.lower()
    assert "dispute" not in user and "alt_leaf" not in user
    assert "Established spans must be incorrect." not in system
    assert unit_lines(user) == [
        ("R0.U1", False, "Picasso"),
        ("R0.U2", False, "painted it."),
        ("R0.U3", False, "Fine."),
    ]
    props = cm.SCHEMA["schema"]["properties"]["units"]["items"]["properties"]
    assert "dispute" not in props and "alt_leaf" not in props and "probe_target" in props
    for s in tx.definitions():
        assert s in cm.SYSTEM, s
    for s in tx.definitions(anchored=True):
        assert s in cm.SYSTEM_V5C, s
    # the anchored prompt keeps rule 2 and the dispute tag
    _s, user_v5c = cm.render_prompt("q", "a⟦READ⟧", units, anchor="v5c")
    assert "ESTABLISHED" in user_v5c and "Established spans must be incorrect." in cm.SYSTEM_V5C
    with pytest.raises(ValueError):
        cm.render_prompt("q", "a⟦READ⟧", units, anchor="bogus")


def test_plain_units_keep_ids_and_offsets():
    units, _ = split_cell(["Picasso painted it."], [["Picasso"]])
    plain = cm.plain_units(units)
    assert [(u.id, u.start, u.end, u.text) for u in plain] == [
        (u.id, u.start, u.end, u.text) for u in units
    ]
    assert not any(u.established for u in plain)


# ------------------------------------------------------------------ probe targets


def test_probe_target_parsing_and_missing_count():
    units = split_cell(["The model thinks it is a test. Next comes therapy. Paris is big. Yes."])[0]
    raw = {
        "units": [
            # rev 19: model_state is the thinky verdict; supplementary_false is a probe candidate
            e("R0.U1", "supplementary_false", probe_target="  this is an evaluation "),
            e("R0.U2", "ambiguous_topic"),  # missing probe_target: counted, leaf kept
            e(
                "R0.U3",
                "junk",
                splits=[
                    sp(
                        "Paris",
                        "substitution_sibling",
                        pair={"said": "Paris", "actual": "Rome"},
                        probe_target="Paris",
                    )
                ],
                rest_leaf="correct",
                probe_target="dropped on the rest: not a probe candidate",
            ),
            e("R0.U4", "correct", probe_target="not a probe candidate"),
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units)
    got = [(s.text, s.leaf, s.probe_target) for s in segs]
    assert got == [
        ("The model thinks it is a test.", "supplementary_false", "this is an evaluation"),
        ("Next comes therapy.", "ambiguous_topic", None),
        ("Paris", "substitution_sibling", "Paris"),
        ("is big.", "correct", None),
        ("Yes.", "correct", None),
    ]
    assert c["n_probe_candidates"] == 3 and c["n_missing_probe_targets"] == 1
    assert segs[2].pair == {"said": "Paris", "actual": "Rome"}


def test_probe_target_on_a_rest_piece_falls_back_to_the_unit():
    units = split_cell(["Hmm, the user is testing me."])[0]
    raw = {
        "units": [
            e(
                "R0.U1",
                "junk",
                splits=[sp("Hmm,", "junk")],
                rest_leaf="ambiguous_topic",  # rev 19: a probe candidate (model_state is not)
                probe_target="this is a test",
            )
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units)
    assert [(s.leaf, s.probe_target) for s in segs] == [
        ("junk", None),
        ("ambiguous_topic", "this is a test"),
    ]
    assert c["n_missing_probe_targets"] == 0


# ------------------------------------------------------------------ single-call pipeline


def test_single_call_map_needs_no_span_or_verify_and_subsamples_v5c(
    tmp_path, prose_readouts, monkeypatch
):
    picked = "chat-dailydialog-0000"
    monkeypatch.setattr(wrun, "in_v5c_subsample", lambda item_id, seed=0: item_id == picked)
    out = tmp_path / "out"
    fake = MapFake()
    _run([("toy", prose_readouts)], out, fake, stages=["map"])
    cells = [c for c in _cells(out, "toy") if c["status"] != "empty"]
    n_picked = sum(c["id"] == picked for c in cells)
    assert 0 < n_picked < len(cells)
    assert fake.n("map") == len(cells) == 15
    assert fake.n("span") == n_picked  # only the v5c comparison subsample
    assert fake.n("verify") == 0 and fake.n("classify") == 0
    for c in cells:
        assert c["stages"]["map"] == "ok" and c["map"]["anchor"] == "none"
        assert not any(s["established"] for s in c["map"]["segments"])
        assert c["map"]["counts"]["n_established_units"] == 0
        assert (c["stages"]["span"] == "ok") == (c["id"] == picked)
    user = next(c[3] for c in fake.calls if c[0] == "map")
    assert "ESTABLISHED" not in user
    d = wrun.arm_dir(out, wrun.Models(), "toy")
    s2 = json.loads((d / "summary_v2.json").read_text())
    assert s2["config"]["map_anchor"] == "none" and s2["config"]["pre_gate"] is True
    cmp_ = s2["overall"]["v5c_comparison"]
    assert 0 < cmp_["n_compared"] < cmp_["n_readouts"]
    # the readouts of cells without a span result still count everywhere else
    assert s2["overall"]["n_rate_readouts"]["all"] == cmp_["n_readouts"]
    assert sum(v["value"] for v in s2["overall"]["char_share_verdict"].values()) == pytest.approx(
        1.0
    )
    # re-run: cached; anchored mode is a different prompt and needs the span stage
    fake2 = MapFake()
    _run([("toy", prose_readouts)], out, fake2, stages=["map"])
    assert fake2.calls == []
    fake3 = MapFake()
    _run([("toy", prose_readouts)], out, fake3, V5C, stages=["map"])
    assert fake3.n("map") == 15 and fake3.n("span") == 15 - n_picked


def test_v5c_subsample_is_seeded_and_about_ten_percent():
    ids = sorted({it["id"] for it in bank.load_items()})
    a = [i for i in ids if wrun.in_v5c_subsample(i)]
    assert a == [i for i in ids if wrun.in_v5c_subsample(i)]  # deterministic
    assert 0.05 < len(a) / len(ids) < 0.15
    assert a != [i for i in ids if wrun.in_v5c_subsample(i, seed=1)]


# ------------------------------------------------------------------ metrics: verdicts + v5c


def _nov5c(rec):
    rec = dict(rec)
    rec["verdict"] = None
    rec["status"] = "pending"  # single-call: the span stage has no result for this cell
    return rec


def test_verdict_shares_rates_and_the_v5c_comparison():
    recs = [
        _rec("a", 44, [[("false_claim", 30), ("junk", 10)]], cls="hallucinated"),
        _rec("b", 44, [[("model_state", 5), ("supplementary_true", 5)]], cls="consistent"),
        _rec("c", 44, [[("unverifiable", 8)]], cls="generic"),
        _nov5c(_rec("d", 44, [[("detail", 4), ("junk", 4)]])),
    ]
    acc = score_v2.AccumulatorV2()
    for r in recs:
        acc.add(r)
    b = score_v2.block(acc.cells, n_boot=50)
    cv = {v: x["value"] for v, x in b["char_share_verdict"].items()}
    assert set(cv) == set(tx.VERDICT_ROWS)
    assert sum(cv.values()) == pytest.approx(1.0)
    assert cv["hallucination"] == pytest.approx((0.75 + 0.5) / 4)
    assert cv["probe_candidate"] == 0  # rev 19: model_state is thinky
    assert cv["thinky"] == pytest.approx(0.5 / 4)
    assert cv["correct"] == pytest.approx(0.5 / 4)
    assert cv["unresolved"] == pytest.approx(1 / 4)
    assert cv["junk"] == pytest.approx((0.5 + 0.25) / 4)  # rev 18: off-topic + non-claim
    rates = b["rates"]
    assert b["n_rate_readouts"]["all"] == 4
    assert rates["verdict_hallucination"]["all"]["value"] == pytest.approx(2 / 4)
    assert rates["verdict_thinky"]["all"]["value"] == pytest.approx(1 / 4)
    assert rates["verdict_junk_only"]["all"]["value"] == 0
    assert rates["verdict_hallucination"]["all"]["ci95"] is not None
    cmp_ = b["v5c_comparison"]
    assert cmp_["n_compared"] == 3 and cmp_["n_readouts"] == 4  # cell d has no v5c verdict
    assert cmp_["v5c_rate"]["value"] == pytest.approx(1 / 3)
    assert cmp_["v2_rate"]["value"] == pytest.approx(1 / 3)
    assert cmp_["v5c_rate_specific"]["value"] == pytest.approx(1 / 2)  # a, b are specific
    assert cmp_["v2_rate_specific"]["value"] == pytest.approx(1 / 2)
    assert "None" not in b["transition"] and None not in b["transition"]


# ------------------------------------------------------------------ calibration


def test_old_gold_names_are_remapped_on_load(tmp_path):
    c = {
        "case_id": "old",
        "part": "synthetic",
        "rules": [],
        "prompt": "Why is the sky blue?",
        "response": "Rayleigh scattering⟦READ⟧ makes it blue.",
        "readouts": ["Lord Rayleigh found it. Mars has a blue sky. Fine and dandy."],
    }
    c = cal.skeleton(c)
    for g, lf in zip(c["gold"]["units"], ["world_true", None, "junk"], strict=True):
        g["leaf"] = lf
    g = c["gold"]["units"][1]
    g["splits"] = [{"text": "Mars", "leaf": "possible_belief"}]
    g["rest_leaf"] = "possible_belief"
    assert cal.validate_case(c) == []  # old names are accepted
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps(c) + "\n")
    (loaded,) = cal.load_cases("dev", p)
    gu = loaded["gold"]["units"]
    assert gu[0]["leaf"] == "supplementary_true"
    assert gu[1]["splits"][0]["leaf"] == "supplementary_false"
    assert gu[1]["rest_leaf"] == "supplementary_false"
    assert c["gold"]["units"][0]["leaf"] == "world_true"  # the input is not changed


def test_verdict_accuracy_gate_row_and_probe_target_anchors():
    c = cal.skeleton(
        {
            "case_id": "v",
            "part": "synthetic",
            "rules": [],
            "prompt": "q",
            "response": "a⟦READ⟧ b",
            "readouts": ["The model is being tested. Paris is lovely. Rome was founded."],
        }
    )
    for g, lf in zip(
        c["gold"]["units"],
        ["ambiguous_topic", "unverifiable", "supplementary_true"],  # rev 19: a probe leaf
        strict=True,
    ):
        g["leaf"] = lf
    c["gold"]["units"][0]["probe_target"] = "this is a test"
    units = cal.mode_units(c)
    pred = cm.parse(
        {
            "units": [
                e("R0.U1", "ambiguous_topic", probe_target="a test"),  # same verdict
                e("R0.U2", "unverifiable"),  # same verdict (unresolved), other group
                e("R0.U3", "false_claim"),  # wrong verdict
            ],
            "rationale": "",
        },
        units,
    )
    m = cal.run_metrics([c], [pred])
    assert m["verdict_acc_unit"] == pytest.approx(2 / 3)
    # U1 (ambiguous_topic both sides) and U2 (unverifiable both sides) share their group
    assert m["group_acc_unit"] == pytest.approx(2 / 3)
    assert m["probe_targets"] == [
        {
            "case_id": "v",
            "unit": "R0.U1",
            "gold_leaf": "ambiguous_topic",
            "pred_leaf": "ambiguous_topic",
            "gold": "this is a test",
            "pred": "a test",
        }
    ]
    res = cal.score([c], [[pred], [pred]], "dev")
    rows = {g["key"]: g for g in res["gate"]}
    assert rows["verdict_acc_unit"]["threshold"] == 0.85
    assert rows["verdict_acc_unit"]["measure"] == "verdict accuracy (6 verdicts), unit level"
    assert rows["verdict_acc_unit"]["passed"] is False
    assert res["anchor"] == "none" and len(res["probe_targets"]) == 2
    assert "verdict accuracy" in cal.render_markdown(res)


def test_dev_v2_has_rev13_anchors():
    cases = cal.load_cases("dev")
    cov = cal.coverage(cases)
    assert cov["segments_per_leaf"]["unverifiable"] >= 3
    probes = [
        g
        for c in cases
        for g in c["gold"]["units"]
        if g.get("probe_target") and g["leaf"] in tx.PROBE_CANDIDATE_LEAVES
    ]
    assert len(probes) >= 3
    rules = {r for c in cases for r in c["rules"]}
    assert {"settlement-not-subject", "supplementary-unsettled", "probe-target"} <= rules
    raw = (cal.CAL_DIR / "dev_v2.jsonl").read_text()
    assert '"world_true"' not in raw and '"possible_belief"' not in raw


def test_gate_passed_reads_the_heldout_result(tmp_path):
    assert cal.gate_passed(tmp_path, "claude-opus-5-5") is False
    d = tmp_path / "calibration" / "claude-opus-5-5"
    d.mkdir(parents=True)
    (d / f"{cal.result_stem('heldout')}.json").write_text(json.dumps({"gate_passed": True}))
    assert cal.gate_passed(tmp_path, "claude-opus-5-5") is True
    assert cal.gate_passed(tmp_path, "claude-opus-5-5", "v5c") is False


# ------------------------------------------------------------------ report


def test_report_writes_probe_candidates_and_marks_pre_gate(tmp_path, prose_readouts):
    def mapper(user):
        out = []
        for uid, _est, text in unit_lines(user):
            if "Paris" in text:
                out.append(
                    e(
                        uid,
                        "substitution_sibling",
                        pair={"said": "Paris", "actual": "London"},
                        probe_target="Paris",
                        contradicted_by=prompt_quote(user),
                    )
                )
            else:
                out.append(e(uid, "ambiguous_topic"))  # probe candidate without a probe_target
        return {"units": out, "rationale": "r"}

    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake({"map": mapper}), stages=["map"])
    root = report_v2.report(out, "claude-opus-5-5", audit_n=2, verdict_n=3)
    rows = [json.loads(x) for x in (root / "probe_candidates.jsonl").read_text().splitlines()]
    assert rows and not (root / "substitution_pairs.jsonl").exists()
    keys = {"arm", "layer", "item", "pos", "leaf", "probe_target", "actual", "segment", "readout"}
    assert all(keys <= set(r) and r["pre_gate"] is True for r in rows)
    sub = [r for r in rows if r["leaf"] == "substitution_sibling"]
    assert sub and sub[0]["probe_target"] == "Paris" and sub[0]["actual"] == "London"
    assert any(r["probe_target"] is None for r in rows)
    md = (root / "report_v2.md").read_text()
    assert "PRE-GATE" in md and "Headline: character-weighted verdict shares" in md
    assert "v2 hallucination rate vs v5c on the same readouts" in md
    assert "Probe candidates by leaf" in md
    audit = (root / "audit_samples_v2.md").read_text()
    assert "PRE-GATE" in audit and "## verdict probe_candidate" in audit
    assert json.loads((root / "report_v2.json").read_text())["pre_gate"] is True
