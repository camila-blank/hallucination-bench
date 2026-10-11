"""claimmap-v2 (plan 0002): taxonomy render, prompt, schema, parse (splits, rest, missing units,
the forced / disputed established-span rules), validation, the :M stage and --stages, and the
v2 metrics (identities, character-weighted shares)."""

import asyncio
import json
import re
from pathlib import Path

import pytest
from conftest import FakeLLM

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cli import build_parser
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.evals.wsbench_chat import score_v2
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.llm import CallResult, Usage
from hallucination_bench.units import split_cell

# single-call lines are `id | text`; anchored (--anchor v5c) lines are `id | flag | text`
_UNIT = re.compile(r'^(R\d+\.U\d+) \| (?:(ESTABLISHED|-) \| )?(".*")$', re.MULTILINE)


def unit_lines(user: str) -> list[tuple[str, bool, str]]:
    return [(m[1], m[2] == "ESTABLISHED", json.loads(m[3])) for m in _UNIT.finditer(user)]


def e(uid, leaf="correct", **kw):
    d = {
        "id": uid,
        "leaf": leaf,
        "hedged": False,
        "world_false": False,
        "cruxy": False,
        "dispute": False,
        "alt_leaf": None,
        "pair": None,
        "probe_target": None,
        "reason": None,
        "source": None,
        "same_slot_alternative": False,
        "world": False,
        "prediction": False,
        "contradicted_by": None,
        "supported_by": None,
        "why": "w",
        "splits": [],
        "rest_leaf": None,
    }
    d.update(kw)
    return d


def sp(text, leaf, **kw):
    d = {k: v for k, v in e("x", leaf, **kw).items() if k not in ("id", "splits", "rest_leaf")}
    return {"text": text, **d}


def prompt_quote(user: str) -> str:
    """A verbatim span of the conversation (the start of the prompt) for evidence quotes."""
    return user.split("<prompt>\n", 1)[1].split("\n</prompt>", 1)[0][:30]


def default_map(user: str) -> dict:
    q = prompt_quote(user)  # rev 17: incorrect / correct segments must quote the conversation
    out = []
    for uid, est, text in unit_lines(user):
        if est:
            leaf = "detail"
        elif text.startswith("<"):
            leaf = "non_claim"
        elif "Paris" in text:
            leaf = "false_claim"
        else:
            leaf = "correct"
        quotes = {
            "contradicted_by": q if leaf in tx.INCORRECT_LEAVES else None,
            "supported_by": q if leaf == "correct" else None,
        }
        out.append(e(uid, leaf, **quotes))
    return {"units": out, "rationale": "r"}


class MapFake(FakeLLM):
    def __init__(self, responders=None):
        super().__init__({"map": default_map, **(responders or {})})

    async def call(self, *, model, system, user, schema):
        if system in (cm.SYSTEM, cm.SYSTEM_V5C):
            self.calls.append(("map", model, system, user))
            return CallResult(self.responders["map"](user), Usage(calls=1, usd=0.02))
        return await super().call(model=model, system=system, user=user, schema=schema)


# ------------------------------------------------------------------ taxonomy + prompt


def test_tree_has_12_leaves_in_8_groups_and_pseudo_rows():
    assert len(tx.LEAF_NAMES) == 12 and len(tx.GROUPS) == 8
    assert set(tx.GROUP_OF.values()) == set(tx.ROWS)
    assert tx.SUBSTITUTION_LEAVES == ("substitution_sibling", "substitution_category")
    assert {tx.group_of(x) for x in ("detail", "false_claim")} == {tx.HALLUCINATION}
    assert tx.group_of("unverifiable") == tx.THINKY
    assert tx.group_of("off_topic") == tx.JUNK and tx.group_of("ambiguous_topic") == tx.UNSETTLED


def test_prompt_contains_every_leaf_definition_rule_and_tag():
    for s in tx.definitions():
        assert s in cm.SYSTEM, s
    for lf in tx.LEAVES.values():
        assert f"- `{lf.name}`: {lf.definition}" in cm.SYSTEM
    assert tx.UNKNOWN_INCORRECT not in cm.SYSTEM  # pseudo-leaves are code-only


def test_precedence_is_listed_in_the_planned_order():
    pos = [cm.SYSTEM.index(f"\n{n}. **{tx.RULES[n].title}**") for n in tx.RULE_ORDER]
    assert pos == sorted(pos)
    assert tx.RULE_ORDER == (1, 3, 4, 5, 6, 10, 11, 12, 13, 7, 8, 9)  # single-call: no rule 2
    assert tx.RULE_ORDER_ANCHORED == (2, 1, 3, 4, 5, 6, 10, 11, 12, 13, 7, 8, 9)  # rev 16: no 14
    pos = [
        cm.SYSTEM_V5C.index(f"\n{n}. **{tx.RULES_ANCHORED[n].title}**")
        for n in tx.RULE_ORDER_ANCHORED
    ]
    assert pos == sorted(pos)
    # revision-10 wording
    # rev 16: predictions are ordinary claims, so the "non-prediction" qualifier is gone
    assert "A world claim is incorrect only if the conversation contradicts" in (
        cm.SYSTEM.replace("**", "")
    )
    # rev 15: unrealised continuations are incorrect only if the text contradicts them
    assert "incorrect if the text obviously contradicts it, before or after the marker" in cm.SYSTEM
    assert "Hedge words stay inside their claim segment" in cm.SYSTEM


def test_open_decisions_default_and_switch(monkeypatch):
    assert tx.CHOSEN == {k: d.default for k, d in tx.OPEN_DECISIONS.items()}
    assert "Picasso for da Vinci is `substitution_sibling`" in cm.SYSTEM
    assert tx.decision_text("hedged_inference") in cm.SYSTEM
    monkeypatch.setitem(tx.CHOSEN, "subject_name", "false_claim")
    assert "it is `false_claim`, not a substitution" in tx._rules()[8].body[1]


def test_render_prompt_lists_units_with_established_flags():
    units, _ = split_cell(["Picasso painted it. Fine.", "{braces} ok"], [["Picasso"], []])
    system, user = cm.render_prompt("Who {painted}?", "Leonardo⟦READ⟧ did.", units, anchor="v5c")
    assert system == cm.SYSTEM_V5C and "Who {painted}?" in user
    assert unit_lines(user) == [
        ("R0.U1", True, "Picasso"),
        ("R0.U2", False, "painted it."),
        ("R0.U3", False, "Fine."),
        ("R1.U1", False, "{braces} ok"),
    ]
    assert cm.TOKENS_NOTE not in user
    assert cm.TOKENS_NOTE in cm.render_prompt("p", "r⟦READ⟧", units, kind="tokens")[1]
    with pytest.raises(ValueError):
        cm.render_prompt("p", "r", [])


@pytest.mark.parametrize("schema", [cm.SCHEMA, cm.SCHEMA_V5C])
def test_schema_is_strict_with_the_leaf_enum(schema):
    item = schema["schema"]["properties"]["units"]["items"]
    assert item["additionalProperties"] is False and set(item["required"]) == set(
        item["properties"]
    )
    assert item["properties"]["leaf"]["enum"] == list(tx.LEAF_NAMES)
    split = item["properties"]["splits"]["items"]
    assert split["additionalProperties"] is False and "text" in split["required"]


# ------------------------------------------------------------------ parse


def _units(text="Alpha is red and Paris is big. Second one.", est=()):
    return split_cell([text], [list(est)])[0]


def test_parse_splits_rest_and_unlocated_subspans():
    units = _units()
    raw = {
        "units": [
            e(
                "R0.U1",
                "non_claim",  # ignored when splitting
                splits=[
                    sp("alpha is red", "correct"),
                    sp("nowhere to be found", "detail"),
                    sp("Paris is big", "false_claim", cruxy=True),
                ],
                rest_leaf="non_claim",
            ),
            e("R0.U2", "correct", cruxy=True),
        ],
        "rationale": "r",
    }
    segs, c = cm.parse(raw, units)
    got = [(s.text, s.leaf, s.kind) for s in segs]
    assert got == [
        ("Alpha is red", "correct", "split"),
        ("and", "non_claim", "rest"),
        ("Paris is big", "false_claim", "split"),
        (".", "non_claim", "rest"),
        ("Second one.", "correct", "unit"),
    ]
    assert c["n_splits"] == 3 and c["n_unlocated_subspans"] == 1
    assert segs[2].cruxy and not segs[4].cruxy  # cruxy only on incorrect segments
    text = "Alpha is red and Paris is big. Second one."
    assert all(text[s.start : s.end] == s.text for s in segs)


def test_parse_overlapping_split_is_rejected_and_missing_rest_falls_back():
    units = _units("one two three")
    raw = {
        "units": [
            e("R0.U1", "non_claim", splits=[sp("one two", "detail"), sp("two three", "detail")])
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units)
    assert c["n_unlocated_subspans"] == 1 and c["n_missing_rest"] == 1
    assert [(s.text, s.leaf) for s in segs] == [("one two", "detail"), ("three", "non_claim")]


def test_parse_missing_duplicate_extra_and_failed_call():
    units = _units("A one. B two. C three.")
    raw = {
        "units": [e("R0.U1"), e("r0.u1", "detail"), e("R0.U9"), e("R0.U3", "not_a_leaf")],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units)
    assert [s.leaf for s in segs] == ["correct", tx.FAILED, tx.FAILED]
    assert [s.status for s in segs] == ["ok", "missing", "bad_leaf"]
    assert c["n_duplicate_units"] == 1 and c["n_extra_units"] == 1
    assert c["n_missing_units"] == 1 and c["n_bad_leaves"] == 1
    segs, c = cm.parse(None, units)
    assert {s.leaf for s in segs} == {tx.FAILED} and c["n_missing_units"] == 0


def test_validate_readout_count_and_missing_share():
    units, _ = split_cell([". ".join(f"Item {c}" for c in "abcdefghijkl") + ".", "Other one."])
    assert len(units) == 13
    ids = [u.id for u in units]
    full = {"units": [e(i) for i in ids], "rationale": ""}
    assert cm.validate(full, units)
    one_missing = {"units": [e(i) for i in ids[1:]], "rationale": ""}
    assert cm.validate(one_missing, units)  # 1 of 13 = 7.7% <= 10%
    two_missing = {"units": [e(i) for i in ids[2:]], "rationale": ""}
    assert not cm.validate(two_missing, units)  # 15% > 10%
    no_r1 = {"units": [e(i) for i in ids if i.startswith("R0")], "rationale": ""}
    assert not cm.validate(no_r1, units)  # wrong readout count
    extra_r = {"units": [*full["units"], e("R5.U1")], "rationale": ""}
    assert not cm.validate(extra_r, units)
    assert not cm.validate({"units": "x"}, units) and not cm.validate(None, units)


def test_established_unit_forced_disputed_and_kept():
    text = "It was Picasso. It was Monet. It was Dali."
    units = _units(text, ["It was Picasso", "It was Monet", "It was Dali"])
    est = [u.id for u in units if u.established]
    assert len(est) == 3
    raw = {
        "units": [
            e(est[0], "substitution_sibling", pair={"said": "Picasso", "actual": "da Vinci"}),
            e(est[1], "correct"),  # non-incorrect, no dispute -> forced
            e(est[2], "detail", dispute=True, alt_leaf="correct"),  # disputed
            *(e(u.id, "non_claim") for u in units if not u.established),
        ],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units)
    by = {s.unit: s for s in segs if s.established}
    assert by[est[0]].leaf == "substitution_sibling" and by[est[0]].status == "ok"
    assert by[est[1]].leaf == tx.UNKNOWN_INCORRECT and by[est[1]].status == "forced"
    assert by[est[1]].raw_leaf == "correct"
    assert by[est[2]].leaf == tx.UNKNOWN_INCORRECT and by[est[2]].status == "disputed"
    assert by[est[2]].alt_leaf == "correct"
    assert c["n_forced_established"] == 1 and c["n_disputed_established"] == 1
    assert c["n_missing_pairs"] == 0
    assert cm.resolved_leaf(by[est[1]]) == "correct"  # forced: raw leaf
    assert cm.resolved_leaf(by[est[2]]) == "correct"  # disputed: alt_leaf
    assert cm.alt_leaf_view(by[est[1]]) == tx.UNKNOWN_INCORRECT


def test_established_split_needs_one_incorrect_subspan():
    units = _units("Picasso painted the Mona Lisa", ["Picasso painted the Mona Lisa"])
    (u,) = units
    ok = {
        "units": [e(u.id, splits=[sp("Picasso", "substitution_sibling")], rest_leaf="correct")],
        "rationale": "",
    }
    segs, c = cm.parse(ok, units)
    assert [s.leaf for s in segs] == ["substitution_sibling", "correct"]
    assert c["n_forced_established"] == 0 and c["n_missing_pairs"] == 1
    forced = {
        "units": [
            e(
                u.id,
                splits=[
                    sp("Picasso", "detail", dispute=True, alt_leaf="correct"),
                    sp("painted", "non_claim"),
                ],
                rest_leaf="correct",
            )
        ],
        "rationale": "",
    }
    segs, c = cm.parse(forced, units)
    assert [(s.leaf, s.status) for s in segs] == [
        (tx.UNKNOWN_INCORRECT, "disputed"),
        (tx.UNKNOWN_INCORRECT, "forced"),
        (tx.UNKNOWN_INCORRECT, "forced"),  # the rest counts as a sub-span
    ]
    assert c["n_disputed_established"] == 1 and c["n_forced_established"] == 2


def test_dispute_on_an_ordinary_unit_is_ignored():
    units = _units("Plain claim.")
    raw = {
        "units": [e("R0.U1", "supplementary_true", dispute=True, alt_leaf="detail")],
        "rationale": "",
    }
    segs, c = cm.parse(raw, units)
    assert segs[0].leaf == "supplementary_true" and not segs[0].dispute and segs[0].alt_leaf is None
    assert c["n_dispute_ignored"] == 1


# ------------------------------------------------------------------ pipeline: the :M stage


def _run(arms, out: Path, fake, models=None, **kw):
    return asyncio.run(
        wrun.run_arms_async(
            arms,
            out,
            models or wrun.Models(),
            layers=[44],
            limit=None,
            rpm=1e9,
            concurrency=4,
            allow_missing=True,
            llm=fake,
            **kw,
        )
    )


def _cells(out: Path, arm: str) -> list[dict]:
    d = wrun.arm_dir(out, wrun.Models(), arm)
    return [json.loads(x) for x in (d / "cells.jsonl").read_text().splitlines()]


def test_stage_keys_and_dependencies():
    assert wrun.stage_of_key("k__L044__p3:M") == "map"
    assert wrun.expand_stages(["map"]) == {"map", "summary"}  # single-call: no span, no verify
    assert wrun.expand_stages(["map"], anchor="v5c") == {"map", "span", "summary"}
    assert wrun.expand_stages(["classify"]) == {"classify", "verify", "span", "summary"}
    assert wrun.expand_stages(None) == set(wrun.V1_STAGES)
    assert "classify" not in wrun.expand_stages(None, classify=False)
    with pytest.raises(ValueError):
        wrun.expand_stages(["bogus"])
    args = build_parser().parse_args(["run", "--arm", "a=b"])
    assert args.stages is None  # the CLI default (every stage) is applied in cmd_run


V5C = wrun.Models(map_anchor="v5c")


def test_map_stage_runs_span_first_and_resumes(tmp_path, prose_readouts):
    """--anchor v5c (plan rev 12 behaviour)."""
    out = tmp_path / "out"
    fake = MapFake()
    s = _run([("toy", prose_readouts)], out, fake, V5C, stages=["map"])["toy"]
    assert fake.n("verify") == 0 and fake.n("classify") == 0
    assert fake.n("span") == 15 and fake.n("map") == 15
    rec = _cells(out, "toy")[0]
    assert rec["stages"]["map"] == "ok" and rec["stages"]["verify"] == "pending"
    leaves = [x["leaf"] for x in rec["map"]["segments"]]
    assert leaves[0] == "correct" and "detail" in leaves  # Paris is established
    est = [x for x in rec["map"]["segments"] if x["established"]]
    assert [x["text"] for x in est] == ["Paris"]
    s2 = json.loads((wrun.arm_dir(out, wrun.Models(), "toy") / "summary_v2.json").read_text())
    assert (
        s2["health"]["n_map_ok"] == 15
        and s2["config"]["prompt_versions"]["map"] == cm.PROMPT_VERSION
    )
    assert s["health"]["n_failed_map"] == 0
    fake2 = MapFake()
    _run([("toy", prose_readouts)], out, fake2, V5C, stages=["map"])
    assert fake2.calls == []  # all cached


def test_invalid_map_answer_is_cached_failed_and_requeued(tmp_path, prose_readouts):
    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake({"map": lambda u: {"units": []}}), stages=["map"])
    rec = _cells(out, "toy")[0]
    assert rec["stages"]["map"] == "failed"
    assert {x["leaf"] for x in rec["map"]["segments"]} == {tx.FAILED}
    fake = MapFake()
    _run([("toy", prose_readouts)], out, fake, stages=["map"])
    assert fake.n("map") == 15 and fake.n("span") == 0
    assert _cells(out, "toy")[0]["stages"]["map"] == "ok"


def test_token_arm_maps_the_summary(tmp_path, token_readouts):
    out = tmp_path / "out"
    fake = MapFake()
    _run([("tok", token_readouts)], out, fake, stages=["map"])
    assert fake.n("summary") == 3 and fake.n("map") == 3
    user = next(c[3] for c in fake.calls if c[0] == "map")
    assert cm.TOKENS_NOTE in user
    # single-call (default): no established span, so "Paris" is not cut out
    assert [t for _i, _e, t in unit_lines(user)] == ["A friendly note about Paris and the teacher."]
    out2 = tmp_path / "out2"
    fake = MapFake()
    _run([("tok", token_readouts)], out2, fake, V5C, stages=["map"])
    user = next(c[3] for c in fake.calls if c[0] == "map")
    assert [t for _i, _e, t in unit_lines(user)] == [
        "A friendly note about",
        "Paris",
        "and the teacher.",
    ]


# ------------------------------------------------------------------ metrics


def _rec(item, layer, readouts, *, map_status="ok", cls="consistent"):
    """readouts: list of segment lists [(leaf, chars, extra-dict)]."""
    segs = []
    verdict = []
    for i, rs in enumerate(readouts):
        pos = 0
        for leaf, n, *extra in rs:
            d = {
                "unit": f"R{i}.U1",
                "readout": i,
                "start": pos,
                "end": pos + n,
                "text": "x" * n,
                "leaf": leaf,
                "raw_leaf": leaf,
                "status": "ok",
                "established": False,
                "kind": "unit",
                "alt_leaf": None,
                "hedged": False,
                "world_false": False,
                "cruxy": False,
                "dispute": False,
                "pair": None,
                "why": "",
            }
            d.update(extra[0] if extra else {})
            segs.append(d)
            pos += n + 1
        verdict.append({"idx": i, "class": cls, "quotes": None})
    return {
        "id": item,
        "layer": layer,
        "status": "judged",
        "stages": {"map": map_status},
        "samples": ["y" * 100 for _ in readouts],
        "verdict": verdict,
        "map": {"segments": segs, "counts": {"n_units": len(segs)}},
    }


def test_character_weighted_shares_average_per_readout_and_identities_hold():
    recs = [
        # readout 0: 30 chars false_claim, 10 chars filler; readout 1: all correct
        _rec("a", 44, [[("false_claim", 30), ("non_claim", 10)], [("correct", 5)]]),
        _rec("b", 44, [[(tx.UNKNOWN_INCORRECT, 4, {"status": "forced"}), ("correct", 4)]]),
        _rec("c", 44, [[(tx.FAILED, 7)]], map_status="failed"),
        _rec("d", 44, [[("detail", 9)]], map_status="pending"),
    ]
    acc = score_v2.AccumulatorV2()
    for r in recs:
        acc.add(r)
    b = score_v2.block(acc.cells, n_boot=50)
    cs = {g: v["value"] for g, v in b["char_share_group"].items()}
    assert b["n_share_readouts"] == 4  # pending cell left out
    assert cs[tx.HALLUCINATION] == pytest.approx((30 / 40) / 4)
    assert cs[tx.CORRECT] == pytest.approx((1 + 0.5) / 4)
    assert cs[tx.UNKNOWN_INCORRECT] == pytest.approx(0.5 / 4)
    assert cs[tx.FAILED] == pytest.approx(1 / 4)
    assert sum(cs.values()) == pytest.approx(1.0)
    assert sum(v["value"] for v in b["char_share_leaf"].values()) == pytest.approx(1.0)
    assert sum(v["value"] for v in b["seg_share_group"].values()) == pytest.approx(1.0)
    # rates: over ok cells only (3 readouts); unknown_incorrect counts as hallucination
    rates = b["rates"]
    assert b["n_rate_readouts"]["all"] == 3
    assert rates["hallucination"]["all"]["value"] == pytest.approx(2 / 3)
    assert rates["hallucination_excl_unknown"]["all"]["value"] == pytest.approx(1 / 3)
    assert rates["correct"]["all"]["value"] == pytest.approx(2 / 3)
    assert b["rates"]["hallucination"]["all"]["ci95"] is not None
    assert b["health"]["n_map_pending"] == 1 and b["health"]["n_map_failed"] == 1


def test_specific_denominator_transition_and_alt_view():
    recs = [
        _rec("a", 44, [[("false_claim", 3)]], cls="hallucinated"),
        _rec("b", 44, [[("non_claim", 3)]], cls="generic"),
        _rec(
            "c",
            44,
            [[(tx.UNKNOWN_INCORRECT, 3, {"status": "disputed", "alt_leaf": "correct"})]],
            cls="hallucinated",
        ),
    ]
    acc = score_v2.AccumulatorV2()
    for r in recs:
        acc.add(r)
    b = score_v2.block(acc.cells, n_boot=20)
    assert b["n_rate_readouts"] == {"all": 3, "specific": 2}
    assert b["rates"]["hallucination"]["specific"]["value"] == 1.0
    assert b["rates"]["hallucination_alt"]["specific"]["value"] == 0.5
    assert b["rates"]["nonclaim_or_junk_only"]["all"]["value"] == pytest.approx(1 / 3)
    assert b["v5c_hallucination_rate_same_readouts"]["value"] == 1.0
    assert b["transition"] == {
        "generic": {tx.NON_CLAIM: 1},
        "hallucinated": {tx.HALLUCINATION: 1, tx.UNKNOWN_INCORRECT: 1},
    }
    assert b["char_share_group_alt"][tx.CORRECT]["value"] == pytest.approx(1 / 3)


def test_verify_agreement_uses_the_majority_leaf_of_the_quote():
    r = _rec("a", 44, [[("correct", 10), ("supplementary_false", 20)]])
    r["samples"] = ["a" * 10 + " " + "b" * 20]
    r["verdict"][0]["quotes"] = {
        "true": ["aaaaaaaaaa"],
        "unverifiable": ["bbbb"],
        "disputed": ["aaa bbbbbbbbbbbb"],
        "off_topic": ["zzz"],
    }
    acc = score_v2.AccumulatorV2()
    acc.add(r)
    ag = score_v2.agreement(acc.cells)
    assert ag["true"]["n_agree"] == 1 and ag["unverifiable"]["n_agree"] == 1
    assert ag["disputed"]["leaves"] == {"supplementary_false": 1} and ag["disputed"]["n_agree"] == 1
    assert ag["off_topic"]["n_unlocated"] == 1


def test_pilot_dump_v2_gives_unlabelled_cells_with_production_units(tmp_path, prose_readouts):
    from hallucination_bench import calibrate_v2, report_v2

    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, MapFake(), stages=["span"])
    cases = report_v2.pilot_dump_v2(out, "claude-opus-5-5", ["toy"], per_stratum=2, seed=0)
    assert cases and all(c["part"] == "real" for c in cases)
    classes = {c["source"]["v5c_cell_class"] for c in cases}
    assert "hallucinated" in classes
    for c in cases:
        assert calibrate_v2.validate_case(c, require_gold=False) == []
        assert all(g["leaf"] is None for g in c["gold"]["units"])
    hal = next(c for c in cases if c["source"]["v5c_cell_class"] == "hallucinated")
    assert any(g["established"] and g["text"] == "Paris" for g in hal["gold"]["units"])


def test_rev11_lens_framing_rule_is_rendered():
    assert tx.LENS_FRAMING_RULE in cm.SYSTEM
    assert "The model is representing / anticipating / focused on / gearing up for X" in cm.SYSTEM
    assert "It is **never `model_state`**" in cm.SYSTEM
    assert cm.PROMPT_VERSION == "claimmap-v2.7"
