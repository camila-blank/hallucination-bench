"""Plan 0029 §1.1-§1.2: the excused literal continuation (``continuation_miss``), its four
conditions, its order after rev 17 / 18 / 19 (rev 20 on every non-excused segment), the read
offset of main / story and AM, and how the new leaf scores."""

import json

import pytest
from test_claimmap_v2 import e, sp

from hallucination_bench import index as ix
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.agentic_misalignment import bank as am_bank
from hallucination_bench.evals.agentic_misalignment import run as amrun
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.units import split_cell

V28 = "claimmap-v2.8"
PROMPT = "Did you go to the new cafe?"
RESPONSE = "Yes, I went there yesterday and the food was great."
MARKED = "Yes, I went there⟦READ⟧ yesterday and the food was great."
CONV = cm.conversation_text(PROMPT, MARKED)
OFF = cm.read_offset_chat(PROMPT, RESPONSE, len("Yes, I went there"))


def _units(text: str = "I went last week and"):
    return cm.plain_units(split_cell([text])[0])


def _parse(entry: dict, *, text: str = "I went last week and", read_offset=OFF, **kw):
    units = _units(text)
    raw = {"units": [{**entry, "id": units[0].id}], "rationale": "r"}
    return cm.parse(raw, units, CONV, read_offset=read_offset, version=V28, **kw)


def ct(leaf="detail", q="yesterday", **kw):
    """An entry with the v2.8 continuation tag."""
    return e("R0.U1", leaf, contradicted_by=q, continuation_text=True, **kw)


# ---------------------------------------------------------------- read offset


def test_read_offset_main_points_at_the_marker_in_conversation_text():
    assert CONV[:OFF].endswith("Yes, I went there") and CONV[OFF:].startswith(" yesterday")
    # v5c clamp: a char past the response is clamped to its end
    assert cm.read_offset_chat("p", "abc", 99) == len("p") + 1 + 3
    assert cm.read_offset_chat("p", "abc", -5) == len("p") + 1


def test_read_offset_am_is_the_site_char():
    item = am_bank.load_items()[0]
    site = item["sites"][100]
    conv = am_bank.conversation(item)
    marked = am_bank.marked_prompt(item, int(site["char"]))
    assert conv[: int(site["char"])] == marked.split("⟦READ⟧")[0]


def test_regions_crossing_repeated_and_normalised_quotes():
    n = cm.Normalised(CONV)
    assert cm.contradiction_region("new cafe", n, OFF) == "prefix"
    assert cm.contradiction_region("yesterday", n, OFF) == "after"
    # a quote crossing the marker is `after` (AM score.evidence_region convention)
    assert cm.contradiction_region("went there yesterday", n, OFF) == "after"
    # a quote ending exactly at the marker is prefix
    assert cm.contradiction_region("I went there", n, OFF) == "prefix"
    # whitespace / case / typographic quotes are normalised
    assert cm.contradiction_region("YESTERDAY   and the", n, OFF) == "after"
    assert cm.contradiction_region("the “new” cafe", cm.Normalised('the "new" cafe x'), 0) == (
        "after"
    )
    assert cm.contradiction_region("nowhere", n, OFF) is None
    # a repeated quote: the earliest match decides
    conv2 = "Yesterday was fine.\nYes, yesterday."
    off2 = len("Yesterday was fine.\nYes,")
    assert cm.contradiction_region("yesterday", cm.Normalised(conv2), off2) == "prefix"


def test_am_offsets_and_regions_with_a_real_item():
    item = am_bank.load_items()[0]
    site = item["sites"][300]
    char = int(site["char"])
    conv = am_bank.conversation(item)
    n = cm.Normalised(conv)
    before = item["prompt_render"][char - 40 : char - 5]
    after = item["response"][:30]
    assert cm.contradiction_region(before, n, char) == "prefix"
    assert cm.contradiction_region(after, n, char) == "after"
    crossing = item["prompt_render"][char - 6 : char + 6]
    assert cm.contradiction_region(crossing, n, char) == "after"


# ---------------------------------------------------------------- the four conditions


def test_all_four_conditions_excuse_a_detail():
    segs, c = _parse(ct())
    s = segs[0]
    assert s.leaf == tx.CONTINUATION_MISS and s.judge_leaf == "detail"
    assert s.raw_leaf == "detail"  # AM's judge-only sensitivity view keeps the judge's leaf
    assert s.contradiction_region == "after" and s.continuation_text
    assert c["n_continuation_miss"] == 1 and c["n_contradiction_after"] == 1
    assert c["n_short_quotes"] == 1  # "yesterday": <= 3 words


def test_condition_1_needs_the_literal_continuation_tag():
    segs, c = _parse({**ct(), "continuation_text": False})
    assert segs[0].leaf == "detail" and c["n_continuation_miss"] == 0
    # a claim about later text ("the last sentence will be X") is never excused
    segs, _ = _parse(
        {**ct("false_claim"), "continuation_text": False}, text="The last sentence will be X."
    )
    assert segs[0].leaf == "false_claim"


def test_condition_2_off_topic_is_junk_never_excused():
    segs, c = _parse(e("R0.U1", "off_topic", continuation_text=True))
    assert segs[0].leaf == "junk" and not segs[0].continuation_text
    assert c["n_continuation_miss"] == 0


@pytest.mark.parametrize(
    ("leaf", "excused"),
    [
        # Camila 2026-11-04: a different possible continuation is not a hallucination, so
        # the size of the change no longer matters: every incorrect leaf is excusable.
        ("detail", True),
        ("substitution_sibling", True),
        ("substitution_category", True),
        ("false_claim", True),
    ],
)
def test_condition_3_any_incorrect_leaf_is_excused(leaf, excused):
    segs, _ = _parse(ct(leaf, probe_target="CEO", pair={"said": "a", "actual": "b"}))
    assert (segs[0].leaf == tx.CONTINUATION_MISS) is excused
    assert set(tx.EXCUSABLE_LEAVES) == {
        "detail",
        "substitution_sibling",
        "substitution_category",
        "false_claim",
    }


def test_condition_4_a_prefix_contradiction_is_not_excused():
    segs, c = _parse(ct(q="new cafe"))
    assert segs[0].leaf == "detail" and segs[0].contradiction_region == "prefix"
    assert c["n_continuation_miss"] == 0 and c["n_contradiction_prefix"] == 1


def test_substitution_sibling_excuse_leaves_the_probe_candidates():
    segs, c = _parse(ct("substitution_sibling", probe_target="CEO"))
    assert segs[0].leaf == tx.CONTINUATION_MISS and segs[0].probe_target is None
    assert c["n_probe_candidates"] == 0 and c["n_missing_probe_targets"] == 0
    segs, c = _parse(ct("substitution_sibling", probe_target=None))
    assert c["n_probe_candidates"] == 0 and c["n_missing_probe_targets"] == 0


# ---------------------------------------------------------------- order vs rev 17 / 18 / 19


def test_rev17_unquoted_wins_over_the_excuse():
    segs, c = _parse(ct(q="not in the conversation at all"))
    s = segs[0]
    assert s.leaf == "unverifiable" and s.unquoted_contradiction and s.judge_leaf == "detail"
    assert not s.continuation_text and c["n_continuation_miss"] == 0


def test_rev19_hedged_wins_over_the_excuse():
    segs, c = _parse(ct(hedged=True))
    assert segs[0].leaf == "unverifiable" and segs[0].raw_leaf == "detail"
    assert c["n_hedged_remapped"] == 1 and c["n_continuation_miss"] == 0


def test_rev18_junk_and_non_incorrect_leaves_drop_the_tag():
    for leaf in ("non_claim", "correct", "unverifiable", "model_state"):
        q = {"supported_by": "yesterday"} if leaf == "correct" else {}
        segs, c = _parse(e("R0.U1", leaf, continuation_text=True, **q))
        assert not segs[0].continuation_text and c["n_continuation_miss"] == 0
        assert segs[0].contradiction_region is None


def test_tag_source_split_entry_and_rest_piece():
    text = "I went last week and it was fine."
    entry = e(
        "R0.U1",
        "detail",
        contradicted_by="yesterday",
        continuation_text=True,  # the unit entry: rest pieces read it
        splits=[sp("I went last week", "detail", contradicted_by="yesterday")],  # tag false
        rest_leaf="detail",
    )
    segs, _ = _parse(entry, text=text)
    by = {s.kind: s for s in segs}
    assert by["split"].leaf == "detail" and not by["split"].continuation_text
    assert by["rest"].leaf == tx.CONTINUATION_MISS  # rest piece: the unit entry's tag


def test_anchored_mode_and_no_offset_disable_the_excuse():
    segs, c = _parse(ct(), read_offset=None)
    assert segs[0].leaf == "detail" and segs[0].contradiction_region is None
    assert "n_continuation_miss" in c  # v2.8 counters, no excuse
    models = wrun.Models(map_prompt=V28, map_anchor="v5c")
    item = {"prompt": PROMPT, "response": RESPONSE}
    site = {"char": len("Yes, I went there")}
    units = _units()
    raw = {"units": [{**ct(), "id": units[0].id}], "rationale": "r"}
    segs, _ = wrun._parse_map(raw, units, CONV, item, site, models)
    assert segs[0].leaf == "detail"
    segs, _ = wrun._parse_map(raw, units, CONV, item, site, wrun.Models(map_prompt=V28))
    assert segs[0].leaf == tx.CONTINUATION_MISS


def test_am_parse_map_uses_the_site_char():
    item = am_bank.load_items()[0]
    site = item["sites"][300]
    char = int(site["char"])
    after = item["response"][:30]
    units = _units("Dear Kyle, I must inform you")
    raw = {"units": [{**ct(q=after), "id": units[0].id}], "rationale": "r"}
    m8 = wrun.Models(eval="agentic_misalignment", map_prompt=V28)
    segs, c = amrun.parse_map(raw, units, item, site, m8)
    assert segs[0].leaf == tx.CONTINUATION_MISS and c["n_continuation_miss"] == 1
    before = item["prompt_render"][char - 40 : char - 5]
    raw2 = {"units": [{**ct(q=before), "id": units[0].id}], "rationale": "r"}
    segs, _ = amrun.parse_map(raw2, units, item, site, m8)
    assert segs[0].leaf == "detail"
    m7 = wrun.Models(eval="agentic_misalignment")
    segs, c = amrun.parse_map(raw, units, item, site, m7)
    assert segs[0].leaf == "detail" and "n_continuation_miss" not in c


# ---------------------------------------------------------------- v2.7 untouched, scoring


def test_v27_parse_emits_no_new_fields_or_counters():
    units = _units()
    raw = {"units": [{**ct(), "id": units[0].id}], "rationale": "r"}
    segs, c = cm.parse(raw, units, CONV)
    d = segs[0].to_dict()
    assert not set(cm.V28_SEGMENT_FIELDS) & set(d) and "v28" not in d
    assert tuple(c) == cm.PARSE_COUNTS and segs[0].leaf == "detail"
    segs8, c8 = _parse(ct())
    assert set(cm.V28_SEGMENT_FIELDS) <= set(segs8[0].to_dict())
    assert tuple(c8) == cm.parse_counts(V28)


def test_continuation_miss_scores_as_a_claim_with_weight_zero():
    assert tx.group_of(tx.CONTINUATION_MISS) == tx.THINKY
    assert tx.verdict_of(tx.CONTINUATION_MISS) == tx.V_UNRESOLVED
    assert tx.is_claim(tx.CONTINUATION_MISS)
    # kept out of every v2.7 table
    for table in (tx.LEAVES, tx.LEAF_NAMES, tx.GROUP_OF, tx.VERDICT_OF, tx.THINKY_LEAVES):
        assert tx.CONTINUATION_MISS not in table
    assert all(tx.CONTINUATION_MISS not in v for v in tx.VERDICTS.values())
    for w in (ix.MAIN_LEAF_WEIGHTS, ix.STORY_LEAF_WEIGHTS):
        assert w.get(tx.CONTINUATION_MISS, 0) == 0


def test_index_counts_the_excused_segment_in_the_denominator(tmp_path):
    segs = [
        {"readout": 0, "start": 0, "end": 3, "leaf": tx.CONTINUATION_MISS},
        {"readout": 0, "start": 4, "end": 9, "leaf": "detail"},
    ]
    rec = {"id": "i", "layer": 20, "stages": {"map": "ok"}, "map": {"segments": segs}}
    p = tmp_path / "cells.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    mat = ix.claim_rows(p, ix.MAIN)[20]
    assert mat[0, ix.C_CLAIMS] == 2 and mat[0, ix.C_WEIGHTED] == pytest.approx(0.2)
