"""Plan 0029 §1.2 as corrected by Camila (2026-11-04: "a readout that says a different possible
continuation is not a hallucination"): a segment is excused (``continuation_miss``, weight 0)
when it is tagged ``continuation_text`` (literal, on topic, possible given the text up to the
marker) and its located contradiction is only after the marker, whatever its incorrect leaf.
v2.7 is untouched."""

import pytest
from test_claimmap_v2 import e
from test_continuation import _parse, ct

from hallucination_bench import index as ix
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges import claimmap_v2 as cm

V27, V28 = "claimmap-v2.7", "claimmap-v2.8"
S7, S8 = cm.VERSIONS[V27].system, cm.VERSIONS[V28].system
R8 = tx.rules(False, V28)


def test_every_incorrect_judge_leaf_is_excusable():
    assert set(tx.EXCUSABLE_LEAVES) == {
        "false_claim",
        "substitution_category",
        "substitution_sibling",
        "detail",
    }
    assert set(tx.EXCUSABLE_LEAVES) <= set(tx.INCORRECT_LEAVES)


@pytest.mark.parametrize("leaf", tx.EXCUSABLE_LEAVES)
def test_tagged_continuation_contradicted_after_is_excused(leaf):
    segs, c = _parse(ct(leaf, probe_target="x", pair={"said": "a", "actual": "b"}))
    s = segs[0]
    assert s.leaf == tx.CONTINUATION_MISS and s.judge_leaf == leaf
    assert s.contradiction_region == "after" and c["n_continuation_miss"] == 1


def test_false_claim_continuation_after_region_is_excused():
    segs, c = _parse(ct("false_claim"))
    assert segs[0].leaf == tx.CONTINUATION_MISS and segs[0].judge_leaf == "false_claim"
    assert c["n_continuation_miss"] == 1


@pytest.mark.parametrize("leaf", tx.EXCUSABLE_LEAVES)
def test_a_prefix_contradiction_is_never_excused(leaf):
    segs, c = _parse(ct(leaf, q="new cafe", probe_target="x", pair={"said": "a", "actual": "b"}))
    assert segs[0].leaf == leaf and segs[0].contradiction_region == "prefix"
    assert c["n_continuation_miss"] == 0


def test_a_prediction_is_not_excused():
    segs, c = _parse(
        {**ct("false_claim"), "continuation_text": False}, text="The last sentence will be X."
    )
    assert segs[0].leaf == "false_claim" and c["n_continuation_miss"] == 0


def test_leaves_without_a_located_contradiction_are_not_excused():
    for leaf in ("unverifiable", "supplementary_false"):
        segs, c = _parse(e("R0.U1", leaf, continuation_text=True))
        assert segs[0].leaf == leaf and c["n_continuation_miss"] == 0
    # an incorrect segment whose quote is not in the conversation fails closed (rev 17)
    segs, c = _parse(ct("false_claim", q="nowhere in the text"))
    assert segs[0].leaf == "unverifiable" and c["n_continuation_miss"] == 0


def test_continuation_miss_weighs_zero_in_every_claim_family():
    assert tx.CONTINUATION_MISS not in ix.MAIN_LEAF_WEIGHTS
    assert tx.CONTINUATION_MISS not in ix.STORY_LEAF_WEIGHTS
    assert tx.is_claim(tx.CONTINUATION_MISS)  # still in the denominator
    assert tx.verdict_of(tx.CONTINUATION_MISS) == tx.V_UNRESOLVED
    assert tx.CONTINUATION_MISS not in tx.LEAF_NAMES  # v2.7 tables untouched


NEW_TEXT = (
    "on topic, and a possible continuation given the text up to the marker",
    "**possible given the text up to the marker**: nothing before the marker rules it out",
    "The size of the difference from what the text goes on to say does not matter",
    "The tag never changes the leaf",
    "possible continuation that only the text after the marker rules out",
)
GONE = (
    "says something different",
    "ONE wrong specific",
    "**one** wrong specific",
    "small change, so never excused",
    "Tagging it never makes a segment a small change",
)


@pytest.mark.parametrize("text", NEW_TEXT)
def test_v28_prompt_states_the_possible_continuation_rule(text):
    assert text in S8 and text not in S7


@pytest.mark.parametrize("text", GONE)
def test_v28_prompt_drops_the_small_change_wording(text):
    assert text not in S8


def test_rule_placement_and_kept_clauses():
    assert "commentary about the text" in R8[3].body[2]
    assert R8[8].body[2].startswith("Step 2.") and "continuation" not in R8[8].body[2]
    step3 = R8[8].body[3]
    assert "never split it into one segment per specific" in step3
    tag = tx.TAGS_V28["continuation_text"]
    assert "neither is commentary about the text, even when it quotes or predicts it" in tag
    assert "the last sentence will be X" in tag  # predictions are never continuation text
    seek = next(x for x in tx.EXAMPLES_V28 if x.startswith('"2. Seek Professional Help"'))
    assert "`false_claim` with `continuation_text = true`" in seek
    weather = next(x for x in tx.EXAMPLES_V28 if x.startswith('"The last sentence will be'))
    assert "`continuation_text = false`" in weather
