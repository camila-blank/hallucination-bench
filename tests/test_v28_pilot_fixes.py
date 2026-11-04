"""Plan 0029 pilot fixes to claimmap-v2.8 (edited in place; v2.7 byte-identical):
A. (revised 2026-11-04, Camila) a literal, on-topic, possible continuation is excused whatever its
   leaf; it is one segment, never split; commentary about the text is never `continuation_text`;
B. an invented name for the right role holder is `detail`;
C. a segment re-labelled `unverifiable` (rev-17 quote check, rev-19 hedge) drops
   `continuation_text`.
"""

import pytest
from test_claimmap_v2 import e
from test_continuation import _parse, ct

from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges import claimmap_v2 as cm

V27, V28 = "claimmap-v2.7", "claimmap-v2.8"
S7, S8 = cm.VERSIONS[V27].system, cm.VERSIONS[V28].system
R8 = tx.rules(False, V28)

A_TEXT = (
    "neither is commentary about the text, even when it quotes or predicts it",
    "(never for commentary about the text)",
    "possible given the text up to the marker",
    '"2. Seek Professional Help" written as the next list item when the text has "1. Allow '
    'Yourself to Feel": `false_claim` with `continuation_text = true`',
    '"I\'m Elara" written as the response\'s own next words when it says "I\'m Clara": `detail` '
    "with `continuation_text = true`",
    "\"Mirrors the earlier timestamp '15:42', so '39' must complete it\": commentary about the "
    "text, `continuation_text = false`",
)
B_TEXT = (
    "An invented name for the right role holder is a name, not an identity",
    '"Thompson (CTO)" or "CEO Mark Evans"',
    "The right person with the wrong title, and another person of the text in the role, stay "
    "`substitution_sibling`",
    '"Thompson (CTO)" when the text\'s CTO is Kyle and no Thompson appears: `detail`',
    'an invented name for the right role holder ("Thompson (CTO)" when the CTO is Kyle)',
)


@pytest.mark.parametrize("text", A_TEXT + B_TEXT)
def test_pilot_fix_text_is_in_v28_only(text):
    assert text in S8
    assert text not in S7 and text not in cm.SYSTEM


def test_a_rules_land_in_rule_3_and_the_tag():
    assert "commentary about the text" in R8[3].body[2]
    assert "possible continuation given the text up to the marker" in R8[3].body[2]
    tag = tx.TAGS_V28["continuation_text"]
    assert "commentary about the text" in tag and "possible given the text up to the marker" in tag
    assert "does not matter" in tag and "ONE segment" in tag
    # the superseded small-change rule is gone
    for gone in ("says something different", "**one** wrong specific", "ONE wrong specific"):
        assert gone not in S8


def test_b_keeps_the_role_substitution_rules():
    s1 = R8[8].body[1]
    assert s1.startswith("Step 1.")
    # the existing role rules are unchanged
    assert '"Kyle, the CEO" when Kyle is the CTO (the right person with the wrong title)' in s1
    assert "another distinct entity that appears in the text, **or a role or title**" in s1
    assert "Thompson (CTO)" in s1 and "`detail` (right role, wrong name)" in s1


def _example(start: str) -> str:
    return next(x for x in tx.EXAMPLES_V28 if x.startswith(start))


def test_examples_v28_are_consistent():
    assert "`substitution_sibling`" in _example('"Kyle, the CEO"')  # right person, wrong title
    thompson = _example('"Thompson (CTO)"')  # right role, invented name
    assert "`detail`" in thompson and "`substitution_sibling`" not in thompson
    assert "`false_claim`" in _example('"2. Seek Professional Help"')
    assert "`detail` with `continuation_text = true`" in _example('"I\'m Elara"')
    assert "`continuation_text = false`" in _example('"Mirrors the earlier timestamp')


def test_a_false_claim_continuation_is_excused_after_the_marker():
    """Camila 2026-11-04: a different possible continuation (leaf false_claim, contradicted only
    after the marker) is excused."""
    segs, c = _parse(ct("false_claim"))
    assert segs[0].leaf == tx.CONTINUATION_MISS and c["n_continuation_miss"] == 1
    segs, c = _parse(ct("detail"))
    assert segs[0].leaf == tx.CONTINUATION_MISS and c["n_continuation_miss"] == 1


def test_c_relabelled_unverifiable_drops_continuation_text():
    # rev 17: no locatable contradiction quote
    segs, c = _parse(ct(q="not in the conversation at all"))
    s = segs[0]
    assert s.leaf == "unverifiable" and s.unquoted_contradiction
    assert s.continuation_text is False and c["n_continuation_miss"] == 0
    # rev 19: a hedged miss is re-labelled unverifiable (raw leaf kept)
    segs, c = _parse(ct(hedged=True))
    s = segs[0]
    assert s.leaf == "unverifiable" and s.raw_leaf == "detail"
    assert s.continuation_text is False and c["n_continuation_miss"] == 0
    # an incorrect segment keeps the tag
    segs, _ = _parse(e("R0.U1", "false_claim", contradicted_by="yesterday", continuation_text=True))
    assert segs[0].continuation_text is True
