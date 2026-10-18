"""units.py (plan 0002): the deterministic splitter (coverage, stability, rules, the real-row
fixture) and the offset-returning ``locate``."""

import json
from pathlib import Path

import pytest

from hallucination_bench.units import Normalised, locate, locate_established, split_readout

FIXTURE = Path(__file__).parent / "fixtures" / "units_real.json"


def texts(s: str, est=()) -> list[str]:
    return [u.text for u in split_readout(s, 0, est)[0]]


def assert_covers(text: str, units) -> None:
    covered = [False] * len(text)
    prev = 0
    for u in units:
        assert u.start >= prev, "units overlap or are out of order"
        assert text[u.start : u.end] == u.text and u.text == u.text.strip() and u.text
        for i in range(u.start, u.end):
            covered[i] = True
        prev = u.end
    assert all(c or text[i].isspace() for i, c in enumerate(covered)), "non-whitespace dropped"


# ------------------------------------------------------------------ the real-row fixture


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_spans_the_three_lens_styles_and_the_stray_quote_row():
    rows = _fixture()
    sources = " ".join(r["source"] for r in rows)
    assert "s3d-rl600" in sources and "nla-rl-L42" in sources and "jlens" in sources
    assert any('</"' in r["text"] for r in rows)  # the stray unbalanced quote row
    assert sum(bool(r["established"]) for r in rows) >= 5


@pytest.mark.parametrize("row", _fixture(), ids=lambda r: r["source"])
def test_splitter_matches_the_checked_in_real_rows(row):
    units, unloc = split_readout(row["text"], 0, row["established"])
    assert [[u.start, u.end, u.text, u.established] for u in units] == row["expected"]
    assert unloc == row["n_unlocated_established"] == 0
    assert_covers(row["text"], units)
    assert [u.id for u in units] == [f"R0.U{n}" for n in range(1, len(units) + 1)]
    # stable: the same input gives the same units
    assert split_readout(row["text"], 0, row["established"])[0] == units
    # every established span is exactly one unit
    for s in row["established"]:
        assert any(u.established and s.strip() in u.text for u in units)


# ------------------------------------------------------------------ rules


def test_newlines_markers_and_sentence_ends():
    s = "- 2: I'd be delighted to\n- 你好！That's fun! Next one.\n1) Item. Two\n* last"
    assert texts(s) == [
        "- 2: I'd be delighted to",
        "- 你好！That's fun!",
        "Next one.",
        "1) Item.",
        "Two",
        "* last",
    ]


def test_no_split_after_digits_abbreviations_or_without_following_space():
    assert texts("Pi is 3.14 today. It rose 2. Then fell") == [
        "Pi is 3.14 today.",
        "It rose 2. Then fell",
    ]
    assert texts(
        "Use e.g. tea, i.e. a drink, vs. coffee. Dr. Who met Mr. X in the U.S. today."
    ) == [
        "Use e.g. tea, i.e. a drink, vs. coffee.",
        "Dr. Who met Mr. X in the U.S. today.",
    ]
    assert texts("see example.com/a.b?x=1 now. ok") == ["see example.com/a.b?x=1 now.", "ok"]
    assert texts("Wait... what?! Yes…  Fine") == ["Wait...", "what?!", "Yes…", "Fine"]


def test_quotes_suppress_splits_and_reset_at_newlines():
    assert texts('He said "Stop. Now." Then left.') == ['He said "Stop. Now." Then left.']
    assert texts("She wrote “Hi. Bye.” and went. Next") == [
        "She wrote “Hi. Bye.” and went.",
        "Next",
    ]
    assert texts("「你好。 再见。」 ok. Then") == ["「你好。 再见。」 ok.", "Then"]
    # a stray quote swallows only the rest of its own line
    s = 'A stray </" quote. Still one unit.\nNew line. Splits again.'
    assert texts(s) == [
        'A stray </" quote. Still one unit.',
        "New line.",
        "Splits again.",
    ]


def test_semicolons_colons_and_tags():
    assert texts("one; two；three") == ["one;", "two；", "three"]
    assert texts("Note: at 10:30 see http://x.y/z ok") == ["Note:", "at 10:30 see http://x.y/z ok"]
    assert texts("<explanation>\nText here</response>more\n</explanation>") == [
        "<explanation>",
        "Text here",
        "</response>",
        "more",
        "</explanation>",
    ]


def test_established_spans_become_their_own_units():
    s = "The response attributes it to Picasso, and the dates follow. Done."
    units, unloc = split_readout(s, 2, ["attributes it to Picasso, and the"])
    assert [u.text for u in units] == [
        "The response",
        "attributes it to Picasso, and the",
        "dates follow.",
        "Done.",
    ]
    assert [u.established for u in units] == [False, True, False, False]
    assert units[1].id == "R2.U2" and unloc == 0
    # a span that straddles a sentence end suppresses the cut inside it
    units, _ = split_readout("It is red. It is big. Fine.", 0, ["red. It is"])
    assert [u.text for u in units] == ["It is", "red. It is", "big.", "Fine."]


def test_established_merging_duplicates_and_unlocated():
    s = "Picasso painted it. Later Picasso slept."
    spans, n = locate_established(
        s, ["Picasso painted", "painted it", "Picasso", "Picasso", "Monet"]
    )
    # nested / overlapping -> merged; the duplicate takes the next occurrence; Monet is unlocated
    assert spans == [(0, 18), (26, 33)] and n == 1
    units, n = split_readout(s, 0, ["Picasso painted", "painted it", "Picasso", "Picasso", "Monet"])
    assert [(u.text, u.established) for u in units] == [
        ("Picasso painted it", True),
        (".", False),
        ("Later", False),
        ("Picasso", True),
        ("slept.", False),
    ]
    assert n == 1
    # a span listed twice with one occurrence is merged, not unlocated
    assert locate_established("only Paris here", ["Paris", "paris"]) == ([(5, 10)], 0)


def test_whitespace_only_text_gives_no_units():
    assert split_readout("  \n \n", 0)[0] == []


# ------------------------------------------------------------------ locate


def test_locate_returns_raw_offsets_under_verify_quote_normalisation():
    t = "He said  “Hello,\n  World” to me"
    a, b = locate('said "hello, world"', t)
    assert t[a:b] == "said  “Hello,\n  World”"
    assert locate("x", "a x b x") == (2, 3)  # no minimum length
    assert locate("x", "a x b x", 3) == (6, 7)  # start after a previous match
    assert locate("nope", t) is None and locate("  ", t) is None


def test_locate_maps_through_length_changing_lowercase():
    t = "İstanbul is big"  # 'İ'.lower() is two characters
    assert len(Normalised(t).norm) == len(t) + 1
    a, b = locate("is big", t)
    assert t[a:b] == "is big"
    a, b = locate("İSTANBUL", Normalised(t))
    assert (a, b) == (0, 8) and t[a:b] == "İstanbul"
