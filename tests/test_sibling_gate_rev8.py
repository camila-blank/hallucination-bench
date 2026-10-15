"""Plan 0009 rev 8 / 8.1: sibgen-chat-v2, the token-class key (one function for read positions and
eval sites), the per-class merge, the read-position caps, and the per-(layer, key) null check."""

import json

import numpy as np
import pytest

from hallucination_bench.evals.sibling_gate import ladder as L
from hallucination_bench.evals.sibling_gate import texts as T

K = T.TOKEN_CLASSES.index

# the top 60 tokens by frequency of /workspace/hb-readouts/s3d-rl600-all11/hallucination.jsonl
# (byte-level BPE with real spaces, as the readouts store them) -> class
TOP60 = {
    ",": "comma",
    ".": "period",
    "Ċ": "newline",
    "ĊĊ": "newline",
    " **": "bold",
    "!": "exclaim",
    "?": "question",
    "-": "dash",
    ":": "colon",
    " (": "open_bracket",
    "*": "star",
    "**": "bold",
    ' "': "quote",
    ":**": "bold",
    "**:": "bold",
    "###": "heading",
    " *": "star",
    " -": "dash",
    ")": "close_bracket",
    "),": "close_bracket",
    "#": "heading",
    "âĢĶ": "dash",  # —
    '"': "quote",
    "**.": "bold",
    "####": "heading",
    "?**": "bold",
    ")**": "bold",
    " ĊĊ": "newline",
    ").": "close_bracket",
    " ?": "question",
    "  Ċ": "newline",
    "?)": "question",
    ".,": "period",
    "()": "open_bracket",
    " '": "other",
    "):": "close_bracket",
    " âĢĵ": "dash",  # –
    "*)": "star",
    "?âĢĿ": "question",  # ?”
    "*,": "star",
    ',"': "comma",
    ' **"': "bold",
    "âĢĵ": "dash",
    "/": "other",
    "**,": "bold",
    " &": "other",
    ";": "other",
    "__(": "other",
    "*(": "star",
    ")*": "close_bracket",
    "!)": "exclaim",
    ".).": "period",
    "...*": "period",
    " (*": "open_bracket",
    '."': "period",
    "**?": "bold",
    ".)": "period",
    ",âĢĿ": "comma",  # ,”
    "__": "other",
    " /": "other",
}


@pytest.mark.parametrize("tok,name", list(TOP60.items()))
def test_token_class_top60(tok, name):
    assert T.TOKEN_CLASSES[T.token_class(tok)] == name
    assert T.is_read_token(tok)  # every eval-site token is a read-token candidate


def test_top60_matches_readouts_file():
    p = "/workspace/hb-readouts/s3d-rl600-all11/hallucination.jsonl"
    try:
        from collections import Counter

        with open(p) as fh:
            c = Counter(json.loads(line)["token"] for line in fh)
    except FileNotFoundError:
        pytest.skip("hb-readouts absent")
    assert {t for t, _ in c.most_common(60)} <= set(TOP60)


def test_normalisation_bpe_and_decoded_agree():
    assert T.decode_bpe("âĢĶ") == "—" and T.decode_bpe("âĢĵ") == "–"
    assert T.decode_bpe("âĢĿ") == "”" and T.decode_bpe("ĠĊ") == " \n"
    for bpe, text in (("âĢĶ", "—"), ("ĊĊ", "\n\n"), (" ĊĊ", " \n\n"), ("?âĢĿ", "?”")):
        assert T.token_class(bpe) == T.token_class(text)
    assert T.decode_bpe("é") == "é"  # decoded text that is not valid BPE stays as is
    assert T.token_class("“") == K("quote") and T.token_class("”") == K("quote")
    assert T.token_class("！") == K("exclaim") and T.token_class("，") == K("comma")
    assert T.token_class(" the") == T.OTHER and not T.is_read_token(" the")
    assert not T.is_read_token(" ") and not T.is_read_token("😊")


def test_read_positions_caps_after_merge():
    toks = ["Alibaba", *[","] * 10, *["."] * 3, *[";"] * 3]
    starts = list(range(len(toks)))
    pos = T.read_positions(toks, starts, 0, 0)
    assert [T.token_class(toks[p]) for p in pos].count(K("comma")) == 8  # <= 8 per key
    assert len(pos) == 8 + 3 + 3
    # merged: period joins other, so other (3 ';' + 3 '.') is capped at 8 together
    merge = {c: (T.OTHER if c in (K("period"), K("comma")) else c) for c in range(T.N_CLASSES)}
    assert len(T.read_positions(toks, starts, 0, 0, merge)) == 8
    many = ["x", *[",", ".", "!", "?", "-", "(", ")", '"', ":"] * 8]
    assert len(T.read_positions(many, list(range(len(many))), 0, 0)) == T.MAX_READ == 32


def test_class_merge_fixed_point():
    # 60 implied texts: 8 commas each (480 >= 400); 40 with 9 newlines (cap 8 -> 320 < 400)
    texts = [[K("comma")] * 8] * 60 + [[K("newline")] * 9] * 40
    m = T.class_merge(texts)
    assert m[K("comma")] == K("comma") and m[K("newline")] == T.OTHER
    assert m[T.OTHER] == T.OTHER
    assert T.class_merge([]) == {c: T.OTHER for c in range(T.N_CLASSES)}


def test_generator_v2():
    assert T.GEN_VERSION == "sibgen-chat-v2"
    for s in ("###", "bold", "bullet", "!", "?", "parentheses", "early"):
        assert s in T.GEN_SYSTEM


# ---------------------------------------------------------------- per-key null check

CONCEPTS = [f"c{i}" for i in range(12)]
COMMA, NL = K("comma"), K("newline")


def keyed_probes(z_at, site_key, null_ok):
    """``site_key(item, pos)`` -> key; ``null_ok[layer] = {key: ok}`` (missing layer: all ok)."""
    cp = L.ClassProbes("K", list(CONCEPTS))
    cp.z = L.ZTable(CONCEPTS)
    for la in L.PROBE_LAYERS:
        cp.null_ok[la] = null_ok.get(la, {COMMA: True, NL: True, T.OTHER: True})
        for c in CONCEPTS:
            cp.validity[(c, la)] = 0.95
        for item in ("i", "j"):
            for p in (1, 2, 3, 4):
                zz = z_at(item, la, p)
                cp.z.add(item, la, p, [zz.get(c, 0.0) for c in CONCEPTS], site_key(item, p))
    return cp


def kunit(key=None, gold="c1", layer=44):
    return L.UnitProbe(
        "u", "s3d-rl600-all11", "i", layer, 2, "K", "c0", gold, [f"c{i}" for i in range(2, 10)], key
    )


def comma_at_2(item, p):
    return COMMA if p == 2 else NL


def fire_at(la):
    return lambda i, L_, p: {"c1": 5.0, **({"c0": 9.0} if (L_ == la and p == 2) else {})}


def comma_at_23(item, p):
    return COMMA if p in (2, 3) else NL


def test_failing_key_makes_only_its_units_na():
    bad_nl = {44: {COMMA: True, NL: False}}
    cp = keyed_probes(fire_at(44), comma_at_23, bad_nl)  # site 3 (comma) keeps it readable
    # the unit sits on a comma (passes): the class stays usable at L44
    assert L.rung2(kunit(), cp, {})["status"] == "probe_faithful"
    assert L.rung2(kunit(key=COMMA), cp, {})["status"] == "probe_faithful"
    # the same class, a unit on the failing newline key: n/a null_fail
    r = L.rung2(kunit(key=NL), cp, {})
    assert r["status"] == "probe_na" and r["na_reason"] == "null_fail"
    # a key absent from the per-key state (unchecked) or unresolved fails
    cp2 = keyed_probes(fire_at(44), comma_at_2, {44: {NL: True}})
    assert L.rung2(kunit(), cp2, {})["na_reason"] == "null_fail"
    cp3 = keyed_probes(fire_at(44), lambda i, p: None, {44: {COMMA: True}})
    assert L.rung2(kunit(), cp3, {})["na_reason"] == "null_fail"


def test_other_fails_by_design():
    cp = keyed_probes(fire_at(44), lambda i, p: T.OTHER if p == 2 else COMMA, {})
    cp.null_ok[44] = {COMMA: True, T.OTHER: False}
    assert L.rung2(kunit(), cp, {})["na_reason"] == "null_fail"
    # the unit's raw key goes through the class merge: ';' -> other
    cp.merge = {c: (c if c == COMMA else T.OTHER) for c in range(T.N_CLASSES)}
    assert L.rung2(kunit(key=K("colon")), cp, {})["na_reason"] == "null_fail"
    assert L.rung2(kunit(key=COMMA), cp, {})["status"] == "probe_faithful"


def test_readability_counts_only_passing_keys():
    # the gold beats its distractors only at newline sites (1, 3, 4 are newlines; 2 = unit)
    def gold_on_nl(i, la, p):
        return {"c1": 5.0} if p != 2 else {}

    ok = keyed_probes(gold_on_nl, comma_at_2, {})
    u = kunit()
    assert L.readable_share(ok, u, 44) == 1.0
    assert L.na_reason(ok, u, 44) is None
    # newline fails at L44: no passing site is left -> unreadable (not null_fail: the unit's
    # own comma key passes)
    no_nl = keyed_probes(gold_on_nl, comma_at_2, {44: {COMMA: True, NL: False}})
    assert L.readable_share(no_nl, u, 44) == 0.0
    assert L.na_reason(no_nl, u, 44) == "unreadable"

    # mixed: sites 1 (newline, gold wins) and 3, 4 (comma, gold loses); newline fails ->
    # only the comma sites count -> share 0 although the gold wins at a failing-key site
    def key_mix(i, p):
        return NL if p == 1 else COMMA

    def gold_at_1(i, la, p):
        return {"c1": 5.0} if p == 1 else {}

    mix = keyed_probes(gold_at_1, key_mix, {44: {COMMA: True, NL: False}})
    assert L.readable_share(mix, u, 44) == 0.0
    mix_ok = keyed_probes(gold_at_1, key_mix, {})
    assert L.readable_share(mix_ok, u, 44) == pytest.approx(1 / 3)


def test_nearby_only_where_the_unit_key_passes():
    good = keyed_probes(fire_at(48), comma_at_2, {})
    assert L.rung2(kunit(), good, {})["status"] == "probe_nearby"
    bad48 = keyed_probes(fire_at(48), comma_at_2, {48: {COMMA: False, NL: True}})
    assert L.rung2(kunit(), bad48, {})["status"] == "probe_unconfirmed"


def test_load_class_probes_per_key_state(tmp_path):
    fd = tmp_path / "fit" / "k"
    fd.mkdir(parents=True)
    merge = {str(c): (c if c in (COMMA, NL) else T.OTHER) for c in range(T.N_CLASSES)}
    res = {
        "layer": 44,
        "validity": {"a": [0.95] * len(CONCEPTS)},
        "null_check": {"keys": {str(COMMA): {"ok": True}, str(NL): {"ok": False}}},
        "merge": merge,
    }
    (fd / "fit_L44.json").write_text(json.dumps(res))
    (fd / "fit_L48.json").write_text(
        json.dumps({"layer": 48, "error": "x", "validity": {"a": [None] * len(CONCEPTS)}})
    )
    sd = tmp_path / "score" / "k"
    sd.mkdir(parents=True)
    np.savez(
        sd / "L44.npz",
        item=np.array(["i", "i"]),
        pos=np.array([1, 2]),
        za=np.zeros((2, len(CONCEPTS)), np.float32),
        key=np.array([NL, COMMA]),
    )
    cp = L.load_class_probes(tmp_path, {"K": {"concepts": CONCEPTS}})["K"]
    assert cp.null_ok[44] == {COMMA: True, NL: False} and cp.null_ok[48] == {}
    assert cp.merged(K("colon")) == T.OTHER and cp.merged(COMMA) == COMMA
    assert cp.site_ok("i", 44, 2) and not cp.site_ok("i", 44, 1)
    assert not cp.key_ok(48, COMMA)


def test_per_key_report():
    rows = [
        {"key": "comma", "rung1": "stable", "final": "probe_faithful"},
        {"key": "comma", "rung1": "stable", "final": "probe_na"},
        {"key": "newline", "rung1": "sparse", "final": "probe_na"},
        {"key": "newline", "rung1": "unstable", "final": "unstable"},
    ]
    s = L.per_key_summary(rows)
    assert list(s) == ["comma", "newline"]
    assert s["comma"]["n_rung2"] == 2 and s["comma"]["rung2_share"] == pytest.approx(2 / 3)
    assert s["comma"]["probe_na_rate"] == 0.5 and s["comma"]["probe_faithful_rate"] == 0.5
    assert s["newline"]["rung1"] == {"sparse": 1, "unstable": 1}
    assert s["newline"]["probe_na_rate"] == 1.0
    md = "\n".join(L.per_key_table(rows))
    assert "| comma | 2 |" in md and "Rung 2 by eval-site key class" in md
    row = L.label_row(
        {
            "unit": "u",
            "arm": "a",
            "id": "i",
            "layer": 44,
            "pos": 2,
            "said": "s",
            "actual": "t",
            "segments": [],
        },
        {"rung1": "stable"},
        {"status": "probe_na"},
        None,
        key="comma",
    )
    assert row["key"] == "comma"


def test_distractors_exclude_compounds_and_near_duplicates():
    from hallucination_bench.evals.sibling_gate import ladder as L

    concepts = [
        "students",
        "employees",
        "students and departments",
        "students table",
        "GPT",
        "ChatGPT",
        "Claude",
        "Gemini",
        "Scala, TypeScript",
        "C",
        "Dart",
        "Rust",
    ]
    d = L.select_distractors("u1", "students", concepts, {}, ["employees"])
    assert "students and departments" not in d["distractors"]
    assert "students table" not in d["distractors"]
    assert "Scala, TypeScript" not in d["distractors"]
    d2 = L.select_distractors("u2", "ChatGPT", concepts, {}, ["Claude"])
    assert "GPT" not in d2["distractors"]
    # short names do not trigger the substring rule ("C" vs "Claude")
    d3 = L.select_distractors("u3", "C", concepts, {}, [])
    assert "Claude" in d3["distractors"] or d3["n_eligible"] >= 8
    assert L.overlaps("Dart", ["Rust"]) is False


def test_counts_drops_other_objects_that_are_said_or_actual():
    from hallucination_bench.evals.sibling_gate import fill_judge as F

    ans = [
        {"block": 1, "answer": "other", "objects": ["ChatGPT"]},
        {"block": 2, "answer": "other", "objects": ["Claude"]},
        {"block": 3, "answer": "other", "objects": ["Qwen "]},
    ]
    c = F.counts(ans, "ChatGPT", "Qwen")
    assert c["n_other"] == 1 and c["n_other_dropped"] == 2
    assert F.counts(ans)["n_other"] == 3  # without said/actual: unchanged behaviour


def test_nearby_ok_does_not_retest_readability():
    import inspect

    from hallucination_bench.evals.sibling_gate import ladder as L

    src = inspect.getsource(L.nearby_ok)
    assert "readable_share" not in src and "key_ok" in src
    assert "nearby_ok(cp, u, L)" in inspect.getsource(L.rung2)


def test_overlap_rule_is_narrow():
    from hallucination_bench.evals.sibling_gate import ladder as L

    assert L.overlaps("Romania", ["Oman"]) is False  # no affix match inside a word
    assert L.overlaps("Somalia", ["Mali"]) is False
    assert L.overlaps("GPT", ["ChatGPT"]) is True
    common = L.common_words(["Intel Core i5", "Intel Core i7", "Intel Xeon", "AMD Ryzen"])
    assert L.overlaps("Intel Core i7", ["Intel Core i5"], common) is False
    assert L.overlaps("North Korea", ["South Korea"]) is True  # shared non-common word
    # siblings other than gold / said no longer shrink the pool via word overlap
    concepts = [
        "France",
        "Germany",
        "Spain",
        "New Zealand",
        "New Guinea",
        "Italy",
        "Peru",
        "Chile",
        "Japan",
        "Kenya",
        "Egypt",
        "Nepal",
    ]
    d = L.select_distractors("u", "France", concepts, {}, ["New Zealand"], said="Germany")
    assert "New Guinea" in d["distractors"] or d["n_eligible"] == 9
