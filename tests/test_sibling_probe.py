"""Plan 0003 stage 1: concept inventory, alias matcher, bank checks and ReadSpec writers."""

import json
import re
import sys
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from hallucination_bench import paths
from hallucination_bench.evals.sibling_probe import CATEGORIES, SYSTEM_PROMPT
from hallucination_bench.evals.sibling_probe import bank as B
from hallucination_bench.evals.sibling_probe import concepts as C
from hallucination_bench.readouts import _bytes_to_unicode

INV = C.load_inventory()
WSB_SRC = paths.WSBENCH_PUB / "src"
PILOT = Path(__file__).resolve().parents[1] / "data" / "sibling_probe" / "pilot"


@cache
def tokenizer():
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    except Exception as e:  # no network and no HF cache
        pytest.skip(f"Qwen3.6-27B tokenizer unavailable: {e}")


def m(cat: str) -> C.Matcher:
    return C.Matcher(INV[cat])


def raw_token(text: str) -> str:
    """Text -> its byte-level BPE vocabulary spelling (what J-lens readouts store)."""
    b2u = _bytes_to_unicode()
    return "".join(b2u[b] for b in text.encode("utf-8"))


# ------------------------------------------------------------------ inventory


def test_inventory_shape():
    assert tuple(INV) == CATEGORIES
    for cat in INV.values():
        assert 10 <= len(cat.concepts) <= 16, cat.name
    assert C.validate_inventory(INV) == []


def test_inventory_questions_and_requests_are_the_plans():
    bio = "Write a hypothetical but realistic Wikipedia biography infobox for me."
    assert INV["country"].requests == (bio,)
    assert INV["occupation"].requests == (bio,)
    assert INV["emotion"].requests[1] == "Write a short journal entry about my day."
    assert INV["farm_animal"].requests[1] == "Write a short caption for a photo of my morning."
    assert INV["capital_city"].requests[1] == "Write a short itinerary for my first day."
    assert INV["tree"].requests[1] == "Write a short description of my garden."
    assert INV["holiday"].requests[1] == "Write a short note to my neighbors about our plans."
    assert INV["country"].question == (
        "Based on the message, which country is the user from? Answer in one or two words."
    )
    assert INV["emotion"].question == (
        "Based on the message, how does the user feel? Answer in one word."
    )
    for cat in INV.values():
        assert cat.question.startswith("Based on the message, ")


def test_pilot_requests_are_the_first_request_sentence():
    meta = {r["id"]: r for r in json.loads((PILOT / "meta.json").read_text())}
    for s in json.loads((PILOT / "specs.json").read_text()):
        cat = INV[meta[s["id"]]["category"]]
        assert B.split_cue(B.user_text(s["messages"]), cat.requests)[1] == 0
        assert meta[s["id"]]["gold"] in cat.ids


def test_every_alias_matches_only_its_own_concept():
    for cat in INV.values():
        assert C.overlap_problems(cat) == [], cat.name


def test_cjk_aliases_have_two_chars_and_exclusion_lists():
    for cat in INV.values():
        for c in cat.concepts:
            for a in c.aliases:
                if C.is_cjk(a):
                    assert len(a) >= 2 and a in c.exclude, (cat.name, c.id, a)


def test_emotion_synonym_clusters_merged():
    emo = INV["emotion"]
    for cid, members in {
        "embarrassed": ["ashamed", "humiliated", "mortified"],
        "angry": ["furious", "frustrated"],
        "sad": ["grieving", "heartbroken"],
    }.items():
        for w in members:
            assert m("emotion").mentions(f"I feel {w}.") == {cid}
    assert {g["concept"] for g in emo.merges} >= {"embarrassed", "angry", "sad"}


# ------------------------------------------------------------------ matcher


def test_matcher_word_boundaries_and_case():
    mc = m("country")
    assert mc.mentions("She is KOREAN by birth.") == {"south_korea"}
    assert mc.mentions("Korea的文化") == {"south_korea"}  # CJK neighbour is a boundary
    assert mc.mentions("a North Korean defector") == set()  # exclusion phrase
    assert mc.mentions("Ukrainian embroidery") == {"ukraine"}
    assert mc.mentions("Belarusian") == {"belarus"}  # 'russian' inside is not a word
    assert mc.mentions("白俄罗斯") == {"belarus"}  # 俄罗斯 inside is excluded
    assert mc.mentions("Latin American music") == set()
    assert mc.mentions("I polish the silver") == set()
    assert mc.mentions("a Polish grandmother") == {"poland"}
    assert m("farm_animal").mentions("the hens and the chickens") == {"chicken"}
    assert m("farm_animal").mentions("then we went") == set()
    assert m("tree").mentions("a pineapple") == set()


def test_matcher_happy_to_help_excluded():
    me = m("emotion")
    assert me.mentions("I'd be happy to help you write that!") == set()
    assert me.mentions("I’d be happy to help") == set()  # curly apostrophe
    assert me.mentions("很高兴为您服务") == set()
    assert me.mentions("She was so happy when she won.") == {"happy"}
    assert me.mentions("Happy birthday to my friend") == set()


def test_matcher_holiday_generic_uses():
    mh = m("holiday")
    assert mh.mentions("our holiday party") == set()
    assert mh.mentions("the holidays are coming") == set()
    assert mh.mentions("Lunar New Year dumplings") == {"lunar_new_year"}
    assert mh.mentions("New Year's Eve countdown") == {"new_year"}
    assert mh.mentions("Valentine’s Day roses") == {"valentines"}
    assert mh.mentions("my birthday") == set()


def test_matcher_holi_not_inside_holiday():
    cat = C.Category(
        "h",
        "q",
        ("r",),
        (C.Concept("holi", "Holi", "h", True, ("Holi",), {"Holi": ("holiday",)}),),
    )
    mt = C.Matcher(cat)
    assert mt.mentions("a holiday") == set()
    assert mt.mentions("the holidays") == set()
    assert mt.mentions("Holi colours") == {"holi"}


def test_matcher_cjk_and_squirrel():
    mt = m("tree")
    assert mt.mentions("松鼠在院子里") == set()  # squirrel: never a pine
    assert mt.mentions("院子里有一棵松树") == {"pine"}
    assert mt.mentions("枫叶红了") == {"maple"}
    mc = m("capital_city")
    assert mc.mentions("东京都的街道") == {"tokyo"}  # not Kyoto (京都 inside 东京都)
    assert mc.mentions("京都的寺庙") == {"kyoto"}
    assert mc.mentions("罗马尼亚") == set()  # Romania is not Rome
    assert mc.mentions("George Washington's portrait") == set()


def test_hits_offsets_point_at_the_alias():
    text = "We flew from Paris to 东京 last May."
    hits = m("capital_city").hits(text)
    assert [(h.concept, text[h.start : h.end]) for h in hits] == [
        ("paris", "Paris"),
        ("tokyo", "东京"),
    ]


def test_containment_rule_prefers_longer_hit():
    cat = C.Category(
        "x",
        "q",
        ("r",),
        (
            C.Concept("a", "New Year", "x", True, ("New Year",)),
            C.Concept("b", "Lunar New Year", "x", True, ("Lunar New Year",)),
        ),
    )
    assert C.Matcher(cat).mentions("Lunar New Year feast") == {"b"}


def test_token_mentions_raw_and_decoded():
    mc = m("country")
    assert mc.token_mentions([raw_token(" Korea"), raw_token(" bowl")]) == {"south_korea"}
    assert mc.token_mentions([raw_token("韩国")]) == {"south_korea"}
    assert mc.token_mentions([" korea"], raw=False) == {"south_korea"}  # case-folded
    assert mc.token_mentions([raw_token(" Kore")]) == set()  # no first-subtoken matching
    assert mc.token_concept(raw_token(" China")) == "china"


def test_token_eligible_iff_token_aliases():
    for cat in INV.values():
        for c in cat.concepts:
            assert c.token_eligible == bool(c.token_aliases)


def test_token_aliases_are_single_qwen_tokens():
    tok = tokenizer()
    for cat in INV.values():
        for c in cat.concepts:
            for t in c.token_aliases:
                assert len(tok.encode(t, add_special_tokens=False)) == 1, (cat.name, c.id, t)


def test_token_aliases_name_their_concept():
    """Each token alias, stripped, is a text alias (or a token-only extra of a multiword
    alias) of its own concept."""
    for cat in INV.values():
        mt = C.Matcher(cat)
        for c in cat.concepts:
            for t in c.token_aliases:
                got = mt.mentions(t.strip())
                assert got in ({c.id}, set()), (cat.name, c.id, t, got)


# ------------------------------------------------------------------ leak + 8-gram


def test_leak_check():
    mc = m("country")
    assert B.leaks("I can't start my day without kimchi jjigae.", mc) == set()
    assert B.leaks("My Korean grandmother makes kimchi.", mc) == {"south_korea"}
    assert B.leaks("I love sopa paraguaya.", mc) == set()  # 'paraguaya' is not the word


def test_word_ngrams_and_overlap():
    a = "I find comfort in my grandmother's homemade sopa paraguaya on Sundays."
    assert len(B.word_ngrams(a)) == len(re.findall(r"\S+", a)) - 7
    idx = B.ngram_index([a])
    assert B.shares_ngram("Honestly I find comfort in my grandmother's homemade sopa too.", idx)
    assert not B.shares_ngram("I find comfort in my aunt's homemade soup on Sundays.", idx)
    assert not B.shares_ngram("too short to share", idx)


def test_cue_format():
    assert B.cue_ok("We tap the tree every spring.")
    assert B.cue_ok("What a day!")
    assert not B.cue_ok('He said "great."')
    assert not B.cue_ok("no punctuation")
    assert B.split_cue("A cue. Write a short poem about my yard.", INV["tree"].requests) == (
        "A cue.",
        0,
    )
    with pytest.raises(ValueError):
        B.split_cue("A cue.Write a short poem about my yard.", INV["tree"].requests)


# ------------------------------------------------------------------ window identity


def test_window_identity_on_pilot_and_quote_failure():
    tok = tokenizer()
    wc = B.WindowChecker(tok)
    req = INV["tree"].requests[0]
    assert wc.ok(B.messages_for("The tree drops acorns.", req), req)
    assert wc.ok(B.messages_for("Is it an oak? I think so!", req), req)
    assert not wc.ok(B.messages_for('My son called it "the giant."', req), req)
    canon = wc.canonical(req)
    assert tok.decode(canon).startswith(" " + req + "<|im_end|>")
    assert tok.decode(canon).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.skipif(
    not WSB_SRC.is_dir(),
    reason=f"wsbench-pub checkout absent at {WSB_SRC} (set HBENCH_WSBENCH_PUB)",
)
def test_from_last_sentence_start_matches_wsbench():
    tok = tokenizer()
    sys.path.insert(0, str(WSB_SRC))
    from wsbench.produce.render import render
    from wsbench.readplan import ReadSpec, resolve

    specs = json.loads((PILOT / "specs.json").read_text())[:6]
    extra = B.messages_for("Odd... ending? Yes!", INV["tree"].requests[1])
    for msgs in [s["messages"] for s in specs] + [extra]:
        spec = ReadSpec("f", "x", "chat_context", dict(B.POSITIONS), [28], messages=msgs)
        rd = render(spec, tok)
        ids = B.render_ids(tok, msgs)
        assert rd.ids == ids
        assert B.display_tokens(tok, ids) == rd.tokens
        assert B.from_last_sentence_start(rd.tokens) == resolve(B.POSITIONS, rd.tokens)


# ------------------------------------------------------------------ coverage rule


def test_bank_size():
    assert B.bank_size(10) == 20
    assert B.bank_size(11) == 22
    assert B.bank_size(16) == 32
    assert B.bank_size(20) == 32


def test_allocate_fewest_first_and_cap():
    ids = ["a", "b", "c", "d"]
    gen = B.allocate(ids, {"a": 5, "b": 1}, 12)
    # 6 slots, each to the fewest-items concept: c, d, b (tie -> order), c, d, b
    assert gen == {"a": 0, "b": 2, "c": 2, "d": 2}
    assert sum(gen.values()) == 6
    gen = B.allocate(ids, {}, 40)
    assert gen == dict.fromkeys(ids, 3)  # cap 3 per gold
    gen = B.allocate(ids, {}, 8, available={"c": 0, "d": 1})
    assert gen == {"a": 3, "b": 3, "c": 0, "d": 1}


def test_coverage_problems_detects_violations():
    cat = C.Category(
        "x",
        "q",
        ("r",),
        tuple(C.Concept(i, i, "x", True, (i,)) for i in ("aa", "bb")),
    )
    items = [
        {"id": f"g{i}", "category": "x", "gold": "aa", "source": "generated"} for i in range(4)
    ]
    errs = B.coverage_problems(items, cat)
    assert any("bb is the gold of no item" in e for e in errs)
    assert any("4 generated items for aa" in e for e in errs)
    ok = [
        {"id": "w1", "category": "x", "gold": "aa", "source": "wsbench"},
        {"id": "g1", "category": "x", "gold": "bb", "source": "generated"},
    ]
    assert B.coverage_problems(ok, cat) == []


# ------------------------------------------------------------------ writers


def test_readspec_writers(tmp_path):
    items = [
        {"id": "gen-tree-oak-a0-00", "category": "tree", "messages": B.messages_for("x.", "y.")},
        {"id": "pilot-emotion-00", "category": "emotion", "messages": B.messages_for("a.", "b.")},
    ]
    n = B.write_plan_files(items, tmp_path / "plan", B.EVAL_FAMILY_PREFIX)
    assert n == {"sibling_tree": 1, "sibling_emotion": 1}
    row = json.loads((tmp_path / "plan" / "sibling_tree.jsonl").read_text())
    assert row["render"] == "chat_context"
    assert row["positions"] == {"kind": "from_last_sentence_start"}
    assert row["layers"] == [28, 36, 42, 44, 52]
    assert row["family"] == "sibling_tree"
    assert B.write_s3d_specs(items, tmp_path / "s3d.json") == 2
    specs = json.loads((tmp_path / "s3d.json").read_text())
    assert {s["id"] for s in specs} == {it["id"] for it in items}
    assert all(s["layers"] == [28, 36, 44, 52] for s in specs)
    with pytest.raises(ValueError):
        B.write_plan_files(items + items[:1], tmp_path / "p2", "x")
    with pytest.raises(ValueError):
        B.readspec_row("f", "bad id", items[0]["messages"], [28])


# ------------------------------------------------------------------ the generated bank


@pytest.mark.skipif(not B.ITEMS_PATH.exists(), reason="items.json not generated yet")
def test_bank_items_follow_the_rules():
    items = B.load_items()
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))
    for it in items:
        cat = INV[it["category"]]
        assert it["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert len(it["messages"]) == 2
        cue, k = B.split_cue(B.user_text(it["messages"]), cat.requests)
        assert (cue, k) == (it["cue"], it["request_idx"])
        assert it["gold"] in cat.ids
        if it["source"] == "generated":
            assert B.leaks(B.user_text(it["messages"]), C.Matcher(cat)) == set(), it["id"]
            assert it["implication"]["pass"] is True
    for cat in INV.values():
        assert B.coverage_problems(items, cat) == [], cat.name
        mine = [it for it in items if it["category"] == cat.name]
        assert len(mine) <= max(B.bank_size(len(cat.ids)), 32)


@pytest.mark.skipif(not B.PROBE_TEXTS_PATH.exists(), reason="probe texts not generated yet")
def test_probe_texts_follow_the_rules():
    rows = B.load_probe_texts()
    assert len({r["id"] for r in rows}) == len(rows)
    eval_cues = [it["cue"] for it in B.load_items()] if B.ITEMS_PATH.exists() else []
    idx = B.ngram_index(eval_cues)
    per = Counter()
    for r in rows:
        cat = INV[r["category"]]
        mt = C.Matcher(cat)
        assert r["messages"] == B.messages_for(r["context"], cat.requests[r["request_idx"]])
        hits = mt.mentions(r["context"])
        if r["kind"] == "explicit":
            assert hits == {r["concept"]}, r["id"]
        else:
            assert hits == set(), r["id"]
        assert not B.shares_ngram(r["context"], idx), r["id"]
        per[(r["category"], r["concept"], r["kind"], r["request_idx"])] += 1
    for cat in INV.values():
        nreq = len(cat.requests)
        for k in range(nreq):
            for cid in cat.ids:
                for kind in ("explicit", "implied"):
                    assert per[(cat.name, cid, kind, k)] == 24 // nreq, (cat.name, cid, kind)
            assert per[(cat.name, None, "neutral", k)] == 24 // nreq


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("me, the author of this endless loop of failing PRs", False),
        ("a fictional author bio", False),
        ("an author bio for the back cover", False),
        ("the author’s note", False),
        ("the authors of the report", False),
        ("she is a published author", True),
        ("works as an author", True),
        ("a novelist and author", True),
    ],
)
def test_author_alias_generic_uses_excluded(text, hit):
    """Review fix: bare 'author' counts only as an occupation, not a generic use."""
    assert ("writer" in C.matcher("occupation").mentions(text)) is hit
