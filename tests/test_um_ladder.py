"""Plan 0006: the user-modelling (country) sibling ladder."""

import numpy as np
import pytest

from hallucination_bench.evals.sibling_probe import um_ladder as U
from hallucination_bench.evals.sibling_probe.ladder_lex import Matcher


@pytest.fixture(scope="module")
def fm() -> U.FastMatcher:
    return U.FastMatcher(U.load_lexicon())


def _item(gold: str, user: str = "I love my food. Write a bio for me.", resp: str = "") -> dict:
    return {
        "id": f"it-{gold}",
        "gold": gold,
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": user}],
        "response": resp,
    }


# ------------------------------------------------------------------ lexicon patch


def test_korean_maps_to_south_korea_north_korea_separate(fm):
    assert fm.concepts_in("a Korean grandmother") == {"south_korea"}
    assert fm.concepts_in("born in North Korea") == {"north_korea"}
    assert fm.concepts_in("朝鲜族的传统") == {"south_korea"}


def test_blocked_phrases_and_regional(fm):
    assert fm.concepts_in("New South Wales and the Gold Coast") == set()
    assert fm.concepts_in("Latin American and South American food") == set()
    assert fm.concepts_in("Mexico City") == {"mexico"}


def test_capitalised_only_and_capitals(fm):
    assert fm.concepts_in("pole vault, polish the floor, a male singer") == set()
    assert fm.concepts_in("a Pole from Warsaw") == {"poland"}
    assert fm.concepts_in("grew up in Beijing") == {"china"}
    assert fm.concepts_in("Washington") == set()
    assert fm.token_concept(" turkey") is None
    assert fm.token_concept(" Turkey") == "turkey"
    assert fm.token_concept("中国") == "china"


def test_fast_matcher_equals_matcher_on_texts(fm):
    m = Matcher(U.load_lexicon(), U.CATEGORY)
    texts = [
        "Born in Seoul to a Korean family, she moved to Toronto, Canada, then Paris.",
        "A Ukrainian SSR passport; Russian and Belarusian relatives; 中国 and 日本.",
        "No country here at all.",
    ]
    for t in texts:
        a = [(h.concept, h.start, h.end) for h in m.find(t) if not fm._lowercase_only(h.alias)]
        b = [(h.concept, h.start, h.end) for h in fm.find(t)]
        assert a == b


def test_soviet_stratum_vs_russia(fm):
    ukr = U.item_context(fm, _item("ukraine"))
    chn = U.item_context(fm, _item("china"))
    row = {
        "id": "x",
        "layer": 20,
        "pos": 1,
        "samples": ["- born in the Ukrainian SSR, Soviet Union"],
    }
    units, strata = U.build_mentions(fm, "s3d", row, ukr)
    assert not [u for u in units if u["concept"] in ("russia", U.SOVIET)]
    assert any(s["stratum"] == "soviet" for s in strata)
    units, _ = U.build_mentions(fm, "s3d", row, chn)
    assert "russia" in {u["concept"] for u in units}


def test_uk_equivalents_and_gold_not_mentions(fm):
    ctx = U.item_context(fm, _item("uk"))
    row = {"id": "x", "layer": 20, "pos": 1, "samples": ["- a Scottish and British and Irish user"]}
    units, _ = U.build_mentions(fm, "s3d", row, ctx)
    assert {u["concept"] for u in units} == {"ireland"}


def test_response_named_is_alias_level(fm):
    ctx = U.item_context(fm, _item("ukraine", resp="Born in the Soviet Union; Russian speaker."))
    row = {
        "id": "x",
        "layer": 20,
        "pos": 1,
        "samples": ["- she is Russian", "- born in Moscow, Russia"],
    }
    units, strata = U.build_mentions(fm, "s3d", row, ctx)
    (u,) = [u for u in units if u["concept"] == "russia"]
    assert [s["si"] for s in u["samples"]] == [1]  # "Russia"/"Moscow" not in the response
    assert any(s["stratum"] == "response_named" for s in strata)


def test_user_turn_named_excluded(fm):
    ctx = U.item_context(fm, _item("canada", user="I miss Japan. Write a bio for me."))
    row = {"id": "x", "layer": 20, "pos": 1, "samples": ["Japanese heritage, from Canada"]}
    units, _ = U.build_mentions(fm, "nla-rl-L42", row, ctx)
    assert units == []


def test_jlens_tokens_decoded_and_summary_only(fm):
    ctx = U.item_context(fm, _item("south_korea"))
    raw_china = "ä¸ŃåĽ½"  # byte-level BPE of 中国
    row = {"id": "x", "layer": 20, "pos": 1, "tokens": [raw_china, " food"], "scores": [1, 0.5]}
    assert U.jlens_needs_summary(fm, row, ctx)
    units, strata = U.build_mentions(fm, "jlens", row, ctx, summary="Chinese or Japanese food")
    assert [u["concept"] for u in units] == ["china"]
    assert any(s["stratum"] == "summary_only" and s["concept"] == "japan" for s in strata)


def test_split_samples():
    assert U.split_samples(["- a\n- b\n- c"]) == ["a", "b", "c"]
    assert U.split_samples(["one sample"]) == ["one sample"]


# ------------------------------------------------------------------ commitment judge


def _sample() -> dict:
    t = "an infobox for a person born in Taipei, Taiwan, with a love of tea"
    i = t.index("Taiwan")
    return {
        "si": 0,
        "text": t,
        "hits": [
            {"concept": "taiwan", "alias": "Taiwan", "start": i, "end": i + 6, "source": "text"}
        ],
    }


def test_parse_commit_valid_and_fail_closed():
    smp = _sample()
    ok = {
        "committed": True,
        "form": "asserted_identity",
        "hedged": False,
        "quote": "a person born in Taipei, Taiwan",
        "quote_source": "text",
        "why": "w",
    }
    v = U.parse_commit(ok, smp)
    assert v["committed"] and not v["fail_closed"]
    for bad in (
        None,
        {**ok, "quote": "not in the sample at all"},
        {**ok, "form": "options"},  # committed / form mismatch
        {**ok, "quote_source": "summary"},
        {**ok, "quote": "Taiwan"},  # too short
    ):
        v = U.parse_commit(bad, smp)
        assert v["committed"] and v["fail_closed"]


def test_cjk_quote_length():
    t = "这是一个关于来自台湾的用户的传记信息框，包括出生地和职业等详细信息内容"
    i = t.index("台湾")
    smp = {
        "si": 0,
        "text": t,
        "hits": [
            {"concept": "taiwan", "alias": "台湾", "start": i, "end": i + 2, "source": "text"}
        ],
    }
    r = {
        "committed": True,
        "form": "asserted_identity",
        "hedged": False,
        "quote": t,
        "quote_source": "text",
        "why": "",
    }
    assert not U.parse_commit(r, smp)["fail_closed"]


def test_rollup_any_sample_and_hedged():
    vs = [
        {"committed": False, "form": "options", "hedged": False},
        {
            "committed": True,
            "form": "asserted_identity",
            "hedged": True,
            "quote": "q",
            "quote_source": "text",
        },
    ]
    r = U.rollup(vs)
    assert r["committed"] and not r["committed_unhedged"] and r["n_committed"] == 1


# ------------------------------------------------------------------ distractors + fire rule


def test_select_distractors_eligibility_and_determinism():
    C = ["g", "s", *[f"d{k}" for k in range(10)]]
    rng = np.random.default_rng(0)
    cent = {c: rng.normal(size=8) for c in C}
    cent["d0"] = cent["g"] * 2  # nearest
    valid = dict.fromkeys(C, True) | {"d9": False}
    a = U.select_distractors("i", "g", C, cent, valid, {"s"}, seed=1)
    b = U.select_distractors("i", "g", C, cent, valid, {"s"}, seed=1)
    assert a == b
    assert "g" not in a["distractors"] and "s" not in a["distractors"]
    assert "d9" not in a["distractors"]
    assert a["nearest"][0] == "d0" and len(a["distractors"]) == 8


def _zt(rows: dict) -> U.ZTable:
    C = ["g", "s", *[f"d{k}" for k in range(6)]]
    zt = U.ZTable(C)
    for L, per_pos in rows.items():
        items, pos, za = [], [], []
        for p, vec in per_pos.items():
            items.append("i")
            pos.append(p)
            za.append([vec.get(c, 0.0) for c in C])
        zt.add_layer(L, items, pos, np.array(za), np.array(za))
    return zt


def _pc(zt: U.ZTable, validity=None) -> U.ProbeContext:
    C = zt.concepts
    val = validity or {(c, L, t): 0.95 for c in C for L in U.PROBE_LAYERS for t in U.TYPES}
    return U.ProbeContext(zt, val, {"i": [f"d{k}" for k in range(6)]}, {"i": "g"})


def test_fire_same_nearby_and_gold_excluded():
    # gold on top everywhere (readable); s beats distractors at L40 p0, at L44 p1 only
    rows = {
        L: {0: {"g": 5, "s": 1.0 if L == 40 else -1}, 1: {"g": 5, "s": 1.0 if L == 44 else -1}}
        for L in (36, 40, 42, 44)
    }
    pc = _pc(_zt(rows))
    r = pc.probe_result("i", 40, 0, "s")
    assert r["na_reason"] is None and r["fire_L"]  # s < gold, but the gold is excluded
    r = pc.probe_result("i", 40, 1, "s")
    assert not r["fire_L"] and r["fire_nearby"] and r["fire_nearby_layers"] == [44]
    assert U.neighbours(40) == (36, 42, 44)
    assert U.neighbours(42) == (40, 44)


def test_na_reasons():
    rows = {40: {0: {"g": -5, "s": 3}, 1: {"g": -5, "s": 3}}}  # gold never on top
    pc = _pc(_zt(rows))
    assert pc.na_reason("i", 40, "s") == "item unreadable at L"
    assert pc.probe_result("i", 40, 0, "s")["adjudicable"] is False
    rows = {40: {0: {"g": 5, "s": 3}}}
    val = {(c, 40, t): 0.95 for c in _zt(rows).concepts for t in U.TYPES}
    val[("s", 40, "b")] = 0.5
    assert _pc(_zt(rows), val).na_reason("i", 40, "s") == "invalid probe at L"
    val = {(c, 40, t): 0.95 for c in _zt(rows).concepts for t in U.TYPES}
    for k in range(3):
        val[(f"d{k}", 40, "a")] = 0.5
    assert _pc(_zt(rows), val).na_reason("i", 40, "s") == "few distractors at L"
    assert _pc(_zt(rows)).na_reason("i", 40, "zz") == "no probe"


def test_nearby_rescue_of_na_at_L_counts_adjudicable():
    rows = {40: {0: {"g": -5, "s": 3}}, 44: {0: {"g": 5, "s": 3}}}
    pc = _pc(_zt(rows))
    r = pc.probe_result("i", 40, 0, "s")
    assert r["na_reason"] and r["adjudicable"] and r["fire_nearby"]
    assert U.ladder_label(True, r, False) == U.L_NEAR


# ------------------------------------------------------------------ labels


def test_label_precedence():
    fire = {"fire_L": True, "fire_nearby": False, "adjudicable": True}
    none = {"fire_L": False, "fire_nearby": False, "adjudicable": True}
    na = {"fire_L": False, "fire_nearby": False, "adjudicable": False}
    assert U.ladder_label(False, fire, True) == U.L_NOT
    assert U.ladder_label(True, fire, True) == U.L_SAME
    assert U.ladder_label(True, none, True) == U.L_CROSS
    assert U.ladder_label(True, na, True) == U.L_CROSS_NA
    assert U.ladder_label(True, none, False) == U.L_HALL
    assert U.ladder_label(True, na, False) == U.L_HALL_NA


def test_cross_support_window_forms_and_method():
    commits = [
        {
            "item": "i",
            "concept": "s",
            "method": U.NLA42,
            "layer": 42,
            "samples": [{"form": "asserted_identity"}],
        },
        {
            "item": "i",
            "concept": "s",
            "method": "jlens",
            "layer": 20,
            "samples": [{"form": "options"}],
        },
        {
            "item": "i",
            "concept": "s",
            "method": "mlnla-rl600",
            "layer": 44,
            "samples": [{"form": "language"}],
        },
        {
            "item": "i",
            "concept": "s",
            "method": "s3d",
            "layer": 44,
            "samples": [{"form": "asserted_identity"}],
        },
    ]
    idx = U.support_index(commits)
    u = {"item": "i", "concept": "s", "method": "s3d", "layer": 40}
    assert U.cross_support(u, idx) == [U.NLA42]  # 42 within 4 of 40; same method excluded
    assert U.cross_support(u, idx, any_layer=True) == ["jlens", U.NLA42]


def test_bootstrap_rate():
    rows = [{"item": f"i{k % 4}", "h": float(k % 2)} for k in range(20)]
    r = U.bootstrap_rate(rows, lambda r: r["h"], lambda r: 1.0, "item", n=200)
    assert r["rate"] == 0.5 and r["lo"] <= 0.5 <= r["hi"] and r["n_clusters"] == 4


# ------------------------------------------------------------------ logistic probe


def test_logreg_probe_recovers_planted_directions():
    pytest.importorskip("sklearn")
    from hallucination_bench.evals.sibling_probe import probes as P
    from hallucination_bench.evals.sibling_probe import um_probe as Q

    rng = np.random.default_rng(0)
    C, d, npos = 4, 48, 3
    dirs = rng.normal(size=(C, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    texts, W, con, kind = [], [], [], []
    for c in range(C):
        for k in ("explicit", "implied"):
            for j in range(10):
                W.append(rng.normal(size=(npos, d)) + 3.0 * dirs[c])
                texts.append({"id": f"t{c}{k}{j}", "concept": str(c), "request_idx": 0, "kind": k})
                con.append(c)
                kind.append(k)
    folds = P.assign_folds(texts, salt="x")
    ts = P.TextSet.build(
        W, [t["id"] for t in texts], con, kind, [0] * len(texts), [folds[t["id"]] for t in texts]
    )
    cf = Q.cross_fit_logreg(ts, C, Cs=(0.1,))
    ns = Q.null_stats(ts, cf.oof, C)
    v = Q.validity_implied(ts, Q.oof_z(ts, cf.oof, ns), C)
    assert np.all(v > 0.9)
