"""Plan 0005: lexicon matcher, units, commitment judge parse / roll-up, cross-method, fire rule,
labels, probe math (synthetic)."""

import numpy as np
import pytest

from hallucination_bench.evals.sibling_probe import ladder as LD
from hallucination_bench.evals.sibling_probe import ladder_lex as LX
from hallucination_bench.evals.sibling_probe import ladder_probe as LP

LEX = {
    "categories": {
        "sport": {
            "concepts": {
                "tennis": {"label": "Tennis", "en": ["tennis"], "cjk": ["网球"], "tokens": []},
                "table_tennis": {
                    "label": "Table tennis",
                    "en": ["table tennis", "ping pong"],
                    "cjk": ["乒乓球"],
                    "tokens": [],
                },
                "ice_hockey": {
                    "label": "Ice hockey",
                    "en": ["ice hockey", "hockey"],
                    "cjk": ["冰球"],
                    "tokens": [],
                },
                "field_hockey": {
                    "label": "Field hockey",
                    "en": ["field hockey"],
                    "cjk": [],
                    "tokens": [],
                },
                "swimming": {
                    "label": "Swimming",
                    "en": ["swimming", "swimmer"],
                    "cjk": ["游泳"],
                    "tokens": ["Swim"],
                },
                "boxing": {
                    "label": "Boxing",
                    "en": ["boxing", "boxer"],
                    "cjk": ["拳击"],
                    "tokens": [],
                },
                "soccer": {"label": "Soccer", "en": ["soccer"], "cjk": ["足球"], "tokens": []},
                "american_football": {
                    "label": "American football",
                    "en": ["American football", "NFL"],
                    "cjk": [],
                    "tokens": [],
                },
                "track_and_field": {
                    "label": "Track and field",
                    "en": ["track and field", "athletics", "running"],
                    "cjk": [],
                    "tokens": [],
                },
            },
            "block": [],
        },
        "country": {
            "concepts": {
                "united_kingdom": {
                    "label": "UK",
                    "en": ["United Kingdom", "UK", "England", "Britain", "British"],
                    "cjk": ["英国"],
                    "tokens": [],
                },
                "russia": {
                    "label": "Russia",
                    "en": ["Russia", "Soviet Union", "USSR"],
                    "cjk": [],
                    "tokens": [],
                },
                "united_states": {
                    "label": "US",
                    "en": ["United States", "US", "America"],
                    "cjk": [],
                    "tokens": [],
                },
                "turkey": {"label": "Turkey", "en": ["Turkey"], "cjk": [], "tokens": []},
                "india": {
                    "label": "India",
                    "en": ["India", "Indian"],
                    "cjk": ["印度"],
                    "tokens": [],
                },
            },
            "block": ["Mexico City", "印度洋", "Indian Ocean"],
        },
    },
    "ambiguous": {"sport": {"football": ["american_football", "soccer"]}},
}


@pytest.fixture
def sport():
    return LX.Matcher(LEX, "sport")


def concepts(hits):
    return [h.concept for h in hits if h.concept]


def test_longest_match_wins(sport):
    assert concepts(sport.find("He plays table tennis, not tennis.")) == ["table_tennis", "tennis"]
    assert concepts(sport.find("field hockey and hockey")) == ["field_hockey", "ice_hockey"]
    assert concepts(sport.find("乒乓球 and 网球")) == ["table_tennis", "tennis"]


def test_word_boundaries_and_plural(sport):
    assert concepts(sport.find("two boxers")) == ["boxing"]
    assert concepts(sport.find("unboxing a parcel")) == []


def test_football_resolves_by_gold_and_item(sport):
    h = sport.find("He played football.", gold="soccer", item="br-ent-ath-messi")
    assert concepts(h) == ["soccer"] and h[0].ambiguous
    h = sport.find("He played football.", gold="american_football", item="br-ent-ath-brady")
    assert concepts(h) == ["american_football"]
    h = sport.find("He played football.", gold="boxing", item="br-ent-ath-ali")
    assert concepts(h) == ["american_football"]  # FOOTBALL_DEFAULT
    h = sport.find("He played football.", gold="tennis", item="br-ent-ath-federer")
    assert concepts(h) == ["soccer"]


def test_country_case_sensitivity_and_blocks():
    m = LX.Matcher(LEX, "country")
    assert concepts(m.find("Let us go")) == []  # "US" upper-case only
    assert concepts(m.find("the US and Turkey")) == ["united_states", "turkey"]
    assert concepts(m.find("a turkey dinner")) == []
    assert concepts(m.find("Mexico City and the Indian Ocean, 印度洋")) == []
    assert concepts(m.find("England, the Soviet Union")) == ["united_kingdom", "russia"]


def test_token_concept(sport):
    assert sport.token_concept(" Swim") == ("swimming", False)
    assert sport.token_concept(" Boxers") == ("boxing", False)
    assert sport.token_concept("网球运动") == ("tennis", False)
    assert sport.token_concept(" football", gold="soccer") == ("soccer", True)
    assert sport.token_concept(" athletic") is None


def test_overlap_report_flags_duplicates():
    lex = {
        "categories": {
            "x": {
                "concepts": {"a": {"en": ["foo"], "cjk": ["一"]}, "b": {"en": ["Foo"], "cjk": []}}
            }
        }
    }
    rep = LX.overlap_report(lex)["x"]
    assert rep["duplicates"] == {"foo": ["a", "b"]}
    assert rep["short_cjk"] == ["a:一"]


def test_split_samples():
    assert LD.split_samples(["- a b\n- c\n- d"]) == ["a b", "c", "d"]
    assert LD.split_samples(["Genre: x\n- not a split at start"]) == [
        "Genre: x\n- not a split at start"
    ]


ITEMS = {
    "br-ent-ath-ali": {
        "id": "br-ent-ath-ali",
        "category": "sport",
        "gold": "boxing",
        "entity": "Muhammad Ali",
        "prompt": "The athlete Muhammad Ali plays the sport of",
    },
    "br-ent-lmk-big": {
        "id": "br-ent-lmk-big",
        "category": "country",
        "gold": "england",
        "entity": "Big Ben",
        "prompt": "The Big Ben is located in the country of",
    },
}


def test_build_units_excludes_gold_equivalents_and_prompt():
    m = {"sport": LX.Matcher(LEX, "sport"), "country": LX.Matcher(LEX, "country")}
    cells = {
        "s3d": [
            {
                "item": "br-ent-ath-ali",
                "layer": 20,
                "pos": 7,
                "samples": [
                    "He is a boxer who swims: swimming.",
                    "The answer is tennis",
                    "Boxing.",
                ],
            },
            {
                "item": "br-ent-lmk-big",
                "layer": 20,
                "pos": 8,
                "samples": ["It is in England, Britain. Not Russia."],
            },
        ],
        "jlens": [
            {
                "item": "br-ent-ath-ali",
                "layer": 36,
                "pos": 7,
                "tokens": [" Swim", " boxing", " athletic"],
                "summary": "Probably swimming or tennis.",
            },
        ],
    }
    units = LD.build_units(cells, ITEMS, m)
    uids = {u["uid"] for u in units}
    assert uids == {
        "s3d|br-ent-ath-ali|20|7|swimming",
        "s3d|br-ent-ath-ali|20|7|tennis",
        "s3d|br-ent-lmk-big|20|8|russia",
        "jlens|br-ent-ath-ali|36|7|swimming",
        "jlens|br-ent-ath-ali|36|7|tennis",
    }
    js = next(u for u in units if u["uid"] == "jlens|br-ent-ath-ali|36|7|swimming")
    assert js["sources"] == ["summary", "token"] and len(js["samples"]) == 1
    s3 = next(u for u in units if u["uid"] == "s3d|br-ent-ath-ali|20|7|swimming")
    assert s3["n_cell_samples"] == 3 and [s["si"] for s in s3["samples"]] == [0]


def _sample(text="He is an Olympian who competes in the sport of swimming", alias="swimming"):
    i = text.index(alias)
    return {
        "si": 0,
        "text": text,
        "hits": [{"source": "text", "alias": alias, "start": i, "end": i + len(alias)}],
    }


def test_render_commit_marks_and_hides_layer():
    u = {"category": "sport", "layer": 20}
    sys_, user = LD.render_commit(
        u, _sample(), "The athlete Muhammad Ali plays the sport of", "Swimming"
    )
    assert "⟦swimming⟧" in user and "20" not in user and "layer" not in user.lower()
    assert "quiz_distractor" in sys_


def test_parse_commit_valid_and_fail_closed():
    s = _sample()
    ok = {
        "committed": True,
        "form": "fact_about_subject",
        "subject": "prompt_subject",
        "quote": "competes in the sport of ⟦swimming⟧",
        "why": "x",
    }
    v = LD.parse_commit(ok, s)
    assert v["committed"] and not v["fail_closed"] and "⟦" not in v["quote"]
    for bad in (
        None,
        {**ok, "quote": "not in the text at all"},
        {**ok, "quote": "swimming"},  # < 3 words
        {**ok, "quote": "He is an Olympian who"},  # no mention in the quote
        {**ok, "committed": False},  # inconsistent with the form
        {**ok, "form": "nonsense"},
    ):
        v = LD.parse_commit(bad, s)
        assert v["committed"] and v["fail_closed"]


def test_rollup_any_sample_and_support():
    nc = {
        "committed": False,
        "form": "negation",
        "subject": "none",
        "quote": "q",
        "fail_closed": False,
    }
    c = {
        "committed": True,
        "form": "substituted_entity_answer",
        "subject": "other_entity",
        "quote": "Usain Bolt track",
        "fail_closed": False,
    }
    r = LD.rollup([nc, c])
    assert (
        r["committed"]
        and r["n_committed"] == 1
        and r["entity_substitution"]
        and r["quote"] == "Usain Bolt track"
    )
    r = LD.rollup([nc])
    assert not r["committed"] and not r["supports"]


def test_cross_support_grid_and_nla42():
    u = {"method": "s3d", "layer": 40}
    others = [
        {"method": "nla-rl-L42", "layer": 42, "supports": True},
        {"method": "jlens", "layer": 52, "supports": True},
        {"method": "mlnla-rl600", "layer": 44, "supports": False},
        {"method": "s3d", "layer": 40, "supports": True},
    ]
    r = LD.cross_support(u, others, lambda m: 0.6 if m == "nla-rl-L42" else 0.1)
    assert r["primary"]["methods"] == ["nla-rl-L42"] and r["primary"]["support"]
    assert not r["primary"]["support_no_prior"]
    assert (
        r["any_layer"]["methods"] == ["jlens", "nla-rl-L42"] and r["any_layer"]["support_no_prior"]
    )
    n = LD.cross_support(
        {"method": "nla-rl-L42", "layer": 42},
        [{"method": "s3d", "layer": 48, "supports": True}],
        lambda m: 0,
    )
    assert not n["primary"]["support"] and n["any_layer"]["support"]


def _cell(**kw):
    c = {
        "classes": ["boxing", "swimming", "tennis"],
        "n_same": 20,
        "auroc": {"logreg": {"swimming": 0.9}, "dom": {"swimming": 0.95}},
        "p_concept": {"logreg": {"swimming": 0.04}, "dom": {"swimming": 0.03}},
        "p_gold": {"logreg": 0.01, "dom": 0.02},
    }
    c.update(kw)
    return c


def test_fire_rule():
    assert LD.fires(_cell(), "swimming", "boxing")["status"] == "fires"
    assert (
        LD.fires(
            _cell(p_concept={"logreg": {"swimming": 0.04}, "dom": {"swimming": 0.2}}),
            "swimming",
            "boxing",
        )["status"]
        == "no_fire"
    )
    assert "few refs" in LD.fires(_cell(n_same=18), "swimming", "boxing")["reason"]
    assert (
        "validity"
        in LD.fires(
            _cell(auroc={"logreg": {"swimming": 0.84}, "dom": {"swimming": 0.95}}),
            "swimming",
            "boxing",
        )["reason"]
    )
    assert (
        "gold control"
        in LD.fires(_cell(p_gold={"logreg": 0.06, "dom": 0.0}), "swimming", "boxing")["reason"]
    )
    assert "not a bank class" in LD.fires(_cell(), "golf", "boxing")["reason"]
    assert LD.fires(None, "swimming", "boxing")["status"] == "n/a"
    # the probe pair is configurable (wdm as the DiffMean slot)
    c = _cell(
        auroc={"logreg": {"swimming": 0.9}, "wdm": {"swimming": 0.9}},
        p_concept={"logreg": {"swimming": 0.01}, "wdm": {"swimming": 0.01}},
        p_gold={"logreg": 0.01, "wdm": 0.01},
    )
    assert LD.fires(c, "swimming", "boxing", probes=("logreg", "wdm"))["fires"]


def test_labels():
    assert LD.ladder_label(False, True, True) == LD.L_NOT
    assert LD.ladder_label(True, True, False) == LD.L_PROBE
    assert LD.ladder_label(True, False, True) == LD.L_CROSS
    assert LD.ladder_label(True, False, False) == LD.L_HALL


def test_emp_p_and_auroc():
    assert LP.emp_p(5.0, np.arange(19)) == pytest.approx((1 + 14) / 20)
    assert LP.emp_p(100.0, np.arange(19)) == pytest.approx(1 / 20)
    assert LP.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert LP.auroc(np.array([1.0]), np.array([1.0])) == 0.5


def test_whitened_dm_matches_direct_solve():
    rng = np.random.default_rng(0)
    n, d, C = 60, 20, 3
    X = rng.normal(size=(n, d))
    y = np.arange(n) % C
    t = (np.arange(n) // C) % 2
    w = LP.WhitenedDM(X, y, t, C)
    lam = 0.1
    R = w.R
    S = R.T @ R / w.m
    a = lam * np.trace(S) / d
    direct = np.linalg.solve(S + a * np.eye(d), w.d0.T).T
    direct /= np.linalg.norm(direct, axis=1, keepdims=True)
    assert np.allclose(w.directions(lam), direct, atol=1e-8)


def test_fit_bank_layer_synthetic():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    C, E, D = 4, 25, 64
    mus = rng.normal(size=(C, D))
    X, y, t, g = [], [], [], []
    for c in range(C):
        for e in range(E):
            base = rng.normal(size=D) * 0.5
            for ti in range(3):
                X.append(base + mus[c] + rng.normal(size=D) * 0.3)
                y.append(c)
                t.append(ti)
                g.append(f"{c}-{e}")
    X, y, t, g = map(np.array, (X, y, t, g))
    Xi = (mus[[0]] + mus[[1]] * 0.8).astype(np.float32)
    r = LP.fit_bank_layer(X, y, t, g, Xi, C, ks=(8, 16), lams=(0.1, 1.0))
    classes = ["a", "b", "c", "d"]
    au = LP.class_aurocs(r["oof"], y, t, classes)
    assert min(au["logreg"].values()) > 0.95 and min(au["dom"].values()) > 0.95
    cell = LP.item_cell(
        {p: r["items"][p][0] for p in LP.PROBE_TYPES}, r["oof"], y, t, classes, "a", au
    )
    assert cell["n_same"] == E
    # the item carries class b on top of its gold a: b fires vs same-gold references
    assert cell["p_concept"]["dom"]["b"] <= 0.05 and cell["p_gold"]["dom"] <= 0.05
    assert LD.fires(cell, "b", "a", probes=("logreg", "dom"))["status"] in ("fires", "n/a")


def test_viewer_ladder_badge_and_writer(tmp_path):
    import json
    import sys
    from pathlib import Path

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "scripts" / "sibling_probe" / "viewer")
    )
    import build_viewer as BV

    u = {
        "uid": "s3d|it2|20|0|swimming",
        "method": "s3d",
        "item": "it2",
        "layer": 20,
        "eff_layer": 20,
        "pos": 0,
        "concept": "swimming",
        "label": LD.L_HALL,
        "committed": True,
        "n_committed": 1,
        "n_not_committed": 0,
        "quote": "competes in swimming",
        "probe": {"status": "no_fire", "reason": "", "pair": ["logreg", "dom"], "view": {}},
        "cross": {
            "primary": {"layers": [], "support_no_prior": False},
            "any_layer": {"layers": [["jlens", 44]]},
        },
        "hits": [{"si": 1, "source": "text", "start": 3, "end": 11}],
    }
    b = BV.ladder_badge(u, {1: 10})
    assert b["spans"] == [{"src": "text", "start": 13, "end": 21}] and b["label"] == LD.L_HALL
    items = [
        {
            "set": "B",
            "id": "it2",
            "prompt": "x",
            "category": "sport",
            "gold": "g",
            "response": "",
            "text": "x",
            "pos": [[0, 0, 1]],
        }
    ]
    idx = BV.write_viewer(items, {}, [], tmp_path, ladder={"it2": [b]})
    assert idx["ladder"] and idx["items"][0]["ladder"] == {LD.L_HALL: 1}
    d = json.loads((tmp_path / "items" / "B__it2.json").read_text())
    assert d["ladder"][0]["uid"] == u["uid"]
    idx2 = BV.write_viewer(items, {}, [], tmp_path / "plain")
    assert "ladder" not in idx2 and "ladder" not in idx2["items"][0]


def test_bare_country_names_in_real_lexicon():
    import json

    lex = json.loads(LX.LEXICON_PATH.read_text())
    m = LX.Matcher(lex, "country")
    ids = lambda t: [h.concept for h in m.find(t) if h.concept]  # noqa: E731
    assert ids("Paris is in Georgia") == ["georgia"]
    assert ids("a georgia peach") == []  # lower-case: case-sensitive
    assert ids("Jordan, Guinea, Niger and Luxembourg") == [
        "jordan",
        "guinea",
        "niger",
        "luxembourg",
    ]
    assert ids("Papua New Guinea") == ["papua_new_guinea"]
    assert ids("Guinea-Bissau and Equatorial Guinea") == ["guinea_bissau", "equatorial_guinea"]
    assert ids("Georgia Tech and the Jordan River") == []
    # athlete items are sport-category: a person name is not a country mention there
    sport = LX.Matcher(lex, "sport")
    assert [h.concept for h in sport.find("Michael Jordan") if h.concept] == []


def test_fire_nan_auroc_and_fail_closed_support():
    c = {
        "classes": ["a", "b"],
        "n_same": 30,
        "auroc": {"logreg": {"a": float("nan")}, "dom": {"a": 0.95}},
        "p_concept": {"logreg": {"a": 0.01}, "dom": {"a": 0.01}},
        "p_gold": {"logreg": 0.5, "dom": 0.5},
    }
    r = LD.fires(c, "a", "b")
    assert r["status"] == "n/a" and r["reason"] == "validity_nan"
    fc = {"committed": True, "form": "fail_closed", "fail_closed": True}
    ok = {"committed": True, "form": "answer", "fail_closed": False}
    assert LD.rollup([fc])["supports"] is False
    assert LD.rollup([fc, ok])["supports"] is True
