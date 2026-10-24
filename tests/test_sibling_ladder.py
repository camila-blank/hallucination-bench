"""Plan 0005: lexicon matcher, units, commitment judge parse / roll-up, cross-method, fire rule,
labels, probe math (synthetic)."""

import pytest

from hallucination_bench.evals.sibling_probe import ladder_lex as LX

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
