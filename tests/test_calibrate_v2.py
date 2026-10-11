"""calibrate_v2 (plan 0002): v2 case validation against the production splitter, unit-level and
segment-level scoring, the gate, the dev coverage, and fresh-call repeats."""

import asyncio
import copy
import json

import pytest
from test_claimmap_v2 import MapFake, e, sp, unit_lines

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.cache import Cache
from hallucination_bench.judges import claimmap_v2 as cm

M = "⟦READ⟧"


def case(**kw):
    c = {
        "case_id": "t1",
        "part": "synthetic",
        "rules": ["x"],
        "prompt": "Who painted the Mona Lisa?",
        "response": f"Leonardo da Vinci{M} painted it.",
        "readouts": ["It was Picasso. A painting Q&A. Next comes the date."],
        "established": [["It was Picasso."]],
    }
    c.update(kw)
    c = cal.skeleton(c)
    leaves = kw.pop("leaves", None) or ["substitution_sibling", "restates_prefix", "stated_after"]
    for g, lf in zip(c["gold"]["units"], leaves, strict=True):
        g["leaf"] = lf
    return c


def test_skeleton_and_validation_follow_the_splitter():
    c = case()
    assert [g["text"] for g in c["gold"]["units"]] == [
        "It was Picasso.",
        "A painting Q&A.",
        "Next comes the date.",
    ]
    assert c["gold"]["units"][0]["established"] is True
    assert cal.validate_case(c) == []
    bad = copy.deepcopy(c)
    bad["gold"]["units"][1]["text"] = "A painting"
    assert any("does not match the splitter" in x for x in cal.validate_case(bad))
    bad = copy.deepcopy(c)
    bad["gold"]["units"][0]["leaf"] = "restates_prefix"  # established, not incorrect, no dispute
    assert any("established unit" in x for x in cal.validate_case(bad))
    bad["gold"]["units"][0]["dispute"] = True
    assert cal.validate_case(bad) == []
    bad = copy.deepcopy(c)
    bad["gold"]["units"][2]["splits"] = [{"text": "comes", "leaf": "filler"}]
    assert any("rest_leaf is required" in x for x in cal.validate_case(bad))
    bad["gold"]["units"][2]["splits"] = [{"text": "nope", "leaf": "filler"}]
    bad["gold"]["units"][2]["rest_leaf"] = "stated_after"
    assert any("not a verbatim" in x for x in cal.validate_case(bad))
    assert any("exactly once" in x for x in cal.validate_case({**c, "response": "no marker"}))
    assert any("not found" in x for x in cal.validate_case({**c, "established": [["Monet"]]}))


def _preds(c, entries):
    units, _ = cal.case_units(c)
    return cm.parse({"units": entries, "rationale": ""}, units)


def test_unit_level_scoring_majority_leaf_and_segment_recall():
    c = case(
        readouts=["It was Picasso. Alpha is red and Paris is big. Next comes the date."],
        leaves=["substitution_sibling", None, "stated_after"],
    )
    # gold splits unit 2: a minority false_claim sub-span inside a mostly-correct unit
    g = c["gold"]["units"][1]
    g["splits"] = [{"text": "Paris is big", "leaf": "false_claim"}]
    g["rest_leaf"] = "restates_prefix"
    assert cal.validate_case(c) == []
    pred = _preds(
        c,
        [
            e("R0.U1", "substitution_category"),  # wrong sub-type, right group
            e("R0.U2", "restates_prefix"),  # unsplit: misses the minority false_claim
            e("R0.U3", "restates_prefix", dispute=True),  # wrong leaf, right group
        ],
    )
    m = cal.run_metrics([c], [pred])
    assert m["n_units"] == 3 and m["n_gold_segments"] == 4  # the "." rest piece is skipped
    assert m["leaf_acc_unit"] == pytest.approx(1 / 3)  # unit 2's majority is restates_prefix
    assert m["group_acc_unit"] == 1.0  # every unit-majority group is right
    assert m["recall_false_claim"] == 0.0  # the minority segment is scored on its own chars
    assert m["recall_correct"] == 1.0
    assert m["recall_substitution"] == 1.0 and m["sibling_vs_category"] == 0.0
    assert m["leaf_recall"]["false_claim"] == {"recall": 0.0, "n": 1}
    assert 0 < m["claim_f1_char"] <= 1
    assert m["group_acc_char"] < 1 and m["leaf_acc_char"] < m["group_acc_char"]


def test_forced_and_disputed_predictions_are_scored_by_raw_and_alt_leaf():
    c = case()
    c["gold"]["units"][0]["leaf"] = "restates_prefix"
    c["gold"]["units"][0]["dispute"] = True
    forced = _preds(c, [e("R0.U1", "restates_prefix"), e("R0.U2"), e("R0.U3", "stated_after")])
    assert forced[1]["n_forced_established"] == 1
    disputed = _preds(
        c,
        [
            e("R0.U1", "detail", dispute=True, alt_leaf="restates_prefix"),
            e("R0.U2"),
            e("R0.U3", "stated_after"),
        ],
    )
    assert disputed[1]["n_disputed_established"] == 1
    for p in (forced, disputed):
        m = cal.run_metrics([c], [p])
        assert m["leaf_acc_unit"] == 1.0
    assert cal.run_metrics([c], [forced])["n_forced_established"] == 1


def test_gate_requires_wilson_lower_bound_on_recalls_and_flags_flips():
    c = case()
    good = _preds(c, [e("R0.U1", "substitution_sibling"), e("R0.U2"), e("R0.U3", "stated_after")])
    other = _preds(c, [e("R0.U1", "substitution_sibling"), e("R0.U2"), e("R0.U3", "filler")])
    res = cal.score([c], [[good], [other]], "heldout")
    assert res["mean"]["group_flip_rate"] == pytest.approx(1 / 3)
    rows = {g["key"]: g for g in res["gate"]}
    # recall_substitution = 1.0 but n = 1: the Wilson lower bound is far below 0.58
    assert rows["recall_substitution"]["value"] == 1.0
    assert rows["recall_substitution"]["wilson_lo"] < 0.58
    assert rows["recall_substitution"]["passed"] is False
    assert rows["group_flip_rate"]["passed"] is False and not res["gate_passed"]
    assert rows["real_group_acc_unit"]["passed"] is False  # no real part on held-out: fail
    dev = cal.score([c], [[good], [good]], "dev")
    assert {g["key"]: g for g in dev["gate"]}["real_group_acc_unit"]["passed"] is None
    assert "disagreements" in dev and "disagreements" not in res


def test_repeats_are_fresh_calls_and_then_cached(tmp_path):
    cases = [case(), case(case_id="t2")]
    fake = MapFake()

    def run(llm):
        with Cache(tmp_path / "c.jsonl") as cache:
            return asyncio.run(
                cal.map_cases(
                    cases,
                    split="dev",
                    model="claude-opus-5-5",
                    effort="low",
                    cache=cache,
                    llm=llm,
                    repeats=2,
                )
            )

    preds = run(fake)
    assert fake.n("map") == 4  # 2 cases x 2 repeats, each a separate call
    assert len(preds) == 2 and all(len(p) == 2 for p in preds)
    rows = [json.loads(x) for x in (tmp_path / "c.jsonl").read_text().splitlines()]
    assert sorted(r["key"] for r in rows) == [
        "v2:dev:t1:r0",
        "v2:dev:t1:r1",
        "v2:dev:t2:r0",
        "v2:dev:t2:r1",
    ]
    assert len({r["fingerprint"] for r in rows if r["key"].startswith("v2:dev:t1")}) == 2
    fake2 = MapFake()
    run(fake2)
    assert fake2.n("map") == 0
    # single-call (default, plan rev 13): the established unit keeps its own unit but is not
    # flagged; under --anchor v5c it is flagged as in rev 12 production
    user = fake.calls[0][3]
    assert unit_lines(user)[0] == ("R0.U1", False, "It was Picasso.")
    assert "ESTABLISHED" not in user.split("<readout")[1]
    fake3 = MapFake()
    with Cache(tmp_path / "c.jsonl") as cache:
        asyncio.run(
            cal.map_cases(
                cases[:1],
                split="dev",
                model="claude-opus-5-5",
                effort="low",
                cache=cache,
                llm=fake3,
                repeats=1,
                anchor="v5c",
            )
        )
    assert fake3.n("map") == 1  # a different prompt: not served from the single-call cache
    assert unit_lines(fake3.calls[0][3])[0] == ("R0.U1", True, "It was Picasso.")


def test_invalid_answer_scores_failed():
    c = case()
    fake = MapFake({"map": lambda u: {"units": [], "rationale": ""}})
    preds = asyncio.run(
        cal.map_cases(
            [c],
            split="dev",
            model="m",
            effort=None,
            cache=_MemCache(),
            llm=fake,
            repeats=1,
        )
    )
    segs, _ = preds[0][0]
    assert {s.leaf for s in segs} == {tx.FAILED}


class _MemCache:
    def __init__(self):
        self.rows = {}

    def get(self, key, fp, include_failed=False):
        r = self.rows.get((key, fp))
        return r if r is not None and (include_failed or r["result"] is not None) else None

    def put(self, key, fp, payload):
        self.rows[(key, fp)] = payload


def test_dev_v2_is_valid_and_meets_the_coverage_requirements():
    cases = cal.load_cases("dev")
    cov = cal.coverage(cases)
    assert cov["n_units"] >= cal.DEV_MIN_UNITS
    assert all(n >= cal.DEV_MIN_PER_LEAF for n in cov["segments_per_leaf"].values()), cov
    assert cov["n_established_units"] >= 5 and cov["n_disputed_established"] >= 1
    rules = {r for c in cases for r in c["rules"]}
    for anchor in (
        "rule10-asserted-invented-event",
        "rule10-unrealised-prediction",
        "world-true",
        "possible-belief",
        "sibling-substitution",
        "category-substitution",
        "detail-fictional-name",
        "sibling-subject-name",
        "detail-incidental-number",
        "real-event-date",
        "count-flips-claim",
        "hedged-hit",
        "hedged-miss",
        "eval-awareness-model-state",
        "token-framing-vs-interpretation",
        "looks-off-topic-but-on-topic",
        "established-disputed",
        "rule12-hedged",
        "rule12-flat",
    ):
        assert anchor in rules, anchor
    assert any(c.get("kind") == "tokens" for c in cases)
    assert cov["segments_per_leaf"]["detail"] >= 12
    framing = [c for c in cases if "token-model-is-framing" in c["rules"]]
    assert len(framing) >= 2 and all(c["kind"] == "tokens" for c in framing)
    # every split / hedged / dispute gold is consistent with the production splitter
    assert all(cal.validate_case(c) == [] for c in cases)


def test_schema_example_in_readme_is_valid():
    from hallucination_bench import DATA

    text = (DATA / "calibration" / "README.md").read_text(encoding="utf-8")
    block = text.split("<!-- v2-example -->")[1].split("```json")[1].split("```")[0]
    assert cal.validate_case(json.loads(block)) == []


def test_sp_helper_matches_split_schema():
    assert set(sp("t", "detail")) == set(cm.SPLIT_SCHEMA_V5C["required"])
    assert set(cm.SPLIT_SCHEMA["required"]) == set(sp("t", "detail")) - {"dispute", "alt_leaf"}
