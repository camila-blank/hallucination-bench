"""taxonomy-v1: prompt render, schema, span-index matching, fail-closed parsing, routing,
calibration cases and repeats."""

import asyncio
import json
from collections import Counter

import pytest
from conftest import FakeLLM, readout_texts, span_lines

from hallucination_bench import calibrate
from hallucination_bench import taxonomy as tx
from hallucination_bench.cache import Cache
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import taxonomy_v1 as tv1

SPANS = [
    tv1.SpanIn(0, "Picasso", "contradicted"),
    tv1.SpanIn(1, "his sister Maria", "unsupported"),
    tv1.SpanIn(1, "angry", "disputed"),
]


def entry(sid, typ="concept", target="entity", sev=2, uk=None, why="w"):
    return {
        "span_id": sid,
        "type": typ,
        "target": target,
        "severity": sev,
        "unsupported_kind": uk,
        "why": why,
    }


# ------------------------------------------------------------------ prompt


def test_prompt_contains_every_label_and_definition():
    for s in tx.definitions():
        assert s in tv1.SYSTEM, s
    for label in [*tx.TYPES, *tx.TARGETS, *tx.UNSUPPORTED_KINDS, *tx.GROUNDINGS]:
        assert f"`{label}`" in tv1.SYSTEM
    assert "Picasso" in tv1.SYSTEM and "Steve" in tv1.SYSTEM


def test_render_numbers_spans_with_readout_and_grounding():
    system, user = tv1.render_prompt(
        "Tell me {about} da Vinci", "Leonardo⟦READ⟧ painted.", [(0, "Picasso"), (2, "x")], SPANS
    )
    assert system == tv1.SYSTEM
    assert "Tell me {about} da Vinci" in user  # braces are never expanded
    assert span_lines(user) == [
        ("S1", 0, "contradicted", "Picasso"),
        ("S2", 1, "unsupported", "his sister Maria"),
        ("S3", 1, "disputed", "angry"),
    ]
    assert readout_texts(user) == ["Picasso", "x"]
    assert '<readout idx="2">' in user
    with pytest.raises(ValueError):
        tv1.render_prompt("p", "r", [(0, "x")], [])


def test_schema_is_strict_with_enums():
    item = tv1.SCHEMA["schema"]["properties"]["spans"]["items"]
    assert item["additionalProperties"] is False
    required = {"span_id", "type", "target", "severity", "unsupported_kind", "why"}
    assert set(item["required"]) == required
    assert item["properties"]["type"]["enum"] == ["concept", "detail"]
    assert item["properties"]["severity"]["enum"] == [1, 2, 3]
    assert len(item["properties"]["target"]["enum"]) == 7


# ------------------------------------------------------------------ parse / validate


def test_parse_happy_path_and_kind_rules():
    raw = {
        "spans": [
            entry("S1", uk="inference"),  # kind on a contradicted span: ignored
            entry("s2", "detail", "quantity", 1, "fabricated"),
            entry("3", "concept", "stance", 3, None),
        ],
        "rationale": "r",
    }
    assert tv1.validate(raw, SPANS)
    labels, counts = tv1.parse(raw, SPANS)
    assert [lab.type for lab in labels] == ["concept", "detail", "concept"]
    assert labels[0].unsupported_kind is None and counts["n_kind_ignored"] == 1
    assert labels[1].unsupported_kind == "fabricated" and labels[1].severity == 1
    assert labels[2].unsupported_kind is None
    assert counts["n_unclassified"] == 0


def test_parse_fails_closed_on_missing_duplicate_and_extra_ids():
    raw = {
        "spans": [entry("S1"), entry("S1", "detail"), entry("S2", uk=None), entry("S9")],
        "rationale": "",
    }
    assert tv1.validate(raw, SPANS)  # id problems do not invalidate
    labels, counts = tv1.parse(raw, SPANS)
    assert labels[0].type == "unknown"  # duplicated
    assert labels[1].type == "concept" and labels[1].unsupported_kind == "unknown"  # kind missing
    assert labels[2].type == "unknown" and labels[2].severity is None  # missing
    assert counts == {
        "n_unclassified": 2,
        "n_missing_ids": 1,
        "n_duplicate_ids": 1,
        "n_extra_ids": 1,
        "n_bad_labels": 0,
        "n_kind_ignored": 0,
    }


def test_bad_enum_invalidates_and_reads_as_unknown_labels():
    raw = {"spans": [entry("S1", target="planet"), entry("S2", sev=7, uk="guess")], "rationale": ""}
    assert not tv1.validate(raw, SPANS)
    assert not tv1.validate({"spans": [entry("S1", sev=True)]}, SPANS)
    assert not tv1.validate({"nope": 1}, SPANS)
    labels, counts = tv1.parse(raw, SPANS)
    assert labels[0].type == "concept" and labels[0].target == "unknown"
    assert labels[1].severity is None and labels[1].unsupported_kind == "unknown"
    assert counts["n_bad_labels"] == 3
    all_unknown = tv1.unknown_labels(SPANS)
    assert [lab.type for lab in all_unknown] == ["unknown"] * 3
    assert [lab.unsupported_kind for lab in all_unknown] == [None, "unknown", None]


# ------------------------------------------------------------------ routing in the pipeline


def _span_answer(kinds_and_spans):
    def f(user):
        texts = readout_texts(user)
        return {
            "samples": [
                {"idx": j, "kind": kinds_and_spans[j][0],
                 "spans": [{"text": s, "type": ty, "why": "EARLIER-WHY"}
                           for s, ty in kinds_and_spans[j][1]],
                 "rationale": "EARLIER-RATIONALE"}
                for j in range(len(texts))
            ]
        }  # fmt: skip

    return f


def test_routing_on_class_not_kind(tmp_path):
    from conftest import toy_sites, write_jsonl

    iid, pos = toy_sites()[0]
    p = write_jsonl(
        tmp_path / "r.jsonl",
        [{"id": iid, "layer": 44, "pos": pos,
          "samples": ["Paris story by the teacher", "plain friendly generic", "a stray code"]}],
    )  # fmt: skip
    answer = _span_answer(
        [
            ("generic", [("Paris", "wrong")]),  # generic kind + wrong -> hallucinated -> classified
            ("generic", []),  # generic class -> never seen by the classifier
            ("specific", [("stray code", "off_topic")]),  # off_topic class: spans not classified
        ]
    )

    def verify(user):
        return {
            "samples": [
                {"idx": 0, "kind": "specific", "rationale": "",
                 "claims": [{"quote": "the teacher", "status": "unverifiable", "why": "V-WHY"}]},
                {"idx": 1, "kind": "specific", "rationale": "",
                 "claims": [{"quote": "plain friendly", "status": "unverifiable", "why": "x"}]},
                {"idx": 2, "kind": "specific", "rationale": "",
                 "claims": [{"quote": "a stray", "status": "unverifiable", "why": "x"},
                            {"quote": "stray code", "status": "disputed", "why": "x"}]},
            ]
        }  # fmt: skip

    fake = FakeLLM({"span": answer, "verify": verify})
    asyncio.run(
        wrun.run_arms_async(
            [("r", p)], tmp_path / "o", wrun.Models(), layers=[44], limit=None, rpm=1e9,
            concurrency=2, allow_missing=True, llm=fake,
        )
    )  # fmt: skip
    cl = [c[3] for c in fake.calls if c[0] == "classify"]
    assert len(cl) == 1
    assert span_lines(cl[0]) == [
        ("S1", 0, "contradicted", "Paris"),
        ("S2", 0, "unsupported", "the teacher"),
        ("S3", 2, "unsupported", "a stray"),
    ]  # readout 2's disputed quote overlaps its off_topic span: dropped and counted
    assert readout_texts(cl[0]) == ["Paris story by the teacher", "a stray code"]
    assert "EARLIER-WHY" not in cl[0] and "V-WHY" not in cl[0] and "EARLIER-RATIONALE" not in cl[0]
    rec = json.loads((wrun.arm_dir(tmp_path / "o", wrun.Models(), "r") / "cells.jsonl").read_text())
    assert rec["n_offtopic_overlap_quotes"] == 1


# ------------------------------------------------------------------ calibration


def test_dev_cases_are_valid_and_meet_the_coverage_requirements():
    cases = calibrate.load_cases("dev")
    spans = [(c, s) for c in cases for s in c["spans"]]
    gold = [s["gold"] for _c, s in spans]
    assert len(spans) >= 40
    types = Counter(g["type"] for g in gold)
    assert types["concept"] >= 12 and types["detail"] >= 12
    assert {g["target"] for g in gold} == set(tx.TARGETS)
    rules = Counter(c.get("rule") for c in cases if c.get("rule"))
    assert sum(rules.values()) >= 8
    unsup = [(s, s["gold"]) for _c, s in spans if s["grounding"] == "unsupported"]
    assert len(unsup) >= 10
    assert {g["unsupported_kind"] for _s, g in unsup} == set(tx.UNSUPPORTED_KINDS)
    assert {g["type"] for _s, g in unsup} == {"concept", "detail"}
    sev = Counter(g["severity"] for g in gold)
    assert all(sev[k] >= 5 for k in (1, 2, 3))
    assert sum(s["grounding"] == "disputed" for _c, s in spans) >= 4
    texts = {s["text"] for _c, s in spans}
    assert any("Picasso" in t for t in texts) and any("Steve" in t for t in texts)
    assert all(c.get("part", "synthetic") == "synthetic" for c in cases)


def test_case_validation_catches_bad_spans():
    case = {
        "case_id": "x",
        "prompt": "p",
        "response": "no marker",
        "readouts": ["abc"],
        "spans": [
            {"readout_idx": 0, "text": "zzz", "grounding": "contradicted",
             "gold": {"type": "concept", "target": "entity", "severity": 2,
                      "unsupported_kind": "inference"}},
        ],
    }  # fmt: skip
    errs = calibrate.validate_case(case)
    assert any("⟦READ⟧" in e for e in errs)
    assert any("verbatim" in e for e in errs)
    assert any("must be null" in e for e in errs)


def test_two_repeats_make_two_separate_calls(tmp_path):
    case = {
        "case_id": "c1",
        "prompt": "Who painted the Mona Lisa?",
        "response": "Leonardo da Vinci⟦READ⟧ painted it.",
        "readouts": ["Picasso painted it."],
        "spans": [
            {"readout_idx": 0, "text": "Picasso", "grounding": "contradicted",
             "gold": {"type": "concept", "target": "entity", "severity": 3,
                      "unsupported_kind": None}},
        ],
    }  # fmt: skip
    fake = FakeLLM()
    with Cache(tmp_path / "c.jsonl") as cache:
        labels = asyncio.run(
            calibrate.classify_cases(
                [case], split="dev", model="m", effort="low", cache=cache, llm=fake, repeats=2
            )
        )
        assert fake.n("classify") == 2
        assert {cache.get(calibrate.cal_key("dev", "c1", r), "x") is None for r in (0, 1)} == {True}
        fake2 = FakeLLM()
        asyncio.run(
            calibrate.classify_cases(
                [case], split="dev", model="m", effort="low", cache=cache, llm=fake2, repeats=2
            )
        )
        assert fake2.n("classify") == 0  # each repeat cached under its own key
    res = calibrate.score([case], labels, "dev")
    assert res["mean"]["type_accuracy"] == 1.0 and res["mean"]["type_flip_rate"] == 0.0
    assert "disagreements" in res
    held = calibrate.score([case], labels, "heldout")
    assert "disagreements" not in held


def test_qwk_and_wilson():
    assert calibrate.qwk([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert calibrate.qwk([1, 2, 3, 1], [3, 2, 1, 3]) < 0
    lo, hi = calibrate.wilson(0.9, 50)
    assert lo < 0.9 < hi
