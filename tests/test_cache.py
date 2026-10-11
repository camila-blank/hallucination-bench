"""Cache + resume: append-only rows, fingerprint mismatch re-queries, partial / invalid answers
are re-queued, stale classify rows are never attached, rebuild makes no calls."""

import asyncio
import json
from pathlib import Path

import pytest
from conftest import FakeLLM, default_span, readout_texts

from hallucination_bench.cache import Cache, fingerprint
from hallucination_bench.evals.wsbench_chat import run as wrun


def _run(arms, out: Path, fake: FakeLLM, models=None, **kw):
    models = models or wrun.Models()
    return asyncio.run(
        wrun.run_arms_async(
            arms,
            out,
            models,
            layers=kw.pop("layers", [44]),
            limit=kw.pop("limit", None),
            rpm=1e9,
            concurrency=4,
            allow_missing=True,
            llm=fake,
            **kw,
        )
    )


def _cells(out: Path, arm: str, models=None) -> list[dict]:
    d = wrun.arm_dir(out, models or wrun.Models(), arm)
    return [json.loads(x) for x in (d / "cells.jsonl").read_text().splitlines()]


def test_cache_rows_fingerprints_and_torn_lines(tmp_path):
    p = tmp_path / "c.jsonl"
    with Cache(p) as c:
        c.put("k", "fp1", {"result": {"a": 1}})
        c.put("k", "fp2", {"result": None, "meta": {"raw": {"bad": 1}}})
        assert c.get("k", "fp1") == c.get("k", "fp1", include_failed=True)
        assert c.get("k", "fp2") is None  # failed rows are re-queued
        assert c.get("k", "fp2", include_failed=True)["meta"]["raw"] == {"bad": 1}
        assert c.get("k", "other") is None
        with pytest.raises(ValueError):
            c.put("k", "fp", {"fingerprint": "x"})
    with p.open("a") as fh:
        fh.write('{"key": "torn"')  # interrupted write
    with Cache(p) as c:
        assert len(c) == 2 and c.get("k", "fp1")["result"] == {"a": 1}
        c.put("k2", "f", {"result": {}})
    assert len(Cache(p, readonly=True)) == 3
    assert fingerprint("a", 1) != fingerprint("a", 2)


def test_end_to_end_resume_makes_no_new_calls(tmp_path, prose_readouts):
    fake = FakeLLM()
    out = tmp_path / "out"
    s = _run([("toy", prose_readouts)], out, fake)["toy"]
    n_cells = 15  # 16 toy sites, the last is empty
    assert fake.n("span") == n_cells and fake.n("verify") == n_cells
    assert fake.n("classify") == 1  # only site 0 has spans to classify
    assert s["health"]["n_empty_cells"] == 1 and s["health"]["n_judged_cells"] == n_cells
    rec = _cells(out, "toy")[0]
    assert rec["status"] == "judged" and rec["stages"]["classify"] == "ok"
    assert [(x["grounding"], x["text"], x["type"]) for x in rec["spans"]] == [
        ("contradicted", "Paris", "concept"),
        ("unsupported", "the teacher", "detail"),
    ]
    fake2 = FakeLLM()
    _run([("toy", prose_readouts)], out, fake2)
    assert fake2.calls == []  # everything served from the cache
    assert _cells(out, "toy") == _cells(out, "toy")


def test_classifier_change_requeues_only_classify(tmp_path, prose_readouts):
    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, FakeLLM())
    fake = FakeLLM()
    _run([("toy", prose_readouts)], out, fake, models=wrun.Models(classifier="claude-other"))
    assert [c[0] for c in fake.calls] == ["classify"]


def test_partial_span_answer_is_failed_and_requeued(tmp_path, prose_readouts):
    def partial(user):
        r = default_span(user)
        r["samples"] = r["samples"][:1]  # judges only readout 0
        return r

    out = tmp_path / "out"
    fake = FakeLLM({"span": partial})
    s = _run([("toy", prose_readouts)], out, fake)["toy"]
    assert s["health"]["n_failed_span"] == 15  # every toy cell has >= 2 readouts
    assert fake.n("verify") == 0
    rows = [json.loads(x) for x in (wrun.arm_dir(out, wrun.Models(), "toy") / "cache.jsonl").open()]
    assert all(r["result"] is None and "raw" in r["meta"] for r in rows)
    fake2 = FakeLLM()
    s = _run([("toy", prose_readouts)], out, fake2)["toy"]
    assert fake2.n("span") == 15 and s["health"]["n_failed_span"] == 0


def test_invalid_classify_enum_is_requeued_and_read_leniently(tmp_path, prose_readouts):
    def bad(user):
        return {
            "spans": [
                {"span_id": "S1", "type": "concept", "target": "planet", "severity": 2,
                 "unsupported_kind": None, "why": "w"},
                {"span_id": "S2", "type": "detail", "target": "entity", "severity": 1,
                 "unsupported_kind": "fabricated", "why": "w"},
            ],
            "rationale": "r",
        }  # fmt: skip

    out = tmp_path / "out"
    s = _run([("toy", prose_readouts)], out, FakeLLM({"classify": bad}))["toy"]
    assert s["health"]["n_failed_classify"] == 1
    spans = _cells(out, "toy")[0]["spans"]
    assert spans[0]["type"] == "concept" and spans[0]["target"] == "unknown"  # bad label only
    assert spans[1]["unsupported_kind"] == "fabricated"
    fake = FakeLLM()
    _run([("toy", prose_readouts)], out, fake)
    assert [c[0] for c in fake.calls] == ["classify"]


def test_verify_failure_classifies_contradicted_only_then_reclassifies(tmp_path, prose_readouts):
    out = tmp_path / "out"
    fake = FakeLLM({"verify": lambda u: None})
    s = _run([("toy", prose_readouts)], out, fake)["toy"]
    assert s["health"]["n_failed_verify"] == 15
    rec = _cells(out, "toy")[0]
    assert [x["grounding"] for x in rec["spans"]] == ["contradicted"]
    assert rec["spans"][0]["type"] == "concept"  # classified in the same run
    # a later run: verify succeeds -> the span list changes -> the cell is re-classified
    fake2 = FakeLLM()
    _run([("toy", prose_readouts)], out, fake2)
    assert fake2.n("verify") == 15 and fake2.n("classify") == 1
    rec = _cells(out, "toy")[0]
    assert [x["grounding"] for x in rec["spans"]] == ["contradicted", "unsupported"]
    assert all(x["type"] != "unknown" for x in rec["spans"])


def test_stale_classify_row_is_never_attached(tmp_path, prose_readouts):
    """Contradicted-only classify row cached, verify row later succeeds but classify is skipped:
    rebuild must not attach the stale labels to the new S-ids."""
    out = tmp_path / "out"
    _run([("toy", prose_readouts)], out, FakeLLM({"verify": lambda u: None}))
    _run([("toy", prose_readouts)], out, FakeLLM(), classify=False)
    s = wrun.rebuild_arm("toy", prose_readouts, out, wrun.Models())
    rec = _cells(out, "toy")[0]
    assert [x["grounding"] for x in rec["spans"]] == ["contradicted", "unsupported"]
    assert all(x["type"] == "unknown" for x in rec["spans"])
    assert s["health"]["n_unclassified"] == 2


def test_rebuild_makes_no_calls_and_covers_every_judged_layer(tmp_path, prose_readouts):
    rows = [json.loads(x) for x in prose_readouts.read_text().splitlines()]
    for r in list(rows):
        rows.append({**r, "layer": 60})
    p = tmp_path / "two_layers.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    _run([("toy", p)], out, FakeLLM(), layers=[44])
    _run([("toy", p)], out, FakeLLM(), layers=[60], limit=3)  # stage 2 extends stage 1
    s = wrun.rebuild_arm("toy", p, out, wrun.Models())
    assert s["config"]["judged_layers"] == [44, 60]
    assert set(s["by_layer"]) == {"44", "60"}
    assert s["by_layer"]["60"]["health"]["n_judged_cells"] == 3
    assert s["health"]["n_pending_cells"] == 12


def test_token_arm_is_summarised_then_judged_k1(tmp_path, token_readouts):
    out = tmp_path / "out"
    fake = FakeLLM()
    s = _run([("tok", token_readouts)], out, fake)["tok"]
    assert fake.n("summary") == 3 and fake.n("span") == 3
    users = [c[3] for c in fake.calls if c[0] == "summary"]
    assert "TOKEN READOUTS:\n Paris (2.00) |  teacher (1.00)" in users[0]
    spans = [c[3] for c in fake.calls if c[0] == "span"]
    assert len(readout_texts(spans[0])) == 1
    assert s["config"]["k"] == 1 and s["config"]["prompt_versions"]["summary"] == "interp-v1"


def test_failed_summary_leaves_the_cell_unjudged(tmp_path, token_readouts):
    out = tmp_path / "out"
    fake = FakeLLM({"summary": lambda u: {"interpretation": "  "}})
    s = _run([("tok", token_readouts)], out, fake)["tok"]
    assert fake.n("span") == 0 and s["health"]["n_failed_summary"] == 3
    assert s["overall"]["rates"]["hallucination_rate"] is None
