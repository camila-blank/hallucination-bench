"""Plan 0029 review fix 7: in v8 roots a refused AM J-lens interp-v1 summary is re-asked once
on Sonnet (the detector's rule): the row stays under the Opus fingerprint and records
``meta.answered_by``; v7 roots never fall back."""

import asyncio

from hallucination_bench import summarizer
from hallucination_bench.cache import Cache
from hallucination_bench.evals.agentic_misalignment import run as amrun
from hallucination_bench.evals.wsbench_chat.run import Models
from hallucination_bench.llm import CallResult, Usage
from hallucination_bench.readouts import Cell

M7 = Models(eval="agentic_misalignment")
M8 = Models(eval="agentic_misalignment", map_prompt="claimmap-v2.8")
CELL = Cell("am-x", 20, 5, None, ("hello", "world"), (0.9, 0.1))


class RefusingLLM:
    def __init__(self, refuse_on: set[str]):
        self.refuse_on, self.calls = refuse_on, []

    async def call(self, *, model, system, user, schema):
        assert system == summarizer.INTERP_SYSTEM
        self.calls.append(model)
        if model in self.refuse_on:
            return CallResult(None, Usage(calls=1, refusals=1))
        return CallResult({"interpretation": f"text by {model}"}, Usage(calls=1, usd=0.01))


def _call(tmp_path, models, llm):
    cache = Cache(tmp_path / "cache.jsonl")
    try:
        st, text = asyncio.run(amrun.summary_call(cache, llm, CELL, models, Usage()))
        rows = list(cache.rows())
    finally:
        cache.close()
    return st, text, rows


def test_v8_refused_summary_is_reasked_once_on_sonnet(tmp_path):
    llm = RefusingLLM({"claude-opus-5-5"})
    st, text, rows = _call(tmp_path, M8, llm)
    assert llm.calls == ["claude-opus-5-5", amrun.DETECT_FALLBACK_MODEL]
    assert st == "ok" and text == f"text by {amrun.DETECT_FALLBACK_MODEL}"
    (row,) = rows
    assert row["key"] == f"summ:{CELL.key}"
    assert row["meta"]["model"] == "claude-opus-5-5"
    assert row["meta"]["answered_by"] == amrun.DETECT_FALLBACK_MODEL
    assert row["meta"]["usage"]["refusals"] == 1 and row["meta"]["usage"]["calls"] == 2
    # cached under the Opus fingerprint: a cache-only read finds it
    cache = Cache(tmp_path / "cache.jsonl", readonly=True)
    try:
        st2, text2 = asyncio.run(amrun.summary_call(cache, None, CELL, M8, Usage()))
    finally:
        cache.close()
    assert (st2, text2) == ("ok", text)


def test_v8_both_refuse_is_one_failed_row(tmp_path):
    llm = RefusingLLM({"claude-opus-5-5", amrun.DETECT_FALLBACK_MODEL})
    st, text, rows = _call(tmp_path, M8, llm)
    assert len(llm.calls) == 2 and st == "failed" and text is None
    (row,) = rows
    assert row["result"] is None and row["meta"]["answered_by"] == amrun.DETECT_FALLBACK_MODEL


def test_v8_answered_summary_records_no_fallback(tmp_path):
    llm = RefusingLLM(set())
    st, _text, rows = _call(tmp_path, M8, llm)
    assert st == "ok" and llm.calls == ["claude-opus-5-5"]
    assert "answered_by" not in rows[0]["meta"]


def test_v7_roots_never_fall_back(tmp_path):
    llm = RefusingLLM({"claude-opus-5-5"})
    st, _text, rows = _call(tmp_path, M7, llm)
    assert llm.calls == ["claude-opus-5-5"] and st == "failed"
    assert "answered_by" not in rows[0]["meta"]
