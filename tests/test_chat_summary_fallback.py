"""Plan 0029 review: in v8 (claimmap-v2.8) chat / story roots a refused J-lens interp-v1
``summ:`` call is re-asked once on Sonnet, the AM driver's rule (``cached_call(fallback=)``):
the row stays under the Opus key and fingerprint with ``meta.answered_by`` and the summed
usage; v7 roots never fall back. Both refused = a failed row (re-asked on a later judge run,
as AM), a ``failed_summary`` cell that ``hbench final`` counts failed per layer and joint."""

import json

from conftest import TOY_ITEMS, write_jsonl
from test_claimmap_v28 import Map28Fake

from hallucination_bench import final
from hallucination_bench import index as ix
from hallucination_bench.evals.agentic_misalignment import run as amrun
from hallucination_bench.evals.wsbench_chat import bank
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.llm import CallResult, Usage

OPUS = "claude-opus-5-5"
SONNET = wrun.SUMMARY_FALLBACK_MODEL
M7 = wrun.Models()
M8 = wrun.Models(map_prompt="claimmap-v2.8")
BAD = "ĠZurich"  # the token that makes a summary call refusable


class RefusingFake(Map28Fake):
    """Refuses a summary call whose bag holds ``Zurich`` when asked on a model in ``refuse_on``."""

    def __init__(self, refuse_on: set[str]):
        super().__init__()
        self.refuse_on = refuse_on

    async def call(self, *, model, system, user, schema):
        if system == wrun.summarizer.INTERP_SYSTEM and "Zurich" in user and model in self.refuse_on:
            self.calls.append(("summary", model, system, user))
            return CallResult(None, Usage(calls=1, refusals=1))
        return await super().call(model=model, system=system, user=user, schema=schema)

    def summary_models(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "summary" and "Zurich" in c[3]]


def _tokens(tmp_path, layers=(44,), bad_layer=44):
    """Token readouts on the first item's first two sites; site 0 at ``bad_layer`` reads Zurich."""
    item = next(it for it in bank.load_items() if it["id"] == TOY_ITEMS[0])
    rows = []
    for la in layers:
        for n, s in enumerate(item["sites"][:2]):
            tok = BAD if (n == 0 and la == bad_layer) else "ĠParis"
            rows.append(
                {
                    "id": item["id"],
                    "layer": la,
                    "pos": int(s["pos"]),
                    "tokens": [tok, "Ġteacher"],
                    "scores": [2.0, 1.0],
                }
            )
    return write_jsonl(tmp_path / "t.jsonl", rows)


def _run(tmp_path, readouts, models, fake):
    return wrun.run_arms(
        [("jl", readouts)],
        tmp_path / "out",
        models,
        layers=None,
        limit=None,
        rpm=1e6,
        concurrency=2,
        allow_missing=True,
        llm=fake,
        stages=["map"],
    )


def _summ_rows(tmp_path, models):
    d = wrun.arm_dir(tmp_path / "out", models, "jl")
    rows = [json.loads(x) for x in (d / "cache.jsonl").open()]
    return [r for r in rows if r["key"].startswith("summ:")], d


def _bad_rows(rows):
    return [r for r in rows if r["meta"]["usage"].get("refusals")]


def _cells(d, name="cells.jsonl"):
    return [json.loads(x) for x in (d / name).read_text().splitlines()]


def test_fallback_model_is_the_am_drivers():
    assert SONNET == amrun.DETECT_FALLBACK_MODEL


def test_v8_refused_summary_is_answered_by_sonnet(tmp_path):
    fake = RefusingFake({OPUS})
    _run(tmp_path, _tokens(tmp_path), M8, fake)
    assert fake.summary_models() == [OPUS, SONNET]
    rows, d = _summ_rows(tmp_path, M8)
    (row,) = _bad_rows(rows)
    assert row["result"] == {"interpretation": "A friendly note about Paris and the teacher."}
    assert row["meta"]["model"] == OPUS and row["meta"]["answered_by"] == SONNET
    assert row["meta"]["usage"]["calls"] == 2 and row["meta"]["usage"]["refusals"] == 1
    assert row["meta"]["usage"]["usd"] == 0.01
    # the other summary was answered by Opus: no answered_by
    assert all("answered_by" not in r["meta"] for r in rows if r is not row)
    # kept under the Opus fingerprint: every cell is mapped, and a rerun asks nothing again
    assert all(c["stages"]["map"] == "ok" for c in _cells(d))
    n = len(fake.calls)
    _run(tmp_path, _tokens(tmp_path), M8, fake)
    assert len(fake.calls) == n


def test_v7_root_never_falls_back(tmp_path):
    fake = RefusingFake({OPUS})
    _run(tmp_path, _tokens(tmp_path), M7, fake)
    assert fake.summary_models() == [OPUS]
    rows, d = _summ_rows(tmp_path, M7)
    (row,) = _bad_rows(rows)
    assert row["result"] is None and "answered_by" not in row["meta"]
    assert row["meta"]["usage"]["calls"] == 1
    bad = [c for c in _cells(d) if c["status"] == "failed_summary"]
    assert len(bad) == 1 and bad[0]["stages"]["summary"] == "failed"


def test_v8_both_refuse_is_cached_failed_and_counted_failed_by_final(tmp_path):
    fake = RefusingFake({OPUS, SONNET})
    _run(tmp_path, _tokens(tmp_path), M8, fake)
    assert fake.summary_models() == [OPUS, SONNET]
    rows, d = _summ_rows(tmp_path, M8)
    (row,) = _bad_rows(rows)
    assert row["result"] is None and row["meta"]["model"] == OPUS
    assert row["meta"]["answered_by"] == SONNET and row["meta"]["usage"]["refusals"] == 2
    cells = _cells(d)
    bad = [c for c in cells if c["status"] == "failed_summary"]
    assert len(bad) == 1 and bad[0]["stages"]["summary"] == "failed"
    assert bad[0]["stages"]["map"] is None
    # hbench final: a failed (not empty, not dropped) cell of its layer's entry
    e = {"family": ix.MAIN, "method": "jlens", "unit": "L44"}
    n, f, _seen = final.recount(e, d / "cells.jsonl")
    assert (n, f) == (2, 1)
    # a cache-only rebuild keeps it failed (not pending)
    wrun.rebuild_arm("jl", tmp_path / "t.jsonl", tmp_path / "out", M8)
    assert [c["status"] for c in _cells(d)] == [c["status"] for c in cells]
    # a later judge run re-asks it (Opus, then Sonnet), as the AM driver does
    _run(tmp_path, _tokens(tmp_path), M8, fake)
    assert fake.summary_models() == [OPUS, SONNET, OPUS, SONNET]


def test_joint_fallback_and_both_refused_joint_cell(tmp_path):
    t = _tokens(tmp_path, layers=(20, 24), bad_layer=24)
    ok = RefusingFake({OPUS})
    wrun.run_joint("jl", [("jl", t)], tmp_path / "a", M8, layers=(20, 24), llm=ok)
    assert ok.summary_models() == [OPUS, SONNET]
    da = wrun.arm_dir(tmp_path / "a", M8, "jl")
    judged = [c for c in _cells(da, wrun.JOINT_CELLS) if c["status"] != "empty"]
    assert len(judged) == 2 and all(c["stages"]["map"] == "ok" for c in judged)

    no = RefusingFake({OPUS, SONNET})
    wrun.run_joint("jl", [("jl", t)], tmp_path / "b", M8, layers=(20, 24), llm=no)
    assert no.summary_models() == [OPUS, SONNET]
    db = wrun.arm_dir(tmp_path / "b", M8, "jl")
    cells = _cells(db, wrun.JOINT_CELLS)
    bad = [c for c in cells if c["status"] == "failed_summary"]
    assert len(bad) == 1 and bad[0]["stages"] == {"summary": "failed", "map": None}
    e = {"family": ix.MAIN, "method": "jlens", "unit": ix.JOINT_LABEL}
    n, f, _seen = final.recount(e, db / wrun.JOINT_CELLS)
    assert (n, f) == (2, 1)
