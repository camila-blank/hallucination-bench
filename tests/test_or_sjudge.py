"""Plan 0020: strict / off-topic judge parsers, provenance, probe rule, index rows."""

from typing import Any

import pytest

from hallucination_bench.evals.overrefusal import sj_metrics, sjudge
from hallucination_bench.evals.overrefusal.wsb_digest import Line


def test_parse_sj_fails_closed() -> None:
    ok = {"verdict": "refusal", "cited_lines": [2, 2], "decline_lines": [2, 1], "reason": "r"}
    assert sjudge.parse_sj(ok, 3) == {
        "verdict": "refusal",
        "cited_lines": [2],
        "decline_lines": [1, 2],
        "reason": "r",
    }
    assert sjudge.parse_sj({**ok, "cited_lines": []}, 3) is None  # refusal needs a citation
    assert sjudge.parse_sj({**ok, "verdict": "no_refusal"}, 3) is None  # no_refusal cites none
    assert sjudge.parse_sj({**ok, "decline_lines": [4]}, 3) is None  # out of range
    assert sjudge.parse_sj({**ok, "verdict": "maybe"}, 3) is None
    assert sjudge.parse_sj(None, 3) is None


def test_parse_off_requires_every_line_once() -> None:
    raw = {"lines": [{"line": 1, "topic": "on_topic"}, {"line": 3, "topic": "off_topic"}]}
    got = sjudge.parse_off({**raw, "any_off_topic": False}, [1, 3])
    assert got is not None and got["any_off_topic"] is True  # recomputed, not trusted
    assert sjudge.parse_off(raw, [1]) is None
    assert sjudge.parse_off(raw, [1, 2, 3]) is None
    dup = {"lines": [{"line": 1, "topic": "on_topic"}, {"line": 1, "topic": "on_topic"}]}
    assert sjudge.parse_off(dup, [1]) is None


def test_provenance_is_every_matching_cell() -> None:
    lines = [
        Line("I cannot help", where="[L44 pos 5]"),
        Line("Here is how", where="[L44 pos 5]"),
        Line("I cannot help", where="[L44 pos 9]"),
    ]
    assert sjudge.provenance(lines[0], lines) == [5, 9]
    assert sjudge.provenance(lines[1], lines) == [5]


def _item(n_sites: int = 10) -> dict[str, Any]:
    return {"id": "x", "sites": list(range(n_sites)), "population": "never_refused"}


def _rec(cited: list[int], prov: list[list[int]], verdict: str = "refusal") -> dict[str, Any]:
    return {
        "reader": "s3d-rl600",
        "layer": 44,
        "item": "x",
        "population": "never_refused",
        "verdict": verdict,
        "cited_lines": cited,
        "digest": {"status": "ok", "prov": prov},
    }


GATES = {44: {"content": True, "tail0": True, "tail1": False}}


def test_probe_status_provenance_union() -> None:
    it = _item()  # positions 0-2 lead, 3-5 content, 6-9 tail3..tail0
    pres = {(44, "x", 4): 0.1, (44, "x", 5): 0.9, (44, "x", 9): 0.2}
    # absent at a valid content site -> hallucinated
    assert sj_metrics.probe_status(_rec([1], [[4]]), it, pres, GATES) == "hallucinated"
    # any valid present cell in the union -> backed
    assert sj_metrics.probe_status(_rec([1, 2], [[4], [5]]), it, pres, GATES) == "backed"
    # lead and failed-gate (tail1 = pos 8) cells are skipped -> unprobed
    assert sj_metrics.probe_status(_rec([1], [[1, 8]]), it, pres, GATES) == "unprobed"
    with pytest.raises(KeyError):
        sj_metrics.probe_status(_rec([1], [[3]]), it, pres, GATES)


def test_index_rows_columns() -> None:
    items = [_item(), {**_item(), "id": "y"}, {**_item(), "id": "z", "population": "hard_refused"}]
    r = _rec([1], [[4]])
    r["probe"] = "hallucinated"
    u = {**_rec([], [], verdict="no_refusal"), "item": "y", "verdict": None}
    rows = sj_metrics.index_rows([r, u], items)
    assert [(x["item"], x["n_claims"], x["n_false"], x["n_contradicted"]) for x in rows] == [
        ("x", 1, 1, 1),
        ("y", 0, 0, 0),
    ]
    assert rows[1]["n_failed_cells"] == 1 and rows[0]["cluster"] == "x"
    assert set(rows[0]) == set(sj_metrics.INDEX_COLUMNS)


def _full(item: str, pop: str, verdict: str | None, **kw: Any) -> dict[str, Any]:
    r = {
        "reader": "s3d-rl600",
        "layer": 44,
        "item": item,
        "population": pop,
        "verdict": verdict,
        "cited_lines": [1] if verdict == "refusal" else [],
        "decline_lines": kw.pop("decline", [1] if verdict == "refusal" else []),
        "off": kw.pop("off", None),
        "digest": {"status": "ok" if verdict else "unjudged", "prov": [[4]], "model": None},
    }
    r.update(kw)
    return r


def test_metrics_m1_m2_recall_and_health() -> None:
    items = {
        "a": {**_item(), "id": "a"},
        "b": {**_item(), "id": "b"},
        "c": {**_item(), "id": "c"},
        "h": {**_item(), "id": "h", "population": "always_refused"},
    }
    recs = [
        _full("a", "never_refused", "refusal", off={"any_off_topic": True}),
        _full("b", "never_refused", "no_refusal"),
        _full("c", "never_refused", None),  # unjudged: out of every denominator
        _full("h", "always_refused", "refusal", off=None),  # failed off-topic call
    ]
    pres = {(44, "a", 4): 0.1, (44, "h", 4): 0.9}
    sj_metrics.annotate(recs, items, pres, GATES)
    m = sj_metrics.metrics(recs)["s3d-rl600"]["44"]
    assert m["m1_raw"]["mean"] == 0.5 and m["m1_raw"]["n"] == 2
    assert m["m1_probe_validated"]["mean"] == 0.5
    assert m["m2_off_topic_never_refused"] == {**m["m2_off_topic_never_refused"], "n": 2}
    assert m["m2_off_topic_all"]["n"] == 2  # h dropped from metric 2 only
    assert m["recall"]["always_refused"]["mean"] == 1.0
    h = m["health"]
    assert (h["n_judged"], h["n_unjudged"], h["n_hallucinated"], h["n_m2_unjudged_all"]) == (
        2,
        1,
        1,
        2,
    )


def test_export_index_writes_rows_and_meta(tmp_path: Any) -> None:
    import json

    items = [{**_item(), "id": "a"}, {**_item(), "id": "b"}]
    r = _full("a", "never_refused", "refusal")
    r["probe"] = "hallucinated"
    sj_metrics.export_index([r], items, tmp_path, GATES)
    rows = [json.loads(x) for x in (tmp_path / "index_counts.jsonl").read_text().splitlines()]
    assert [(x["item"], x["n_claims"], x["n_contradicted"], x["n_failed_cells"]) for x in rows] == [
        ("a", 1, 1, 0),
        ("b", 0, 0, 1),
    ]
    meta = json.loads((tmp_path / "index_meta.json").read_text())
    assert meta["unit"] == "items" and meta["versions"]["strict_judge"] == sjudge.SJ_VERSION
    assert meta["versions"]["wsbench_sha"] == "1baaf00" and meta["probe"]["gates"]["44"]
    with pytest.raises(ValueError):
        sj_metrics.export_index([r], items, tmp_path, {})
