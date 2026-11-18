"""Plan 0022: the overrefusal offtopic export (the ``refusal_claims`` root hindex-v8 reads) and
the index's absent rule on it. The hindex-v6 / v5 output tests were removed with those versions
(plan 0030)."""

import json
from typing import Any

import pytest
from index_helpers import _or_row, _v6_root

from hallucination_bench import index as ix
from hallucination_bench.evals.overrefusal import offtopic_export as ox
from hallucination_bench.evals.overrefusal import run as or_run

# ---------------------------------------------------------------- export


def _rec(
    reader: str,
    item: str,
    layer: int,
    dl: list[int] | None = None,
    topics: dict[int, str] | None = None,
    *,
    status: str = "ok",
    verdict: str | None = "no_refusal",
    off_failed: bool = False,
    pop: str = "never_refused",
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "reader": reader,
        "item": item,
        "layer": layer,
        "population": pop,
        "digest": {"status": status},
        "verdict": verdict,
        "decline_lines": list(dl or []),
    }
    if dl:  # an off-topic call exists only with decline lines (sjudge.py)
        if off_failed:
            r["off"] = None
        else:
            t = {str(k): v for k, v in sorted((topics or {}).items())}
            r["off"] = {"topics": t, "any_off_topic": "off_topic" in t.values(), "reason": ""}
    return r


ITEMS = [
    {"id": "a", "population": "never_refused"},
    {"id": "b", "population": "hard_refused"},
    {"id": "c", "population": "always_refused"},
    {"id": "d", "population": "never_refused"},
]


def _recs(layer: int = 20, reader: str = "jlens") -> list[dict[str, Any]]:
    return [
        # 3 decline lines, 1 off-topic (int lines, string topic keys)
        _rec(reader, "a", layer, [2, 5, 9], {2: "off_topic", 5: "on_topic", 9: "on_topic"}),
        # a refused item: 2 decline lines, both off-topic
        _rec(reader, "b", layer, [1, 3], {1: "off_topic", 3: "off_topic"}, verdict="refusal"),
        # judged, no decline lines: a zero row (no "off" key at all)
        _rec(reader, "c", layer),
        # off-topic call failed, despite decline lines: a failed row with all counts 0
        _rec(reader, "d", layer, [4], off_failed=True, verdict="refusal"),
    ]


def test_record_counts_failure_rules():
    assert ox.record_counts(_rec("r", "x", 20, status="unjudged", verdict=None)) == (
        0,
        0,
        "unjudged",
    )
    assert ox.record_counts(_rec("r", "x", 20, verdict=None)) == (0, 0, "strict_judge_failed")
    assert ox.record_counts(_rec("r", "x", 20, [3], off_failed=True)) == (0, 0, "off_topic_failed")
    assert ox.record_counts(_rec("r", "x", 20)) == (0, 0, None)  # judged zero row
    ok = _rec("r", "x", 20, [3, 7], {3: "off_topic", 7: "on_topic"})
    assert ox.record_counts(ok) == (2, 1, None)


def test_key_mismatch_is_an_error():
    bad = _rec("r", "x", 20, [3, 7], {3: "off_topic", 8: "on_topic"})
    with pytest.raises(ValueError, match=r"off\.topics keys"):
        ox.record_counts(bad)
    # int keys in topics would also mismatch the JSON-string convention
    bad2 = _rec("r", "x", 20, [3], {3: "on_topic"})
    bad2["off"]["topics"] = {3: "on_topic"}
    with pytest.raises(ValueError, match=r"off\.topics keys"):
        ox.record_counts(bad2)


def test_index_rows_zero_failed_populations():
    rows, fails = ox.index_rows(_recs(), ITEMS, readers=["jlens"])
    by = {r["item"]: r for r in rows}
    assert [r["item"] for r in rows] == ["a", "b", "c", "d"]
    assert set(rows[0]) == set(ox.COLUMNS)
    assert (by["a"]["n_claims"], by["a"]["n_false"], by["a"]["n_contradicted"]) == (3, 1, 1)
    assert (by["b"]["n_claims"], by["b"]["n_contradicted"]) == (2, 2)
    assert (by["c"]["n_claims"], by["c"]["n_failed_cells"]) == (0, 0)
    assert (by["d"]["n_claims"], by["d"]["n_contradicted"], by["d"]["n_failed_cells"]) == (0, 0, 1)
    assert {r["population"] for r in rows} == {"never_refused", "hard_refused", "always_refused"}
    assert all(r["n_unprobed"] == r["n_unresolved"] == 0 for r in rows)
    assert all(r["cluster"] == r["item"] for r in rows)
    assert fails == {"unjudged": 0, "strict_judge_failed": 0, "off_topic_failed": 1}
    # the index's share: (1 + 2) / (3 + 2)
    r, _ = ix.safety_metric(rows)
    assert r is not None and r["v"] == 3 / 5 and r["n"] == 5


def test_index_rows_errors_and_reader_allowlist():
    recs = _recs()
    with pytest.raises(ValueError, match="no record"):
        ox.index_rows(recs[:3], ITEMS, readers=["jlens"])
    with pytest.raises(ValueError, match="repeated record"):
        ox.index_rows([*recs, recs[0]], ITEMS, readers=["jlens"])
    with pytest.raises(ValueError, match="unknown item"):
        ox.index_rows([*recs, _rec("jlens", "zz", 20)], ITEMS, readers=["jlens"])
    rows, _ = ox.index_rows([*recs, *_recs(reader="other")], ITEMS, readers=["jlens"])
    assert {r["arm"] for r in rows} == {"jlens"}


def test_summary_secondary_numbers():
    rows, _ = ox.index_rows([*_recs(20), *_recs(24)], ITEMS, readers=["jlens"])
    claims = [
        {"arm": "jlens", "layer": 20, "item": "a", "n_claims": 30, "n_failed_cells": 0},
        {"arm": "jlens", "layer": 20, "item": "d", "n_claims": 10, "n_failed_cells": 0},
    ]
    items = [
        {
            "arm": "jlens",
            "layer": 20,
            "item": i,
            "cluster": i,
            **dict.fromkeys(ix.SAFETY_INT_COLS, 0),
        }
        | {"n_claims": 1, "n_contradicted": int(i == "a")}
        for i in ("a", "d")
    ]
    m = ox.summary(rows, claims, items)
    c20 = m["jlens"]["20"]
    assert c20["index"]["v"] == 3 / 5 and c20["index"]["n_claims"] == 5
    assert c20["index"]["n_failed"] == 1 and c20["index"]["n_items"] == 4
    assert c20["never_refused"]["v"] == 1 / 3 and c20["never_refused"]["n_items"] == 2
    # d's off-topic call failed: d is out of both sides of (ii)
    assert c20["per_disposition_claim"] == {
        "n_items": 1,
        "n_off": 1,
        "n_disposition_claims": 30,
        "v": 1 / 30,
    }
    assert c20["item_metric"]["v"] == 0.5
    assert m["jlens"]["24"]["item_metric"]["v"] is None  # no item rows at L24: absent


def test_export_writes_separate_root(tmp_path, monkeypatch):
    root = or_run.eval_root(tmp_path, "m")
    root.mkdir(parents=True)
    (root / ox.RECORDS).write_text("".join(json.dumps(r) + "\n" for r in _recs()))
    meta0 = {"unit": "items", "versions": {"digest": "digest-v3", "off_topic_judge": "oroff-v1"}}
    (root / ox.ITEM_META).write_text(json.dumps(meta0))
    (root / ox.ITEM_COUNTS).write_text("")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    monkeypatch.setattr(ox.data, "load_items", lambda: ITEMS)
    monkeypatch.setattr(ox.index_export, "READERS", ("jlens",))
    res = ox.export(tmp_path, "m")
    assert res["off_topic_failed"] == 1 and res["n_decline_lines"] == 5
    dest = root / ox.OFFTOPIC_DIR
    meta = json.loads((dest / ox.ITEM_META).read_text())
    assert meta["unit"] == ix.UNIT_REFUSAL_CLAIMS == ox.UNIT
    assert meta["population"] == "all" and meta["gates"] == {} and meta["n_items"] == 4
    assert meta["versions"] == meta0["versions"]
    rows, ignored = ix.safety_rows(dest, ix.OVERREFUSAL)
    assert not ignored and len(rows["jlens"][20]) == 4
    for name, b in before.items():  # the plan-0020 files are untouched
        assert (root / name).read_bytes() == b
    doc = json.loads((root / ox.SUMMARY).read_text())
    assert doc["metrics"]["jlens"]["20"]["index"]["v"] == 3 / 5
    assert "per_disposition_claim" not in doc["metrics"]["jlens"]["20"]  # no claims rows file
    (root / ox.ITEM_META).write_text(json.dumps({"unit": "refusal_claims"}))
    with pytest.raises(ValueError, match="unit items"):
        ox.export(tmp_path, "m")


# ---------------------------------------------------------------- the index


def test_v6_absent_at_zero_decline_lines(tmp_path):
    root = _v6_root(tmp_path / "rc", [_or_row(20, "o0", 0, 0), _or_row(20, "o1", 0, 0, 1)])
    built = ix.build_index({ix.OVERREFUSAL: [root]}, ix.Methods(), version="v8")
    assert built["rows"] == []  # absent, not 0: no family value, so no row
    (gone,) = built["safety"][ix.OVERREFUSAL]["absent"]
    assert (gone["layer"], gone["absent_reason"], gone["n_failed_cells"]) == (20, "no_claims", 1)
