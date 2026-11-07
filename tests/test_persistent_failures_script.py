"""Plan 0029 Amendment 4: `scripts/v8/persistent_failures.py` -- which failed cells the list
names (blocking stage, reasons, attempts, fingerprints) on synthetic caches."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_index_v8 import _w

from hallucination_bench import final
from hallucination_bench import index as ix

_spec = importlib.util.spec_from_file_location(
    "persistent_failures",
    Path(__file__).resolve().parents[1] / "scripts" / "v8" / "persistent_failures.py",
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


@pytest.fixture(autouse=True)
def _two_layer_grid(monkeypatch):
    monkeypatch.setattr(final, "EXPECTED_LAYERS", (20, 24))


def _row(key, *, fp="f1", ts=1.0, ok=False, refusals=0, errors=0, calls=1, invalid=0, **meta):
    m = {"stage": "x", "model": "claude-opus-5-5", **meta}
    m["usage"] = {"calls": calls, "refusals": refusals, "errors": errors}
    if invalid:
        m["n_invalid_attempts"] = invalid
    return {"key": key, "fingerprint": fp, "ts": ts, "result": {"a": 1} if ok else None, "meta": m}


def _refused(key, n=3, **kw):
    return [_row(key, ts=float(i), refusals=2, **kw) for i in range(n)]


def test_reason_classification():
    assert pf.reason(_row("k", refusals=1), None) == "refused"
    assert pf.reason(_row("k", invalid=3), None) == "invalid"
    assert pf.reason({"meta": {"reason": "malformed"}}, None) == "invalid"
    assert pf.reason(_row("k", errors=1, calls=0), None) == "error"
    assert pf.reason(_row("k"), None) == "unknown"
    # an AM batched detection row: the batch row's usage decides
    det = {"key": "c:D", "meta": {"batch": "batchD:1", "answered_by": "claude-sonnet-5-5"}}
    assert pf.reason(det, _row("batchD:1", refusals=2)) == "refused"


def test_failed_call_uses_the_newest_fingerprint():
    rows = {"k": [_row("k", fp="old", ts=1, ok=True), *_refused("k", 2, fp="new")]}
    for r in rows["k"][1:]:
        r["ts"] += 10
    c = pf.failed_call("k", rows)
    assert c["attempts"] == 2 and c["reason"] == "refused"  # the old ok answer does not settle
    rows["k"].append(_row("k", fp="new", ts=99, ok=True))
    assert pf.failed_call("k", rows) is None
    assert pf.failed_call("absent", rows) is None


def _arm(tmp: Path, fam_dir: str, cells: list[dict], cache: list[dict], file=ix.CELLS) -> Path:
    d = tmp / "roots" / fam_dir / "jlens"
    _w(d / file, cells)
    _w(d / "cache.jsonl", cache)
    return d


def _entry(tmp, fam, unit, fam_dir, file) -> dict:
    return final.entry(
        fam,
        "jlens",
        unit,
        Path("roots") / fam_dir,
        f"jlens/{file}",
        base=tmp,
        arm="jlens",
        prompt_version="claimmap-v2.9",
        judge_model="m",
    )


def _build(tmp, entries, or_joint=None) -> dict:
    m = tmp / "m.json"
    m.write_text(json.dumps({"index_version": "hindex-v8", "entries": entries}))
    return pf.build(m, tmp, or_joint)


def test_main_cells_by_blocking_stage(tmp_path):
    ok = {"map": "ok", "cverify": "ok"}
    cells = [
        {"key": "a__L020__p1", "layer": 20, "stages": {"map": "failed"}},  # map refused x3
        {"key": "b__L020__p1", "layer": 20, "stages": {"map": "failed"}},  # map refused x2
        {"key": "c__L020__p1", "layer": 20, "stages": {"map": "failed"}},  # transport error
        {"key": "d__L020__p1", "layer": 20, "stages": {"map": "ok", "cverify": "failed"}},
        {"key": "e__L020__p1", "layer": 20, "stages": {"map": "failed"}},  # never asked
        {"key": "f__L020__p1", "layer": 20, "stages": ok},
    ]
    cache = [
        *_refused("a__L020__p1:M"),
        *_refused("b__L020__p1:M", 2),
        *[_row("c__L020__p1:M", ts=float(i), errors=1, calls=0) for i in range(4)],
        _row("d__L020__p1:M", ok=True),
        _row("d__L020__p1:CV:0", ok=True),
        *_refused("d__L020__p1:CV:3", answered_by="claude-sonnet-5-5"),
        *_refused("e__L020__p1:CV:0"),  # a refused call of another stage never explains a map
    ]
    _arm(tmp_path, "main", cells, cache)
    doc = _build(tmp_path, [_entry(tmp_path, ix.MAIN, "L20", "main", ix.CELLS)])
    got = {c["cell"]: (c["stage"], c["reason"], c["attempts"]) for c in doc["cells"]}
    assert got == {"a__L020__p1": ("map", "refused", 3), "d__L020__p1": ("cverify", "refused", 3)}
    why = {c["cell"]: c["why"] for c in doc["not_listed"]}
    assert set(why) == {"b__L020__p1", "c__L020__p1", "e__L020__p1"}
    assert "transient or under-retried" in why["b__L020__p1"]
    assert "transient or under-retried" in why["c__L020__p1"]
    assert "never asked" in why["e__L020__p1"]
    sha = final.sha256_file(tmp_path / "roots" / "main" / "jlens" / ix.CELLS)
    assert {c["file_sha256"] for c in doc["cells"]} == {sha}
    d = next(c for c in doc["cells"] if c["cell"] == "d__L020__p1")
    assert [x["key"] for x in d["calls"]] == ["d__L020__p1:CV:3"]
    assert d["calls"][0]["answered_by"] == "claude-sonnet-5-5"


def test_am_joint_site_blocked_at_its_own_map_not_at_a_detection(tmp_path):
    cells = [
        # pending joint map (never asked), one per-layer detection refused: NOT listed
        {"key": "s__Lall__p5", "id": "s", "layer": "all", "status": "ok", "map_status": "pending"},
        # a joint site still waiting on a refused per-layer detection: listed at detect
        {"key": "t__Lall__p5", "id": "t", "layer": "all", "status": "pending"},
        # Sonnet answers R0 only: invalid x4
        {"key": "u__Lall__p5", "id": "u", "layer": "all", "status": "ok", "map_status": "failed"},
    ]
    cache = [
        *_refused("s__L024__p5:D"),
        *_refused("t__L020__p5:Dsum", 6),  # detection: 2 rows per pass -> needs 6
        _row("t__L024__p5:D", ok=True),
        *[_row("u__Lall__p5:M", ts=float(i), invalid=3, calls=3) for i in range(4)],
    ]
    _arm(tmp_path, "am", cells, cache, ix.CELLS_JOINT)
    doc = _build(tmp_path, [_entry(tmp_path, ix.AM_CLAIMS, "all", "am", ix.CELLS_JOINT)])
    got = {c["cell"]: (c["stage"], c["reason"], c["attempts"]) for c in doc["cells"]}
    assert got == {"t__Lall__p5": ("detect", "refused", 6), "u__Lall__p5": ("map", "invalid", 4)}
    (nl,) = doc["not_listed"]
    assert nl["cell"] == "s__Lall__p5" and "never asked" in nl["why"]


def test_overrefusal_first_unsettled_stage(tmp_path):
    root = tmp_path / "roots" / "or" / "overrefusal" / "m"
    rows = [
        {"arm": "jlens", "layer": 20, "item": i, "cluster": i, "n_failed_cells": int(i != "ok")}
        for i in ("dg", "sj", "never", "ok")
    ]
    _w(root / "offtopic_index" / ix.SAFETY_COUNTS, rows)
    _w(
        root / "digest" / "jlens.jsonl",
        [
            *_refused("dg:jlens:dg:L20"),
            _row("dg:jlens:sj:L20", ok=True),
            _row("dg:jlens:never:L20", ok=True),
        ],
    )
    _w(root / "sjudge" / "jlens.jsonl", _refused("sj:jlens:sj:L20", 4))
    e = final.entry(
        ix.OVERREFUSAL,
        "jlens",
        "L20",
        Path("roots/or/overrefusal/m/offtopic_index"),
        ix.SAFETY_COUNTS,
        base=tmp_path,
        arm="jlens",
        prompt_version="n/a",
        judge_model="m",
    )
    doc = _build(tmp_path, [e])
    got = {c["cell"]: (c["stage"], c["attempts"]) for c in doc["cells"]}
    assert got == {"dg": ("digest", 3), "sj": ("sjudge", 4)}
    (nl,) = doc["not_listed"]
    assert nl["cell"] == "never" and nl["stage"] == "sjudge" and "never asked" in nl["why"]


def test_the_written_list_passes_final(tmp_path):
    """The script's output is a list `final.load_persistent` accepts."""
    cells = [{"key": "a__L020__p1", "layer": 20, "stages": {"map": "failed"}}]
    _arm(tmp_path, "main", cells, _refused("a__L020__p1:M"))
    e = _entry(tmp_path, ix.MAIN, "L20", "main", ix.CELLS)
    doc = _build(tmp_path, [e])
    p = tmp_path / "pf.json"
    p.write_text(json.dumps(doc))
    m = {"entries": [e], "persistent_failures": {"path": "pf.json", "sha256": final.sha256_file(p)}}
    got = final.load_persistent(m, tmp_path)
    assert got == {(ix.MAIN, "jlens", "L20"): {"a__L020__p1": ("refused", e["sha256"])}}


def test_a_detection_needs_twice_the_attempts(tmp_path):
    cells = [{"key": "v__L020__p5", "id": "v", "layer": 20, "status": "failed"}]
    _arm(tmp_path, "am", cells, _refused("v__L020__p5:D", 5))
    doc = _build(tmp_path, [_entry(tmp_path, ix.AM_CLAIMS, "L20", "am", ix.CELLS)])
    assert not doc["cells"] and "under-retried" in doc["not_listed"][0]["why"]
