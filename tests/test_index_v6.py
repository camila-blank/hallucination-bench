"""Plan 0022: hindex-v6 (overrefusal on wrong-topic refusal claims), the offtopic export, the unit
guard, and the hindex-v5 dev-box golden."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_index_v5 import _canon, _golden_methods, _pinned_roots_v4, _sources_v5

from hallucination_bench import REPO_ROOT, cli
from hallucination_bench import index as ix
from hallucination_bench.evals.overrefusal import offtopic_export as ox
from hallucination_bench.evals.overrefusal import run as or_run

FIX = REPO_ROOT / "tests" / "fixtures"


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


# ---------------------------------------------------------------- index v6


def _v6_root(tmp: Path, rows: list[dict[str, Any]], unit: str | None = "refusal_claims") -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / ix.SAFETY_COUNTS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if unit is not None:
        (tmp / ix.SAFETY_META).write_text(json.dumps({"unit": unit, "gates": {}}))
    return tmp


def _or_row(layer: int, item: str, n: int, off: int, failed: int = 0) -> dict[str, Any]:
    return {
        "arm": "jlens",
        "layer": layer,
        "item": item,
        "cluster": item,
        **dict.fromkeys(ix.SAFETY_INT_COLS, 0),
        "n_claims": n,
        "n_false": off,
        "n_contradicted": off,
        "n_failed_cells": failed,
        "population": "never_refused",
    }


def test_v6_reads_refusal_claims_root_rest_identical_to_v5(tmp_path):
    src = _sources_v5(tmp_path / "src")
    # L20: 1 + 0 off of 4 + 4 decline lines; L24: no decline lines at all -> absent
    rows = [_or_row(20, "o0", 4, 1), _or_row(20, "o1", 4, 0), _or_row(20, "o2", 0, 0, failed=1)]
    rows += [_or_row(24, "o0", 0, 0), _or_row(24, "o1", 0, 0)]
    src6 = {**src, ix.OVERREFUSAL: [_v6_root(tmp_path / "or6", rows)]}
    methods = ix.Methods(notes={"k": "methods note"})
    v5 = ix.build_index(src, methods, version="v5")
    v6 = ix.build_index(src6, methods, version="v6")
    assert v6["version"] == "hindex-v6"
    assert v6["notes"] == {"k": "methods note", **ix.NOTES_V6}
    assert "2 of 921 J-lens decline lines" in ix.NOTES_V6["hindex-v6"]
    assert v6["safety"][ix.OVERREFUSAL]["unit"] == ["refusal_claims"]
    assert v6["safety"][ix.OVERREFUSAL]["share"] == ix.SHARE_V6
    assert v5["safety"][ix.OVERREFUSAL]["share"] == ix.SHARE_V4
    r5 = {r["layer"]: r for r in v5["rows"]}
    r6 = {r["layer"]: r for r in v6["rows"]}
    for L in (20, 24):
        for f in (ix.MAIN, ix.AM, ix.STORY, ix.SIBLING):  # bit-identical to v5 (AM on candidates)
            assert json.dumps(r6[L]["families"].get(f)) == json.dumps(r5[L]["families"].get(f))
    or6 = r6[20]["families"][ix.OVERREFUSAL]
    assert (or6["v"], or6["n"], or6["n_items"]) == (1 / 8, 8, 3)
    assert or6["health"]["n_failed_cells"] == 1
    assert r5[20]["families"][ix.OVERREFUSAL]["v"] == 0.25
    per = r6[20]["families"]
    assert json.dumps(r6[20]["matched"]) == json.dumps(ix.weighted_macro(per, ix.MATCHED))
    assert json.dumps(r6[20]["full_v3"]) == json.dumps(r5[20]["full_v3"])  # no overrefusal
    # L24: 0 decline lines -> absent (not 0); AM is also absent there (0 candidates, as in v5)
    assert ix.OVERREFUSAL not in r6[24]["families"]
    assert r6[24]["absent"][ix.OVERREFUSAL] == {"arm": "jlens", "absent_reason": "no_claims"}
    md = ix.render_md(v6)
    assert md.startswith("# Hallucination index (hindex-v6)")
    assert "**refusal claims**" in md and "with v6 family definitions" in md
    assert "jlens.overrefusal@24 (no_claims)" in md
    assert "- hindex-v6: " in md and "- hindex-v5: " not in md


def test_unit_guard_both_ways(tmp_path):
    rows = [_or_row(20, "o0", 1, 0)]
    claims = _v6_root(tmp_path / "rc", rows)
    items = _v6_root(tmp_path / "it", rows, unit="items")
    bare = _v6_root(tmp_path / "bare", rows, unit=None)
    for v in ("v4", "v5"):
        with pytest.raises(ValueError, match="refusal_claims"):
            ix.build_index({ix.OVERREFUSAL: [claims]}, ix.Methods(), version=v)
        for ok in (items, bare):  # items, or a meta-less (synthetic) root: accepted
            ix.build_index({ix.OVERREFUSAL: [ok]}, ix.Methods(), version=v)
    for bad in (items, bare):
        with pytest.raises(ValueError, match="hindex-v6 needs"):
            ix.build_index({ix.OVERREFUSAL: [bad]}, ix.Methods(), version="v6")
    ix.build_index({ix.OVERREFUSAL: [claims]}, ix.Methods(), version="v6")
    ix.build_index({ix.OVERREFUSAL: [claims]}, ix.Methods(), version="v3")  # v3: not read
    ix.build_index({}, ix.Methods(), version="v6")  # no overrefusal root: family absent


def test_v6_absent_at_zero_decline_lines(tmp_path):
    root = _v6_root(tmp_path / "rc", [_or_row(20, "o0", 0, 0), _or_row(20, "o1", 0, 0, 1)])
    built = ix.build_index({ix.OVERREFUSAL: [root]}, ix.Methods(), version="v6")
    assert built["rows"] == []  # absent, not 0: no family value, so no row
    (gone,) = built["safety"][ix.OVERREFUSAL]["absent"]
    assert (gone["layer"], gone["absent_reason"], gone["n_failed_cells"]) == (20, "no_claims", 1)


def _cli_argv(tmp_path: Path, or_root: Path) -> list[str]:
    src = _sources_v5(tmp_path / "src")
    (tmp_path / "methods.json").write_text("{}")
    argv = ["index", "--out", str(tmp_path / "none"), "--methods", str(tmp_path / "methods.json")]
    for fam, flag in (
        (ix.MAIN, "--main-root"),
        (ix.AM, "--am-root"),
        (ix.STORY, "--story-root"),
        (ix.SIBLING, "--um-root"),
    ):
        argv += [flag, str(src[fam][0])]
    return [*argv, "--overrefusal-root", str(or_root)]


def test_cli_default_is_v6_and_unit_guard(tmp_path, monkeypatch):
    for var in ix.SAFETY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    rc = _v6_root(tmp_path / "rc", [_or_row(20, "o0", 4, 1), _or_row(24, "o0", 2, 0)])
    argv = _cli_argv(tmp_path, rc)
    assert cli.main([*argv, "--dest", str(tmp_path / "d")]) == 0
    out = json.loads((tmp_path / "d" / "index.json").read_text())
    assert out["version"] == "hindex-v6"
    assert {r["layer"]: r["families"][ix.OVERREFUSAL]["v"] for r in out["rows"]} == {
        20: 0.25,
        24: 0.0,
    }
    for v in ("v4", "v5"):  # a refusal_claims root under v4 / v5: refused with a clear message
        with pytest.raises(SystemExit, match="unit 'refusal_claims'"):
            cli.main([*argv, "--version", v])
    items = _v6_root(tmp_path / "it", [_or_row(20, "o0", 1, 0)], unit="items")
    with pytest.raises(SystemExit, match="hindex-v6 needs"):
        cli.main(_cli_argv(tmp_path / "x", items))
    # the pre-existing checks run first (their messages are unchanged)
    with pytest.raises(SystemExit, match="not a directory"):
        cli.main(_cli_argv(tmp_path / "y", tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match=r"no index_counts\.jsonl"):
        cli.main(_cli_argv(tmp_path / "z", tmp_path / "empty"))


def test_report_refuses_a_refusal_claims_root(tmp_path, monkeypatch, capsys):
    """`hbench report`'s index refresh (library default v4) refuses a v6 root from the env."""
    monkeypatch.setattr(cli, "_cmd_report", lambda args: 0)
    rc = _v6_root(tmp_path / "rc", [_or_row(20, "o0", 4, 1)])
    monkeypatch.setenv("HBENCH_SAFETY_OR_OUT", str(rc))
    assert cli.main(["report", "--out", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "hallucination index not built: ValueError" in err and "refusal_claims" in err
    assert not (tmp_path / "index").exists()


# ---------------------------------------------------------------- v5 dev-box golden


def test_v5_devbox_golden_reproduced_exactly(tmp_path):
    """The official hindex-v5 run (#45, ``/workspace/hb-index-v5``), safety root mirrored."""
    golden = json.loads((FIX / "index_v5_devbox.json").read_text())
    assert golden["version"] == "hindex-v5"
    roots = [Path(r) for rs in golden["sources"].values() for r in rs]
    if missing := [r for r in roots if not r.is_dir()]:
        pytest.skip(f"dev-box sources absent: {missing[:2]}")
    built = ix.build_index(
        _pinned_roots_v4(golden, tmp_path), _golden_methods(golden), workers=8, version="v5"
    )
    assert _canon(built) == _canon(golden)
    assert ix.render_md(built) == (FIX / "index_v5_devbox.md").read_text()


def test_v6_note_disjoint():
    assert not set(ix.NOTES_V6) & (set(ix.NOTES_V4) | set(ix.NOTES_V5))
