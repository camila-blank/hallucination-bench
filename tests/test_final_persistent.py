"""Plan 0029 Amendment 4: persistent judge failures -- `hbench final` excludes, counts and
reports the cells a hashed persistent-failures list names, and nothing else."""

import json
from pathlib import Path

import pytest
from test_final import _main, _manifest
from test_index_v8 import _am_cell, _chat_cell, _w

from hallucination_bench import final
from hallucination_bench import index as ix


@pytest.fixture(autouse=True)
def _two_layer_grid(monkeypatch):
    monkeypatch.setattr(final, "EXPECTED_LAYERS", (20, 24))
    # two items per unit: one excluded cell is 50%; only the bound test lowers this
    monkeypatch.setattr(final, "MAX_EXCLUDED_SHARE", 1.0)


def _list(tmp: Path, cells: list[dict], m: dict | None = None, **kw) -> dict:
    """The list file; each cell's ``file_sha256`` defaults to its manifest entry's sha256."""
    sha = {(e["family"], e["method"], e["unit"]): e["sha256"] for e in (m or {}).get("entries", [])}
    cells = [
        c
        if "file_sha256" in c or "family" not in c
        else {**c, "file_sha256": sha.get((c["family"], c["method"], c["unit"]), "0" * 64)}
        for c in cells
    ]
    doc = {"version": final.PERSISTENT_VERSION, "min_attempts": 3, "cells": cells, **kw}
    p = tmp / "persistent.json"
    p.write_text(json.dumps(doc))
    return {"path": "persistent.json", "sha256": final.sha256_file(p)}


def _cell(fam=ix.MAIN, unit="L20", cell="i0__x", reason="refused", attempts=3) -> dict:
    return {
        "family": fam,
        "method": "jlens",
        "unit": unit,
        "cell": cell,
        "reason": reason,
        "attempts": attempts,
    }


def _fail_main_cverify(roots: Path) -> None:
    cells = [_chat_cell(i, L, ["false_claim"]) for L in (20, 24) for i in ("i0", "i1")]
    cells[0]["stages"]["cverify"] = "failed"  # i0 @ L20: both verifier models refused
    _w(roots / "main" / "jlens" / ix.CELLS, cells)


def _with(tmp: Path, cells: list[dict], prep=_fail_main_cverify, **kw):
    def mutate(m):
        m["persistent_failures"] = _list(tmp, cells, m, **kw)

    return _manifest(tmp, prep=prep, mutate=mutate)


def test_without_a_list_nothing_changes(tmp_path, capsys):
    """The Amendment 2 rule is retained: an unlisted unsettled verdict still refuses."""
    assert _main(tmp_path, _manifest(tmp_path, prep=_fail_main_cverify)) == 2
    assert "1 cells with a failed / pending continuation-verifier" in capsys.readouterr().err


def test_a_listed_cell_is_excluded_counted_and_reported(tmp_path):
    m = _with(tmp_path, [_cell()])
    assert _main(tmp_path, m) == 0
    built = json.loads((tmp_path / "final" / "index.json").read_text())
    pf = built["manifest"]["persistent_failures"]
    assert pf["n_excluded"] == 1 and pf["path"] == "persistent.json"
    (x,) = pf["entries"]
    assert (x["family"], x["unit"], x["n_cells"], x["n_excluded"]) == (ix.MAIN, "L20", 2, 1)
    assert x["by_reason"] == {"refused": 1} and x["n_unsettled_verdicts"] == 1
    md = (tmp_path / "final" / final.EXCLUDED_MD).read_text()
    assert "| main | jlens | L20 | 2 | 1 | 50.00 | refused 1 | review |" in md
    assert "Total excluded: 1" in md
    index_md = (tmp_path / "final" / "index.md").read_text()
    assert "## Cells excluded as persistent judge failures" in index_md
    assert "| main | jlens | L20 | 2 | 1 | 50.00 | refused 1 | review |" in index_md
    # the excluded cell is dropped from the index (never scored): i1 alone at L20
    rows = {r["layer"]: r for r in built["rows"] if r["method"] == "jlens"}
    assert rows[20]["families"][ix.MAIN]["n_items"] == 1


def test_the_list_needs_its_hash_and_version(tmp_path, capsys):
    def bad_sha(m):
        m["persistent_failures"] = {**_list(tmp_path / "a", [_cell()], m), "sha256": "0" * 64}

    (tmp_path / "a").mkdir()
    assert (
        _main(tmp_path / "a", _manifest(tmp_path / "a", prep=_fail_main_cverify, mutate=bad_sha))
        == 2
    )
    assert "persistent_failures: sha256 mismatch" in capsys.readouterr().err
    assert _main(tmp_path / "b", _with(tmp_path / "b", [_cell()], version="v0")) == 2
    assert "version must be" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("cell", "msg"),
    [
        (_cell(reason="error"), "is not persistent"),
        (_cell(reason="unknown"), "is not persistent"),
        (_cell(attempts=2), "2 attempts < 3"),
        (_cell(unit="L99"), "which no manifest entry has"),
        ({"family": ix.MAIN, "method": "jlens", "unit": "L20", "cell": "i0__x"}, "misses"),
        (
            {**_cell(), "file_sha256": "f" * 64},
            "the persistent-failure list was built from another",
        ),
    ],
)
def test_only_persistent_cells_may_be_listed(tmp_path, capsys, cell, msg):
    assert _main(tmp_path, _with(tmp_path, [cell])) == 2
    assert msg in capsys.readouterr().err


def test_a_cell_listed_twice_refuses(tmp_path, capsys):
    assert _main(tmp_path, _with(tmp_path, [_cell(), _cell()])) == 2
    assert "listed twice" in capsys.readouterr().err


def test_a_stale_list_refuses(tmp_path, capsys):
    """A listed cell that is not failed in the hashed file (re-run since) refuses the manifest."""
    assert _main(tmp_path, _with(tmp_path, [_cell(cell="i1__x")])) == 2
    err = capsys.readouterr().err
    assert "1 persistent-failure cells are not failed" in err and "stale list" in err


def test_unlisted_failures_keep_the_one_percent_rule(tmp_path, capsys):
    def two_failed(roots: Path) -> None:
        cells = [_chat_cell(i, L, ["false_claim"]) for L in (20, 24) for i in ("i0", "i1")]
        cells[0]["stages"]["cverify"] = "failed"
        cells[1]["stages"]["map"] = "failed"  # i1 @ L20, transient: not listed
        _w(roots / "main" / "jlens" / ix.CELLS, cells)

    assert _main(tmp_path, _with(tmp_path, [_cell()], prep=two_failed)) == 2
    assert "main x jlens x L20: 1 of 2 cells failed" in capsys.readouterr().err


def test_declared_n_excluded_must_agree(tmp_path, capsys):
    def mutate(m):
        m["persistent_failures"] = _list(tmp_path, [_cell()], m)
        e = next(x for x in m["entries"] if (x["family"], x["unit"]) == (ix.MAIN, "L20"))
        e["n_excluded"] = 2

    assert _main(tmp_path, _manifest(tmp_path, prep=_fail_main_cverify, mutate=mutate)) == 2
    assert "declared n_excluded 2 != 1" in capsys.readouterr().err


def test_the_excluded_share_is_bounded(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(final, "MAX_EXCLUDED_SHARE", 0.4)  # 1 of 2 = 50%
    assert _main(tmp_path, _with(tmp_path, [_cell()])) == 2
    assert "excluded as persistent failures (50.00% > 40%)" in capsys.readouterr().err


def test_am_joint_and_safety_cells(tmp_path):
    def prep(roots: Path) -> None:
        _fail_main_cverify(roots)
        joint = [_am_cell(i, "all", ["false_claim"]) for i in ("i0", "i1")]
        joint[1]["map_status"] = "failed"  # invalid after the retries
        _w(roots / "am" / "jlens" / ix.CELLS_JOINT, joint)
        p = roots / "or" / ix.SAFETY_COUNTS
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        r = next(x for x in rows if x["layer"] == 24)
        r.update({c: 0 for c in ix.SAFETY_INT_COLS}, n_failed_cells=1)  # digest refused
        _w(p, rows)
        return r["item"]

    item = {}

    def prep2(roots):
        item["or"] = prep(roots)

    src_tmp = tmp_path

    def mutate(m):
        m["persistent_failures"] = _list(
            src_tmp,
            [
                _cell(),
                _cell(ix.AM_CLAIMS, "all", "i1__x", "invalid", 4),
                _cell(ix.OVERREFUSAL, "L24", item["or"]),
            ],
            m,
        )

    m = _manifest(tmp_path, prep=prep2, mutate=mutate)
    assert _main(tmp_path, m) == 0
    built = json.loads((tmp_path / "final" / "index.json").read_text())
    got = {
        (x["family"], x["unit"]): x["n_excluded"]
        for x in built["manifest"]["persistent_failures"]["entries"]
    }
    assert got == {(ix.MAIN, "L20"): 1, (ix.AM_CLAIMS, "all"): 1, (ix.OVERREFUSAL, "L24"): 1}


def test_recount_is_unchanged_by_the_ids(tmp_path):
    """`recount` still counts failed records (key-less rows too), as before Amendment 4."""
    p = tmp_path / "cells.jsonl"
    rows = [
        {"layer": 20, "status": "judged", "stages": {"map": "failed"}},
        {"layer": 20, "status": "judged", "stages": {"map": "failed"}},
        {"layer": 20, "status": "judged", "stages": {"map": "ok"}},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    e = {"family": ix.MAIN, "unit": "L20", "method": "m"}
    assert final.recount(e, p)[:2] == (3, 2)
    n, failed, _ = final.scan(e, p)
    assert n == 3 and [cv for _c, cv in failed] == [False, False]


def test_duplicate_failed_ids_refuse_a_listed_cell(tmp_path, capsys):
    def dup(roots: Path) -> None:
        cells = [_chat_cell(i, L, ["false_claim"]) for L in (20, 24) for i in ("i0", "i1")]
        cells[0]["stages"]["cverify"] = "failed"
        cells.append({**cells[0]})  # the same key twice at L20
        _w(roots / "main" / "jlens" / ix.CELLS, cells)

    assert _main(tmp_path, _with(tmp_path, [_cell()], prep=dup)) == 2
    assert "not unique among the failed records" in capsys.readouterr().err
