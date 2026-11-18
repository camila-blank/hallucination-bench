"""The index code every version shared, kept for hindex-v8 (plan 0030; moved unchanged from the
plan-0013 / 0019 / 0021 test files): the claim reader, the sibling metric, ``weighted_macro``,
the methods config, the safety metric and reader, and the AM candidate rule."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hallucination_bench import cli
from hallucination_bench import index as ix

# ---------------------------------------------------------------- claims, sibling, methods


def _segs(leaves: list[str]) -> list[dict[str, Any]]:
    return [{"readout": 0, "leaf": lf} for lf in leaves]


def _chat(item: str, layer: int, leaves: list[str], map_ok: bool = True) -> dict[str, Any]:
    stages = {"summary": "ok", "map": "ok" if map_ok else "pending"}
    return {"id": item, "layer": layer, "stages": stages, "map": {"segments": _segs(leaves)}}


def _write_arm(root: Path, arm: str, cells: list[dict[str, Any]]) -> Path:
    d = root / arm
    d.mkdir(parents=True)
    (d / "cells.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cells))
    return d


# layer 20: i1 = 3 claims (fc, det, correct; junk is not a claim) + 1 (sib; non_claim is not),
# i2 = 2 claims (cat, model_state); i3's cell is not mapped, i4's has no segments -> 6 claims over
# 2 items. Weighted: i1 1 + 0.2 + 0.5 = 1.7, i2 0.5 -> 2.2 / 6. Layer 24 has no claims.
MAIN_CELLS = [
    _chat("i1", 20, ["false_claim", "detail", "correct", "junk"]),
    _chat("i1", 20, ["substitution_sibling", "non_claim"]),
    _chat("i2", 20, ["substitution_category", "model_state"]),
    _chat("i3", 20, ["false_claim"], map_ok=False),
    _chat("i4", 20, []),
    _chat("i1", 24, ["junk", "off_topic"]),
    {"id": "i5", "layer": 20, "stages": {"map": "pending"}},  # no segments: skipped unparsed
]


def test_claim_rows_counts_and_main_weighted_share(tmp_path):
    d = _write_arm(tmp_path, "a", MAIN_CELLS)
    rows = ix.claim_rows(d / "cells.jsonl", ix.MAIN)
    assert sorted(rows) == [20, 24]
    assert rows[20].shape == (2, ix.N_COLS)  # i1, i2 (i3 unmapped, i4 empty)
    assert rows[24][0][ix.C_CLAIMS] == 0
    assert ix.claim_metric(rows[24], ix.MAIN) is None  # no claims: family absent at L24
    r = ix.claim_metric(rows[20], ix.MAIN)
    assert r is not None
    assert r["v"] == pytest.approx(2.2 / 6)
    assert (r["n"], r["n_items"]) == (6, 2)
    for k in ("fc", "cat", "sib", "det"):
        assert r[k] == pytest.approx(1 / 6)
    # every 2-item draw is {i1,i1} = 3.4/8, {i2,i2} = 0.5/2 or mixed = 2.2/6
    assert r["ci"] == pytest.approx([0.25, 0.425])


def _unit(item: str, fire: bool, cross: bool, committed: bool = True, na: str | None = None):
    return {
        "method": "m",
        "item": item,
        "layer": 20,
        "committed": committed,
        "probe": {"na_reason": na, "fire_L": fire},
        "cross": ["x"] if cross else [],
    }


def _write_um(root: Path, units: dict[str, list[dict[str, Any]]]) -> Path:
    for dom, us in units.items():
        p = ix.um_paths(root)[dom]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(u) + "\n" for u in us))
    return root


UM = {
    # faithful (fire_L wins over cross), cross, hallucination; uncommitted / probe n/a excluded
    "country": [
        _unit("c1", True, True),
        _unit("c1", False, True),
        _unit("c2", False, False),
        _unit("c2", False, False, committed=False),
        _unit("c3", False, False, na="invalid probe at L"),
    ],
    "capital_city": [_unit("k1", False, False), _unit("k2", True, False)],
    "holiday": [_unit("h1", False, False), *(_unit(f"h{i}", True, False) for i in (2, 3, 4))],
}


def test_um_label_rule_and_sibling_macro(tmp_path):
    units = ix.um_rows(_write_um(tmp_path, UM))
    assert [lb for _, lb in units[("m", 20, "country")]] == ["faithful", "cross", "hallucination"]
    r = ix.sibling_metric({d: units[("m", 20, d)] for d in ix.UM_DOMAINS})
    assert r is not None
    assert r["v"] == pytest.approx(13 / 36)  # (1/3 + 1/2 + 1/4) / 3
    assert [r["per_domain"][d]["hall"] for d in ix.UM_DOMAINS] == pytest.approx(
        [1 / 3, 1 / 2, 1 / 4]
    )
    assert (r["n"], r["n_items"]) == (9, 8)
    assert r["ci"][0] <= r["v"] <= r["ci"][1]
    assert ix.sibling_metric({"country": units[("m", 20, "country")]}) is None


def test_weighted_macro_hand_computed():
    per = {
        ix.MAIN: {"v": 0.1, "ci": [0.08, 0.12]},
        ix.AM: {"v": 0.2, "ci": [0.1, 0.3]},
        ix.SIBLING: {"v": 0.3, "ci": [0.2, 0.4]},
        ix.STORY: {"v": 0.4, "ci": [0.3, 0.5]},
    }
    full = ix.weighted_macro(per, ix.FAMILIES_V3)
    assert full["v"] == pytest.approx(2.5 / 15)  # 0.9 + 0.6 + 0.6 + 0.4 over 9 + 3 + 2 + 1
    se = math.sqrt(0.36**2 + 0.6**2 + 0.4**2 + 0.2**2) / 3.92 / 15  # w_f * CI width
    assert full["ci"] == pytest.approx([2.5 / 15 - 1.96 * se, 2.5 / 15 + 1.96 * se])
    assert ix.weighted_macro(per, ix.MATCHED_V3)["v"] == pytest.approx(1.9 / 13)


def _methods(tmp_path: Path, cfg: dict[str, Any]) -> ix.Methods:
    p = tmp_path / "methods.json"
    p.write_text(json.dumps(cfg))
    return ix.load_methods(p)


def test_duplicate_arm_and_bad_config(tmp_path):
    for r in ("r1", "r2"):
        _write_arm(tmp_path / r, "a", [_chat("i1", 20, ["detail"])])
    with pytest.raises(ValueError, match="under two roots"):
        ix.build_index({ix.MAIN: [tmp_path / "r1", tmp_path / "r2"]}, ix.Methods())
    with pytest.raises(SystemExit, match="under two roots"):
        cli.main(
            ["index", "--main-root", str(tmp_path / "r1"), "--main-root", str(tmp_path / "r2")]
        )
    with pytest.raises(ValueError, match="excluded"):
        _methods(tmp_path, {"methods": {"M": {"main": ["a"]}}, "exclude": ["a"]})
    with pytest.raises(ValueError, match="unknown families"):
        _methods(tmp_path, {"methods": {"M": {"chat": ["a"]}}})


def test_cli_index_writes_json_and_md(tmp_path, capsys):
    (tmp_path / "methods.json").write_text("{}")
    _write_arm(tmp_path / "main", "a", [_chat("i1", 20, ["detail"])])
    dest = tmp_path / "dest"
    argv = ["index", "--out", str(tmp_path / "none"), "--main-root", str(tmp_path / "main")]
    argv += ["--dest", str(dest), "--methods", str(tmp_path / "methods.json")]
    argv += ["--version", "v8"]
    assert cli.main(argv) == 0
    out = json.loads((dest / "index.json").read_text())
    assert out["version"] == ix.INDEX_VERSION and "git_sha" in out
    assert [(r["method"], r["layer"]) for r in out["rows"]] == [("a", 20)]
    assert (
        "| a | 20 |" in (dest / "index.md").read_text() and "| a | 20 |" in capsys.readouterr().out
    )


def test_default_methods_config_loads():
    m = ix.load_methods(cli.INDEX_METHODS)
    assert m.methods["s3d-rl600"]["sibling"] == ["s3d"]
    assert {"nla-rl-L42", "nla-sft-L42"} <= m.exclude


def test_arm_listed_by_two_methods_and_unreadable_config(tmp_path):
    with pytest.raises(ValueError, match="listed by 'M' and 'N'"):
        _methods(tmp_path, {"methods": {"M": {"main": ["a"]}, "N": {"main": ["b", "a"]}}})
    _methods(tmp_path, {"methods": {"M": {"main": ["a"]}, "N": {"story": ["a"]}}})  # per family
    with pytest.raises(SystemExit, match=r"missing\.json"):
        cli.main(["index", "--out", str(tmp_path), "--methods", str(tmp_path / "missing.json")])
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(SystemExit):
        cli.main(["index", "--out", str(tmp_path), "--methods", str(tmp_path / "bad.json")])


# ---------------------------------------------------------------- safety families


def _row(arm: str, layer: int, item: str, cluster: str | None = None, **kw: int) -> dict:
    r: dict[str, Any] = {"arm": arm, "layer": layer, "item": item, "cluster": cluster or item}
    r |= dict.fromkeys(ix.SAFETY_INT_COLS, 0) | kw
    return r


def _write_safety(root: Path, rows: list[dict], meta: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / ix.SAFETY_COUNTS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if meta is not None:
        (root / ix.SAFETY_META).write_text(json.dumps(meta))
    return root


# i1: 4 claims, 3 false (2 contradicted, 1 unprobed); i2: 2 claims, 1 false (unprobed);
# i3: no claims (a zero row); one unresolved / failed cell are health counts only
SHARE_ROWS = [
    _row("jlens", 20, "i1", n_claims=4, n_false=3, n_contradicted=2, n_unprobed=1, n_units=5),
    _row("jlens", 20, "i2", n_claims=2, n_false=1, n_unprobed=1, n_unresolved=1, n_units=3),
    _row("jlens", 20, "i3", n_failed_cells=2, n_units=2),
]


def test_safety_share_hand_computed():
    r, counts = ix.safety_metric(SHARE_ROWS)
    assert r is not None
    assert r["v"] == pytest.approx(2 / 6)  # contradicted / claims; unprobed stay in denominator
    assert r["raw"] == pytest.approx(4 / 6)
    assert (r["n"], r["n_items"], r["n_clusters"]) == (6, 3, 3)
    assert (r["n_false"], r["n_contradicted"], r["n_unprobed"]) == (4, 2, 2)
    assert r["health"] == {"n_unprobed": 2, "n_unresolved": 1, "n_failed_cells": 2}
    assert r["units"]["n_units"] == 10
    assert counts["n_items"] == 3
    # clusters 2/4, 0/2, 0/0: every defined draw lies in [0, 1/2] and both ends are reached; the
    # all-i3 draws (0/0, ~1/27 of them) are skipped, not propagated as NaN
    assert r["ci"] == [0.0, 0.5]


def test_safety_absent_rules():
    none, c = ix.safety_metric([_row("jlens", 20, "i1"), _row("jlens", 20, "i2")])
    assert none is None and c["absent_reason"] == ix.ABSENT_NO_CLAIMS and c["n_items"] == 2
    rows = [_row("jlens", 20, "i1", n_claims=3, n_false=2, n_unprobed=2)]
    none, c = ix.safety_metric(rows)
    assert none is None and c["absent_reason"] == ix.ABSENT_ALL_UNPROBED
    # claims but no false claim: a real 0, not absent
    r, _ = ix.safety_metric([_row("jlens", 20, "i1", n_claims=3)])
    assert r is not None and r["v"] == 0.0 and r["ci"] == [0.0, 0.0]
    # one probed false claim that is not contradicted: present (0), not all-unprobed
    r, _ = ix.safety_metric([_row("jlens", 20, "i1", n_claims=3, n_false=2, n_unprobed=1)])
    assert r is not None and r["v"] == 0.0


def test_cluster_bootstrap_resamples_clusters_in_sorted_order():
    # clusters "b" = items i1, i2 (each 1/1 contradicted), "a" = i3, i4 (each 0/1)
    rows = [
        _row("jlens", 20, "i1", "b", n_claims=1, n_false=1, n_contradicted=1),
        _row("jlens", 20, "i2", "b", n_claims=1, n_false=1, n_contradicted=1),
        _row("jlens", 20, "i3", "a", n_claims=1),
        _row("jlens", 20, "i4", "a", n_claims=1),
    ]
    r, _ = ix.safety_metric(rows)
    assert r is not None and r["v"] == 0.5 and r["n_clusters"] == 2
    mat = np.array([[0.0, 2.0], [2.0, 2.0]])  # sorted clusters: a, b
    idx = np.random.default_rng(0).integers(0, 2, size=(ix.N_BOOT_SAFETY, 2))
    bs = mat[idx].sum(1)
    assert r["ci"] == np.nanpercentile(bs[:, 0] / bs[:, 1], [2.5, 97.5]).tolist()
    assert r["ci"] == [0.0, 1.0]  # cluster draws are 0, 1/2 or 1 only
    assert ix.safety_metric(list(reversed(rows)))[0] == r  # row order is irrelevant; fresh RNG


def test_safety_reader_allowlist_and_columns(tmp_path):
    root = _write_safety(
        tmp_path / "or",
        [
            _row("mlnla-rl600", 20, "i1", n_claims=1),  # not overrefusal's dir name
            _row("mlnla-rl600-all11", 20, "i1", n_claims=1),
            _row("textreader-x", 20, "i1", n_claims=1),
        ],
    )
    rows, ignored = ix.safety_rows(root, ix.OVERREFUSAL)
    assert set(rows) == {"mlnla-rl600-all11"} and ignored == {"mlnla-rl600", "textreader-x"}
    assert ix.safety_rows(tmp_path / "empty", ix.OVERREFUSAL) == ({}, set())
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / ix.SAFETY_COUNTS).write_text(json.dumps({"arm": "jlens", "layer": 20}) + "\n")
    with pytest.raises(ValueError, match="missing columns"):
        ix.safety_rows(bad, ix.OVERREFUSAL)
    dup = _write_safety(tmp_path / "dup", [_row("jlens", 20, "i1"), _row("jlens", 20, "i1")])
    with pytest.raises(ValueError, match="duplicate"):
        ix.safety_rows(dup, ix.OVERREFUSAL)
    with pytest.raises(ValueError, match="not a safety family"):
        ix.safety_rows(root, ix.MAIN)


def test_safety_arm_under_two_roots(tmp_path):
    meta = {"unit": "refusal_claims"}
    a = _write_safety(tmp_path / "a", [_row("jlens", 20, "i1", n_claims=1)], meta)
    b = _write_safety(tmp_path / "b", [_row("jlens", 24, "i1", n_claims=1)], meta)
    with pytest.raises(ValueError, match="under two roots"):
        ix.build_index({ix.OVERREFUSAL: [a, b]}, ix.Methods())


def test_methods_json_maps_safety_arms():
    m = ix.load_methods(cli.INDEX_METHODS)
    arms = {
        ix.OVERREFUSAL: ["s3d-rl600", "mlnla-rl600-all11", "jlens"],
    }
    res = ix.resolve_methods(m, arms)
    for f in ix.SAFETY_FAMILIES:
        assert res["s3d-rl600"][f] == ["s3d-rl600"]
        assert res["jlens"][f] == ["jlens"]
    assert res["mlnla-rl600"][ix.OVERREFUSAL] == ["mlnla-rl600-all11"]
    assert "mlnla-rl600-all11" not in res  # no stray identity method
    assert set(ix.resolve_methods(m, arms, ix.FAMILIES_V3)["s3d-rl600"]) == set(ix.FAMILIES_V3)


# ---------------------------------------------------------------- the candidate rule


def _seg(readout: int, start: int, end: int, leaf: str) -> dict[str, Any]:
    return {"readout": readout, "start": start, "end": end, "leaf": leaf}


def _ment(readout: int, start: int, end: int) -> dict[str, Any]:
    return {"readout": readout, "start": start, "end": end}


def _am(
    item: str, layer: int, segs: list[dict], ments: list[dict], ok: bool = True
) -> dict[str, Any]:
    return {
        "id": item,
        "layer": layer,
        "map_status": "ok" if ok else "failed",
        "map": {"segments": segs},
        "mentions": ments,
    }


# layer 20, i1: mentions [9, 15) + [14, 25) of readout 0 merge to [9, 25).
#   [0, 10) false_claim overlaps by 1 char -> candidate (hallucinated)
#   [10, 20) correct inside -> candidate; [25, 30) detail touches (end == 25) -> not
#   readout 1 [0, 10) false_claim: no mention in readout 1 -> not; [12, 18) non_claim: not a claim
# layer 20, i2: one false claim [0, 50), mention at [100, 110) -> a zero row
# layer 20, i3: map failed -> not an item. layer 24, i1: mentions miss every claim -> absent
AM_CELLS = [
    _am(
        "i1",
        20,
        [
            _seg(0, 0, 10, "false_claim"),
            _seg(0, 10, 20, "correct"),
            _seg(0, 25, 30, "detail"),
            _seg(1, 0, 10, "false_claim"),
            _seg(0, 12, 18, "non_claim"),
        ],
        [_ment(0, 14, 25), _ment(0, 9, 15)],
    ),
    _am("i2", 20, [_seg(0, 0, 50, "false_claim")], [_ment(0, 100, 110)]),
    _am("i3", 20, [_seg(0, 0, 50, "false_claim")], [_ment(0, 0, 50)], ok=False),
    _am("i1", 24, [_seg(0, 0, 10, "detail")], [_ment(0, 10, 20), _ment(1, 0, 10)]),
]


def _write_cells(root: Path, arm: str, cells: list[dict[str, Any]]) -> Path:
    (root / arm).mkdir(parents=True)
    (root / arm / ix.CELLS).write_text("".join(json.dumps(c) + "\n" for c in cells))
    return root / arm / ix.CELLS


def test_is_candidate_overlap_rule():
    spans = {0: [(9, 25)]}
    assert ix.is_candidate(_seg(0, 0, 10, "x"), spans)  # 1 char
    assert not ix.is_candidate(_seg(0, 0, 9, "x"), spans)  # touching at the start
    assert not ix.is_candidate(_seg(0, 25, 30, "x"), spans)  # touching at the end
    assert ix.is_candidate(_seg(0, 24, 30, "x"), spans)
    assert not ix.is_candidate(_seg(1, 10, 20, "x"), spans)  # another readout
    assert ix.is_candidate(_seg(0, 0, 100, "x"), spans)  # containing


def test_claim_rows_candidates(tmp_path):
    p = _write_cells(tmp_path, "a", AM_CELLS)
    rows = ix.claim_rows(p, ix.AM, am_candidates=True)
    assert sorted(rows) == [20, 24]
    m20 = rows[20]
    assert m20.shape == (2, ix.N_COLS_CANDIDATES)  # i1, i2 (i3 unmapped)
    i1, i2 = m20
    assert (i1[ix.C_CLAIMS], i1[ix.C_FC], i1[ix.C_DET], i1[ix.C_ALL_CLAIMS]) == (2, 1, 0, 4)
    assert (i2[ix.C_CLAIMS], i2[ix.C_ALL_CLAIMS]) == (0, 1)  # a zero row, kept
    # readout / segment health columns: unchanged from the all-claims reader
    base = ix.claim_rows(p, ix.AM)
    assert base[20].shape == (2, ix.N_COLS)
    cols = [ix.C_READOUTS, ix.C_SEGMENTS]
    assert np.array_equal(m20[:, cols], base[20][:, cols])
    assert (base[20][0][ix.C_CLAIMS], base[20][1][ix.C_CLAIMS]) == (4, 1)
    r = ix.claim_metric(m20, ix.AM)
    assert r is not None and (r["v"], r["n"], r["n_items"]) == (0.5, 2, 2)
    assert ix.claim_metric(rows[24], ix.AM) is None  # 0 candidates: absent
    # the flag only applies to AM
    for fam in (ix.MAIN, ix.STORY):
        assert ix.claim_rows(p, fam, am_candidates=True) == {}  # AM-shaped cells: not mapped
    chat = [{"id": "c", "layer": 20, "stages": {"map": "ok"}, "map": AM_CELLS[0]["map"]}]
    q = _write_cells(tmp_path, "chat", chat)
    assert ix.claim_rows(q, ix.MAIN, am_candidates=True)[20].shape == (1, ix.N_COLS)


def test_merged_mentions_count_once(tmp_path):
    """Overlapping mentions merge: a claim overlapping both is one candidate, and a claim
    overlapping only their union (the gap is covered by the other one) is a candidate."""
    cells = [
        _am(
            "i1",
            20,
            [_seg(0, 4, 6, "false_claim"), _seg(0, 13, 15, "detail")],
            [_ment(0, 0, 10), _ment(0, 5, 14)],
        )
    ]
    (row,) = ix.claim_rows(_write_cells(tmp_path, "a", cells), ix.AM, am_candidates=True)[20]
    assert (row[ix.C_CLAIMS], row[ix.C_FC], row[ix.C_DET]) == (2, 1, 1)


# ---------------------------------------------------------------- v8 guards (plan 0030)


def test_unit_guard_v8(tmp_path):
    """Overrefusal needs a `refusal_claims` root, benign_roleplay a `prompts` root; a cheating
    root is on no index; only v8 is accepted."""
    row = [_row("jlens", 20, "i1", n_claims=1)]
    good_or = _write_safety(tmp_path / "or", row, {"unit": "refusal_claims"})
    good_br = _write_safety(tmp_path / "br", row, {"unit": "prompts", "instrument": {"a": 1}})
    ix.check_safety_units({ix.OVERREFUSAL: [good_or], ix.BENIGN_ROLEPLAY: [good_br]}, "v8")
    for meta in ({"unit": "items"}, None):
        bad = _write_safety(tmp_path / f"or_{bool(meta)}", row, meta)
        with pytest.raises(ValueError, match="needs a 'refusal_claims' root"):
            ix.check_safety_units({ix.OVERREFUSAL: [bad]}, "v8")
        with pytest.raises(ValueError, match="needs a 'prompts' root"):
            ix.check_safety_units({ix.BENIGN_ROLEPLAY: [bad]}, "v8")
    other = _write_safety(tmp_path / "br2", row, {"unit": "prompts", "instrument": {"a": 2}})
    with pytest.raises(ValueError, match="different instruments"):
        ix.check_safety_units({ix.BENIGN_ROLEPLAY: [good_br, other]}, "v8")
    with pytest.raises(ValueError, match="on no index"):
        ix.check_safety_units({ix.CHEATING: [tmp_path / "ch"]}, "v8")
    with pytest.raises(ValueError, match="hindex-v8 is the only index version"):
        ix.check_safety_units({ix.OVERREFUSAL: [good_or]}, "v6")


def test_unlisted_reader_never_becomes_a_method(tmp_path):
    root = _write_safety(
        tmp_path / "or",
        [_row("textreader-x", 20, "i1", n_claims=2, n_false=1, n_contradicted=1)],
        {"unit": "refusal_claims"},
    )
    rows, ignored = ix.safety_rows(root, ix.OVERREFUSAL)
    assert rows == {} and ignored == {"textreader-x"}


def test_safety_roots_are_opt_in(tmp_path, monkeypatch):
    for var in ix.SAFETY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    src = ix.default_sources(tmp_path, cli.DEFAULT_MODEL)
    assert not set(src) & set(ix.safety_families())  # the dev-box defaults are never used
    monkeypatch.setenv("HBENCH_SAFETY_OR_OUT", str(tmp_path / "or"))
    assert ix.default_sources(tmp_path, cli.DEFAULT_MODEL)[ix.OVERREFUSAL] == [tmp_path / "or"]


def test_overrefusal_export_readers():
    from hallucination_bench.evals.overrefusal import index_export

    assert index_export.READERS == ("s3d-rl600", "mlnla-rl600-all11", "jlens")
    assert index_export.EXPORT_READERS == (*index_export.READERS, "final-olens-rl600-x4")
