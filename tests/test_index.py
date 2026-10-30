"""Plan 0013 Part B: the hallucination index v3 on tiny synthetic fixtures."""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from hallucination_bench import cli
from hallucination_bench import index as ix


def _segs(leaves: list[str]) -> list[dict[str, Any]]:
    return [{"readout": 0, "leaf": lf} for lf in leaves]


def _chat(item: str, layer: int, leaves: list[str], map_ok: bool = True) -> dict[str, Any]:
    stages = {"summary": "ok", "map": "ok" if map_ok else "pending"}
    return {"id": item, "layer": layer, "stages": stages, "map": {"segments": _segs(leaves)}}


def _am(item: str, layer: int, leaves: list[str]) -> dict[str, Any]:
    return {"id": item, "layer": layer, "map_status": "ok", "map": {"segments": _segs(leaves)}}


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


def test_unweighted_share_and_am_reader(tmp_path):
    d = _write_arm(tmp_path, "s", MAIN_CELLS)
    r = ix.claim_metric(ix.claim_rows(d / "cells.jsonl", ix.STORY)[20], ix.STORY)
    assert r is not None and r["v"] == pytest.approx(3 / 6)  # fc + cat + det, sib not counted
    assert "fc" not in r
    am = _write_arm(tmp_path, "m", [_am("x", 20, ["detail", "correct"]), MAIN_CELLS[0]])
    rows = ix.claim_rows(am / "cells.jsonl", ix.AM)  # the chat-shaped cell has no map_status
    assert rows[20].shape[0] == 1
    r = ix.claim_metric(rows[20], ix.AM)
    assert r is not None and r["v"] == pytest.approx(1 / 2)


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


def test_build_index_mapping_exclusion_and_partial_rows(tmp_path):
    main, am, story = tmp_path / "main", tmp_path / "am", tmp_path / "story"
    one = [_chat("i1", 20, ["false_claim", "correct"])]  # main 1/2, unweighted 1/2
    _write_arm(main, "a", [*one, _chat("i1", 24, ["junk"])])  # L24: no claims, no row
    _write_arm(main, "b", [_chat("i1", 20, ["correct"]), _chat("i1", 24, ["detail"])])
    _write_arm(main, "c", one)
    _write_arm(main, "x", one)
    _write_arm(main, "d", one)
    _write_arm(am, "c", [_am("i1", 20, ["correct", "correct", "detail", "correct"])])
    _write_arm(am, "d", [_am("i1", 20, ["detail"])])
    _write_arm(story, "c", [_chat("s1", 20, ["correct"])])
    um = _write_um(tmp_path / "um", {d: [{**u, "method": "c"} for u in us] for d, us in UM.items()})
    methods = _methods(tmp_path, {"methods": {"M": {"main": ["a", "b"]}}, "exclude": ["x"]})
    sources = {ix.MAIN: [main], ix.AM: [am], ix.STORY: [story], ix.SIBLING: [um]}
    out = ix.build_index(sources, methods)
    rows = {(r["method"], r["layer"]): r for r in out["rows"]}
    assert sorted(rows) == [("M", 20), ("M", 24), ("c", 20), ("d", 20)]  # a, b claimed; x out
    # the first arm in the list that has the layer wins
    assert rows[("M", 20)]["families"][ix.MAIN]["arm"] == "a"
    assert rows[("M", 20)]["families"][ix.MAIN]["v"] == pytest.approx(0.5)
    assert rows[("M", 24)]["families"][ix.MAIN]["arm"] == "b"  # falls through a's empty L24
    assert rows[("M", 24)]["families"][ix.MAIN]["v"] == pytest.approx(0.2)
    c = rows[("c", 20)]
    assert set(c["families"]) == set(ix.FAMILIES_V3)
    assert c["families"][ix.AM]["v"] == pytest.approx(0.25)
    assert c["families"][ix.STORY]["v"] == 0.0
    # (9 * 0.5 + 3 * 0.25 + 2 * 13/36 + 1 * 0) / 15 and (4.5 + 0.75 + 0) / 13
    assert c["full_v3"]["v"] == pytest.approx((4.5 + 0.75 + 26 / 36) / 15)
    assert c["matched_v3"]["v"] == pytest.approx(5.25 / 13)
    d = rows[("d", 20)]  # no story, no sibling: partial row, neither macro
    assert set(d["families"]) == {ix.MAIN, ix.AM}
    assert "full_v3" not in d and "matched_v3" not in d
    assert "full" not in d and "matched" not in d
    assert out["version"] == ix.INDEX_VERSION
    md = ix.render_md(out)
    assert "| c | 20 |" in md and "Excluded arms: x." in md


def test_missing_sibling_keeps_matched(tmp_path):
    root = tmp_path / "r"
    for fam in ("main", "am", "story"):
        _write_arm(root / fam, "c", [(_am if fam == "am" else _chat)("i1", 20, ["detail"])])
    sources = {f: [root / n] for f, n in ((ix.MAIN, "main"), (ix.AM, "am"), (ix.STORY, "story"))}
    (row,) = ix.build_index(sources, ix.Methods())["rows"]
    assert "full_v3" not in row and row["matched_v3"]["v"] == pytest.approx((9 * 0.2 + 3 + 1) / 13)


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
    argv += ["--version", "v4"]
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


def test_report_exit_code_survives_index_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cmd_report", lambda args: 0)

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(ix, "build_index", boom)
    assert cli.main(["report", "--out", str(tmp_path)]) == 0
    assert "hallucination index not built: RuntimeError: boom" in capsys.readouterr().err


def test_report_system_exit_skips_index(tmp_path, monkeypatch):
    def fail(args: object) -> int:
        raise SystemExit("report failed")

    def never(*a: object, **k: object) -> None:
        raise AssertionError("the index must not run")

    monkeypatch.setattr(cli, "_cmd_report", fail)
    monkeypatch.setattr(ix, "build_index", never)
    with pytest.raises(SystemExit, match="report failed"):
        cli.main(["report", "--out", str(tmp_path)])
    assert not (tmp_path / "index").exists()


def test_report_builds_index_at_default_dest(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_cmd_report", lambda args: 5)
    _write_arm(tmp_path / "wsbench_chat" / cli.DEFAULT_MODEL, "a", [_chat("i1", 20, ["detail"])])
    assert cli.main(["report", "--out", str(tmp_path)]) == 5  # report's own code is kept
    out = json.loads((tmp_path / "index" / cli.DEFAULT_MODEL / "index.json").read_text())
    assert [(r["method"], r["layer"]) for r in out["rows"]] == [("a", 20)]


def test_arm_listed_by_two_methods_and_unreadable_config(tmp_path):
    with pytest.raises(ValueError, match="listed by 'M' and 'N'"):
        _methods(tmp_path, {"methods": {"M": {"main": ["a"]}, "N": {"main": ["b", "a"]}}})
    _methods(tmp_path, {"methods": {"M": {"main": ["a"]}, "N": {"story": ["a"]}}})  # per family
    with pytest.raises(SystemExit, match=r"missing\.json"):
        cli.main(["index", "--out", str(tmp_path), "--methods", str(tmp_path / "missing.json")])
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(SystemExit):
        cli.main(["index", "--out", str(tmp_path), "--methods", str(tmp_path / "bad.json")])
