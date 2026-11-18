"""Plan 0027: the benign_roleplay index export (the ``prompts`` root hindex-v8 reads) and its
exact equivalence with the eval's own rate + CI. The hindex-v7 output tests were removed with
that version (plan 0030)."""

import json
from pathlib import Path
from typing import Any

import pytest
from index_helpers import INST, ITEMS, _items, _summary

from hallucination_bench import cli
from hallucination_bench import index as ix
from hallucination_bench.evals.benign_roleplay import index_export as bx
from hallucination_bench.evals.benign_roleplay import metrics
from hallucination_bench.evals.benign_roleplay.wsb import WsbError

FLAGS = {20: {"s0", "s5", "h1"}, 24: {"s3"}}


def _br_root(tmp: Path, arm: str = "jlens", flags: dict[int, set[str]] | None = None) -> Path:
    sp = _summary(tmp / "summary.json", ITEMS, {arm: flags or FLAGS})
    bx.export([sp], tmp / "root", ITEMS)
    return tmp / "root"


# ---------------------------------------------------------------- export == eval


def test_export_rows_and_meta(tmp_path):
    root = _br_root(tmp_path)
    rows = [json.loads(x) for x in (root / ix.SAFETY_COUNTS).read_text().splitlines()]
    assert len(rows) == 2 * 12 and {r["layer"] for r in rows} == {20, 24}  # no "all" unit
    r = {(x["layer"], x["item"]): x for x in rows}
    assert r[(20, "s0")]["n_contradicted"] == r[(20, "s0")]["n_false"] == 1
    assert r[(20, "s1")]["n_contradicted"] == 0 and r[(20, "s1")]["n_claims"] == 1
    assert r[(20, "s5")]["cluster"] == "t0" and r[(20, "h1")]["cluster"] == "h1"  # id fallback
    meta = json.loads((root / ix.SAFETY_META).read_text())
    assert (meta["unit"], meta["arms"], meta["layers"]) == (ix.UNIT_PROMPTS, ["jlens"], [20, 24])
    assert meta["instrument"] == INST


def test_index_value_and_ci_equal_the_eval_exactly(tmp_path):
    """safety_metric on the exported rows == metrics.rate_ci, bit for bit (plan 0027)."""
    root = _br_root(tmp_path)
    rows, _ = ix.safety_rows(root, ix.BENIGN_ROLEPLAY, "v8")
    s = json.loads((tmp_path / "summary.json").read_text())
    for L in (20, 24):
        got, _ = ix.safety_metric(rows["jlens"][L])
        u = s["rates"]["jlens"]["units"][f"L{L}"]
        assert got["v"] == u["false_recognition_rate"]
        assert got["ci"] == u["ci95"]
        assert got["n"] == u["n"] == 12


def test_export_refusals(tmp_path):
    good = {"jlens": FLAGS}
    sp = _summary(tmp_path / "a.json", ITEMS, good)
    with pytest.raises(WsbError, match="not index readers"):
        bx.export([sp], tmp_path / "x", ITEMS, arm_map={"jlens": "nope"})
    other = _summary(tmp_path / "b.json", ITEMS, {"s3d-rl600": FLAGS})
    s = json.loads(other.read_text())
    s["rates"]["s3d-rl600"]["instrument"]["prompt_version"] = "jb-v4"
    other.write_text(json.dumps(s))
    with pytest.raises(WsbError, match="different instruments"):
        bx.export([sp, other], tmp_path / "x", ITEMS)
    with pytest.raises(WsbError, match="two summaries"):
        bx.export([sp, sp], tmp_path / "x", ITEMS)
    with pytest.raises(WsbError, match="bank counts"):
        bx.export([sp], tmp_path / "x", _items(n=13))
    # a non-reader arm (e.g. plan 0023's mlnla-rl2-2000) is skipped and recorded, not refused
    meta0 = bx.export(
        [_summary(tmp_path / "c.json", ITEMS, {"mlnla11-rl600": FLAGS, "jlens": FLAGS})],
        tmp_path / "x0",
        ITEMS,
    )
    assert (meta0["arms"], meta0["skipped_arms"]) == (["jlens"], ["mlnla11-rl600"])
    with pytest.raises(WsbError, match="not arms of any summary"):
        bx.export([sp], tmp_path / "x", ITEMS, arm_map={"mlnla11-rl60": "mlnla-rl600-all11"})
    # mixed instruments within one summary
    two = _summary(tmp_path / "t.json", ITEMS, {"jlens": FLAGS, "s3d-rl600": FLAGS})
    s2 = json.loads(two.read_text())
    s2["rates"]["s3d-rl600"]["instrument"]["judge_model"] = "claude-opus-5-5"
    two.write_text(json.dumps(s2))
    with pytest.raises(WsbError, match="different instruments"):
        bx.export([two], tmp_path / "x", ITEMS)
    # --arm-map renames an eval arm to an index reader name
    meta = bx.export(
        [_summary(tmp_path / "d.json", ITEMS, {"mlnla11-rl600": FLAGS})],
        tmp_path / "y",
        ITEMS,
        arm_map={"mlnla11-rl600": "mlnla-rl600-all11"},
    )
    assert meta["arms"] == ["mlnla-rl600-all11"]

    def bad_unit(**kw: Any) -> Path:
        p = _summary(tmp_path / "e.json", ITEMS, good)
        s = json.loads(p.read_text())
        s["rates"]["jlens"]["units"]["L20"] |= kw
        p.write_text(json.dumps(s))
        return p

    with pytest.raises(WsbError, match="incomplete"):
        bx.export([bad_unit(complete=False, n_unjudged=1)], tmp_path / "x", ITEMS)
    with pytest.raises(WsbError, match="flagged_ids not within"):
        bx.export([bad_unit(flagged_ids=["s0", "zz", "h1"])], tmp_path / "x", ITEMS)
    p = bad_unit(n=11)  # pre-0027 summary (no scored_ids) with n short of the bank: refused
    s = json.loads(p.read_text())
    del s["rates"]["jlens"]["units"]["L20"]["scored_ids"]
    p.write_text(json.dumps(s))
    with pytest.raises(WsbError, match="no scored_ids"):
        bx.export([p], tmp_path / "x", ITEMS)


def test_export_old_summary_full_bank_and_unjudged(tmp_path):
    p = _summary(tmp_path / "a.json", ITEMS, {"jlens": FLAGS}, accept_refused_as_flagged=True)
    s = json.loads(p.read_text())
    u = s["rates"]["jlens"]["units"]
    for k in u:
        del u[k]["scored_ids"]  # a summary from before plan 0027: n == bank size is accepted
    u["L20"] |= {"complete": False, "n_unjudged": 1, "unjudged_ids": ["s5"]}
    p.write_text(json.dumps(s))
    bx.export([p], tmp_path / "root", ITEMS)
    rows = [json.loads(x) for x in (tmp_path / "root" / ix.SAFETY_COUNTS).read_text().splitlines()]
    failed = {(r["layer"], r["item"]) for r in rows if r["n_failed_cells"]}
    assert failed == {(20, "s5")}


def test_score_unit_records_scored_and_unjudged_ids():
    rows = [
        {"id": "a", "state": "judged", "any_recognition": True},
        {"id": "b", "state": "refused"},
    ]
    u = metrics.score_unit([{"numbers": {"extras": {"item_rows": rows}}}], ["a", "b", "c"], {})
    assert u["scored_ids"] == ["a", "c"]  # b refused: out of the denominator
    assert "c" in u["unjudged_ids"] and u["n"] == 2


# ---------------------------------------------------------------- the index


def test_v7_methods_json_maps_nla_br_arm_end_to_end(tmp_path):
    """Method mlnla-rl600 picks up its BR value from arm mlnla-rl600-all11 (methods.json)."""
    br = _br_root(tmp_path / "br", arm="mlnla-rl600-all11")
    built = ix.build_index(
        {ix.BENIGN_ROLEPLAY: [br]}, ix.load_methods(cli.INDEX_METHODS), version="v8"
    )
    rows = {(r["method"], r["layer"]): r for r in built["rows"]}
    got = rows[("mlnla-rl600", 20)]["families"][ix.BENIGN_ROLEPLAY]
    assert got["arm"] == "mlnla-rl600-all11"
    assert ("mlnla-rl600-all11", 20) not in rows  # claimed by the method, no identity row
