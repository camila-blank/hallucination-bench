"""Plan 0027: hindex-v7 (benign_roleplay on the index, weight 2; sibling weight 1), the
benign_roleplay index export, and its exact equivalence with the eval's own rate + CI."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_index_v5 import _sources_v5
from test_index_v6 import _or_row, _v6_root

from hallucination_bench import cli, paths
from hallucination_bench import index as ix
from hallucination_bench.evals.benign_roleplay import bank, metrics
from hallucination_bench.evals.benign_roleplay import index_export as bx
from hallucination_bench.evals.benign_roleplay.wsb import GRID, WsbError

INST = {
    "prompt_version": "jb-v3+digest-v5-task-extractive+sj-v10",
    "judge_model": "claude-sonnet-5",
}


# ---------------------------------------------------------------- export fixtures


def _items(n: int = 12, n_hb: int = 4, templates: int = 5) -> dict[str, Any]:
    items = [
        {"id": f"s{k}", "hb_only": False, "template_id": f"t{k % templates}"}
        for k in range(n - n_hb)
    ]
    items += [{"id": f"h{k}", "hb_only": True, "template_id": None} for k in range(n_hb)]
    return {"items": items}


def _unit(items_doc: dict[str, Any], flagged: set[str], **kw: Any) -> dict[str, Any]:
    """A summary unit exactly as metrics.score_unit builds it (the real function)."""
    clusters = {it["id"]: str(it.get("template_id") or it["id"]) for it in items_doc["items"]}
    ids = [it["id"] for it in items_doc["items"]]
    flags = {i: i in flagged for i in ids}
    rate, ci = metrics.rate_ci(flags, clusters)
    u = {
        "false_recognition_rate": rate,
        "ci95": list(ci),
        "n": len(ids),
        "n_flagged": len(flagged),
        "n_unjudged": 0,
        "complete": True,
        "flagged_ids": sorted(flagged),
        "scored_ids": sorted(ids),
        "unjudged_ids": [],
    }
    return u | kw


def _summary(
    path: Path, items_doc: dict[str, Any], arms: dict[str, dict[int, set[str]]], **kw: Any
) -> Path:
    items = items_doc["items"]
    n_hb = sum(it["hb_only"] for it in items)
    s = {
        "bank": {"n_items": len(items), "n_shared": len(items) - n_hb, "n_hb_only": n_hb},
        "accept_refused_as_flagged": False,
        "rates": {
            arm: {
                "instrument": dict(INST),
                "units": {
                    "all": _unit(items_doc, set().union(*by.values())),
                    **{f"L{L}": _unit(items_doc, fl) for L, fl in by.items()},
                },
            }
            for arm, by in arms.items()
        },
    } | kw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s))
    return path


ITEMS = _items()
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
    rows, _ = ix.safety_rows(root, ix.BENIGN_ROLEPLAY, "v7")
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
    with pytest.raises(WsbError, match="not an index reader"):
        bx.export(
            [_summary(tmp_path / "c.json", ITEMS, {"mlnla11-rl600": FLAGS})], tmp_path / "x", ITEMS
        )
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


def _v7_sources(tmp: Path, br: Path) -> dict[str, list[Path]]:
    src = _sources_v5(tmp / "src")
    # overrefusal: jlens + a final-olens row (the v6 allowlist must hold under v7)
    rows = [_or_row(20, "o0", 4, 1), _or_row(20, "o1", 4, 0)]
    rows += [{**_or_row(20, "o0", 2, 1), "arm": "final-olens-rl600-x4"}]
    return {**src, ix.OVERREFUSAL: [_v6_root(tmp / "or6", rows)], ix.BENIGN_ROLEPLAY: [br]}


def test_v7_weights_and_macros(tmp_path):
    br = _br_root(tmp_path / "br")
    src = _v7_sources(tmp_path, br)
    v6 = ix.build_index(
        {k: v for k, v in src.items() if k != ix.BENIGN_ROLEPLAY}, ix.Methods(), version="v6"
    )
    v7 = ix.build_index(src, ix.Methods(notes={"k": "n"}), version="v7")
    assert v7["version"] == "hindex-v7"
    assert sum(ix.FAMILY_WEIGHTS_V7.values()) == 17
    assert ix.FAMILY_WEIGHTS_V7[ix.SIBLING] == 1 and ix.FAMILY_WEIGHTS_V7[ix.BENIGN_ROLEPLAY] == 2
    assert v7["weights"]["family"] == ix.FAMILY_WEIGHTS_V7
    assert v7["weights"]["family_v6"] == ix.FAMILY_WEIGHTS
    assert v7["notes"] == {"k": "n", **ix.NOTES_V6, **ix.NOTES_V7}
    r6 = {(r["method"], r["layer"]): r for r in v6["rows"]}
    r7 = {(r["method"], r["layer"]): r for r in v7["rows"]}
    row = r7[("jlens", 20)]
    per = row["families"]
    assert set(per) == set(ix.FAMILIES_V7)
    w = ix.FAMILY_WEIGHTS_V7
    full = sum(w[f] * per[f]["v"] for f in ix.FAMILIES_V7) / 17
    assert row["full"]["v"] == pytest.approx(full)
    matched = sum(w[f] * per[f]["v"] for f in ix.MATCHED_V7) / 16
    assert row["matched"]["v"] == pytest.approx(matched)
    # continuity: v7's full_v6 / matched_v6 / full_v3 are exactly the v6 build's macros
    for key6, key7 in (("full", "full_v6"), ("matched", "matched_v6"), ("full_v3", "full_v3")):
        assert json.dumps(row[key7]) == json.dumps(r6[("jlens", 20)][key6])
    # every v6 family value is bit-identical under v7, final-olens's overrefusal included
    for k, r in r6.items():
        for f, v in r["families"].items():
            assert json.dumps(r7[k]["families"][f]) == json.dumps(v), (k, f)
    assert ("final-olens-rl600-x4", 20) in r7
    # BR value == the eval's own
    s = json.loads((tmp_path / "br" / "summary.json").read_text())
    assert (
        per[ix.BENIGN_ROLEPLAY]["v"]
        == s["rates"]["jlens"]["units"]["L20"]["false_recognition_rate"]
    )
    assert v7["safety"][ix.BENIGN_ROLEPLAY]["share"] == ix.SHARE_BR
    assert v7["safety"][ix.OVERREFUSAL]["share"] == ix.SHARE_V6
    md = ix.render_md(v7)
    assert md.startswith("# Hallucination index (hindex-v7)")
    assert "| full | matched | full_v6 | main |" in md and "| overrefusal | benign_roleplay |" in md
    assert "benign_roleplay 2" in md and "sibling 1" in md


def test_v7_without_br_has_no_full_but_keeps_v6_macros(tmp_path):
    src = _v7_sources(tmp_path, tmp_path / "unused")
    del src[ix.BENIGN_ROLEPLAY]
    (row,) = [
        r
        for r in ix.build_index(src, ix.Methods(), version="v7")["rows"]
        if r["layer"] == 20 and r["method"] == "jlens"
    ]
    assert "full" not in row and "matched" not in row
    assert "full_v6" in row and "matched_v6" in row


def test_v3_to_v6_unchanged_and_refuse_br(tmp_path):
    br = _br_root(tmp_path / "br")
    src = _v7_sources(tmp_path, br)
    base = {k: v for k, v in src.items() if k != ix.BENIGN_ROLEPLAY}
    for v in ("v3", "v4", "v5", "v6"):
        with pytest.raises(ValueError, match="hindex-v7 only"):
            ix.build_index(src, ix.Methods(), version=v)
        assert ix.BENIGN_ROLEPLAY not in ix.safety_families(v)
        assert ix.safety_readers(ix.BENIGN_ROLEPLAY, v) == frozenset()
    ix.build_index({**base, ix.OVERREFUSAL: base[ix.OVERREFUSAL]}, ix.Methods(), version="v3")
    assert ix.FAMILIES == (*ix.FAMILIES_V3, ix.OVERREFUSAL)
    assert ix.FAMILY_WEIGHTS[ix.SIBLING] == 2.0  # v4-v6 table untouched


def test_v7_unit_guard(tmp_path):
    rows = [{**r, "arm": "jlens"} for r in [_or_row(20, "o0", 1, 0)]]
    bad = _v6_root(tmp_path / "bad", rows, unit="items")
    or6 = _v6_root(tmp_path / "or6", [_or_row(20, "o0", 1, 0)])
    with pytest.raises(ValueError, match="'prompts' root"):
        ix.build_index({ix.BENIGN_ROLEPLAY: [bad]}, ix.Methods(), version="v7")
    with pytest.raises(ValueError, match="hindex-v7 needs a 'refusal_claims'"):
        ix.build_index({ix.OVERREFUSAL: [bad]}, ix.Methods(), version="v7")
    ix.build_index({ix.OVERREFUSAL: [or6]}, ix.Methods(), version="v7")
    a = _br_root(tmp_path / "a", arm="jlens")
    b = _br_root(tmp_path / "b", arm="s3d-rl600")
    m = json.loads((b / ix.SAFETY_META).read_text())
    m["instrument"] = {**m["instrument"], "prompt_version": "jb-v4"}
    (b / ix.SAFETY_META).write_text(json.dumps(m))
    with pytest.raises(ValueError, match="different instruments"):
        ix.build_index({ix.BENIGN_ROLEPLAY: [a, b]}, ix.Methods(), version="v7")


def test_methods_json_maps_br_under_v7_only(tmp_path):
    m = ix.load_methods(cli.INDEX_METHODS)
    assert m.methods["mlnla-rl600"][ix.BENIGN_ROLEPLAY] == ["mlnla-rl600-all11"]
    r7 = ix.resolve_methods(m, {}, ix.families_for("v7"))
    assert r7["mlnla-rl600"][ix.BENIGN_ROLEPLAY] == ["mlnla-rl600-all11"]
    r6 = ix.resolve_methods(m, {}, ix.families_for("v6"))
    assert ix.BENIGN_ROLEPLAY not in r6["mlnla-rl600"]


def test_env_root_reaches_v7_only(tmp_path, monkeypatch):
    for var in ix.SAFETY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HBENCH_SAFETY_BR_OUT", str(tmp_path / "br"))
    for v in ("v4", "v6"):
        assert ix.BENIGN_ROLEPLAY not in ix.default_sources(tmp_path, "claude-opus-5-5", v)
    assert ix.default_sources(tmp_path, "claude-opus-5-5", "v7")[ix.BENIGN_ROLEPLAY] == [
        tmp_path / "br"
    ]


def test_cli_v7_and_br_root_refused_before_v7(tmp_path, monkeypatch):
    for var in ix.SAFETY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    br = _br_root(tmp_path / "br")
    src = _v7_sources(tmp_path, br)
    (tmp_path / "methods.json").write_text("{}")
    argv = ["index", "--out", str(tmp_path / "none"), "--methods", str(tmp_path / "methods.json")]
    for fam, flag in (
        (ix.MAIN, "--main-root"),
        (ix.AM, "--am-root"),
        (ix.STORY, "--story-root"),
        (ix.SIBLING, "--um-root"),
        (ix.OVERREFUSAL, "--overrefusal-root"),
    ):
        argv += [flag, str(src[fam][0])]
    with pytest.raises(SystemExit, match="hindex-v7 only"):
        cli.main([*argv, "--benign-roleplay-root", str(br), "--dest", str(tmp_path / "d6")])
    assert (
        cli.main(
            [
                *argv,
                "--version",
                "v7",
                "--benign-roleplay-root",
                str(br),
                "--dest",
                str(tmp_path / "d7"),
            ]
        )
        == 0
    )
    out = json.loads((tmp_path / "d7" / "index.json").read_text())
    assert out["version"] == "hindex-v7"
    assert any(ix.BENIGN_ROLEPLAY in r["families"] for r in out["rows"])


# ---------------------------------------------------------------- dev box: the final-olens run

DEVBOX_SUMMARY = paths.BR_FINAL_OLENS_SUMMARY


@pytest.mark.skipif(not DEVBOX_SUMMARY.is_file(), reason="dev-box final-olens summary absent")
def test_devbox_final_olens_export_equals_eval(tmp_path):
    meta = bx.export([DEVBOX_SUMMARY], tmp_path / "root", bank.load_items())
    assert meta["layers"] == GRID
    rows, _ = ix.safety_rows(tmp_path / "root", ix.BENIGN_ROLEPLAY, "v7")
    s = json.loads(DEVBOX_SUMMARY.read_text())
    for L in GRID:
        got, _ = ix.safety_metric(rows["final-olens-rl600-x4"][L])
        u = s["rates"]["final-olens-rl600-x4"]["units"][f"L{L}"]
        assert got["v"] == pytest.approx(u["false_recognition_rate"], abs=0, rel=0)
        assert got["ci"] == u["ci95"]
