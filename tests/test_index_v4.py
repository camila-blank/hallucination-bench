"""Plan 0019: hindex-v4 (the safety families) and the hindex-v3 golden outputs."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hallucination_bench import REPO_ROOT, cli
from hallucination_bench import index as ix

FIX = REPO_ROOT / "tests" / "fixtures"
V3_FIX = FIX / "index_v3"
VOLATILE = ("git_sha", "sources")  # checkout-specific: excluded from the golden comparison


def _canon(index: dict[str, Any]) -> str:
    """Golden comparison text (NaN CIs compare as text, not as floats)."""
    return json.dumps({k: v for k, v in index.items() if k not in VOLATILE}, sort_keys=True)


# ---------------------------------------------------------------- v3 goldens


def test_v3_synthetic_golden_reproduced_exactly():
    sources = {
        ix.MAIN: [V3_FIX / "main"],
        ix.AM: [V3_FIX / "am"],
        ix.STORY: [V3_FIX / "story"],
        ix.SIBLING: [V3_FIX / "um"],
        # v3 never reads a safety root
        ix.OVERREFUSAL: [V3_FIX / "nonexistent"],
    }
    built = ix.build_index(sources, ix.load_methods(V3_FIX / "methods.json"), version="v3")
    golden = json.loads((V3_FIX / "golden" / "index.json").read_text())
    assert built["version"] == "hindex-v3"
    assert _canon(built) == _canon(golden)
    assert ix.render_md(built) == (V3_FIX / "golden" / "index.md").read_text()


def _pinned_roots(golden: dict[str, Any], tmp: Path) -> dict[str, list[Path]]:
    """Mirrors of the golden's source roots holding only its arms (claim families: symlinked arm
    dirs; sibling: units files filtered to its methods), so arms added later are ignored."""
    arms = {f: {a for m in golden["methods"].values() for a in m[f]} for f in ix.FAMILIES_V3}
    out: dict[str, list[Path]] = {}
    for f, roots in golden["sources"].items():
        out[f] = []
        for k, root in enumerate(map(Path, roots)):
            mirror = tmp / f / str(k)
            mirror.mkdir(parents=True)
            out[f].append(mirror)
            if f == ix.SIBLING:
                for dom, src in ix.um_paths(root).items():
                    if not src.exists():
                        continue
                    dst = ix.um_paths(mirror)[dom]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    keep = [
                        ln
                        for ln in src.read_text(encoding="utf-8").splitlines(keepends=True)
                        if json.loads(ln)["method"] in arms[f]
                    ]
                    dst.write_text("".join(keep), encoding="utf-8")
                continue
            for a in sorted(arms[f]):
                if (root / a / ix.CELLS).is_file():
                    (mirror / a).symlink_to(root / a)
    return out


def test_v3_devbox_golden_reproduced_exactly(tmp_path):
    golden = json.loads((FIX / "index_v3_devbox.json").read_text())
    roots = [Path(r) for rs in golden["sources"].values() for r in rs]
    if missing := [r for r in roots if not r.is_dir()]:
        pytest.skip(f"dev-box sources absent: {missing[:2]}")
    methods = ix.Methods(
        methods={m: {f: list(a) for f, a in fams.items()} for m, fams in golden["methods"].items()},
        exclude=set(golden["exclude"]),
    )
    built = ix.build_index(_pinned_roots(golden, tmp_path), methods, workers=8, version="v3")
    assert _canon(built) == _canon(golden)
    assert ix.render_md(built) == (FIX / "index_v3_devbox.md").read_text()


def test_v3_notes_are_the_frozen_methods_notes():
    golden = json.loads((V3_FIX / "golden" / "index.json").read_text())
    assert golden["notes"] == ix.NOTES_V3
    built = ix.build_index({}, ix.Methods(notes={"new": "v4 note"}), version="v3")
    assert built["notes"] == ix.NOTES_V3
    assert ix.build_index({}, ix.Methods(notes={"new": "v4 note"}))["notes"] == {"new": "v4 note"}


# ---------------------------------------------------------------- v4 fixtures


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


def test_textreader_never_becomes_a_method(tmp_path):
    root = _write_safety(
        tmp_path / "or", [_row("textreader-x", 20, "i1", n_claims=2, n_false=1, n_contradicted=1)]
    )
    out = ix.build_index({ix.OVERREFUSAL: [root]}, ix.Methods())
    assert out["rows"] == [] and "textreader-x" not in out["methods"]
    assert out["safety"][ix.OVERREFUSAL]["ignored_arms"] == ["textreader-x"]


# a 5-family (method, layer): claim cells, UM units and overrefusal rows all named "jlens"


def _cells(root: Path, arm: str, cells: list[dict]) -> None:
    (root / arm).mkdir(parents=True)
    (root / arm / ix.CELLS).write_text("".join(json.dumps(c) + "\n" for c in cells))


def _chat(item: str, leaves: list[str], am: bool = False) -> dict[str, Any]:
    segs = {"segments": [{"readout": 0, "leaf": lf} for lf in leaves]}
    st: dict[str, Any] = {"map_status": "ok"} if am else {"stages": {"map": "ok"}}
    return {"id": item, "layer": 20, "map": segs, **st}


def _sources(tmp: Path, sibling: bool = True, or_rows: list[dict] | None = None) -> dict:
    _cells(tmp / "main", "jlens", [_chat("i1", ["false_claim", "correct"])])  # 1/2
    _cells(tmp / "am", "jlens", [_chat("i1", ["detail", "correct", "correct", "correct"], True)])
    _cells(tmp / "story", "jlens", [_chat("i1", ["correct"])])  # 0
    src: dict[str, list[Path]] = {
        ix.MAIN: [tmp / "main"],
        ix.AM: [tmp / "am"],
        ix.STORY: [tmp / "story"],
    }
    if sibling:  # one hallucination unit per domain of two -> 1/2 in every domain
        for d, p in ix.um_paths(tmp / "um").items():
            p.parent.mkdir(parents=True, exist_ok=True)
            us = [
                {
                    "method": "jlens",
                    "item": f"{d}{k}",
                    "layer": 20,
                    "committed": True,
                    "probe": {"na_reason": None, "fire_L": k == 0},
                    "cross": [],
                }
                for k in range(2)
            ]
            p.write_text("".join(json.dumps(u) + "\n" for u in us))
        src[ix.SIBLING] = [tmp / "um"]
    meta = {
        "gates": {"20": {"tail0": True, "content": False}},
        "population": "never_refused",
        "unit": "items",
    }
    # plan 0020's rows: one verdict per item (n_claims = judged), unit columns 0; o1 a
    # contradicted false refusal claim, o2 an unprobed one, o3 / o4 no false claim -> 1/4
    rows = or_rows or [
        _row("jlens", 20, "o1", n_claims=1, n_false=1, n_contradicted=1),
        _row("jlens", 20, "o2", n_claims=1, n_false=1, n_unprobed=1),
        _row("jlens", 20, "o3", n_claims=1),
        _row("jlens", 20, "o4", n_claims=1),
    ]
    src[ix.OVERREFUSAL] = [_write_safety(tmp / "or", rows, meta)]
    return src


def test_full_and_matched_over_five_families(tmp_path):
    (row,) = ix.build_index(_sources(tmp_path), ix.Methods())["rows"]
    per = row["families"]
    assert set(per) == set(ix.FAMILIES)
    vals = {
        ix.MAIN: 0.5,
        ix.AM: 0.25,
        ix.SIBLING: 0.5,
        ix.STORY: 0.0,
        ix.OVERREFUSAL: 0.25,
    }
    assert {f: per[f]["v"] for f in ix.FAMILIES} == pytest.approx(vals)
    w = {**ix.FAMILY_WEIGHTS_V3, ix.OVERREFUSAL: 1.0}
    assert w == ix.FAMILY_WEIGHTS

    def mean(fams: tuple[str, ...]) -> float:
        return sum(w[f] * vals[f] for f in fams) / sum(w[f] for f in fams)

    assert row["full"]["v"] == pytest.approx(mean(ix.FAMILIES))  # (4.5+.75+1+0+.25) / 16
    assert row["full"]["v"] == pytest.approx(6.5 / 16)
    assert row["matched"]["v"] == pytest.approx(5.5 / 14)  # sibling excluded
    assert ix.SIBLING not in ix.MATCHED and len(ix.MATCHED) == 4
    assert row["full_v3"]["v"] == pytest.approx(mean(ix.FAMILIES_V3))
    assert row["matched_v3"]["v"] == pytest.approx(mean(ix.MATCHED_V3))
    # safety extras: raw share, counts, gates from index_meta.json
    o = per[ix.OVERREFUSAL]
    assert (o["raw"], o["n_claims"], o["n_items"], o["n_unprobed"]) == (0.5, 4, 4, 1)
    assert o["gates"] == {"tail0": True, "content": False}
    out = ix.build_index(_sources(tmp_path / "again"), ix.Methods())
    assert out["version"] == "hindex-v4"
    (m,) = out["safety"][ix.OVERREFUSAL]["meta"]
    assert m["meta"]["population"] == "never_refused"
    md = ix.render_md(out)
    assert "| method | layer | full | matched | full_v3 | main |" in md
    assert "| story | overrefusal | main fc" in md
    assert "mask" not in md and "strongreject" not in md
    assert ix.FAMILIES == (*ix.FAMILIES_V3, ix.OVERREFUSAL)
    assert out["safety"][ix.OVERREFUSAL]["unit"] == ["items"]


def test_missing_family_drops_full_but_matched_survives_sibling(tmp_path):
    (row,) = ix.build_index(_sources(tmp_path, sibling=False), ix.Methods())["rows"]
    assert "full" not in row and "full_v3" not in row
    assert "matched" in row and "matched_v3" in row


def test_absent_safety_family_listed_and_drops_macros(tmp_path):
    unprobed = [_row("jlens", 20, "o1", n_claims=1, n_false=1, n_unprobed=1)]
    out = ix.build_index(_sources(tmp_path, or_rows=unprobed), ix.Methods())
    (row,) = out["rows"]
    assert ix.OVERREFUSAL not in row["families"]
    gone = {"arm": "jlens", "absent_reason": ix.ABSENT_ALL_UNPROBED}
    assert row["absent"] == {ix.OVERREFUSAL: gone}
    assert "full" not in row and "matched" not in row
    assert "full_v3" in row and "matched_v3" in row  # the v3 macros don't need safety families
    (a,) = out["safety"][ix.OVERREFUSAL]["absent"]
    assert (a["arm"], a["layer"], a["absent_reason"], a["n_claims"]) == (
        "jlens",
        20,
        ix.ABSENT_ALL_UNPROBED,
        1,
    )
    assert "jlens.overrefusal@20 (no_probed_false_claims)" in ix.render_md(out)


def test_safety_arm_under_two_roots(tmp_path):
    a = _write_safety(tmp_path / "a", [_row("jlens", 20, "i1", n_claims=1)])
    b = _write_safety(tmp_path / "b", [_row("jlens", 24, "i1", n_claims=1)])
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


def test_safety_roots_are_opt_in(tmp_path, monkeypatch):
    for var in ix.SAFETY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    src = ix.default_sources(tmp_path, cli.DEFAULT_MODEL)
    assert set(src) == set(ix.FAMILIES_V3)  # the dev-box defaults are never used
    monkeypatch.setenv("HBENCH_SAFETY_SR_OUT", str(tmp_path / "sr"))  # not on the index
    monkeypatch.setenv("HBENCH_SAFETY_MASK_OUT", str(tmp_path / "mask"))
    assert set(ix.default_sources(tmp_path, cli.DEFAULT_MODEL)) == set(ix.FAMILIES_V3)
    monkeypatch.setenv("HBENCH_SAFETY_OR_OUT", str(tmp_path / "or"))
    src = ix.default_sources(tmp_path, cli.DEFAULT_MODEL)
    assert src[ix.OVERREFUSAL] == [tmp_path / "or"] and set(src) == set(ix.FAMILIES)


def test_cli_version_and_safety_roots(tmp_path, monkeypatch):
    for var in ix.SAFETY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "methods.json").write_text("{}")
    root = _write_safety(
        tmp_path / "or", [_row("jlens", 20, "i1", n_claims=2, n_false=1, n_contradicted=1)]
    )
    base = ["index", "--out", str(tmp_path / "none"), "--methods", str(tmp_path / "methods.json")]
    assert cli.main([*base, "--overrefusal-root", str(root), "--dest", str(tmp_path / "d4")]) == 0
    out = json.loads((tmp_path / "d4" / "index.json").read_text())
    assert out["version"] == "hindex-v4"
    assert out["rows"][0]["families"][ix.OVERREFUSAL]["v"] == 0.5
    assert cli.main([*base, "--version", "v3", "--dest", str(tmp_path / "d3")]) == 0
    assert json.loads((tmp_path / "d3" / "index.json").read_text())["version"] == "hindex-v3"
    with pytest.raises(SystemExit, match="hindex-v4 only"):
        cli.main([*base, "--version", "v3", "--overrefusal-root", str(root)])
    with pytest.raises(SystemExit, match="not a directory"):
        cli.main([*base, "--overrefusal-root", str(tmp_path / "missing")])
    (tmp_path / "empty").mkdir()  # an opted-in root without index_counts.jsonl is an error
    with pytest.raises(SystemExit, match=r"no index_counts\.jsonl"):
        cli.main([*base, "--overrefusal-root", str(tmp_path / "empty")])
    monkeypatch.setenv("HBENCH_SAFETY_OR_OUT", str(tmp_path / "empty"))
    with pytest.raises(SystemExit, match=r"no index_counts\.jsonl"):
        cli.main(base)
    assert cli.main([*base, "--version", "v3", "--dest", str(tmp_path / "d3b")]) == 0
    monkeypatch.delenv("HBENCH_SAFETY_OR_OUT")
    for flag in ("--strongreject-root", "--mask-root"):  # not on the index (revs 6 / 5)
        with pytest.raises(SystemExit):
            cli.main([*base, flag, str(root)])
