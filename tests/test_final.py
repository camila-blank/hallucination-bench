"""Plan 0029 §3.9: ``hbench final`` -- the v8 manifest checks (sha256, recounted completeness,
the unit grid, one accepted prompt version backed by each arm's ``arm.json``), staging exactly
the manifest's files, and the plots, on synthetic roots."""

import json
from pathlib import Path

import pytest
from test_index_v8 import _chat_cell, _w, v8_sources

from hallucination_bench import cli, final
from hallucination_bench import index as ix

V28 = "claimmap-v2.8"
V29 = "claimmap-v2.9"  # plan 0029 amendment 1: the only version `hbench final` accepts
CLAIM_DIRS = {ix.MAIN: "main", ix.STORY: "story", ix.AM_CLAIMS: "am"}


@pytest.fixture(autouse=True)
def _two_layer_grid(monkeypatch):
    monkeypatch.setattr(final, "EXPECTED_LAYERS", (20, 24))


def _arm_json(d: Path, prompt: str = V29, bank: dict | None = None) -> None:
    (d / "arm.json").write_text(json.dumps({"arm": d.name, "map_prompt": prompt, **(bank or {})}))


X3_RECORD = {"bank": "x3", "bank_source": "plan0028-x3:/x3/items.json", "bank_sha256": "0" * 64}


def _files(fam: str, unit: str) -> list[str]:
    if fam in CLAIM_DIRS:
        return [f"jlens/{ix.CELLS_JOINT if unit == 'all' else ix.CELLS}"]
    if fam == ix.AM_SITUATION:
        return [f"jlens/{ix.SITUATION}"]
    if fam == ix.SIBLING:
        return ["label/units_labeled.jsonl"] + [
            f"{d}/label/units_labeled.jsonl" for d in ix.UM_DOMAINS if d != "country"
        ]
    return [ix.SAFETY_COUNTS_JOINT if unit == "all" else ix.SAFETY_COUNTS]


def _entries(tmp: Path, src: dict, version=V29, *, extra_arm: str | None = None) -> list[dict]:
    srcs = {**src, ix.AM_SITUATION: src[ix.AM_CLAIMS]}
    entries = []
    for fam, roots in srcs.items():
        root = roots[0].relative_to(tmp)
        units = ["L20", "L24"] + (["all"] if fam in ix.JOINT_FAMILIES_V8 else [])
        for unit in units:
            for file in _files(fam, unit):
                v = version if fam in final.CLAIM_MAP_FAMILIES else "n/a"
                entries.append(
                    final.entry(
                        fam,
                        "jlens",
                        unit,
                        root,
                        file,
                        base=tmp,
                        arm="jlens",
                        prompt_version=v,
                        judge_model="claude-opus-5-5",
                    )
                )
    return entries


def _manifest(tmp: Path, *, version=V29, recorded=V29, mutate=None, prep=None) -> Path:
    src = v8_sources(tmp / "roots")
    for sub in CLAIM_DIRS.values():
        _arm_json(tmp / "roots" / sub / "jlens", recorded, X3_RECORD if sub == "main" else None)
    if prep:
        prep(tmp / "roots")
    m = {
        "index_version": "hindex-v8",
        "max_failed_share": 0.01,
        "entries": _entries(tmp, src, version),
    }
    if mutate:
        mutate(m)
    p = tmp / "v8_manifest.json"
    p.write_text(json.dumps(m, indent=1))
    return p


def _methods(tmp: Path) -> Path:
    p = tmp / "methods.json"
    p.write_text(json.dumps({"methods": {}, "exclude": [], "notes": {}}))
    return p


def _main(tmp: Path, manifest: Path) -> int:
    return final.main(
        manifest, out=tmp / "final", plots=tmp / "plots", methods=_methods(tmp), base=tmp
    )


def test_final_builds_the_index_and_draws_the_plots(tmp_path):
    m = _manifest(tmp_path)
    assert _main(tmp_path, m) == 0
    built = json.loads((tmp_path / "final" / "index.json").read_text())
    assert built["version"] == "hindex-v8" and built["manifest"]["prompt_version"] == V29
    # the note names the checked version (generated, not hard-coded)
    assert "Claim maps judged with claimmap-v2.9" in built["notes"]["hindex-v8"]
    rows = {r["layer"]: r for r in built["rows"] if r["method"] == "jlens"}
    assert set(rows) == {20, 24, "all"} and "full_joint" in rows["all"]
    assert set(rows[20]["families"]) == set(ix.FAMILIES_V8) and "full" in rows[20]
    for name in ("index_per_layer.png", "families_per_method.png", "macro_all11.png"):
        p = tmp_path / "plots" / name
        assert p.is_file() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # sources point at the manifest roots; nothing staged is left behind
    assert built["sources"][ix.MAIN] == [str((tmp_path / "roots" / "main").resolve())]
    assert not (tmp_path / "final" / "_sources").exists()
    assert "hbench-final-" not in (tmp_path / "final" / "index.json").read_text()
    assert _main(tmp_path, m) == 0  # a second run stages afresh


def test_a_dropped_arm_leaves_no_trace(tmp_path):
    def ghost(roots: Path) -> None:
        for sub in ("main", "story", "am"):
            g = roots / sub / "ghost-arm"
            g.mkdir()
            for f in (roots / sub / "jlens").iterdir():
                (g / f.name).write_bytes(f.read_bytes())
        rows = [json.loads(x) for x in (roots / "or" / ix.SAFETY_COUNTS).read_text().splitlines()]
        assert rows
        # the ghost is a reader the index would accept: only the manifest keeps it out

    m = _manifest(tmp_path, prep=ghost)
    assert _main(tmp_path, m) == 0
    for p in (tmp_path / "final" / "index.json", tmp_path / "final" / "index.md"):
        assert "ghost-arm" not in p.read_text()


def test_joint_files_are_staged_only_when_an_all_entry_hashes_them(tmp_path):
    src = v8_sources(tmp_path / "roots")
    entries = [e for e in _entries(tmp_path, src) if e["unit"] != "all"]
    stage = tmp_path / "stage"
    sources, back = final.stage_sources({"entries": entries}, tmp_path, stage)
    staged = {p.name for p in stage.rglob("*") if p.is_symlink()}
    assert ix.CELLS in staged and ix.CELLS_JOINT not in staged
    assert ix.SAFETY_COUNTS_JOINT not in staged and ix.SAFETY_META in staged
    built = ix.build_index_v8(sources, ix.Methods())
    assert all(r["layer"] != "all" for r in built["rows"])  # no joint data leaked in
    assert set(back.values()) >= {str((tmp_path / "roots" / "main").resolve())}


def test_sha_mismatch_exits_2(tmp_path, capsys):
    m = _manifest(tmp_path)
    cells = tmp_path / "roots" / "main" / "jlens" / ix.CELLS
    cells.write_text(cells.read_text() + "\n")
    assert _main(tmp_path, m) == 2
    err = capsys.readouterr().err
    assert "sha256 mismatch" in err and "main x jlens" in err
    assert not (tmp_path / "plots").exists()
    rc = cli.main(
        [
            "final",
            "--manifest",
            str(m),
            "--out",
            str(tmp_path / "f2"),
            "--plots",
            str(tmp_path / "p2"),
            "--methods",
            str(_methods(tmp_path)),
        ]
    )
    assert rc == 2


def test_counts_are_recomputed_from_the_hashed_file(tmp_path, capsys):
    def fail_one(roots: Path) -> None:
        cells = [_chat_cell(i, L, ["false_claim"]) for L in (20, 24) for i in ("i0", "i1")]
        cells[0]["stages"]["map"] = "failed"
        _w(roots / "main" / "jlens" / ix.CELLS, cells)

    assert _main(tmp_path / "a", _manifest(tmp_path / "a", prep=fail_one)) == 2
    assert "1 of 2 cells failed" in capsys.readouterr().err

    def lie(m):  # a declared count must agree with the recount
        m["entries"][0]["n_cells"] = 999

    assert _main(tmp_path / "b", _manifest(tmp_path / "b", mutate=lie)) == 2
    assert "declared n_cells 999" in capsys.readouterr().err

    def honest(m):
        m["entries"][0]["n_cells"], m["entries"][0]["n_failed"] = 4, 0

    assert _main(tmp_path / "c", _manifest(tmp_path / "c", mutate=honest)) == 0

    def stray_layer(roots: Path) -> None:
        p = roots / "story" / "jlens" / ix.CELLS
        p.write_text(p.read_text() + json.dumps(_chat_cell("i0", 42, ["correct"])) + "\n")

    assert _main(tmp_path / "d", _manifest(tmp_path / "d", prep=stray_layer)) == 2
    assert "outside the grid" in capsys.readouterr().err


def test_the_unit_grid_must_be_complete(tmp_path, capsys):
    def drop(m):
        m["entries"] = [
            e
            for e in m["entries"]
            if not (e["family"] == ix.STORY and e["unit"] == "all")
            and not (e["family"] == ix.CHEATING and e["unit"] == "L24")
        ]

    assert _main(tmp_path, _manifest(tmp_path, mutate=drop)) == 2
    err = capsys.readouterr().err
    assert "incomplete unit grid" in err
    assert "story x jlens: ['all']" in err and "cheating x jlens: ['L24']" in err


def test_prompt_versions_declared_and_recorded(tmp_path, capsys):
    def mixed(m):
        next(e for e in m["entries"] if e["family"] == ix.STORY)["prompt_version"] = "claimmap-v2.7"

    assert _main(tmp_path / "a", _manifest(tmp_path / "a", mutate=mixed)) == 2
    assert "different prompt versions" in capsys.readouterr().err
    assert (
        _main(
            tmp_path / "b",
            _manifest(tmp_path / "b", version="claimmap-v2.7", recorded="claimmap-v2.7"),
        )
        == 2
    )
    assert "not an accepted v8 version" in capsys.readouterr().err
    # the manifest says v2.9 but the arm dirs were judged with v2.7
    assert _main(tmp_path / "c", _manifest(tmp_path / "c", recorded="claimmap-v2.7")) == 2
    assert "records map_prompt 'claimmap-v2.7'" in capsys.readouterr().err

    def am3(m):  # an AM entry may declare its AM version
        for e in m["entries"]:
            if e["family"] == ix.AM_CLAIMS:
                e["prompt_version"] = "claimmap-v2.9+am3"

    assert _main(tmp_path / "d", _manifest(tmp_path / "d", mutate=am3)) == 0
    assert final.chat_version("claimmap-v2.9+am3") == final.chat_version(V29) == V29
    assert final.chat_version("claimmap-v2.8+am3") == final.chat_version(V28) == V28

    def no_arm_json(roots: Path) -> None:
        (roots / "am" / "jlens" / "arm.json").unlink()

    assert _main(tmp_path / "e", _manifest(tmp_path / "e", prep=no_arm_json)) == 2
    assert "no arm.json" in capsys.readouterr().err


def test_final_accepts_only_v29(tmp_path, capsys):
    """Plan 0029 amendment 1: ``V8_FINAL_PROMPTS`` = {claimmap-v2.9}; a v2.8 manifest and a
    mixed v2.8 / v2.9 one (declared or recorded) are refused."""
    assert final.V8_FINAL_PROMPTS == {V29}
    m9 = _manifest(tmp_path / "ok")
    assert final.check(final.load_manifest(m9), tmp_path / "ok") == V29
    # a consistent v2.8 manifest (declared and recorded): refused
    m8 = _manifest(tmp_path / "v28", version=V28, recorded=V28)
    with pytest.raises(final.ManifestError, match="not an accepted v8 version"):
        final.check(final.load_manifest(m8), tmp_path / "v28")
    assert _main(tmp_path / "v28", m8) == 2
    assert "'claimmap-v2.8' is not an accepted v8 version" in capsys.readouterr().err

    def mixed(m):  # story declared v2.8, the rest v2.9
        for e in m["entries"]:
            if e["family"] == ix.STORY:
                e["prompt_version"] = V28

    mm = _manifest(tmp_path / "mixed", mutate=mixed)
    with pytest.raises(final.ManifestError, match="different prompt versions"):
        final.check(final.load_manifest(mm), tmp_path / "mixed")
    assert _main(tmp_path / "mixed", mm) == 2
    assert "different prompt versions" in capsys.readouterr().err

    def am_v28(m):  # the AM entries declare the v2.8 AM version
        for e in m["entries"]:
            if e["family"] == ix.AM_CLAIMS:
                e["prompt_version"] = "claimmap-v2.8+am3"

    ma = _manifest(tmp_path / "mixed_am", mutate=am_v28)
    with pytest.raises(final.ManifestError, match="different prompt versions"):
        final.check(final.load_manifest(ma), tmp_path / "mixed_am")

    def story_v28(roots: Path) -> None:  # declared v2.9 everywhere, one arm judged with v2.8
        _arm_json(roots / "story" / "jlens", V28)

    mr = _manifest(tmp_path / "mixed_rec", prep=story_v28)
    with pytest.raises(final.ManifestError, match=r"records map_prompt 'claimmap-v2\.8'"):
        final.check(final.load_manifest(mr), tmp_path / "mixed_rec")


def test_index_v8_note_names_the_prompt_version():
    assert "claimmap-v2.9" in ix.notes_v8("claimmap-v2.9")["hindex-v8"]
    assert "claimmap-v2.9" not in ix.notes_v8("claimmap-v2.8")["hindex-v8"]
    assert "not checked" in ix.notes_v8(None)["hindex-v8"]
    with pytest.raises(ValueError, match="no hindex-v8 note"):
        ix.notes_v8("claimmap-v2.7")


def test_manifest_schema(tmp_path):
    p = tmp_path / "m.json"
    good = {
        "family": "main",
        "method": "m",
        "unit": "L20",
        "root": "r",
        "file": "f",
        "sha256": "x",
        "prompt_version": V28,
        "judge_model": "j",
    }
    for doc, msg in (
        ({"index_version": "hindex-v7", "entries": [good]}, "index_version"),
        ({"index_version": "hindex-v8", "entries": []}, "no entries"),
        ({"index_version": "hindex-v8", "entries": [{"family": "main"}]}, "misses"),
        (
            {
                "index_version": "hindex-v8",
                "entries": [{k: v for k, v in good.items() if k != "judge_model"}],
            },
            "judge_model",
        ),
        (
            {
                "index_version": "hindex-v8",
                "entries": [{k: v for k, v in good.items() if k != "prompt_version"}],
            },
            "prompt_version",
        ),
    ):
        p.write_text(json.dumps(doc))
        with pytest.raises(final.ManifestError, match=msg):
            final.load_manifest(p)
    p.write_text(json.dumps({"index_version": "hindex-v8", "entries": [good]}))
    assert final.load_manifest(p)["entries"][0]["judge_model"] == "j"


def test_filter_keeps_only_manifest_units():
    index = {
        "rows": [
            {"method": "m", "layer": 20, "families": {"main": {"v": 0.1}, "story": {"v": 0.2}}},
            {"method": "m", "layer": "all", "families": {"main": {"v": 0.3}}},
        ]
    }
    shown = final.filter_index(index, {("m", "main"): {20}})
    assert [(r["layer"], set(r["families"])) for r in shown["rows"]] == [(20, {"main"})]
