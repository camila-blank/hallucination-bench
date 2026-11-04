"""Plan 0029 §3.9: ``hbench final`` -- the v8 manifest checks (sha256, completeness, one accepted
prompt version) and the plots, on synthetic roots."""

import json
from pathlib import Path

import pytest
from test_index_v8 import v8_sources

from hallucination_bench import cli, final
from hallucination_bench import index as ix

V28 = "claimmap-v2.8"


def _manifest(tmp: Path, *, version=V28, failed=0, mutate=None) -> Path:
    src = v8_sources(tmp / "roots")
    entries = []
    for fam, roots in src.items():
        root = roots[0].relative_to(tmp)
        for unit in ("L20", "L24", "all"):
            if fam in (ix.MAIN, ix.STORY, ix.AM_CLAIMS):
                file = f"jlens/{ix.CELLS_JOINT if unit == 'all' else ix.CELLS}"
            elif fam == ix.SIBLING:
                if unit == "all":
                    continue
                file = "label/units_labeled.jsonl"
            elif fam == ix.CHEATING:
                if unit == "all":
                    continue
                file = ix.SAFETY_COUNTS
            else:
                file = ix.SAFETY_COUNTS_JOINT if unit == "all" else ix.SAFETY_COUNTS
            kw = {
                "prompt_version": version if fam in final.CLAIM_MAP_FAMILIES else "n/a",
                "judge_model": "claude-opus-5-5",
                "n_cells": 400,
                "n_failed": failed,
            }
            entries.append(final.entry(fam, "jlens", unit, root, file, base=tmp, arm="jlens", **kw))
        if fam == ix.AM_CLAIMS:
            for unit in ("L20", "L24", "all"):
                entries.append(
                    final.entry(
                        ix.AM_SITUATION,
                        "jlens",
                        unit,
                        root,
                        f"jlens/{ix.SITUATION}",
                        base=tmp,
                        arm="jlens",
                        n_cells=400,
                        n_failed=0,
                    )
                )
    m = {"index_version": "hindex-v8", "max_failed_share": 0.01, "entries": entries}
    if mutate:
        mutate(m)
    p = tmp / "v8_manifest.json"
    p.write_text(json.dumps(m, indent=1))
    return p


def _main(tmp: Path, manifest: Path) -> int:
    return final.main(
        manifest, out=tmp / "final", plots=tmp / "plots", methods=_methods(tmp), base=tmp
    )


def _methods(tmp: Path) -> Path:
    p = tmp / "methods.json"
    p.write_text(json.dumps({"methods": {}, "exclude": [], "notes": {}}))
    return p


def test_final_builds_the_index_and_draws_the_plots(tmp_path):
    m = _manifest(tmp_path)
    assert _main(tmp_path, m) == 0
    built = json.loads((tmp_path / "final" / "index.json").read_text())
    assert built["version"] == "hindex-v8" and built["manifest"]["prompt_version"] == V28
    assert any(r["layer"] == "all" and "full_joint" in r for r in built["rows"])
    for name in ("index_per_layer.png", "families_per_method.png", "macro_all11.png"):
        p = tmp_path / "plots" / name
        assert p.is_file() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # only the manifest's arm dirs were staged
    stage = tmp_path / "final" / "_sources"
    assert sorted(p.name for p in (stage / ix.MAIN).iterdir()) == ["jlens"]


def test_sha_mismatch_exits_2(tmp_path, capsys):
    m = _manifest(tmp_path)
    cells = tmp_path / "roots" / "main" / "jlens" / ix.CELLS
    cells.write_text(cells.read_text() + "\n")
    assert _main(tmp_path, m) == 2
    err = capsys.readouterr().err
    assert "sha256 mismatch" in err and "main x jlens" in err
    assert not (tmp_path / "plots").exists()
    # through the CLI too (paths relative to the working directory)
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


def test_completeness_and_prompt_version_checks(tmp_path, capsys):
    assert _main(tmp_path / "a", _manifest(tmp_path / "a", failed=5)) == 2
    assert "the hole must be re-run" in capsys.readouterr().err
    assert _main(tmp_path / "b", _manifest(tmp_path / "b", failed=4)) == 0  # 1% exactly

    def mixed(m):
        next(e for e in m["entries"] if e["family"] == ix.STORY)["prompt_version"] = "claimmap-v2.7"

    assert _main(tmp_path / "c", _manifest(tmp_path / "c", mutate=mixed)) == 2
    assert "different prompt versions" in capsys.readouterr().err
    assert _main(tmp_path / "d", _manifest(tmp_path / "d", version="claimmap-v2.7")) == 2
    assert "not an accepted v8 version" in capsys.readouterr().err

    def gone(m):
        m["entries"][0]["file"] = "nope/cells.jsonl"

    assert _main(tmp_path / "e", _manifest(tmp_path / "e", mutate=gone)) == 2


def test_manifest_schema(tmp_path):
    p = tmp_path / "m.json"
    for doc, msg in (
        ({"index_version": "hindex-v7", "entries": [{}]}, "index_version"),
        ({"index_version": "hindex-v8", "entries": []}, "no entries"),
        ({"index_version": "hindex-v8", "entries": [{"family": "main"}]}, "misses"),
    ):
        p.write_text(json.dumps(doc))
        with pytest.raises(final.ManifestError, match=msg):
            final.load_manifest(p)


def test_filter_keeps_only_manifest_units():
    index = {
        "rows": [
            {"method": "m", "layer": 20, "families": {"main": {"v": 0.1}, "story": {"v": 0.2}}},
            {"method": "m", "layer": "all", "families": {"main": {"v": 0.3}}},
        ]
    }
    shown = final.filter_index(index, {("m", "main"): {20}})
    assert [(r["layer"], set(r["families"])) for r in shown["rows"]] == [(20, {"main"})]
