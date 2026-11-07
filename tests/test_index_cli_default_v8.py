"""Plan 0029 Amendment 4: `hbench index` defaults to hindex-v8 (was v6)."""

import json

from test_index_v8 import v8_sources

from hallucination_bench import cli
from hallucination_bench import index as ix


def test_hbench_index_defaults_to_v8(tmp_path, monkeypatch):
    for var in ix.SAFETY_ENV.values():
        monkeypatch.delenv(var, raising=False)
    assert cli.build_parser().parse_args(["index"]).version == "v8"
    src = v8_sources(tmp_path)
    argv = ["index", "--out", str(tmp_path / "none"), "--dest", str(tmp_path / "d")]
    argv += ["--workers", "1"]
    for fam, flag in (
        (ix.MAIN, "--main-root"),
        (ix.STORY, "--story-root"),
        (ix.AM_CLAIMS, "--am-root"),
        (ix.SIBLING, "--um-root"),
        (ix.OVERREFUSAL, "--overrefusal-root"),
        (ix.BENIGN_ROLEPLAY, "--benign-roleplay-root"),
    ):
        argv += [flag, str(src[fam][0])]
    assert cli.main(argv) == 0  # no --version
    built = json.loads((tmp_path / "d" / "index.json").read_text())
    assert built["version"] == "hindex-v8"
    assert {r["layer"] for r in built["rows"]} == {20, 24, "all"}
