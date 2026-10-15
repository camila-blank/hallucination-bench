"""Pilot fixes (plan 0009 step 0/2a): the read generator needs its ::main entrypoint, and
collect must ignore the patchscopes ``_work/`` shard stores."""

import json
from pathlib import Path

from hallucination_bench.evals.sibling_gate import resample as R


def test_read_job_names_the_main_entrypoint():
    pool = R.POOLS["s3d-rl600"]
    line = next(x for x in R.job_lines(pool, [44], Path("d"), []) if "wsbench_read_modal" in x)
    assert "wsbench_read_modal.py::main " in line


def test_iter_gen_rows_skips_work_dirs(tmp_path):
    (tmp_path / "chat-a").mkdir()
    (tmp_path / "chat-a" / "L044.jsonl").write_text(
        json.dumps({"label": "chat-a", "layer": 44, "pos": 3, "samples": ["x"]}) + "\n"
    )
    (tmp_path / "_work").mkdir()
    (tmp_path / "_work" / "L044_shard0.jsonl").write_text(json.dumps({"key": "k"}) + "\n")
    rows = list(R.iter_gen_rows(tmp_path))
    assert [r["id"] for r in rows] == ["chat-a"]


def test_iter_gen_rows_finds_nested_run_dirs(tmp_path):
    d = tmp_path / "gen-run" / "chat-a"
    d.mkdir(parents=True)
    (d / "L044.jsonl").write_text(
        json.dumps({"label": "chat-a", "layer": 44, "pos": 3, "samples": ["x"]}) + "\n"
    )
    (tmp_path / "gen-run" / "_work").mkdir()
    (tmp_path / "gen-run" / "_work" / "L044_s0.jsonl").write_text(json.dumps({"k": 1}) + "\n")
    assert [r["id"] for r in R.iter_gen_rows(tmp_path)] == ["chat-a"]


def test_l42_acts_dir_is_remapped():
    p = R._vol_path("/acts/acts-jlens-pr-L42/chat-a.safetensors", "x")
    assert str(p) == "/acts-hallucination-chat-L42cap/chat-a.safetensors"


def test_dedupe_apply_merges_only_renamed_and_chains() -> None:
    from hallucination_bench.evals.sibling_gate.classes import apply_merges

    names = ["a", "b", "c", "d"]
    out = apply_merges(names, [{"name": "a", "canonical": "b"}, {"name": "b", "canonical": "c"}])
    assert out == {"a": "c", "b": "c", "c": "c", "d": "d"}
    assert apply_merges(["x"], [{"name": "x", "canonical": "x"}]) == {"x": "x"}
