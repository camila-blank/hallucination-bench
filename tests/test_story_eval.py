"""Plan 0007: ``--eval story`` runs the shared pipeline on the story bank, writes under
``<out>/story/<model>/``, is always pre-gate, and leaves ``wsbench_chat`` untouched."""

import ast
import json
from pathlib import Path

import pytest
from conftest import write_jsonl
from test_claimmap_v2 import MapFake, _run

from hallucination_bench import cli, report, report_v2
from hallucination_bench.evals import bank_for
from hallucination_bench.evals.story import bank as story_bank
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm

JUDGE = "claude-opus-5-5"
STORY = wrun.Models(eval="story")
TOY = ["hal-cre-000", "hal-cre-001"]  # neither is in the seeded 10% v5c subsample


@pytest.fixture
def story_readouts(tmp_path: Path) -> Path:
    items = {it["id"]: it for it in story_bank.load_items()}
    rows = []
    for iid in TOY:
        for n, s in enumerate(items[iid]["sites"]):
            samples = ["The story is set in Paris." if n == 0 else "A girl named Elara waits."]
            rows.append({"id": iid, "layer": 44, "pos": int(s["pos"]), "samples": samples})
    return write_jsonl(tmp_path / "story.jsonl", rows)


def _pass_chat_gate(out: Path) -> None:
    from hallucination_bench.calibrate_v2 import result_stem

    d = out / "calibration" / wrun.model_slug(JUDGE)
    d.mkdir(parents=True)
    (d / f"{result_stem('heldout')}.json").write_text(json.dumps({"gate_passed": True}))


def test_open_arm_uses_the_story_bank_and_output_root(tmp_path, story_readouts):
    arm = wrun.open_arm("toy", story_readouts, tmp_path, STORY)
    assert arm.bank is story_bank
    assert arm.dir == tmp_path / "story" / wrun.model_slug(JUDGE) / "toy"
    assert set(arm.items) == {it["id"] for it in story_bank.load_items()}
    assert len(arm.cells) == 16
    assert wrun.arm_dir(tmp_path, wrun.Models(), "toy") == (
        tmp_path / "wsbench_chat" / wrun.model_slug(JUDGE) / "toy"
    )
    assert wrun.open_arm("c", story_readouts, tmp_path, wrun.Models()).bank is bank_for(
        "wsbench_chat"
    )


def test_empty_cache_rebuild_records_the_story_bank(tmp_path, story_readouts):
    s = wrun.rebuild_arm("toy", story_readouts, tmp_path, STORY)
    assert s["config"]["eval"] == "story"
    assert s["config"]["bank_source"] == story_bank.BANK_SOURCE
    assert s["config"]["bank_commit"] == story_bank.BANK_COMMIT
    assert "eval" not in s["config"]["models"]


def test_story_map_run_is_pre_gate_even_after_a_chat_gate_pass(tmp_path, story_readouts):
    out = tmp_path / "out"
    _pass_chat_gate(out)
    fake = MapFake()
    _run([("toy", story_readouts)], out, fake, models=STORY, stages=["map"])
    assert fake.n("map") == 16
    d = wrun.arm_dir(out, STORY, "toy")
    arm_json = json.loads((d / "arm.json").read_text())
    assert arm_json["eval"] == "story"
    s1 = json.loads((d / "summary.json").read_text())
    assert s1["config"]["eval"] == "story" and "eval" not in s1["config"]["models"]
    s2 = json.loads((d / "summary_v2.json").read_text())
    assert s2["config"]["pre_gate"] is True
    # every :M prompt rendered a story item, never a chat one
    users = [c[3] for c in fake.calls if c[0] == "map"]
    assert all("<prompt>" in u for u in users)
    prompts = {it["prompt"] for it in story_bank.load_items() if it["id"] in TOY}
    assert all(any(p in u for p in prompts) for u in users)


def test_chat_gate_logic_is_unchanged(tmp_path, prose_readouts):
    out = tmp_path / "out"
    _pass_chat_gate(out)
    _run([("toy", prose_readouts)], out, MapFake(), stages=["map"])
    d = wrun.arm_dir(out, wrun.Models(), "toy")
    s2 = json.loads((d / "summary_v2.json").read_text())
    assert s2["config"]["pre_gate"] is False and s2["config"]["eval"] == "wsbench_chat"
    s1 = json.loads((d / "summary.json").read_text())
    assert "eval" not in s1["config"]["models"]
    assert json.loads((d / "arm.json").read_text())["eval"] == "wsbench_chat"


def test_story_reports_render_with_story_text(tmp_path, story_readouts):
    out = tmp_path / "out"
    _run([("toy", story_readouts)], out, MapFake(), models=STORY, stages=["map"])
    root = report_v2.report(out, JUDGE, audit_n=2, verdict_n=2, eval_name="story")
    assert root == out / "story" / wrun.model_slug(JUDGE)
    md = (root / "report_v2.md").read_text()
    assert md.startswith("# story claim map (taxonomy v2)")
    assert "held-out gate is chat gold; story has no gate of its own" in md
    assert "judge-noise sample is chat-only; not estimated for story" in md
    assert "consistency-v2` on a seeded" not in md
    audit = (root / "audit_samples_v2.md").read_text()
    toy_prompts = [it["prompt"] for it in story_bank.load_items() if it["id"] in TOY]
    assert any(json.dumps(p) in audit for p in toy_prompts)
    assert "chat-dailydialog" not in audit
    root1 = report.report(out, JUDGE, audit_n=1, eval_name="story")
    assert (root1 / "report.md").read_text().startswith("# story report")
    assert not (out / "wsbench_chat").exists()


def test_run_py_reads_the_bank_only_through_the_arm():
    # no module-global chat bank: open_arm resolves it locally, everything else uses arm.bank
    assert not hasattr(wrun, "bank") and not hasattr(wrun, "NAME")
    tree = ast.parse(Path(wrun.__file__).read_text())
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "bank":
                    assert fn.name == "open_arm", (fn.name, node.attr)
            if isinstance(node, ast.Attribute) and node.attr == "bank":
                assert isinstance(node.value, ast.Name) and node.value.id == "arm", fn.name


def test_cli_eval_flag():
    p = cli.build_parser()
    a = p.parse_args(["run", "--eval", "story", "--readouts-root", "x"])
    assert cli._models(a).eval == "story"
    assert cli._models(p.parse_args(["run", "--readouts-root", "x"])).eval == "wsbench_chat"
    assert p.parse_args(["report", "--eval", "story", "--view", "v2"]).eval == "story"
    with pytest.raises(SystemExit):
        p.parse_args(["run", "--eval", "nope"])
    with pytest.raises(SystemExit):
        p.parse_args(["consistency-v2", "--eval", "story"])


def test_claim_map_prompt_is_unchanged_for_story():
    assert cm.PROMPT_VERSION == "claimmap-v2.7"
