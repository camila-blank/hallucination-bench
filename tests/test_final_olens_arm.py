"""Plan 0024: the final O-lens campaign arm (final-olens-rl600-x4) on the index families."""

import json
from pathlib import Path
from typing import Any

import pytest

from hallucination_bench.evals.benign_roleplay import metrics, wsb

ARM = "final-olens-rl600-x4"
GRID = wsb.GRID
ROOT = Path(__file__).resolve().parents[1]


def _result(ids: dict[str, tuple[str, bool]], unit: str) -> dict[str, Any]:
    rows = [
        {"id": i, "kind": k, "any_recognition": hit, "state": "judged", "best_label": "topic",
         "quotes": ["q"] if hit else [], "flagged": hit}
        for i, (k, hit) in ids.items()
    ]  # fmt: skip
    ctl = [r for r in rows if r["kind"] == "control"]
    config = {
        "reads": "full",
        "reads_checked": "sidecar",
        "sampling_source": "declared",
        "layers_judged": GRID if unit == "all" else [int(unit[1:])],
        "prompt_version": "jb-v2",
        "judge_model": "claude-sonnet-5",
    }
    fpr = sum(r["flagged"] for r in ctl) / len(ctl)
    return {"complete": True, "config": config,
            "numbers": {"extras": {"item_rows": rows, "false_recognition_rate": fpr}}}  # fmt: skip


def _write(p: Path, res: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))


def test_benign_roleplay_nested_campaign_arm(tmp_path: Path) -> None:
    camp, out = tmp_path / "judged", tmp_path / "hb"
    shared = {"c1": ("control", True), "c2": ("control", False), "j1": ("jailbreak", True)}
    for unit in ["all", *[f"L{k}" for k in GRID]]:
        assert wsb.campaign_dir(camp, ARM, unit) == camp / ARM / unit
        _write(wsb.result_path(camp / ARM / unit), _result(shared, unit))
    calls = wsb.plan_calls(campaign=camp, readouts=tmp_path / "ro", out=out, arms=[ARM])
    assert [str(c.out.relative_to(out)) for c in calls if c.out] == [f"{ARM}/hb_only"]
    hb = {"h1": ("control", False)}
    for tag in ["all", *[f"L{k}" for k in GRID]]:
        _write(wsb.result_path(out / ARM / "hb_only", tag), _result(hb, tag))
    items = {
        "kind_mix": {"twin": 1, "roleplay": 2},
        "items": [
            {"id": "c1", "hb_only": False, "template_id": "t1", "control_type": "twin"},
            {"id": "c2", "hb_only": False, "template_id": "t2", "control_type": "roleplay"},
            {"id": "h1", "hb_only": True, "template_id": "t3", "control_type": "roleplay"},
        ],
    }
    s = metrics.summarize(items, out=out, campaign=camp, arms=[ARM], layers=None)
    assert s["complete"]
    assert s["rates"][ARM]["units"]["L44"]["false_recognition_rate"] == 1 / 3  # c1 of 3
    (wsb.result_path(camp / ARM / "L44")).unlink()
    with pytest.raises(wsb.WsbError, match="campaign config missing"):
        wsb.plan_calls(campaign=camp, readouts=tmp_path / "ro", out=out, arms=[ARM])
    assert ARM in wsb.KNOWN_ARMS and ARM not in wsb.ARMS  # opt-in, default runs unchanged
