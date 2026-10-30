"""Plan 0024: the final O-lens campaign arm (final-olens-rl600-x4) on the index families."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from hallucination_bench import index
from hallucination_bench.evals.benign_roleplay import metrics, wsb
from hallucination_bench.evals.overrefusal import index_export

ARM = "final-olens-rl600-x4"
GRID = wsb.GRID
ROOT = Path(__file__).resolve().parents[1]


def test_overrefusal_allowlist_is_versioned() -> None:
    frozen = frozenset({"s3d-rl600", "mlnla-rl600-all11", "jlens"})
    assert index.SAFETY_READERS[index.OVERREFUSAL] == frozen  # v4 / v5 unchanged
    for v in ("v4", "v5"):
        assert index.safety_readers(index.OVERREFUSAL, v) == frozen
    assert index.safety_readers(index.OVERREFUSAL, "v6") == frozen | {ARM}
    assert index_export.READERS == ("s3d-rl600", "mlnla-rl600-all11", "jlens")
    assert index_export.EXPORT_READERS == (*index_export.READERS, ARM)


def _or_row(arm: str, layer: int, item: str, n: int, off: int) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(index.SAFETY_COLS, 0)
    row.update(arm=arm, layer=layer, item=item, cluster=item, n_claims=n, n_false=off,
               n_contradicted=off)  # fmt: skip
    return row


def test_safety_rows_read_the_arm_under_v6_only(tmp_path: Path) -> None:
    rows = [_or_row(ARM, 44, "i1", 4, 1), _or_row("jlens", 44, "i1", 2, 0)]
    (tmp_path / index.SAFETY_COUNTS).write_text("".join(json.dumps(r) + "\n" for r in rows))
    v6, ign6 = index.safety_rows(tmp_path, index.OVERREFUSAL, "v6")
    assert set(v6) == {ARM, "jlens"} and not ign6
    for v in ("v4", "v5"):
        got, ign = index.safety_rows(tmp_path, index.OVERREFUSAL, v)
        assert set(got) == {"jlens"} and ign == {ARM}
    assert index.safety_rows(tmp_path, index.OVERREFUSAL)[1] == {ARM}  # library default v4


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


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "final_olens" / f"{name}.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_export_um_arm_keeps_only_the_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exp = _load_script("export_um_arm")
    um, dest = tmp_path / "um", tmp_path / "dest"
    for d, p in index.um_paths(um).items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"uid": f"{d}-a", "method": "s3d"}) + "\n"
            + json.dumps({"uid": f"{d}-b", "method": ARM, "x": "ü"}, ensure_ascii=False) + "\n"
        )  # fmt: skip
    monkeypatch.setattr("sys.argv", ["x", "--arm", ARM, "--um", str(um), "--dest", str(dest)])
    exp.main()
    for d, p in index.um_paths(dest).items():
        got = [json.loads(x) for x in p.read_text().splitlines()]
        assert got == [{"uid": f"{d}-b", "method": ARM, "x": "ü"}]
