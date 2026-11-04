"""Plan 0029 §3.1 / §6: a v7 (claimmap-v2.7 / +am2) root rebuilds byte-identically under the v8
code. The fixture (inputs: readouts, caches, intent cache; outputs: what the pre-change code
rebuilt) was written once by ``scripts/index/build_v7_fixture.py`` with the pre-change code."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from hallucination_bench.evals.agentic_misalignment import intent
from hallucination_bench.evals.agentic_misalignment import run as amrun
from hallucination_bench.evals.wsbench_chat import run as wrun

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "v7_rebuild"
_spec = importlib.util.spec_from_file_location(
    "build_v7_fixture", REPO / "scripts" / "index" / "build_v7_fixture.py"
)
assert _spec is not None and _spec.loader is not None
bvf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bvf)


@pytest.fixture(autouse=True)
def _no_real_overrides(monkeypatch):
    monkeypatch.setattr(intent, "OVERRIDES_PATH", FIX / "no_overrides.json")


def _expected(d: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted((d / "expected").iterdir())}


@pytest.mark.parametrize("arm", ["prose", "tokens"])
@pytest.mark.parametrize("arm_json", [None, "claimmap-v2.7"])
def test_chat_v7_rebuild_is_byte_identical(tmp_path, arm, arm_json):
    src = FIX / "chat" / arm
    models = wrun.Models()
    r = shutil.copy(src / "readouts.jsonl", tmp_path / f"chat_{arm}.jsonl")
    d = wrun.arm_dir(tmp_path / "out", models, arm)
    d.mkdir(parents=True)
    shutil.copy(src / "cache.jsonl", d / "cache.jsonl")
    if arm_json is not None:  # an explicit v2.7 record behaves as a missing one
        (d / "arm.json").write_text(json.dumps({"arm": arm, "map_prompt": arm_json}))
    wrun.rebuild_arm(arm, Path(r), tmp_path / "out", models)
    got = bvf.chat_outputs(d, Path(r))
    assert got == _expected(src)


@pytest.mark.parametrize("arm", ["prose", "tokens"])
def test_am_v7_rebuild_is_byte_identical(tmp_path, arm):
    src = FIX / "am" / arm
    models = wrun.Models(eval="agentic_misalignment")
    out = tmp_path / "out"
    r = bvf.expand_am_readouts(src / "readouts_sparse.jsonl", tmp_path / f"am_{arm}.jsonl")
    d = amrun.arm_dir(out, models, arm)
    d.mkdir(parents=True)
    shutil.copy(src / "cache.jsonl", d / "cache.jsonl")
    shutil.copy(FIX / "am" / "intent.jsonl", amrun.intent_cache_path(out, models))
    amrun.rebuild(arm, r, out, models)
    got = bvf.am_outputs(d, r)
    want = _expected(src)
    for name in want:  # one assert per file keeps a failure readable
        assert got[name] == want[name], name
    assert set(got) == set(want)
