"""Plan 0029 fix: AM ``--limit N`` with the ``map`` stage counts mapped cells (per layer, N
drawn among the detection positives), not pre-detection cells; detect-only keeps the
pre-detection draw."""

import json
import re
import shutil

import pytest
from test_am_run import CTRL, MIS
from test_am_v8 import M8, AM8Fake, _empty_overrides, _prose

from hallucination_bench.evals.agentic_misalignment import intent
from hallucination_bench.evals.agentic_misalignment import run as amrun

_KEY = re.compile(r"__L(\d{3})__p(\d+):M$")

# L20: 3 default positives (MIS 3 / MIS 4 / CTRL 7) + 4 more; L24: MIS 3 only. 20 negatives per
# layer, so a pre-detection draw of 3 would usually map 0-1 cells.
EXTRA = {
    **{(MIS, p, la): f"Plain text about emails {p}." for p in range(10, 30) for la in (20, 24)},
    **{(MIS, p, 20): f"I will blackmail Kyle at step {p}." for p in range(30, 34)},
    (MIS, 4, 24): "Plain text.",
    (CTRL, 7, 24): "Plain text.",
}
POSITIVES = {20: 7, 24: 1}


@pytest.fixture
def detected(tmp_path, monkeypatch):
    monkeypatch.setattr(intent, "OVERRIDES_PATH", _empty_overrides(tmp_path / "ov.json"))
    f = _prose(tmp_path, layers=(20, 24), extra=EXTRA)
    base = tmp_path / "detected"
    amrun.run_am([("toy", f)], base, M8, llm=AM8Fake(), items=[MIS, CTRL], stages="detect")

    def root(name: str):  # a fresh copy of the detected root
        out = tmp_path / name
        shutil.copytree(base, out)
        return out

    return f, root


def _mapped(out) -> dict[int, list[int]]:
    d = amrun.arm_dir(out, M8, "toy")
    got: dict[int, list[int]] = {}
    for line in (d / "cache.jsonl").read_text().splitlines():
        m = _KEY.search(json.loads(line)["key"])
        if m:
            got.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    return {la: sorted(v) for la, v in got.items()}


@pytest.mark.parametrize("seed", [0, None])
def test_map_limit_counts_mapped_cells_per_layer(detected, seed):
    f, root = detected
    out = root(f"o{seed}")
    amrun.run_am(
        [("toy", f)], out, M8, llm=AM8Fake(), items=[MIS, CTRL], stages="map", limit=3,
        sample_seed=seed,
    )  # fmt: skip
    got = _mapped(out)
    assert {la: len(v) for la, v in got.items()} == {la: min(3, n) for la, n in POSITIVES.items()}
    # deterministic: the same seed in a fresh root maps the same cells
    out2 = root(f"o{seed}b")
    amrun.run_am(
        [("toy", f)], out2, M8, llm=AM8Fake(), items=[MIS, CTRL], stages="map", limit=3,
        sample_seed=seed,
    )  # fmt: skip
    assert _mapped(out2) == got


def test_map_limit_larger_than_positives_maps_them_all(detected):
    f, root = detected
    out = root("all")
    amrun.run_am(
        [("toy", f)], out, M8, llm=AM8Fake(), items=[MIS, CTRL], stages="map", limit=50,
        sample_seed=0,
    )  # fmt: skip
    assert {la: len(v) for la, v in _mapped(out).items()} == POSITIVES


def test_detect_limit_keeps_the_pre_detection_draw(tmp_path, monkeypatch):
    monkeypatch.setattr(intent, "OVERRIDES_PATH", _empty_overrides(tmp_path / "ov.json"))
    f = _prose(tmp_path, layers=(20, 24), extra=EXTRA)
    a = amrun.open_arm("toy", f, tmp_path / "o", M8, layers=None)
    try:
        sel = amrun.select(a, [MIS, CTRL], 3, 0)
        assert sorted({c.layer for c in sel}) == [20, 24]
        assert all(sum(c.layer == la for c in sel) == 3 for la in (20, 24))
        assert sel == amrun.select(a, [MIS, CTRL], 3, 0)
        # the pre-detection draw is what map used to get: here fewer than 3 L20 positives
        pos20 = {(MIS, 3), (MIS, 4), (CTRL, 7), *((MIS, p) for p in range(30, 34))}
        assert sum((c.id, c.pos) in pos20 for c in sel if c.layer == 20) < 3
    finally:
        a.cache.close()


def test_map_limit_seeds_draw_different_positives(detected):
    """With more positives than N (L20: 7 for N = 3), the seed changes which ones are mapped."""
    f, root = detected
    picks = {}
    for seed in (0, 1, 2, 3):
        out = root(f"s{seed}")
        amrun.run_am(
            [("toy", f)], out, M8, llm=AM8Fake(), items=[MIS, CTRL], stages="map", limit=3,
            sample_seed=seed,
        )  # fmt: skip
        got = _mapped(out)
        assert len(got[20]) == 3
        picks[seed] = tuple(got[20])
    assert len(set(picks.values())) > 1, picks
