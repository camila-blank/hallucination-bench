"""Plan 0029 fix (Camila, 2026-11-04): under claimmap-v2.8 every date or time is `detail`,
real-world events included. The v2.8 prompt says so, v2.7 is unchanged, and calibrating v2.8
applies the version-aware gold overrides (v2.7 keeps the shared gold)."""

import pytest

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges import claimmap_v2 as cm

V27, V28 = "claimmap-v2.7", "claimmap-v2.8"
S7, S8 = cm.VERSIONS[V27].system, cm.VERSIONS[V28].system

DATE_TEXT = (
    "every date or time, including the date of a real-world event",
    "and so is every date or time of a real-world event",
    '"14 April 1911" for the Titanic\'s sinking, actually 1912',
    '"The Titanic sank on 14 April 1911" when the text says 1912: `detail`',
    "quantities of **real-world events** are `false_claim`",
)


@pytest.mark.parametrize("text", DATE_TEXT)
def test_v28_prompt_says_real_world_dates_are_detail(text):
    assert text in S8
    assert text not in S7 and text not in cm.SYSTEM


def test_v28_carve_out_no_longer_makes_real_world_dates_false_claims():
    co = tx.DETAIL_CARVE_OUT_V28
    assert "quantities and dates of" not in co
    assert "Quantities and dates of **real events**" not in S8
    assert tx.detail_carve_out(V28) == co
    # v2.7 keeps its carve-out (real-event dates are false_claim there)
    assert tx.detail_carve_out(V27) == tx.DETAIL_CARVE_OUT
    assert "Quantities and dates of **real events**" in S7
    assert "Quantities and dates of **real events**" in tx.DETAIL_CARVE_OUT


def _leaf(cases, cid, uid, split=None):
    c = next(c for c in cases if c["case_id"] == cid)
    g = next(g for g in c["gold"]["units"] if g["id"] == uid)
    if split is None:
        return g["leaf"]
    return next(s["leaf"] for s in g["splits"] if s["text"] == split)


def test_gold_overrides_apply_under_v28_only():
    dev = cal.load_cases("dev")
    assert _leaf(dev, "dev2-06-titanic", "R0.U3", "14 April 1911") == "false_claim"
    assert _leaf(dev, "dev2-23-penicillin", "R0.U3") == "false_claim"
    assert cal.gold_overrides(V27) == {}
    assert cal.apply_gold_overrides(dev, V27) == dev
    d8 = cal.apply_gold_overrides(dev, V28)
    assert _leaf(d8, "dev2-06-titanic", "R0.U3", "14 April 1911") == "detail"
    assert _leaf(d8, "dev2-23-penicillin", "R0.U3") == "detail"
    # the inputs are untouched and nothing else changes
    assert _leaf(dev, "dev2-06-titanic", "R0.U3", "14 April 1911") == "false_claim"
    changed = [
        (a["case_id"], ga["id"])
        for a, b in zip(dev, d8, strict=True)
        for ga, gb in zip(a["gold"]["units"], b["gold"]["units"], strict=True)
        if ga != gb
    ]
    assert sorted(changed) == [("dev2-06-titanic", "R0.U3"), ("dev2-23-penicillin", "R0.U3")]


def test_gold_override_with_a_moved_target_fails_loudly():
    dev = cal.load_cases("dev")
    c = next(c for c in dev if c["case_id"] == "dev2-23-penicillin")
    bad = {**c, "gold": {**c["gold"], "units": [{**g, "id": "X"} for g in c["gold"]["units"]]}}
    with pytest.raises(ValueError, match="gold override targets not found"):
        cal.apply_gold_overrides([bad], V28)
    # a split without the overridden case is fine
    assert cal.apply_gold_overrides([], V28) == []


@pytest.mark.parametrize("prompt", [V27, V28])
def test_calibrate_run_uses_the_overrides_of_its_version(tmp_path, monkeypatch, prompt):
    seen = {}

    async def fake_map_cases(cases, **kw):
        return [[([], {}) for _ in cases]]

    def fake_score(cases, preds, split, anchor, prompt_):
        seen["cases"] = cases
        return {"split": split, "prompt_version": prompt_}

    class FakeLLM:
        def __init__(self, **kw):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(cal, "map_cases", fake_map_cases)
    monkeypatch.setattr(cal, "score", fake_score)
    monkeypatch.setattr(cal, "LLM", FakeLLM)
    monkeypatch.setattr(cal, "render_markdown", lambda res: "")
    res = cal.run("dev", model="m", effort=None, out_root=tmp_path, repeats=1, prompt=prompt)
    want = "detail" if prompt == V28 else "false_claim"
    assert _leaf(seen["cases"], "dev2-06-titanic", "R0.U3", "14 April 1911") == want
    assert _leaf(seen["cases"], "dev2-23-penicillin", "R0.U3") == want
    if prompt == V28:
        assert {o["case_id"] for o in res["gold_overrides"]} == {
            "dev2-06-titanic",
            "dev2-23-penicillin",
        }
    else:
        assert "gold_overrides" not in res
