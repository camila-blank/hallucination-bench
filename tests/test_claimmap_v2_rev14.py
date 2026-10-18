"""Plan 0002 rev 14: the narrow unrealised_option, unrealised predictions are incorrect, the
name anchors, the verdict update, in-run retries of invalid answers and the dev re-label."""

import asyncio
import json

import pytest
from test_claimmap_v2 import MapFake, _cells, _run, default_map

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.evals.wsbench_chat import run as wrun
from hallucination_bench.judges import claimmap_v2 as cm


def test_unrealised_leaves_merge_into_unverifiable():
    # rev 16: unrealised_option / unrealised_prediction are unverifiable + tags
    assert "unrealised_option" not in tx.LEAVES and "unrealised_prediction" not in tx.LEAVES
    assert tx.LEAF_RENAMES["unrealised_option"] == "unverifiable"
    assert tx.MERGE_TAGS["unrealised_option"] == {"same_slot_alternative": True, "prediction": True}
    assert tx.verdict_of("unverifiable") == tx.V_UNRESOLVED
    assert tx.FALLBACK_RENAMES == {}


@pytest.mark.parametrize("system", [cm.SYSTEM, cm.SYSTEM_V5C])
def test_rules_3_9_10_and_name_anchors_are_rendered(system):
    flat = system.replace("**", "")
    # rev 16: the same-slot example is a tag on unverifiable
    assert '"3. Baking soda" where item 3 is coffee grounds' in flat
    r3 = tx.RULES[3].body
    assert any(
        "`same_slot_alternative = true` if it fills the same open-ended slot" in b for b in r3
    )
    # rule 8 name anchors
    assert '"Miss Li" and the readout says "Miss Wang"' in system
    assert '("我是 ChatGPT" where it is Qwen) is `substitution_sibling`' in system


def test_unrealised_prediction_gold_loads_without_a_fallback(tmp_path):
    c = cal.skeleton(
        {
            "case_id": "old",
            "part": "synthetic",
            "rules": [],
            "prompt": "q",
            "response": "a⟦READ⟧ b",
            "readouts": ["Next it will mention cats. Fine."],
        }
    )
    c["gold"]["units"][0]["leaf"] = "unrealised_prediction"  # a pre-rev-16 leaf
    c["gold"]["units"][1]["leaf"] = "filler"
    assert cal.validate_case(c) == []
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps(c) + "\n")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # rev 16: a mechanical merge, no fallback warning
        (loaded,) = cal.load_cases("dev", p)
    g0, g1 = loaded["gold"]["units"]
    assert g0["leaf"] == "unverifiable" and g0["tags"] == {"prediction": True}
    assert g1["leaf"] == "junk" and g1["tags"] == {"reason": "filler"}


def test_dev_v2_is_relabelled_with_rev14_anchors():
    cases = {c["case_id"]: c for c in cal.load_cases("dev")}
    rules = {r for c in cases.values() for r in c["rules"]}
    assert {
        "unrealised-option",
        "unrealised-different-answer",
        "self-identity-substitution",
        "user-supplied-name-detail",
    } <= rules
    assert cal.coverage(list(cases.values()))["segments_per_leaf"]["unverifiable"] >= 3
    ident = cases["dev2-37-identity"]["gold"]["units"][0]
    assert ident["leaf"] == "substitution_sibling" and ident["probe_target"] == "ChatGPT"
    assert cases["dev2-38-miss-li"]["gold"]["units"][0]["leaf"] == "detail"


# ------------------------------------------------------------------ in-run retries


def _flaky(n_bad):
    state = {"n": 0}

    def responder(user):
        state["n"] += 1
        if state["n"] <= n_bad:
            return {"units": [], "rationale": "truncated"}  # invalid: readouts missing
        return default_map(user)

    return responder


def _one_cell(prose_readouts, tmp_path):
    rows = [json.loads(x) for x in prose_readouts.read_text().splitlines()][:1]
    p = tmp_path / "one.jsonl"
    p.write_text(json.dumps(rows[0]) + "\n")
    return p


def test_invalid_map_answer_is_retried_in_run_then_succeeds(tmp_path, prose_readouts):
    out = tmp_path / "out"
    fake = MapFake({"map": _flaky(2)})
    _run([("one", _one_cell(prose_readouts, tmp_path))], out, fake, stages=["map"])
    assert fake.n("map") == 3  # 2 invalid + 1 valid, same run
    (rec,) = [c for c in _cells(out, "one") if c["status"] != "empty"]
    assert rec["stages"]["map"] == "ok"
    d = wrun.arm_dir(out, wrun.Models(), "one")
    rows = [json.loads(x) for x in (d / "cache.jsonl").read_text().splitlines()]
    m = [r for r in rows if r["key"].endswith(":M")]
    assert len(m) == 1 and m[0]["result"] is not None
    assert m[0]["meta"]["n_invalid_attempts"] == 2 and m[0]["meta"]["usage"]["calls"] == 3


def test_invalid_map_answer_is_cached_failed_after_two_retries(tmp_path, prose_readouts):
    out = tmp_path / "out"
    fake = MapFake({"map": _flaky(99)})
    _run([("one", _one_cell(prose_readouts, tmp_path))], out, fake, stages=["map"])
    assert fake.n("map") == 1 + wrun.MAP_INVALID_RETRIES == 3
    (rec,) = [c for c in _cells(out, "one") if c["status"] != "empty"]
    assert rec["stages"]["map"] == "failed"


def test_calibration_retries_invalid_answers(tmp_path):
    c = cal.skeleton(
        {
            "case_id": "t",
            "part": "synthetic",
            "rules": [],
            "prompt": "q",
            "response": "a⟦READ⟧ b",
            "readouts": ["One claim. Two claim."],
        }
    )
    for g in c["gold"]["units"]:
        g["leaf"] = "correct"
    from hallucination_bench.cache import Cache

    fake = MapFake({"map": _flaky(1)})
    with Cache(tmp_path / "c.jsonl") as cache:
        preds = asyncio.run(
            cal.map_cases(
                [c], split="dev", model="m", effort=None, cache=cache, llm=fake, repeats=1
            )
        )
    assert fake.n("map") == 2
    segs, _ = preds[0][0]
    assert {s.leaf for s in segs} == {"correct"}
