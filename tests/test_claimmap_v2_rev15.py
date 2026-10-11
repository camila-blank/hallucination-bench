"""Plan 0002 rev 15: hallucination = obviously contradicted by the text (still in force in rev
16; the rev-15 leaves themselves were merged by rev 16, see test_claimmap_v2_rev16.py)."""

import pytest

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges import claimmap_v2 as cm


@pytest.mark.parametrize("system", [cm.SYSTEM, cm.SYSTEM_V5C])
def test_hallucination_gate_is_rendered(system):
    assert tx.HALLUCINATION_GATE in system
    flat = system.replace("**", "")
    assert "only if the conversation obviously contradicts it" in flat
    assert "or that the text simply never mentions the claim" in flat
    assert '"Alibaba Group engineers who created me"' in system
    assert "Never the incorrect branch: that needs an obvious contradiction." in system
    for s in tx.definitions():
        assert s in cm.SYSTEM, s


def test_rule_3_order_puts_realised_first_then_contradiction():
    body = tx.RULES[3].body
    i_real = next(i for i, b in enumerate(body) if "in any " in b and "`correct`" in b)
    i_contra = next(i for i, b in enumerate(body) if "obviously contradicts" in b)
    i_unv = next(i for i, b in enumerate(body) if b.startswith("otherwise `unverifiable`"))
    assert i_real < i_contra < i_unv


def test_dev_v2_keeps_rev15_anchors():
    cases = {c["case_id"]: c for c in cal.load_cases("dev")}
    by_rule: dict[str, int] = {}
    for c in cases.values():
        for r in c["rules"]:
            by_rule[r] = by_rule.get(r, 0) + 1
    assert by_rule.get("realised-paraphrase-stated-after", 0) >= 2
    assert by_rule.get("contradicted-continuation-incorrect", 0) >= 2
    assert [g["leaf"] for g in cases["dev2-39-creator"]["gold"]["units"]] == ["correct"] * 2
    assert [g["leaf"] for g in cases["dev2-40-flight-book"]["gold"]["units"]] == [
        "correct",
        "unverifiable",
        "false_claim",
    ]
