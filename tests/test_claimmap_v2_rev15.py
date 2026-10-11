"""Plan 0002 rev 15: hallucination = obviously contradicted by the text; unsupported_assertion;
the reinstated unrealised_prediction; the `alternative` verdict; dev anchors."""

import pytest

from hallucination_bench import calibrate_v2 as cal
from hallucination_bench import taxonomy_v2 as tx
from hallucination_bench.judges import claimmap_v2 as cm


def test_22_leaves_and_the_rev15_verdict_table():
    assert len(tx.LEAF_NAMES) == 22
    assert {"unsupported_assertion", "unrealised_prediction", "unrealised_option"} <= set(tx.LEAVES)
    assert tx.VERDICTS == {
        "hallucination": ("detail", "false_claim"),
        "probe_candidate": (
            "substitution_sibling",
            "substitution_category",
            "supplementary_false",
            "model_state",
            "ambiguous_topic",
        ),
        "alternative": ("unrealised_option", "unrealised_prediction"),
        "correct": ("restates_prefix", "stated_after", "entailed", "supplementary_true"),
        "unresolved": (
            "other_on_topic",
            "unsupported_assertion",
            "supplementary_unsettled",
            tx.UNKNOWN_INCORRECT,
        ),
        "junk": ("off_topic",),
        "non_claim": tx.NON_CLAIM_LEAVES,
    }
    mapped = [lf for leaves in tx.VERDICTS.values() for lf in leaves]
    assert sorted(mapped) == sorted((*tx.LEAF_NAMES, tx.UNKNOWN_INCORRECT))
    assert tx.group_of("unsupported_assertion") == tx.THINKY
    assert tx.VERDICT_ROWS[-1] == tx.FAILED and len(tx.VERDICT_ROWS) == 8


@pytest.mark.parametrize("system", [cm.SYSTEM, cm.SYSTEM_V5C])
def test_hallucination_gate_and_rules_are_rendered(system):
    assert tx.HALLUCINATION_GATE in system
    flat = system.replace("**", "")
    assert "only if the conversation obviously contradicts it" in flat
    assert "or that the text simply never mentions the claim" in flat
    # rule 10: unsupported assertions, realised paraphrases, contradicted continuations
    assert '"the user is a teacher"; "her sister Maria"' in system
    assert '"Alibaba Group engineers who created me"' in system
    assert "This beats both unrealised leaves." in system
    assert "A prediction or continuation that the text obviously contradicts" in system
    # rule 9 order, rule 12 flat assertion, detail rule
    assert (
        "`unrealised_option` > `unrealised_prediction` > `model_state` > "
        "`unsupported_assertion` > `other_on_topic`." in system
    )
    assert "text never addresses is `unsupported_assertion` (rule 10)" in system
    assert "so it is `unsupported_assertion` (rule 10)" in system  # invented specifics
    assert "including invented arbitrary specifics" not in system
    # probe_target no longer asks for the alternatives
    assert "`model_state` and `ambiguous_topic` segment" in system
    for s in tx.definitions():
        assert s in cm.SYSTEM, s


def test_rule_3_order_puts_realised_first_then_contradiction():
    body = tx.RULES[3].body
    i_real = next(i for i, b in enumerate(body) if "in any " in b and "stated_after" in b)
    i_contra = next(i for i, b in enumerate(body) if "obviously contradicts" in b)
    i_opt = next(i for i, b in enumerate(body) if b.startswith("`unrealised_option`"))
    i_pred = next(i for i, b in enumerate(body) if b.startswith("`unrealised_prediction`"))
    assert i_real < i_contra < i_opt < i_pred


def test_invented_specific_decision_defaults_to_unsupported_assertion():
    assert tx.CHOSEN["invented_specific"] == "unsupported_assertion"
    assert not any("invented" in d for d in tx.DETAIL_ITEMS)


def test_dev_v2_has_rev15_anchors():
    cases = cal.load_cases("dev")
    cov = cal.coverage(cases)["segments_per_leaf"]
    assert cov["unsupported_assertion"] >= 3 and cov["unrealised_prediction"] >= 3
    by_rule: dict[str, int] = {}
    for c in cases:
        for r in c["rules"]:
            by_rule[r] = by_rule.get(r, 0) + 1
    assert by_rule.get("realised-paraphrase-stated-after", 0) >= 2
    assert by_rule.get("contradicted-continuation-incorrect", 0) >= 2
    ids = {c["case_id"]: c for c in cases}
    leaves = [g["leaf"] for g in ids["dev2-39-creator"]["gold"]["units"]]
    assert leaves == ["stated_after", "stated_after"]
    assert [g["leaf"] for g in ids["dev2-40-flight-book"]["gold"]["units"]] == [
        "stated_after",
        "unrealised_prediction",
        "false_claim",
    ]
