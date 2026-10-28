"""Plan 0017: the verbatim MASK prompts and the per-archetype prompt choice."""

import hashlib
import os
import runpy
from pathlib import Path

import pytest

from hallucination_bench.evals.mask import source as S

NAMES = (
    "BELIEF_BINARY_JUDGE_PROMPT_DIRECT",
    "BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY",
    "BINARY_JUDGE_PROMPT_TEMPLATE",
    "DISINFORMATION_RULES",
    "CONTINUATIONS_RULES",
    "DOUBLING_DOWN_BINARY_PROMPT",
    "NUMERICAL_JUDGE_PROMPT",
)
# sha256 of the literals joined by NUL, taken from mask @ 25e0b12 (plan 0017)
LITERALS_SHA = "a02a94de7a0636f3308ff52f4607bfca64fc910e9861259a448c044372d93c1d"


def test_literals_pinned():
    blob = "\x00".join(getattr(S, n) for n in NAMES)
    assert hashlib.sha256(blob.encode()).hexdigest() == LITERALS_SHA


def test_literals_match_upstream_checkout():
    root = os.environ.get("HBENCH_MASK_REPO")  # a centerforaisafety/mask checkout @ 25e0b12
    src = Path(root or ".") / "mask/prompts/evaluation_prompts.py"
    if not root or not src.exists():
        pytest.skip("no centerforaisafety/mask checkout ($HBENCH_MASK_REPO)")
    up = runpy.run_path(str(src))
    for n in NAMES:
        assert up[n] == getattr(S, n), n


def test_parse_proposition():
    a, b = S.parse_proposition(
        "Adolf Hitler <pivotal>died in Berlin/escaped to Argentina</pivotal> in 1945."
    )
    assert a == "Adolf Hitler died in Berlin in 1945."
    assert b == "Adolf Hitler escaped to Argentina in 1945."
    assert S.parse_proposition("There were <pivotal><value></pivotal> million") is None
    assert S.parse_proposition("no pivot here") is None


def _item(arch: str) -> dict:
    return {
        "task_id": "t",
        "archetype": arch,
        "system_prompt": "SYS",
        "user_prompt": "USR",
        "dd_turn_1_prompt": "U1",
        "dd_turn_1_response": "R1",
        "proposition": "X <pivotal>is/is not</pivotal> true.",
        "belief_elicit_1": "BE1",
        "belief_elicit_2": "BE2",
    }


def test_statement_prompt_rules():
    p = S.statement_prompt(_item("disinformation"), "RESP")
    assert S.DISINFORMATION_RULES in p and "RESP" in p and '"X is true."' in p
    assert S.CONTINUATIONS_RULES in S.statement_prompt(_item("continuations"), "R")
    kf = S.statement_prompt(_item("known_facts"), "R")
    assert S.DISINFORMATION_RULES not in kf and S.CONTINUATIONS_RULES not in kf
    dd = S.statement_prompt(_item("doubling_down_known_facts"), "FINAL")
    assert "AI Response 1 (forced response" in dd and "R1" in dd and "FINAL" in dd


def test_belief_prompt_direct_vs_consistency():
    d = S.belief_prompt(_item("known_facts"), 1, "R")
    c = S.belief_prompt(_item("known_facts"), 2, "R")
    assert "indicates agreement with only statement A" in d and "BE1" in d
    assert "only consistent with statement A" in c and "BE2" in c
    assert S.BELIEF_SYSTEM in d
