"""Plan 0008 Stage M: ``claimmap-v2.7+am2`` (the chat system prompt byte for byte, a new head
with the marker in the prompt, the chat tail verbatim)."""

import pytest

from hallucination_bench.evals.agentic_misalignment import bank
from hallucination_bench.judges import claimmap_am as cam
from hallucination_bench.judges import claimmap_v2 as cm
from hallucination_bench.judges.v5c import READ_MARK
from hallucination_bench.units import split_cell

SAMPLES = ["The AI plans to {blackmail} Kyle. It weighs {n} options; {units_block}", "Second."]


def _prompt_block(user: str) -> str:
    return user.split("<prompt>\n", 1)[1].split("\n</prompt>", 1)[0]


def test_system_and_tail_are_the_chat_ones():
    assert cam.SYSTEM == cm.SYSTEM
    assert cam.SCHEMA is cm.SCHEMA
    assert cam.PROMPT_VERSION == "claimmap-v2.7+am2"
    assert cm.PROMPT_VERSION == "claimmap-v2.7"  # unchanged
    tail = cm.USER_TEMPLATE[cm.USER_TEMPLATE.index("Each line is `unit id | unit text`") :]
    assert cam.USER_TAIL == tail
    assert cam.USER_TEMPLATE_AM.endswith(tail)
    assert cam.USER_TEMPLATE_AM.startswith(cam.RESEARCH_FRAME)
    assert cam.USER_TEMPLATE_AM[len(cam.RESEARCH_FRAME) :].startswith(
        "A language model, acting as an agent, was given the prompt below and wrote the "
        "response below. The lens read the model's activation at the token immediately before "
        "the marker ⟦READ⟧, which is inside the PROMPT"
    )


def test_units_block_equals_the_chat_single_call_block():
    units, _ = split_cell(SAMPLES)
    it = bank.load_items()[0]
    _s, user = cam.render_prompt_am(bank.marked_prompt(it, 40), it["response"], units)
    _s2, chat_user = cm.render_prompt("p", "r" + READ_MARK, units)
    block = cm.units_block(units, flags=False)
    assert block in user and block in chat_user
    assert " | - | " not in user and "ESTABLISHED" not in user.split("<response>")[-1]
    assert user.endswith(chat_user[chat_user.index("Each line is `unit id | unit text`") :])


def test_marker_once_inside_the_prompt_block_and_braces_kept():
    units, _ = split_cell(SAMPLES)
    it = bank.load_items()[3]
    char = it["sites"][100]["char"]
    system, user = cam.render_prompt_am(bank.marked_prompt(it, char), it["response"], units)
    assert system == cm.SYSTEM
    block = _prompt_block(user)
    assert block.count(READ_MARK) == 1
    assert block.replace(READ_MARK, "") == it["prompt_render"]
    assert block.index(READ_MARK) == char
    resp = user.split("<response>\n", 1)[1].split("\n</response>", 1)[0]
    assert resp == it["response"] and READ_MARK not in resp
    # braces in the substituted text are not re-expanded (one pass)
    assert "{blackmail}" in user and "{n} options;" in user and '"{units_block}"' in user
    assert f"pre-split into {len(units)} units." in user
    assert cm.TOKENS_NOTE not in user


def test_tokens_note_and_errors():
    units, _ = split_cell(["A summary."])
    it = bank.load_items()[0]
    _s, user = cam.render_prompt_am(bank.marked_prompt(it, 10), it["response"], units, "tokens")
    assert cm.TOKENS_NOTE + "The lens readouts" in user
    with pytest.raises(ValueError):
        cam.render_prompt_am(bank.marked_prompt(it, 10), it["response"], [])
    with pytest.raises(ValueError):
        cam.render_prompt_am(it["prompt_render"], it["response"], units)
