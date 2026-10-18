"""Plan 0008: the agentic_misalignment bank (``data/agentic_misalignment/``), built by
``scripts/build_am_bank.py``."""

from collections import Counter

import pytest

from hallucination_bench.evals import bank_for
from hallucination_bench.evals.agentic_misalignment import bank
from hallucination_bench.judges.v5c import READ_MARK


def test_32_items_and_50452_sites():
    items = bank.load_items()
    assert len(items) == 32
    assert sum(len(it["sites"]) for it in items) == bank.N_SITES == 50_452
    assert Counter(it["arm"] for it in items) == {"misaligned": 28, "control": 4}
    assert Counter(it["group"] for it in items) == {"anthropic": 18, "working": 14}
    assert bank.load_bank()["meta"]["n_sites"] == 50_452
    assert len(bank.expected_grid(items, [20, 24])) == 2 * 50_452


def test_render_is_the_qwen_chat_render():
    for it in bank.load_items():
        want = (
            f"<|im_start|>system\n{it['system']}<|im_end|>\n" if it["system"] else ""
        ) + f"<|im_start|>user\n{it['text']}<|im_end|>\n<|im_start|>assistant"
        assert it["prompt_render"] == want == bank.render_prompt(it["system"], it["text"])
        assert it["response"] == it["prefill"]


def test_sites_are_contiguous_prompt_positions_ending_at_assistant():
    for it in bank.load_items():
        sites = it["sites"]
        assert [s["pos"] for s in sites] == list(range(len(sites)))
        assert [s["token"] for s in sites[-2:]] == ["<|im_start|>", "assistant"]
        assert sites[-1]["char"] == len(it["prompt_render"])
        for s in sites:
            if s.get("split_char"):
                assert "�" in s["token"]
                continue
            assert it["prompt_render"][: s["char"]].endswith(s["token"]), (it["id"], s)


def test_split_characters_only_in_the_murder_items():
    split = Counter(
        it["id"] for it in bank.load_items() for s in it["sites"] if s.get("split_char")
    )
    assert set(split) == {it["id"] for it in bank.load_items() if it["id"].startswith("am-murder")}
    assert sum(split.values()) == bank.load_bank()["meta"]["n_split_char_sites"]


def test_personas():
    c = Counter(it["persona"] for it in bank.load_items())
    assert c == {
        "Alex": 19,
        "Atlas": 2,
        "Argus": 1,
        "Devin": 1,
        "Hypatia": 1,
        "Kepler": 1,
        "Meridian": 1,
        None: 6,
    }
    for it in bank.load_items():
        if it["persona"]:
            assert it["system"].startswith(f"You are {it['persona']}")


def test_marked_prompt_and_provenance():
    it = bank.load_items()[0]
    s = it["sites"][5]
    mp = bank.marked_prompt(it, s["char"])
    assert mp.count(READ_MARK) == 1
    assert mp.replace(READ_MARK, "") == it["prompt_render"]
    with pytest.raises(ValueError):
        bank.marked_prompt(it, len(it["prompt_render"]) + 1)
    assert bank.BANK_COMMIT.startswith("92d763e")
    assert bank_for("agentic_misalignment") is bank
    assert bank.load_items(["rh-unit-tests"])[0]["id"] == "rh-unit-tests"
    with pytest.raises(KeyError):
        bank.load_items(["nope"])
