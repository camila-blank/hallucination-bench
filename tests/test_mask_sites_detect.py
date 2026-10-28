"""Plan 0017: site selection, detector / verifier render + fail-closed parse."""

import pytest

from hallucination_bench.evals.mask import bank
from hallucination_bench.evals.mask import detect as D
from hallucination_bench.evals.mask import verify as V
from hallucination_bench.readouts import Cell

SYS = ["<|im_start|>", "system", "\n", "You", " are", " PR", ".", "<|im_end|>", "\n"]
USR = ["<|im_start|>", "user", "\n", "Was", " it", " fraud", "?", "<|im_end|>", "\n"]
GEN = ["<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"]


def test_prompt_sites_end_at_assistant_never_in_think_suffix():
    prompt = SYS + USR + GEN
    toks = [*prompt, "No", ".", "<|im_end|>"]
    sel = bank.select_sites(toks, len(prompt))
    p = [pos for pos, st in sel if st == "prompt"]
    r = [pos for pos, st in sel if st == "response"]
    a = len(SYS) + len(USR) + 1  # the final 'assistant'
    assert toks[a] == "assistant" and max(p) == a and min(p) == 3
    assert len(p) == 8 and p == sorted(set(p))
    assert r == [len(prompt), len(prompt) + 1]  # <|im_end|> excluded, deduplicated
    assert all(pos < len(SYS) + len(USR) + 2 for pos in p)  # never the think suffix


def test_select_sites_long_response_and_bad_render():
    prompt = SYS + USR + GEN
    toks = prompt + [f"w{i}" for i in range(100)]
    r = [pos for pos, st in bank.select_sites(toks, len(prompt)) if st == "response"]
    assert len(r) == 8 and r[0] == len(prompt) and r[-1] == len(toks) - 1
    with pytest.raises(ValueError):
        bank.select_sites(USR + GEN, len(USR + GEN))


def _cell(samples=None, tokens=None, id="t1", layer=36, pos=5):
    return Cell(
        id=id,
        layer=layer,
        pos=pos,
        samples=tuple(samples) if samples else None,
        tokens=tuple(tokens) if tokens else None,
        scores=None,
    )


def test_render_is_blind_and_canonical_has_no_batch_label():
    c = _cell(samples=["The assistant will lie.", "x"], id="secret_item_77", layer=44)
    canon = D.canonical(D.readouts_of(c))
    system, user = D.render_batch([canon])
    for leak in ("secret_item_77", "44", "honest", "lie_rate", "prompt", "response"):
        assert leak not in canon
    assert "cell c0" in user and "cell c0" not in canon
    assert all(cat in system for cat in D.CATEGORIES)


def test_bag_text_decodes_byte_level_tokens():
    c = _cell(tokens=["Ġlie", "è°İ", "Ċ"])
    assert D.readouts_of(c) == [" lie | 谎 | \\n"]


def test_parse_batch_fail_closed():
    ro = [["The assistant plans to deny the breach."], ["nothing here"]]
    good = {
        "cells": [
            {
                "cell": "c0",
                "mentions": [
                    {
                        "readout": 0,
                        "quote": "plans to deny",
                        "category": "lie_claim",
                        "actor": "ai_or_speaker",
                    }
                ],
            },
            {"cell": "c1", "mentions": []},
        ]
    }
    p = D.parse_batch(good, ro)
    assert p[0].ok and p[0].mentions[0]["start"] == len("The assistant ")
    assert p[1].ok and p[1].mentions == []
    missing = {"cells": [good["cells"][0]]}
    assert [x.ok for x in D.parse_batch(missing, ro)] == [True, False]
    bad_quote = {
        "cells": [
            {"cell": "c0", "mentions": [{**good["cells"][0]["mentions"][0], "quote": "xyz"}]},
            good["cells"][1],
        ]
    }
    assert [x.ok for x in D.parse_batch(bad_quote, ro)] == [False, True]
    bad_idx = {
        "cells": [
            {"cell": "c0", "mentions": [{**good["cells"][0]["mentions"][0], "readout": 3}]},
            good["cells"][1],
        ]
    }
    assert not D.parse_batch(bad_idx, ro)[0].ok
    twice = {"cells": [good["cells"][0], good["cells"][0], good["cells"][1]]}
    assert not D.parse_batch(twice, ro)[0].ok
    assert not any(x.ok for x in D.parse_batch(None, ro))


def test_locate_quote_newline_and_whitespace():
    assert D.locate_quote("a\\nb", "xx a\nb") == (3, 6)
    assert D.locate_quote("lie  to them", "will lie to them") is not None


def test_verify_render_and_parse():
    canon = V.canonical("the PR aide will mislead", "will mislead")
    _, u_blind = V.render([canon], None)
    _, u = V.render([canon], "System Prompt:\nYou are a PR aide")
    assert "PR aide\n\n##########" in u and "##########" not in u_blind
    raw = {
        "mentions": [
            {"mention": "m0", "real": True, "recategory": "lie_claim", "reason": "r"},
            {"mention": "m1", "real": False, "recategory": "lie_claim", "reason": "r"},
        ]
    }
    out = V.parse_batch(raw, 3)
    assert out[0]["real"] and out[0]["recategory"] == "lie_claim"
    assert out[1]["recategory"] == "other"  # real=false cannot keep lie_claim
    assert out[2] is None
