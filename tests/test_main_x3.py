"""Plan 0028: the 3x main bank — read sites are punctuation/whitespace-only response tokens,
the WSB 8 sites are a subset of the new 16, and the WSB items are byte-identical."""

import json
import os
import unicodedata
from pathlib import Path

import pytest

from hallucination_bench import DATA
from hallucination_bench.evals.wsbench_chat import expand

WSB_DIR = DATA / "wsbench_chat"
X3_ITEMS = Path(os.environ.get("HBENCH_X3_BANK", DATA / "wsbench_chat_x3")) / "items.json"
WSB_ITEM_KEYS = ("id", "stratum", "source", "prompt", "response", "response_start")


def punct_ws_only(text: str) -> bool:
    """Independent of ``expand``: non-empty, every char Unicode punctuation or whitespace, and
    at least one non-space char (a bare " " token is not a site)."""
    return bool(text.strip(" ")) and all(
        unicodedata.category(c).startswith("P") or c.isspace() for c in text
    )


def wsb_items() -> list[dict]:
    return json.loads((WSB_DIR / "items.json").read_text())["items"]


@pytest.fixture(scope="module")
def tok():
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    except Exception as e:
        pytest.skip(f"Qwen3.6-27B tokenizer unavailable: {e}")


# ---------------------------------------------------------------- pure rules


@pytest.mark.parametrize("n", range(1, 401))
def test_thin_16_keeps_every_thin_8_site(n):
    sites = [{"pos": i} for i in range(n)]
    t8 = expand.thin(sites, 8)
    t16 = expand.thin(sites, 16)
    assert len(t16) == min(n, 16)
    assert all(s in t16 for s in t8)
    assert [s["pos"] for s in t16] == sorted({s["pos"] for s in t16})


def test_site_kind_rules():
    assert expand.site_kind(",") == "clause"
    assert expand.site_kind(".**") == "sentence"
    assert expand.site_kind("\n\n") == "newline"
    assert expand.site_kind(' "') == "quote"
    assert expand.site_kind(" **") == "markup"
    for word in ("the", " 's", "`", "<|im_end|>", " ", "", "a."):
        assert expand.site_kind(word) is None


def test_screen_chat_seed():
    ok = "What is the capital of France, and why?"
    assert expand.screen_chat_seed(ok)
    assert not expand.screen_chat_seed("too short")
    assert not expand.screen_chat_seed("x" * 2001)
    assert not expand.screen_chat_seed(ok + " see https://example.com")
    assert not expand.screen_chat_seed(ok + " ```code```")
    assert not expand.screen_chat_seed("Hello NAME_1, how are you doing today?")
    assert not expand.screen_chat_seed("这是一个中文的问题，请回答一下好吗？谢谢你的帮助")


# ---------------------------------------------------------------- the WSB copy (always present)


def test_every_wsb_site_is_punctuation_or_whitespace_only():
    items = wsb_items()
    n = 0
    for it in items:
        assert 1 <= len(it["sites"]) <= 8
        for s in it["sites"]:
            assert punct_ws_only(s["token"]), (it["id"], s)
            assert "<|" not in s["token"]
            n += 1
    assert n == 1123


def test_wsb_sites_reproduced_and_kept_by_the_16_thin(tok):
    raw = tok.backend_tokenizer

    def dec(ids):
        return raw.decode([int(t) for t in ids], skip_special_tokens=False)

    rows = {r["id"]: r for r in json.loads((WSB_DIR / "capture_rows.json").read_text())}
    for it in wsb_items():
        r = rows[it["id"]]
        s8 = expand.item_sites(dec, r["input_ids"], r["prompt_len"], 8)
        s16 = expand.item_sites(dec, r["input_ids"], r["prompt_len"], 16)
        assert s8 == it["sites"], it["id"]
        assert all(s in s16 for s in it["sites"]), it["id"]


# ---------------------------------------------------------------- the 3x bank (local until frozen)

x3 = pytest.mark.skipif(not X3_ITEMS.exists(), reason="3x bank not built here (local until frozen)")


@pytest.fixture(scope="module")
def x3_items() -> list[dict]:
    return json.loads(X3_ITEMS.read_text())["items"]


@x3
def test_x3_every_site_is_punctuation_or_whitespace_only(x3_items):
    for it in x3_items:
        sites = it["sites"]
        assert 1 <= len(sites) <= 16, it["id"]
        assert [s["pos"] for s in sites] == sorted({s["pos"] for s in sites}), it["id"]
        for s in sites:
            assert punct_ws_only(s["token"]), (it["id"], s)
            assert "<|" not in s["token"] and s["kind"] != "eos", (it["id"], s)
            assert s["pos"] >= it["response_start"], (it["id"], s)
            assert 0 < s["char"] <= len(it["response"]), (it["id"], s)


@x3
def test_x3_wsb_subset_is_byte_identical(x3_items):
    wsb = wsb_items()
    got = [it for it in x3_items if it["subset"] == "wsb149"]
    assert [it["id"] for it in got] == [it["id"] for it in wsb]
    assert x3_items[: len(wsb)] == got  # WSB items first, in WSB order
    for old, new in zip(wsb, got, strict=True):
        for k in WSB_ITEM_KEYS:
            assert new[k] == old[k], (old["id"], k)
        keys = ("pos", "token", "kind", "char")
        keep = [{k: s[k] for k in keys} for s in new["sites"] if s["wsb"]]
        assert keep == old["sites"], old["id"]
    assert sum(s["wsb"] for it in got for s in it["sites"]) == 1123


@x3
def test_x3_new_items_are_distinct_and_balanced(x3_items):
    new = [it for it in x3_items if it["subset"] == "x3new"]
    assert {it["subset"] for it in x3_items} <= {"wsb149", "x3new"}
    assert not any(s["wsb"] for it in new for s in it["sites"])
    for field in ("id", "seed_key", "prompt"):
        vals = [it[field] for it in x3_items]
        assert len(vals) == len(set(vals)), field
    assert {it["source"] for it in new} <= {"lmsys", "dailydialog"}


@x3
def test_x3_sites_checked_against_token_ids(x3_items, tok):
    """Each site's pos points at a non-special token whose decode is the stored token; the
    capture rows read exactly the sites; no rollout ends in a special token (no <|im_end|>)."""
    raw = tok.backend_tokenizer
    special = set(tok.all_special_ids) | {
        i for t, i in tok.get_vocab().items() if t.startswith("<|") and t.endswith("|>")
    }
    rows = {r["label"]: r for r in json.loads((X3_ITEMS.parent / "capture_rows.json").read_text())}
    assert set(rows) == {it["id"] for it in x3_items}
    for it in x3_items:
        r = rows[it["id"]]
        ids = r["input_ids"]
        assert r["prompt_len"] == it["response_start"]
        assert r["eval_positions"] == [s["pos"] for s in it["sites"]], it["id"]
        assert ids[-1] not in special, it["id"]
        for s in it["sites"]:
            assert it["response_start"] <= s["pos"] < len(ids), (it["id"], s)
            assert ids[s["pos"]] not in special, (it["id"], s)
            assert raw.decode([ids[s["pos"]]], skip_special_tokens=False) == s["token"]


@x3
def test_x3_subset_counts(x3_items):
    by = {}
    for it in x3_items:
        by[(it["subset"], it["source"])] = by.get((it["subset"], it["source"]), 0) + 1
    assert by[("wsb149", "lmsys")] == 74 and by[("wsb149", "dailydialog")] == 75
    meta = json.loads(X3_ITEMS.read_text())["meta"]
    assert meta["n_items"] == len(x3_items)
    assert meta["n_sites"] == sum(len(it["sites"]) for it in x3_items)
