"""Plan 0007: the story bank (``data/story/``), built by ``scripts/build_story_bank.py``."""

import json

from hallucination_bench.evals import bank_for
from hallucination_bench.evals.story import bank


def test_twenty_items_eight_sites_each():
    items = bank.load_items()
    assert len(items) == 20
    assert all(len(it["sites"]) == 8 for it in items)
    assert len(bank.expected_grid(items, [42])) == 160
    assert len(bank.expected_grid(items, [20, 24])) == 320
    assert bank.load_bank()["meta"]["n_sites"] == 160


def test_sites_mark_their_token_and_skip_eos():
    for it in bank.load_items():
        assert it["stratum"] == "story" and it["source"] == "hallucination_creative"
        for s in it["sites"]:
            assert s["kind"] != "eos"
            tok = s["token"].replace("Ġ", " ").replace("Ċ", "\n")
            assert it["response"][: s["char"]].endswith(tok), (it["id"], s)
            marked = bank.marked_response(it, s)
            assert marked.count("⟦READ⟧") == 1


def test_capture_rows_target_exactly_the_bank_sites():
    rows = json.loads((bank.BANK_DIR / "capture_rows.json").read_text())
    by_id = {it["id"]: it for it in bank.load_items()}
    assert sorted(r["label"] for r in rows) == sorted(by_id)
    for r in rows:
        it = by_id[r["label"]]
        assert r["eval_positions"] == sorted(int(s["pos"]) for s in it["sites"])
        assert r["user"] == it["prompt"] and r["prefill"] == it["response"]
        assert all(p < len(r["input_ids"]) for p in r["eval_positions"])


def test_bank_api_matches_the_chat_bank():
    chat = bank_for("wsbench_chat")
    for name in (
        "load_bank",
        "load_items",
        "positions",
        "sites_by_item",
        "marked_response",
        "expected_grid",
        "BANK_SOURCE",
        "BANK_COMMIT",
    ):
        assert hasattr(bank, name) and hasattr(chat, name), name
    assert bank_for("story") is bank
    assert chat.BANK_COMMIT == "92d763e"
    assert bank.BANK_COMMIT == bank.load_bank()["meta"]["source_commit"]
    assert "hallucination/creative" in bank.BANK_SOURCE
    ids = {it["id"] for it in bank.load_items()}
    assert not ids & {it["id"] for it in chat.load_items()}
