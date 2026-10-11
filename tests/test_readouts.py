"""Readout contract loader, decode_token, verify_quote, empty cells."""

from conftest import write_jsonl

from hallucination_bench.readouts import decode_token, load_readouts, normalize_ws, verify_quote


def test_prose_and_token_rows_and_kinds(tmp_path):
    p = write_jsonl(
        tmp_path / "a.jsonl",
        [
            {"id": "x", "layer": 44, "pos": 1, "samples": ["a", "b"], "token": ","},
            {"id": "x", "layer": 44, "pos": 2, "samples": [""]},
        ],
    )
    cells, rep = load_readouts(p)
    assert rep.kind == "prose" and [c.kind for c in cells] == ["prose", "prose"]
    assert cells[0].key == "x__L044__p1" and cells[0].token == ","
    assert cells[1].empty and rep.n_empty == 1
    t = write_jsonl(
        tmp_path / "t.jsonl",
        [{"id": "x", "layer": 20, "pos": 1, "tokens": ["Ġa", "b"], "scores": [1, 2.5]}],
    )
    cells, rep = load_readouts(t)
    assert rep.kind == "tokens" and cells[0].scores == (1.0, 2.5) and not cells[0].empty


def test_malformed_mixed_duplicate_and_off_site_rows(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        '{"id": "x", "layer": 44, "pos": 1, "samples": ["a"]}\n'
        "not json\n"
        '{"id": "x", "layer": 44, "pos": 1, "samples": ["dup"]}\n'
        '{"id": "x", "layer": 44, "pos": 2, "tokens": ["a"]}\n'
        '{"id": "x", "layer": 44, "pos": 9, "samples": ["off"]}\n'
        '{"id": "y", "layer": 44, "pos": 1, "samples": ["unknown id"]}\n'
        '{"id": "x", "layer": "44", "pos": 3, "samples": ["bad layer"]}\n'
    )
    cells, rep = load_readouts(p, positions={"x": [1, 2]})
    assert [c.samples for c in cells] == [("a",)]  # first copy wins
    assert rep.skipped["malformed"] == 2
    assert rep.skipped["duplicate"] == 1
    assert rep.skipped["mixed_kind"] == 1
    assert rep.skipped["pos_not_selected"] == 1
    assert rep.skipped["unknown_id"] == 1


def test_empty_token_cell_and_whitespace_samples(tmp_path):
    p = write_jsonl(
        tmp_path / "e.jsonl", [{"id": "x", "layer": 1, "pos": 1, "samples": [" ", "\n"]}]
    )
    cells, _ = load_readouts(p)
    assert cells[0].empty
    q = write_jsonl(tmp_path / "f.jsonl", [{"id": "x", "layer": 1, "pos": 1, "tokens": []}])
    cells, _ = load_readouts(q)
    assert cells[0].empty


def test_decode_token_byte_level_bpe():
    assert decode_token("æĹ¥æľ¬èªŀ") == "日本語"
    assert decode_token("Ġlanguages") == " languages"
    assert decode_token(" AI") == " AI"


def test_verify_quote_normalisation():
    assert normalize_ws("  “Hi”\n  There ") == '"hi" there'
    assert verify_quote("the  MOON", ["The moon rises"])
    assert verify_quote("“Basic Facts”", ['the "Basic Facts" heading'])
    assert not verify_quote("ab", ["ab cd"])  # under 3 characters
    assert not verify_quote("moon", ["sun"])
