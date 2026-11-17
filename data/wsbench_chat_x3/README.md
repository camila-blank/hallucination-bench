# data/wsbench_chat_x3 — the main bank at 3x prompts, 2x read positions (plan 0028)

**Frozen (hindex-v8).** These are the files the frozen v8 index scored: `items.json` is the bank
`data/index/v8_manifest.json` records (`provenance.bank.sha256`), and every file is pinned by
`MANIFEST.sha256` (`sha256sum -c MANIFEST.sha256`; `tests/test_main_x3.py`). Never edit. Built by `scripts/main_x3/` (see
`plans/0028-main-bank-expansion.md`).

| file | what |
|---|---|
| `items.json` | `{meta, items}` in the `wsbench_chat` schema plus `subset` (`wsb149` / `x3new`), `seed_key`, and `sites[].wsb` |
| `capture_rows.json` | one row per item: exact `input_ids`, `prompt_len`, `eval_positions` = every site |
| `capture_new_a.json`, `capture_new_b.json` | the cells read per batch. `a` holds the 905 new sites of the WSB items plus the new dailydialog items; `b` holds the new lmsys items |
| `seeds.json`, `rollouts_<source>.json` | provenance: screen counts, the WSB re-derivation check, generation settings and versions |

- **WSB parity:** take `subset == "wsb149"` and sites with `wsb: true`. Together these are exactly
  `data/wsbench_chat/items.json` (1,123 sites), and their item fields are byte-identical.
- **Read sites:** every site is a response token made only of punctuation and/or whitespace, with
  ≤16 per item spread evenly (`chat_bank.thin`). There is no `<|im_end|>` site.
- **Use with the chat pipeline:** `hbench run|report --bank x3` (plan 0029; another location:
  `HBENCH_X3_BANK=<dir>`; `run` reads `$HBENCH_X3_READOUTS` unless `--readouts-root` / `--arm`
  is given), or the older `HBENCH_WSBENCH_CHAT_ITEMS=data/wsbench_chat_x3/items.json`.
- **Readouts:** `$HBENCH_X3_READOUTS/<arm>/hallucination.jsonl`.
