# `data/sad/`: the SAD stages-oversight bank (plan 0018)

**No SAD text is in git.** SAD (Laine et al. 2024, `LRudL/sad`) ships its questions in
password-protected zips and asks that its texts never appear in plain text anywhere scrapable,
including private repos. This repo is public, so only ids, hashes, gate counts and read-site
positions are committed.

- `scripts/sad/fetch_sad.py` downloads `sad/stages/oversight/structs.zip` at the pinned commit
  (`LRudL/sad@dfc5c98`, zip sha256 pinned), unpacks it with the password SAD publishes in its own
  `unzip.sh`, and writes the texts to **`raw/items.jsonl` (gitignored)** plus `manifest.json`.
  `--check` re-derives the manifest and compares.
- `manifest.json`: `{meta, items: [{id, sad_id, provenance, source, sha256, n_chars}]}`. 399 items:
  `test_oversight` -> provenance `eval` (Anthropic evals 67, BIG-bench 67, WMDP 66) and
  `deploy_oversight` -> `deploy` (199 WildChat; one sample listed twice is kept once).
  `id = "sad-" + first 12 hex of SAD's MCQuestion id`.
- `gate.json` (`hbench sad gate`): Qwen3.6-27B's own answer to SAD's question, 10 samples per
  item: `{meta: {question, survival, ...}, items: {id: {n, n_eval, n_deploy, n_unparsed, majority,
  agree, provenance, belief}}}`; `belief` is set when >= 9 of 10 agree.
- `selected.json` (`hbench sad select`): the pilot items and read sites,
  `{meta, items: [{id, belief, provenance, source, agree, n_tokens, sites: [{pos, region}]}]}` -
  positions only (token strings would spell the text). Token strings and char offsets are
  re-derived locally by `select.sites` from `raw/`.
