# Plan 0014 — DeepSeek-V4-Flash arms on the hallucination index

Camila, 2026-10-24: put the V4-Flash HallucinationBench results on the official index
(hindex-v3, plan 0013) and land the V4 port (branch `camila/v4-flash-hb`, commit 49b78b3).
**No change to any weight, the claim definition, the bootstrap or `INDEX_VERSION`.**

## What lands

1. **The V4 port** (cherry-pick of 49b78b3, conflict-free on main): env-overridable chat / story
   / AM banks (`HBENCH_*_ITEMS`, `HBENCH_WSBENCH_PUB_V4`), the V4 AM bank builder, and the story
   and UM-ladder V4 drivers under `scripts/` (see `CLAUDE.md` §DeepSeek-V4-Flash port).
2. **V4 arms on the index by name.** `hbench index` already matches an arm across families by
   directory / `method` name, and its roots are repeatable flags, so the V4 arms
   (`v4-jlens-std-n1000`, `v4-s3don-sft`, `v4-s3don-rl600`) need no method entry: their roots
   are passed beside the Qwen ones. No V4 path is hard-coded; the V4 outputs follow the index's
   own root layout (`<root>/<arm>/cells.jsonl` per family, `<um root>/{,<domain>/}label/`).
3. **One name collision, fixed at export, not in the index.** The V4 UM ladder labels the V4
   J-lens arm `jlens` (the ladder code's fixed J-lens name), which collides with the Qwen
   `jlens` of the official sibling root (`_um_units` rejects a method under two roots — kept).
   `scripts/v4_um/export_index_root.py` copies the three domains' `units_labeled.jsonl` into
   `outputs/v4/sibling/um/` with `jlens` → `v4-jlens-std-n1000` (the same rename `um3_v4.py`
   already applies), all other fields and line order unchanged. `index.py` is untouched.
4. `data/index/methods.json` gains two notes only (`v4`, `v4-jlens-std-n1000.sibling`); no
   method entry, no exclusion.

## Invariants (checked)

- The Qwen rows are bit-for-bit unchanged (checked before and after the V4 re-run): every one of
  the 272 Qwen (method, layer) rows of the combined index equals, as a JSON value, the row of the plan 0013 command on unmodified
  main, and every row of the `docs/results/0013` table appears verbatim in the new `index.md`.
- V4 AM / story values equal the exploratory `outputs/v4/index/hclaims_v4.json` unweighted
  numbers (max |diff| 0 over 24 cells); the re-run leaves V4 story at the old layers unchanged.

## V4 macros: chat + story re-read at 21/27/33 (Camila, 2026-10-24: approved)

`full` needs all four families at a layer, and `matched` needs main + AM + story. V4 AM was read
at layers 21/27/33 (depth-matched to Qwen 32/42/48, `scripts/run_am_v4.sh`). V4 chat and story
were read on the V4 bench grid 15/24/30/36/39. So at first no V4 row had a macro. The gap is not
bridged by pairing nearby layers or averaging over layers, because that would change the
definition. Instead, chat and story were re-read and re-judged at 21/27/33 for all three arms.
This side is cheaper: about 1.2k cells per arm-layer, against about 47.6k for AM. The UM ladder
already has rows at those layers.

- **Readouts** (`scripts/v4_index/readout_jobs.py`):
  - The bench producers only read the bench grid's layers, so the cells go through their jobs
    mode (global-workspace `wsbench_v4_modal.py` / `wsbench_v4_jlens_modal.py::fanout_jobs`, one
    Modal app per arm).
  - The recipe is the same:
    - capture: B=1 block-output capture, with the adapter disabled for the O-lens;
    - O-lens: AO carrier with slot injection, 4 samples, T 1, top-p 0.95, top-k 64, 256 new
      tokens;
    - J-lens: stream mean, cosine top-10.
  - The jobs are the bank's `capture_rows` (input ids, read positions) at 21/27/33. The 3
    regenerated chat items (below) are read at all 8 layers.
  - `merge` converts the rows to the bench contract (O-lens samples joined by the producer's
    `join_samples`). It adds them to the existing readouts, minus the 3 regenerated items, and
    checks the exact grid.
  - Check: 24 J-lens cells of unchanged items were also read at layer 15. They match the bench
    producer's rows (same tokens; scores within 1e-4, last-digit rounding).
- **Judging** (`scripts/v4_index/judge_v4r.sh`):
  - The judge and recipe are the existing V4 runs': chat runs every stage on the V4 chat bank,
    story runs `--stages map`.
  - Output goes to `outputs/v4r/`, seeded with the old runs' `cache.jsonl`. The cache is keyed by
    content, so old cells are cache hits and changed items are re-judged. The V4 story values at
    15/24/30/36/39 equal the earlier run's exactly.
  - Coverage: ≥ 99.8% of cells per arm-layer have a claim map. The few left are the judge's
    deterministic refusals, as in the earlier runs.
  - Spend was about $790 (chat about $720, story $65).
- **Index:** main and story read `outputs/v4r`; AM and sibling read `outputs/v4`. V4 `full` and
  `matched` exist at 21/27/33 only.
- `v4-s3don-sft14500` was read on the pre-fix bank and not re-run. It is left out of the index
  sources and the results table. The index mechanism is unchanged.

## Audit of V4 vLLM generations (2026-10-24 corruption modes)

Only the chat and story banks contain V4-generated text:
- AM keeps the pinned Qwen rollouts.
- UM responses come from HF `generate` with B=1.

**Chat bank** (wsbench-pub 8e59016, `gen_vllm_modal.py`, unpadded, one 149-request call):
- Items `chat-lmsys-0000..0002` were garbage (the head-of-call mode). The rest, including the
  tail, are coherent.
- **Fixed** (`scripts/v4_index/regen_chat3.py`):
  - The 3 items were re-asked with the bank's own recipe on global-workspace main's
    `gen_requests_modal.py`. The render was asserted equal to the bank's prompt ids. The recipe
    is V4 chat with thinking off, T 1.0, top-p 1.0, top-k off, ≤ 512 tokens, seed 7.
  - The runner wraps the call in 300 head and tail pads that run to max_tokens. The Jaws + Paris
    canary passed.
  - All three responses are coherent.
  - Sites were rebuilt with `wsbench_v4_content.py build`'s rule (8 each; the bank goes from
    1,045 to 1,051 sites).
  - The 3 items were spliced into both V4 bench copies (`wsbench-pub-v4-opus`, `wsbench-pub-v4`).
    The originals are kept as `items.json.corrupt` / `capture_rows.json.corrupt`.
- Main values at the old layers move by ≤ 0.13 pt.

**Story bank** (20 requests, eager, `max_num_seqs=1`, unpadded): all 20 are coherent, so there was
nothing to rerun.
