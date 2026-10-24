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
   (`v4-jlens-std-n1000`, `v4-s3don-sft`, `v4-s3don-rl600`; plus the chat-only intermediate
   `v4-s3don-sft14500`) need no method entry: their roots are passed beside the Qwen ones. No V4
   path is hard-coded; the V4 outputs follow the index's own `--out` layout
   (`<out>/{wsbench_chat,story,agentic_misalignment}/<judge>/<arm>/cells.jsonl`,
   `<out>/sibling/um/...`), so `hbench index --out outputs/v4` indexes V4 alone.
3. **One name collision, fixed at export, not in the index.** The V4 UM ladder labels the V4
   J-lens arm `jlens` (the ladder code's fixed J-lens name), which collides with the Qwen
   `jlens` of the official sibling root (`_um_units` rejects a method under two roots — kept).
   `scripts/v4_um/export_index_root.py` copies the three domains' `units_labeled.jsonl` into
   `outputs/v4/sibling/um/` with `jlens` → `v4-jlens-std-n1000` (the same rename `um3_v4.py`
   already applies), all other fields and line order unchanged. `index.py` is untouched.
4. `data/index/methods.json` gains two notes only (`v4`, `v4-jlens-std-n1000.sibling`); no
   method entry, no exclusion.

## Invariants (checked)

- The Qwen rows are bit-for-bit unchanged: every one of the 272 Qwen (method, layer) rows of
  the combined index equals, as a JSON value, the row of the plan 0013 command on unmodified
  main, and every row of the `docs/results/0013` table appears verbatim in the new `index.md`.
- V4 AM / story values equal the exploratory `outputs/v4/index/hclaims_v4.json` unweighted
  numbers (max |diff| 0 over 24 cells).

## Known gap: no V4 macro yet

The index's unit is one (method, layer) (`full` needs all four families at that layer,
`matched` main + AM + story). V4 AM was read at layers 21/27/33 (depth-matched to Qwen
32/42/48, `scripts/run_am_v4.sh`) while V4 chat and story were read on the V4 bench grid
15/24/30/36/39, so **no V4 row has `full` or `matched`**. Not bridged by pairing nearby layers or
averaging over layers (a definition change). Closing it needs one side re-read and re-judged at
the other's layers; the cheaper side is chat + story at 21/27/33 (≈1.2k cells per arm-layer vs
≈47.6k for AM), where the UM ladder already has rows, giving `full` for SFT / RL600 and
`matched` for all three arms. Coordinator decision (GPU readouts + Opus judging).

## Audit of V4 vLLM generations (2026-10-24 corruption modes)

Only the chat and story banks contain V4-generated text (AM keeps the pinned Qwen rollouts; UM
responses come from HF `generate`, B=1). Chat bank (wsbench-pub 8e59016, `gen_vllm_modal.py`,
unpadded, one 149-request call): items `chat-lmsys-0000..0002` are garbage (head-of-call
mode); the rest, incl. the tail, are coherent. Story bank (20 requests, eager, `max_num_seqs=1`,
unpadded): all 20 coherent. Dropping the 3 chat items moves V4 main by ≤ 0.1 pt at every layer;
the official values keep them and the results doc records it. Fix when the chat bank is next
touched: regenerate the 3 through `gen_requests_modal.padded` + canary, re-read, re-judge.
