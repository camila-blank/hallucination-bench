# 0028: the main bank at 3x prompts and 2x read positions (`wsbench_chat_x3`)

Requested by Camila on 2026-11-04. The first version asked for 5x prompts; the same day she changed it
to **3x prompts**, so this plan is written for 3x. Grow main (`wsbench_chat`) to **3x the prompts**
(≈447 items) and **2x the read positions per prompt** (≤16 sites). Read sites stay
**punctuation/whitespace-only response tokens**, as in the WSB bank. Readouts are captured for the
index arms. **No judge, summary-of-claims or claim map runs here.** Plan 0029 (hindex-v8) is
rewriting the claim-map prompt, and judging happens after it lands. This plan builds data and
readouts only. It changes no index definition, prompt, judge or existing bank file.

## Invariants retained

- `data/wsbench_chat/` (the verbatim WSB copy @ 92d763e) is **not edited**. Its sha256 pins and
  `tests/test_v5c_frozen.py` stay as they are. The 149 WSB items appear in the new bank with
  **byte-identical** `id / stratum / source / prompt / response / response_start` and the same token
  ids (`capture_rows.json` `input_ids`, `prompt_len`). They are marked `subset: "wsb149"`, so WSB
  parity stays computable: filter to `subset == "wsb149"` and to sites with `wsb: true` to get
  exactly the old 1,123-site grid.
- **The WSB 8 sites are a subset of the new 16**, for every WSB item. The thinning is
  `chat_bank.thin`, which keeps `sites[int(i * n / m)]`, and `int(i*n/8) == int(2i*n/16)` exactly.
  Verified on all 149 items: the port reproduces the 8 sites byte-for-byte (pos / token / kind /
  char) and the 16 are a superset (2,028 sites).
- Each site's token is made only of Unicode punctuation (`P*`) and/or whitespace (`is_summary_token`).
  There is no `eos` site. Rollouts carry no `<|im_end|>` (EOS is cut at generation), so the final
  `<|im_end|>` can never be a site. The test also asserts that no site token is a special token.
- The old readouts of the 1,123 WSB cells per layer are **reused unchanged** in every merged
  readout file, so old numbers stay comparable. Only the cells that are new (new items, plus the
  WSB items' 905 new sites) are read.

## Sources and filters (recovered, written down)

The WSB chat bank is the chat half of global-workspace's J-lens P/R items. The rules come from
`scripts/oracle_lens_evals/jlens_pr/{build_seeds.py, items.py, modal_pipeline.py}` and
`scripts/oracle_lens_evals/hallucination/chat_bank.py` (GW main), ported literally into
`src/hallucination_bench/evals/wsbench_chat/expand.py`:

- **lmsys:** the stripped first user turn of every `lmsys/lmsys-chat-1m` row whose conversation
  starts with a user turn. Parquet shards are read in sorted filename order, and `source_id` is the
  stream row index.
- **dailydialog:** `ConvLab/dailydialog` train split, streaming. The seed is the first turn if its
  speaker is `user`, and `source_id` is the `dialogue_id`.
- **Screen:** 20 to 2,000 characters after stripping; no ```` ``` ````, no `http(s)://`, no
  `NAME_\d+`; at least 98% ASCII characters. The ASCII rule is the English filter: the original
  builder used no language-ID field. Seeds are deduplicated by `seed_key = chat:<source>:<blake2b-16>`.
  There is no `used_seeds` exclusion; the WSB build recorded `guard.asserted: false`.
- **WSB pool rule:** the first `4 * n_per_source` unique screened seeds of each source,
  shuffled with one `random.Random(7)` shared across sources (lmsys first), keeping the first 75.
  `build_seeds.py` always re-derives this draw and **fails unless it picks exactly the
  bank's seeds**. dailydialog was verified 75/75 on 2026-11-04. For lmsys, 75 seeds are picked and
  74 are in the bank, because `chat-lmsys-0013` was dropped for having no sites.
- **Expansion draw:** the pool is the first `4 * 3 * 75 = 900` unique screened seeds per source.
  The WSB pool of 300 is its prefix. Remove the 75 WSB picks by key, shuffle the rest with
  `random.Random("hb0028:<source>")`, and keep the first **150 per source** in that seeded order.
  This gives 300 new seeds. Empty rollouts and items with no sites are dropped, as WSB did, which
  should leave ≈298 new items and ≈447 in total. The balance stays 1:1 lmsys/dailydialog. Dedup
  against the WSB 149 holds by construction: the WSB keys are removed, and keys are unique within
  the pool.

## Generation (same settings as the WSB bank)

`scripts/main_x3/gen_modal.py` (Modal, one H200, weights from the `jlens-hf-cache` volume) uses
these settings:

- `Qwen/Qwen3.6-27B`, HF `transformers.generate`, bf16, SDPA;
- left-padded batches of 16;
- `do_sample`, T=1.0, top-p 1.0, top-k 0 (full vocabulary);
- max 512 new tokens;
- chat template with `enable_thinking=False` (an empty think block);
- output cut at the first stop id (EOS not kept);
- empty rollouts dropped.

These match `modal_pipeline.py` stage 1 and the bank meta (`T=1.0, top-p 1.0, top-k 0, max 512
new tokens, seed 7; chat template with an empty think block; EOS not kept`). The one addition is
`torch.manual_seed(7)`. The WSB run seeded only its read-position RNG, so neither run is
bit-reproducible. Labels continue the WSB numbering per source in seeded order
(`chat-lmsys-0075...`, `chat-dailydialog-0075...`).

## Bank (`data/wsbench_chat_x3/`, local until frozen)

`scripts/main_x3/build_bank.py` writes the following files. It uses the Qwen tokenizer, with
per-token decodes for sites and whole-span decodes for `char`, as `chat_bank.py` does.

- **`items.json`:** `{meta, items: [...]}` in the `wsbench_chat` schema plus `subset`
  (`wsb149` | `x3new`) and `seed_key`; `sites[]` gains `wsb: bool`. The 149 WSB items come first,
  in WSB order: they are rebuilt from `data/wsbench_chat/capture_rows.json` ids, and the build
  asserts their item fields equal the WSB copy byte for byte and that their `wsb` sites equal the
  WSB sites. New items follow in label order. Sites are `thin(response_sites(...), 16)`.
- **`capture_rows.json`:** one row per item (`label, family, source, source_id, input_ids,
  prompt_len, eval_positions` = all ≤16 sites, `targets: [], look_for: "", render: plain`).
- **`capture_new.json`:** the same rows restricted to the cells to read. These are every site of a
  new item and only the non-`wsb` sites of a WSB item. Items with nothing to read are omitted.
- **`seeds.json`, `rollouts_*.json`:** the provenance (screen counts, the `verify_wsb` result,
  generation settings, library versions).
- **`MANIFEST.sha256`:** written at freeze.

The bank loads through the existing override `HBENCH_WSBENCH_CHAT_ITEMS=<x3>/items.json`, so the
`wsbench_chat` pipeline runs on it unchanged, with the same schema and extra keys ignored. There
is no new eval name and no CLI change. Data, readouts and bundles stay **local (not pushed) until
frozen**. Code, tests, the plan and the data README are pushed.

## Readouts (Modal, the HB pattern of `scripts/overrefusal/reads.sh`)

`scripts/main_x3/reads.sh [capture|read-s3d|read-mlnla|read-jlens|fetch|merge]` runs from a local
GW main clone (`HBENCH_GW_CAPTURE`) with `/root/.local/bin/modal`:

- **capture:** `wsbench_capture_modal.py --prompts-json capture_new.json --acts-subdir
  acts-hb28-main-x3<TAG> --layers 20:61:4 --max-positions 2048`.
- **s3d:** `wsbench_read_modal.py::main --ckpt
  agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 --merged-name s3d-rl600 --k 3
  --prompt-kind concepts_raw --transform unit`, at all 11 layers. This is one read. The new cells
  feed **both** s3d arms: `s3d-rl600` at its 5 layers (20/36/44/48/60) and `s3d-rl600-all11` at
  all 11. methods.json picks `s3d-rl600` at its layers and `-all11` for the rest, so the index
  never sees the same new readout twice.
- **mlnla:** `wsbench_nla_modal.py --lens mlnla-rl600 --n-samples 3` at all 11 layers.
- **jlens:** `wsbench_tokenlens_modal.py --lens jlens` (top-10) at all 11 layers.
- **final-olens-rl600-x4:** dropped by Camila (2026-11-04). Its producer is also a different path:
  the campaign's plan-file fanout, not these readers.
- **fetch:** `wsbench convert-gen-dir` (wsbench-pub @ 92d763e), then a grid check that the rows
  are exactly `capture_new` × layers.
- **merge:** the old arm file (the WSB cells, unchanged) plus the new rows go to
  `$HBENCH_X3_READOUTS/<arm>/hallucination.jsonl`. This is followed by a full-grid check against
  the x3 bank at the arm's layers: no missing, off-grid or duplicate rows.

Old readout sources:

| arm | old readout file |
|---|---|
| `s3d-rl600` | `/workspace/hb-readouts/s3d-rl600` |
| `s3d-rl600-all11` | `/workspace/hb-readouts/s3d-rl600-all11` |
| `mlnla-rl600-all11` | `/workspace/hb-readouts/mlnla-rl600-all11` |
| `jlens` | `/workspace/hb-readouts/jlens` |

Outputs are written to `/root/tmp`, then rsynced to `/workspace/hb-readouts-x3/` when finished.

**J-lens interp-v1 summaries** (k=1) are the J-lens readout text that a judge reads, so they are
produced here. The command is `hbench run --stages summary --arms jlens` with the bank override,
into the x3 output root. The old cells' `summ:` rows are first copied from the official jlens cache
(`hb-taxonomy-v2/outputs/wsbench_chat/claude-opus-5-5/jlens/cache.jsonl`), so only new cells call
the API. No span, verify, classify or map stage runs.

## Joint all-layer bundle

`scripts/main_x3/joint_bundle.py` writes one JSONL per index method (`s3d-rl600`, `mlnla-rl600`,
`jlens`). Each record is one (item, site):

- `{id, subset, pos, kind, char, token, wsb}`;
- `layers: {"20": {arm, samples[:k]} | {arm, tokens, summary}, ...}` for all 11 layers;
- prose readouts get the first k=3 non-empty samples (`HAL_K`), and J-lens gets the decoded
  top-10 plus the interp-v1 summary (k=1);
- the source arm per layer follows `data/index/methods.json`.

An `index.json` gives counts, the layers, each method's arm-per-layer map, and the sha256 of each
readout file. No judging is done.

## Tests (new files only; no pre-existing test is edited)

`tests/test_main_x3.py`:

1. On the WSB copy (always runs): every one of the 1,123 sites is punctuation/whitespace-only and
   not special. The port reproduces the WSB 8 sites from `capture_rows` ids, and the 16-thin keeps
   them. A tokenizer is needed, so this part is skipped without `HBENCH_QWEN_TOKENIZER`.
2. Pure unit tests: the subset property of `thin` for n = 1 to 400, the screen rules, and `site_kind`.
3. On the x3 bank (skipped when absent): every site is punctuation/whitespace-only and not special,
   no `<|im_end|>`, ≤16 per item, in position order. For each WSB item, the item fields equal the
   WSB copy and its `wsb` sites equal the WSB sites. There are no duplicate seed keys or prompts,
   and the subset counts are as stated.

## Estimated claim-map spend (for planning; nothing judged here)

Let S be the bank's total sites. The WSB items have 2,028 sites at ≤16; the new items are estimated
at ≈13.6 per item. Cells per arm-layer = S. Per-layer claim map = S × 11 per method. The joint
judge = S calls per method (one per site, with all 11 layers). Final numbers are in the PR.

## Steps

1. Seeds: dailydialog done. **lmsys is blocked**: the box's HF token is not on the gated
   `lmsys/lmsys-chat-1m` access list (403), and Camila must grant access or provide the shard.
   `build_seeds.py --sources lmsys` then appends to the same `seeds.json` (same rule, same verify).
2. Generate the dailydialog rollouts now, and lmsys when unblocked.
3. Build the bank, then capture, read, fetch, merge, summaries and bundle. A batch `TAG` lets the
   lmsys half be captured and read later with no change to the done half.
4. Freeze: a manifest, rsync to /workspace, and push the data in a follow-up PR once complete.
