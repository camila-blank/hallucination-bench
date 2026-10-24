# Results 0013: the hallucination index (hindex-v3)

Plan: [`plans/0013-final-sibling-eval-and-hallucination-index.md`](../../plans/0013-final-sibling-eval-and-hallucination-index.md)
(definition, CIs, method mapping). Computed by `uv run hbench index` (also at the end of every
`hbench report`) from the families' existing outputs; nothing is re-judged.

- **main** (`wsbench_chat`, weight 9): weighted % of claims hallucinated (false_claim 1,
  substitution_category 0.5, substitution_sibling 0.5, detail 0.2).
- **agentic_misalignment** (3), **story** (1): unweighted % of claims hallucinated
  (false_claim + substitution_category + detail).
- **sibling** (2): the 3-domain user-modelling ladder (country, capital_city, holiday), domain
  macro of the hallucination share of committed, probe-testable mentions.
- `full` = all four families; `matched` = main + AM + story (9:3:1), the only macro J-lens has.

Reproduces the exploratory 2026-10-20 numbers exactly (max |diff| = 0 on every per-family value,
CI, `full` and `matched`). Command used (outputs live in the family worktrees on the dev box):

```bash
uv run hbench index --dest outputs/index/claude-opus-5-5 \
  --main-root /workspace/hb-taxonomy-v2/outputs/wsbench_chat/claude-opus-5-5 \
  --main-root /workspace/hb-rlrecon/outputs/wsbench_chat/claude-opus-5-5 \
  --story-root /workspace/hb-story/outputs/story/claude-opus-5-5 \
  --am-root /workspace/hb-am-halluc/outputs/agentic_misalignment/claude-opus-5-5 \
  --um-root /workspace/hb-um-ladder/outputs/sibling/um
```

## Table (the macro3 methods)

`mlnla64-rl600-all11` also has main / AM / story rows and so a `matched` value (see the generated
`index.md`); every other arm has main-only rows.

% of claims hallucinated per (method, layer), 95% CIs. Family weights main 9, agentic_misalignment 3, sibling 2, story 1; `matched` = main + agentic_misalignment + story. main leaf weights false_claim 1, substitution_category 0.5, substitution_sibling 0.5, detail 0.2; agentic_misalignment / story: unweighted false_claim + substitution_category + detail; sibling: 3-domain UM ladder.

| method | layer | full | matched | main | agentic_misalignment | sibling | story | main fc / cat / sib / det |
|---|---:|---|---|---|---|---|---|---|
| jlens | 20 | — | 7.9 [6.7, 9.2] | 9.4 [7.9, 11.0] | 3.4 [1.1, 6.3] | — | 8.6 [5.0, 12.7] | 9.2 / 0.2 / 0.1 / 0.1 |
| jlens | 24 | — | 6.4 [5.4, 7.5] | 7.5 [6.2, 8.9] | 3.1 [1.7, 4.9] | — | 7.1 [5.0, 9.4] | 7.4 / 0.1 / 0.0 / 0.3 |
| jlens | 28 | — | 5.3 [4.5, 6.2] | 6.1 [5.0, 7.2] | 2.5 [1.8, 3.2] | — | 7.0 [4.0, 10.7] | 5.9 / 0.1 / 0.0 / 0.4 |
| jlens | 32 | — | 5.5 [4.7, 6.4] | 6.0 [4.9, 7.1] | 4.0 [2.7, 5.3] | — | 6.4 [4.2, 8.7] | 5.8 / 0.1 / 0.2 / 0.3 |
| jlens | 36 | — | 2.8 [2.2, 3.4] | 3.0 [2.3, 3.9] | 1.6 [1.1, 2.2] | — | 4.7 [3.0, 6.5] | 2.9 / 0.0 / 0.1 / 0.3 |
| jlens | 40 | — | 2.6 [2.1, 3.2] | 3.0 [2.3, 3.8] | 1.7 [1.3, 2.1] | — | 2.0 [0.6, 3.7] | 3.0 / 0.0 / 0.1 / 0.1 |
| jlens | 44 | — | 2.3 [1.8, 2.8] | 2.4 [1.8, 3.1] | 1.5 [1.0, 2.0] | — | 3.2 [1.6, 5.0] | 2.3 / 0.0 / 0.1 / 0.4 |
| jlens | 48 | — | 2.9 [2.4, 3.5] | 3.1 [2.4, 3.7] | 2.1 [1.4, 2.8] | — | 4.5 [1.6, 8.2] | 2.9 / 0.1 / 0.1 / 0.3 |
| jlens | 52 | — | 2.6 [2.0, 3.2] | 2.8 [2.1, 3.6] | 1.5 [1.0, 2.3] | — | 4.1 [1.8, 7.3] | 2.6 / 0.0 / 0.2 / 0.5 |
| jlens | 56 | — | 2.7 [2.1, 3.2] | 2.8 [2.2, 3.5] | 1.7 [0.8, 2.8] | — | 4.0 [2.0, 6.1] | 2.6 / 0.1 / 0.2 / 0.4 |
| jlens | 60 | — | 3.1 [2.5, 3.7] | 3.3 [2.6, 4.0] | 2.1 [1.0, 3.5] | — | 4.2 [2.4, 6.0] | 3.1 / 0.0 / 0.2 / 0.5 |
| mlnla-rl600 | 20 | 29.8 [28.5, 31.2] | 27.8 [26.9, 28.8] | 19.9 [18.7, 21.1] | 48.4 [46.7, 50.2] | 42.8 [34.3, 51.0] | 37.8 [35.1, 40.6] | 12.3 / 2.7 / 4.0 / 21.0 |
| mlnla-rl600 | 24 | 29.3 [27.4, 31.1] | 28.8 [27.9, 29.7] | 20.5 [19.4, 21.7] | 50.4 [48.9, 51.8] | 32.6 [19.4, 44.5] | 38.8 [35.6, 41.9] | 12.8 / 3.5 / 3.4 / 21.0 |
| mlnla-rl600 | 28 | 29.0 [27.8, 30.1] | 28.1 [27.2, 28.9] | 20.2 [19.1, 21.3] | 48.2 [46.5, 49.7] | 34.8 [28.0, 41.5] | 38.9 [36.3, 41.6] | 12.1 / 3.7 / 4.1 / 20.7 |
| mlnla-rl600 | 32 | 28.3 [26.7, 30.0] | 27.7 [26.8, 28.7] | 19.6 [18.4, 20.9] | 48.1 [46.4, 49.6] | 32.3 [22.4, 43.8] | 39.9 [37.1, 42.4] | 11.5 / 3.7 / 3.9 / 21.6 |
| mlnla-rl600 | 36 | 26.0 [24.6, 27.4] | 25.6 [24.7, 26.5] | 17.4 [16.2, 18.6] | 46.1 [44.5, 47.5] | 29.1 [20.8, 38.1] | 37.5 [35.3, 40.1] | 10.3 / 2.0 / 3.7 / 21.2 |
| mlnla-rl600 | 40 | 27.0 [25.6, 28.3] | 26.9 [26.1, 27.8] | 18.8 [17.8, 20.0] | 47.9 [46.2, 49.4] | 27.2 [19.4, 36.2] | 37.0 [34.1, 39.7] | 11.4 / 2.4 / 3.9 / 21.3 |
| mlnla-rl600 | 44 | 26.7 [25.3, 28.0] | 26.4 [25.5, 27.2] | 18.1 [17.0, 19.2] | 47.2 [45.6, 48.7] | 28.5 [20.8, 37.7] | 38.6 [36.1, 40.9] | 10.6 / 2.6 / 3.8 / 21.7 |
| mlnla-rl600 | 48 | 27.1 [25.9, 28.3] | 26.8 [25.9, 27.6] | 18.5 [17.4, 19.7] | 47.8 [46.2, 49.3] | 29.3 [22.5, 37.1] | 37.9 [35.3, 40.3] | 11.1 / 2.5 / 3.8 / 21.5 |
| mlnla-rl600 | 52 | 27.0 [25.4, 28.6] | 26.4 [25.5, 27.3] | 18.2 [17.2, 19.4] | 46.9 [45.4, 48.4] | 30.7 [20.8, 42.5] | 38.4 [35.8, 40.9] | 11.5 / 1.5 / 3.1 / 22.2 |
| mlnla-rl600 | 56 | 29.2 [27.7, 30.8] | 28.3 [27.4, 29.3] | 20.7 [19.5, 21.9] | 48.1 [46.7, 49.4] | 35.0 [24.6, 45.1] | 37.9 [35.6, 40.6] | 14.4 / 1.2 / 2.7 / 21.8 |
| mlnla-rl600 | 60 | 31.1 [29.6, 32.5] | 28.6 [27.7, 29.4] | 21.1 [20.1, 22.2] | 47.7 [46.0, 49.3] | 47.4 [39.4, 57.9] | 38.2 [35.2, 41.4] | 15.0 / 1.3 / 2.3 / 21.8 |
| s3d-rl600 | 20 | 16.1 [14.2, 18.0] | 12.5 [11.6, 13.4] | 9.8 [8.7, 11.1] | 17.8 [16.8, 18.8] | 39.4 [29.9, 56.0] | 20.6 [17.1, 23.8] | 6.5 / 1.9 / 3.2 / 3.9 |
| s3d-rl600 | 24 | 14.4 [13.1, 15.6] | 13.7 [12.7, 14.8] | 10.6 [9.3, 12.0] | 21.5 [20.0, 23.2] | 18.7 [13.5, 26.5] | 18.7 [15.5, 21.5] | 7.0 / 2.1 / 3.6 / 3.6 |
| s3d-rl600 | 28 | 15.0 [13.4, 16.5] | 13.2 [12.2, 14.3] | 10.2 [8.9, 11.7] | 19.7 [18.6, 21.0] | 26.1 [17.6, 36.4] | 21.1 [17.4, 24.4] | 6.3 / 2.2 / 4.1 / 3.4 |
| s3d-rl600 | 32 | 13.1 [12.1, 14.2] | 12.5 [11.5, 13.5] | 9.3 [8.0, 10.7] | 19.2 [18.3, 20.2] | 17.1 [12.4, 21.6] | 21.6 [18.1, 24.7] | 5.7 / 1.8 / 3.7 / 4.1 |
| s3d-rl600 | 36 | 13.4 [11.9, 14.8] | 12.2 [11.2, 13.2] | 7.8 [6.6, 9.2] | 21.3 [20.1, 22.4] | 21.1 [12.8, 30.8] | 23.9 [18.9, 28.1] | 4.5 / 1.4 / 3.6 / 4.3 |
| s3d-rl600 | 40 | 12.3 [11.1, 13.5] | 11.6 [10.7, 12.5] | 7.5 [6.4, 8.7] | 20.7 [19.7, 21.6] | 16.7 [9.6, 23.3] | 21.5 [17.4, 25.4] | 4.4 / 1.2 / 3.3 / 4.3 |
| s3d-rl600 | 44 | 13.1 [11.9, 14.4] | 11.8 [10.9, 12.7] | 7.3 [6.2, 8.5] | 22.4 [21.2, 23.5] | 21.9 [14.9, 30.1] | 21.0 [17.2, 24.6] | 4.3 / 1.0 / 3.2 / 4.6 |
| s3d-rl600 | 48 | 13.9 [12.6, 15.2] | 12.0 [11.1, 12.9] | 8.0 [6.9, 9.2] | 21.2 [20.1, 22.2] | 26.0 [18.2, 33.4] | 20.9 [16.8, 24.8] | 4.9 / 1.1 / 3.2 / 4.8 |
| s3d-rl600 | 52 | 13.2 [11.8, 14.6] | 11.6 [10.7, 12.5] | 7.6 [6.4, 8.8] | 20.7 [19.7, 21.7] | 23.7 [15.6, 33.2] | 20.4 [16.3, 24.3] | 4.5 / 0.9 / 3.5 / 4.5 |
| s3d-rl600 | 56 | 12.9 [11.3, 14.5] | 10.8 [9.9, 11.7] | 7.4 [6.2, 8.7] | 18.6 [17.7, 19.5] | 26.8 [17.7, 38.2] | 18.0 [14.3, 21.4] | 4.6 / 0.7 / 3.2 / 4.1 |
| s3d-rl600 | 60 | 13.5 [11.6, 15.5] | 9.8 [9.0, 10.7] | 6.8 [5.7, 7.9] | 16.0 [15.0, 17.0] | 37.5 [25.3, 52.2] | 19.3 [15.3, 23.4] | 4.3 / 0.6 / 2.7 / 4.0 |

Arms used under another name (at one or more layers): mlnla-rl600.agentic_misalignment = mlnla-rl600-all11; mlnla-rl600.main = mlnla-rl600-all11; mlnla-rl600.story = mlnla-rl600-all11; s3d-rl600.main = s3d-rl600-all11; s3d-rl600.sibling = s3d; s3d-rl600.story = s3d-rl600-all11.

Excluded arms: nla-rl-L42, nla-sft-L42.

- jlens.sibling: no sibling score: J-lens has committed, probe-testable UM units in at most two of the three domains at every layer, so only `matched` compares it
- nla-rl-L42: single-layer L42 NLA arms (nla-rl-L42, nla-sft-L42) excluded (Camila 2026-10-20)
- s3d-rl600.main: the s3d-rl600 arm wins at the layers it has, s3d-rl600-all11 fills the rest

## Caveats

- Every family's claim map is **pre-gate** (the held-out gate is chat gold), so these are
  provisional; judge = Opus 5.5, single samples.
- Each (method, layer) is its own row: compare methods at a layer, not averaged over layers.
- **J-lens has no sibling score**: its committed, probe-testable UM units cover at most two of the
  three domains at any layer, so only `matched` compares it with the others.
- Single-layer L42 NLA arms (`nla-rl-L42`, `nla-sft-L42`) are excluded.
- `s3d-rl600` main reads the 5-layer `s3d-rl600` arm where it exists and `s3d-rl600-all11` at the
  other layers; its story score comes from `s3d-rl600-all11`, its sibling score from the ladder's
  `s3d` method (`data/index/methods.json`).
- The sibling family covers 3 domains; occupation (judge committed precision 0.50) and emotion
  were dropped and removed from the repo in plan 0013.

## DeepSeek-V4-Flash (plan 0014)

Plan: [`plans/0014-v4-flash-on-the-index.md`](../../plans/0014-v4-flash-on-the-index.md). The V4
arms are matched by name, and their roots are passed beside the Qwen ones. The Qwen rows above
are unchanged in the combined index, bit for bit. First, the V4 sibling root is exported from
the V4 UM ladder, with its `jlens` renamed to `v4-jlens-std-n1000`:

```bash
V=/workspace/hb-v4/outputs/v4   # V4 run outputs (dev box)
uv run python scripts/v4_um/export_index_root.py --um $V/um --dest $V/sibling/um
uv run hbench index --dest outputs/index/claude-opus-5-5 <the five Qwen roots above> \
  --main-root $V/wsbench_chat/claude-opus-5-5 --story-root $V/story/claude-opus-5-5 \
  --am-root $V/agentic_misalignment/claude-opus-5-5 --um-root $V/sibling/um
```

**No V4 row has a `full` or `matched` value.** V4 agentic_misalignment was read at layers
21/27/33, which are depth-matched to Qwen 32/42/48. V4 chat and story were read at the V4 bench
grid, 15/24/30/36/39. So no V4 layer has main, AM and story together. The index does not pair
nearby layers or average over layers. To get the macros, chat and story must be re-read and
re-judged at 21/27/33, or AM at the chat layers.

V4 has 43 layers and Qwen3.6-27B has 64. A V4 layer number therefore sits at about 2/3 the depth
of the same Qwen layer number. To compare with the Qwen rows above, look at about 1.5× the V4
layer number.

| method | layer | full | matched | main | agentic_misalignment | sibling | story | main fc / cat / sib / det |
|---|---:|---|---|---|---|---|---|---|
| v4-jlens-std-n1000 | 15 | — | — | 4.9 [4.0, 5.9] | — | — | 8.4 [6.1, 10.8] | 4.7 / 0.0 / 0.3 / 0.3 |
| v4-jlens-std-n1000 | 21 | — | — | — | 3.1 [2.3, 4.1] | — | — | — |
| v4-jlens-std-n1000 | 24 | — | — | 4.0 [3.2, 4.8] | — | — | 5.7 [3.3, 8.3] | 3.7 / 0.2 / 0.1 / 0.5 |
| v4-jlens-std-n1000 | 27 | — | — | — | 3.6 [2.4, 4.9] | — | — | — |
| v4-jlens-std-n1000 | 30 | — | — | 2.8 [2.1, 3.5] | — | — | 5.3 [2.6, 8.4] | 2.7 / 0.1 / 0.1 / 0.2 |
| v4-jlens-std-n1000 | 33 | — | — | — | 3.5 [2.0, 5.2] | — | — | — |
| v4-jlens-std-n1000 | 36 | — | — | 3.0 [2.3, 3.8] | — | — | 4.6 [2.7, 6.5] | 2.8 / 0.0 / 0.1 / 0.4 |
| v4-jlens-std-n1000 | 39 | — | — | 4.5 [3.5, 5.4] | — | — | 5.2 [3.0, 7.9] | 4.3 / 0.2 / 0.0 / 0.2 |
| v4-s3don-rl600 | 15 | — | — | 9.8 [8.4, 11.2] | — | 67.9 [59.7, 83.2] | 22.0 [16.8, 27.0] | 5.6 / 2.3 / 4.5 / 3.7 |
| v4-s3don-rl600 | 18 | — | — | — | — | 61.2 [51.8, 68.9] | — | — |
| v4-s3don-rl600 | 21 | — | — | — | 23.0 [22.0, 24.0] | 53.2 [29.1, 69.8] | — | — |
| v4-s3don-rl600 | 24 | — | — | 7.8 [6.6, 9.1] | — | 41.0 [28.0, 54.9] | 18.2 [13.3, 23.6] | 4.7 / 1.5 / 3.3 / 3.5 |
| v4-s3don-rl600 | 27 | — | — | — | 22.3 [21.0, 23.9] | 52.7 [40.4, 66.2] | — | — |
| v4-s3don-rl600 | 30 | — | — | 6.8 [5.6, 8.0] | — | 42.3 [29.9, 54.3] | 14.4 [10.8, 18.0] | 4.4 / 0.6 / 2.5 / 4.3 |
| v4-s3don-rl600 | 33 | — | — | — | 22.4 [20.9, 24.1] | 50.0 [38.5, 60.3] | — | — |
| v4-s3don-rl600 | 36 | — | — | 6.4 [5.2, 7.6] | — | 42.4 [28.5, 62.6] | 14.4 [10.8, 17.8] | 4.5 / 0.4 / 1.9 / 3.5 |
| v4-s3don-rl600 | 39 | — | — | 5.8 [4.7, 7.0] | — | 38.7 [24.5, 51.6] | 13.5 [10.0, 17.1] | 3.8 / 0.5 / 2.0 / 4.0 |
| v4-s3don-rl600 | 42 | — | — | — | — | 59.6 [33.7, 81.2] | — | — |
| v4-s3don-sft | 15 | — | — | 8.0 [6.8, 9.2] | — | 47.6 [40.6, 56.9] | 18.9 [14.1, 23.8] | 5.4 / 2.2 / 1.9 / 2.6 |
| v4-s3don-sft | 18 | — | — | — | — | 13.9 [5.6, 22.2] | — | — |
| v4-s3don-sft | 21 | — | — | — | 16.7 [15.8, 17.6] | 48.9 [36.6, 62.3] | — | — |
| v4-s3don-sft | 24 | — | — | 7.7 [6.5, 8.8] | — | 14.1 [3.0, 36.3] | 17.1 [13.0, 21.0] | 5.5 / 1.1 / 1.8 / 3.6 |
| v4-s3don-sft | 27 | — | — | — | 18.4 [17.2, 19.8] | 45.0 [21.9, 70.7] | — | — |
| v4-s3don-sft | 30 | — | — | 6.0 [5.1, 7.0] | — | 48.9 [25.2, 71.1] | 12.7 [9.2, 16.2] | 4.6 / 0.4 / 1.0 / 3.7 |
| v4-s3don-sft | 33 | — | — | — | 19.2 [18.1, 20.5] | 39.0 [19.4, 52.2] | — | — |
| v4-s3don-sft | 36 | — | — | 6.2 [5.0, 7.3] | — | 12.3 [4.0, 22.2] | 13.9 [10.5, 17.1] | 4.6 / 0.5 / 1.0 / 3.9 |
| v4-s3don-sft | 39 | — | — | 5.9 [4.8, 7.0] | — | 55.7 [37.0, 69.1] | 16.4 [12.6, 20.1] | 4.2 / 0.4 / 1.4 / 3.8 |
| v4-s3don-sft | 42 | — | — | — | — | 26.2 [15.7, 37.7] | — | — |

`v4-s3don-sft14500` is an intermediate SFT checkpoint with chat results only. The generated
`index.md` also has its main rows.

### V4 deviations and caveats

- **The banks were regenerated for V4.**
  - chat: the V4 `hallucination` bank of `wsbench-pub-v4-opus` (149 prompts, V4 responses
    generated with vLLM on 2026-10-12).
  - story: an on-policy V4 bank (`outputs/v4/story_bank/items.json`, 20 items, set with
    `HBENCH_STORY_ITEMS`).
  - AM: the same 32 pinned Qwen3.6-27B rollouts, re-tokenized for V4
    (`scripts/build_am_bank_v4.py`). The read sites come from V4 J-lens columns.
  - UM ladder: V4 responses (from HF `generate`) and V4 probes. The pool of cross-method
    siblings contains only the three V4 arms.
- **Each family uses different layers.** chat and story: 15/24/30/36/39. AM: 21/27/33.
  sibling: every 3rd layer from 15 to 42.
- **Three chat responses are corrupted.** The V4 chat bank came from an unpadded vLLM call
  (`gen_vllm_modal.py`). Its first three items (`chat-lmsys-0000..0002`) are garbage, which is
  the head-of-call failure mode. The other 146, including the tail, are coherent. The official
  values keep the three items. Dropping them would move V4 main by at most 0.1 pt at any layer.
  The other banks are clean:
  - The story bank is coherent throughout. It was also unpadded, but it ran eager with one
    sequence at a time (20 requests).
  - AM contains no V4-generated text.
  - UM used HF `generate`.
- `v4-jlens-std-n1000` has no sibling score, for the same reason as Qwen `jlens`: it has no
  committed, probe-testable country units at any layer.
