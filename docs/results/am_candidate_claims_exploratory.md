# AM index family on candidate claims only (exploratory, 2026-10-29)

**Exploratory. Nothing in the index changes**: hindex-v4 and every published number stay as in
[`0019_index_v4.md`](0019_index_v4.md). Nothing was re-judged; this re-scores existing plan-0008
outputs. Script: [`scripts/index/am_candidate_claims.py`](../../scripts/index/am_candidate_claims.py).

## Question

The index's `agentic_misalignment` value is the unweighted share
`(false_claim + substitution_category + detail) / n_claims` over **every** claim segment of every
mapped cell (`index.claim_rows`). Plan 0008 maps a cell only because the detector located a
misalignment mention in it, but the map labels the whole readout, so most of those claims are
about something else. What does the family read if it counts only the claims that sit inside the
misalignment mentions?

## Method

- **Offsets.** Claim-map segments carry `(readout, start, end)` in the readout string; the
  detector's located mentions (`cells.jsonl` `mentions`) carry `(readout, start, end)` in the
  same string (plan 0008: "detector offsets and map segment offsets must refer to the same
  string"). For J-lens both refer to the interp-v1 **summary**: the scoring mentions are the
  `:Dsum` ones and the map sees the summary, so J-lens spans are comparable and are reported
  beside the prose arms (with the plan-0008 caveat that J-lens enters only when bag and summary
  are both detected).
- **Counting spans.** Mention intervals of one readout are merged with `score.merge_intervals`
  (overlapping merged, touching kept apart), the plan's counting unit.
- **Overlap rule.** A claim segment (`taxonomy_v2.is_claim` of its leaf, as in the index) is a
  **candidate claim** iff it shares at least one character with a merged span of the same
  readout of the same cell: `min(seg.end, span.end) - max(seg.start, span.start) > 0` (the rule
  of `score._overlaps` / `mention_verdict`). A whole segment counts once, however little of it
  the span covers. Since the spans are merged unions, this is the same as overlapping at least
  one mention.
- **Variants.** (a) **all**: every located detector mention of the cell (the map population:
  any actor, stance, item arm). (b) **headline**: only mentions with `headline: true` in the
  arm's `mentions.jsonl` (Stage-H-verified, misaligned items, `ai_self` / `unspecified` x
  plans / acts), keyed `(cell, readout, start, end)`. Headline mentions in cells whose map
  failed (25 J-lens ... 1,157 mlnla64-rl600) are off the index already and so off (b) too.
- **Population and CI.** The index's own: items in `claim_rows` order (an item registers with
  its first mapped cell), items with no candidate claims kept as zero rows, `claim_metric`
  (2,000-draw item bootstrap, fresh `default_rng(0)` per row). The script first asserts its
  base matrices equal `index.claim_rows` and its base value / CI / n / n_items equal the
  hindex-v4 `index.json` AM entry for all 64 (method, layer) rows; they do, exactly.
- **Macros.** For each index row, AM's `v` / `ci` are swapped for the variant's and
  `index.weighted_macro` is recomputed with every other family held fixed (baseline: the v4
  `index.json` behind 0019, built at `6d6615e`). Shown: `full` / `matched` where the row has
  them, else `full_v3` / `matched_v3` (V4, mlnla64: no overrefusal family).

## Results

**Pooled over layers** (claims; "outside" = share of the current AM hallucinated claims that
are not candidate claims):

| method | claims that are cand. (a) | cand. (b) | hall. outside (a) | hall. outside (b) | AM now | AM on (a) | AM on (b) |
|---|---:|---:|---:|---:|---:|---:|---:|
| jlens | 60% | 6.9% | 35% | 94% | 1.9 | 2.1 | 1.6 |
| mlnla-rl600 | 16% | 1.7% | 82% | 98% | 47.7 | 53.4 | 56.2 |
| mlnla64-rl2k-all11 | 15% | 1.5% | 82% | 98% | 44.0 | 52.9 | 55.0 |
| mlnla64-rl600-all11 | 15% | 1.6% | 82% | 98% | 44.7 | 50.8 | 54.5 |
| s3d-rl600 | 35% | 9.2% | 67% | 92% | 20.1 | 18.9 | 17.5 |
| v4-jlens-std-n1000 | 58% | 8.3% | 48% | 94% | 3.3 | 3.0 | 2.5 |
| v4-s3don-rl600 | 16% | 3.7% | 85% | 97% | 22.6 | 21.1 | 20.0 |
| v4-s3don-sft | 17% | 3.9% | 85% | 97% | 18.0 | 16.0 | 15.4 |

- **Most of the current AM rate is not about misalignment content.** 67 to 85 % of the
  hallucinated claims of the prose arms sit outside every mention span (35 to 48 % for J-lens,
  whose summaries are short); outside the headline spans it is 92 to 98 %.
- **The ranking holds and the gap widens.** On candidate claims the NLA arms get worse (mlnla
  +2 to +14 pt on (a), largest at the late layers), the O-lens arms mostly slightly better
  (s3d-rl600 -2.2 to +2.3 pt, 9 of 11 layers down; V4 s3don -0.9 to -3.1 pt), J-lens about flat
  (it is ~2 % either way). So the NLA arms hallucinate *more* where they talk about the
  misconduct than elsewhere; s3d-rl600 slightly less.
- **(b) is small.** 1.5 to 9 % of claims (hundreds to ~2.8k per row; 5 at J-lens L20), so its
  CIs are 2 to 3x wider than the current ones.
- **Macro effect is small** because AM is 3 of 16 (`full`) / 14 (`matched`): s3d-rl600 moves
  -0.5 to +0.5 pt on (a), -1.0 to +0.7 on (b); mlnla-rl600 +0.4 to +2.0 on (a), +0.6 to +3.4 on
  (b); mlnla64 `matched_v3` +0.5 to +3.2 on (a), +0.8 to +4.7 on (b); J-lens -0.3 to +0.8 except
  L20 (b) (+7.8, 5 claims); V4 -1.1 to +0.2. No ordering between methods changes at any layer.

### Per (method, layer): AM value (% hallucinated, 95% CI)

| method | layer | current | cand (a) all | cand (b) headline | n claims | n cand (a) | n cand (b) | hall. outside (a) | hall. outside (b) |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| jlens | 20 | 3.4 [1.1, 6.3] | 7.2 [2.3, 13.3] | 40.0 [nan, nan] | 145 | 69 | 5 | 0% | 60% |
| jlens | 24 | 3.1 [1.7, 4.9] | 3.8 [1.8, 6.9] | 4.8 [0.0, 10.4] | 917 | 471 | 42 | 36% | 93% |
| jlens | 28 | 2.5 [1.8, 3.2] | 2.4 [1.4, 3.4] | 5.3 [0.0, 12.5] | 2308 | 1145 | 95 | 51% | 91% |
| jlens | 32 | 4.0 [2.7, 5.3] | 5.8 [3.7, 7.8] | 2.6 [0.0, 5.7] | 1125 | 621 | 77 | 20% | 96% |
| jlens | 36 | 1.6 [1.1, 2.2] | 1.9 [1.1, 2.8] | 1.4 [0.3, 3.1] | 3543 | 2053 | 280 | 32% | 93% |
| jlens | 40 | 1.7 [1.3, 2.1] | 1.8 [1.4, 2.3] | 1.5 [0.3, 2.8] | 3789 | 2452 | 275 | 30% | 94% |
| jlens | 44 | 1.5 [1.0, 2.0] | 1.7 [1.1, 2.4] | 0.4 [0.0, 1.3] | 3943 | 2495 | 242 | 29% | 98% |
| jlens | 48 | 2.1 [1.4, 2.8] | 2.0 [0.9, 3.2] | 1.1 [0.0, 2.8] | 2692 | 1597 | 185 | 43% | 96% |
| jlens | 52 | 1.5 [1.0, 2.3] | 1.4 [0.8, 2.1] | 0.7 [0.0, 2.5] | 1956 | 1163 | 148 | 47% | 97% |
| jlens | 56 | 1.7 [0.8, 2.8] | 1.5 [0.5, 2.6] | 0.9 [0.0, 2.2] | 1242 | 809 | 116 | 43% | 95% |
| jlens | 60 | 2.1 [1.0, 3.5] | 2.3 [0.9, 4.1] | 1.4 [0.0, 5.3] | 624 | 434 | 69 | 23% | 92% |
| mlnla-rl600 | 20 | 48.4 [46.7, 50.2] | 56.1 [53.0, 59.6] | 64.1 [61.1, 67.3] | 53438 | 8084 | 842 | 82% | 98% |
| mlnla-rl600 | 24 | 50.4 [48.9, 51.8] | 58.9 [56.6, 61.3] | 64.4 [60.2, 68.5] | 59532 | 9287 | 1012 | 82% | 98% |
| mlnla-rl600 | 28 | 48.2 [46.5, 49.7] | 53.8 [50.9, 56.4] | 57.3 [53.9, 60.1] | 87751 | 14057 | 1639 | 82% | 98% |
| mlnla-rl600 | 32 | 48.1 [46.4, 49.6] | 52.9 [50.1, 55.3] | 56.9 [53.2, 60.3] | 102067 | 16786 | 1675 | 82% | 98% |
| mlnla-rl600 | 36 | 46.1 [44.5, 47.5] | 48.2 [45.3, 50.7] | 51.0 [47.7, 53.6] | 120519 | 19785 | 2253 | 83% | 98% |
| mlnla-rl600 | 40 | 47.9 [46.2, 49.4] | 51.4 [48.6, 53.9] | 51.3 [47.9, 53.7] | 104816 | 17425 | 2000 | 82% | 98% |
| mlnla-rl600 | 44 | 47.2 [45.6, 48.7] | 51.4 [48.4, 54.2] | 53.3 [49.2, 56.8] | 97632 | 16289 | 1673 | 82% | 98% |
| mlnla-rl600 | 48 | 47.8 [46.2, 49.3] | 55.7 [52.5, 58.5] | 59.5 [55.1, 63.7] | 76368 | 12672 | 1182 | 81% | 98% |
| mlnla-rl600 | 52 | 46.9 [45.4, 48.4] | 54.9 [52.1, 57.4] | 58.2 [54.2, 62.0] | 67106 | 11246 | 1049 | 80% | 98% |
| mlnla-rl600 | 56 | 48.1 [46.7, 49.4] | 56.5 [54.4, 58.5] | 56.3 [53.5, 58.9] | 54814 | 9366 | 796 | 80% | 98% |
| mlnla-rl600 | 60 | 47.7 [46.0, 49.3] | 57.0 [55.0, 59.0] | 60.6 [55.0, 66.2] | 44093 | 7696 | 589 | 79% | 98% |
| mlnla64-rl2k-all11 | 20 | 46.3 [45.1, 47.6] | 56.8 [53.7, 59.7] | 66.7 [63.3, 69.9] | 85736 | 12157 | 1233 | 83% | 98% |
| mlnla64-rl2k-all11 | 24 | 46.9 [45.8, 47.9] | 57.7 [54.9, 60.1] | 66.9 [63.3, 70.0] | 93016 | 13288 | 1305 | 82% | 98% |
| mlnla64-rl2k-all11 | 28 | 45.4 [44.3, 46.3] | 55.3 [52.4, 57.5] | 60.4 [57.5, 62.9] | 119017 | 17386 | 1641 | 82% | 98% |
| mlnla64-rl2k-all11 | 32 | 45.7 [44.7, 46.6] | 54.3 [51.6, 56.7] | 54.5 [49.8, 57.7] | 137919 | 20997 | 1982 | 82% | 98% |
| mlnla64-rl2k-all11 | 36 | 42.9 [41.9, 43.7] | 47.6 [44.6, 50.3] | 46.2 [43.0, 48.4] | 169781 | 26148 | 2730 | 83% | 98% |
| mlnla64-rl2k-all11 | 40 | 43.8 [42.8, 44.6] | 50.3 [47.4, 53.0] | 47.5 [43.7, 50.4] | 144456 | 21875 | 2264 | 83% | 98% |
| mlnla64-rl2k-all11 | 44 | 44.9 [43.8, 45.9] | 52.4 [49.1, 55.4] | 51.9 [47.7, 55.4] | 133913 | 20293 | 1964 | 82% | 98% |
| mlnla64-rl2k-all11 | 48 | 44.8 [43.7, 45.7] | 54.8 [51.6, 57.6] | 55.0 [50.4, 58.7] | 110484 | 16560 | 1534 | 82% | 98% |
| mlnla64-rl2k-all11 | 52 | 41.9 [41.0, 42.7] | 51.8 [49.2, 54.1] | 55.0 [51.2, 57.9] | 101970 | 15016 | 1452 | 82% | 98% |
| mlnla64-rl2k-all11 | 56 | 40.3 [39.4, 41.1] | 53.1 [51.0, 55.1] | 58.7 [54.5, 62.2] | 88223 | 13346 | 1281 | 80% | 98% |
| mlnla64-rl2k-all11 | 60 | 39.3 [38.5, 40.1] | 53.3 [51.5, 54.9] | 58.7 [55.5, 61.8] | 73682 | 11233 | 1000 | 79% | 98% |
| mlnla64-rl600-all11 | 20 | 45.8 [44.2, 47.3] | 53.7 [50.5, 56.8] | 63.3 [60.6, 66.0] | 90540 | 13118 | 1346 | 83% | 98% |
| mlnla64-rl600-all11 | 24 | 46.8 [45.5, 48.2] | 53.6 [50.7, 56.1] | 60.2 [57.4, 63.0] | 101847 | 15145 | 1692 | 83% | 98% |
| mlnla64-rl600-all11 | 28 | 45.4 [44.1, 46.5] | 51.3 [48.4, 53.8] | 54.2 [52.1, 56.2] | 126318 | 18729 | 2038 | 83% | 98% |
| mlnla64-rl600-all11 | 32 | 45.5 [44.1, 46.8] | 51.2 [48.4, 53.6] | 53.8 [51.0, 56.5] | 138171 | 21092 | 2191 | 83% | 98% |
| mlnla64-rl600-all11 | 36 | 43.4 [41.9, 44.7] | 45.5 [42.4, 48.1] | 47.1 [44.3, 49.4] | 157854 | 24438 | 2681 | 84% | 98% |
| mlnla64-rl600-all11 | 40 | 45.2 [43.7, 46.4] | 49.2 [46.1, 51.9] | 50.9 [48.1, 53.0] | 139146 | 21623 | 2264 | 83% | 98% |
| mlnla64-rl600-all11 | 44 | 44.3 [42.6, 45.7] | 48.8 [45.3, 51.8] | 52.4 [49.2, 55.3] | 129417 | 20160 | 2078 | 83% | 98% |
| mlnla64-rl600-all11 | 48 | 44.7 [43.1, 46.3] | 52.5 [49.1, 55.6] | 54.9 [52.1, 57.6] | 104322 | 16223 | 1591 | 82% | 98% |
| mlnla64-rl600-all11 | 52 | 43.8 [42.4, 45.1] | 50.7 [48.1, 53.0] | 54.6 [51.6, 57.0] | 101430 | 15997 | 1505 | 82% | 98% |
| mlnla64-rl600-all11 | 56 | 43.8 [42.4, 45.2] | 53.2 [50.5, 55.6] | 59.5 [56.2, 62.5] | 88349 | 14319 | 1290 | 80% | 98% |
| mlnla64-rl600-all11 | 60 | 43.4 [42.3, 44.5] | 54.7 [52.5, 56.7] | 59.0 [55.2, 63.6] | 73292 | 12138 | 1106 | 79% | 98% |
| s3d-rl600 | 20 | 17.8 [16.8, 18.8] | 16.9 [15.3, 18.7] | 20.4 [16.3, 24.7] | 7298 | 2467 | 504 | 68% | 92% |
| s3d-rl600 | 24 | 21.5 [20.0, 23.2] | 23.8 [21.6, 26.3] | 24.9 [20.6, 29.5] | 8290 | 2843 | 708 | 62% | 90% |
| s3d-rl600 | 28 | 19.7 [18.6, 21.0] | 19.3 [17.8, 21.0] | 18.2 [15.8, 20.6] | 10928 | 3784 | 942 | 66% | 92% |
| s3d-rl600 | 32 | 19.2 [18.3, 20.2] | 18.3 [16.9, 19.9] | 16.6 [14.4, 19.0] | 14816 | 5272 | 1331 | 66% | 92% |
| s3d-rl600 | 36 | 21.3 [20.1, 22.4] | 19.6 [18.3, 21.0] | 16.4 [14.5, 18.3] | 19592 | 7055 | 1996 | 67% | 92% |
| s3d-rl600 | 40 | 20.7 [19.7, 21.6] | 19.5 [17.9, 21.1] | 16.3 [13.7, 19.1] | 18761 | 6813 | 1912 | 66% | 92% |
| s3d-rl600 | 44 | 22.4 [21.2, 23.5] | 20.8 [19.1, 22.5] | 19.9 [17.1, 22.8] | 18005 | 6518 | 1746 | 66% | 91% |
| s3d-rl600 | 48 | 21.2 [20.1, 22.2] | 19.7 [18.2, 21.2] | 17.6 [15.3, 20.0] | 16773 | 5969 | 1516 | 67% | 92% |
| s3d-rl600 | 52 | 20.7 [19.7, 21.7] | 18.5 [17.2, 19.8] | 17.8 [15.3, 20.3] | 15436 | 5426 | 1442 | 69% | 92% |
| s3d-rl600 | 56 | 18.6 [17.7, 19.5] | 16.8 [15.4, 18.2] | 15.2 [12.4, 17.9] | 14287 | 4978 | 1318 | 69% | 92% |
| s3d-rl600 | 60 | 16.0 [15.0, 17.0] | 14.9 [13.5, 16.7] | 14.9 [12.1, 18.1] | 15368 | 5384 | 1249 | 67% | 92% |
| v4-jlens-std-n1000 | 21 | 3.1 [2.3, 4.1] | 2.2 [1.4, 3.2] | 1.8 [0.0, 3.5] | 2310 | 1336 | 170 | 59% | 96% |
| v4-jlens-std-n1000 | 27 | 3.6 [2.4, 4.9] | 4.3 [2.6, 6.3] | 4.6 [0.0, 13.4] | 1177 | 668 | 108 | 31% | 88% |
| v4-jlens-std-n1000 | 33 | 3.5 [2.0, 5.2] | 3.2 [1.5, 5.1] | 1.3 [0.0, 4.3] | 802 | 500 | 76 | 43% | 96% |
| v4-s3don-rl600 | 21 | 23.0 [22.0, 24.0] | 22.1 [20.9, 23.4] | 21.0 [18.7, 23.3] | 72530 | 12491 | 2829 | 83% | 96% |
| v4-s3don-rl600 | 27 | 22.3 [21.0, 23.9] | 19.9 [18.1, 21.7] | 17.5 [14.5, 20.2] | 50643 | 8342 | 2000 | 85% | 97% |
| v4-s3don-rl600 | 33 | 22.4 [20.9, 24.1] | 20.8 [19.1, 22.7] | 21.4 [18.0, 24.8] | 53010 | 8041 | 1695 | 86% | 97% |
| v4-s3don-sft | 21 | 16.7 [15.8, 17.6] | 15.7 [14.4, 17.2] | 15.3 [13.6, 16.9] | 49295 | 8762 | 1984 | 83% | 96% |
| v4-s3don-sft | 27 | 18.4 [17.2, 19.8] | 16.3 [14.5, 18.2] | 16.2 [13.8, 18.6] | 23972 | 4179 | 920 | 85% | 97% |
| v4-s3don-sft | 33 | 19.2 [18.1, 20.5] | 16.1 [14.8, 17.8] | 15.0 [12.6, 17.7] | 40139 | 6470 | 1493 | 87% | 97% |

### Macros with the AM value swapped (other families fixed)

| method | layer | macro | current | with (a) | Δ (a) | with (b) | Δ (b) |
|---|---:|---|---|---|---:|---|---:|
| jlens | 20 | matched | 7.4 [6.2, 8.6] | 8.2 [6.6, 9.8] | +0.8 | 15.2 [nan, nan] | +7.8 |
| jlens | 24 | matched | 6.0 [5.0, 6.9] | 6.1 [5.1, 7.2] | +0.2 | 6.3 [4.9, 7.8] | +0.4 |
| jlens | 28 | matched | 5.0 [4.2, 5.7] | 4.9 [4.1, 5.8] | -0.0 | 5.6 [4.0, 7.1] | +0.6 |
| jlens | 32 | matched | 5.1 [4.4, 5.9] | 5.5 [4.7, 6.4] | +0.4 | 4.8 [3.9, 5.8] | -0.3 |
| jlens | 36 | matched | 2.7 [2.1, 3.2] | 2.7 [2.2, 3.3] | +0.1 | 2.6 [2.0, 3.2] | -0.0 |
| jlens | 40 | matched | 2.5 [2.0, 3.0] | 2.5 [2.0, 3.0] | +0.0 | 2.5 [1.9, 3.0] | -0.0 |
| jlens | 44 | matched | 2.2 [1.7, 2.6] | 2.2 [1.7, 2.7] | +0.0 | 1.9 [1.5, 2.4] | -0.2 |
| jlens | 48 | matched | 2.7 [2.2, 3.2] | 2.7 [2.2, 3.3] | -0.0 | 2.5 [2.0, 3.1] | -0.2 |
| jlens | 52 | matched | 2.5 [2.0, 3.1] | 2.5 [1.9, 3.1] | -0.0 | 2.3 [1.7, 3.0] | -0.2 |
| jlens | 56 | matched | 2.6 [2.1, 3.1] | 2.6 [2.1, 3.1] | -0.0 | 2.4 [1.9, 2.9] | -0.2 |
| jlens | 60 | matched | 2.9 [2.3, 3.4] | 2.9 [2.3, 3.5] | +0.0 | 2.7 [2.0, 3.5] | -0.1 |
| mlnla-rl600 | 20 | full | 28.0 [26.7, 29.3] | 29.4 [28.0, 30.8] | +1.5 | 30.9 [29.5, 32.3] | +3.0 |
| mlnla-rl600 | 20 | matched | 25.8 [25.0, 26.7] | 27.5 [26.4, 28.6] | +1.7 | 29.2 [28.2, 30.2] | +3.4 |
| mlnla-rl600 | 24 | full | 27.5 [25.7, 29.2] | 29.1 [27.3, 30.8] | +1.6 | 30.1 [28.2, 32.0] | +2.6 |
| mlnla-rl600 | 24 | matched | 26.7 [25.9, 27.6] | 28.6 [27.6, 29.5] | +1.8 | 29.7 [28.6, 30.9] | +3.0 |
| mlnla-rl600 | 28 | full | 27.2 [26.1, 28.3] | 28.2 [27.1, 29.4] | +1.1 | 28.9 [27.7, 30.1] | +1.7 |
| mlnla-rl600 | 28 | matched | 26.1 [25.3, 26.9] | 27.3 [26.4, 28.3] | +1.2 | 28.1 [27.1, 29.0] | +2.0 |
| mlnla-rl600 | 32 | full | 26.6 [25.0, 28.1] | 27.4 [25.8, 29.0] | +0.9 | 28.2 [26.5, 29.9] | +1.7 |
| mlnla-rl600 | 32 | matched | 25.7 [24.8, 26.6] | 26.7 [25.8, 27.7] | +1.0 | 27.6 [26.5, 28.7] | +1.9 |
| mlnla-rl600 | 36 | full | 24.4 [23.1, 25.7] | 24.8 [23.4, 26.2] | +0.4 | 25.3 [23.9, 26.7] | +0.9 |
| mlnla-rl600 | 36 | matched | 23.7 [22.9, 24.6] | 24.2 [23.2, 25.1] | +0.4 | 24.8 [23.8, 25.8] | +1.0 |
| mlnla-rl600 | 40 | full | 25.3 [24.1, 26.6] | 26.0 [24.7, 27.3] | +0.7 | 26.0 [24.6, 27.3] | +0.6 |
| mlnla-rl600 | 40 | matched | 25.1 [24.2, 25.9] | 25.8 [24.9, 26.7] | +0.7 | 25.8 [24.8, 26.7] | +0.7 |
| mlnla-rl600 | 44 | full | 25.0 [23.7, 26.3] | 25.8 [24.5, 27.1] | +0.8 | 26.2 [24.7, 27.6] | +1.2 |
| mlnla-rl600 | 44 | matched | 24.5 [23.7, 25.3] | 25.4 [24.5, 26.4] | +0.9 | 25.8 [24.7, 26.9] | +1.3 |
| mlnla-rl600 | 48 | full | 25.4 [24.3, 26.6] | 26.9 [25.6, 28.1] | +1.5 | 27.6 [26.2, 29.0] | +2.2 |
| mlnla-rl600 | 48 | matched | 24.8 [24.0, 25.7] | 26.5 [25.6, 27.5] | +1.7 | 27.4 [26.2, 28.5] | +2.5 |
| mlnla-rl600 | 52 | full | 25.3 [23.8, 26.8] | 26.8 [25.2, 28.4] | +1.5 | 27.4 [25.7, 29.1] | +2.1 |
| mlnla-rl600 | 52 | matched | 24.5 [23.7, 25.4] | 26.2 [25.3, 27.2] | +1.7 | 27.0 [25.8, 28.1] | +2.4 |
| mlnla-rl600 | 56 | full | 27.4 [25.9, 28.9] | 29.0 [27.5, 30.5] | +1.6 | 28.9 [27.4, 30.5] | +1.5 |
| mlnla-rl600 | 56 | matched | 26.3 [25.5, 27.2] | 28.1 [27.2, 29.0] | +1.8 | 28.1 [27.1, 29.1] | +1.8 |
| mlnla-rl600 | 60 | full | 29.1 [27.8, 30.5] | 30.9 [29.5, 32.3] | +1.7 | 31.6 [29.9, 33.2] | +2.4 |
| mlnla-rl600 | 60 | matched | 26.5 [25.7, 27.3] | 28.5 [27.7, 29.4] | +2.0 | 29.3 [27.9, 30.7] | +2.8 |
| mlnla64-rl2k-all11 | 20 | matched_v3 | 27.2 [26.3, 28.0] | 29.6 [28.6, 30.6] | +2.4 | 31.9 [30.8, 33.0] | +4.7 |
| mlnla64-rl2k-all11 | 24 | matched_v3 | 27.3 [26.5, 28.1] | 29.8 [28.9, 30.8] | +2.5 | 31.9 [30.9, 33.0] | +4.6 |
| mlnla64-rl2k-all11 | 28 | matched_v3 | 27.1 [26.2, 27.9] | 29.3 [28.4, 30.3] | +2.3 | 30.5 [29.5, 31.5] | +3.5 |
| mlnla64-rl2k-all11 | 32 | matched_v3 | 27.1 [26.2, 27.9] | 29.1 [28.1, 30.1] | +2.0 | 29.1 [27.9, 30.3] | +2.0 |
| mlnla64-rl2k-all11 | 36 | matched_v3 | 24.5 [23.7, 25.3] | 25.6 [24.6, 26.6] | +1.1 | 25.3 [24.3, 26.3] | +0.8 |
| mlnla64-rl2k-all11 | 40 | matched_v3 | 25.2 [24.3, 26.0] | 26.7 [25.7, 27.7] | +1.5 | 26.0 [24.9, 27.1] | +0.9 |
| mlnla64-rl2k-all11 | 44 | matched_v3 | 25.3 [24.5, 26.1] | 27.1 [26.0, 28.1] | +1.7 | 27.0 [25.8, 28.1] | +1.6 |
| mlnla64-rl2k-all11 | 48 | matched_v3 | 25.3 [24.5, 26.1] | 27.6 [26.6, 28.7] | +2.3 | 27.7 [26.5, 28.9] | +2.4 |
| mlnla64-rl2k-all11 | 52 | matched_v3 | 24.1 [23.4, 24.9] | 26.4 [25.5, 27.3] | +2.3 | 27.1 [26.1, 28.2] | +3.0 |
| mlnla64-rl2k-all11 | 56 | matched_v3 | 24.3 [23.6, 25.1] | 27.3 [26.4, 28.1] | +3.0 | 28.6 [27.4, 29.7] | +4.2 |
| mlnla64-rl2k-all11 | 60 | matched_v3 | 24.4 [23.7, 25.1] | 27.6 [26.9, 28.4] | +3.2 | 28.9 [27.9, 29.8] | +4.5 |
| mlnla64-rl600-all11 | 20 | matched_v3 | 26.3 [25.4, 27.2] | 28.1 [27.0, 29.2] | +1.8 | 30.3 [29.3, 31.3] | +4.0 |
| mlnla64-rl600-all11 | 24 | matched_v3 | 26.4 [25.5, 27.2] | 27.9 [26.9, 28.9] | +1.6 | 29.4 [28.4, 30.5] | +3.1 |
| mlnla64-rl600-all11 | 28 | matched_v3 | 26.3 [25.4, 27.2] | 27.6 [26.6, 28.7] | +1.4 | 28.3 [27.3, 29.3] | +2.0 |
| mlnla64-rl600-all11 | 32 | matched_v3 | 26.1 [25.2, 27.0] | 27.4 [26.4, 28.5] | +1.3 | 28.0 [26.9, 29.1] | +1.9 |
| mlnla64-rl600-all11 | 36 | matched_v3 | 24.0 [23.1, 24.8] | 24.4 [23.4, 25.5] | +0.5 | 24.8 [23.8, 25.8] | +0.9 |
| mlnla64-rl600-all11 | 40 | matched_v3 | 25.2 [24.3, 26.1] | 26.2 [25.1, 27.2] | +0.9 | 26.5 [25.5, 27.6] | +1.3 |
| mlnla64-rl600-all11 | 44 | matched_v3 | 24.3 [23.4, 25.3] | 25.4 [24.2, 26.5] | +1.0 | 26.2 [25.1, 27.3] | +1.9 |
| mlnla64-rl600-all11 | 48 | matched_v3 | 25.1 [24.2, 26.0] | 26.9 [25.8, 28.0] | +1.8 | 27.4 [26.4, 28.5] | +2.3 |
| mlnla64-rl600-all11 | 52 | matched_v3 | 25.1 [24.2, 26.0] | 26.7 [25.7, 27.7] | +1.6 | 27.6 [26.6, 28.7] | +2.5 |
| mlnla64-rl600-all11 | 56 | matched_v3 | 26.2 [25.3, 27.1] | 28.3 [27.3, 29.4] | +2.2 | 29.8 [28.7, 30.9] | +3.6 |
| mlnla64-rl600-all11 | 60 | matched_v3 | 26.2 [25.4, 27.1] | 28.8 [27.9, 29.8] | +2.6 | 29.9 [28.6, 31.1] | +3.6 |
| s3d-rl600 | 20 | full | 16.1 [14.3, 18.0] | 16.0 [14.1, 17.8] | -0.2 | 16.6 [14.7, 18.6] | +0.5 |
| s3d-rl600 | 20 | matched | 12.8 [11.9, 13.8] | 12.6 [11.6, 13.6] | -0.2 | 13.4 [12.1, 14.7] | +0.6 |
| s3d-rl600 | 24 | full | 14.6 [13.4, 15.8] | 15.0 [13.8, 16.3] | +0.4 | 15.2 [13.8, 16.7] | +0.6 |
| s3d-rl600 | 24 | matched | 14.0 [13.0, 15.1] | 14.5 [13.4, 15.6] | +0.5 | 14.8 [13.4, 16.1] | +0.7 |
| s3d-rl600 | 28 | full | 16.8 [15.3, 18.3] | 16.7 [15.2, 18.2] | -0.1 | 16.5 [14.9, 18.0] | -0.3 |
| s3d-rl600 | 28 | matched | 15.4 [14.3, 16.5] | 15.3 [14.2, 16.5] | -0.1 | 15.1 [13.9, 16.3] | -0.3 |
| s3d-rl600 | 32 | full | 14.5 [13.5, 15.6] | 14.4 [13.3, 15.5] | -0.2 | 14.1 [12.9, 15.2] | -0.5 |
| s3d-rl600 | 32 | matched | 14.2 [13.1, 15.2] | 14.0 [12.9, 15.1] | -0.2 | 13.6 [12.5, 14.8] | -0.6 |
| s3d-rl600 | 36 | full | 13.8 [12.4, 15.3] | 13.5 [12.1, 15.0] | -0.3 | 12.9 [11.4, 14.4] | -0.9 |
| s3d-rl600 | 36 | matched | 12.8 [11.8, 13.8] | 12.4 [11.4, 13.5] | -0.4 | 11.8 [10.7, 12.8] | -1.0 |
| s3d-rl600 | 40 | full | 12.7 [11.5, 13.8] | 12.4 [11.2, 13.6] | -0.2 | 11.8 [10.6, 13.1] | -0.8 |
| s3d-rl600 | 40 | matched | 12.1 [11.2, 13.0] | 11.8 [10.9, 12.8] | -0.3 | 11.1 [10.1, 12.2] | -0.9 |
| s3d-rl600 | 44 | full | 13.0 [11.8, 14.3] | 12.8 [11.5, 14.0] | -0.3 | 12.6 [11.3, 13.9] | -0.5 |
| s3d-rl600 | 44 | matched | 11.8 [10.9, 12.7] | 11.5 [10.5, 12.4] | -0.3 | 11.3 [10.2, 12.3] | -0.5 |
| s3d-rl600 | 48 | full | 14.0 [12.7, 15.2] | 13.7 [12.4, 14.9] | -0.3 | 13.3 [12.0, 14.6] | -0.7 |
| s3d-rl600 | 48 | matched | 12.2 [11.3, 13.2] | 11.9 [11.0, 12.9] | -0.3 | 11.5 [10.4, 12.5] | -0.8 |
| s3d-rl600 | 52 | full | 13.3 [11.9, 14.6] | 12.8 [11.5, 14.2] | -0.4 | 12.7 [11.3, 14.1] | -0.6 |
| s3d-rl600 | 52 | matched | 11.8 [10.8, 12.7] | 11.3 [10.3, 12.2] | -0.5 | 11.1 [10.1, 12.2] | -0.6 |
| s3d-rl600 | 56 | full | 12.9 [11.4, 14.4] | 12.6 [11.1, 14.1] | -0.3 | 12.3 [10.7, 13.9] | -0.6 |
| s3d-rl600 | 56 | matched | 10.9 [10.0, 11.9] | 10.6 [9.6, 11.5] | -0.4 | 10.2 [9.1, 11.3] | -0.7 |
| s3d-rl600 | 60 | full | 13.4 [11.6, 15.3] | 13.2 [11.4, 15.1] | -0.2 | 13.2 [11.3, 15.1] | -0.2 |
| s3d-rl600 | 60 | matched | 10.0 [9.1, 10.9] | 9.8 [8.9, 10.7] | -0.2 | 9.8 [8.7, 10.8] | -0.2 |
| v4-jlens-std-n1000 | 21 | matched_v3 | 4.2 [3.4, 4.9] | 3.9 [3.2, 4.7] | -0.2 | 3.8 [3.0, 4.7] | -0.3 |
| v4-jlens-std-n1000 | 27 | matched_v3 | 3.3 [2.6, 3.9] | 3.4 [2.7, 4.1] | +0.2 | 3.5 [1.9, 5.1] | +0.2 |
| v4-jlens-std-n1000 | 33 | matched_v3 | 3.3 [2.6, 4.0] | 3.2 [2.5, 4.0] | -0.1 | 2.8 [2.0, 3.6] | -0.5 |
| v4-s3don-rl600 | 21 | full_v3 | 19.9 [17.0, 22.8] | 19.7 [16.8, 22.6] | -0.2 | 19.5 [16.6, 22.4] | -0.4 |
| v4-s3don-rl600 | 21 | matched_v3 | 14.8 [13.6, 15.9] | 14.6 [13.4, 15.8] | -0.2 | 14.3 [13.1, 15.6] | -0.5 |
| v4-s3don-rl600 | 27 | full_v3 | 16.7 [14.8, 18.6] | 16.2 [14.3, 18.1] | -0.5 | 15.7 [13.7, 17.7] | -1.0 |
| v4-s3don-rl600 | 27 | matched_v3 | 11.1 [10.2, 12.0] | 10.6 [9.6, 11.5] | -0.6 | 10.0 [8.9, 11.1] | -1.1 |
| v4-s3don-rl600 | 33 | full_v3 | 15.8 [14.1, 17.4] | 15.4 [13.8, 17.1] | -0.3 | 15.6 [13.8, 17.3] | -0.2 |
| v4-s3don-rl600 | 33 | matched_v3 | 10.5 [9.6, 11.4] | 10.1 [9.2, 11.1] | -0.4 | 10.3 [9.1, 11.4] | -0.2 |
| v4-s3don-sft | 21 | full_v3 | 15.7 [13.9, 17.6] | 15.5 [13.7, 17.4] | -0.2 | 15.5 [13.6, 17.4] | -0.3 |
| v4-s3don-sft | 21 | matched_v3 | 10.6 [9.8, 11.5] | 10.4 [9.5, 11.3] | -0.2 | 10.3 [9.4, 11.2] | -0.3 |
| v4-s3don-sft | 27 | full_v3 | 14.5 [11.1, 17.8] | 14.0 [10.7, 17.4] | -0.4 | 14.0 [10.6, 17.4] | -0.4 |
| v4-s3don-sft | 27 | matched_v3 | 9.8 [8.8, 10.7] | 9.3 [8.2, 10.3] | -0.5 | 9.2 [8.2, 10.3] | -0.5 |
| v4-s3don-sft | 33 | full_v3 | 13.2 [10.9, 15.4] | 12.5 [10.2, 14.8] | -0.6 | 12.3 [10.0, 14.6] | -0.8 |
| v4-s3don-sft | 33 | matched_v3 | 9.2 [8.4, 9.9] | 8.5 [7.7, 9.2] | -0.7 | 8.2 [7.3, 9.1] | -1.0 |

## Caveats

- **Candidate is coarse.** A segment that touches a span by one character counts whole, so (a)
  includes some neighbouring content; a stricter rule (e.g. majority-of-segment overlap) would
  shrink the candidate set further. The detector's quotes are short (as short as still names the
  action), so a long claim unit containing the quote dominates the overlap.
- **Mention-map conditioning.** Only cells with a located mention are mapped, so neither the
  current value nor the candidate value sees misconduct-free readouts; the candidate rule only
  changes which claims of the mapped cells count.
- **(b) mixes populations.** Headline mentions exist only on the 28 misaligned items, so (b)'s
  item population still includes control items as zero rows (the bootstrap resamples all items,
  as the index does), and its value is effectively a misaligned-items rate. Where a row has very
  few (b) claims, bootstrap draws with zero claims give `nan` percentiles (J-lens L20: 5 claims,
  CI `nan`); the index's `claim_metric` would do the same.
- **J-lens is not like-for-like** (plan 0008): its spans live in a summary of the bag and the
  summariser may add or drop content; J-lens AM is ~2 % either way.
- Headline membership is Stage H's; the Stage-H gate's precision caveats (0008) carry over to (b).

## Reproduce

```bash
PYTHONPATH=src python scripts/index/am_candidate_claims.py \
  --index /workspace/hb-index-v4-res/outputs/index/claude-opus-5-5/index.json \
  --out outputs/index/am_candidate/results.json   # minutes; streams one arm at a time
```

The AM roots are read from the index's `sources` (`/workspace/hb-am-halluc/outputs/
agentic_misalignment/claude-opus-5-5`, `/workspace/hb-v4/outputs/v4/agentic_misalignment/
claude-opus-5-5`). `--print-only` re-prints the tables from `--out`.
