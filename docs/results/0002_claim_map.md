# Results 0002 — the claim map (taxonomy v2)

Plan: [`plans/0002-claim-map-taxonomy-v2.md`](../../plans/0002-claim-map-taxonomy-v2.md)
(Revision 10). Status: **stub** — implementation, smoke and dev calibration done; held-out gate
and the full `:M` run pending (the held-out split is written by a blind author).

## Setup

- Judge `claude-opus-5-5`, keyless Anthropic auth, no temperature, `output_config.effort=low`,
  structured output (`claimmap_v2.SCHEMA`). Prompt version `claimmap-v2`, cache key `{key}:M`.
- One call per non-empty, span-judged cell. Established spans = v5c verified wrong strings
  (revoked or not), each its own unit. Verify outputs are never shown to the judge.
- Open decisions use their defaults (`taxonomy_v2.CHOSEN`): Picasso-for-da-Vinci is a
  substitution; invented arbitrary specifics are `detail`; `cruxy` is a tag; the conversation
  wins over the world (tag `world_false`); hedged user/setting inferences are `other_on_topic`.

## Smoke (2026-10-11; copies of the plan-0001 caches, 10 cells per arm)

| arm | cells | failed | units | established | splits | unlocated sub-spans | forced | disputed | mean out tok | max out tok | mean $/cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s3d-rl600 L44 | 10 | 0 | 247 | 10 | 25 | 0 | 0 | 0 | 2,667 | 3,194 | 0.092 |
| nla-rl-L42 | 10 | 0 | 341 | 75 | 51 | 0 | 0 | 2 | 3,710 | 5,218 | 0.115 |
| jlens L44 | 10 | 0 | 29 | 4 | 26 | 1 | 0 | 0 | 645 | 817 | 0.049 |

No parse failures, missing or duplicate units, or unlocated established spans. The smoke
$/cell includes a cache write of the ~6.6k-token system prompt on every call (all calls were
concurrent); a long run reads the cache instead (~$0.03 less per call). Worst case 5,218 output
tokens: `max_tokens` 16,000 (the default) is ample (+50% = 7.8k).

## Calibration

### dev (`dev_v2.jsonl`, 27 synthetic cases, 130 gold units, 141 gold segments, 2 runs; $0.97)

| measure | mean | Wilson 95% | threshold | pass |
|---|---|---|---|---|
| group accuracy (7 groups), unit level | 0.954 | [0.903, 0.979] | 0.85 | yes |
| group accuracy, character level | 0.945 | [0.938, 0.952] | 0.85 | yes |
| leaf accuracy (19 leaves), unit level | 0.950 | [0.898, 0.976] | 0.70 | yes |
| recall false_claim (segment) | 1.000 | [0.701, 1.000] | 0.75 | yes |
| recall detail (segment) | 1.000 | [0.566, 1.000] | 0.75 | no (n = 5: Wilson bound) |
| recall substitution (segment) | 0.944 | [0.629, 0.994] | 0.75 | yes |
| recall correct (segment) | 0.900 | [0.799, 0.953] | 0.75 | yes |
| sibling-vs-category (segment) | 1.000 | [0.701, 1.000] | 0.75 | yes |
| claim/non-claim char F1 | 0.985 | — | 0.85 | yes |
| group flip rate | 0.031 | [0.012, 0.076] | 0.07 | yes |
| unlocated sub-spans / splits | 0.000 | — | 0.02 | yes |

Weakest leaves on dev: `ambiguous_topic` (0.50, read as `off_topic`), `restates_prefix`
(0.78, often read as `generic` for one-line topic descriptions).

### held-out

Pending (blind author; ≤ 3 checks, logged in `data/calibration/README.md`).

## Headline tables per arm

Pending the full `:M` run (`hbench report --view v2`).

## Spend

Implementation and smoke: about $3.5 (`:M` smoke 30 cells $2.55, dev calibration $0.97).
Full-run estimate at the smoke rates with cache reads: ~$0.06 (prose) / ~$0.02 (tokens) per
cell, i.e. on the order of $2–3k for ~48k cells.

## Caveats

- Gold (dev and held-out real part) is labelled by Claude, the judge's model family, so
  agreement is optimistic.
- Token arms are judged through the `interp-v1` summariser, which commits to a specific reading.
- The transition table assigns each readout the first group present in the order
  hallucination > unknown_incorrect > co-activation > thinky > unsettled > correct > junk >
  non-claim (an implementation choice; the plan does not fix one).

## Rev 13 (claimmap-v2.3): single-call pilot on s3d-rl600 L44 — PRE-GATE

Option A tree (20 leaves / 8 groups, supplementary branch, settlement-not-subject rule),
verdict layer, probe targets, single-call `:M` (no established spans, no verify). Dev
calibration (single-call, 34 cases, 2 runs, $1.42): every dev row passes (group acc 0.940,
verdict acc 0.940, leaf acc 0.921, flip 0.044). Held-out not yet re-checked, so the pilot is
pre-gate.

Pilot: 1,123 cells, 11 failed after one retry pass (1.0%), `:M` spend $59.61. Char-weighted
verdict shares: hallucination 7.5% [6.0, 9.2], probe_candidate 47.6%, correct 32.6%, unresolved
2.3%, junk 1.7%, non_claim 7.2%, failed 1.0%. Readout-level hallucination rate (v2) 24.4%
[20.9, 27.9] vs v5c 46.3% [42.1, 50.4] on the same 3,333 readouts (full comparison: v5c is
cached for every s3d cell). Probe candidates: 10,908 segments (8,812 unrealised_prediction,
1,097 sibling, 840 category, 110 supplementary_false, 41 ambiguous_topic, 8 model_state).
Files: `outputs/wsbench_chat/claude-opus-5-5/{report_v2.md,probe_candidates.jsonl,audit_samples_v2.md}`.
