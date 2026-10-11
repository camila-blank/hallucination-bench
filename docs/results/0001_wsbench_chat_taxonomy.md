# Results 0001: wsbench_chat with the taxonomy-v1 judge

Plan: [`plans/0001-bootstrap-wsbench-chat-taxonomy.md`](../../plans/0001-bootstrap-wsbench-chat-taxonomy.md).
Full generated tables (every arm × layer, CIs, health): [`0001_report_full.md`](0001_report_full.md).
Run date 2026-10-11, judge `claude-opus-5-5` for every stage (summariser, v5c span, v5c verify,
taxonomy-v1 classifier). Readouts: `/workspace/wsbench-paper-data/readouts/` (11 arms, 149
items, 1,123 sites per arm-layer).

**Status:** taxonomy-v1 is superseded by the claim map (plan 0002, PR #4). This run is its
baseline. Its span and verify caches are reused by v2.

## Calibration (held-out gate, taxonomy-v1)

- Gate passed on the first held-out check: 92 gold spans, 2 repeats, $2.15.
  - Part (a): 46 blind synthetic spans.
  - Part (b): 46 real spans, hand-labelled blind.
- Results:

  | measure | value |
  |---|---|
  | type accuracy | 97.8% |
  | type accuracy, real part only | 95.7% |
  | concept recall | 100% |
  | detail recall | 96% |
  | severity, exact | 81% |
  | severity, QWK | 0.85 |
  | target accuracy | 86% |
  | `unsupported_kind` accuracy | 82% |
  | type flip rate (gold spans) | 0.0% |

- Dev (42 spans) also passed with no prompt edits.
- The log is in `data/calibration/README.md`.

## Headline, all layers pooled

Rates are the share of v5c-specific readouts. Item-bootstrap 95% CIs are in the full report.

- **v5c:** at least one contradicted span (the unchanged instrument).
- **concept / detail:** at least one contradicted span of that type.
- **strict:** at least one contradicted, disputed or unsupported span. This is Camila's
  definition: unrealised predictions and fabricated specifics count.

### Prose arms

| arm | layers | v5c | concept | detail | major (sev 3) | strict |
|---|---|---|---|---|---|---|
| patchscopes | 5 | 6.1 | 5.4 | 0.9 | 3.2 | 11.4 |
| O-lens SFT (s3d-sft251) | 5 | 35.9 | 25.4 | 13.8 | 10.9 | 90.8 |
| O-lens RL (s3d-rl600) | 5 | 48.2 | 35.9 | 18.3 | 16.7 | 95.8 |
| NLA-RL | L42 | 98.2 | 68.4 | 83.1 | 44.2 | 99.1 |
| NLA-SFT | L42 | 98.4 | 70.5 | 83.6 | 38.5 | 99.2 |

### Token arms (judged through the interp-v1 summary)

| arm | layers | v5c | concept | detail | major | strict |
|---|---|---|---|---|---|---|
| jlens | 5 | 45.7 | 37.5 | 12.9 | 14.4 | 84.0 |
| rlens | 5 | 51.8 | 35.8 | 24.2 | 11.5 | 84.0 |
| tuned | 5 | 57.3 | 49.2 | 14.6 | 22.3 | 84.8 |
| logit | 5 | 69.8 | 67.0 | 4.0 | 41.6 | 84.5 |
| template-v3 | 5 | 74.7 | 70.1 | 12.9 | 44.4 | 92.0 |
| sae-l42 | L42 | 76.8 | 57.2 | 39.2 | 23.7 | 87.8 |

## Concept vs detail share of contradicted spans (span level)

| arm | concept | detail | top targets |
|---|---|---|---|
| logit | 95% | 5% | topic 73%, structure 16% |
| template-v3 | 89% | 11% | topic 63% |
| patchscopes | 86% | 14% | entity 39%, topic 26% |
| tuned | 81% | 19% | topic 55% |
| jlens | 77% | 23% | topic 51%, structure 28% |
| O-lens RL | 70% | 30% | entity 34%, topic 22%, structure 14% |
| O-lens SFT | 67% | 33% | entity 33%, topic 20%, structure 15% |
| sae-l42 | 66% | 34% | topic 49%, structure 42% |
| rlens | 64% | 36% | structure 49%, topic 37% |
| NLA-RL | 46% | 54% | structure 61% |
| NLA-SFT | 44% | 56% | structure 69% |

## Findings

- **Each lens has a different error profile, not just a different rate.**
  - NLA's errors are mostly *details*: wrong wording in its long quoted continuations (target
    `structure`).
  - Logit lens and template lens errors are almost all *concept*: the summary misreads the
    topic.
  - O-lens sits in between. Entities (wrong named works and people) are its largest error
    target at about a third of spans, ahead of topic, structure, attribute and quantity at 10–22%
    each.
- **Patchscopes has the lowest rate, but it rarely commits to anything.** 24% of its cells are
  empty, and most of its readouts are off-topic junk (off-topic class 88%). A low hallucination
  rate is not the same as a useful readout.
- **The strict rate is near-saturated for most lenses (84–99%).** Almost every specific readout
  contains at least one unsupported specific. Readout-level "any error" metrics therefore
  separate lenses poorly. This motivates the per-segment claim map in plan 0002.
- **Layer trends are flat for O-lens.** For RL, v5c is 46–54% across L20–L60, with L20 worst.
  For SFT it is 33.5–38.1%, with L36 worst. See the full
  report for every arm × layer.

## Example spans (from `audit_samples.md`)

- *concept / entity* (O-lens RL): "*The Rats of Narnia*". A different source work replaces the
  novel the response names.
- *concept / topic* (logit): "suggesting the model is mid-way through a multilingual or garbled
  stretch of text". The response is a coherent English answer about DETR.
- *concept / topic* (jlens): "how the body/market/person reacts to…". The text is about the
  chemical reactivity of sodium hydride.
- *concept / attribute* (O-lens RL): "I can continue the movie while you…". The response says
  the AI has no physical capability.

## Spend and health

- Total spend is $1,842 for all cached calls: summariser, span, verify and classify over about
  48k cells, plus the pilot and smoke runs. Per-arm spend is in the full report.
- Failed calls after retries are ≤ 1.0% per arm per stage. The worst is NLA-SFT verify at 11 of
  1,123 cells, which are persistent refusals. The others are ≤ 0.2%. No missing cells.

## Caveats

- The judge is Opus 5.5, not WorkspaceBench's pinned Gemini 3.8 Flash, so absolute v5c rates are
  not comparable to published WorkspaceBench numbers.
- Token arms are judged through a summariser that is told to commit to a specific reading.
  Their labels describe the summary, and their unsupported and strict rates are inflated by
  design.
- Labels are single samples: the Anthropic route sends no temperature.
- The real-part gold labeller is the same model family as the judge, so agreement is
  optimistic.
- Treating `concept` for unsupported spans as invented events or stances extends Camila's
  wording "entirely contradictory claims".
