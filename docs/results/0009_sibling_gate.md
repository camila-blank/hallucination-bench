# Results 0009: the sibling gate (10-sample resample check, then probes)

Plan `plans/0009-sibling-resample-probe-gate.md` (rev 8.2). Run 2026-10-14/15.
- **Judge:** `claude-opus-5-5`: slot-fill judge `sibfill-v1`, class judge `sibclass-v1`
  (dedupe v2), probe texts `sibgen-chat-v2`.
- **Scoring:** rev 21.
- **Machine outputs:**
  - `outputs/sibling_gate/full_summary.md` holds every table below at full width, plus 10
    examples per group;
  - `outputs/sibling_gate/labels/<arm>.jsonl`;
  - the re-scored report `docs/results/0002_report_full_v2.8.md`;
  - plots in `plots/full_run_v2.8/` and `plots/sibling_gate/`.

Camila asked for three things:
- sibling substitutions count as hallucinations, "but we should only have unambiguous ones";
- resample 10 times, and "if lots of different items come up… we can tell that canada was a
  hallucination";
- where it is still unclear, build probes like the user-modelling ladder.

## The procedure, as run

1. **Units.** One unit is one `substitution_sibling` segment group, keyed by (arm, item, layer,
   position, said, actual), with repeats across the cell's 3 samples merged. There are
   **25,789 units** (34,377 segments) across the 14 arms.
2. **Rung 1, the resample check.**
   - Each cell with a unit was sampled **10 more times** from the same lens and checkpoint, with
     the same activation and the same sampling settings as the original readouts. That is 9,269
     requested cells.
   - All generators ran on H200, as the original readouts did. The sglang image has no B300
     kernels, and Camila said "do h200s".
   - The slot-fill judge read the 10 new blocks. For each block it labelled what fills the
     claim's slot: the said sibling, the right answer, another object, or nothing.
   - A unit is **`unstable` (a hallucination)** when the sibling comes back in at most 2 of
     the 10 blocks **and** either ≥ 2 different other objects appear (branch A, Camila's
     criterion) or the right answer appears in ≥ 4 blocks (branch B, a pre-registered
     extension).
   - With n_said ≥ 3 the unit is `stable`. Otherwise it is `sparse`. Both go to rung 2.
3. **Rung 2, probes** (the plan 0006/0007 recipe, adapted):
   - **Classes.** Each said object is assigned a class ("programming languages", "countries",
     "pharmaceutical drugs"…), and each class gets a concept set of at least 16 members.
   - **Probe texts.** Per concept, 24 explicit and 24 implied wsbench_chat-style texts, read at
     punctuation and markdown tokens.
   - **Probe.** Shrinkage-whitened DiffMean.
   - **Null.** A per-fold null z, keyed by the token class of the read position (comma,
     period, newline, bold, heading…).
   - **Fire.** At the unit's (item, layer, position), the sibling's z must beat all valid
     distractors. There are 8 distractors: the 4 nearest the gold and 4 random.
   - **Then:** a fire at a neighbouring layer gives `probe_nearby`; agreement from a different
     lens family gives `cross_method`; otherwise the unit is **`probe_unconfirmed` (a
     hallucination)**.
   - Units the probe cannot judge are `probe_na` and are **not** counted as hallucinations.
4. **Scoring.** The two hallucination statuses, `unstable` and `probe_unconfirmed`, become the
   scoring-only leaf `substitution_sibling_gated`. That is the fourth hallucination component,
   alongside false_claim, detail and substitution_category. Every other sibling segment keeps
   its rev-20 verdict, `probe_candidate`. Without a labels file, scoring is exactly rev 20, and
   a test checks this against a frozen copy of the rev-20 scorer.

## Headline: hallucination share, rev 20 → rev 21

Character-weighted share and readout-level rate, both with item-bootstrap 95% CIs.

| arm | char-weighted rev 20 | char-weighted rev 21 | readout-level rev 20 | readout-level rev 21 |
|---|---|---|---|---|
| s3d-rl600-all11 | 10.1 [8.7, 11.5] | **12.1** [10.4, 13.9] | 30.8 | **35.0** |
| s3d-rl600 | 10.1 [8.7, 11.5] | **11.9** [10.2, 13.6] | 30.5 | **34.5** |
| s3d-sft251 | 7.4 [6.4, 8.5] | **9.0** [7.6, 10.3] | 21.2 | **24.6** |
| mlnla-rl600-all11 | 35.7 [34.2, 37.3] | **38.1** [36.5, 39.9] | 93.4 | **94.8** |
| mlnla-rl2-800-all11 | 37.8 [36.5, 39.2] | **40.5** [39.0, 42.1] | 93.2 | **94.8** |
| nla-rl-L42 | 32.6 [30.8, 34.6] | **36.0** [33.9, 38.4] | 83.7 | **85.8** |
| nla-sft-L42 | 39.2 [37.8, 40.8] | **41.4** [40.0, 43.0] | 89.6 | **91.1** |
| patchscopes | 2.5 [2.2, 2.9] | **3.0** [2.6, 3.5] | 3.5 | **4.0** |
| token arms (jlens, logit, tuned, rlens, template-v3, sae-l42) | | unchanged (±0.1) | | |

- **Prose arms gain 1.5–3.4 points.** The arm ranking does not change.
- **Most sibling claims are unambiguous hallucinations.** 18,184 of 25,789 units (71%) are
  `unstable`, and almost all of those (17,467) come from branch A, where several different items
  come up. Typical cases:
  - metformin → ibuprofen, venlafaxine, modafinil…;
  - Infosys → Accenture, IBM, PwC…;
  - India → Hong Kong, Wales, Canada….
- **Token arms have almost no siblings** (≤ 158 units each). They have no resample, so they rely
  on rung 2 alone.

## Status mix

| status | units | share | counted as |
|---|---|---|---|
| unstable (rung 1) | 18,184 | 70.5% | hallucination |
| probe_unconfirmed | 174 | 0.7% | hallucination |
| probe_faithful | 324 | 1.3% | not (probe confirms) |
| probe_nearby | 133 | 0.5% | not |
| cross_method | 230 | 0.9% | not (another lens family agrees) |
| probe_na | 6,669 | 25.9% | not (cannot adjudicate) |
| resample_failed | 75 | 0.3% | not |

**Rung 2 is mostly n/a.** 6,669 of the 7,530 rung-2 units could not be adjudicated:

| n/a reason | units |
|---|---|
| fewer than 6 valid distractors | 2,394 |
| class too small (< 4 pairs) | 1,231 |
| invalid sibling probe | 1,180 |
| item unreadable (gold does not beat distractors) | 1,054 |
| no class | 515 |
| no scores | 220 |
| no gold | 74 |
| null check failed | 1 |

The "class too small" row includes 15 labelled `small_class`. The "no scores" units are in 5
fitted classes that have a concept with no training texts. The single `null_fail` unit shows
that the rev-8 key fix worked.

**Where it can adjudicate (861 units), the probe carries signal:**
- the sibling beats all its distractors at the claimed layer in **37.6%** of units;
- that rises to 53.1% with neighbouring layers;
- on the matched null (the same sibling and the same distractors at sites where no arm names
  it), the fire rate is **20.5%** (5,860 / 28,624);
- 20.2% of adjudicable units end `probe_unconfirmed`.

Some stable siblings are therefore in the activations. Examples:
- "William Shakespeare" for Douglas Adams: z 2.2 vs 0.2;
- "cannabis" for psychedelics: z 1.9 vs 1.1;
- "Python" for C++: z 1.3 vs 0.5.

## Calibration of the slot-fill judge

| set | units | per-block agreement | rung-1 decision agreement | target |
|---|---|---|---|---|
| dev (blind subagent, stratified by rung-1 status) | 40 | 0.912 | 0.900 | ≥ 0.90 / ≥ 0.85 |
| gold (a second blind subagent, labelling from the plan's §2b text only; uniform random, seed 0) | 80 | **0.938** | **0.925** | ≥ 0.90 / ≥ 0.85 |

- **No prompt revision was needed.**
- **The disagreements lean one way.** The judge sometimes labels a block "other" where the
  labeller saw the said sibling (23 of 800 blocks).
- **Effect on the gold `unstable` calls:** precision is 0.90 (47 of 52 judge-unstable units are
  confirmed) and recall is 0.98. The gate may therefore over-count this component by about
  one unit in ten.
- **Not stratified:** the gold sample was drawn uniformly, not stratified as plan §2b says.
  This gives the less biased agreement number.

## Sensitivity (gated units)

| arm | main | rev-1 rule | branch A only | n_said ≤ 1 | n_said ≤ 3 | cross same-pos | cross same-family | shared-prior discount |
|---|---|---|---|---|---|---|---|---|
| s3d-rl600-all11 | 3,837 | 3,771 | 3,747 | 3,464 | 4,121 | 3,858 | 3,826 | 3,862 |
| mlnla-rl600-all11 | 4,689 | 4,597 | 4,462 | 4,251 | 5,022 | 4,719 | 4,672 | 4,720 |
| nla-rl-L42 | 471 | 472 | 449 | 421 | 516 | 472 | 471 | 477 |

All rows are within −10% / +11% of the main rule. The cut-off on n_said matters most.
`full_summary.md` §5 has every arm.

## Deviations, flagged

1. **H200, not B300,** for every resample generator (Camila, 2026-10-15).
2. **The probe class cap is ≥ 4 pairs** (rev 8.2). With no cap the run would have needed about
   763k probe texts and about 7 TB of activations. Units in smaller classes are `probe_na`
   (`class_too_small`).
3. **The rung-2 pilot changed the recipe** (rev 8 / 8.1). The pilot's plain-prose probe texts
   left the merged `other` token key failing the null check in 10 of 17 classes. The fix was
   markdown-style probe texts, byte-level token classes matched to the eval sites, per-key n/a
   and ≥ 400 null positions per key.
4. **Shared pilot run directory.** The full s3d resample shared a run directory with the pilot,
   and the generator skipped the pilot's (prompt, layer) files. Those 92 files were deleted and
   regenerated, leaving 6 short cells, which are `resample_failed`.
5. **Post-result change: distractor exclusion.** After seeing the first full-run examples, I
   found that list-style claims had put compound concepts into concept sets ("students and
   departments", "JavaScript and C++"). Near-duplicates had also slipped past the exact-name
   exclusion ("GPT" for ChatGPT, "students table" for students), so a distractor could contain
   the gold itself.
   - Compound and word-overlapping concepts are now never distractors (`ladder.overlaps`).
   - Before the fix: unconfirmed 250, faithful 485, adjudicable 1,220.
   - After the fix: unconfirmed 177, faithful 324, adjudicable 861 (more units now have too few
     distractors).
   - Hallucination shares moved by at most 0.1 point.
6. **Fixes from the PR review, applied before the final labels:**
   - **Nearby follows the plan literally.** At a neighbour layer, nearby needs a valid S, at least
     6 valid distractors and a passing key. Readability is not re-tested there. This moved 3
     units from unconfirmed and 17 from cross_method to `probe_nearby`.
   - **"Other" objects that normalise to the said or actual object are dropped** from n_other.
     The judge sometimes files an alias as "other". This moved 9 units out of `unstable`.
   - **Final numbers** (this doc): unconfirmed 174, faithful 324, nearby 133, cross_method 230.
7. **Scope of the gate:** 13 report arms plus mlnla-rl600-all11. The mlnla-ck* sweep arms are
   ungated and score exactly as rev 20.

## Spend

| item | USD |
|---|---|
| slot-fill judge (26,960 calls incl. pilot + calibration) | 561 |
| class / dedupe / concept judges | 18 |
| probe texts (188k texts kept) | 1,112 |
| **API total** | **1,691** |
| Modal (estimate): probe fit about 308, capture about 80, score about 10 (incl. the re-score) | ~400 |
| Modal resample (7 pools × 16 H200 containers; not metered here) | not measured |

**Provenance:**
- generator snapshot: `gw-mlnla` @ f0d704ec (`wsbench_read_modal` unchanged; `modal_lensagree` GPU
  H100 → H200);
- capture snapshot: `gw-wrong-bridge` @ 2e9413d3.
