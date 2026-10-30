# 0024: the final O-lens arm (final-olens-rl600-x4) on the hallucination index

Requested 2026-10-30 (WorkspaceBench fix campaign): WorkspaceBench now reports the HB index in
place of its removed `hallucination` family (workspace-bench plan 0040), and the campaign's lens is
the final O-lens `final-olens-rl600-x4` (HF `agu18dec/final_olens`, folder `lens_rl_step600`; 4
samples per activation, kept as 4 separate samples, sampling `bullet4`; producer =
global-workspace PR #646). This plan puts that arm on every index family and on `benign_roleplay`.
No index definition, weight, prompt, judge or bank changes; the index version stays hindex-v6
(the `hbench index` CLI default) and every v3 / v4 / v5 output stays bit-identical.

## Inputs (all at the 11-layer grid 20..60:4)

Read by the campaign producer in plan-file mode (`wsbench_public_modal.py::fanout --plan-file`),
one ReadSpec file per family, rows built from HB's own frozen data:

| family | plan rows | cells |
|---|---|---|
| main (`wsbench_chat`) | `data/wsbench_chat/capture_rows.json` (`render: captured`, the bank's `sites[].pos`) | 149 items, 12,353 |
| story | `data/story/capture_rows.json` (`eval_positions`) | 20 items, 1,760 |
| sibling (UM ladder) | the frozen s3d sibling ReadSpecs (`hb-sibling-probes/outputs/sibling/s3d_specs.json`), the ladder's 3 domains, re-pointed at the grid | 86 items, 18,205 |
| overrefusal | `data/overrefusal/items.json` `prompt_render` (`render: plain`), every site | 427 items, 124,575 |
| agentic_misalignment | REUSED: the campaign's `agentic_misalignment` readouts (bank v2 at workspace-bench 831f22f holds the 32 HB items with identical `system` / `text`), filtered to HB's 50,452 sites per layer | 32 items |
| benign_roleplay | REUSED: the campaign's judged `jailbreak_recognition` configs + its `hb_only` plan-file read | 150 items |

Readouts keep the arm's 4 samples per cell. Each family applies its own unchanged sample rule
(e.g. the claim map's `HAL_K = 3` first non-empty samples; the UM ladder splits "- " bullets as
for s3d).

## Code changes (this PR)

1. **Overrefusal reader allowlist, v6 only.** `index.SAFETY_READERS` (plan 0019) is unchanged;
   new `index.SAFETY_READERS_V6_ADDED = {overrefusal: {final-olens-rl600-x4}}` and
   `index.safety_readers(family, version)`: v6 reads the allowlist plus the added readers, v4 / v5
   read exactly `SAFETY_READERS` (so the v4 / v5 goldens and `index.json` `safety.readers` stay
   bit-identical; under v5 the arm's overrefusal rows are listed in `ignored_arms`, and the arm has
   no v5 overrefusal value: the v4 / v5 unit is the probe-validated item metric, not exported for
   it). `index_export.READERS` (plan 0019) is unchanged and still drives the dev-box reconcile
   tests; new `index_export.EXPORT_READERS = READERS + (final-olens-rl600-x4,)` is the reader set
   of the plan-0020 (`sj_metrics.index_rows`) and plan-0022 (`offtopic_export.index_rows`)
   exports. The arm's records live in their own `$HBENCH_OR_OUT` root, so no earlier export
   changes.
2. **benign_roleplay campaign layout.** `wsb.CAMPAIGN_NESTED = (final-olens-rl600-x4,)`: its
   reused campaign dirs are `<campaign>/<arm>/all` and `<campaign>/<arm>/L<k>` (every config
   judged by the campaign, so HB issues only the `hb_only` call). `wsb.KNOWN_ARMS = ARMS +
   CAMPAIGN_NESTED` is what `--arms` accepts; the default `ARMS` (and so every default run and its
   tests) is unchanged.
3. **UM ladder with one extra arm** (`scripts/final_olens/um_ladder_arm.py`): the landed driver's
   own stages with `$UM_ARM` appended to the method pool, every output path pointed at
   `$UM_ROOT`, a COPY of the official um root. The frozen concept set C, probe texts, probes,
   distractors and probe scores are reused (never re-run `concepts`, `probe-texts`,
   `distractors`), so an arm commit to a concept outside C is untestable and drops out like any
   other untestable unit. Per domain: `mentions -> judge -> label`. Adding a method relabels the
   frozen arms' units in the copy (cross-method support), so only the arm's units are exported
   (`scripts/final_olens/export_um_arm.py`) into a separate sibling root; the frozen arms keep
   their official root. Check run before labelling: relabelling the copy with the frozen four only
   reproduces the official `units_labeled.jsonl` in every index-relevant field (committed,
   `probe.na_reason`, `probe.fire_L`, `cross`) in all 3 domains; capital_city and holiday are
   byte-identical, country differs only in the nearby-layer fields (the copy has no `probes_adj`
   scores), which the index does not read.
4. Tests: `tests/test_final_olens_arm.py` (versioned allowlist, v6-only safety rows, nested
   campaign dirs + summary end to end, the UM export). No pre-existing test is modified.

## Judging (no change to any instrument)

Opus 5.5 at effort low, keyless `ant` OAuth token per command. main / story: `hbench run --stages
map` (claim map + the seeded 10% span subsample); AM: `detect -> verify -> map` (+ the per-item
intent cache copied from the official root); overrefusal: `scripts/overrefusal/sj_run.py --tag
full` (digest -> orsj-v3 -> oroff-v1) -> `sj_report.py` -> `hbench overrefusal offtopic-export`;
sibling: the wrapper above; benign_roleplay: `hbench benign-roleplay run --arms
final-olens-rl600-x4` (WSB's pinned jailbreak judge at `wsb.WSB_SHA`).

## Index

`hbench index` (v6) at the merge commit of this PR with the official source roots plus the arm's
roots (`--main-root`, `--am-root`, `--story-root`, `--um-root`, `--overrefusal-root` repeated). The
arm is matched by name in every family (no `methods.json` entry). Results:
`docs/results/0024_final_olens_arm.md` (follow-up PR).
