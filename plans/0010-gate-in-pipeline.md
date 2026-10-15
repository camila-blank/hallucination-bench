# Plan 0010: the sibling gate as a normal pipeline step

Issue #12. Author: Camila Blank (planned with Claude), 2026-10-15. Revision 2 (after critique 1;
a full rewrite, so the body is the spec).
Branch `camila/gate-pipeline`, stacked on `camila/taxonomy-v2` (it includes #11 / plan 0009).

Camila: "can you figure out how to automatically integrate this as a step in our normal
hallucination bench pipeline?"

## 0. Goal, model and retained invariants

**Goal.** One command gates the arms after the claim map:
`hbench gate [--plan] [--yes] [--arms …]`. `scripts/run_all_arms.sh` calls it after the last
`map` stage.

**The gate is a "generation", not a patch.** The gate's outputs depend on the whole gated-arm
set, because these are computed over every gated arm:
- the cross-method index;
- the shared-prior discount;
- the matched null;
- class pair counts and the ≥ 4-pair cap;
- `item_sibling_names`.

So each run re-labels every arm in the persisted gated-arm set (§1). The expensive work stays
incremental through caches: resampling, LLM calls, capture and per-class fits. Labels of
previously gated arms can change when arms are added. This is intended, and it is reported
(§6). Each labels file carries a generation stamp.

**Retained, unchanged:**
- the plan 0009 method as merged: the rung-1 rule, the rung-2 recipe and fire rule, strict
  precedence, the ≥ 4-pair class cap, the merged distractor exclusion (compound, gold/said word
  overlap with class-common words ignored, affix match), the per-key null with ≥ 400 null
  positions, plan-literal nearby, and alias-other dropping;
- fail-safe scoring: with no labels the scores are exactly rev 20;
- the judge side, which reads raw leaves only.

## 1. Arm registry (`data/arms.json`, `src/hallucination_bench/arms.py`)

- **`gated_arms`**: the persisted default gate set. It starts with the 14 plan-0009 arms (13
  report arms plus mlnla-rl600-all11). `hbench gate --add-arm X` appends to it. The cross
  index, class counts, cap, matched null and `item_sibling_names` are always computed over this
  set, never over `--arms`. `--arms` only narrows what is printed, and resampling still covers
  only the arms that need it.
- **`arms.<name>`** records, for each arm:
  - `cells`, the claim-map cells directory;
  - `readouts`;
  - `kind`, either `sampled` or `token`;
  - `family`;
  - `layers`;
  - `pool`, for sampled arms.
- **`pools.<pool>`** records, for each pool:
  - `generator` (`read` / `nla` / `lensagree`) and its args (lens, merged dir, acts subdirs for
    the 11-layer bank and L42);
  - `sampling`, copied from the original `run_config.json`;
  - `orig_readouts`, the primary arm's `hallucination.jsonl`, used for the shape check;
  - `gpu: "H200"`;
  - `samples_per_run`: 10, or 1 for patchscopes, which needs 10 seeded runs.
- **The plan-0009 values:**
  - pool s3d-rl600 serves s3d-rl600 and s3d-rl600-all11, and its `orig_readouts` is
    s3d-rl600-all11's file;
  - patchscopes uses per-seed runs.
- **The validation arm.** `mlnla-ck2000-all11` gets pool `mlnla-ck2000`: generator `nla`, lens
  `mlnla-rl2-2000` (`ckpt_sweep.sh` `lens_of`), acts `acts-hallucination-chat`, all 11
  layers, T 1.0 / top_p 0.95 / top_k 64. Its cells are in
  `/workspace/hb-taxonomy-v2/outputs/wsbench_chat/claude-opus-5-5/mlnla-ck2000-all11`.
- **Derived views.** `sibling_gate.units.family_of`, `resample.POOLS` and `ARM_OF` become views
  of the registry. A test asserts that they equal the plan-0009 values.
- **`env`**:
  - `MODAL_RESOURCE_ENV=camila-blank`;
  - `WSBENCH_GEN_VOLUME=sibgate-gen`;
  - `WSBENCH_READ_GPU=H200`;
  - the modal command `uvx --python 3.12 modal@latest`;
  - the snapshot root.
- **`snapshots`**: gw-mlnla at f0d704ec and gw-wrong-bridge at 2e9413d3, each with its source
  clone path (`/workspace/gw-mlnla`, `/workspace/gw-wrong-bridge`) and its allowed patches. The
  only patch is modal_lensagree's GPU, H100 → H200.
  - `hbench gate snapshot` runs `git archive` from the local clone. It fails if the commit is
    missing, applies the patches, and records a tree checksum.
  - The gate refuses a snapshot that drifts from this, unless `--allow-snapshot-drift` is given
    (which is logged).
- **Unregistered arms.** A sampled arm without a pool exits with code 3. `--rung1 skip` makes
  its units `deterministic` instead, and the report flags this.

## 2. Resampling (no collisions, recoverable state)

- **The volume is the source of truth.** Collect lists the volume's `gen-sibgate-<pool>-*` runs,
  `volume get`s the new ones and reads every run's rows. A key's sample count is the sum over
  all runs, so the 10 patchscopes seed runs add up to 10.
- **Complete cells.** A cell is complete at ≥ 10 samples. Extra samples are cut to the first 10
  in run-name order.
- **Run names.** Each launch is named `gen-sibgate-<pool>-<sha8>-a<attempt>`, where sha8 is the
  hash of the sorted requested keys. A retry of the same keys therefore gets a new name. This
  removes the plan-0009 pilot bug, where a generator skipped existing partial files.
- **Retries.** Requested = keys of the gated units minus complete keys. A key still short after
  2 attempts becomes `resample_failed`.
- **Launch and wait.** The gate starts each pool's job with `modal run --detach`, all pools in
  parallel, and polls the volume:
  - it logs a heartbeat every 5 minutes;
  - the timeout is 10 h by default (`--resample-timeout`);
  - on timeout the pool is left partial, the gate exits with code 4, and a re-run resumes.
- **Fatal errors.** A Modal spend-limit, auth or quota error exits with code 3 and the message.
- **Precheck.** The 5-cell key and shape check runs once per new pool, before its first full
  launch, and is recorded in `outputs/sibling_gate/resample/<pool>/precheck.json`.

## 3. Units and the cells guard

- **Where cells come from.** Units are built from each gated arm's registered `cells`. These
  default to `<out>/wsbench_chat/<model>/<arm>/cells.jsonl`, written by the same pipeline run.
- **The unit-set guard.** Each labels file records a hash of its arm's unit set: the unit keys
  with their segment keys. If the claim map is re-run and the unit set changes, the gate notices
  the mismatch, re-gates that arm and logs it.
- **A missing join is never silent.** Scoring with a labels file whose unit hash does not match
  the cells is refused and treated as no labels (exact rev 20), with a warning.

## 4. LLM stages: incremental without re-queuing

- **Fill** is per unit and keeps its cache. Calibration is not re-run.
  `outputs/sibling_gate/calibration.json` records the dev and gold file hashes and the agreement,
  and the gate warns if the `sibfill` prompt version differs from the calibrated one.
- **Classes** are per (said, item) and keep their cache.
- **Dedupe becomes incremental.** The existing mapping (raw class name → canonical) is frozen in
  `data/sibling_gate/class_map.json`. Only new raw names go to a new prompt,
  `dedupe-incr-v1`, which maps new names onto the existing canonical list or creates new
  canonical classes. Existing assignments never change. With no new names there is no call.
- **Fillers are frozen.** A class's filler concepts are fixed at first generation and stored in
  `concepts.json`. New said or gold members are appended to the class without regenerating the
  fillers.
- **The probe-text bank.** Texts are stored per (class, concept, kind) in
  `data/sibling_gate/probe_bank/<class_slug>.jsonl.gz`. Only missing (concept, kind) cells, and
  the neutral texts of new classes, are generated. The generator's `avoid` list is the class's
  concept list at the moment the concept is first generated, and it is stored with the texts.
  Existing texts are never regenerated.
- **A version bump.** Prompts now take the stored avoid list, which changes the cache key, so
  the version becomes `sibgen-chat-v2.1`. The plan-0009 texts are imported into the bank as is,
  without re-asking. The idempotence test (§7) runs on the imported bank.

## 5. Probe bank, capture and fits

- **Capture is append-only.** New texts go into a new shard `acts-sibgate-probe-<sha8>`, and each
  class records its shard list.
- **The key merge is frozen per class** (`merge.json`, load and merge). It is set at the class's
  first fit. A key class that later crosses 400 null positions stays merged into `other`, which
  avoids recapturing old texts. A class is re-merged only with `--remerge <class>`, which
  recaptures that class's texts and is printed in `--plan`.
- **Refit only what changed.** A class whose concept or text set changed is refitted in full:
  folds are re-dealt, so old texts' folds can change, and DiffMean and the null change too.
  Unchanged classes keep their fit, verified by an input hash recorded in
  `probes/fit/<class>/inputs.sha256`.
- **Cap crossing.** The cap is re-evaluated over the gated-arm set every run. A class that
  crosses it gets concepts, texts, capture and fit. Its units get distractors for the first time
  (their **first freeze**).
- **Distractors.** Each unit's set is frozen at its first freeze, in
  `data/sibling_gate/distractors/<class_slug>.json`, with a sha per unit. Score records the
  per-unit shas it used, and label checks them unit by unit, replacing the whole-file guard.
  - **Exclusions are taken at first freeze.** The `item_sibling_names` exclusion is evaluated
    then. A sibling name added later by a new arm is not removed from an old set.
  - **Accepted order dependence.** Gating A then B can differ from gating A and B together,
    but only through distractor sets. The generation stamp records the freeze generation of
    every set.
- **Rescore** runs for touched classes: new fits, or new eval sites from new units. Label then
  re-runs over every gated arm.

## 6. Reporting changes, the stamp, and cost

- **The generation stamp.** Every labels file and `outputs/sibling_gate/generation.json` carry:
  - the gated-arm set;
  - the class-map, concepts, bank and fit hashes;
  - the per-arm unit hash;
  - the code revision;
  - the spend.
- **What moved.** Each run writes `outputs/sibling_gate/changes.md`: per previously gated arm,
  the units whose final status changed and why (cross_method from a new arm, a class refit, or
  a cap crossing), plus the change in that arm's gated share. `--plan` lists these expected flip
  sources before the run.
- **Default labels in the report.** `hbench report --view v2` reads
  `<out>/sibling_gate/labels` when it exists, and `--no-sibling-labels` turns that off. It
  prints the gate generation and a coverage table per arm: gated, ungated, rung-2 pending.
- **Cost guard.** `--plan` prints all the work and an estimate:
  - new units and cells per pool, and generated cells;
  - fill calls;
  - new class names, concepts, texts and capture prompts;
  - refits.

  The estimate uses the plan-0009 unit costs. A real run proceeds without `--yes` only if the
  estimate is ≤ `$GATE_AUTO_USD`, default $300. Otherwise it prints the plan and exits with code
  2. In `run_all_arms.sh`, `GATE=1` is the default, and an exit 2 is reported as "gate needs
  --yes", without failing the run. The actual spend is logged in `generation.json`.

## 7. Tests

1. The registry reproduces the plan-0009 families, pools, `ARM_OF` and orig readouts. An
   unregistered sampled arm exits with code 3.
2. Run names carry the attempt, and completeness sums across runs. Two regression tests:
   - the plan-0009 collision: a partial run, then a full request, ends with no missing keys;
   - patchscopes seed runs add up to 10.
3. `--plan` on unchanged inputs reports zero work and zero flip sources.
4. Text bank: adding a concept to a class requests texts for that concept only. Existing
   texts' cache keys are untouched, and the fillers are unchanged.
5. Dedupe: a new raw name triggers one incremental call. Old mappings are unchanged.
6. The merge is frozen: new texts crossing the threshold do not change `merge.json` unless
   `--remerge` is given.
7. Distractors: an existing unit's set is unchanged after a new arm adds a sibling name, and the
   per-unit sha guard holds.
8. The cells guard: a changed unit set triggers re-gating, and scoring with a mismatched labels
   file falls back to rev 20.
9. Report: labels are read by default, and `--no-sibling-labels` gives rev 20.

## 8. Validation run (tmux `hbgate`, `logs/gate_<ts>.log`)

1. **Import and idempotence.**
   - Import the plan-0009 state: the bank, class map, frozen distractors, merge, fits and runs.
   - `hbench gate --plan` must report zero work.
   - `hbench gate` must make zero API calls and launch no Modal jobs, and its key-sorted
     labels must equal the merged plan-0009 labels unit for unit (final status and rung-1
     fields).
   - Byte equality is required for the labels files written in a stable order. Report any
     difference.
2. **A new arm.**
   - `hbench gate --add-arm mlnla-ck2000-all11 --plan` prints the estimate. Then run the gate.
   - Report:
     - ck2000's rung-1 / rung-2 mix;
     - its rev-20 → rev-21 share;
     - `changes.md` for the 14 existing arms (expected: some cross_method flips);
     - spend and wall-clock.
3. **The plan-0009 doc numbers.** Results 0009 stays valid for its generation, and the
   validation doc notes which generation it is.

## 9. Files

```
plans/0010-gate-in-pipeline.md
data/arms.json
data/sibling_gate/{class_map.json, probe_bank/, distractors/}
src/hallucination_bench/arms.py
src/hallucination_bench/evals/sibling_gate/
  gate.py        driver: plan, cost guard, stage order, exit codes, generation stamp, changes.md
  resample.py    volume-listed runs, attempt-suffixed names, summed counts
  classes.py     frozen class map + dedupe-incr-v1; frozen fillers
  texts.py       per-(concept, kind) bank, stored avoid lists (sibgen-chat-v2.1)
  stages.py / ladder.py   per-unit distractor freeze + sha; registry views
scripts/sibling_gate/probes_modal.py   per-class input hash; frozen merge
src/hallucination_bench/cli.py          + `hbench gate`
src/hallucination_bench/report_v2.py    default labels; generation + coverage
scripts/run_all_arms.sh                 GATE step (exit 2 = needs --yes)
tests/test_gate_pipeline.py
docs/results/0010_gate_pipeline.md
CLAUDE.md, README.md
```
