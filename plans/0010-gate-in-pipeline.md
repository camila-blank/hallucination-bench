# Plan 0010: the sibling gate as a normal pipeline step

Issue #12. Author: Camila Blank (planned with Claude), 2026-10-15. Revision 1.
Branch `camila/gate-pipeline`, stacked on `camila/taxonomy-v2` (it includes #11 / plan 0009).

Camila: "can you figure out how to automatically integrate this as a step in our normal
hallucination bench pipeline?"

## 0. Goal and retained invariants

**Goal.** One command gates every arm after the claim map has run: `hbench gate --arms …`,
called by `scripts/run_all_arms.sh` after the `map` stage. It is idempotent and incremental:
- a re-run with nothing new makes no API calls and launches no Modal job;
- a new arm, or new cells, only pays for its own units;
- new concepts only add their own texts, captures and class refits.

**Retained, unchanged:**
- the plan 0009 method: the rung-1 rule, the rung-2 recipe and fire rule, strict precedence, the
  rev-8.2 class cap, the distractor rules as merged, and the per-key null;
- the fail-safe scoring: no labels file means exact rev-20 scores;
- the judge side, which reads raw leaves.

Plan 0010 changes orchestration only. A test asserts that gating the plan-0009 inputs again
reproduces the merged labels byte for byte, with zero new calls.

## 1. Arm registry

`src/hallucination_bench/arms.py` plus `data/arms.json`. Each arm declares:
- `readouts`: its hallucination.jsonl;
- `kind`: `sampled` or `token`;
- `family`: s3d / mlnla / nla / patchscopes / token arm;
- `layers`;
- for sampled arms, a `resample` block:
  - `pool`: the shared lens and checkpoint key;
  - `generator`: `read` / `nla` / `lensagree`;
  - generator args: the lens, merged dir and acts subdir;
  - the sampling config, copied from the original `run_config.json`;
  - `gpu: "H200"`, the default per Camila 2026-10-15.

The registry rules:
- **Families come from here.** `sibling_gate.units.family_of` and the pools in
  `resample.POOLS` are derived from the registry. Their hard-coded copies become thin views of
  it, and a test asserts the registry reproduces plan 0009's families and pools.
- **The 15 plan-0009 arms** are registered from their current values. The mlnla-ck* sweep arms
  are also registered (pool `mlnla-ck<N>`) but stay out of the default gate set.
- **Unregistered arms fail.** `hbench gate` on a sampled arm without a `resample` block exits
  with an error, unless `--rung1 skip` is given; with it the arm's units become `deterministic`
  and are flagged in the report. Token arms never need a block.

## 2. Generator snapshot

- **Pinning.** The generator and capture scripts live outside the repo. The pinned commits move
  into `data/arms.json` (`snapshots`: gw-mlnla f0d704ec, gw-wrong-bridge 2e9413d3), together
  with the allowed snapshot patches. The only patch is modal_lensagree's GPU line, H100 to H200.
- **`hbench gate snapshot`** builds the snapshots under `$HB_SNAP_ROOT`, default
  `/workspace/sibgate-snap`, with `git archive` of the pinned commit, applies the declared
  patches and verifies a checksum.
- **The gate refuses a snapshot** whose commit or patch set differs from the registry, unless
  `--allow-snapshot-drift` is given. That flag is logged.

## 3. Resampling without the pilot collision

- **Content-addressed run names.** Plan 0009 hit a bug: the pilot shared the full run's
  generator directory, and the generator skipped its partial (prompt, layer) files. Each launch
  now writes to `gen-sibgate-<pool>-<sha8>`, where sha8 is the hash of the sorted requested
  (id, layer, pos) keys.
- **Every known run is read.** `resample-collect` reads all `gen-sibgate-<pool>-*` runs known
  to a local index, `outputs/sibling_gate/resample/<pool>/runs.json`.
- **A cell is complete** when some run has 10 samples for its key.
- **Only incomplete cells are requested.** Before launching, the gate computes the requested
  keys minus the complete ones. An empty set means no launch.
- **Shorts.** A key that is still short after 2 launch attempts becomes `resample_failed`.
- **Launch and wait from Python, not bash.** The gate launches `modal run --detach` per pool in
  parallel, polls the volume for unit files (with a timeout and a heartbeat in the log), then
  collects. Plan 0009's lesson applies: subagent- or client-bound runs die with their parent,
  so the job is detached.
- **The `prechecks` stage** (the 5-cell key and shape check) runs once per new pool, recorded in
  `runs.json`, and is skipped afterwards.

## 4. LLM stages (already cached; incremental by construction)

- **Fill, classes, dedupe, concepts and texts** all keep their caches.
  - Fill and classes are per unit and per (said, item), so they are already incremental.
  - **Dedupe.** A new class name runs one dedupe call against the existing canonical list,
    which maps new names onto existing classes or creates new ones. Existing assignments
    never change, so no relabelling cascades.
  - **Concepts and texts.** A new class gets concept generation. An existing class that gains
    said or gold names gets texts for the new concepts only. Existing texts are never
    regenerated.
- **Calibration does not re-run.** The gate records the slot-judge calibration (dev and gold
  file hashes plus agreement) in `outputs/sibling_gate/calibration.json`. It warns if the fill
  prompt version changes without a fresh calibration.

## 5. Probe bank, incremental

- **The probe bank** is `data/sibling_gate/probe_bank.json`: the classes, their concepts, their
  text ids, a capture shard per text, and the fitted-model hash per class. Capture appends: new
  texts go into a new shard (`acts-sibgate-probe-<sha8>`), and every class records its shard
  list.
- **Refit only what changed.** A class whose concept or text set changed is refitted in full,
  because DiffMean and the null depend on every concept in the class. Unchanged classes keep
  their fitted models, and the gate checks this by hash.
- **The class cap is re-evaluated each run** (≥ 4 pairs, counted over all gated arms). A class
  that crosses the cap gets concepts, texts, capture and fit.
- **Rescore only what changed.** Score runs for touched classes (new fits or new eval units),
  and label re-runs over all units, which is cheap.
- **Distractors** are frozen per unit and never changed for existing units. New units get new
  sets. The sha guard covers the whole file, as before.

## 6. Pipeline wiring

- **`scripts/run_all_arms.sh`** gets a `gate` step after the last `map` stage and before
  `report`. It runs `uv run hbench gate --arms "$ARMS" --out "$OUT"`, and `GATE=0` skips it.
- **`hbench report --view v2`** reads `<out>/sibling_gate/labels` by default when it exists;
  `--no-sibling-labels` turns that off. It prints the gate's coverage per arm: gated, ungated,
  and rung-2 pending.
- **A plan / dry-run mode.** `hbench gate --plan` prints what would run, with no calls:
  - new units and cells to resample per pool;
  - generated cells;
  - fill calls;
  - new classes and concepts, texts to generate, capture prompts, classes to refit;
  - a cost estimate from the plan-0009 per-unit costs.

  The real run prints the same block first.

## 7. Tests

- Registry: it reproduces the plan-0009 families and pools; an unregistered sampled arm errors.
- Run names are content-addressed, and completeness is computed across runs. A test reproduces
  the pilot-collision scenario: a run with partial files, then a full request, ends with no
  missing keys.
- `--plan` on unchanged inputs reports zero work.
- Incremental probe bank: adding a concept to a class marks that class (only that class) for
  refit, and text generation is requested for new concepts only.
- The report reads labels by default; `--no-sibling-labels` gives rev 20.

## 8. Validation run

1. **Idempotence.** `hbench gate --plan` on the merged plan-0009 outputs reports zero work.
   `hbench gate` then reproduces the labels byte for byte with $0 spend.
2. **A new arm, end to end:** `mlnla-ck2000-all11`, currently ungated. Gate it and report its
   rev-20 vs rev-21 share, the spend and the wall-clock. This exercises new-pool resampling,
   new units against the existing probe bank, and any new classes or concepts.
3. All of it runs in tmux `hbgate`, logged to `logs/gate_<ts>.log`.

## 9. Files

```
plans/0010-gate-in-pipeline.md
data/arms.json
src/hallucination_bench/arms.py
src/hallucination_bench/evals/sibling_gate/{gate.py (driver), units.py, resample.py, classes.py, texts.py, stages.py}
src/hallucination_bench/cli.py              + `hbench gate [--plan] [--rung1 skip] ...`
src/hallucination_bench/report_v2.py        labels by default; coverage table
scripts/run_all_arms.sh                     gate step
tests/test_gate_pipeline.py
docs/results/0010_gate_pipeline.md          validation (idempotence + ck2000 arm)
CLAUDE.md, README.md
```
