# Results 0010: the sibling gate as a pipeline step

Plan `plans/0010-gate-in-pipeline.md` (rev 5). Runs on 2026-10-15/16.

## What it is

`hbench gate` runs the plan-0009 sibling gate (10-sample resample check, then probes) over the
registered gate set in `data/arms.json`. `scripts/run_all_arms.sh` calls it between the last
claim-map stage and the final report.

Each run is a **generation**: every gated arm is labelled again. The expensive work is only
redone where its inputs changed, because it is cached or frozen:
- resample runs;
- the LLM caches;
- the class map and the class fillers;
- the probe-text bank, one cell per (concept, kind);
- the token-key merge;
- the per-class fits;
- the per-unit distractor sets.

Commands:
- `hbench gate --plan` prints the planned work and a cost estimate before anything runs.
- Above `$GATE_AUTO_USD` ($300) the gate stops and needs `--yes`.
- Every run writes `changes.md`, which lists the existing labels that moved and why.

## Validation 1: idempotence on the plan-0009 state

- **Import.** The plan-0009 run state was imported. Every cross-check passed:
  - the class map reproduces `classes.json`;
  - the 25,789 units are unchanged;
  - the fit inputs are identical for all 223 classes;
  - run0 is byte-identical to each pool's `resample.jsonl`.
- **Plan.** `hbench gate --plan` reported zero work, $0.
- **Run.** `hbench gate --offline` made **0 LLM calls and 0 Modal launches**.
- **Diff.** Against the merged plan-0009 labels, there were **0 differences** over 25,789
  units. All 14 labels files and `matched_null.json` are byte-identical.

## Validation 2: a new arm end to end (`mlnla-ck2000-all11`)

`hbench gate --add-arm mlnla-ck2000-all11 --yes` estimated $1,071, ran the steps below, and
produced generation 1 (32,410 units).

**What it ran:**
- **Precheck:** 5 cells.
- **Resample:** 2,622 cells × 10 samples on 16 H200s, 19:27 to 22:54.
- **Slot-fill judge:** 6,621 units.
- **Classes:** 1,072 new (said, item) pairs. This took one incremental dedupe call, and 41
  classes crossed the 4-pair cap.
- **Probe texts:** 1,975 new cells in 152 classes.
- **Capture:** 44,035 prompts on 30 containers, 8 minutes.
- **Fits:** 152 classes.
- **Distractors:** a first freeze for 2,327 units.
- **Rescore:** 217 classes.

**Checkpoint 2000's results:**

| | |
|---|---|
| sibling units | 6,621 |
| rung 1 | unstable 4,377 (66%), stable 1,981, sparse 250, resample_failed 13 |
| rung 2 | faithful 98, nearby 31, cross_method 76, **unconfirmed 67**, n/a 1,959 |
| hallucination share, char-weighted | 37.8% [36.6, 39.1] → **40.3%** [38.9, 41.8] |
| hallucination share, readout-level | 91.4% → **93.0%** |

These are in line with the other ML-NLA arms: rl600 goes 35.7 → 38.1, and rl2-800 goes
37.8 → 40.5.

**What moved in the 14 existing arms (`changes.md`):**
- 219 units changed final status: 213 because of a class refit, 6 because of cross_method from
  the new arm.
- Most gated shares moved by less than 0.3 points. Two exceptions, both in small arms:
  - template-v3: 4.4 → 6.3% (12 units);
  - patchscopes: 70.8 → 67.4% (4 units).
- The plan-0009 numbers (results 0009) are those of generation 0, the 14-arm set.

**Spend for the new arm:**
- **API: $406.** The slot-fill judge cost $152, probe texts $251 and classes $3.
- **Modal (estimate): about $500.** That is about 56 H200-hours of resample, the failed
  single-container capture attempts, the 30-way recapture, and 152 CPU fits.

The pre-run estimate was $1,071. It over-counted refits: it assumed about 244 classes, but 152
were refit.

## Bugs found by the live run, all fixed in this PR

1. **The LLM client was reused across `asyncio.run` stages.** Its semaphore stayed bound to a
   closed event loop, so the class stage crashed right after fill. The client is now rebuilt
   for each event loop.
2. **Capture ran in one container, without `--resume`.** 44k prompts took about 3.5 hours. A
   container restart then refused the existing directory, and the crash left 42,636 zero-byte
   files.
   - Capture now splits into parallel sub-shards (`--capture-shards`, automatically one per
     1,500 prompts) and always passes `--resume`.
3. **The snapshot tree hash included `__pycache__`.** That made a false drift check fatal on
   retry. Bytecode is now ignored.

## Pipeline usage

```bash
uv run hbench gate --plan                     # work + cost, no calls
uv run hbench gate [--yes]                    # run (exit 0 done / 2 needs --yes / 3 fatal / 4 partial)
uv run hbench gate --add-arm <arm> --plan     # a new arm (register its pool in data/arms.json)
scripts/run_all_arms.sh                       # GATE=1 by default; GATE=0 skips
uv run hbench report --out outputs --view v2  # reads the gate labels by default
```
