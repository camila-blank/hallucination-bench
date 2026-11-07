# hindex-v8: frozen results (plan 0029)

Plan: [`plans/0029-index-v8-overhaul.md`](../../plans/0029-index-v8-overhaul.md) and its
Amendments 1–4. The index was built by `hbench final --manifest data/index/v8_manifest.json`.
That manifest has 282 entries: 7 families × 3 methods × (L20…L60 + the joint unit) where the
family has one. It names the persistent-failures list `data/index/v8_persistent_failures.json`
(Amendment 4).
- Judges and versions: claim maps `claimmap-v2.9` (AM `+am3` on Sonnet 5.5) with the continuation
  verifier `cverify-v1`, judged by Opus 5.5; main on plan 0028's x3 bank; AM situation `amsit-v2`.
- Strict `final` exited **0** on 2026-11-07. The full generated table (every row, plus the
  exclusion table) is [`0029_index_v8_full.md`](0029_index_v8_full.md). The plots are in
  [`plots/final/`](../../plots/final/).
- Judge outputs (caches, cells files, safety roots) stay local / on the dev box under
  `outputs/v8-full/` (archive: `/workspace/hb-v8-full/`). The manifest pins each file by sha256.

Lower is better (the share of claims hallucinated). Methods: O-lens = `s3d-rl600` (main per layer
from `s3d-rl600` at L20/36/44/48/60 and `s3d-rl600-all11` elsewhere), NLA = `mlnla-rl600`
(`mlnla-rl600-all11`), J-lens = `jlens`. J-lens has no sibling score, so its headline is `matched`.

## Headline: 11-layer means and the joint unit

| method | macro | frozen (%) | provisional (%) | Δ (pp) |
|---|---|---|---|---:|
| O-lens (s3d-rl600) | full | 9.88 [9.41, 10.36] | 9.88 [9.42, 10.36] | +0.000 |
| O-lens (s3d-rl600) | matched | 9.08 [8.60, 9.54] | 9.08 [8.60, 9.55] | +0.000 |
| O-lens (s3d-rl600) | full_joint | 9.59 [8.94, 10.24] | 9.59 [8.94, 10.24] | -0.003 |
| NLA (mlnla-rl600) | full | 31.72 [30.48, 32.86] | 31.72 [30.48, 32.86] | +0.001 |
| NLA (mlnla-rl600) | matched | 31.62 [30.35, 32.76] | 31.62 [30.35, 32.76] | +0.001 |
| NLA (mlnla-rl600) | full_joint | 36.02 [34.62, 37.43] | 36.00 [34.59, 37.40] | +0.028 |
| J-lens (jlens) | matched | 4.71 [4.46, 4.96] | 4.71 [4.46, 4.96] | +0.001 |
| J-lens (jlens) | full_joint | 5.06 [4.60, 5.51] | 5.06 [4.60, 5.51] | +0.000 |

`full` = all seven families (Σ 20), `matched` = all but sibling (Σ 19), `full_joint` = the
six-family macro on the joint 11-layer unit. Shared item bootstrap over layers, 95% CIs.

## Per layer

| method | layer | full / matched (frozen %) | provisional % | Δ (pp) |
|---|---|---|---|---:|
| O-lens | 20 | full 11.01 [10.14, 11.88] | 11.01 | +0.000 |
| O-lens | 24 | full 10.72 [10.09, 11.36] | 10.72 | +0.000 |
| O-lens | 28 | full 11.61 [10.85, 12.37] | 11.62 | -0.010 |
| O-lens | 32 | full 10.87 [10.22, 11.53] | 10.88 | -0.003 |
| O-lens | 36 | full 9.14 [8.44, 9.84] | 9.14 | -0.001 |
| O-lens | 40 | full 9.58 [8.95, 10.22] | 9.57 | +0.014 |
| O-lens | 44 | full 9.69 [9.01, 10.38] | 9.69 | +0.001 |
| O-lens | 48 | full 10.00 [9.32, 10.68] | 10.00 | +0.002 |
| O-lens | 52 | full 8.50 [7.88, 9.13] | 8.50 | -0.001 |
| O-lens | 56 | full 8.59 [7.87, 9.32] | 8.59 | -0.002 |
| O-lens | 60 | full 8.93 [8.11, 9.76] | 8.93 | -0.001 |
| O-lens | all | full_joint 9.59 [8.94, 10.24] | 9.59 | -0.003 |
| NLA | 20 | full 32.37 [31.07, 33.66] | 32.37 | +0.003 |
| NLA | 24 | full 33.60 [32.25, 34.94] | 33.60 | +0.000 |
| NLA | 28 | full 33.91 [32.57, 35.25] | 33.91 | +0.000 |
| NLA | 32 | full 33.33 [31.80, 34.86] | 33.33 | +0.000 |
| NLA | 36 | full 30.32 [28.87, 31.77] | 30.32 | -0.000 |
| NLA | 40 | full 31.30 [29.94, 32.66] | 31.28 | +0.015 |
| NLA | 44 | full 30.34 [28.98, 31.70] | 30.34 | +0.001 |
| NLA | 48 | full 30.85 [29.53, 32.17] | 30.85 | +0.000 |
| NLA | 52 | full 29.87 [28.39, 31.35] | 29.87 | +0.000 |
| NLA | 56 | full 30.53 [29.29, 31.76] | 30.53 | -0.003 |
| NLA | 60 | full 32.51 [31.34, 33.68] | 32.51 | +0.000 |
| NLA | all | full_joint 36.02 [34.62, 37.43] | 36.00 | +0.028 |
| J-lens | 20 | matched 10.03 [9.06, 11.00] | 10.03 | +0.000 |
| J-lens | 24 | matched 7.01 [6.44, 7.58] | 7.01 | +0.000 |
| J-lens | 28 | matched 6.09 [5.56, 6.62] | 6.09 | +0.000 |
| J-lens | 32 | matched 6.23 [5.63, 6.83] | 6.23 | +0.000 |
| J-lens | 36 | matched 3.28 [2.91, 3.65] | 3.28 | +0.000 |
| J-lens | 40 | matched 3.34 [3.02, 3.66] | 3.34 | -0.001 |
| J-lens | 44 | matched 3.12 [2.74, 3.50] | 3.12 | -0.002 |
| J-lens | 48 | matched 3.43 [2.99, 3.86] | 3.43 | -0.001 |
| J-lens | 52 | matched 2.92 [2.53, 3.30] | 2.90 | +0.015 |
| J-lens | 56 | matched 2.91 [2.57, 3.25] | 2.91 | +0.000 |
| J-lens | 60 | matched 3.45 [3.09, 3.80] | 3.45 | +0.000 |
| J-lens | all | full_joint 5.06 [4.60, 5.51] | 5.06 | +0.000 |

max |Δ| per row: 0.028 pp

**Frozen vs provisional.** The provisional index (2026-11-05, `outputs/v8-full/final_provisional/`,
built by dropping every failed cell without a gate) and the frozen one agree to within
**0.03 pp** on every headline row and macro. The largest change is NLA `full_joint` +0.028 pp,
where 2026-11-07's retry answered ~300 AM joint sites. Family values move by at most 0.5 pp
(NLA overrefusal joint 41.39 → 41.94%). No conclusion changes.

## Persistent judge failures (Amendment 4)

**Retries.** Every failed row had already been re-asked in 3–8 run passes on 2026-11-05. It was
re-asked once more on 2026-11-07, as one pass per stage group: main map / cverify / joint, AM
detect, AM map / cverify / joint / situation (two passes), overrefusal per layer and joint
(two passes each); logs `logs/retry-*_20261107_0824.log`. Effect of the extra pass:
- AM NLA joint: 698 → 404 failed sites.
- Overrefusal per layer: 385 → 376 failed item-layers. Joint: 71 → 68.
- Main: a few per-layer and joint maps answered; 28 Opus-refused or invalid maps and 5 J-lens
  summaries remain.
- AM verifier: 13 → 8 cells with an unsettled verdict (10 verdicts, all O-lens).

The documented fallback (Sonnet on refusal) applies where the stage has one: verifier, AM
detector, J-lens summaries, overrefusal digests.

**What is left.** 1,413 failed cells; `scripts/v8/persistent_failures.py` explains each one.
- **1,404 are persistent** and listed. Each has ≥ 3 refused / invalid attempts at its blocking stage
  (transport-error rows not counted), and ≥ 6 for AM detections. Reasons: **1,006 refused** (both models where there is a fallback) and **398
  invalid**: 391 NLA AM joint sites where Sonnet maps readout `R0` only; 6 O-lens main maps; 1
  O-lens AM joint.
  - By blocking stage: AM detection 490, overrefusal digest 441, AM map 423 (joint 414), main map
    28, AM verifier 8, AM situation 7, main J-lens summary 5, overrefusal strict judge 2.
- **9 are not listed**, because they are transient or under-retried:
  - 6 AM joint maps that time out (`error`);
  - 1 AM joint map with 2 invalid attempts;
  - 1 J-lens AM detection and 1 overrefusal strict judge, each with 2 attempts.

  They stay under the 1% allowance in their entries and are dropped like any failed cell.

Per family × method (summed over the units that have exclusions):

| family | method | unit | excluded | of cells (units with exclusions) | % | worst unit % | reasons |
|---|---|---|---:|---:|---:|---:|---|
| am_claims | jlens | per layer (10 units) | 65 | 504520 | 0.01 | 0.03 | refused 65 |
| am_claims | mlnla-rl600 | joint (1 unit) | 398 | 35443 | 1.12 | 1.12 | invalid 391, refused 7 |
| am_claims | mlnla-rl600 | per layer (11 units) | 344 | 554972 | 0.06 | 0.13 | refused 344 |
| am_claims | s3d-rl600 | joint (1 unit) | 20 | 14760 | 0.14 | 0.14 | invalid 1, refused 19 |
| am_claims | s3d-rl600 | per layer (11 units) | 94 | 554972 | 0.02 | 0.03 | refused 94 |
| am_situation | mlnla-rl600 | per layer (4 units) | 6 | 2048 | 0.29 | 0.39 | refused 6 |
| am_situation | s3d-rl600 | joint (1 unit) | 1 | 512 | 0.20 | 0.20 | refused 1 |
| main | jlens | joint (1 unit) | 2 | 5924 | 0.03 | 0.03 | refused 2 |
| main | jlens | per layer (2 units) | 3 | 11848 | 0.03 | 0.03 | refused 3 |
| main | mlnla-rl600 | joint (1 unit) | 2 | 5924 | 0.03 | 0.03 | refused 2 |
| main | mlnla-rl600 | per layer (3 units) | 3 | 17772 | 0.02 | 0.02 | refused 3 |
| main | s3d-rl600 | joint (1 unit) | 4 | 5924 | 0.07 | 0.07 | refused 4 |
| main | s3d-rl600 | per layer (9 units) | 19 | 53307 | 0.04 | 0.08 | invalid 6, refused 13 |
| overrefusal | jlens | joint (1 unit) | 3 | 427 | 0.70 | 0.70 | refused 3 |
| overrefusal | jlens | per layer (5 units) | 12 | 2135 | 0.56 | 1.17 | refused 12 |
| overrefusal | mlnla-rl600 | joint (1 unit) | 37 | 427 | 8.67 | 8.67 | refused 37 |
| overrefusal | mlnla-rl600 | per layer (11 units) | 171 | 4697 | 3.64 | 5.85 | refused 171 |
| overrefusal | s3d-rl600 | joint (1 unit) | 28 | 427 | 6.56 | 6.56 | refused 28 |
| overrefusal | s3d-rl600 | per layer (11 units) | 192 | 4697 | 4.09 | 7.03 | refused 192 |

No exclusions in story, sibling or benign_roleplay. Four entries are above the 5% review flag, all
of them overrefusal: O-lens L32 7.03% and joint 6.56%, NLA L24 5.85% and joint 8.67%. All are
under the 15% broken-run bound.

**The bias, stated.** Exclusions are not random.
- Opus and Sonnet refuse to digest readouts that sound harmful. On overrefusal those are the O-lens
  and NLA readouts that echo the harmful-sounding prompt, which J-lens rarely produces (0.3–0.7%
  excluded vs 4–9%).
- So each method's overrefusal value is over its own item set. The 391 NLA AM joint `invalid`
  sites are a Sonnet behaviour on the longest NLA joint prompts, not a refusal.
- Overrefusal on the **common item set** (per unit, the items excluded under no method;
  `scripts/v8/or_common.py`):
  - per layer, the common-set value differs from the headline by at most 2.3 pp (NLA), with a mean
    of −0.1 pp per method;
  - joint: O-lens 31.65 → 31.50%, NLA 41.94 → 41.26%, J-lens 0.00 → 0.00%.
  - The excluded items lean slightly toward off-topic refusal claims. The effect is small next to
    the CIs and the between-method gaps.
- AM joint has no common item set: each method's joint population is its own (sites with a
  sample-0 mention). The NLA joint value is over 98.9% of its sites.
- Excluded overrefusal rows are all-zero rows: they still count in `n_items` and as bootstrap
  clusters (zero claims), as before.

## Reproduce

```bash
# from the repo root, with outputs/v8-full pointing at the judge outputs
python scripts/v8/persistent_failures.py data/index/v8_manifest.json \
  --or-joint-cache outputs/v8-full/safety/or/overrefusal/claude-opus-5-5 \
  --dest data/index/v8_persistent_failures.json   # must reproduce the committed list
hbench final --manifest data/index/v8_manifest.json --out outputs/final --plots plots/final \
  --methods data/index/methods.json --workers 4
```

`hbench index` now defaults to `--version v8`. A plain `hbench index` build has no manifest gate
and no exclusion report; the official numbers are `hbench final`'s.
