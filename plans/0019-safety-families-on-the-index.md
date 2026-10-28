# Plan 0019: the safety families on the hallucination index (hindex-v4), all 11 layers

Author: Camila Blank (planned with Claude), 2026-10-29. Revision 1.
Builds on plan 0013 (the index, `hindex-v3`), 0014 (V4 rows), 0015 (overrefusal), 0016
(strongreject), 0017 (mask). Everything not restated here is unchanged.

## Why

The three safety evals (overrefusal, strongreject, mask) are frozen and merged, but they are not on
the hallucination index, and they were read at 4 layers (28 / 36 / 44 / 52) while every index
family has all 11 (20:61:4). Camila (2026-10-29): put them on the index, at all 11 layers for all
three methods.

## Decisions (Camila, 2026-10-29)

1. **Unit: % of claims**, as every existing family. A family value is
   `hallucinated claims / disposition claims`, pooled over every judged cell of the (arm, layer)
   on the benign population (definitions below).
2. **Score: probe-validated.** A hallucinated claim is a false claim that the eval's probe does not
   back (plans 0015 rev 12 / 0016 rev 6 / 0017 rev 5 definitions). A probe-backed false claim stays
   in the denominator as a non-hallucinated claim.
3. **Family weights: 1 each** (overrefusal 1, strongreject 1, mask 1). Existing weights unchanged:
   main 9, agentic_misalignment 3, sibling 2, story 1.
4. **Positions: all prompt positions, for all three evals.** For mask this means the prompt
   positions only (system + user turns up to and including `assistant`); response positions are
   not in the index (they stay in the mask HB report).

## Families

Each family reads only its eval's **benign population** (as the HB scores do):
`overrefusal` = the 160 frozen `never_refused` prompts; `strongreject` = the 150 frozen
`unsuccessful` items; `mask` = the 155 gated `honest_resisted` items of the frozen HB set as
merged in #29 (the WorkspaceBench funnel's 154 is a different, WSB-side filter). Items are unchanged: no new items, no
regating. Sites = every prompt position of every item (overrefusal: positions 0..n−1 incl. the
template tail; strongreject: 0..n−1; mask: 0..`assistant`).

A **claim** is one verified detector mention span of a readout, overlapping spans of the same
readout merged (AM convention), restricted to the family's disposition labels. "Verified" means
after the eval's own verification stage where it exists, the detector label otherwise — exactly as
each eval's merged report counts it:

| family | disposition claims (denominator) | false claims | probe-backed (removed from the numerator) |
|---|---|---|---|
| overrefusal | `refusal_intent` (after `orverify`) + `comply_intent` | `refusal_intent` | at a site where `refusal_probe_v2` says refusal is present |
| strongreject | the compliance claims #33's `raw` counts + `refusal_intent` | the compliance claims #33's `raw` counts | at a site where `refusal_probe_v2` says refusal is ABSENT |
| mask | `lie_claim` (after the verifier the merged HB score uses) + `honest_claim` | `lie_claim` | at a site where the deception probe says a lie is represented |

Family value = `(false − probe-backed false) / disposition claims`. The raw share
(`false / disposition claims`) is reported beside it in `index.json` (not in the macro).

- A (method, layer) with **0 disposition claims** on the family's population has no value for that
  family (absent, as plan 0013 treats `n_claims == 0`), never 0. J-lens bags make few disposition
  claims; expect wide CIs or absence, and report `n_claims`.
- **Probe for mask:** the sensitivity threshold (τ = 95th percentile of the out-of-fold honest
  scores of the training archetypes, leave-one-archetype-out), not the pre-registered one, which
  was degenerate (plan 0017 rev 5). Prompt-site probe only.
- **Probe for overrefusal / strongreject:** `refusal_probe_v2` (plan 0015 rev 7 recipe: harmful vs
  harmless DiffMean, threshold = 95th percentile of the calibration half of the answered-prompt
  pool, present iff ≥ 2 of 4 folds). Content sites use plan 0015's content-site rule, as the
  merged probe reports do.
- **Failed judge cells** are excluded from both numerator and denominator (as the HB reports do);
  the count per (family, arm, layer) goes in `index.json` health.
- **CIs:** 2,000-draw item bootstrap, a fresh `np.random.default_rng(0)` per (family, arm, layer)
  row, items = the family's items (prompts / MASK items), as the claim families do.

## Extending to 11 layers

Layers 20, 24, 32, 40, 48, 56, 60 are new for all three evals × three readers (O-lens
`s3d-rl600` k = 3 prose, multilayer NLA `mlnla-rl600-all11` 3 samples, J-lens n1000 top-10). The
existing 28 / 36 / 44 / 52 cells are reused unchanged (same cache keys and fingerprints, so nothing
is re-judged).

For each eval, **with every retained invariant of its merged plan**:
1. **Capture + reads** at the 7 new layers, every prompt position (mask: prompt positions only),
   same capture conventions, same reader settings, same `convert-gen-dir` path; grid check (every
   (item, site, layer) present). Modal names prefixed `hb-<eval>-L11-`. Fan out as many GPUs as
   the work divides into (the NLA read is slow: shard wide).
2. **Detect + verify** with the **same prompt versions, models, fallback chain and quorum** as the
   eval's merged 4-layer run (overrefusal: `ordetect-v2` / `orverify-v3` on Sonnet 5.5;
   strongreject: `srdetect-v1` + echo with the Opus → Sonnet fallback and #25's failed-echo rule;
   mask: its detector + verifier with the Opus → Sonnet fallback). Judge concurrency ≤ 64 (shared
   org limit). Retry failed cells once via the fallback; leftovers are health counts.
3. **Probes at the 7 new layers** with the same recipe and threshold rule; the existing 4 layers'
   probe entries stay byte-identical (assert). Validity per new layer reported (held-out AUROC,
   holdout FPR for the refusal probe, transfer AUROC for both); a layer whose probe fails its
   validity gate keeps its family value as **raw** for that layer and is flagged in `index.json`
   (`probe_fallback: true`), never silently.
4. **Index export:** each eval writes `outputs/<eval>/<judge>/index_counts.jsonl`, one row per
   (arm, layer, item): `{arm, layer, item, n_claims, n_false, n_false_probe_backed, n_failed_cells}`
   (claims over that item's prompt sites at that layer), plus `index_meta.json` (prompt versions,
   probe artifact sha256, population, n items). Written by each eval's `report` from its caches.
5. Each eval's HB report and results doc gain the 7 layers (the HB score definitions are
   unchanged); the frozen manifests gain the new probe artifacts.

## Index changes (`index.py`, `INDEX_VERSION = "hindex-v4"`)

- `FAMILIES` adds `overrefusal`, `strongreject`, `mask`; `FAMILY_WEIGHTS` adds 1.0 each.
- `full` = weighted mean over all 7 families (only when all 7 are present).
  `matched` = all families except sibling (main, AM, story + the 3 safety families), for methods
  without a sibling score (J-lens).
- **Legacy columns, for continuity:** `full_v3` and `matched_v3` = the hindex-v3 definitions
  (4 / 3 families, v3 weights), so V4 rows (plan 0014), which have no safety families, keep a
  comparable macro. `hbench index --version v3` must reproduce `docs/results/0013` (and 0014's V4
  rows) exactly — the existing reproduction test stays and passes unchanged.
- Family readers for the three safety families read `index_counts.jsonl` via new
  `--overrefusal-root / --strongreject-root / --mask-root` options (defaults from `paths.py`
  `HBENCH_SAFETY_*`, extending #31's variables), mapped through `data/index/methods.json`
  (add per-family arm lists: s3d-rl600 → `s3d-rl600`; mlnla-rl600 → `mlnla-rl600-all11` for
  overrefusal / mask and `mlnla-rl600` for strongreject; jlens identity).
- `render_md` adds the three family columns; `index.json` adds raw shares, `n_claims`, n items,
  health and `probe_fallback` per family.

## Tests

- Index: hand-computed probe-validated share on a tiny `index_counts.jsonl`; absent family on
  `n_claims == 0`; `full` absent when any of 7 is missing; `matched` excludes sibling; weights;
  v3 reproduction unchanged (existing test) and `--version v3` output identical to before.
- Each eval: `index_counts.jsonl` rows sum to the report's pooled counts at 28/36/44/52 (the
  existing numbers); the 4 old layers' cells and probe entries unchanged (hash / count asserts);
  grid completeness at 11 layers.

## Run, deliverables

- Long jobs in named tmux sessions (`hb-<eval>-L11-*`), unbuffered, logs in `logs/<name>_<ts>.log`.
- Results doc `docs/results/0019_index_v4.md`: the v4 table (method × 11 layers: `full`,
  `matched`, all seven families with CIs), the three new families' raw vs probe-validated and
  `n_claims`, probe validity at the new layers, `probe_fallback` cells, health, spend; a figure
  (`full` / `matched` by layer per method; the three safety families by layer per method).
- README / CLAUDE.md: the index section (v4 definition, new options, weights).
- Cost estimate: ≈ 1.75× the three evals' 4-layer spend (mask dominates), plus the probe fits.

## Retained invariants (restated at the point of change)

- The v3 index definition, its RNG consumption order and its reproduction of 0013 / 0014 are kept
  bit for bit; v4 only adds families and columns.
- Each eval's frozen population, gates, detector / verifier prompts and HB score definitions are
  unchanged; the 4 existing layers' cells and probe entries are reused byte-identically.
- Every (method, layer) is its own index row; no all-layer roll-up.
