# Plan 0009: sibling substitutions become hallucinations, gated by a 10-sample resample check and probes

Issue #10. Author: Camila Blank (planned with Claude), 2026-10-14. Revision 6 (after critiques 1–5;
rev 3 was a full rewrite with no override sections, so the body is the spec; revs 4–6 edit it in place).

Branch `camila/sibling-resample`, stacked on `camila/taxonomy-v2` (PR #4, plan 0002 rev 20 at
f969f83). This plan changes how `substitution_sibling` segments are **scored**. It does not
change or re-run the claim-map judge.

Camila (2026-10-14): "for the substitution sibling category by default we should consider these
hallucinations but we should only have unambiguous ones. By default we should have a step where
for those sibling claims we sample 10 times and see if lots of different items come up (i.e.
sometimes canada pops up, sometimes russia, sometimes uk), then we can tell that canada was a
hallucination. And then for ones where we are still unsure let's actually go ahead and make the
probes and test this like we did for the user modeling eval."

## 0. What changes, what is retained

- **The gate.** Every sibling unit (§1) goes through a two-rung ladder:
  - rung 1, the resample check (§2);
  - rung 2, probes, for the units rung 1 leaves unsure (§3).

  Each unit ends with one final status. Exactly two final statuses are hallucinations:
  `unstable` and `probe_unconfirmed`. Their segments form a fourth hallucination component,
  the scoring-only leaf `substitution_sibling_gated`.
- **"Only unambiguous ones".** Every other outcome is not counted as a hallucination: the probe
  confirms the sibling, another method agrees, the probe cannot adjudicate, or a stage failed
  or has not run. Those segments keep the raw leaf `substitution_sibling`, which stays a
  `probe_candidate`.
- **Fail-safe scoring (retained invariant).** `taxonomy_v2.VERDICTS`, `VERDICT_ROWS`,
  `PROBE_CANDIDATE_LEAVES` and the leaf assert are **unchanged** from rev 20:
  - hallucination = `false_claim`, `detail`, `substitution_category`;
  - probe_candidate = `substitution_sibling`, `supplementary_false`, `ambiguous_topic`;
  - thinky = `model_state`;
  - the rest as before.

  The gated leaf is added only to the derived scoring maps (§4). So a consumer that misses the
  status join falls back to exact rev-20 behaviour, and it never silently counts an ungated
  sibling as a hallucination. With no labels file, every arm scores identically to rev 20, and
  a test checks this.
- **The judge side stays raw.** `consistency_v2` (κ, agreement, noise metrics), `calibrate_v2`
  and the `claimmap_v2` parser read raw leaves only. The gate never enters calibration or
  judge-noise agreement. The judge prompt (`claimmap-v2.7`), the `:M` cache and `cells.jsonl`
  are untouched.
- **No commitment judge.** Plans 0006/0007 start with one because lexicon mentions include
  options and comparisons. Here the claim map has already decided that the segment is a
  contradicted claim.

## 1. Units

- A **sibling unit** groups the `substitution_sibling` segments (raw leaf, `status == "ok"`)
  that share all of:
  - arm, id, layer and the cell's pos;
  - normalised `pair.said` and normalised `pair.actual` (NFKC, casefold, punctuation and
    whitespace stripped; if that leaves an empty string, the casefolded raw string is used).
- **Scope of a unit.** Repeats of a pair across the cell's 3 samples are one unit, and every
  segment inherits its unit's status. The same (arm, id, said, actual) at another layer or pos
  is a separate unit, with its own resample and status.
- **Alt leaves.** Segments whose `alt_leaf` is `substitution_sibling` but whose raw leaf is
  something else form no unit. Their alt leaf stays raw.
- **`item_sibling_names`** (`evals/sibling_gate/units.py`; it is always imported qualified as
  `sibling_gate.units`, because `hallucination_bench/units.py` already exists) maps item → the said objects of every in-scope arm's
  units, with each object's arms and layers. It is built once, before any distractor is drawn,
  and used for the distractor exclusion (§3d) and for cross-method (§3e). It is keyed by
  `said_canonical` once §3a has run, and by the normalised said before that.
- **Expected size**, from the v2.7 cells:

  | arm | segments | cells |
  |---|---|---|
  | s3d-rl600-all11 | 7,488 | 1,661 |
  | mlnla-rl600-all11 | 8,474 | 2,757 |
  | mlnla-rl2-800-all11 | 9,085 | 2,727 |
  | s3d-sft251 | 3,983 | 1,271 |
  | s3d-rl600 | 3,233 | 753 |
  | nla-rl-L42 | 1,185 | 271 |
  | nla-sft-L42 | 563 | 204 |
  | patchscopes | 109 | 81 |
  | each token arm | ≤ 34 | |

  The `units` stage prints exact unit and cell counts per pool.
- **Arms in scope:** the 14 arms of `docs/results/0002_report_full_v2.7.md`, plus
  mlnla-rl600-all11. The mlnla-ck* sweep arms are out of scope and stay ungated.

## 2. Rung 1: the resample check

### 2a. Resampling

For each cell that holds at least one unit, draw **10 fresh samples** from the same lens at the
same (id, layer, pos):
- the same activation: `acts-hallucination-chat`, or `acts-hallucination-chat-L42` for the NLA
  L42 arms;
- the same sampling parameters and prompt format as the original run, taken from its
  `run_config.json`: T = 1.0, top_p 0.95, top_k 64, max_new 256 for s3d and NLA, and its own
  config for patchscopes.

"Fresh" means the cell's 3 original samples are not counted. The cell was selected because one
of them named the sibling, so counting them would bias the estimate.

**Pools.** Arms that share a lens and checkpoint share one pool per cell. A pool's resample
layer list is the union of its units' layers.

| pool | arms | generator |
|---|---|---|
| s3d-rl600 | s3d-rl600, s3d-rl600-all11 | `wsbench_read_modal.py --k 10` |
| s3d-sft251 | s3d-sft251 | `wsbench_read_modal.py --k 10` |
| mlnla-rl600 | mlnla-rl600-all11 | `wsbench_nla_modal.py --n-samples 10` |
| mlnla-rl2-800 | mlnla-rl2-800-all11 | `wsbench_nla_modal.py --n-samples 10` |
| nla-rl | nla-rl-L42 (L42) | `wsbench_nla_modal.py --n-samples 10` |
| nla-sft | nla-sft-L42 (L42) | `wsbench_nla_modal.py --n-samples 10` |
| patchscopes | patchscopes | `modal_lensagree` (conditional: if the pre-check fails, `resample_failed`) |

- **Script snapshots.** The generators run from a snapshot copy of `/workspace/gw-mlnla` at its
  current HEAD. The commit is recorded in the results doc, and the scripts are never edited in
  place.
- **Hardware.** Every GPU job runs on B300:8, sharded for speed.
- **Subsets.** Each pool gets a new acts subdir `acts-sibgate-<pool>` on the same volume. It
  holds **copies** of the needed prompts' acts files (scripts are never edited, and Modal
  volumes do not reliably support symlinks) and a trimmed `manifest.json`:
  - only the needed prompts;
  - the original layer list and plane indices, so that `ActsManifest.load` and
    `check_bank_layers` pass;
  - each entry's own `eval_positions` set to the union of positions its units need
    (`common.effective_positions` reads the entry).
  `--eval-positions` is **not** used, because it overwrites every entry. `--layers` is the pool's
  layer union.
- **Over-generation.** Both generators generate every listed position at every listed layer per
  prompt (`units = prompts × layers`). The collect stage keeps only the requested
  (id, layer, pos) keys and discards the rest. The cost estimate counts the generated cells, not
  the requested ones.
- **Output:** `outputs/sibling_gate/resample/<pool>/resample.jsonl`, with rows
  `{id, layer, pos, samples: [10 strings], config}` in the readout contract.

**Step 0, generator pre-check, before the pilot:**
- For each generator, run a 5-cell trimmed manifest. It must load through `ActsManifest.load`
  and `check_bank_layers`. The output (id, layer, pos) key set must
  equal the request.
- For 20 cells per pool, the block shape must match the originals: the number of "- " lines per
  sample, and the length distribution.
- A pool whose generator cannot be driven this way stops. Its units become `resample_failed`,
  and the results doc says so.

**Deterministic arms.** jlens, logit, tuned, rlens, template-v3 and sae-l42 are token bags with
no sampling. Their units get rung-1 status `deterministic` and go to rung 2. The report marks
their gated shares "probes only (no resample)" and keeps them in the token-arm tables.

### 2b. Slot-fill judge (`sibfill-v1`)

- **Call.** One call per unit: `claude-opus-5-5`, effort low, no temperature. It is cached in
  `outputs/sibling_gate/fill/cache.jsonl` with key `fill:{arm}:{cellkey}:{said}:{actual}` and
  the standard fingerprint.
- **Input:**
  - the original segment text;
  - `pair.said`, `pair.actual` and `contradicted_by`;
  - a ±400-character window of the item conversation (`claimmap_v2.conversation_text`) around
    `contradicted_by`;
  - the 10 resampled blocks, numbered.
- **Output, per block:** what fills **the same slot**, meaning the role the said object plays
  in the original claim (e.g. "the company that built me):
  - `said`: the said object; aliases count ("OpenAI's ChatGPT" for ChatGPT);
  - `actual`: the right answer for the slot as the item text gives it, in any form, including a
    description ("the team that created me");
  - `other`: any different object filling the slot, whatever its class; the judge names it,
    normalised;
  - `none`: the slot is not filled. A mention of the said object elsewhere in the block, in a
    different role, is `none`.
- **Multi-line blocks.** A block is a multi-line "- " list and gets one answer. Precedence:
  `said` if any line fills the slot with it, else `other` (listing every distinct other
  object), else `actual`, else `none`.
- **Quotes.** Every non-`none` answer carries a verbatim quote from that block, checked with
  `readouts.verify_quote`. A failed quote turns the block into `none`. The count of failed
  quotes is reported.
- **Retries.** An invalid answer is re-asked up to 2 times. After that the unit becomes
  `resample_failed`, which is not a hallucination.
- **Calibration**, plan 0006's protocol:
  - 40 dev units, labelled by a fresh subagent that is blind to the judge, with one allowed
    prompt revision;
  - 80 gold units (seed 0), labelled by a different fresh subagent working only from this
    section's text, stratified over pools and the dev-judge rung-1 decision;
  - stratifying on the judge's own decisions can inflate agreement, so unstratified agreement
    on a random 20 of the 80 is also reported;
  - target: per-block fill agreement ≥ 90%, and unit-level rung-1 decision agreement ≥ 85%;
  - below target, the doc says "judge (below target)", and the run is not blocked.

### 2c. Rung-1 rule (pre-registered)

From the 10 fresh blocks:
- n_said = the number of blocks answered `said`;
- n_actual = the number of blocks answered `actual`;
- n_other = the number of distinct `other` objects over the 10 blocks.

| condition | rung-1 status | next |
|---|---|---|
| n_said ≤ 2 **and** (n_other ≥ 2 **or** n_actual ≥ 4) | `unstable` | final: **hallucination** |
| n_said ≥ 3 | `stable` | rung 2 |
| n_said ≤ 2 and the unstable condition fails | `sparse` | rung 2 |

- **Branch A** (n_other ≥ 2) is Camila's criterion as stated: several different items come up
  (Canada once, Russia once, the UK once).
- **Branch B** (n_actual ≥ 4, sibling rare) is a **deliberate, pre-registered extension**: the
  lens mostly gives the right answer, so the sibling is sampling noise. The report splits
  `unstable` by branch (A only, B only, both).
- **`sparse`** covers a rarely filled slot (including all 10 blocks `none`) and a single repeated other object. Neither can
  separate noise from a real alternative read.
- **Sensitivity rows.** Each one re-labels only; a unit moved into rung 2 uses its rung-2
  outcome.
  - (i) The rev-1 rule: unstable iff n_said ≤ 2 and n_said + n_actual + #`other` blocks ≥ 4.
  - (ii) Branch A only.
  - (iii) A cut-off of n_said ≤ 1.
  - (iv) A cut-off of n_said ≤ 3, with stable at ≥ 4. The n_other / n_actual conditions are
    unchanged.

  So that every row has a rung-2 outcome to use, rung 2 also runs on all units with
  2 ≤ n_said ≤ 3 and on the branch-B-only units. Those outcomes are used only in sensitivity
  rows. A unit that still lacks one counts as `probe_na` there.

## 3. Rung 2: probes, following the plan 0006/0007 ladder

Rung 2 runs on every `stable`, `sparse` and `deterministic` unit, plus the sensitivity-only units
of §2c.

### 3a. Classes and concepts

- **Class judge `sibclass-v1`.** Opus 5.5, cached, one call per distinct (normalised said, item)
  pair. It is given the said object, the segment and the item window, and returns:
  - `class`: a short class name ("AI assistants", "countries", "Chinese tech companies"), or
    `no_class` when the said object has no enumerable class with at least 9 well-known members;
  - `said_canonical`;
  - `gold`: the canonical same-class entity that the item text makes right for the slot (e.g.
    "Alibaba" for "the engineers who created me" when the text says Qwen is by Alibaba), or
    `no_gold`;
  - `compatible`: concepts that are true together with the gold, such as the parent company,
    an alias, or the gold's own product.
- **Class dedupe.** One call maps the full list of class names to canonical classes.
- **Concept set C per class:**
  - every `said_canonical` and `gold` in the class;
  - well-known fillers, generated in one call per class (`sibconcepts-v1`), until C has at
    least 16 concepts. A filler is never a gold or said object of the class.
- **Join key.** `said_canonical` is the key for cross-method and for the distractor exclusion.
  So "ChatGPT" and "OpenAI's ChatGPT" match, while "OpenAI" and "ChatGPT" do not.

### 3b. Probe texts and capture

- **Texts.** Per concept: 24 explicit and 24 implied texts. Per class: 24 neutral texts (not a
  probe class). This follows plan 0006 §4a. Generator `sibgen-chat-v1`, `claude-opus-5-5`.
- **Format.** Unlike plans 0006/0007, the texts are short wsbench_chat-style exchanges (system,
  user turn, assistant reply):
  - explicit: the reply names the concept and no other concept of C;
  - implied: the reply makes the concept unambiguous without naming it ("the company behind the
    Ascend chips");
  - neutral: the reply names nothing in C.
- **Code checks** (`um_ladder.text_check`, adapted to C): naming as specified, no word 8-gram
  shared with any wsbench_chat item text, and no duplicates.
- **Capture.**
  - Script: `wsbench_capture_modal.py`, from a snapshot of
    `/workspace/gw-wrong-bridge/scripts/olens_suite/workspace_bench` at its current HEAD
    (recorded, never edited).
  - Layers: 12 (20…60 step 4, plus 42).
  - Read positions: every punctuation token in the reply after the concept's first explicit or
    implied mention, up to 24 per text.
  - Acts subdir: `acts-sibgate-probe` (one bank, with the class in the text metadata).

### 3c. Probes: a new key, the same estimator

- **Estimator.** Type (a) whitened DiffMean (`probes.cross_fit_whitened`):
  - lambda by inner CV;
  - 4 folds by text id, balanced by (concept, kind), with each concept's explicit and implied
    texts spread across folds (`assign_folds`);
  - per-fold null z on other concepts' implied positions.
- **Vendored code.** `probes.py` and `um_probe.py` are copied verbatim from
  `camila/um-ladder-v2`. The only edit is the import line: the
  primary import points to `hallucination_bench.evals.sibling_gate.probes`, and the `sibprobes`
  Modal fallback stays.
- **New recipe, stated as such.** Centring and the null are keyed by the **punctuation token
  class** of the read position, not by plan 0006's (ending, offset).
  - The classes are `.` `,` `!` `?` `:` `;` newline `)` `。` `，` `！` `？` `：` and other. A
    class with fewer than 50 training positions merges into `other`.
  - The key reuses `_key` / `eval_key` with `ending = 0` and `offset = class index`. Eval sites
    get their class from their token.
- **Validity** per (concept, layer): OOF implied AUROC ≥ 0.90.
- **Null check** per class and layer, on held-out folds: the null z of non-target concepts has
  |mean| ≤ 0.1 and sd between 0.8 and 1.25 at every key.
  - A failing key merges into `other`.
  - If `other` fails too, every unit of that class at that layer is `probe_na`.
- **Transfer caveat.** Validity is measured on probe texts only. The eval sites are in the chat
  items, so the readability rule (§3d) is the transfer check, and the doc says so.
- **No type (b).** Logistic regression is not run: it was not evaluable in plan 0006 rev 8.

### 3d. Distractors, n/a, fire

**S** is the probe for the unit's `said_canonical` concept, which is in C (§3a). `no_gold`
units use the same S.

**Order of decisions (strict):** n/a → fire → nearby → cross_method (§3e) → probe_unconfirmed.
An n/a unit is final `probe_na`, and never reaches cross_method or probe_unconfirmed. A `no_gold`
unit that does not fire is `probe_na` and skips cross_method too.

- **Distractors.** Per unit, 8 distractors from C:
  - the 4 nearest the gold by cosine of the grand-mean-centred implied centroids at L44 (for
    `no_gold` units, the 4 nearest `said_canonical`), plus 4 seeded-random (seed = unit key);
  - never the gold, the said object, a compatible concept of that gold, or a concept in
    `item_sibling_names[id]`.

  Distractors are frozen before scoring. Their sha256 is recorded and checked by `label`.
- **n/a** (final `probe_na`), when any of these holds:
  - `no_class`, or the class has fewer than 9 concepts;
  - S has no valid probe at L, or fewer than 6 distractors are valid at L;
  - the class fails the null check at L;
  - the item is **unreadable** at L: over all the item's `eval_positions` at L except the unit's
    own pos, the gold beats all its distractors at fewer than 50% of sites.
- **`no_gold` units.** Readability cannot be checked. If S fires at the same layer or nearby,
  the unit takes that fire status (`probe_faithful` or `probe_nearby`). If it does not, the
  unit becomes `probe_na`, flagged `no_gold`, and never `probe_unconfirmed`.
- **Fire** (final `probe_faithful`): at (id, L, pos), z_S exceeds the z of every valid
  distractor.
- **Nearby** (final `probe_nearby`): it fires at the same (id, pos) at a neighbouring layer:
  - for all-layer arms, the neighbours at distance exactly 4, plus L42 for L40 and L44;
  - for the L42 arms, L40 and L44.
- **Eval residuals.** These come from `acts-hallucination-chat`, or `acts-hallucination-chat-L42`
  for the L42 arms. They are scored by a Modal CPU job adapted from
  `um_probes_modal.py --phase fit|score`: one container per layer, volume `sibling-probes`,
  directory `/sibgate-v1/`.
- **Matched null** (reported, not used for labels): the same S against the same distractors, at
  the item's other sites and at other items with the same gold, where no arm names S.

### 3e. Cross-method, then label

This step applies only if the probe fired at neither the same layer nor a nearby one.

- **`cross_method`** (final): an arm of a **different lens family** names the same
  `said_canonical` for the same item, at a layer within ±4, at any of the item's positions.
  This is the lenient choice, and the doc flags it. A same-pos variant is a sensitivity row.
  - The families are s3d-*, mlnla-*, nla-*, patchscopes, and each token arm on its own.
  - `cross_method` means agreement between methods, not truth, and the results doc says so.
- **`probe_unconfirmed`** (final, **hallucination**): every other unit.

Sensitivity rows:
- same-family agreement also counts;
- the shared-prior discount: no `cross_method` when the agreeing arm names S for at least 50% of
  the items whose gold is in the same class.

## 4. Labels file and scoring

- **Labels file.** `outputs/sibling_gate/labels/<arm>.jsonl` has one row per unit:
  - the unit key, and its segment keys (cell key, readout, start, end);
  - `rung1` ∈ {`unstable`, `stable`, `sparse`, `deterministic`, `resample_failed`}, with the
    rung-1 counts and the branch;
  - `final` ∈ {`unstable`, `probe_faithful`, `probe_nearby`, `cross_method`,
    `probe_unconfirmed`, `probe_na`, `resample_failed`}, or null while rung 2 is unfinished;
  - flags: `no_gold`, `no_class`, and `diverse` (n_other ≥ 2).
- **Field name.** The segment-level field is **`sibling_status`**, which holds `final`. It is
  never `status`, which segments already use for "ok".
- **`taxonomy_v2.py`** (VERDICTS and its assert unchanged):
  - `SIBLING_GATED = "substitution_sibling_gated"`;
  - `SIBLING_HALLUCINATION = frozenset({"unstable", "probe_unconfirmed"})`;
  - `VERDICT_OF[SIBLING_GATED] = V_HALLUCINATION`, added after the dict is built;
  - `GROUP_OF[SIBLING_GATED] = group_of("substitution_sibling")`, so group shares are unchanged;
  - `effective_leaf(leaf, sibling_status)` returns `SIBLING_GATED` iff the leaf is
    `substitution_sibling` and the status is in `SIBLING_HALLUCINATION`, else `leaf`. A null or
    missing status keeps the raw leaf.
- **`score_v2.py`:**
  - The slim loader joins `sibling_status` by segment key and applies `effective_leaf` to the
    tuple's leaf (`s[0]`) once, at load time. Downstream code is unchanged.
  - `s[1]` is `effective_leaf(cm.alt_leaf_view(s), status)` for non-disputed segments (where the
    alt view equals the raw leaf), so `hallucination_alt >= hallucination` holds. Disputed segments
    whose alt is a raw sibling stay raw (§1). A test checks `hallucination_alt >= hallucination`
    per readout when labels are present. The `agreement` spans that `slim()` builds also stay
    raw, so judge-noise agreement stays raw. Only the `segs` tuple gets the effective leaf.
    `slim()` is the only place `SIBLING_GATED` enters scoring.
  - **Threading.** `slim(rec)` and `AccumulatorV2.add(rec)` have no arm today.
    `AccumulatorV2` gets an optional `sibling_labels: dict[segkey, final] | None` in its
    constructor. The segment key is (`rec["key"]`, readout idx, start, end).
    The call site is the rebuild path (`run.py`, the `AccumulatorV2()` construction near
    line 769, which writes `summary_v2.json`). It and `report_v2` load
    `outputs/sibling_gate/labels/<arm>.jsonl` when present and pass it in, so None means rev-20
    behaviour.
  - Raw-sibling segments with `status == "disputed"` form no unit and stay raw. This includes
    the alt view used by the `acg_*` columns.
  - **Leaf rows.** `score_v2.LEAF_ROWS = (*tx.ALL_LEAVES, SIBLING_GATED)` (taxonomy's
    `ALL_LEAVES` and the leaf assert are untouched). The `cs_` / `sc_` columns and
    `char_share_leaf` / `seg_share_leaf` therefore get a gated row, and the raw
    `substitution_sibling` row is net of gated segments. The `report_v2` loops over
    `tx.ALL_LEAVES` (about lines 314 and 632) switch to `score_v2.LEAF_ROWS`. With no labels the
    gated row is all zero. The rev-20-identity test runs the full `summarize` and compares
    every rev-20 key.
  - **Flags.** `readout_flags` gains `substitution_sibling_gated` (like `substitution_category`),
    and the v5c comparison gains `n_cmp_v2_gated`, counted inside the comparison's
    hallucination, as the other components are. Both are added to `FLAGS` and `_columns()`, so
    `readout_vector` finds them. The summary gets the matching keys
    (`v2_substitution_sibling_gated_rate` beside the other component rates, and the comparison count).
    `report_v2` passes the same arm's labels wherever it builds an accumulator. `LEAF_ROWS` is defined before `_columns()` runs.
  - `HALLUCINATION_LEAVES` gains `SIBLING_GATED`.
  - `HALLUCINATION_COMPONENTS` becomes `(false_claim, detail, substitution_category,
    SIBLING_GATED)`, in that order.
  - `hedged_origin` keeps testing the raw judge leaf against a local rev-20 three-component
    constant, so the `n_hedged_from_*` counts are unchanged.
- **`report_v2.py`:**
  - The hallucination columns gain `↳ substitution_sibling (gated)`, and the leaf tables gain a
    `substitution_sibling_gated` row, with raw `substitution_sibling` net of it.
  - The verdicts line prints the scoring mapping.
  - `hallucination_examples` and `probe_candidates()` iterate raw `cells.jsonl`, so they get the
    status from the shared helper `sibling_gate.units.load_labels(arm) -> {segkey: final}`. They look up each
    segment key (`rec["key"]`, readout idx, start, end) and then call `effective_leaf`.
    The `slim` join uses the same helper.
  - `probe_candidates.jsonl` keeps the raw `leaf` and adds `sibling_status`. Gated segments
    leave it.
  - A new per-arm table shows units and segments by `rung1` and by `final`, the unstable branch
    split, the `no_gold` and `no_class` counts, and the sensitivity rows (§2c, §3e).
  - The judge-noise column shows "—" for the gated component, because judge noise is measured
    on raw leaves.
- **Plot scripts.** Any direct leaf test switches to `effective_leaf`. The implementer greps
  `verdict_of`, `VERDICTS`, `HALLUCINATION_` and `leaf in` across `src/` and `scripts/`, and
  lists every call site in the PR as keep-raw or effective.
- **Tests.** Existing tests are **not modified**; VERDICTS is unchanged, so the rev-13, 16, 19
  and 20 tests pass as they are. A new test file covers:
  - the rung-1 rule and its sensitivity rows;
  - `effective_leaf`;
  - the label join (alt stays raw; a null final keeps the raw leaf);
  - the token-class key;
  - `no_gold` never producing `probe_unconfirmed`;
  - the cross-method family exclusion;
  - no labels file giving a summary identical to rev 20.

## 5. Execution

0. The generator pre-check (§2a).
1. Code and tests: `uv run pytest`, `ruff check`, `ruff format --check`.
2. **Pilot:** 300 cells from s3d-rl600-all11 at L36, L44 and L52 (all in its 11-layer bank), plus nla-rl-L42. Run resample,
   the slot judge, dev calibration, then rung 2 on the pilot's rung-2 units. The pilot gate:
   - the block-shape check passes;
   - slot-judge dev agreement is at least 90% after at most one revision;
   - at least 3 classes have valid probes at L44;
   - the null check passes at L44.
3. **Full run** in tmux session `sibgate`, logging to `logs/sibgate_<timestamp>.log`. The chained
   driver is `scripts/sibling_gate/run_all.sh`, run from a `/tmp` snapshot because editing a
   running bash script breaks it. Its stages: units → resample pools → fill → classes →
   concepts → probe texts → capture → fit → score → label → report → figures.
4. Gold calibration of the slot judge, then the results doc, report v2.8 and the plots.

## 6. Outputs and docs

- **Report.** `docs/results/0002_report_full_v2.8.md` is the v2.7 report re-scored under rev 21,
  at $0 on the judge side. Its plots go in `plots/full_run_v2.8/`. The v2.7 files stay.
- **Results doc** `docs/results/0009_sibling_gate.md`:
  - the procedure as run;
  - the headline: hallucination share per arm, rev 20 vs rev 21, with item-bootstrap CIs;
  - the rung-1 and final status mix per arm, and the unstable branch split;
  - calibration;
  - the matched null;
  - the sensitivity rows;
  - 10 hand-read examples each of unstable, stable-faithful and stable-unconfirmed;
  - the generator and capture commits;
  - spend.
- `CLAUDE.md` gets a "Rev 21 (sibling gate)" gotcha and layout lines. Plan 0002 gets a one-line
  pointer to this plan.

## 7. Files

```
plans/0009-sibling-resample-probe-gate.md
src/hallucination_bench/
  taxonomy_v2.py                    + SIBLING_GATED, SIBLING_HALLUCINATION, effective_leaf (VERDICTS unchanged)
  evals/wsbench_chat/score_v2.py    status join at slim load; 4 hallucination components
  report_v2.py                      gated column, sibling-gate table, probe_candidates status
  cli.py                            + `hbench sibling-gate <stage>`
  evals/sibling_gate/
    __init__.py
    units.py        units, normalisation, item_sibling_names, label join
    resample.py     pools, trimmed manifests, job specs, block-shape check, collect
    fill_judge.py   sibfill-v1 prompt/schema/parse/quote check; rung-1 rule + sensitivity rows
    classes.py      sibclass-v1, class dedupe, sibconcepts-v1, compatible stratum
    texts.py        sibgen-chat-v1 texts + checks; punctuation token-class key
    probes.py       vendored verbatim from camila/um-ladder-v2
    um_probe.py     vendored verbatim from camila/um-ladder-v2
    ladder.py       distractors, n/a, fire / nearby, cross-method, matched null, labels
scripts/sibling_gate/
  run_all.sh  probes_modal.py  plot_sibling_gate.py
data/sibling_gate/
  fill_dev.jsonl  fill_gold.jsonl  classes.json  concepts.json  probe_texts.jsonl
tests/test_sibling_gate.py
docs/results/0009_sibling_gate.md  docs/results/0002_report_full_v2.8.md
plots/sibling_gate/  plots/full_run_v2.8/
outputs/sibling_gate/   (gitignored) resample/ fill/ classes/ probes/ labels/
logs/sibgate_*.log
```

CLI stages: `units`, `resample-manifests`, `resample-collect`, `fill`, `fill-calib`, `classes`,
`concepts`, `probe-texts`, `capture-manifest`, `fit`, `score`, `label`, `report`, `figures`.

## 8. Cost estimate (updated from the pilot)

- **Resampling:** about 9k requested cells (the `units` stage counts them exactly), with more
  generated through over-generation (§2a), which the `resample-manifests` stage counts. Each costs a
  (prompt, layer) prefill plus 10 generations per position on B300:8.
- **Slot judge:** one call per unit, about 6k input and 0.5k output tokens.
- **Class judge:** one call per (said, item).
- **Probe texts:** about 72 per concept, for a few hundred concepts.
- **Capture:** one GPU job.
- **Fit and score:** Modal CPU, one container per layer.

Camila set no budget cap. The spend is reported in the results doc.
