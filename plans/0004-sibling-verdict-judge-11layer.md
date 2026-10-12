# Plan 0004: 11-layer sibling claims, probe validation, and a post-probe verdict judge

Issue #5 (continued), PR #6. Author: Camila Blank (planned with Claude), 2026-10-12. Revision 3 (after critiques 1–2 + Camila's NLA choice).
Builds on plan 0003 (rev 11): the concept inventory, the whitened-DiffMean probe recipe (V2) and
its gates, cross-fitting, thresholds, and the judge wiring. Everything here not stated otherwise
reuses plan 0003.

## Revision 3 changes (Camila 2026-10-12 + critique 2). These override revision 2 and the body.

1. **Methods (Camila: "use the l42 nla for now for the 3-overlap post hoc metric, but also try
   the ml-nla on the side"):**
   - **Primary methods:** s3d O-lens (11 layers), **NLA-RL L42** (layer 42 only; set-A readouts
     and v2.7 judge cells already exist; set B from `wsbench-paper-data/readouts/nla-rl-L42/basic_readout.jsonl`)
     and J-lens (11 layers).
   - **ML-NLA RL600** (11 layers) gets the full claim → probe → verdict treatment as a
     **side method**, with its own rows in every figure and table. It also gets an alternative
     consensus row with ML-NLA in place of NLA-RL L42.
2. **Consensus (post hoc), final definition:**
   - At one (item, position, concept), all three primary methods make a **sibling claim**
     (assertion or option; not the gold; not named in the prompt or response).
   - s3d and J-lens must claim it within ±1 grid step of each other (the grid is 4 layers:
     20, 24, …, 60). NLA-RL L42 must claim it at that position.
   - A unit gets the consensus label only if **its own layer** is within ±1 grid step of the
     other s3d / J-lens claim. For NLA-RL L42 units, either other method's claim may sit at any
     layer.
   - **The ML-NLA variant:** all three of s3d, J-lens and ML-NLA claim the concept within ±1
     grid step.
   - Report how many consensus units are early-layer.
3. **The verdict judge also labels consensus units** (and validated units) for form. That reports
   how many consensus units it would call hallucination, which checks whether post hoc consensus
   hides shared-bias hallucinations. The category precedence is unchanged.
4. **Depth labels:**
   - The judge always sees the default label: early ≤ 36, mid = 40 (and 42 for NLA-RL), late ≥ 44.
   - The 32 / 36 / 40 / 44 sweep changes only the **rule** label.
   - The Step 3 table's "any layer, any position" and the "early = 20–36" wording are
     superseded by items 2 and 4.
5. **Headline denominator = all claim units.** Probe n/a is its own category, and the
   "with probe verdict" rate is secondary.
6. **Form is judged per segment.** A unit is a hallucination if any of its segments is an
   assertion of the concept; option if all its segments are options; otherwise unclear. The
   viewer shows each segment's form.
7. **Set-B probe coverage:** before building set-B probes, count the claimed concepts that are
   some item's gold (and so can be validated). Report that count. If fewer than 30% of set-B
   claims are validatable, say so as a limitation (set B is then mainly about form / layer, and
   set A carries the probe question).

## Revision 2 changes (critique 1). These override the text below where they conflict.

1. **Set B scope.**
   - Only the `entity` subfamily (35 items) and `implicit` items with a categorical gold enter the
     sibling pipeline. The processing (arithmetic) items are shown in the viewer without probe
     verdicts.
   - s3d / J-lens set-B files are deduplicated by (id, layer, pos): 7 duplicated
     `br-proc-arith-*` rows.
   - Entity-level substitutions (ML-NLA L20 reads Ali as "Usain Bolt") are a separate category,
     **entity substitution**. They get no probe and are counted as a hallucination when asserted
     (the verdict judge labels the form).
2. **Set-B probe read site matched.**
   - Probe texts end with the eval prompts' **own templates** (e.g. "The athlete NAME plays the
     sport of", with a fictional or implied NAME and context before it). Use several templates per
     category.
   - The **transfer gate on the real set-B gold items is mandatory**. A concept's probe must
     reach transfer AUROC ≥ 0.70 on the items where it is the gold. Concepts that are never a gold
     are `probe n/a (unvalidated)`.
3. **Category-first inventory (not built from lens output).** Per set-B category, the inventory is
   the golds plus a fixed Opus-generated list of 15 members, generated before any claims are
   mapped. Claims are then mapped into it. Unmappable / out-of-inventory claims are kept as
   `probe n/a`. Report the map rate; Fable blind spot-checks 50 mappings.
4. **One extraction path for every method.**
   - The claim map runs on every arm's cells; J-lens uses the summary plus the plan-0003
     token+summary rule.
   - **Unit** = (method, item, layer, pos, concept) presence. Also reported per readout sample,
     because s3d has about 4 samples per cell, ML-NLA 1 and J-lens 1 summary.
5. **Consensus (post hoc), tightened.**
   - All three methods must make a sibling claim (an assertion or option, not a bare alias hit;
     not the gold; not named in the prompt or response) for the same concept, at the same item and
     position.
   - Layers must be within ±1 grid step.
   - Report how many consensus units the probe rejects. The "post hoc" label stays.
6. **Early / late is a sensitivity sweep, and the judge is checked layer-blind.**
   - The rule label uses cuts at 32 / 36 / 40 / 44. The default cut is 40: early < 40,
     late ≥ 44, 40 = mid.
   - The verdict judge runs **with** the layer, and a 300-unit subsample also runs **without** it.
     Judge-vs-rule agreement is reported for the layer-blind run, so the layer hint cannot feed
     back.
7. **Calibration.**
   - Per-class precision, recall and the confusion matrix on the blind gold set (100, subagent)
     **plus** a 30-unit human set for Camila, stratified, with the Ali claims as fixed anchors.
     That set is shipped as `data/sibling_probe/verdict_human_todo.jsonl` with a viewer filter.
   - Fable's 40-unit dev labels are written before seeing the judge prompt text.
   - If the gold accuracy on {option, hallucination} is < 85%: one prompt revision on dev, then
     re-run. If it still misses, the judge's categories are reported as **"judge (below target)"**
     next to the rule labels, and the figures show both.
8. **Denominators.**
   - Probe agreement (Figure 1) = detected / units with a probe verdict, with the probe-n/a share
     per method and layer printed beside it.
   - The hallucination rate is given both over units with a verdict and over all units, counting
     n/a as its own bar.
   - Figure 2 shows n on every bar and item-clustered CIs on the shares.
9. **Feasibility.**
   - The set-A sizes, counted: ML-NLA and J-lens 204 items × ~16 positions × 11 layers ≈ 36k
     cells; s3d extra 7 layers ≈ 23k cells.
   - The readout and capture jobs use the existing WorkspaceBench scripts (one GPU per shard,
     B300 first in their fallback list). Camila's "8×B300 always" rule is raised with her; the
     scripts are not edited.
   - The ML-NLA wall clock is measured from the first shards and reported.
   - If `pubplan-q36-all11` lacks basic_readout rows, capture the 35+ set-B prompts (minutes).
10. **Set-B response:** a greedy raw continuation of the plain prompt. At the final-token read
    site it is what the model goes on to say, which is the judge's ground truth for "stated after".
11. **Caches:** `sibverdict-v1`, `brcat-v1` and `brnorm-v1` each get their own cache file under
    `outputs/sibling/verdict/`, `outputs/sibling/br/`. Set-B claim-map cells use a separate arm
    namespace (`br-<arm>`), so nothing collides with the plan-0003 caches.
12. **Viewer size:** one JSON per item, lazy-loaded (the mlnla viewer's pattern). Target < 300 MB
    total, gzip on serve.

## Goal (Camila, 2026-10-12)

1. Find sibling / substitution claims at **all 11 lens layers** (20, 24, …, 60) for three methods:
   - the s3d O-lens (s3d-rl600);
   - **ML-NLA RL600** (the 11-layer NLA; it replaces NLA-RL L42);
   - J-lens.

   This covers the user-modelling items and the WorkspaceBench `basic_readout` prompts.
2. For every layer where such a claim appears, probe the claimed concept. Build probes for any
   concept that has none yet.
3. Add a **post-probe verdict judge**, starting from Camila's rule of thumb:
   - the probe matches → not a hallucination;
   - the probe does not match, late layer → usually an *option* / sample continuation;
   - the probe does not match, early layer → usually a hallucination.

   The judge checks the claim's **form**, not just its layer. In the Muhammad Ali prompt, the
   late layers give "Boxing or football" (a faithful sample option) while the early layers say
   "The sport he practices is Cricket" (a hallucination).
4. **Consensus:** a concept that all three methods name for an item counts as a
   non-hallucination even if the probe disagrees. It is marked separately as **post hoc**.
5. **Deliverables:**
   - an Agam-style readout viewer: every question, the readouts from each method, and labels
     (hallucination or not, category, what the probes say);
   - a figure of how often the probes agree, per method;
   - a figure of how many sibling / substitution claims fall into each category.

## Data sets

| set | items | read sites | readouts |
|---|---|---|---|
| **A: sibling bank** | the 204 plan-0003 items (7 categories) | `from_last_sentence_start` (request sentence + template tail) | **new** at the missing layers (below) |
| **B: basic_readout** | WorkspaceBench `basic_readout`, 100 items (entity / processing / implicit; e.g. `br-ent-ath-ali` "The athlete Muhammad Ali plays the sport of" → boxing) | the final prompt token (as in wsbench) | **existing**, all 11 layers |

- **Set A readouts:**
  - **Capture:** a new capture `acts-sibling-eval-all11` at layers 20..60 step 4, using the same
    `wsbench_public_plan.py prompts` rows and window-identity checks.
  - **J-lens:** all 11 layers, from the new capture (`wsbench_tokenlens_modal.py`).
  - **ML-NLA RL600:** all 11 layers, from the new capture (`wsbench_nla_modal.py --lens
    mlnla-rl600 --layers 20,...,60 --shards 64`). Its fill and convert steps run with all 11
    layers, not `wsb_post_read.sh`'s hard-coded 44.
  - **s3d O-lens:** `fanout_custom` at the 7 missing layers (20, 24, 32, 40, 48, 56, 60), merged
    with the existing 28 / 36 / 44 / 52. The token-identity and cell-count checks are re-run.
- **Set B readouts:**
  - s3d and J-lens come from `/workspace/wsbench-paper-data/readouts/{s3d-rl600,jlens}/basic_readout.jsonl`
    (11 layers).
  - ML-NLA comes from `/workspace/gw-wsb-combos/outputs/wsbench_combos/readouts/mlnla600-L{20..60}/basic_readout.jsonl`.
  - **Check first:** one ML-NLA L20 Ali sample talks about Usain Bolt. Verify that item ids and
    positions line up for every arm before use (decoded `token` at pos == the prompt's last
    token).
- **Set B activations:** `wsbench-acts:pubplan-q36-all11` (all 11 layers, every plan prompt).
  Verify that it contains the basic_readout rows at the same positions. Otherwise capture them
  (100 short prompts, cheap).
- **Set B responses:** the judge needs a response. Use a greedy Qwen3.6-27B continuation of the
  plain prompt (raw text, no chat template, max 48 new tokens), via `scripts/sibling_probe/qwen_gen_modal.py`
  in raw mode, stored in `data/sibling_probe/basic_readout_bank.json`.

## Step 1: find sibling / substitution claims (the standard judge)

- **Judge:** the claim map (`claimmap_v2`, latest `PROMPT_VERSION` on `camila/taxonomy-v2`;
  merge that branch in before judging, as before), single-call, `claude-opus-5-5`, effort low.
- **Set A:** the plan-0003 prefilter (an inventory mention). The v2.7 cache is reused for
  cells already judged (s3d and J-lens at 28 / 36 / 44 / 52); only new cells are called.
- **Set B:** every non-empty cell is judged (100 items × 11 layers × 3 arms ≈ 3.3k cells).
- **A sibling / substitution claim** is a judged segment that meets either condition below,
  with the claimed concept ≠ the item's gold and not named in the prompt or the response:
  - its leaf is `substitution_sibling` or `substitution_category`; or
  - for set A, it contains an inventory sibling mention, with a not-correct verdict (plan 0003
    M).

  The claimed concept is the inventory concept for set A, and the normalised `probe_target`
  for set B (step 2).
- **Unit:** (method, item, layer, pos, concept), deduplicated across segments; it keeps the
  segment texts.

## Step 2: concepts and probes for every claimed concept

- **Set A:** the plan-0003 inventory. Retrain V2 probes at **all 11 layers**; a new probe-text
  capture `acts-sibling-probe-all11` covers the 6 layers not captured yet. Same texts, folds,
  gates and thresholds. The pipeline-vs-sweep equality check is extended to the shared layers.
- **Set B (open inventory):**
  1. **Category per item:** one cached Opus call per item turns (prompt, target) into a
     category name and its gold concept (e.g. Ali → `sport`, `boxing`). Prompt version
     `brcat-v1`.
  2. **Canonical concepts:** each set-B claim's `probe_target` (and, when it is null, the claim
     segment) is mapped by one cached Opus call (`brnorm-v1`) to a canonical concept of the
     item's category, or `none`.
  3. **Inventory per category:** the golds plus every concept claimed at least once. If a
     category has fewer than 8 concepts, Opus adds common members up to 8 so that the
     one-vs-rest contrast is meaningful. The fillers are marked.
  4. **Probe texts:** for every (category, concept), 48 plain-text passages (24 explicit, 24
     implied) plus 24 neutral passages per category, generated by Opus with the plan-0003
     rules (no 8-gram overlap with eval prompts, a leak check for implied texts). Each ends with
     one of 2 fixed category stems ending mid-sentence like the eval prompts (e.g. sport: "…
     The sport in question is"). The probe reads the **final token**, mirroring set B's read
     site.
  5. **Capture and fit:** capture at all 11 layers, then fit V2 with the plan-0003 folds,
     gates and thresholds (single read position, so the per-offset centring has one offset).
  6. **Transfer gate:** run on set-B gold items wherever the concept is some item's gold.
     Concepts that are never a gold are unvalidated and form a flagged row (plan 0003 rule).
- **Probe verdict per claim unit:**
  - **detected** = the fold-averaged detection at τ_cat > 0.5;
  - also reported: z(c | g), the strict τ(c | g) detection, and whether the gold's probe fires
    at that cell (sanity);
  - claims on probes that fail the gates → `probe: n/a`.

## Step 3: the post-probe verdict judge (`sibverdict-v1`)

Each claim unit gets exactly one **category**:

| category | rule | decided by |
|---|---|---|
| **validated** | the probe detects the concept (gates passing) | code |
| **consensus (post hoc)** | not validated, but all three methods name the concept for this item (any layer, any position) | code; counted as a non-hallucination, shown separately |
| **option** | not validated or consensus; the claim offers the concept as one candidate among alternatives or as a sample continuation ("Boxing or football", "e.g. X", a list of choices, "could be X", a multiple-choice line, a question proposing X) | LLM judge |
| **hallucination** | not validated or consensus; the claim asserts the concept as a fact about this context ("The sport he practices is Cricket", "she is a swimmer") | LLM judge |
| **unclear** | not validated or consensus; the judge cannot tell | LLM judge |
| **probe n/a** | the probe fails a gate; the judge still labels the form (option / hallucination / unclear) | reported separately |

- **Judge input:**
  - the prompt (and the response);
  - the full readout with the claim segment marked;
  - the claimed concept;
  - the layer and its depth label ("early" = 20–36, "mid" = 40, "late" = 44–60);
  - the probe result (z, not detected).
- **Instructions:** classify by the claim's **form**; the layer is a hint only. Camila's rule
  of thumb is stated as a prior that the judge must override when the text contradicts it.
- **Model and output:** `claude-opus-5-5`, effort low, structured output
  `{category, form: assertion|alternatives|continuation|question|other, quote, why}`, cached.
  The quote must be verbatim from the readout (fail closed → `unclear`).
- **Rule-only label:** early → hallucination, late → option, mid → unclear. It is computed for
  every non-validated, non-consensus unit and reported as **judge vs rule agreement by layer**.
  That agreement is how the rule of thumb is validated.
- **Calibration:**
  - A **blind gold set**: a fresh subagent, never shown judge outputs, labels 100 claim units
    (stratified by method × depth × set; it includes all Ali claims) with the same category
    definitions.
  - Report judge-vs-gold accuracy with a Wilson CI. Target ≥ 85% on {option, hallucination}.
  - The judge prompt may be revised **once** on a separate 40-unit dev sample (labelled by
    Fable), never on the gold set.

## Step 4: deliverables

1. **Readout viewer (Agam style).**
   - **Base:** the dashboard of `gw-mlnla/scripts/mlnla/build_wsb_viewer.py` +
     `wsb_readout_viewer.html` (compaction `readout_viewer.html` format), copied into
     `scripts/sibling_probe/viewer/`.
   - **Layout:** every item of sets A and B; the tokenised prompt with clickable read positions
     (←/→ steps positions, [ ] steps items); for each position, each method's readouts at all
     11 layers.
   - **Claim badges:** sibling claims are highlighted with a category badge (validated /
     consensus post hoc / option / hallucination / unclear / probe n/a) and a tooltip giving
     the probe (z, detected, gold detected), the judge's form, quote and why, and the rule label.
   - **Filters:** set, category, method, layer and "items with claims only". The gold and the
     response are shown.
   - **Hosting:** built to `outputs/sibling/viewer/`, hosted as Modal app `sibling-viewer`
     (static, link-key gated like `olensmeta-viewer`; the URL and key are given to Camila).
2. **Figure 1, probe agreement per method:** the share of claim units whose probe detects the
   concept, per method (bars, item-clustered 95% CI), with a second panel by layer
   (20 … 60, one line per method). Sets A and B are shown separately.
3. **Figure 2, categories:** the claim units per category (stacked: validated, consensus post
   hoc, option, hallucination, unclear, probe n/a), per method; a second panel by layer depth
   (early / mid / late).
4. **Results doc** `docs/results/0004_sibling_verdicts.md`:
   - the figures;
   - judge-vs-gold accuracy;
   - judge-vs-rule agreement by layer (does Camila's rule hold?);
   - per-method hallucination rate = hallucination / (all claim units with a probe verdict),
     with and without consensus;
   - worked examples (Ali at every layer for each method).

All plots follow the dataviz rules (slide quality, 150 dpi, legends outside).

## Layout

```
plans/0004-sibling-verdict-judge-11layer.md
data/sibling_probe/basic_readout_bank.json        set B items + responses + category/gold
data/sibling_probe/br_concepts.json               set B open inventory (+ fillers marked)
data/sibling_probe/br_probe_texts.jsonl
data/sibling_probe/verdict_gold.jsonl             blind gold (100) + dev (40)
src/hallucination_bench/evals/sibling_probe/
  claims.py      claim units from judge cells (sets A/B), dedup, consensus sets
  br_bank.py     set B loading, category/normalisation calls, inventory
  verdict.py     sibverdict-v1 prompt/schema/parse, rule label, categories
  figures.py     figures 1-2
scripts/sibling_probe/viewer/{build_viewer.py,readout_viewer.html}
scripts/serve/sibling_viewer.py                    Modal static host
docs/results/0004_sibling_verdicts.md
tests/test_sibling_verdict.py
```

## Tests

- Claim-unit extraction and dedup (both sets).
- Consensus across methods at the item level.
- Category precedence: validated > consensus > judge.
- Rule labels by layer.
- The verdict parser fails closed on a non-verbatim quote.
- Set-B position alignment (decoded token == the prompt's last token).
- The viewer data builder emits every item and every claim badge.
- Existing tests pass unchanged.

## Cost (estimate)

- **Modal:** capture all11 (A), ML-NLA 45k cells (64 shards), s3d 28k cells (16 shards),
  J-lens, set-B capture / probe texts and probe fits: about $250–400.
- **Judge:** about 45k new prefiltered set-A cells plus 3.3k set-B cells, plus the verdict calls
  on non-validated claims: about $1.5–2.5k.
- **Generation:** about $30.

Camila has no budget cap; spend is reported.

## Out of scope

- Steering / ablation (the causal test).
- Other lenses.
- The wsbench `user_modeling` family itself (set A already covers user modelling, including its
  country and occupation items).
