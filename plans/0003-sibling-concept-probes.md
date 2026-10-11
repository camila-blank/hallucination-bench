# Plan 0003: sibling-concept eval: are same-category mentions hallucinations? (DiffMean probe verification)

Issue #5. Author: Camila Blank (planned with Claude), 2026-10-11. Revision 10 (after critiques 1–8; amendment below).

This plan is stacked on plan 0002 (PR #4, branch `camila/taxonomy-v2`). It uses plan 0002's
claim-map judge (`judges/claimmap_v2.py`) **at whatever `PROMPT_VERSION` is current on that
branch when the judge stage runs**. The judge is still being revised, so before judging the
branch is rebased onto `origin/camila/taxonomy-v2`. Every output records the judge's
`PROMPT_VERSION` and git sha. If PR #4 changes the prompt before this PR merges, the judge stage
is re-run; it is cache-keyed by fingerprint, so only changed calls repeat.

## Amendment, revision 10 (2026-10-11, after the probe job, before any judge or score result)

The probe validity gate (position-level out-of-fold AUROC, own concept vs the category's other
concepts, pooled over every position of the read window) at 0.85 passed only 165 / 510
(concept, layer) probes. Medians by category: emotion 0.91, occupation 0.85, holiday 0.82,
capital_city 0.82, country 0.78 (L44), farm_animal 0.65, tree 0.68. The eval-site transfer gate
(≥ 0.70) mostly passes. Pooling over every window position (many request-sentence tokens carry
little concept information) makes 0.85 stricter than the question needs, so:

1. The validity gate becomes **≥ 0.70** (the transfer gate's bar). Both gates still apply.
2. A **robustness row** recomputes β₁, β₂ and the headline rates on probes passing the old
   0.85 gate.
3. Categories whose probes mostly fail even 0.70 (expected: farm_animal, tree) are reported as
   "probes too weak to adjudicate", not dropped silently.

This was decided from probe-side numbers only; no judge, mention or M/U statistic had been
computed.

4. **Judge moved to `claimmap-v2.5`** (plan 0002 rev 15) before judging: it adds the
   `alternative` verdict (`unrealised_option`, `unrealised_prediction`) and the
   `unsupported_assertion` leaf (verdict `unresolved`). M becomes verdict in {hallucination,
   probe_candidate, alternative, unresolved, junk}: every not-correct claim verdict, as before.

## Goal

A lens readout sometimes names several instances of one category where the context implies
only one:

- user is South Korean, readout says "South Korea, China, Taiwan";
- user is embarrassed, readout says "embarrassed, angry, sad".

The extra instances (**siblings**) look like hallucinations. Camila's hypothesis: many of them
are not hallucinations, because the model really does represent the sibling concepts at that
activation.

Test it by training one DiffMean probe per concept (France, UK, Germany, …). Then ask: when a
lens mentions sibling *c*, does *c*'s probe fire at that activation more than when the lens does
not mention *c*?

Deliverables:

1. **Verdict.** Probe score vs lens mention: the detection rate of mentioned siblings vs
   unmentioned siblings, and a within-item AUROC.
2. **Probe-verified hallucination rate** per arm (s3d-rl600 O-lens, NLA-RL L42, J-lens). Among a
   lens's sibling mentions that the judge did not mark correct, this is the share whose probe
   does **not** detect the concept (Camila: "hallucination is when the probes don't detect the
   concept").
3. Per-category pilot evidence of how common sibling mentions are, and a list of further
   settings to try.

## Settings (categories)

Each category is a user-modelling-style scenario. A cue implies one instance (the **gold**)
without naming it, and is followed by a generic request sentence.

| category | source of items | example gold → likely siblings |
|---|---|---|
| country | wsbench `user_modeling` selfdescribe Country items (16), verbatim | South Korea → China, Japan |
| occupation | wsbench `user_modeling` selfdescribe Occupation items (14), verbatim | — |
| emotion | new | embarrassed → angry, sad |
| farm_animal | new | chicken → cow, pig |
| capital_city | new | Paris → London, Rome |
| tree | new | maple → oak, birch |
| holiday | new | Halloween → Thanksgiving, Christmas |

## Phase A: pilot gate (which categories show sibling mentions)

- Country and occupation: the existing readouts in
  `/workspace/wsbench-paper-data/readouts/{s3d-rl600,jlens,nla-rl-L42}/user_modeling.jsonl`.
- The 5 new categories: 8 hand-written pilot prompts each
  (`data/sibling_probe/pilot/specs.json`). Readouts come from s3d-rl600 only, via
  `wsbench_public_modal.py::fanout_custom`, at layers 28, 36, 44, 52 with
  `from_last_sentence_start`.
- **Unit = the cell** (one `(id, layer, pos)`; for s3d, its ~4 bullets). In the wsbench country
  readouts, a single bullet rarely names two countries, but the bullets of one cell and the cells
  of one item often do (Korean user → Japan, China; Ukraine → Russia; New Zealand → Australia;
  South Africa → Nigeria).
- A **sibling mention** = the cell names a category instance that is not the gold and is not
  named anywhere in the prompt. Matching uses the concept's alias list (§Concept inventory),
  case-folded, on word boundaries, plus the CJK aliases.
- **Gate (pre-registered):** keep a category if at least 3 of its 8 pilot items (country and
  occupation: 30% of items) have at least one s3d cell with a sibling mention. Fable also reads
  20 sibling-mention cells per category by hand and notes in the results doc whether they look
  like the phenomenon (a different instance of the same category, in a list or as an
  alternative) or like noise (e.g. a stock "Canada" in a generic scenario). A category passing
  the count but failing the read is reported and still run, with that flag.
- **Pilot result (rev 2, run 2026-10-11, `data/sibling_probe/pilot/`):** all five new
  categories pass (8/8 items each; sibling cells 21–37% under the rough pilot matcher), and the
  hand read confirms the phenomenon in each: Berlin landmarks → "central London"; Tokyo →
  "Tokyo & Washington D.C."; maple-syrup tree → "likely a birch or oak"; goats on the roof →
  "cows on the roof"; Easter → Thanksgiving/Halloween; Diwali → Rosh Hashanah; jealous →
  frustration/disappointed (emotion is the fuzziest). Country and occupation pass on the wsbench
  readouts (Ukraine → Russia, New Zealand → Australia, fisherman → farmer/sailor, locksmith →
  plumber, flight attendant → pilot). **All 7 categories are kept.** Pilot matcher noise the
  inventory must avoid: one-character CJK aliases (松 inside 松鼠 "squirrel"; 牛, 鸡, 马), `holi`
  inside "holiday", generic uses ("happy to help", "holiday", "birthday").
- The pilot also feeds the concept inventory: every sibling seen at least twice in a category's
  pilot readouts is added to that category's concepts (cap 16 concepts per category).

## Concept inventory

`data/sibling_probe/concepts.json`:
`{category: [{id, label, aliases: [..en.., ..zh..], gold: bool}]}`.

- 10–16 concepts per category, covering every gold value plus the frequent pilot siblings.
- Aliases cover the English name, demonyms and adjectives ("Korean", "Seoul" is NOT an alias of
  South Korea; only the concept itself and its adjective/demonym count), and the Chinese name (the
  O-lens often writes Chinese).
- CJK aliases are at least 2 characters, and each has an explicit exclusion list (e.g. 松树 for
  pine, never 松). Each English alias has an optional exclusion-phrase list (e.g. "happy to
  help", "holiday" for holi). Generic non-instances (birthday, "the holidays", pets) are not
  concepts.
- **Synonym clusters are merged** before the inventory is fixed (emotion: embarrassed / ashamed /
  humiliated / mortified are one concept; angry / furious / frustrated one; sad / grieving /
  heartbroken one). Each cluster is one concept with all members as aliases; `concepts.json`
  records the merge.
- **Token aliases** (for J-lens): each concept lists its single-token forms (" Korea", "Korean",
  "韩国", " Paris", …), checked against the Qwen tokenizer by a test (each listed alias must be
  one token). A concept with no single-token alias is `token_eligible: false`; it is excluded
  from both sides of every J-lens test, and the J-lens positive control runs on eligible concepts
  only. No first-subtoken matching.
- Aliases of two concepts never overlap. Tests check this, and check that each alias matches its
  own label.

## Phase B: bank

`data/sibling_probe/items.json`, about 20 items per kept category:

- **Country, occupation:** the wsbench items, verbatim (messages, id = wsbench `name`), plus
  generated items under the same coverage rule as every category (below). Verbatim items are
  exempt from the "at most 3 per gold" cap (fisherman and locksmith have 5 each).
- **New categories:** the 8 pilot prompts plus generated items.
  - Generated by `claude-opus-5-5` from a fixed prompt: the cue implies the gold through
    objects, habits or events, and never names the gold or any concept of the category.
  - The request sentence is one of the category's 2 fixed request sentences: the pilot's sentence
    and a second one — emotion "Write a short journal entry about my day."; farm_animal "Write a
    short caption for a photo of my morning."; capital_city "Write a short itinerary for my
    first day."; tree "Write a short description of my garden."; holiday "Write a short note to
    my neighbors about our plans." (Country/occupation keep the single wsbench sentence.)
  - **Every inventory concept is the gold of at least one item** (target 2; at most 3), so every
    sibling has its own eval-site validation. Generated items per category = max(20, 2 ×
    concepts) − (verbatim + pilot items), so that the total reaches max(20, 2 × concepts), capped at
    32 total; generation targets the concepts with the fewest items first.
- **Leak check (code), generated items only:** no alias of any concept in the category may
  appear in the user turn. Failing items are dropped and regenerated. The verbatim wsbench items
  are exempt (e.g. "sopa paraguaya"); any concept named in their prompt is simply excluded from
  that item's siblings (and from gold-mention counts if the gold itself is named).
- **Implication check:** Qwen3.6-27B is asked the category's pre-registered question (stored in
  `concepts.json`), greedy, thinking off: country "Based on the message, which country is the
  user from? Answer in one or two words."; occupation "… what is the user's occupation? …";
  emotion "… how does the user feel? Answer in one word."; farm_animal "… which farm animal is
  the message about? …"; capital_city "… which city is the user visiting? …"; tree "… which
  kind of tree is described? …"; holiday "… which holiday is the user preparing for? …". An item is kept only if the
  answer matches the gold's aliases (the revealed-belief gate of wsbench, at 1 greedy sample
  instead of 10). The verbatim wsbench items already passed the wsbench gate and are not
  re-checked; every generated item (including the 6 new country / occupation items) is.
- **Response:** Qwen3.6-27B's greedy reply to the item (chat template, `enable_thinking=False`,
  max 256 new tokens), stored in the bank. The judge needs it.

## Phase C: readouts and activations (Modal)

All three arms read the same positions: `from_last_sentence_start` on the chat render.

| arm | layers | how |
|---|---|---|
| s3d-rl600 | 28, 36, 44, 52 | `wsbench_public_modal.py::fanout_custom` (k=1, ~4 bullets per cell) |
| jlens | 28, 36, 44, 52 | capture (`wsbench_capture_modal.py --prompts-json`) → `wsbench_tokenlens_modal.py --lens jlens` → convert |
| nla-rl-L42 | 42 | the same capture → `wsbench_nla_modal.py --lens nla-rl400` → convert |

- **Rows come from `wsbench_public_plan.py prompts`, not a hand-rolled writer.** The bank writes
  ReadSpec plan files (`sibling_<category>.jsonl` and `sibprobe_<category>.jsonl` for the probe
  texts; `render: "chat_context"`, `positions: {"kind": "from_last_sentence_start"}`, layers 28,
  36, 42, 44, 52) and runs `prompts` once. That emits capture rows with exact `input_ids` (the
  `capture_acts` passthrough) and absolute `eval_positions`, and writes the `meta/<family>.json`
  that `convert` needs. Ids are filename-safe and unique across every row of one capture
  (`collect_prompt_rows` refuses duplicates). The s3d specs for `fanout_custom` use the same
  render, so ids match by construction; `readplan.resolve` silently drops out-of-range positions,
  so the post-merge cell-count check is the guard, plus a test that the decoded `token` per pos
  in the s3d rows equals `manifest.prompts[].tokens` at the same pos.
- **Formats.** Capture writes `wsbench-acts:/<subdir>/<label>.safetensors` (`acts` [layers,
  n_pos (all positions), 5120] bf16, `input_ids`) plus `manifest.json` (`ActsManifest`: `layers`,
  `prompts[].{label, file, n_pos, tokens, eval_positions}`). Tokenlens jobs
  `{acts, out, layers: [28,36,44,52]}`, k=10, manifest `eval_positions`, output
  `gen-<out>/<label>/L0xx.jsonl` on `WSBENCH_GEN_VOLUME` (default `wsbench-gen-4`). NLA
  `--layers 42 --acts-subdirs <subdir>`. Fetch gen dirs with `modal volume get`, check
  completeness (`wsbench_combos_fill.py`-style), convert J-lens with `convert --kind tokens`, NLA
  with `--strip-nla --layers 42`. s3d needs only the shard merge.
- **Two captures, two acts subdirs:** `prompts` is run twice (separate out-dirs): the eval items
  (`acts-sibling-eval`) and the probe + neutral texts (`acts-sibling-probe`), both at layers 28,
  36, 42, 44, 52. J-lens and NLA run on the eval subdir ONLY (never on the ~5.5k probe texts);
  `probes_modal.py` reads both. The eval capture's residuals are the inputs for scoring the
  probes at every eval cell, so probe scores and J-lens/NLA readouts share one forward pass; the
  s3d capture is a separate forward pass of identical ids (checked as above).
- **Window identity check (code, before capture):** every eval row's and probe row's read-window
  token ids (from `from_last_sentence_start` to the end) must equal the canonical ids of its
  (category, ending), i.e. the ending sentence rendered after the same separator (one space after
  the cue's final punctuation) plus the template tail. Rows that fail are regenerated or
  dropped and counted.
- GPU scripts stay in the global-workspace repo (run from `/workspace/gw-wrong-bridge`).
  hallucination-bench holds the specs/rows writers, the converters into its contract, and the
  commands (in `CLAUDE.md`).
- `--max-positions` must be at least the longest row. After every merge, check that the cell
  counts are complete; a missing cell is exit 2.

## Phase D: judge (latest claim-map judge)

- **Prefilter (code, before any call):** judge only the cells whose readout mentions at least one
  concept of the item's category (gold or sibling). For J-lens the prefilter is a raw top-10
  token matching a token alias; the judged text is the plan 0002 summary (cached `summ:`).
- **J-lens mention rule:** a J-lens mention of *c* requires BOTH a raw top-10 token hit for *c*
  AND an alias hit for *c* in the summary; it takes the verdict of the summary segment that
  contains it. Token-only hits (summary silent) and summary-only hits (summariser invention) are
  counted and reported separately, never in M.
- **Judge:** plan 0002 single-call claim map (`anchor="none"`), `claude-opus-5-5`, effort `low`.
  The prompt is the item's messages, and the marked response has the marker at character 0. Every
  read position is in the prompt, so the activation precedes the whole response. The judge code
  is imported from `judges/claimmap_v2.py` (render, schema, validate, parse) and from the
  `wsbench_chat` cached-call/cache machinery. **No prompt text is copied or edited.** If the
  `wsbench_chat` pipeline cannot take a second bank without a change, refactor it behind a small
  bank interface; the existing `wsbench_chat` tests must pass unchanged.
- **Mention → segment:** each alias hit is attributed to the judge segment that contains it, and
  carries that segment's leaf and verdict (`taxonomy_v2.VERDICT_OF`).
- A **sibling hallucination candidate** is a sibling mention whose verdict is not `correct`.
  Expected leaves are `substitution_sibling`, `false_claim`, `detail` and `off_topic`. A sibling
  the response itself names (e.g. the reply mentions China) is usually judged correct, and is
  excluded and counted.
- **Claim scope split (code + judge leaf):** NLA readouts often reconstruct an imagined prompt
  (a Ukrainian-user item read as "a prompt about Kazakh cuisine"). Each M mention is tagged
  `scope = user` (the segment characterises the user / the user's situation) or
  `scope = other` (a reconstructed prompt, a hypothetical scenario, a list of examples) by one
  extra cheap structured call per M segment (`claude-opus-5-5`, effort low, cached; fixed prompt
  in `judge.py`, prompt version `sibscope-v1`). Every headline number is reported on all M and on
  the `scope = user` slice, and also split by leaf.
- **Inventory coverage:** all headline numbers cover inventory concepts only, and say so. To
  measure what is missed, a seeded 5% sample of the non-prefiltered cells per (arm, category) is
  also judged, and every probe-candidate segment (in any judged cell) whose `probe_target` is a
  same-category instance matching no alias is counted (string match against a per-category
  allow-list of instance words is not possible, so this uses one cheap structured call per such
  segment: "is X an instance of <category>? yes/no", cached, `sibcover-v1`). Report per category
  the share of same-category sibling mentions that fall outside the inventory.
- `probe_target` is recorded for every candidate. When it names a concept of the category, the
  results doc reports how often it agrees with the alias match.

## Phase E: DiffMean probes

- **Training texts** (`data/sibling_probe/probe_texts.jsonl`): for each concept, 48 user
  messages generated by `claude-opus-5-5`, never shown to the judge. 24 state the concept
  outright, 24 only imply it. Contexts are deliberately unlike the eval cues (a test checks no
  training text shares an 8-gram with any eval item's cue).
- **Endings (transfer to the eval read sites):** every probe text ends with one of the category's
  **exact eval request sentences** (wsbench country/occupation: "Write a hypothetical but
  realistic Wikipedia biography infobox for me."; new categories: their 2 fixed sentences),
  balanced. So the read window (ending sentence + template tail) is token-identical between
  probe texts and eval items and aligns exactly by **offset from the end**.
- **Messages template (all banks):** a system turn "You are a helpful assistant." followed by
  one user turn, identical for eval items (wsbench items already have it), generated items,
  probe texts and neutral texts.
- **Neutral texts:** 24 messages per category with no category content, ending in the same
  sentences.
- **Capture:** `acts-sibling-probe` (see Phase C), `from_last_sentence_start` resolved by
  `prompts`, layers 28, 36, 42, 44, 52.
- **Where it runs:** acts stay on the `wsbench-acts` volume (~20 GB; the box has a 32 GB RAM cap).
  A Modal function (`scripts/sibling_probe/probes_modal.py`, CPU or one GPU) streams the acts one
  layer at a time, keeps only the eval positions, fits the probes, scores the held-out texts and
  every eval cell, and writes small outputs: directions (`.safetensors`), held-out score tables
  and the eval score table `{arm-independent: id, layer, pos, concept, s}` (parquet/jsonl).
- **Direction:** per (concept *c*, layer *L*): average each text's eval-position activations
  into one vector per text; d = mean over *c*'s training texts − the equal-weighted mean of the
  other category concepts' per-concept means. True DiffMean (Camila's request), no whitening.
- **Position-aligned scoring:** s = ((h − μ_cat[L, ending, offset]) · d/‖d‖) / σ_cat[L, ending,
  offset, c], where h is one position's residual, μ_cat is the equal-weighted mean over the
  category's concepts of the training texts' residuals at that ending and offset from the end,
  and σ_cat is the sd of the projected training residuals at that (ending, offset) (pooled over
  the category's concepts). This removes the per-position mean AND scale (template tokens have
  very different norms), so pooling thresholds over offsets is valid.
- **Cross-fitting, category-level folds:** every probe text of a category (all concepts and the
  neutral texts) gets one seeded fold id in {0..3}, balanced by concept and ending. Fold model k
  fits, for every concept of the category, d, μ_cat and σ_cat on the non-k texts only. Every
  out-of-fold score (validity gate, τ_cat, τ(c | g), τ_neutral, z, the ceiling's τ(g | g′)) is a
  fold-k text scored by model k, so no threshold text ever contributed to the direction, the
  centring or the scale that scores it. Eval cells are scored by **each fold model k separately** and thresholded against
  fold k's own out-of-fold distribution (fold k's held-out texts); the 4 detection indicators and
  the 4 z values are then averaged per (cell, c). No all-texts refit, no averaged direction. Report threshold CIs by
  text-level bootstrap.
- **Unembedding check:** report the cosine between each direction and its concept's
  token-alias unembedding rows (how much the probe is just the output token).
- **Probe validity gate (pre-registered):** position-level AUROC on out-of-fold scores, *c*'s
  positions vs the other concepts' positions in the category, ≥ 0.85 at layer L. Failing
  (concept, layer) probes are excluded from scoring and listed.
- **Thresholds (out-of-fold scores; scores are already position-centred by μ_cat, so every
  threshold and z pools over endings and offsets):**
  - **τ_cat(c, L)**, pooled = 95th percentile of s_c over the other concepts' out-of-fold positions
    (pooled). "Fires above its general baseline." Used for the headline probe-verified
    hallucination rate (Camila's definition: the probe does not detect the concept).
  - **τ(c | g, L)** = 95th percentile of s_c over the out-of-fold positions of the item's **gold**
    concept *g* (its implied-only texts, which match the eval cues; all of g's texts as a
    robustness row). "Fires above its level in an ordinary g context." This removes the near-concept
    confound (the China probe may fire on most Korea texts); used for the strict rate and for
    the verdict.
  - Robustness row: τ_neutral(c, L) = 95th percentile over the category's neutral texts.
  - z(c | g) = (s_c − mean_g(s_c)) / sd_g(s_c), over g's out-of-fold implied-text positions at L
    (pooled over endings and offsets; ~24 texts × ~15 positions per (c, g, L)).

## Phase F: scoring and the verdict

Unit = (cell, concept): arm × item × layer × pos × concept of the item's category; the probe
score is the cell's own layer (NLA L42). A **sibling** is a concept that is not the gold and not
named in the prompt.

- **Gold-equivalent reference (the ceiling).** τ(c | g) is undefined for c = g (it would be ~5%
  by construction), so the ceiling is computed as: the gold probe g at its own items' cells
  against τ(g | g′), where g′ ranges over the category's other golds (pooled), i.e. "does the
  gold's probe fire above its level in another gold's context". The ceiling is always computed on
  **gold-mention cells** (under τ_cat: plain gold-mention detection); the all-cells version is a
  secondary row.
- **Transfer gate per (concept, layer), pre-registered:** the eval-site AUROC of concept c's
  probe, c-gold items' cells vs the other-gold items' cells of the category (same layer), must be
  ≥ 0.70. A (category, layer) is transfer-passing if at least half its concepts pass. The
  headline hallucination rates and the verdict cover only (cell, c) pairs whose c passes at that
  layer; pairs with a failing or unvalidated c (e.g. a concept whose implication check dropped
  all its items) form a separate flagged row. Also report gold detection rate (τ_cat) over all cells and over gold-mention
  cells.
- **M** = sibling hallucination candidates: mentioned, with verdict in {hallucination,
  probe_candidate, unresolved, junk} (J-lens per the mention rule). Mentions in `non_claim` or
  `failed` segments are excluded from M and reported as their own rows.
- **Unmentioned c at a cell** = no alias hit for c anywhere in the cell's readout text, and for
  J-lens also no raw top-10 token-alias hit and no summary alias hit. Every other mention type
  (correct / non_claim / failed segments, J-lens token-only or summary-only hits) is excluded
  from BOTH sides of Tests 1–3 and the within-item AUROC; those pairs appear only in their own
  rows. A test covers this.
- **Test 1, matched concept (primary):** for each M pair (item, layer, c), the comparison set
  U_matched = the same (item, layer, c) at the cells of that arm that do not mention c.
  **Pre-registered estimand β₁:** the `mentioned` coefficient of the OLS z(c | g) ~ mentioned +
  offset-from-end dummies + (item, layer, c) fixed effects, over M ∪ U_matched, pooled over the
  transfer-passing (category, L), item-clustered bootstrap (2,000). Descriptive companions
  (no verdict weight): detection rate(M) vs detection rate(U_matched) under τ(c | g) and τ_cat,
  per layer and per category.
- **Test 2, within cell (primary):** at each M cell, compare z(c | g) of the mentioned sibling(s)
  with z(c' | g) of the unmentioned siblings at the same cell. **Pre-registered estimand β₂:**
  the mean over M cells of [mean z of mentioned siblings − mean z of unmentioned siblings],
  pooled over transfer-passing (category, L), item-clustered bootstrap. Companion: the share of
  M cells where the mentioned sibling outranks the median unmentioned one. Same activation, same position, gold baseline removed per pair:
  this controls for position and for near/far geometry.
- **Within-item AUROC (descriptive, deliverable 1):** per (arm, item, layer, c) with at least one
  mention cell and one non-mention cell, the AUROC of z(c | g) at mention cells vs non-mention
  cells; mean over groups, item-clustered bootstrap CI.
- **Test 3, residual (robustness):** a linear-probability regression detected (τ(c | g); τ_cat as
  a second row) ~ mentioned, with
  cell fixed effects and (item, c) fixed effects; report the `mentioned` coefficient with a
  cluster bootstrap CI.
- **Unit of analysis:** every test is computed **per arm**. Within an arm a (cell, c) pair is
  counted once even if several segments mention c; a cell mentioning several siblings contributes
  each sibling once.
- **Probe-verified hallucination rate** per arm = share of M whose probe does not detect c (and,
  matching Camila's purely probe-based wording, the same rate over **all** sibling mentions,
  judge verdict ignored; the response-named siblings excluded by the judge are counted). Coverage:
  the share of mentions dropped because their (c, L) probe failed the validity gate or their
  (category, L) failed transfer.
  - **Headline row:** τ_cat, all M. **Strict row:** τ(c | g). **Robustness row:** τ_neutral.
    Plus the all-sibling-mentions row (τ_cat).
  - Each row per category and macro over categories, on all M and on the `scope = user` slice,
    and per leaf.
  - Next to each row: the ceiling (gold-equivalent reference under the same threshold) and the
    floor (detection rate(U_matched) under the same threshold).
- **J-lens is a positive control, not evidence.** A J-lens mention is a top-10 token of a
  linear readout of the same residual, so it should correlate with a linear probe almost
  mechanically. Its tests are reported (they check that the pipeline can see an effect), but the
  verdict rests on the s3d O-lens; NLA is reported alongside.
- **Hypothesis verdict (pre-registered).** Computed for the s3d O-lens (primary) and, by the same
  rule, for NLA (secondary), on transfer-passing (category, L) and transfer-passing concepts.
  "Control passes" = J-lens β₁ > 0 and β₂ > 0, each item-clustered 95% CI excluding 0, on the same
  set restricted to token-eligible concepts. The s3d / NLA β₁, β₂ in the table are computed on
  all transfer-passing concepts; an extra row reports them on the token-eligible subset (the
  control's exact set).

  | outcome | rule |
  |---|---|
  | supported | β₁ > 0 and β₂ > 0, both CIs excluding 0 |
  | contradicted | control passes, and β₁ or β₂ is significantly < 0 |
  | not supported | control passes, and both CIs include 0 |
  | inconclusive (mixed) | control passes, one coefficient significantly > 0 and the other null |
  | inconclusive (control failed) | control fails and the outcome is not *supported* |

  *Strongly supported* = supported AND, under τ_cat (one threshold for all three terms),
  (det(M) − det(U_matched)) / (ceiling − det(U_matched)) > 0.5 with CI excluding 0.5; if the
  denominator's CI includes 0 the strong criterion is *undefined* and reported as such.
- **Plots** (`plots/sibling_probe/`): (1) per arm × category grouped bars of detection rate for
  gold mentions, M and U_matched, with CIs; (2) per arm, the probe-verified hallucination rate
  (τ_cat and τ(c | g)); (3) Test 2 scatter of mentioned vs unmentioned z per M cell. Slide
  quality, 150 dpi, legend outside.

## Layout

```
plans/0003-sibling-concept-probes.md
data/sibling_probe/
  README.md                 schema + provenance
  pilot/specs.json          pilot prompts (40)
  concepts.json             concept inventory
  items.json                bank (+ responses)
  probe_texts.jsonl         probe training + neutral texts
src/hallucination_bench/evals/sibling_probe/
  __init__.py
  concepts.py               inventory loader, alias matcher (mentions in text / token bags)
  bank.py                   items, specs/capture-rows writers, leak + overlap checks
  pilot.py                  pilot gate counts
  judge.py                  prefilter + claim-map judging (imports claimmap_v2; no copied prompts)
  probes.py                 DiffMean fit, held-out AUROC gate, thresholds, scoring (pure numpy; used by the Modal job)
  score.py                  M/U detection, within-item AUROC, bootstrap, verified hallucination rate
  report.py                 results tables + plots
scripts/sibling_probe/
  probes_modal.py           Modal: stream acts, fit + score, write small outputs
  gen_items.py              Opus generation of items + probe texts (cached)
  run_modal.sh              the documented GPU commands (responses, capture, s3d, jlens, nla)
tests/test_sibling_probe.py
docs/results/0003_sibling_probes.md
CLAUDE.md                   + sibling_probe section
```

The CLI gains `hbench sibling {pilot,judge,probes,score,report}`.

## Tests

- The alias matcher: word boundaries, CJK aliases, no cross-concept overlap.
- Leak check and 8-gram overlap check.
- DiffMean on synthetic Gaussians: recovers the planted direction; τ is the 95th percentile.
- M/U assignment: gold, prompt-named, mentioned and unmentioned cases; J-lens token+summary rule.
- τ(c | g) and z(c | g) on synthetic scores; Test 2 on a planted example.
- Positions resolved by the rows writer equal `wsbench.readplan.resolve` on the same tokens.
- The bootstrap is seeded and deterministic.
- The judge module renders exactly `claimmap_v2`'s prompt for a fixed cell (no prompt drift).
- All existing tests pass unchanged; any edit to a pre-existing test is a red flag.

## Cost (estimate)

- Readouts: about 120–140 items × ~20 positions → about 11k s3d cells, about 11k J-lens cells
  and about 2.7k NLA cells.
- Capture: about 120 eval items and about 5.5k probe texts.
- Modal total about $150–300. Judge: prefiltered cells only, about $300–800 (Opus 5.5). Camila
  has no budget cap; spend is reported.

## Out of scope

- Steering or ablation along the probe directions.
- Other lenses beyond the three arms.
- Settings beyond user modelling; the results doc lists candidates (e.g. sports, musical
  instruments, languages, dog breeds, car brands, cuisines).
