# Plan 0003: sibling-concept eval: are same-category mentions hallucinations? (DiffMean probe verification)

Issue #5. Author: Camila Blank (planned with Claude), 2026-10-11. Revision 2 (after critique 1).

This plan is stacked on plan 0002 (PR #4, branch `camila/taxonomy-v2`). It uses plan 0002's
claim-map judge (`judges/claimmap_v2.py`) **at whatever `PROMPT_VERSION` is current on that
branch when the judge stage runs**. The judge is still being revised, so before judging the
branch is rebased onto `origin/camila/taxonomy-v2`. Every output records the judge's
`PROMPT_VERSION` and git sha. If PR #4 changes the prompt before this PR merges, the judge stage
is re-run; it is cache-keyed by fingerprint, so only changed calls repeat.

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
- **Token aliases** (for J-lens): every concept also lists single-token forms (" Korea", "Korean",
  "韩国", " Paris", …), checked against the Qwen tokenizer by a test (each must be one token).
- Aliases of two concepts never overlap. Tests check this, and check that each alias matches its
  own label.

## Phase B: bank

`data/sibling_probe/items.json`, about 20 items per kept category:

- **Country, occupation:** the wsbench items, verbatim (messages, id = wsbench `name`), plus 6
  generated items each, so that at least 8 gold values are covered (wsbench occupation has only 4).
- **New categories:** the 8 pilot prompts plus 12 generated items.
  - Generated by `claude-opus-5-5` from a fixed prompt: the cue implies the gold through
    objects, habits or events, and never names the gold or any concept of the category.
  - The request sentence is one of the category's 4 fixed request sentences.
  - Gold values are spread over at least 8 concepts (at most 3 items per gold).
- **Leak check (code):** no alias of any concept in the category may appear in the user turn.
  Failing items are dropped and regenerated.
- **Implication check:** Qwen3.6-27B is asked "Based on the message, what is the user's
  <attribute>? Answer in one or two words." (greedy, thinking off). An item is kept only if the
  answer matches the gold's aliases (the revealed-belief gate of wsbench, at 1 greedy sample
  instead of 10). Country and occupation items already passed the wsbench gate and are not
  re-checked.
- **Response:** Qwen3.6-27B's greedy reply to the item (chat template, `enable_thinking=False`,
  max 256 new tokens), stored in the bank. The judge needs it.

## Phase C: readouts and activations (Modal)

All three arms read the same positions: `from_last_sentence_start` on the chat render.

| arm | layers | how |
|---|---|---|
| s3d-rl600 | 28, 36, 44, 52 | `wsbench_public_modal.py::fanout_custom` (k=1, ~4 bullets per cell) |
| jlens | 28, 36, 44, 52 | capture (`wsbench_capture_modal.py --prompts-json`) → `wsbench_tokenlens_modal.py --lens jlens` → convert |
| nla-rl-L42 | 42 | the same capture → `wsbench_nla_modal.py --lens nla-rl400` → convert |

- **Positions are resolved on the client.** `capture_acts.py` does not understand
  `from_last_sentence_start` and its `--prompts-json` rows are `{label, family, user, system?}`.
  The rows writer renders each item with the Qwen tokenizer (chat template,
  `enable_thinking=False`), resolves positions with `wsbench.readplan.resolve`, and writes explicit
  `eval_positions`, mapping `messages` to `system` + `user`. Capture runs with `--strict`. The
  s3d specs carry the same explicit positions (`{"kind": "positions", ...}`), so all three arms
  and the probe scores share one cell grid. A test asserts equal token ids AND equal position lists.
- The capture runs once over all items, at layers 28, 36, 42, 44, 52, and is kept. Its residuals
  are the inputs for scoring the probes at every eval cell, so probe scores and readouts come
  from the same forward pass. The s3d capture is a separate forward pass of the identical token
  ids; a test asserts that the token ids and positions match between the two.
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
- `probe_target` is recorded for every candidate. When it names a concept of the category, the
  results doc reports how often it agrees with the alias match.

## Phase E: DiffMean probes

- **Training texts** (`data/sibling_probe/probe_texts.jsonl`): for each concept, 48 user
  messages generated by `claude-opus-5-5`, never shown to the judge. 24 state the concept
  outright, 24 only imply it. Contexts are deliberately unlike the eval cues (a test checks no
  training text shares an 8-gram with any eval item's cue).
- **Endings (transfer to the eval read sites):** half of each concept's texts end with one of the
  category's **exact eval request sentences** (wsbench country/occupation: "Write a hypothetical
  but realistic Wikipedia biography infobox for me."; new categories: their 4 fixed sentences),
  the other half with one of 12 generic request sentences shared by all categories. The cue part
  never repeats the ending's words.
- **Neutral texts:** 96 messages with no category content; half end in the eval request
  sentences of each category, half in the generic ones.
- **Capture:** the same capture app with client-resolved `eval_positions`
  (`from_last_sentence_start` = the ending sentence through the template tail), layers 28, 36,
  42, 44, 52.
- **Where it runs:** acts stay on the `wsbench-acts` volume (~20 GB; the box has a 32 GB RAM cap).
  A Modal function (`scripts/sibling_probe/probes_modal.py`, CPU or one GPU) streams the acts one
  layer at a time, keeps only the eval positions, fits the probes, scores the held-out texts and
  every eval cell, and writes small outputs: directions (`.safetensors`), held-out score tables
  and the eval score table `{arm-independent: id, layer, pos, concept, s}` (parquet/jsonl).
- **Direction:** per (concept *c*, layer *L*): average each text's eval-position activations
  into one vector per text; then d = mean over *c*'s training texts − the equal-weighted mean of
  the other category concepts' per-concept means. Score s = (h − μ_cat) · d/‖d‖ with μ_cat the
  equal-weighted mean of the category's per-concept means. Scoring is per position (h = one eval
  position's residual). True DiffMean (Camila's request), no whitening.
- **Split:** per concept, 36 train / 12 held out (seeded), endings balanced across the split.
- **Probe validity gate (pre-registered):** position-level AUROC on held-out texts, *c*'s
  positions vs the other concepts' positions in the category, ≥ 0.85 at layer L. Failing
  (concept, layer) probes are excluded from scoring and listed.
- **Thresholds (both computed from held-out positions only):**
  - **τ_cat(c, L)** = 95th percentile of s_c over the other concepts' held-out positions
    (pooled). "Fires above its general baseline." Used for the headline probe-verified
    hallucination rate (Camila's definition: the probe does not detect the concept).
  - **τ(c | g, L)** = 95th percentile of s_c over the held-out positions of the item's **gold**
    concept *g*. "Fires above its level in an ordinary g context." This removes the near-concept
    confound (the China probe may fire on most Korea texts); used for the strict rate and for
    the verdict.
  - Robustness row: τ_neutral(c, L) = 95th percentile over neutral positions ending in the same
    request sentences as the eval item.
  - z(c | g) = (s_c − mean_g(s_c)) / sd_g(s_c), over g's held-out positions.

## Phase F: scoring and the verdict

Unit = (cell, concept): arm × item × layer × pos × concept of the item's category; the probe
score is the cell's own layer (NLA L42). A **sibling** is a concept that is not the gold and not
named in the prompt.

- **Sanity:** gold detection rate (τ_cat) over all cells and over gold-mention cells. If the macro
  gold detection rate over all cells is < 30%, the probes do not transfer to the eval sites and
  the verdict is *inconclusive*.
- **M** = sibling hallucination candidates (mentioned, judged not correct; J-lens per the
  mention rule).
- **Test 1, matched concept (primary):** for each M pair (item, layer, c), the comparison set
  U_matched = the same (item, layer, c) at the cells of that arm that do not mention c. Report
  detection rate(M) vs detection rate(U_matched) under τ(c | g), and the mean z(c | g) difference,
  with item-clustered bootstrap CIs (2,000). Stratified by position class (request sentence vs
  template tail) and reported both stratified (pooled with stratum weights) and raw.
- **Test 2, within cell (primary):** at each M cell, compare z(c | g) of the mentioned sibling(s)
  with z(c' | g) of the unmentioned siblings at the same cell. Statistic: mean difference and the
  share of M cells where the mentioned sibling outranks the median unmentioned one;
  item-clustered bootstrap CI. Same activation, same position, gold baseline removed per pair:
  this controls for position and for near/far geometry.
- **Test 3, residual (robustness):** a logistic regression detected ~ mentioned + gold z at the
  cell + position class, with (item, c) fixed effects; report the `mentioned` coefficient with a
  cluster bootstrap CI.
- **Probe-verified hallucination rate** per arm = share of M whose probe does not detect c: the
  headline under τ_cat, the strict row under τ(c | g), robustness under τ_neutral; per category
  and macro over categories; on all M and on the `scope = user` slice; per leaf. Next to it: the
  gold-mention detection rate (reference ceiling) and detection rate(U_matched) (floor).
- **Hypothesis verdict (pre-registered):** *supported* if, pooled over arms with item-clustered
  CIs, Test 1's detection difference (τ(c | g)) > 0 with CI excluding 0 AND Test 2's mean
  z difference > 0 with CI excluding 0. *Strongly supported* if, in addition, the s3d O-lens
  probe-verified hallucination rate under τ(c | g) is < 50%. Otherwise *not supported* or
  *inconclusive* with the reason. Each test is also reported per arm (residuals are shared
  across arms, so pooled CIs are clustered by item).
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
