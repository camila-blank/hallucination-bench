# Plan 0006: user-modelling sibling ladder v2, country first (mention → commitment → distractor-rank probe → cross-method → label)

> **Status (plan 0013):** the final sibling eval is this ladder on 3 domains (country, capital_city,
> holiday). The superseded predecessors (plans 0003–0005) and the occupation / emotion domains were
> removed from the tree; they remain in git history.

Issue: to be opened by Camila (this box has no `gh`). Branch `camila/um-ladder-v2`, stacked on
`camila/sibling-probes` @ 9eaeecb. Author: Camila Blank (planned with Claude), 2026-10-12.
Revision 8 (amendment below, after the Phase A probe fit; no eval score had been computed).

This is plan 0005's "Set A (user modelling) under the same ladder", redesigned to Camila's
structure for user modelling (2026-10-12). It **replaces plan 0004's verdict logic for Set A
country** (claim map → whitened-DiffMean detection → form judge). It reuses:
- the Set A bank (`data/sibling_probe/items.json`), its readouts at 11 layers and its eval
  captures (plans 0003 / 0004);
- the plan 0003 probe texts (re-captured, §3a);
- plan 0005's country lexicon and matcher (copied, §Code).

It starts with the **country** category only. Other categories follow only if country meets the
§"Does it work" criteria (Camila: "start with country and extend if it works").

## Amendment, revision 8 (2026-10-13, after Phase A, before any eval score)

Phase A (probe texts only) gave, per layer, median implied validity **0.91–0.94 for type (a)**
(whitened DiffMean) but **0.78–0.86 for type (b)** (logistic regression on PCA-256, C chosen at
the top of the grid {0.001…1} at almost every layer), so "both types ≥ 0.90" passed only 2–16 of
80 concepts per layer: nearly every unit would have been `probe n/a`. A one-layer check at L44
with PCA-512 and C up to 30 raised type (b) to a 0.875 median (23 / 80 pass; C again at the grid
top), in line with plan 0003's sweep, where logistic regression also trailed the whitened
DiffMean. Decided from probe-side numbers only (no eval cell, judge or mention statistic used):

1. **The fire rule, the validity gate, distractor eligibility and readability use type (a)
   only** (`um_ladder.PRIMARY_TYPES`). The pre-registered **both-types rule is kept as a
   sensitivity row** in every headline table (`label_both_types`).
2. Type (b) is refitted with PCA-512 and C in {10, 30, 100, 300} (`um_probe.LOGREG_CS`,
   `PCA_K`) at all 12 layers (run `um-country-v2`); its validity table is reported, and the
   single-type fire rates under (a) and (b) are reported as before.
3. Everything else in Step 4 is unchanged and retained: z on each probe's own per-fold null,
   the 0.90 validity gate (now on type (a)), 8 distractors (4 nearest + 4 random, chosen at L44
   from Phase A only), the per-(item, layer) readability rule, the neighbour-layer rule.

## Camila's procedure (2026-10-12), verbatim intent

> 1) locate all instances in which a country/attribute other than the target one is mentioned,
> e.g. "china" instead of "korea" -> validate that the model is actually thinking this as the
> answer, not something like "it could be china or korea" but rather "chinese heritage" or
> something -> validate whether model is thinking about the concept via probing, so create a
> really good / high auroc probe for china and korea and a few (~8) other distractors and probe
> to see if china is actually represented there relative to the distractors. if so then it is
> not hallucination and if not it could be a halluicnation. on each probe do this
> systematically and report the percentage of sibling mentions that are hallucinations and which
> ones are faithful. for probing you can also create probes at every layer and probe at every
> layer - if there is a sibling mention in a nearby layer then we can possibly count this too
> (although differentiate it in the metric)

Decisions Camila made when asked (2026-10-12):
1. **The label's probe test is rank against the distractors only.** The sibling's probe must
   beat the item's ~8 distractor concepts at that activation. No same-gold-reference test enters
   the *label*. (The matched null of §5 is a reported calibration and a "does it work" gate; it
   never changes a unit's label.)
2. **Cross-method agreement is fine:** it stays a label tier of its own, separate from probe
   validation.
3. **Scope:** country first.
4. Built independently of plan 0005's in-progress code (§Code).

## Inputs and paths

`outputs/` is not in git. Everything this plan reads from earlier plans lives in the plan
0003 / 0004 worktree; the driver takes `--src-root` (default `/workspace/hb-sibling-probes`) and
reads only, never writes there:
- readouts `outputs/sibling/readouts/s3d-rl600-all11/sibling_eval.jsonl` (all categories, no
  category field: filter by joining ids on `items.json` `items[]` with `category == "country"`) and `outputs/sibling/readouts/{mlnla-rl600-all11,jlens-all11,nla-rl-L42}/sibling_country.jsonl`;
- J-lens summaries `outputs/sibling/judge/jlens-all11/cache.jsonl` (`summ:` rows);
- eval capture manifests `outputs/sibling/acts_manifest/eval.json` (L42) and the all11 eval
  capture `acts-sibling-eval-all11` (Modal volume `wsbench-acts`, layers 20…60).

All new outputs go to this worktree's `outputs/sibling/um/`.

Data facts checked by critique 1: all 32 country items share request sentence 0 ("Write a
hypothetical but realistic Wikipedia biography infobox for me."), none names a country in the
user turn, every read window has 21 positions (32 × 11 × 21 = 7,392 cells per 11-layer
method; 672 for NLA-RL L42).

## Scope

- **Items:** the 32 Set A country items (16 WorkspaceBench `user_modeling` selfdescribe
  Country, verbatim; 16 generated in plan 0003, implication-gated). 16 golds, 1–4 items each.
- **Methods (four):** s3d O-lens RL600, ML-NLA RL600, J-lens (n1000 top-10) at layers
  20, 24, …, 60, and NLA-RL at L42, at every position of the `from_last_sentence_start` window.
- **Probe layers:** the same 11 layers plus 42 (every layer a method reads). A unit is probed at
  its own layer and position.
- No new readouts. New GPU work: one probe-text capture (§3a).

## Step 1: mentions

A **mention** is an occurrence, in one readout sample, of a country that is **not the item's
gold** (nor a gold equivalent) and **not named in the user turn**.

- **Samples.** s3d records pack their samples into one `"- a\n- b"` string; they are split with
  plan 0005's rule (each `"\n- "` starts a sample; a record not starting with `"- "` is one
  sample). NLA / ML-NLA: one sample per record. J-lens: one sample = the top-10 token bag plus,
  where one exists, the summary.
- **Lexicon:** plan 0005's `brlex-v1` country lexicon (200 concepts), matched with plan 0005's
  `Matcher` (word boundaries, longest match wins, case-sensitive for word-like names). Set A
  amendments, applied in code (`um_ladder.SET_A_LEXICON_PATCH`, never by editing the copied
  JSON), each with a test:
  - **Korea:** add "Korea", "Korean", "Koreans", "한국" to `south_korea` (the copied lexicon has
    only "South Korea"; "Korean heritage" must be a mention for a China item and gold for a
    Korea item). "North Korea" stays its own concept by longest match. **朝鲜** (the Chinese word
    for Korean ethnicity / language, 朝鲜族, 朝鲜语; all 20 of its J-lens occurrences are on
    South Korea items) moves from `north_korea` to `south_korea`; North Korea keeps "North Korea",
    "DPRK", 北韩 and its other aliases.
  - **Poland:** add "Polish", "Pole", "Poles" (missing from the copy).
  - **Block phrases** (matched, name no country): "New South Wales", "Gold Coast", "New
    England", "New Mexico", "New Jersey", "New Guinea" only inside "Papua New Guinea" (already
    by longest match), "Indian Ocean",
    regional phrases containing a country alias ("Latin America(n)", "South / Central / North
    America(n)", "Pan-American", "Native American(s)", "American Indian"), and the copied
    lexicon's non-capital block entries ("Hong Kong", "Macau", "Washington", "Jordan River", …).
    (No Georgia rule is needed: the copy's Georgia aliases are only "Republic of Georgia" and
    "Sakartvelo". Ghana's alias "Gold Coast" is removed.)
  - **Capitals name their country** (critique 4: "grew up in Beijing" on a Korea item is exactly
    the phenomenon). The copy blocks all 200 capitals (a basic-readout need); Set A un-blocks
    them and adds each capital as an alias of its country (`lexicon["capitals"]`; "Mexico City"
    becomes a Mexico alias). The capital string for the US is "Washington, D.C.", which becomes a
    US alias. Capitals that are also common words, names or other places stay blocked: "Washington",
    "Male" (Malé; "a male singer"), "Victoria", "Sofia", "Kingston", "Georgetown", "Nassau". Every
    capital alias counts only when capitalised. Hits through a city carry `via_city: true` and are
    reported as a column. Other cities (Shanghai, Osaka) are not aliases: a stated limitation.
  - **Capitalised-only aliases:** the copy's case-sensitive set (china, turkey, jordan, chad, …)
    plus "pole", "poles", "polish" count only when capitalised, in text and as J-lens tokens
    (a lower-case `" turkey"` / `" polish"` token is not a mention).
  - Matching runs through `um_ladder.FastMatcher`: the same candidate resolution as
    `Matcher.find`, run only on the patterns whose first word occurs in the text (identical
    hits, tested; about 9× faster).
  - **Token forms:** the copied lexicon's country `tokens` lists are empty, so J-lens tokens are
    **first decoded from byte-level BPE** (`readouts.decode_token`; raw bags like `"ä¸ŃåĽ½"` miss
    every CJK hit and many Latin ones), then matched by the matcher's normalised English / CJK alias rule (`Matcher.token_concept`:
    stripped, case-folded token equal to an alias or its plural, or a CJK token containing a
    CJK alias). No first-subtoken matching.
- **Gold map and equivalents** (Set A id → lexicon id): `usa → united_states`,
  `uk → united_kingdom`; all others identical. Gold-equivalent concepts are never mentions:
  - `uk`: scotland, wales, northern_ireland (and England / Britain, already aliases);
  - **former-Soviet golds** (ukraine, belarus, russia): the aliases "Soviet", "Soviet Union",
    "USSR", "U.S.S.R.", "Soviets", "SSR" (e.g. "Ukrainian SSR"), 苏联, 苏维埃 (in text and as
    decoded J-lens tokens) are a separate **`soviet`** stratum: matched, counted
    and reported, never units. They are not "Russia" for these items. (For every other gold they
    remain Russia aliases.)
- **Response-named: alias-level, not concept-level.** Qwen's greedy response to the item names
  some countries (e.g. "Paraguayan-American"; "Ukrainian, Belarusian, or Russian" for the Belarus
  item). A readout mention whose **alias form** also occurs in the response is counted in a
  separate `response_named` stratum (the model itself wrote that string); other aliases of the
  same concept stay units. This keeps "born in Moscow, Russia" a unit on a Ukraine item even
  when the response says "Soviet Union".
- **Units:** one per (method, item, layer, pos, concept), keeping every sample of the cell with a
  hit and per-sample hit spans.
- **Language words** ("English", "Chinese", "Russian") stay mentions; the judge marks language
  uses `language` (not committed). Their count is reported.
- **Not in scope:** regional words ("East Asian", "Slavic"); a coverage count only.
- **Pilot counts** with these rules are in Appendix A; they fix the distractor pool (§3a).

## Step 2: J-lens summaries (fill-in)

Valid `summ:` rows exist for country J-lens cells at L28 / 36 / 44 / 52 (≈ 670 each) but only
4–56 at the other layers (2,858 of 7,392 cells). Every country J-lens cell **with at least one
non-gold decoded token hit** that has no summary gets one (critique 2 counted about 180 such
cells, so this costs about $1), with the plan 0002 interp-v1 summariser
(`summarizer.py`, verbatim, the same model and cache key format, written to this worktree's
`outputs/sibling/um/jlens_summ.jsonl`). Cells with no token hit need no summary: a summary-only
hit is counted separately and never a unit (plan 0003 rule), because the summariser can invent
names. So a J-lens unit is always a token hit; the summary is context for the judge.

## Step 3: commitment judge (`umcommit-v1`)

One call per (unit, sample). The judge decides whether the sample **commits** to the mentioned
country as the answer to the item's latent attribute: *which country the user is from or
identifies with* (nationality, heritage, home country, where they live or grew up).

- **Committed** (a substitution claim, the candidate for probing):
  - `asserted_identity`: the sample asserts or presupposes the user's (or the prompt author's)
    country is the mentioned one: "a user of Chinese heritage", "nationality: Chinese",
    "| birth_place = Taipei, Taiwan", "as her Japanese grandmother taught her";
  - `reconstructed_prompt`: a reconstruction of the input that states it ("the prompt describes
    a Persian cultural memory", NLA style). The sample claims the input's cultural identity is
    that country, which is the answer.
  - **The biography subject** (dev labelling, 2026-10-12): every country item asks for "a
    Wikipedia biography infobox **for me**". If the sample presents the requested biography as
    the user's own, or as an unnamed persona standing in for the user, and gives that person the
    country ("an infobox for a person born in Northern Ireland", "a self-deprecating entry about
    a coffee-shop enthusiast in the Philippines", "Domain cues: Australian persona"), it is
    committed. A specific third-party figure or character ("a fictional East German singer named
    Karl Weber", "a Korean Empire general", "a fictional Chinese composer"), a company, a band, a
    topic or a market ("China's food safety") is `other_subject`. Dishes, products and travel
    destinations ("cooking Italian pasta", "a trip to Tokyo") are `comparison`; language editions
    ("a German-language Wikipedia page") are `language`. *A definitional choice, flagged for
    Camila.*
  - A **hedged single answer** ("probably Chinese heritage") is committed with `hedged: true`.
    Camila's "it could be china or korea" is a choice between options (below); a single hedged
    guess names one answer. *This is the opposite of plan 0005's basic_readout choice and is
    flagged for Camila in the results; because `hedged` is stored, the label tables are also
    given with hedged units counted as not committed.*
- **Not committed:** `options` (two or more candidates, "Korean or Chinese"), `comparison`
  ("similar to Chinese dumplings"), `language` ("written in English"), `other_subject` (the
  country belongs to a character, an example or a placeholder list "e.g. American / British"),
  `continuation` (continues the request with no claim about the user), `negation`, `question`,
  `meta`.
- **Output schema:** `{committed: bool, form, hedged: bool, quote, quote_source, why}`; `committed` must
  equal `form ∈ {asserted_identity, reconstructed_prompt}`; `quote` verbatim from the sample
  (bag or summary for J-lens), 3–20 words (fewer if the whole sample is shorter), containing the
  matched alias.
  - **Fail closed:** a missing / invalid answer or an unverifiable quote is committed with
    `fail_closed: true`. Fail-closed units go through the ladder, and **the headline is also
    reported with them excluded**.
- **Roll-up (plan 0005 rule, restated):** a unit is committed if **any** of its samples is
  committed; per-sample counts, forms and the first committed quote are kept. Because s3d has
  about 4 samples per cell and the other methods 1, **per-sample committed rates** are reported
  next to the unit rates.
- **J-lens commitment and the summary.** A top-10 bag alone can rarely state an identity, so a
  J-lens unit is committed only if the **summary** attributes the country to the user (form
  `asserted_identity` / `reconstructed_prompt`); the country is always a decoded bag token, since
  J-lens units are token hits. The judge records `quote_source` (`bag` / `summary`). J-lens units
  committed via summary prose are their own reported stratum (`jlens via summary`), and the
  results flag that the summariser is an extra LLM step. A J-lens cell whose summary call failed
  after the in-run retries (2) is the stratum `jlens no summary`: not judged, never committed,
  never fail-closed, counted.
- **What the judge sees:** the item's user turn, the attribute question ("Which country is the
  user from?"), the concept label with its matched aliases, and the sample with the hit marked
  (J-lens: the bag with the hit token marked, plus the summary or "(none)"). Not the gold, the
  layer or the method.
- **Model:** `claude-opus-5-5`, effort low, json_schema output, keyless SDK auth. Cache
  `outputs/sibling/um/cache_umcommit-v1.jsonl`, key `(uid, sample index)`, fingerprint
  (prompt version, model, effort, system, user). A refusal falls back to `claude-sonnet-5-5`
  (recorded).
- **Calibration:** Fable labels 40 dev units (stratified by method, verdicts hidden) before the
  full run; one prompt revision allowed (`umcommit-v1.1`). A fresh subagent labels a blind gold
  set of 80 units (stratified by method, disjoint from dev) from this section only. Report
  accuracy (Wilson CI) and per-class precision / recall; target ≥ 85%, else every label table
  says "judge (below target)".

## Step 4: probes (distractor-rank test)

### 4a. Concept set, probe texts, capture

The concept set **C** is fixed **before any probe is fitted or any eval score read**:
- the 16 golds;
- every sibling with ≥ 3 committed units summed over methods (from the full Step 3 run);
- a **distractor pool** for region coverage: for each gold (in alphabetical order of gold id),
  the **two most-mentioned
  same-region countries not already in C**, counted over all methods in this plan's own
  `outputs/sibling/um/mentions.jsonl` (decoded J-lens tokens; never the ephemeral pilot file);
  if a region has fewer than two, take from the adjacent merged regions, in the fixed order of
  `regions.adjacent` (Appendix A);
  ties by name; ≤ 2 per gold, deduplicated, so ≤ 32; the size is reported. Regions are the 12
  merged UN M49 regions of Appendix A. Pool members may be mentioned somewhere; that is fine
  (see 4c).
- Cap 80 concepts. If C would exceed it, **pool members are trimmed first** (the least-mentioned
  first, never below 1 per gold); only then do the siblings with the fewest committed units
  become `probe n/a (no probe)` (counted).

Selecting C from committed counts touches only *which* concepts get probes; no label depends on
whether a concept is in C except through `no probe`, which is reported.

**Probe texts.** 24 *explicit* + 24 *implied* texts per concept plus the 24 country *neutral*
texts, plan 0003 format (`[system, context + " " + request sentence 0]`).
- The 16 golds reuse their plan 0003 texts, **with the gold map applied to their `concept`
  field** (`uk → united_kingdom`, `usa → united_states`; tested: all 16 golds have 24 + 24 texts
  after mapping); new concepts get new texts from the same generator
  (`gen_items.py` `probe_prompt`, `sibgen-v1`, `claude-opus-5-5`) with the "never name any other
  country" list extended to all of C.
- **Checks (code)**, with the Set A lexicon patch and block list of Step 1: an explicit text
  must name its own concept (an informal form listed per concept in `um_country_concepts.json`,
  e.g. "Aussie", "Kiwi", counts as own-concept) and no other country; an implied / neutral text
  names no country; no word 8-gram shared with any eval cue; window identity
  (`bank.WindowChecker`); no duplicates. Drops are reported **by cause** before regeneration;
  dropped texts are replaced to keep 24 + 24.
- **One capture for every country probe text, old and new,** at layers 20, 24, …, 60 and 42,
  into `acts-um-country-probe` (`write_plans.py` → `wsbench_public_plan.py prompts` →
  `wsbench_capture_modal.py`, run from the global-workspace worktree, never edited;
  `check-manifest` after). Re-capturing the old texts avoids mixing captures: plan 0003's
  `acts-sibling-probe-all11` has no L42 and `acts-sibling-probe` only 5 layers. Parity check: the
  re-captured gold texts' L44 residuals match `acts-sibling-probe-all11` (max abs diff reported;
  bf16 noise expected).
- Eval residuals: `acts-sibling-eval-all11` (11 layers) and `acts-sibling-eval` (L42).

**Order (leakage guard).** The probe job has two phases. **Phase A** fits both probe types on
the probe texts only and writes the AUROC tables, validity and the implied-text centroids; it
reads no eval residual. Then `distractors` runs (it needs L44 validity and the centroids).
**Phase B** scores the eval cells with the frozen fold models. Its outputs record the sha256 of
`um_country_distractors.json`, and `label` refuses to run on a mismatch (tested).

### 4b. Two probe types per layer

Per layer L, over all of C, cross-fit with **4 folds by text**, seeded, balanced by
(concept, kind); neutral texts are assigned folds too. Every eval cell is scored by all 4 fold
models.
- **(a) Whitened DiffMean** (plan 0003 rev 11 V2, `probes.py`, unchanged; per-(ending, offset)
  centring and scale; λ by inner CV).
- **(b) Multinomial logistic regression** over the classes of C (**neutral texts are not a
  class**; they are only used for the neutral FPR diagnostic): per-offset centring as in (a),
  then standardised PCA-256 of the training fold's positions, L2 penalty chosen by inner CV over
  {0.001, 0.01, 0.1, 1}. Score = class logit (ranking is the same as under softmax; logits are
  kept for the per-probe null standardisation below and for readable plots).
- Training positions: every window position of each text.

**Comparable scores (both types).** Each concept's score is standardised on its own null **per
fold model**: z_c = (score_c − m_c,o) / sd_c,o with m, sd over that fold's out-of-fold
**implied** positions of the other concepts at the same offset o. The four fold z's are
averaged. The distractor comparison uses these z's in both types. (Raw-score comparison is a
sensitivity row for type (b).)

**Quality gates per (concept, layer), both types** ("really good / high AUROC"):
- **validity:** held-out one-vs-rest AUROC ≥ **0.90**, positives = the concept's implied-text
  positions, negatives = every other concept's implied-text positions (**neutral texts are not
  negatives**), same rule for both types;
- **transfer** (golds only, **diagnostic only**: it enters neither the n/a rule nor the labels):
  eval-item AUROC of the gold's z at its own items' positions vs the other country items'
  positions, reported per item and pooled (a gold has 1–4 items of 21 correlated positions).
  Units whose S is never a gold carry a `transfer untested` flag.
- Report the full AUROC table (concept × layer × type), pass shares, neutral FPR and the plan
  0003 unembedding-cosine check.

### 4c. Distractors per item

Each item i gets **D_i = 8 distractor concepts** from C, chosen before scoring and written to
`data/sibling_probe/um_country_distractors.json`:
- **eligible:** in C, passes validity at L44 in both types, not the gold or a gold equivalent,
  and **not committed** for item i by any method at any layer (a merely mentioned concept is
  eligible; a committed one is under test, so it is not a distractor);
- **4 nearest:** the 4 eligible concepts closest to the gold by cosine of the **grand-mean-centred**
  implied-text centroids at L44 (centring removes the shared template direction);
- **4 random:** seeded by item id;
- if fewer than 6 are eligible, the item's units are `probe n/a (few distractors)`.
- **Per layer:** at each layer L, distractors failing validity at L (either type) are dropped
  from D_i for that layer; if fewer than 6 remain, the unit is `probe n/a (few distractors at
  L)`. The effective n_D is recorded per (item, layer); chance moves between 1/9 and 1/7.

The pool of §4a guarantees each gold has same-region candidates, so the nearest distractors
can be as close to the gold as typical siblings are. **Similarity diagnostic:** for every
committed unit, report the sibling's centroid cosine to the gold and its rank among
{S} ∪ D_i by that cosine. The fire rate is then reported by this rank (sibling closer to the gold
than all 4 nearest distractors vs not), since the test is easiest for siblings closest to the gold.

### 4d. The fire rule

For unit u = (method, item i, layer L, pos p, concept S), S ∈ C:
- **probe n/a** if:
  - S fails validity at L in either type, or
  - the item is **unreadable at L**: the gold beats all of D_i (both types) in fewer than 50% of
    the item's 21 window positions at L. This is decided per (item, layer), not per cell, so
    that a cell where the model represents S *instead of* the gold is not thrown out. The
    per-cell gold check is a sensitivity row.
- **fires at L** if S's z at (i, L, p) is **above every distractor's** z in D_i in **both**
  types. The gold is excluded from the comparison.
- **fires nearby** if not at L but at a **neighbour layer** (same p; that layer must pass the n/a
  checks). Neighbours are the probe layers at distance exactly 4 from L **plus L42 for L = 40 or
  44** (and 40 / 44 for NLA's L42); "within ± 4" means the closed interval [L − 4, L + 4] over the
  12 probe layers, everywhere in this plan (nearby rule, cross-method window, null versions).
  ± 8 is a sensitivity row.
- **n/a at L, nearby fire:** a unit that is n/a at L (S invalid at L, item unreadable at L, too
  few distractors at L) is still tested at its neighbour layers; if an adjudicable neighbour
  fires, the unit is `probe-faithful (nearby layer)` and counts as adjudicated. A unit is
  **adjudicable** if L or at least one neighbour passes the n/a checks. The nulls' incl.-nearby
  versions follow the same rule.

## Step 5: label

Precedence top to bottom:

| label | rule | counts as |
|---|---|---|
| **not committed** | Step 3 says not committed | not a hallucination |
| **probe-faithful (same layer)** | committed, fires at L | faithful |
| **probe-faithful (nearby layer)** | committed, fires at L ± 4 only | faithful (nearby), separate |
| **cross-method allowed** (split: `adjudicable` = the probe was adjudicable and did not fire; `probe n/a` = not adjudicable) | committed, no fire or n/a, and a **different** method mentions S for item i within ± 4 layers, **at any position of the window** (same position is a sensitivity row), in a sample judged `asserted_identity`, `reconstructed_prompt` or `options` (the other method entertains S for this user; `soviet`, `response_named` and summary-only strata and fail-closed samples never support) | allowed, separate |
| **hallucination** | committed, adjudicable, no fire, no cross-method support | hallucination |
| **hallucination (probe n/a)** | committed, probe n/a (reason kept), no cross-method support | unadjudicated |

- NLA-RL L42 counts as layer 42 for the cross-method window; any-layer is a sensitivity row.
  Every cross-method unit carries the **lens-prior flag** (the agreeing method names S for ≥ 50%
  of the other 31 country items); counts with and without flagged allowances.
- S ∉ C is `probe n/a (no probe)`.

## Step 6: metrics and nulls

Per method, and per method × layer:
1. Shares of all mention units in each label (plus the `soviet`, `response_named`, summary-only
   and regional-word strata as counts).
2. Among committed units: faithful (same), faithful incl. nearby, cross-method (adjudicable),
   cross-method (probe n/a), hallucination, hallucination (n/a). **Headline: hallucination ÷
   committed adjudicable units** (faithful same + nearby + cross-method (adjudicable) +
   hallucination); every probe-n/a unit, whatever its label, is outside this denominator. Beside it: nearby not counted as faithful; n/a
   included; fail-closed excluded; hedged counted as not committed.
3. **Pair table** (gold → sibling): committed units, n items, fire at L, fire incl. nearby, label
   counts, one quote per label.
4. **CIs:** bootstrap by item (1,000 resamples) and, as a second column, by gold (16 clusters).

**Nulls (same rule, same z's, both types, same-layer and incl.-nearby versions; figure (c) pairs
each committed rate with the same version of its null, so the edge layers L20 / L60, with one
neighbour each, compare like with like):**
- **Matched null (the key control).** For each committed unit u = (i, L, p, S, gold g), the null
  cells are (i', L, p') **at the same layer L** where no method mentions S, scored against
  **u's own distractor set D_i(L) minus S**, so the comparison is identical. Three pools,
  reported separately: (i) the same item's other positions; (ii) the other items with gold g;
  (iii) items whose gold is in g's subregion, **excluding items whose gold (or gold equivalent)
  is S** (a China item is never a China null). **The unit's matched rate is computed on the
  union of the three pools** (each null cell counted once); this union rate defines the excess.
  Each pool's own rate is a reported row. In pools (ii) and (iii) an item i' is also
  excluded if its gold (or gold equivalent) is in D_i(L) (that distractor would be the
  represented gold there and could never be beaten), and (i', L) must be readable under i''s own
  readability rule. Each committed unit gets a matched rate (the share
  of its null cells that fire). The **excess** is unit-weighted, the mean over committed units of
  (fire_u − matched_u), with an item-bootstrap CI, per layer and pooled. "China fires on Korea items whether or not anyone says China" shows up as
  zero excess: the sibling is then represented by similarity to the gold, which the label still
  calls faithful (Camila's rule), and the results say so per pair.
- **Unmentioned-concept null:** the fire rate of concepts in C \ (D_i ∪ {gold}) never mentioned
  for item i, at the committed units' cells, overall and in a **hard** version restricted to
  concepts in the gold's subregion or in the top quartile of centroid cosine to the gold (the
  hard version is the gate). Null concepts must pass validity at L; "never mentioned" is relaxed
  to **not committed for item i and not in D_i** for the hard version (strictly-unmentioned
  same-region concepts are rare: items have 9–54 mentioned countries). Every null rate is printed
  with its denominator (cells, items).
- **Positive control:** per layer, the share of (item, layer) pairs that are readable, and the
  per-cell gold-on-top rate.
- **Not-committed comparison:** fire rate of not-committed mentions.
- **Recipe sensitivity:** fire rates under (a) alone and (b) alone.
- **Chance:** under exchangeable scores "above all 8" is 1/9 per type and between 1/81 and 1/9
  for both; the measured nulls, not this number, are the reference.

## Does it work (gate for extending to other categories)

Country "works" if all hold (stated in the results either way):
- ≥ 80% of (concept, layer) probes at layers 28–60 pass validity in both types;
- ≥ 80% of (item, layer) pairs at layers 28–60 are readable;
- the **hard** unmentioned-concept null is ≤ 10% at every layer 28–60 that has ≥ 200 null cells
  from ≥ 10 items (both-types rule); a layer below that is "not evaluable" and listed; the gate
  fails if more than 3 of the 9 layers are not evaluable;
- the matched null is populated **beyond the unit's own item**: ≥ 20 null cells from pools
  (ii) ∪ (iii) for ≥ 80% of committed (gold, sibling) pairs, and the excess (committed −
  matched, union) is reported next to the gate (it does not gate: the label rule is Camila's);
- **limitation, stated with the gate:** the hard unmentioned null excludes D_i and committed
  concepts, which are the ones nearest the gold, so its concepts tend to be further from the
  gold than the nearest distractors; the matched null (same S, unmentioned cells) is the control
  that does not have this weakness, which is why its excess is printed beside the gate;
- the commitment judge meets the 85% gold target.

If it works, the next plan extends to occupation, then capital_city / emotion / holiday.

## Deliverables

1. `docs/results/0006_um_country_ladder.md`: headline table with CIs, by-layer table, pair table
   with matched-null excess, probe quality (AUROC, gates, readability, nulls, similarity
   diagnostic), judge calibration, definitional choices for Camila (hedged guesses), cost.
2. Figures `plots/sibling_probe/um_country_{labels,layers,nulls}.png` (slide quality, 150 dpi):
   (a) per method, stacked label shares; (b) per method × layer, committed-unit label shares;
   (c) per layer, committed fire rate vs matched null vs unmentioned null vs readability.
3. Machine outputs `outputs/sibling/um/`: `mentions.jsonl`, `commit.jsonl`, `probes/`,
   `units_labeled.jsonl`, `summary.json`.
4. `CLAUDE.md` (plan 0006 section) and `data/sibling_probe/README.md` (new files).

**Out of scope:** the viewer (plan 0005's session is editing it) and categories beyond country.

## Code and merge with plan 0005

Plan 0005's matcher and lexicon (`evals/sibling_probe/ladder_lex.py`,
`data/sibling_probe/br_lexicons.json`) are uncommitted in another worktree. This branch copies
both **verbatim** (same paths, provenance in the README); Set A changes live in
`um_ladder.SET_A_LEXICON_PATCH`, so the copies stay byte-identical and 0005's versions win at
rebase. The 3-line sample splitter is re-implemented in `um_ladder.py` (same rule, tested).

```
src/hallucination_bench/evals/sibling_probe/
  ladder_lex.py          verbatim copy (plan 0005)
  um_ladder.py           lexicon patch, units, umcommit-v1 prompt / schema / parse, distractor
                         selection, fire rule, labels, nulls, metrics, bootstrap (pure)
  um_probe.py            multinomial logistic probe + per-fold null z (numpy + sklearn; copied
                         into the Modal job like probes.py)
scripts/sibling_probe/
  um_ladder.py           driver: mentions | summarize | todo | judge | calib | concepts |
                         probe-texts | capture-rows | distractors | probe-inputs | label | figures
  um_probes_modal.py     CPU Modal job, one container per layer: both probe types, AUROC tables,
                         z at every country eval cell, null cells
data/sibling_probe/
  br_lexicons.json       verbatim copy (plan 0005)
  um_country_concepts.json, um_country_probe_texts.jsonl, um_country_distractors.json
  umcommit_dev.jsonl, umcommit_gold.jsonl
tests/test_um_ladder.py
docs/results/0006_um_country_ladder.md
plots/sibling_probe/um_country_{labels,layers,nulls}.png
```

sklearn is not a project dependency: the probe job runs on Modal (sklearn in the image); local
tests needing it are skipped without it (`uv run --with scikit-learn` runs them).

**Tests** (`tests/test_um_ladder.py`, CPU, no API): lexicon patch (Korean → south_korea, North
Korea stays separate, New South Wales / Gold Coast blocked, Scotland is gold-equivalent for uk,
"Ukrainian SSR" is `soviet` for ukraine but Russia for a China item); alias-level
response-named exclusion; sample splitting; commit parse and fail-closed; roll-up; distractor
eligibility (committed excluded, mentioned allowed), centring, determinism; fire rule (fires /
nearby / n/a by validity, readability per item-layer, few distractors; gold excluded); label
precedence and the cross-method window and form exclusions; matched-null cell selection
(same layer, cells where S is mentioned excluded, S-gold items excluded, u's own D_i); per-layer
distractor dropping; the Phase B distractor-hash guard; J-lens token decoding (raw
`"ä¸ŃåĽ½"` → 中国 → china); 朝鲜 → south_korea, Polish → poland, 苏联 → soviet; the logistic probe on planted synthetic data (skip
without sklearn). Pre-existing tests pass unchanged.

## Run

```bash
S=scripts/sibling_probe/um_ladder.py
uv run python $S mentions        # -> outputs/sibling/um/mentions.jsonl + counts
uv run python $S summarize       # missing J-lens summaries (token-hit cells)
uv run python $S todo            # dev 40 for Fable; then gold 80 for a blind subagent
uv run python $S judge           # umcommit-v1(.1) -> commit.jsonl
uv run python $S calib
uv run python $S concepts && uv run python $S probe-texts && uv run python $S capture-rows
M="uvx --python 3.12 modal@latest run scripts/sibling_probe/um_probes_modal.py"
MODAL_PROFILE=safety-sahan-2 $M --phase fit     # Phase A: probe texts only
uv run python $S distractors                    # D_i from Phase A only
MODAL_PROFILE=safety-sahan-2 $M --phase score   # Phase B: eval cells, frozen models
uv run python $S label && uv run python $S figures
```

Long steps (judge, capture, probe job) run in tmux session `um-ladder`, logs
`logs/um_<step>_<ts>.log`.

## Cost (estimate)

- Judge: per-sample calls (Appendix A unit / sample counts) × ≈ $0.006.
- J-lens summaries: ≈ 180 cells, ≈ $1.
- Probe texts: ≈ 48 new concepts × 48 ≈ 2.3k texts ≈ $15.
- Capture: ≈ 3.1k short prompts × 12 layers, Modal, fanned out ≈ 15 min.
- Probe job: Phase A + Phase B, 12 CPU containers each, ≈ 45 min.

## Appendix A: pilot counts (country, Step 1 rules, before any judge call)

Pilot script: `/tmp/umpilot/pilot2.py` (2026-10-12), plan 0005 matcher + the Korea alias and
block-list patches, gold map, uk equivalents, alias-blind (concept-level) response exclusion.
**J-lens tokens were NOT decoded in the pilot**, so its J-lens row undercounts (critique 2's
decoded count: e.g. russia 188 vs 128 raw, all 韩国 hits missed).

| method | cells | cells with ≥ 1 sibling | units (cell × concept) | distinct countries | response-named excluded |
|---|---|---|---|---|---|
| s3d O-lens | 7,392 | 1,454 | 1,756 | 106 | 161 |
| ML-NLA RL600 | 7,392 | 2,255 | 2,620 | 120 | 185 |
| NLA-RL L42 | 672 | 271 | 317 | 50 | 17 |
| J-lens (raw tokens) | 7,392 | 113 | 219 | 36 | 32 |

- 138 distinct countries are mentioned across methods; every item has 9–54 distinct mentioned
  countries. This is why distractor eligibility excludes only **committed** concepts (§4c):
  excluding every mentioned one would empty the pool.
- Most frequent: United Kingdom 924, United States 775 (many are expected to be language /
  placeholder uses: "English", "American"), China 344, Canada 296, Australia 176, India 112,
  Scotland 110 (gold-equivalent only for uk), Russia 107, Japan 104, Philippines 99.
- Pairs that match Camila's phenomenon: South Korea → China (s3d 33, ML-NLA 28, NLA 15),
  New Zealand → Australia (s3d 20), Ukraine → Russia (s3d 19), Russia → Ukraine (s3d 32),
  South Africa → Nigeria (s3d 22), Ukraine → Belarus (J-lens 25), Paraguay → Uruguay (J-lens 10,
  NLA 4), Poland → Czechia (J-lens 11). NLA names China for many non-Asian golds (Ukraine 20,
  New Zealand 10) — a lens prior the cross-method flag and matched null will expose.
- Qwen's responses name: United States (3 items), Russia (4 Ukraine / Belarus items, mostly via
  "Soviet"), Argentina (a Paraguay item), Ukraine (the Belarus item).
- Expected judge load: ≈ 4.9k units, ≈ 7–8k per-sample calls (s3d ≈ 1.5 samples with a hit per
  unit) ≈ $45.

**Regions** (UN M49 subregions merged to 12; used for the pool and the matched null (iii)):
East Asia; South-East Asia; South Asia; Central + West Asia; Eastern Europe (incl. Russia,
Ukraine, Belarus, Poland, Czechia, Slovakia, Hungary, Romania, Bulgaria, Moldova); Northern
Europe (UK and its nations, Ireland, Nordics, Baltics); Western + Southern Europe; North Africa;
Sub-Saharan Africa; Northern America (US, Canada); Latin America + Caribbean; Oceania. The table
lives in `um_country_concepts.json` (`regions`), built from M49 by code, tested for coverage of
all 200 concepts.

**Pool rule, made concrete:** for each gold in alphabetical order, add the two most-mentioned
same-region countries not already in C (counts from this plan's `mentions.jsonl`, decoded
J-lens tokens; the pilot above is only indicative); if fewer than two exist, take from the
adjacent regions in this fixed order. Ties by name.

| region | adjacent (in order) |
|---|---|
| Northern America | Latin America + Caribbean, Northern Europe |
| Latin America + Caribbean | Northern America, Western + Southern Europe |
| Northern Europe | Western + Southern Europe, Eastern Europe |
| Western + Southern Europe | Northern Europe, Eastern Europe |
| Eastern Europe | Central + West Asia, Northern Europe |
| East Asia | South-East Asia, Central + West Asia |
| South-East Asia | East Asia, Oceania |
| South Asia | Central + West Asia, South-East Asia |
| Central + West Asia | Eastern Europe, South Asia |
| North Africa | Central + West Asia, Sub-Saharan Africa |
| Sub-Saharan Africa | North Africa, South Asia |
| Oceania | South-East Asia, Northern America |
 The pool is computed by `concepts` after the judge,
written, and never changed after Phase A starts. (This is the rule of §4a, restated.)
