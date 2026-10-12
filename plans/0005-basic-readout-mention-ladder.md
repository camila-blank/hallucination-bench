# Plan 0005: basic-readout mention ladder (commitment → probe → cross-method → hallucination)

Issue #5, PR #6. Author: Camila Blank (planned with Claude), 2026-10-12. Revision 2 (after critique 1).

This supersedes plan 0004's verdict logic for basic_readout. Set A (user modelling) follows later
with the same ladder. It reuses plan 0004's readouts, judge plumbing and viewer, and the probe
method of the Ali case study (`gw-ali-probe/scripts/experiments/ali_probe/`,
`docs/project/experiments/ali_probe/README.md`).

## Revision 2 (critique 1). This section is final and overrides the body.

1. **Every mention goes to the judge, J-lens token hits included.** The judge sees the token bag
   plus the summary. The rule that a token-only hit is never committed is dropped.
2. **Unit and roll-up.** The labelled unit is (method, item, layer, pos, concept). It is committed
   if any of its samples is committed. Per-sample counts and quotes are kept.
3. **Cross-method support** (Camila: another method "agrees and says swimming"):
   - Another method's mention of the concept for the same item counts unless the judge marked it
     a negation or a quiz distractor.
   - **Primary rule:** that mention lies within ±1 grid step (4 layers) of the unit's layer.
     NLA-RL L42 counts as layer 42, so it matches units at 40 or 44, and an NLA-RL unit matches
     other methods at 40 or 44.
   - **Sensitivity row:** any layer.
   - Label counts are reported **with and without** allowances that the lens prior flags (the
     agreeing method names the concept for ≥ 50% of the other same-category items).
4. **The probe fires** for concept S on item i at the **unit's own layer** L (L42 for NLA-RL) if
   all of the following hold:
   - (a) the validity gate passes: held-out AUROC ≥ 0.85 for S vs other classes at L, from
     GroupKFold scores;
   - (b) the gold positive control passes at L (the same p-rule, gold vs other-class
     references);
   - (c) the empirical p = (1 + #{same-gold refs scoring ≥ the item}) / (n + 1) ≤ 0.05, with
     n ≥ 19 same-gold references, **in both probe types**.

   If any precondition fails, the result is `probe n/a` with the reason recorded, and the unit
   moves to the cross-method step. Firing at ±1 grid step is a sensitivity row.
   - **FPR calibration:** report the fire rate of each category's control concepts (concepts
     no method names) per layer under the same rule. Category-layers where the control fire rate
     is > 10% are flagged and their validations shown as `probe-validated (weak null)`.
5. **Bank hygiene.**
   - The 35 item entities (Ali, Jordan, Messi, Paris, the Taj Mahal, …) are excluded from
     training rows and references.
   - Report post-filter counts per class **before** fitting. Classes with fewer than 19
     same-gold references make their items' units `probe n/a (few refs)`.
   - **Capitals:** references are non-capital cities of the gold country in the item's template
     ("The city of X is the capital of"). That premise is false, so first pilot the gold control
     and report the capital-vs-reference shift. Golds whose control fails are n/a.
   - **Landmarks:** references are other landmarks in the item's template.
6. **Lexicon equivalences and substrings.**
   - Gold equivalents count as gold, not mentions: England ≈ UK ≈ Britain ≈ United Kingdom;
     Russia ≈ USSR ≈ Soviet Union; football (American, for Brady) vs soccer (for Messi /
     Ronaldo) are distinct classes, with "football" disambiguated by the item's gold; piano ≈
     keyboard; track and field ≈ athletics ≈ sprinting ≈ running.
   - Longest-match wins (field hockey ≠ hockey, table tennis ≠ tennis).
   - Names in the prompt are excluded, including the entity itself and its city.

## Camila's procedure (2026-10-12), verbatim intent

> starting with the basic readout prompts, for example a sports prompt, find all mentions of any
> sport name in the readout … give them to a judge, which should determine if they are being said
> e.g. as sample continuations, as options being considered, etc. if so, then these are not
> hallucinations because the model has not committed to them. but if the model says them as a
> substitution, e.g. "- 2: He is an Olympian who competes in the sport of swimming" then this
> becomes a candidate for exploration. we should then make a probe for e.g. swimming, and if the
> probe fires then it's safe, but if not, then we keep examining. in that case, we ask whether at
> least one of the other three methods agrees and says swimming, in which case we allow it. if
> all of these fail, then we label it a hallucination.

## Scope

- **Items:** the 35 basic_readout `entity` items:
  - athletes → sport (12);
  - capitals → country (10);
  - landmarks → country (7);
  - musicians → instrument (6).
- **Methods (four):** s3d O-lens, ML-NLA RL600 and J-lens at layers 20…60, plus NLA-RL L42. These
  are the existing readouts from plan 0004 set B, deduplicated by (id, layer, pos), read at the
  final prompt token.

## Step 1: mentions

A **mention** is any occurrence, in a method's readout cell, of a member of the item's category
that is **not the gold** and **not named in the prompt**. Examples: any sport name for an athlete
item; any country for a capital or landmark item; any instrument for a musician item.

- **Lexicons:** comprehensive per category (`data/sibling_probe/br_lexicons.json`):
  - sport: about 80 sports;
  - country: every UN member, with demonyms and common short names;
  - instrument: about 60.

  Each has English aliases, CJK forms (≥ 2 characters, with exclusions) and token aliases, and is
  generated once by Opus and then checked by code for overlaps. The country lexicon excludes
  capital-city names (a city is not a country mention).
- **Units:** one per (method, item, layer, pos, concept, sample); s3d has about 4 samples per
  cell. **J-lens:** a mention is a raw top-10 token hit or a hit in the summary; the form is judged
  on the summary. A token-only hit (summary silent) is **not committed** by definition, since a
  token bag lists candidates.
- Entity substitutions (the readout is about another person, e.g. Usain Bolt for Ali) keep a flag
  `entity_substitution`. Their sport mentions are treated like any other mention.

## Step 2: commitment judge (`brcommit-v1`)

For each mention, the judge decides whether the model **commits** to the concept as the answer or
as a fact about the prompt's subject.

- **Committed (a substitution, candidate for exploration):** the sample states the concept as
  the answer or as a fact about the subject: "He is an Olympian who competes in the sport of
  swimming", "The answer is **Swimming**", "Paris is the capital of Spain", or the answer stated
  for a substituted entity ("Usain Bolt … track and field").
  - Each readout sample is its own claim. A sample that asserts one answer is committed even if
    another sample names something else.
- **Not committed (not a hallucination):**
  - an option being considered or one item in a list ("boxing or football", "such as
    baseball");
  - a hedged guess ("possibly baseball", "most likely apples");
  - a sample continuation or example text that is not about the subject;
  - a quiz distractor ("A) Australia B) Canada");
  - a question;
  - a negation ("not swimming");
  - a meta mention.
- **Output:** `{committed: bool, form, quote (verbatim, 3–20 words), why}`. A missing or invalid
  quote fails closed to **committed**: the mention stays a candidate and gets examined, never
  silently cleared.
- **Model:** `claude-opus-5-5`, effort low, cached in its own file. The layer is not shown.
- **Calibration:**
  - Fable labels 40 dev units first, blind to the prompt text. One prompt revision is allowed.
  - A fresh subagent labels a blind gold set of 80 units with the same definitions, plus all
    Ali units as anchors.
  - Report accuracy with a Wilson CI and per-class precision / recall. Target ≥ 85%; if it is
    missed, label the output "judge (below target)".

## Step 3: probes for committed candidates (Ali method, extended)

- **Entity banks** (`data/sibling_probe/br_entity_banks.json`), built from the item's own
  template wording:
  - **sport:** reuse the Ali bank (19 sports × about 20 athletes, plus 3 control sports). Add any
    sport that is committed at least once and missing, with about 20 famous athletes each.
  - **country** (capitals and landmarks, separate templates): for every country that is gold
    or committed at least once, about 20 well-known cities (not the capital) / landmarks.
  - **instrument:** for every instrument that is gold or committed at least once, about 20
    famous players.
  - Entities are generated by Opus and filtered by Fable for obvious errors.
  - **Knowledge filter (Ali method):** keep an entity only if the model's most likely
    category word, among the category's words, names its true class.
- **Capture:** final-token residuals at the 11 lens layers plus 42, for every bank entity in its
  template and for the 35 items.
- **Probes (both, per layer):**
  - (a) multinomial logistic regression on a standardised PCA-256 of the training fold, scored by
    class logit;
  - (b) one-vs-rest difference of means.

  Both use GroupKFold(5) over entities, so the item and every reference are scored by fold
  models that saw neither.
- **"The probe fires"** for concept S on item i at layer L:
  - the item's S-score ranks **≥ 0.95** among the held-out bank entities that **share the
    item's gold** (for Ali, held-out boxers), **in both probe types**;
  - the gold's own positive control passes at that layer: the item's gold score ranks ≥ 0.95
    among held-out entities of other classes.

  If the control fails, or fewer than 10 same-gold references remain, the result is
  `probe n/a`.
- Report the control-concept null (concepts no method names) for each category, as in the Ali
  study.

## Step 4: cross-method check, then the label

| label | rule |
|---|---|
| **not committed** | step 2 says not committed |
| **probe-validated** | committed, and the probe fires at that unit's layer (NLA-RL: L42) |
| **cross-method allowed** | committed, the probe does not fire or is n/a, and at least one of the **other three** methods mentions the concept for the **same item** (any layer, any form) |
| **hallucination** | committed, the probe does not fire or is n/a, and no other method mentions it |

- **Probe n/a** is recorded (the reason is kept) and passes on to the cross-method step, so the
  procedure still terminates in a label.
- **Diagnostics** (they do not change the label):
  - **Lens prior:** for each method and concept, the share of the other same-category items
    for which that method mentions the concept. This flags cross-method allowances that are
    probably a shared generic prior; in the Ali study, O-lens and ML-NLA name swimming / track
    for 10–11 of 11 athletes.
  - **Layer-matched agreement:** whether the agreeing method mentions the concept within ±1
    grid step.

## Deliverables

1. **Per-item breakdown** (the Ali README style) for all 35 items, in
   `docs/results/0005_basic_readout_ladder.md`, with a table per category.
   - **Row** = concept × method.
   - **Columns:** layers mentioned; committed / not committed counts with an example quote;
     probe result per layer (rank, both probes); cross-method support; final label; lens prior.
   - Ali is checked against the Ali case study's verdicts, and any disagreement is explained.
2. **Figures:**
   - (a) per method, the share of mentions in each final label;
   - (b) the same by layer;
   - (c) the share of committed mentions the probe validates, per method and layer.
3. **Viewer:** update the public sibling viewer so every mention carries its ladder label,
   quote, probe ranks and cross-method support, with a filter by label.

## Cost (estimate)

- **Judge:** about 10–20k mentions × about $0.006, roughly $60–120.
- **Bank generation:** about $10.
- **Capture:** about 2k short prompts at 12 layers, 1 GPU, about 10 min.
- **Probes:** CPU.

## Out of scope (next)

- Set A (user modelling) under the same ladder.
- Implicit / processing basic_readout items.
