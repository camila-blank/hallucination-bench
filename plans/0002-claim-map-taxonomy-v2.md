# Plan 0002: claim map (taxonomy v2), an exhaustive, mutually exclusive categorisation of readout content

Issue #3. Author: Camila Blank (planned with Claude), 2026-10-11. Revision 2 (after critique 1).
This plan builds on plan 0001 (PR #2): the v5c span and verify port, the taxonomy-v1 classifier,
and the first full run. It reuses plan 0001's cache layout, client, calibration machinery,
bootstrap and report unless this plan says otherwise.

Items marked **[pending Camila]** use a default until she confirms them (§Open decisions).

## Goal

Every piece of every readout lands in **exactly one leaf** of a fixed tree. The tree is
mutually exclusive and collectively exhaustive (MECE) over the readout's text, and as granular
as we can label reliably. The headline separates seven groups:

- **hallucinations**: wrong details and false claims;
- **plausible co-activation**: substitutions of a related object, which the model may really be
  representing;
- **correct content**;
- **thinky unverifiable content**: unrealised predictions, the model's own state and beliefs, and
  possible model beliefs;
- **unsettled**: ambiguous topic;
- **junk**: off-topic claims;
- **non-claims**.

## Units: code-split units, judge-labelled segments

Code pre-splits each readout into **units**. A unit is a sentence, a bullet line or a clause
ending in `;` or `:`, and splits fall on newlines, sentence punctuation and bullet markers.
Units carry ids (`R{readout}.U{n}`) and their raw character offsets, so together they cover the
readout by construction. Whitespace-only text between units is dropped.

The judge labels each unit. When a unit carries more than one claim, or mixes claim and
non-claim text, the judge **splits** it by quoting sub-spans verbatim. Each sub-span gets its
own leaf, and any unquoted remainder of a split unit gets the judge's `rest` leaf, which is
required when it splits.

Code locates sub-spans with an offset-returning `locate(sub, unit)`. It applies the same
normalisation as plan 0001's `verify_quote` (case, quote marks, whitespace) and maps the match
back to raw offsets. If a sub-span cannot be located, the whole unit falls back to the judge's
`rest` leaf, and the miss is counted (`n_unlocated_subspans`).

A segment is a unit or a located sub-span. Segments never overlap and cover every unit, so
coverage is total by construction. Atomic-claim rule: one segment carries one claim. A sentence
with a correct part and a wrong part must be split.

## The tree (taxonomy v2): 19 judge leaves in 7 groups

```
segment
├─ non_claim                                                        group: non_claim
│   ├─ generic        generic description, nothing concrete ("a friendly reply")
│   ├─ filler         hedges, connectives, repetition, verbatim restatement of the prompt, list scaffolding,
│   │                 and the framing words of token-bag summaries ("the tokens include…", "suggesting")
│   ├─ formatting     markup, bullets, empty headings, emoji, fragments that break off asserting nothing
│   ├─ junk_text      garbled or broken tokens, stray code, byte artifacts
│   └─ lens_meta      statements about the lens output itself rather than the conversation
│                     ("Russian 'холод' (cold) and Chinese '冷的' (cold)", "emoji byte artifacts")
└─ claim
    ├─ correct                                                      group: correct
    │   ├─ restates_prefix   stated in the prompt or in the response before the marker
    │   ├─ stated_after      stated in the response after the marker (a realised prediction OR an asserted
    │   │                    fact that the continuation states)
    │   ├─ entailed          not stated anywhere, but strictly entailed by the conversation
    │   └─ world_true        on topic, true in the world, not stated or entailed by the conversation
    ├─ incorrect
    │   ├─ detail            [hallucination] a wrong specific with no concept of its own (see §Detail rule)
    │   ├─ false_claim       [hallucination] a false assertion about the right object: reversed outcome or stance,
    │   │                    wrong agent (who did what to whom), false relation or cause, a wrong real-world
    │   │                    quantity, or an invented event or fact asserted about the conversation
    │   ├─ substitution_sibling   [co-activation] a different object of the same class in place of the right one
    │   └─ substitution_category  [co-activation] a different object of a different category
    └─ unverifiable
        ├─ on_topic                                                 group: thinky
        │   ├─ unrealised_prediction  framed as anticipation of the continuation, plausible from the text before
        │   │                         the marker, and not said by the response
        │   ├─ model_state            about the model's own beliefs, perceptions, state, intent, plans or awareness,
        │   │                         including claims about the nature or purpose of the conversation (a test, an
        │   │                         evaluation, a roleplay), whatever the framing
        │   ├─ possible_belief        a world claim the judge believes is false, which the conversation neither
        │   │                         states nor contradicts; the model may really hold it
        │   └─ other_on_topic         any other on-topic claim that the conversation and world knowledge cannot
        │                             settle, including contested or doubtful world claims
        ├─ ambiguous_topic                                          group: unsettled
        └─ off_topic                                                group: junk
```

Code-assigned pseudo-leaves are kept as **separate rows in every table**, outside the seven
groups:

- `unknown_incorrect`: an established span the judge failed to cover with an incorrect leaf,
  or one it disputed;
- `failed`: the cell's call failed, so every unit in it gets this.

The identities hold over the seven groups plus these two rows.

### Precedence (in the prompt, applied top-down; the first rule that fits wins)

1. **Non-claim vs claim.** A segment is a claim if it commits to anything concrete about the
   conversation, the model, or the world.
   - Token-bag summaries: the framing ("the tokens include X") is `filler`, and a gloss of the
     tokens is `lens_meta`. The interpretation ("suggesting the passage is about Y") is a claim,
     judged as usual.
2. **Established v5c spans** must be `incorrect` (§Inputs).
3. **Claims about the model's own beliefs, perceptions or the conversation's nature** →
   `model_state`. Evaluation awareness lands here.
4. **A claim about this conversation is never `off_topic`** (v5c's rule). If it is false, it is
   incorrect.
5. **Conversation vs world.** The conversation wins for claims about the conversation and its
   content. **[pending Camila, default:]** world-false content that the response itself states
   is `correct.restates_prefix` or `correct.stated_after`, with the tag `world_false = true`
   (the wrong-bridge case: the response uses the wrong bridge, and the readout finds it).
6. **Inside `correct`:** `restates_prefix` > `stated_after` > `entailed` > `world_true`.
7. **Inside `incorrect`:**
   - A wrong named object that has its own identity, in place of the right one →
     substitution (sibling or category). **[pending Camila]**: see §Open decisions 1 for named
     entities that are the subject of the claim.
   - A false assertion about the right object → `false_claim`.
   - A wrong arbitrary specific → `detail`.
8. **Inside `unverifiable.on_topic`:** `unrealised_prediction` > `model_state` >
   `possible_belief` > `other_on_topic`. A statement of intent that also predicts text ("about
   to apologise") is a prediction: `stated_after` if it is realised, `unrealised_prediction` if
   not.
9. **Framing.** A concrete event or fact **asserted** about the conversation that the
   conversation does not support is `false_claim` (Camila: "the invented event is a false claim.
   it's verifiable"). If it is **framed as anticipation** and plausible, and the response does
   not say it, it is `unrealised_prediction` (Camila: "unrealized predictions are unverifiable
   on topic, they are thinky stuff"). A prediction that the text before the marker rules out is
   incorrect.

### Detail rule [pending Camila: §Open decisions 2]

`detail` covers wrong specifics with no concept representation of their own (Camila: "things
that likely don't have their own concept representation, e.g. names"):
- arbitrary or fictional names;
- incidental numbers, dates or counts in fiction or scaffolding;
- quoted wording or heading text.

Quantities and dates of **real events**, and counts that **flip the claim** ("0 errors" for 3),
are `false_claim`. Wording and digits are treated alike, so "no errors" and "0 errors" both
count as `false_claim`.

### Tags (orthogonal; they do not affect the leaf)

- `hedged` (bool): a short hedged list or disjunction of at most 4 options. It counts as correct
  if the realised or stated option is in it, and it is one segment. A longer scattershot list of
  more than 4 unrelated options is `generic`.
- `world_false` (bool): see precedence rule 5.
- `cruxy` (bool, incorrect segments only) **[pending Camila: §Open decisions 3]**: the segment
  misstates the central subject, claim or answer. Camila's "important/cruxy concepts" are
  `false_claim ∧ cruxy`.
- `established`, `dispute` (bool): see §Inputs.
- `pair` (substitutions only): `{said, actual}`, logged for later probing.

## Decisions (Camila, 2026-10-11, verbatim) and where they are encoded

- "same class sibling and different category substitution are different from factually false
  claims so there should be two categories not two full axes" → two substitution leaves,
  separate from `false_claim`.
- "unrealized predictions are unverifiable on topic, they are thinky stuff" → precedence 9 and
  `unrealised_prediction`.
- "for substitution i think these could plausibly be things the model is really thinking
  about…" → the substitutions form the co-activation group, which is not a hallucination, with
  `pair` logged.
- "the invented event is a false claim. it's verifiable." → precedence 9.
- "world checkable claims that are on topic and correct should be verifiable correct, claims
  that are on topic and incorrect are either unverifiable because we are not sure whether the
  model is actually thinking about this or not or maybe they are just verifiably incorrect
  because they contradict with the following text" → these map as follows:
  - on topic and world-true → `world_true`;
  - world-false and contradicted by the conversation → incorrect;
  - world-false and the conversation silent → `possible_belief`;
  - world-false and the conversation states it → precedence 5 [pending].
- "hedged lists are fine" → the `hedged` tag.
- Original wish list: "eval awareness" as thinky content → precedence 3 and `model_state`.

## Open decisions (defaults are used until Camila answers)

1. **Picasso for da Vinci.** Her first example called this a concept hallucination. Under the
   substitution rule it is `substitution_sibling`. The default is substitution, because her
   later reasoning (co-activation) applies.
2. **Numbers.** The default is the §Detail rule.
3. **Cruxy.** The default is a `cruxy` tag, with cruxy false claims reported as their own row.
4. **The wrong-bridge direction.** The default is precedence 5 (the conversation wins; tag
   `world_false`).

## Inputs and established spans

`claimmap-v2` runs **one call per non-empty cell**, whatever the v5c class of its readouts.
Token arms judge the summariser's interpretation.

The judge sees:

- the prompt and the marked response;
- the units, each with its id;
- the **v5c verified wrong spans** for each readout, marked as established.

Every unit or sub-span that overlaps an established span must get an `incorrect.*` leaf, unless
the judge sets `dispute = true` and gives an alternative leaf. In code:

- **Disputed:** the segment is scored as `unknown_incorrect`, the alternative is logged, and a
  sensitivity row in the report re-scores with the alternatives.
- **Non-incorrect leaf without a dispute:** the segment is forced to `unknown_incorrect` and
  counted.

Verify outputs (plan 0001 stage 3) are **not** shown to the judge. They are used only in the
agreement table, mapped as follows:

| verify | v2 |
|---|---|
| `true` | `restates_prefix` ∪ `stated_after` ∪ `entailed` |
| `unverifiable` | `world_true` ∪ thinky ∪ `false_claim` |
| `disputed` | incorrect ∪ `possible_belief` |
| `off_topic` | `off_topic` ∪ non_claim |

## Output schema and parsing

The schema is
`{units: [{id, leaf, hedged, world_false, cruxy, dispute, alt_leaf|null, pair|null, why,
splits: [{text, leaf, hedged, world_false, cruxy, dispute, alt_leaf|null, pair|null, why}],
rest_leaf|null}], rationale}`.

- `leaf` is a schema enum of the 19 judge leaves. When `splits` is non-empty, `rest_leaf` is
  required and the unit's own `leaf` is ignored.
- Every unit id must appear exactly once.
  - A missing unit gets `unknown`, which counts as `failed` for that unit, and is counted.
  - A duplicate keeps its first entry and is counted.
  - A wrong readout count, or more than 10% missing units, makes the answer invalid. It is
    cached as failed and re-queued on every run, as in plan 0001.
- A substitution with no `pair` is counted, and its leaf is kept.
- Prompt version `claimmap-v2`. Cache key `{key}:M`. The fingerprint is the same as plan 0001's.
- `STAGES` and the stage-of-key mapping gain `:M`. `hbench run --stages span,verify,classify,map`
  selects which stages run; the default is all of them.

## Metrics

Per arm, per layer and overall, each with an item-bootstrap 95% CI (2000 draws, seed 0).

1. **Headline: character-weighted shares.** The share of readout characters in each group and
   each leaf. Each readout is weighted equally: per-readout shares are computed first, then
   averaged over non-empty readouts. The segment-count shares, pooled over segments, are also
   reported.
   - Identity (tested): the shares of the 7 groups plus `unknown_incorrect` plus `failed` sum
     to 1, and so do the leaf shares.
2. **Readout-level rates.** Each rate is computed over two denominators: all non-empty readouts,
   and v5c-specific readouts (`n_specific`, which matches plan 0001's `hallucination_rate`).
   The rates are the share of readouts with at least one segment of:
   - hallucination (`hallucination_rate_v2`);
   - `false_claim`;
   - cruxy `false_claim`;
   - `detail`;
   - co-activation;
   - thinky;
   - correct.

   Also reported: the share of readouts that are entirely non-claim or junk.
3. **Comparisons.**
   - v5c `hallucination_rate` next to `hallucination_rate_v2` on the `n_specific` denominator;
   - a transition table of v5c class × v2 readout group;
   - the verify-vs-v2 agreement table.
4. **Health.**
   - failed cells;
   - `n_missing_units`, `n_duplicate_units`, `n_unlocated_subspans`;
   - `n_forced_established`, `n_disputed_established`;
   - `n_missing_pairs`;
   - spend.

Token arms get their own table, as in plan 0001.

## Calibration (gate before the full run)

Plan 0001's machinery carries over: a dev split, a frozen held-out split, 2 repeats with fresh
calls, Wilson CIs, and at most 3 held-out checks.

**Gold format.** The cases use the production unit split. Gold is a leaf per unit, plus
sub-spans with leaves for units that need splitting. Units are shared between gold and the
prediction, so matching is **one-to-one by unit**. For a unit that is split in gold or in the
prediction, scoring is at character level within the unit, where each character carries the
leaf of its segment.

- **Measures.** Unit-level and character-level leaf accuracy and group accuracy.
- **dev** (written by the implementer): at least 100 gold units, at least 3 per leaf. It must
  include:
  - every decision anchor (asserted vs predicted invented event; world-true; possible_belief;
    sibling vs category; detail vs substitution for names; numbers under the detail rule;
    hedged hit and miss; eval-awareness `model_state`; token-summary framing vs interpretation;
    a claim about the conversation that looks off-topic);
  - established spans, including one that should be disputed.
- **held-out:**
  - (a) at least 120 synthetic gold units, written and labelled by a fresh-context subagent
    that sees only §Units, §The tree, §Precedence, §Detail rule and §Tags;
  - (b) at least 120 real units from stored readouts. Cells are stratified by arm (s3d-rl600,
    nla-rl-L42, jlens, sae-l42, patchscopes) and by v5c class. The main session hand-labels
    them blind to classifier output and freezes them before the first check.
  - Across (a) and (b), at least **20 gold units each** for `false_claim`, `detail`,
    substitution (sibling and category together, at least 8 of each) and `correct`, and at least
    20 thinky units. Sample real cells until these minimums are met.
  - Camila spot-checks the decision anchors and 30 random real-part gold labels before the
    held-out set is frozen.
- **Gate** (held-out, mean of the 2 runs):

  | measure | threshold |
  |---|---|
  | group accuracy (7 groups), unit level | ≥ 85% |
  | group accuracy, character level | ≥ 85% |
  | leaf accuracy (19 leaves), unit level | ≥ 70% |
  | recall of `false_claim`, `detail`, substitution and correct | each ≥ 75% |
  | sibling-vs-category accuracy on gold substitutions | ≥ 75% |
  | claim/non-claim character F1 | ≥ 0.85 |
  | group flip rate between runs | ≤ 7% |
  | real-part group accuracy | ≥ 80% |
  | `n_unlocated_subspans` | ≤ 2% of splits |

## Run

1. Smoke: 20 cells × 3 arms. It prints prompts and checks the unit split, parsing and the
   established-span rule. It also measures cost and the worst-case output length (to size
   `max_tokens` at +50%).
2. Calibration: dev, then held-out (requires the §Open decisions answered or the defaults
   accepted).
3. A full `:M` run over all plan-0001 cells: 11 arms, all layers, about 48k cells. The v5c span
   results come from the cache, and cells whose span stage failed are retried first. It runs as
   one process with one pacer in tmux, and is resumable.
4. `hbench report --view v2` writes:
   - the group and leaf tables, with prose and token arms separate;
   - `substitution_pairs.jsonl`;
   - `audit_samples_v2.md`, with 10 random segments per leaf in context.

## Deliverables

- **Code:**
  - `taxonomy_v2.py`: the leaves, groups, definitions, precedence and decisions. It is the
    single source of truth and is rendered into the prompt; tests check the prompt contains
    every leaf definition.
  - `units.py`: the deterministic splitter and `locate`.
  - `judges/claimmap_v2.py`.
  - the `:M` stage and the `--stages` option.
  - the v2 metrics and report view, and calibrate support for unit gold.
- **Tests:**
  - the splitter (it covers all non-whitespace text and is stable);
  - `locate` offsets;
  - split, rest and missing-unit handling;
  - the forced and disputed established-span rules;
  - the group and leaf identities;
  - the character-weighted shares;
  - unit-level scoring;
  - fresh-call repeats.
- **Docs:** `docs/results/0002_claim_map.md` (calibration numbers, spend, caveats), and updates
  to the README and CLAUDE.md.
- **Acceptance:** pytest, `ruff check` and `ruff format --check` are green; the gate passes;
  fewer than 1% of `:M` cells fail per arm.
