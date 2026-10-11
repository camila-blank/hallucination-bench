# Plan 0002: claim map (taxonomy v2), an exhaustive, mutually exclusive categorisation of readout content

Issue: #3. Author: Camila Blank (planned with Claude), 2026-10-11. Revision 1.
Builds on plan 0001 (PR #2), which ships the v5c span + verify port, the taxonomy-v1 classifier
and the first full run. Plan 0001's cache layout, client, calibration machinery and report are
reused unchanged unless this plan says otherwise.

## Goal

Every piece of every readout lands in **exactly one leaf** of a fixed tree. The tree is
mutually exclusive and collectively exhaustive (MECE) over the readout's text, as granular as
we can label reliably. Camila specified it on 2026-10-11 (decisions are quoted in §Decisions).
The headline separates:

- **hallucinations**: wrong details and false claims;
- **plausible co-activation**: substitutions of a related object, which the model may really be
  representing;
- **correct content**;
- **thinky unverifiable content**: unrealised predictions, the model's own state or plans, and
  possible model beliefs;
- **junk**.

## The tree (taxonomy v2)

A readout is partitioned into **segments**: contiguous verbatim substrings, in order, that
together cover all of its non-whitespace text. The rule for splitting is §Segmentation. Each
segment gets exactly one leaf.

```
segment
├─ non_claim
│   ├─ generic        generic description with nothing concrete ("a friendly reply", "a narrative about a character")
│   ├─ filler         hedges, connectives, repetition, restating the prompt verbatim, list scaffolding
│   ├─ formatting     markup, bullets, headings with no content, emoji, broken-off fragments that assert nothing
│   ├─ junk           garbled or broken tokens, stray code, byte artifacts
│   └─ lens_meta      statements about the lens output itself, not the conversation (token-bag summaries:
│                     "the tokens include Russian 'холод' (cold)", "emoji byte artifacts suggest…")
└─ claim
    ├─ correct
    │   ├─ restates_prefix        stated in the response before the marker (or the prompt)
    │   ├─ realised_prediction    about what comes next, and the response after the marker does say it
    │   ├─ entailed_inference     not stated, but strictly entailed by the conversation
    │   └─ world_true             on-topic fact true in the world, not stated in the conversation
    ├─ incorrect
    │   ├─ detail                 a wrong specific with no concept of its own: an arbitrary name, number, date,
    │   │                         count or wording ("Steve" for the story's Joe; "October 24" for October 29;
    │   │                         a misquoted heading)
    │   ├─ false_claim            a false assertion about the right object: a reversed outcome or stance, the wrong
    │   │                         agent (who did what to whom), a false relation or cause, or an invented event
    │   │                         or fact asserted about the conversation ("the response recommends therapy"
    │   │                         when it does not; "the user is a teacher" when nothing says so)
    │   ├─ substitution_sibling   a different object of the same class in place of the right one
    │   │                         (Germany for France; sunrise for sunset; Picasso for da Vinci)
    │   └─ substitution_category  a different object of a different category (Nazis for the Mafia)
    └─ unverifiable
        ├─ on_topic
        │   ├─ unrealised_prediction  framed as anticipation of what comes next, plausible from the text before
        │   │                         the marker, and the response does not go on to say it
        │   ├─ model_state            about the model's own state, intent, plans or awareness
        │   │                         (e.g. evaluation awareness, "the model is deciding whether to refuse")
        │   ├─ possible_belief        a world claim that is false (or doubtful) but that the conversation neither
        │   │                         states nor contradicts; the model may really hold it
        │   └─ other_on_topic         any other on-topic claim the conversation and world knowledge cannot settle
        ├─ ambiguous_topic            could plausibly be on or off topic
        └─ off_topic                  coherent, but unrelated to this conversation
```

### Group roll-ups (the headline categories)

| group | leaves |
|---|---|
| hallucination | `detail`, `false_claim` |
| co-activation | `substitution_sibling`, `substitution_category` |
| correct | all four `correct.*` leaves |
| thinky | `unrealised_prediction`, `model_state`, `possible_belief`, `other_on_topic` |
| unsettled | `ambiguous_topic` |
| junk | `off_topic` and every `non_claim.*` leaf |

### Tags (orthogonal; they do not affect the leaf)

- `hedged` (bool): the claim is a short hedged list or disjunction ("music, theater or dance").
  The list counts as **correct** if the realised or stated option is in it; it is still one
  segment. A long scattershot list is `non_claim.generic`.
- `established` (bool, set in code): the segment overlaps a v5c verified wrong span (§Inputs).
- `dispute` (bool, judge): the judge believes an established span is not actually wrong (see
  the rules for established spans).

## Decisions (Camila, 2026-10-11) and how the rules encode them

1. **"Invented event is a false claim. It's verifiable."** A concrete event, fact or property
   **asserted** about the conversation (the user, the response, the scenario) that the
   conversation does not support is `incorrect.false_claim`. This holds even when nothing in
   the text explicitly rules it out, because the conversation is the ground truth for claims
   about itself.
2. **"Unrealized predictions are unverifiable on topic; they are thinky stuff."** The boundary
   with decision 1 is **framing**:
   - asserted as fact → decision 1;
   - framed as anticipation of the continuation ("likely continues with", "next it will",
     "expects") and plausible from the text before the marker → `unrealised_prediction`;
   - a prediction that the text before the marker rules out → `incorrect` (subtype by the
     usual rules).
3. **World-checkable claims.**
   - On topic and true in the world → `correct.world_true`.
   - On topic and false in the world, and contradicted by the conversation (before or after
     the marker) → `incorrect.*`.
   - On topic and false in the world, with the conversation silent → `unverifiable.possible_belief`.
     Example: the wrong-bridge multihop eval, where the model really holds the wrong bridge
     entity and O-lens surfaces it.
4. **Substitutions are not hallucinations.** "These could plausibly be things the model is
   really thinking about": if the model is thinking about countries when it thinks about
   France, it may think about Germany too. They are reported separately as co-activation, with
   the (said, actual) pair logged so they can be probed later (§Outputs). Sibling and category
   are two leaves, not an axis.
5. **Detail = no concept representation of its own**: arbitrary names, numbers, dates, counts and
   wording. A wrong number is `detail` even when it flips the claim. The flip is not recorded
   separately, since reversals of *meaning* are `false_claim` only when the readout asserts the
   reversed proposition in words.
6. **Hedged lists are fine** (see the `hedged` tag).

Precedence when two leaves seem to apply (stated in the prompt, in this order):

1. `non_claim` vs `claim`: a segment is a claim if it commits to anything concrete about the
   conversation or the world.
2. **Established v5c spans** stay in `incorrect` (see the rules for established spans).
3. Among the incorrect leaves:
   - **detail vs substitution**: a wrong *named* object that has its own identity (a real person,
     work, place, organisation, or another entity in the text) is a substitution. An arbitrary
     or invented label for the right referent is a detail.
   - **substitution vs false_claim**: substitution swaps the object; false_claim says something
     false about the right object.
4. `unverifiable.on_topic` subtypes:
   - `unrealised_prediction` beats `model_state`, which beats `possible_belief`, which beats
     `other_on_topic`;
   - when a statement about the model's intent is also a prediction of the text ("the model is
     about to apologise"), it is `unrealised_prediction` if not realised, and
     `realised_prediction` if realised.

## Segmentation

- **Coverage.** A segment is a verbatim substring of its readout. The segments of a readout are
  listed in order and must not overlap. Together they must cover the readout up to whitespace
  and punctuation: code checks that ≥ 95% of the readout's non-whitespace characters are
  covered.
- **Atomic claims.** One segment carries one claim. A sentence with a correct part and a wrong
  part is split at the smallest boundary that separates them. A run of non-claim text between
  claims becomes its own non-claim segment.
- **Granularity floor.** No segment shorter than 3 characters unless it is the whole readout.
  Adjacent segments with the same leaf may be merged by the judge.
- **Long readouts.** NLA readouts reach about 1.5k characters × 3. A cell's call carries all of its
  readouts, the same as v5c; `max_tokens` is sized for the worst case measured in the smoke
  run, plus 50%.

## Inputs and the established spans

The v2 judge (`claimmap-v2`) runs **one call per cell** on every non-empty cell, whatever its
readouts' v5c class: generic readouts are segmented too, which is what makes the map
exhaustive. Token arms judge the summariser's interpretation, as in plan 0001.

It sees:
- the prompt and the marked response (same `⟦READ⟧` convention);
- the readouts;
- the **v5c verified wrong spans** for each readout, as "established: a separate audit verified
  these contradict the conversation".

v5c is the calibrated instrument and remains the headline of plan 0001, so the v2 judge may not
silently drop its findings. Each established span is covered by one or more segments, and
every segment that overlaps it must get an `incorrect.*` leaf, unless the judge sets
`dispute = true` with a reason. The span then stays `incorrect` for scoring and is counted in
`n_disputed_established`.

Verify (plan 0001 stage 3) outputs are **not** given to the v2 judge. They are kept for a
comparison: verify `true` vs v2 `correct`, and verify `unverifiable` vs v2 `unverifiable` ∪
`false_claim`, reported as an agreement table.

## Output schema and parsing

`{readouts: [{idx, segments: [{text, leaf, hedged, dispute, pair|null, why}]}], rationale}`

- `leaf` is one of the 20 leaf names (enum in the schema).
- `pair` is required for substitutions: `{said, actual}`, where `actual` is quoted from the
  conversation or named.
- Code checks (all fail closed):
  - **Verbatim:** every `text` must be verbatim in its own readout (plan 0001 `verify_quote`
    normalisation). Segments are matched left to right; an unmatched segment is dropped and
    counted.
  - **Order and overlap:** overlap is resolved by truncating the later segment; an
    out-of-order segment is counted.
  - **Coverage:** uncovered non-whitespace text becomes an `uncovered` pseudo-leaf (its own
    bucket in the junk group's health line, never silently junk) and its share is reported.
    A readout below 95% coverage counts as a coverage failure.
  - **Established spans:** an established span not covered by an `incorrect` segment, and not
    disputed, is forced to `incorrect` with `leaf = unknown_incorrect` and counted.
  - **Pairs:** a substitution missing its `pair` keeps its leaf and is counted.
- **Re-queues.** An invalid answer (schema failure, a wrong readout count) is cached as failed
  and re-queued on the next run. More than 2 coverage failures in a cell also re-queues it,
  once; if it still fails, it is accepted with the `uncovered` share.
- Prompt version `claimmap-v2`. Cache key `…:M`, fingerprint as in plan 0001.

## Metrics

All per arm, per layer, and overall, with item-bootstrap 95% CIs (2000 draws, seed 0). Two
units:

1. **Claim-level shares** (the MECE view): the share of all segments, and of claim segments,
   in each leaf and each group. Shares are weighted equally per segment.
   - The same shares weighted by characters are reported too, since segment counts depend on
     granularity.
   - **Identity, tested:** the leaf shares sum to 1, and the group shares sum to 1.
2. **Readout-level rates**, over non-empty readouts:
   - share with ≥ 1 hallucination segment (`hallucination_rate_v2`);
   - share with ≥ 1 co-activation segment;
   - share with ≥ 1 thinky segment;
   - share that is all junk;
   - share with ≥ 1 correct segment;
   - `false_claim_rate` and `detail_rate`.

Comparisons:
- v5c `hallucination_rate` (plan 0001) next to `hallucination_rate_v2`;
- a transition table: v5c class (hallucinated / consistent / off_topic / generic) × v2 readout
  group;
- the verify vs v2 agreement table.

Health:
- `n_failed`, `coverage_failure_rate`, `uncovered_char_share`;
- `n_unmatched_segments`, `n_forced_established`, `n_disputed_established`,
  `n_missing_pairs`;
- spend.

**Token arms.** As in plan 0001, they are reported in a separate table, because the summariser
introduces `lens_meta` and specificity of its own.

## Calibration (gate before the full run)

The same machinery as plan 0001 (dev for iteration, a frozen held-out set for the gate, 2
repeats with fresh calls, Wilson CIs, at most 3 held-out checks), with a new case format: gold
is a **segmentation with leaves**.

- **Scoring by character alignment.** Each gold segment is matched to the predicted segment
  with the largest character overlap. Accuracy is measured over gold *claim* segments, plus a
  separate non-claim vs claim F1 at character level.
- **dev** (written by the implementer): ≥ 60 gold segments covering all 20 leaves at least
  twice. It must include:
  - the decision-anchor examples (invented event asserted vs predicted; world-true; possible
    belief; sibling vs category substitution; detail vs sibling for names; hedged list hit and
    miss);
  - `lens_meta` and summariser-style readouts;
  - established spans, including one that should be disputed.
- **held-out:**
  - (a) ≥ 60 synthetic gold segments, written and labelled by a fresh-context subagent that
    sees only this plan's §The tree, §Decisions and §Segmentation;
  - (b) ≥ 60 real segments from stored readouts. To build them, sample cells stratified by arm
    (s3d-rl600, nla-rl-L42, jlens, sae-l42, patchscopes) and by v5c class. The main session
    hand-labels the full segmentation of those cells blind to classifier output and freezes it
    before the first check.
- **Gate (held-out, mean of 2 runs):**

  | measure | threshold |
  |---|---|
  | group accuracy (6 groups) | ≥ 85% |
  | leaf accuracy (20 leaves) | ≥ 70% |
  | hallucination-vs-co-activation-vs-correct three-way accuracy on gold incorrect + correct segments | ≥ 80% |
  | recall of `false_claim` | ≥ 75% |
  | recall of `detail` | ≥ 75% |
  | recall of substitution (either leaf) | ≥ 75% |
  | claim/non-claim character F1 | ≥ 0.85 |
  | group flip rate between runs | ≤ 7% |
  | real-part group accuracy | ≥ 80% |
  | coverage failures | ≤ 2% of readouts |

## Run

1. Smoke: 20 cells × 3 arms. It prints prompts and checks parsing, coverage and the
   established-span rule, and measures the cost per cell.
2. Calibration (dev, then held-out).
3. Full run of `claimmap-v2` on **all** plan-0001 cells (all 11 arms, all layers, about 48k
   cells). Span results come from the plan-0001 cache, so v5c is not re-run.
   - The span stage must be complete for a cell before its `:M` call. Cells whose span stage
     failed are retried first.
   - One process with one pacer, in tmux, resumable, the same as `run_all_arms.sh`; there is a
     new `--stages` option.
4. `hbench report --view v2` writes:
   - the leaf/group tables, prose and token arms separately;
   - `substitution_pairs.jsonl` (arm, layer, item, pos, said, actual, subtype, readout text),
     the input for later probing;
   - `audit_samples_v2.md` (10 random segments per leaf with context).

## Deliverables

- `taxonomy_v2.py`: the tree, definitions, decisions and precedence as the single source of
  truth, rendered into the prompt. Tests assert that the prompt contains every leaf definition.
- `judges/claimmap_v2.py`: prompt, schema, parsing and the code checks above.
- Pipeline stage `:M`, metrics, report view, calibrate support for segmentation gold.
- Tests:
  - segment matching, coverage, overlap and order;
  - the established-span rule, including forced and disputed spans;
  - the leaf/group identities;
  - bootstrap;
  - character-alignment scoring;
  - the fresh-call repeats.
- `docs/results/0002_claim_map.md` (with calibration numbers, spend and caveats); README and
  CLAUDE.md updated.
- `uv run pytest`, `ruff check` and `ruff format --check` green. The gate is passed. Failed
  `:M` cells are < 1% per arm.
