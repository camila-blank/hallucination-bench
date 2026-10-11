# Calibration cases for the taxonomy classifier

## Case schema (JSONL, one case per line)

Both `dev.jsonl` and `heldout.jsonl` use this schema. `hbench calibrate` validates every case
(`calibrate.validate_case`) and refuses to run on an invalid file.

```json
{
  "case_id": "dev-01-mona-lisa",
  "part": "synthetic",
  "rule": "name-rule1-subject",
  "prompt": "Who painted the Mona Lisa and when?",
  "response": "The Mona Lisa was painted by Leonardo da Vinci,⟦READ⟧ who worked on it ...",
  "readouts": ["<explanation>\nArt-history Q&A. The response attributes the Mona Lisa to Picasso, ...\n</explanation>"],
  "spans": [
    {
      "readout_idx": 0,
      "text": "attributes the Mona Lisa to Picasso",
      "grounding": "contradicted",
      "gold": {"type": "concept", "target": "entity", "severity": 3, "unsupported_kind": null}
    }
  ]
}
```

| field | type | rule |
|---|---|---|
| `case_id` | string | unique within the file |
| `part` | `"synthetic"` \| `"real"` | `real` = a span sampled from a pilot run (`hbench pilot-dump`); default `synthetic` |
| `rule` | string \| null | optional tag naming the taxonomy rule the case exercises (see below) |
| `prompt` | string | the user prompt the model answered |
| `response` | string | the model's full response with the marker `⟦READ⟧` inserted **exactly once** where the lens read |
| `readouts` | list of 1–3 strings | the lens readouts the classifier sees (production shows only specific-class readouts) |
| `spans` | non-empty list | the spans to label, in order; they become `S1..Sn` in the prompt |
| `spans[].readout_idx` | int | index into `readouts` |
| `spans[].text` | string | a **verbatim substring** of `readouts[readout_idx]` (checked with plain `in`) |
| `spans[].grounding` | `contradicted` \| `disputed` \| `unsupported` | set by the pipeline in production, never by the classifier |
| `spans[].gold.type` | `concept` \| `detail` | |
| `spans[].gold.target` | `entity` \| `quantity` \| `event` \| `attribute` \| `stance` \| `structure` \| `topic` | exactly one |
| `spans[].gold.severity` | `1` \| `2` \| `3` | 1 minor, 2 moderate, 3 major |
| `spans[].gold.unsupported_kind` | `prediction` \| `inference` \| `fabricated` \| `world_knowledge` \| `null` | required for `unsupported` spans, `null` for `contradicted` / `disputed` |

Optional extra fields are ignored by the scorer: `source` (`{arm, key, readout_idx}` for real
spans, written by `pilot-dump`) and `spans[].sampled` (which spans the pilot dump drew).

The label definitions and rules live in `src/hallucination_bench/taxonomy.py` (the single source
of truth, rendered into the classifier prompt). Label from that file, not from memory.

The case is rendered exactly as in production (`taxonomy_v1.render_prompt`): the prompt, the
marked response, every readout as `<readout idx="i">`, and the numbered span list with each
span's readout index and grounding. The classifier never sees earlier stages' `why` text.

### `rule` tags used in dev

`name-rule1-subject` (real entity as the subject of the claim → concept), `name-rule2-passing`
(real entity in a passing mention → detail), `name-rule3-arbitrary` (arbitrary label → detail),
`protagonist-renamed` (arbitrary protagonist renamed → detail), `protagonist-confused`
(protagonist confused with another character in the text → concept), `protagonist-famous`
(arbitrary protagonist swapped for a famous real name → concept), `value-flip` (a value that
flips the claim → concept), `value-noflip` (a wrong value that does not → detail), `magnitude`
(a magnitude error that stays detail; severity rises instead).

## dev.jsonl (synthetic, prompt iteration)

32 cases, 42 spans, written from scratch by the implementer (Claude, 2026-10-11) for plan 0001.
Every case is invented; no real readout or bench item is used. Readout styles mimic the stored
lenses: O-lens bullet lists with code-switched Chinese and English fragments, NLA
`<explanation>` prose, and one-to-two-sentence summariser interpretations ("The model is ...").

Coverage (asserted by `tests/test_taxonomy_judge.py::test_dev_cases_are_valid_and_meet_the_coverage_requirements`):

| requirement | dev |
|---|---|
| anchors from issue #1 | Picasso for da Vinci (`dev-01`, concept), Steve for Joe (`dev-02`, detail) |
| concept / detail spans | 22 / 20 |
| targets | all 7 (entity 9, quantity 9, event 6, stance 6, attribute 6, topic 3, structure 3) |
| rule-tagged cases | 12 (all 9 tags above) |
| unsupported spans | 14: fabricated 4, inference 4, prediction 3, world_knowledge 3; 9 concept, 5 detail |
| severity 1 / 2 / 3 | 13 / 17 / 12 |
| disputed spans | 5 |

Gold labelling notes (judgement calls, all following `taxonomy.py`):

- Unsupported claims about the conversation (an invented intent, the user's situation, a
  claimed fact the response "notes") are `concept`; invented specific values attached to
  something the text discusses (a job title, "serves two people", "324 meters tall", a name
  for the unnamed sister) are `detail`. World-knowledge spans follow the same split: an added
  claim or event is `concept` (`dev-12`, `dev-32`), a value is `detail` (`dev-13`).
- A count of list items ("introduced 3 tips") is target `quantity` (count), not `structure`.
- A wrong country in a location clause (`dev-32` "border of Nepal and India") is `detail`: the
  country is not the subject of the claim and the gist is right.
- A predicted continuation the response never makes is `prediction` even when it also invents an
  event (`dev-24` "Vaaler sued"); an invented name inside a fulfilled prediction is `fabricated`
  (`dev-27` "Joe's sister Maria").

## heldout.jsonl (gate only)

Not written by the implementer. Part (a): at least 30 synthetic spans written and gold-labelled
by a **fresh-context subagent** that never sees the classifier prompt or the dev results. Part
(b): at least 30 real spans sampled with `hbench pilot-dump` from a span + verify pilot (40 cells
each of s3d-rl600 L44, nla-rl-L42 and jlens L44, `hbench run --skip-classify`), stratified by arm
and grounding, hand-labelled blind to classifier output before the classifier prompt was written.
Minimums across both parts: at least 20 unsupported spans with at least 4 per kind, at least 10
spans per severity level, at least 4 disputed spans, and at least 8 concept and 8 detail gold
spans in the real part. Labels are committed, frozen, before the first held-out check.

**Caveat:** the gold labeller (Claude) is the same model family as the classifier, so agreement
is optimistic. A 30-span sample goes into `audit_samples.md` for Camila to spot-check.

## Check log

Every `hbench calibrate --split heldout` run is logged here (at most 3 checks; then stop and
report). Gold-label fixes are recorded here with the reason.

| date | split | prompt version | result | notes |
|---|---|---|---|---|
| 2026-10-11 | heldout (check 1/3) | taxonomy-v1 | **PASS** | type 0.978, concept rec 1.00, detail rec 0.96, sev exact 0.81 / QWK 0.85, target 0.86, ukind 0.82, flip 0.034, real-part type 0.957; 92 spans, 2 runs, $2.15 |
| 2026-10-11 | dev | taxonomy-v1 | all dev thresholds met (type acc 0.940, sev exact 0.869, QWK 0.882, target 0.869, unsupported_kind 0.786, type flip 0.024) | first dev run, no prompt edits |

Gold-label fixes: none.

## Held-out set (frozen 2026-10-11, before the first held-out check)

- Part (a): 46 synthetic spans in 16 cases (`ho-syn-*`). A fresh-context subagent wrote and
  gold-labelled them without seeing the classifier prompt, `dev.jsonl` or any results.
- Part (b): 46 real spans in 41 cases (`real-*`). They were drawn by `hbench pilot-dump
  --per-stratum 6 --seed 0` from a span + verify pilot of 80 cells each of s3d-rl600 L44,
  nla-rl-L42 and jlens L44. That draw gave 54 spans; the main session (Claude) hand-labelled
  them before seeing any classifier output on them.
  - 8 spans were dropped as genuinely ambiguous under the rules. Most were lens meta-claims about
    the tokens themselves (for example, "emoji byte artifacts" or a list of token glosses) rather
    than claims about the conversation.
  - A real case keeps its whole cell for prompt fidelity. Only the drawn spans carry gold
    (`sampled: true`); the other spans are context and are not scored.
- Caveat: the real-part gold labeller is the same model family as the classifier, so agreement
  on part (b) is optimistic.

---

# Claim-map (taxonomy v2) calibration cases (plan 0002)

Files: `dev_v2.jsonl` (implementer), `heldout_v2.jsonl` (blind author only; never written or
edited by the implementer). `hbench calibrate --taxonomy v2 --split dev|heldout` validates every
case (`calibrate_v2.validate_case`) and refuses to run on an invalid file. The leaf definitions
and rules live in `src/hallucination_bench/taxonomy_v2.py` (rendered into the judge prompt);
label from that file, plan 0002 §Units, §The tree, §Precedence, §Detail rule and §Tags.

## Workflow (gold uses the production unit split)

1. Write the case without gold: `case_id`, `part`, `rules`, `kind`, `prompt`, `response` (with
   `⟦READ⟧` exactly once), `readouts`, `established`.
2. Run the production splitter over it:
   `uv run hbench units-v2 --cases draft.jsonl --dest skeleton.jsonl`. This fills
   `gold.units` with one entry per production unit (`id`, `text`, `established`, `leaf: null`).
   Real cells come pre-split from `hbench pilot-dump-v2` (stratified by arm × v5c cell class;
   it never shows claim-map output, so the labeller stays blind).
3. Label exactly those units: set `leaf`, or, for a unit that carries more than one claim or
   mixes claim and non-claim text, give `splits` (verbatim, in-order, non-overlapping sub-spans
   of the unit text, each with a leaf) and `rest_leaf` (the leaf of the unquoted remainder;
   required whenever the splits leave any non-whitespace text). Never edit `id` / `text`: the
   loader re-runs the splitter and rejects a case whose units no longer match.

## Schema (JSONL, one case per line)

<!-- v2-example -->
```json
{"case_id": "dev2-xx-example", "part": "synthetic", "rules": ["sibling-subject-name", "split"], "kind": "prose", "prompt": "Who painted the Mona Lisa?", "response": "The Mona Lisa was painted by Leonardo da Vinci⟦READ⟧ in the early 1500s.", "readouts": ["<explanation>\nThe response attributes it to Picasso.\nThe user, Sarah, asks about a painting.\n</explanation>"], "established": [["The response attributes it to Picasso."]], "gold": {"units": [{"id": "R0.U1", "text": "<explanation>", "established": false, "leaf": "formatting", "splits": [], "rest_leaf": null}, {"id": "R0.U2", "text": "The response attributes it to Picasso.", "established": true, "leaf": "substitution_sibling", "splits": [], "rest_leaf": null}, {"id": "R0.U3", "text": "The user, Sarah, asks about a painting.", "established": false, "leaf": null, "splits": [{"text": "Sarah", "leaf": "detail"}], "rest_leaf": "restates_prefix"}, {"id": "R0.U4", "text": "</explanation>", "established": false, "leaf": "formatting", "splits": [], "rest_leaf": null}]}}
```

| field | type | rule |
|---|---|---|
| `case_id` | string | unique within the file |
| `part` | `"synthetic"` \| `"real"` | `real` = a stored cell (`hbench pilot-dump-v2`); default `synthetic` |
| `rules` | list of strings | optional tags naming the anchors / rules the case exercises |
| `kind` | `"prose"` \| `"tokens"` | `tokens` = the readout is a summariser interpretation of a token bag (the prompt then says so); default `prose` |
| `prompt` | string | the user prompt |
| `response` | string | the full response with `⟦READ⟧` inserted **exactly once** |
| `readouts` | list of 1–3 strings | what the judge sees (production: every non-empty readout of the cell) |
| `established` | list (one per readout) of lists of strings | the v5c verified `wrong` spans of each readout (revoked or not); each must occur in its readout (`verify_quote` normalisation); overlapping ones are merged; each becomes its own unit. Optional (default none) |
| `source` | object | optional (`{arm, key, v5c_cell_class}` for real cells) |
| `gold.units[]` | list | **every** production unit, in order |
| `gold.units[].id`, `.text`, `.established` | | copied from the splitter (`units-v2` / `pilot-dump-v2`); never edit |
| `gold.units[].leaf` | one of the 22 leaves \| `null` | the unit's leaf when it is not split. `null` with no splits = unlabelled context (allowed on `real` cases only; not scored) |
| `gold.units[].splits[]` | `{text, leaf}` | sub-spans, verbatim from the unit, left to right, non-overlapping |
| `gold.units[].rest_leaf` | leaf \| `null` | required when `splits` leave a remainder |
| `gold.units[].probe_target` | string | optional; on a probe-candidate unit, the expected probe target (reported next to the judge's, not gated) |
| `gold.units[].dispute` | bool | optional; on an established unit whose correct leaf is **not** incorrect (the judge should dispute it). An established unit's gold must otherwise contain an incorrect leaf |

Leaves (taxonomy v2, plan rev 13): non_claim — `generic`, `filler`, `formatting`, `junk_text`,
`lens_meta`; correct — `restates_prefix`, `stated_after`, `entailed`; incorrect —
`detail`, `false_claim` (hallucination), `substitution_sibling`, `substitution_category`
(co-activation); supplementary — `supplementary_true`, `supplementary_false`,
`supplementary_unsettled`; thinky — `unrealised_option`, `unrealised_prediction`, `model_state`,
`unsupported_assertion`, `other_on_topic`;
`ambiguous_topic` (unsettled); `off_topic` (junk). Gold never uses the code-only
`unknown_incorrect` / `failed`. Pre-rev-13 names are renamed on load (`world_true` ->
`supplementary_true`, `possible_belief` -> `supplementary_false`); rev 14's
`unrealised_prediction` -> `unrealised_option` only as a fallback, with a warning (re-label it:
most unrealised predictions are incorrect).

Single-call mode (the default since rev 13): established spans still make their own units (ids
stay stable) but the flags are not shown to the judge; an established unit is scored against
its gold leaf (for a disputed one, its non-incorrect gold leaf). The gate adds verdict accuracy
(6 verdicts, unit level, ≥ 85%).

Scoring (`calibrate_v2`): units are shared, so prediction and gold match one to one by unit.
Recall rows, per-leaf recall and sibling-vs-category are per **gold segment** (the
majority-character predicted leaf over that segment's characters); accuracies and the flip rate
are per **unit** (each side's majority-character leaf); the claim/non-claim F1 and the
character accuracies are per non-whitespace character. A predicted `unknown_incorrect` is
scored by the judge's `alt_leaf` (disputed) or raw leaf (forced).

## dev_v2.jsonl (synthetic, prompt iteration)

41 cases, 177 gold units, written from scratch by the implementer (Claude, 2026-10-11) with
`scripts/build_dev_v2.py` (the labels live there; re-run it if the splitter changes). Every
case is invented. Coverage (asserted by `tests/test_calibrate_v2.py`): ≥ 3 gold segments for
every leaf (min 3, e.g. 5 sibling and 4 category substitutions, 9 false_claim, 13 detail;
segment counts skip pure-punctuation fragments), token-summary "The model is …" framing as
`filler` (rev 11, `dev2-28/29`), 8
established units (one should be disputed: `dev2-16`), and every decision anchor (asserted vs
predicted invented event `dev2-04`; world-true `dev2-05/06/22`; possible_belief `dev2-05/24`;
sibling vs category `dev2-21/22/23`; detail vs substitution for names `dev2-01/03/27`; numbers
under the detail rule `dev2-03/06/07/18/23/27`; hedged hit and miss `dev2-08`; eval awareness
`dev2-09/25`; token framing vs interpretation `dev2-11/12`; a claim that looks off-topic but is
on topic `dev2-13`; rule 12 hedged `dev2-09/24` and flat `dev2-10`; the wrong bridge
`dev2-20`). Rev 13: mechanical renames, dev2-24's "Brown eggs taste better" relabelled
`supplementary_unsettled`, plus `dev2-33` (settlement, not subject: Guernica in a Blue Period
answer; 2 unsettled) and `dev2-34` (2 unsettled; probe targets), and gold `probe_target` on
`dev2-05/23/33/34` (4 anchors). Rev 14: every `unrealised_prediction` re-labelled
(dev2-04, dev2-08 -> `unrealised_option`; dev2-07, dev2-17, dev2-34 -> `false_claim`; dev2-25
-> `detail`, ship renamed "Rusty Anchor"), plus dev2-35 (option vs guide), dev2-36 (different
answer), dev2-37 (我是 ChatGPT -> `substitution_sibling`), dev2-38 (Miss Wang for Miss Li ->
`detail`); probe targets now on dev2-05/23/33/35/37 (dev2-34's decaf is no longer one). Rev 15 (incorrect only if obviously contradicted):
uncontradicted invented content -> `unsupported_assertion` (dev2-04, 10, 27, 32); uncontradicted
unrealised moves -> `unrealised_prediction` (dev2-07, 17, 34, 35); new anchors dev2-39
(realised paraphrases), dev2-40 (paraphrase / alternative / contradicted continuation), dev2-41
(contradicted details, unsupported "sister Maria"). dev2-35's option is no longer a probe target.

Gold-labelling notes (judgement calls, following `taxonomy_v2.py` defaults):

- `dev2-20`: the response says India's capital is Mumbai; the readout's "The capital will be
  named as Mumbai" is `stated_after` (rule 6, tag `world_false`), and "The capital of India is
  New Delhi" is contradicted by the conversation, so it is incorrect, and New Delhi in place of
  Mumbai is `substitution_sibling`. This is the conversation-wins default (open decision 4).
- Bare list markers left over when an established span starts mid-line (`-`) are `formatting`;
  `- next:` style scaffolding is `filler`.
- Token-summary lens framing ("The model is representing / anticipating / gearing up to") is
  a `filler` split; what it attributes is judged as an ordinary claim or prediction, never
  `model_state` (plan rev 11, rule 1).

## heldout_v2.jsonl (gate only)

Not written by the implementer. (a) ≥ 120 synthetic gold units by a fresh-context subagent that
sees only plan 0002 §Units, §The tree, §Precedence, §Detail rule and §Tags, following the
workflow above; (b) ≥ 120 real units from `hbench pilot-dump-v2` cells (arms s3d-rl600,
nla-rl-L42, jlens, sae-l42, patchscopes; stratified by v5c class), hand-labelled blind and frozen
before the first check. Minimums and the gate are in plan 0002 §Calibration. At most 3 held-out
checks, each logged below.

## Check log (v2)

| date | split | prompt version | result | notes |
|---|---|---|---|---|
| 2026-10-11 | dev | claimmap-v2 | all dev rows pass except detail recall's Wilson bound (1.000, n = 5, lower 0.566) | group acc unit 0.954 / char 0.945, leaf acc 0.950, recall false_claim 1.00 / substitution 0.944 / correct 0.900, sib-vs-cat 1.00, claim F1 0.985, flip 0.031; 2 runs, $0.97; first dev run, no prompt edits |

Gold-label fixes (v2): none.
| 2026-10-11 | dev | claimmap-v2.1 (rev 11 framing rule; 32 cases, punctuation-only segments excluded) | all dev rows pass | group acc unit 0.966 / char 0.958, leaf acc 0.959, recall false_claim 1.00 / detail 0.923 (n 13) / substitution 1.00 / correct 0.910, sib-vs-cat 0.944, claim F1 0.992, flip 0.000; 2 runs, $2.06 |
| 2026-10-11 | dev | claimmap-v2.3 (rev 13: supplementary branch, verdicts, probe targets; single-call, 34 cases) | all dev rows pass | group acc unit 0.940 / char 0.937, verdict acc 0.940, leaf acc 0.921, recall false_claim 1.00 / detail 0.923 / substitution 0.944 / correct 0.881, sib-vs-cat 0.819, claim F1 0.981, flip 0.044; 1 of 68 calls failed (dev2-32 r1); probe_target missing 0/25 per run; 2 runs, $1.42 |
| 2026-10-11 | dev | claimmap-v2.4 (rev 14: narrow unrealised_option, name anchors, in-run retries; 38 cases) | all dev rows pass | group acc unit 0.955 / char 0.951, verdict acc 0.955, leaf acc 0.943, recall false_claim 0.964 / detail 0.900 / substitution 1.00 / correct 0.903, sib-vs-cat 0.900, claim F1 0.991, flip 0.006; 0 failed calls; unrealised_option recall 1.00 (n 3); 2 runs, $2.16 |
| 2026-10-11 | dev | claimmap-v2.5 (rev 15: hallucination = obviously contradicted; unsupported_assertion; unrealised_prediction reinstated; 41 cases) | all dev rows pass | group acc unit 0.963 / char 0.956, verdict acc (7) 0.963, leaf acc (22) 0.935, recall false_claim 1.00 (n 6) / detail 0.964 / substitution 1.00 / correct 0.900, sib-vs-cat 0.900, claim F1 0.987, flip 0.006; unsupported_assertion 1.00 (n 9), unrealised_prediction 0.667 (n 6, misses go to unrealised_option); 2 runs, $3.73. Gold fix after the first scoring: dev2-36 "development environment" false_claim -> unrealised_prediction (rev 15 item 3; the judge read it as uncontradicted) |
