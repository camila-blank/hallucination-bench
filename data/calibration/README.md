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
