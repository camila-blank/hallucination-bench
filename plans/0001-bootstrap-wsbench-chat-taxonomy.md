# Plan 0001 — Bootstrap HallucinationBench v2; port `wsbench_chat`; add the taxonomy judge

Issue #1. Author: Camila Blank (planned with Claude), 2026-10-11.

## Goal

HallucinationBench v2 is a suite of evals giving a comprehensive view of the **types and
severities** of hallucination that activation-reading lenses produce. This PR:

1. Bootstraps the repo: a slim shared core (LLM client, cache, readout contract, token-bag
   summariser, taxonomy) that later evals reuse.
2. Ports WorkspaceBench's `hallucination_chat` eval as the first eval, `wsbench_chat`, with its
   v5c detection judge reproduced **verbatim** (the existing instrument).
3. Adds a **taxonomy judge** (`taxonomy-v1`) that labels every hallucinated span by type
   (**concept vs detail**), target, severity and basis, and separately finds *unsupported
   fabricated specifics* that v5c deliberately does not mark.
4. Calibrates the taxonomy judge on a hand-labelled case set, then runs both judges over all
   11 stored lens arms and writes per-lens hallucination profiles.

Non-goals: new banks/evals beyond `wsbench_chat` (later plans), producing new readouts (we
re-judge stored ones), GPU code.

## Sources (read-only; port, do not import)

- Public WorkspaceBench, local clone `/workspace/root/wsbench-pub` @ `92d763e` (MIT, same authors):
  - `src/wsbench/evals/hallucination/{prompts,judge,score}.py` — v5c prompt, `verify_quote`,
    `decode_token`, `parse_verdict`, readout classes, rates, item bootstrap.
  - `evals/hallucination/{items.json,capture_rows.json,judge_prompt_v5c_chat.json,README.md}`.
  - `src/wsbench/{llm,cache,readouts,summarizer}.py` — client (Anthropic / OpenRouter routing,
    retries, RPM pacer, spend), JSONL cache with fingerprints, readout loader, `interp-v1`
    summariser.
- Stored readouts (inputs, not copied into git):
  `/workspace/wsbench-paper-data/readouts/<arm>/hallucination.jsonl`, rows
  `{id, layer, pos, token, samples}` (prose) or `{..., tokens, scores}` (token bag).

| arm | kind | layers | k |
|---|---|---|---|
| s3d-rl600 (O-lens RL), s3d-sft251 (O-lens SFT) | prose | 20,36,44,48,60 | 3 |
| nla-rl-L42, nla-sft-L42 | prose | 42 | 3 |
| patchscopes | prose | 20,36,44,48,60 | 1 |
| jlens, logit, tuned, rlens, template-v3 | tokens | 20,36,44,48,60 | 1 (summarised) |
| sae-l42 | tokens | 42 | 1 (summarised) |

## Repo layout (new files)

```
hallucination-bench/
├── CLAUDE.md                      # agent runbook: layout, conventions, gotchas, run commands
├── README.md                      # updated: suite overview, taxonomy, quickstart, results pointer
├── pyproject.toml                 # uv project `hallucination-bench`, py>=3.12, deps: anthropic, openai, numpy; dev: pytest, ruff
├── plans/0001-bootstrap-wsbench-chat-taxonomy.md
├── data/
│   ├── calibration/taxonomy_cases.jsonl   # hand-labelled cases (see §Calibration)
│   └── wsbench_chat/                      # verbatim copies + README.md (provenance, source commit)
│       ├── items.json
│       ├── capture_rows.json
│       └── judge_prompt_v5c_chat.json
├── src/hallucination_bench/
│   ├── __init__.py
│   ├── llm.py            # client: claude-* -> Anthropic SDK, else OpenRouter; retries, pacer, Spend
│   ├── cache.py          # JSONL cache {key, fingerprint, result, meta}; resume-safe, append-only
│   ├── readouts.py       # readout contract loader + decode_token + verify_quote
│   ├── summarizer.py     # interp-v1 token-bag -> prose (verbatim prompt)
│   ├── taxonomy.py       # THE taxonomy: enums + definitions + examples (single source of truth)
│   ├── judges/
│   │   ├── __init__.py
│   │   ├── v5c.py        # verbatim v5c prompt/schema/parse (loads the frozen JSON, asserts equality)
│   │   └── taxonomy_v1.py# taxonomy judge prompt (rendered from taxonomy.py), schema, parse
│   ├── evals/
│   │   ├── __init__.py   # registry: name -> module
│   │   └── wsbench_chat/
│   │       ├── __init__.py
│   │       ├── bank.py   # load items, marked response, expected cell grid
│   │       ├── run.py    # cells -> (summarise) -> v5c -> taxonomy -> cells.jsonl
│   │       └── score.py  # headline + taxonomy metrics, item bootstrap CIs
│   ├── calibrate.py      # run taxonomy judge on data/calibration, report agreement with gold
│   ├── report.py         # cross-arm tables (markdown + JSON), audit sample dump
│   └── cli.py            # `hbench run|calibrate|report` (argparse)
├── scripts/run_all_arms.sh  # tmux driver: all arms, layer slice then all layers; logs/
├── docs/results/0001_wsbench_chat_taxonomy.md  # results write-up (filled after the run)
└── tests/
    ├── test_v5c_frozen.py      # prompt literals == frozen JSON; parse_verdict rules
    ├── test_taxonomy_judge.py  # schema, verbatim-span checks, fail-closed rules, prompt render
    ├── test_score.py           # metrics on synthetic cells (denominators, CIs, decomposition identity)
    ├── test_readouts.py        # loader, decode_token, dedupe, token/prose kinds
    └── test_cache.py           # resume, fingerprint mismatch -> re-query
outputs/   (gitignored) outputs/wsbench_chat/<judge_model_slug>/<arm>/{cells.jsonl,summary.json}
logs/      (gitignored)
```

## The taxonomy (`taxonomy.py`, v1)

Every **hallucinated span** gets four labels. Definitions live once in `taxonomy.py` and are
rendered into the judge prompt; tests assert the prompt contains every label and definition.

1. **grounding** — why it counts as a hallucination:
   - `contradicted`: the response (before or after the marker) or established world knowledge
     rules the span out. These are exactly the spans v5c marks `wrong` (see §Judges).
   - `unsupported`: a *specific* claim (a name, number, date, quotation, event, attribute) that is
     not stated in the response, not realised in it, and not a generic reasonable inference — a
     fabricated specific. Includes a specific prediction of what comes next that the response
     never realises; such spans also get `is_prediction = true` so analyses can include/exclude
     them. (This is the user's definition: unfulfilled predictions and fabricated specifics are
     hallucinations; grounded = stated or realised only.)
2. **type** — the primary axis requested in issue #1:
   - `concept`: the span gets the **meaning** wrong. It substitutes a different, meaningfully
     distinct entity, idea, event or situation, or reverses a claim, stance or outcome. Test: a
     reader who believed the span would misunderstand *what the text is about or what it
     claims*. Examples: "Picasso" when the text is about da Vinci; Johnson where the text says
     Kennedy; "argues remote work hurts productivity" where it argues the opposite; "a recipe
     for bread" where the response is a poem about bread; "the user is angry" where the user is
     grateful.
   - `detail`: the gist is right but a **specific value** is wrong or invented. Test: replacing
     the span with the correct value would leave the reader's understanding of the substance
     intact. Examples: "Steve" when the text is about an arbitrary character named Joe;
     "October 24" where the text says October 29; "3 tips" where it lists 5; a heading quoted as
     "Quick Facts" where it is "Basic Facts"; "Python 3.8" where it says 3.10.
   - Tie-break rule (stated in the prompt): a wrong **named entity with its own real-world
     identity** (a person, work, place, organisation, scientific concept that the reader would
     recognise as a different thing) is `concept`; a wrong **arbitrary label** (a fictional or
     incidental name, an exact number/date/count, wording) is `detail` — *unless* the wrong value
     flips the claim (e.g. "safe" vs "unsafe" dose, "won" vs "lost", "increase" vs "decrease"),
     which is `concept`. Magnitude alone does not make it concept; it raises severity.
3. **target** — what kind of thing is wrong (one of):
   `entity` (named person/place/work/org/product), `quantity` (number, date, time, count,
   measurement), `event` (what happened / who did what), `attribute` (a property or relation
   of something: cause, part-of, attribution of a quote), `stance` (opinion, polarity,
   negation, sentiment, intent), `structure` (headings, list items, quoted wording, format or
   what the response has already written), `topic` (the overall subject or situation of the
   conversation).
4. **severity** — how much the span would mislead a reader about this conversation:
   - `1 minor`: peripheral; does not touch anything the response relies on.
   - `2 moderate`: misstates a specific point the response makes or relies on.
   - `3 major`: misstates the central subject, claim, answer or situation.

Off-topic junk (v5c `off_topic`) is kept as its own category and reported, but it is not a
hallucination and gets no taxonomy labels (same as v5c, where off-topic spans do not count).

## Judges

### v5c (detection; the existing instrument, verbatim)

- Port `prompts.py` / `parse_verdict` / readout classes / rates from wsbench-pub without
  changing a byte of the prompt; `tests/test_v5c_frozen.py` asserts the literals equal
  `data/wsbench_chat/judge_prompt_v5c_chat.json` (same check as wsbench-pub
  `tests/test_hallucination.py`).
- Classes and rules unchanged: span must be verbatim in its own readout or it is dropped and
  counted (`n_unverified_spans`); unknown span type counts as wrong (fail closed); any verified
  wrong span → `hallucinated`, else generic/empty → `generic`, else junk span → `off_topic`,
  else `consistent`. `hallucination_rate = hallucinated / specific`,
  `specific = hallucinated + off_topic + consistent` (unrevoked headline; `revoked` reported).
- Token arms: decode with `decode_token`, summarise with `interp-v1` (`render_bag` of decoded
  tokens + scores), judge the interpretation as one readout (k = 1). A failed summary leaves the
  cell unjudged (counted, never scored).

### taxonomy-v1 (classification + fabricated-specifics detection; new)

- **One call per cell that has at least one `specific` readout under v5c.** Cells where every
  readout is generic/empty are not sent (nothing to classify; recorded as `skipped_nonspecific`).
- Input: the same prompt, marked response and readouts as v5c (same `⟦READ⟧` convention), plus
  the list of v5c's **verified wrong spans** per readout (verbatim text + v5c's `why`).
- Task A — **classify** each given wrong span: `type`, `target`, `severity`, `basis` ∈
  {`before_marker`, `after_marker`, `world_knowledge`} (what rules it out), a one-clause
  `why`, and `agree` (bool: the taxonomy judge also thinks it is wrong). `grounding` is
  `contradicted` for all of these (set in code, not by the judge). `agree = false` spans stay in
  the v5c headline (the instrument is unchanged) but are reported as `disputed` counts.
- Task B — **find unsupported specifics**: list further spans of each readout that are
  `unsupported` per the taxonomy definition, each with `type`, `target`, `severity`,
  `is_prediction`, `why`. Must not repeat a Task-A span (overlap check in code: a Task-B span
  whose normalised text overlaps a Task-A span of the same readout is dropped and counted).
- Every span text must be verbatim in its own readout (`verify_quote`, same normalisation as
  v5c) or it is dropped and counted (`n_unverified_tax_spans`). Task-A entries must match a
  given v5c span (by index); a missing or extra Task-A entry → that v5c span gets
  `type = unknown` and is counted (`n_unclassified`), never silently guessed. Unknown enum
  values are rejected by the JSON schema; if one slips through it becomes `unknown`.
- Output schema (JSON schema enforced via the client):
  `{readouts: [{idx, classified: [{span_idx, agree, type, target, severity, basis, why}],
  unsupported: [{text, type, target, severity, is_prediction, why}]}], rationale}`.
- Prompt version string `taxonomy-v1`; the fingerprint includes it, so prompt edits during
  calibration invalidate cached calls automatically.

### Judge model

- Default `claude-opus-5-5` for v5c, taxonomy and summariser (one model; keyless first-party
  auth through the `ant` CLI profile — the client must construct `anthropic.Anthropic()` with no
  `api_key` when `ANTHROPIC_API_KEY` is unset so the SDK picks up the `ant` OAuth profile; do not
  require the env var). Overridable with `--judge-model`, `--summarizer-model`.
- Note in the README: the WorkspaceBench pin is Gemini 3.8 Flash; our absolute v5c rates are
  therefore not directly comparable to published WorkspaceBench numbers (the AR-ablation study
  also used Opus 5.5 for this family).
- Temperature 0 where supported; structured JSON output; thinking off. Failed/refused calls
  return `None` and are counted, never scored; re-running resumes from the cache and retries
  only failed cells.

## Metrics (`score.py`)

Denominator for readout-level rates = **specific readouts** (as v5c). All rates come with an
item-level bootstrap 95% CI (2000 draws, seed 0, resample items).

Per arm, overall and per layer:
- `hallucination_rate` (v5c headline, unchanged) and `off_topic_rate`.
- Decomposition of contradicted spans: `concept_rate` = share of specific readouts with ≥1
  contradicted `concept` span; `detail_rate` likewise; `concept_only`, `detail_only`, `both`.
  Identity check (tested): `concept_only + detail_only + both + unclassified_only = hallucination_rate`.
- Severity: `major_rate` (≥1 contradicted span with severity 3), and mean max-severity per
  hallucinated readout.
- Unsupported: `unsupported_rate` (≥1 unsupported span), split concept/detail and
  `unsupported_rate_excl_predictions`.
- `strict_hallucination_rate` = ≥1 contradicted **or** unsupported span (user's definition).
- Span-level distributions: counts and shares by grounding × type × target × severity × basis.
- Judge health: `n_cells`, `n_missing`, `n_failed_calls`, `n_unverified_spans`,
  `n_unverified_tax_spans`, `n_unclassified`, `n_disputed` (v5c wrong spans the taxonomy judge
  disagreed with), spend in USD and tokens.

`summary.json` per arm holds all of the above plus config (judge model, prompt versions,
source commit of the bank, readouts path + sha256).

## Calibration (gate before the big run)

`data/calibration/taxonomy_cases.jsonl`: **≥ 40 hand-labelled cases**, each a short prompt +
marked response + 1–3 readouts + gold labels for the spans that matter
(`{readout_idx, text, grounding, type, target, severity}` and an explicit list of spans that
must **not** be flagged). Coverage requirements:
- Issue #1's two anchor examples verbatim in spirit: Picasso-for-da-Vinci (concept) and
  Steve-for-arbitrary-Joe (detail).
- ≥ 10 concept and ≥ 10 detail contradicted spans across all 7 targets; ≥ 4 tie-break cases
  (wrong real named entity vs wrong fictional name; a number that flips the claim vs one that
  does not; a magnitude error that stays detail).
- ≥ 6 unsupported cases (fabricated name/number/quote; unrealised specific prediction) and ≥ 4
  traps that must NOT be flagged unsupported (generic reasonable predictions, paraphrases,
  correct inferences, hedged statements).
- ≥ 3 off-topic junk readouts (must get no taxonomy labels).
- Cases are written by the implementer from scratch (not lifted from lens outputs), then reviewed
  label-by-label by the main session before the gate runs.

`hbench calibrate` feeds the gold contradicted spans to taxonomy-v1 as if they were v5c spans
(Task A) and scores Task B against the gold unsupported spans. Report: type accuracy, target
accuracy, severity exact + within-1, unsupported precision/recall, false flags on traps, and
the confusion matrix. **Gate:** type accuracy ≥ 90%, unsupported precision ≥ 80% and recall
≥ 70%, ≤ 1 trap flagged. If it fails, revise the prompt (bump to `taxonomy-v1.1`, …), never the
gold labels, unless a gold label is shown to be wrong — then fix it and record why in the
calibration README. Report the final calibration numbers in the results doc.

## Run

1. `hbench run --eval wsbench_chat --arm <arm> --readouts <path> [--layers 44] [--concurrency N]`.
2. **Stage 1 (layer-matched slice):** all 11 arms at one layer each — L44 for the 5-layer arms,
   L42 for the single-layer arms (nla-*, sae-l42). ≈ 12.4k cells.
3. **Stage 2:** the remaining layers (20, 36, 48, 60) of the 5-layer arms. ≈ 36k cells.
4. Driver `scripts/run_all_arms.sh` runs everything in one tmux session `hbench`, arms in
   parallel (bounded by `--concurrency` per arm and a global RPM), logging to
   `logs/run_<timestamp>.log`; it ends with `hbench report`. Re-running the script resumes from
   caches. Before the full stage, a 20-cell dry slice per arm prints prompts and checks
   parsing; also print an estimated cost from the slice.
5. `hbench report` writes `outputs/wsbench_chat/<model>/report.{md,json}`: one row per arm
   (and per arm × layer) with every metric above, and `audit_samples.md`: 15 random spans per
   (type × grounding) across arms with context, for human spot-checks.

## Deliverables / acceptance

- `uv run pytest` and `uv run ruff check` + `ruff format --check` pass.
- `test_v5c_frozen.py` proves the v5c prompt is byte-identical to the frozen JSON.
- Calibration gate passed and reported.
- Stage 1 complete for all 11 arms (missing + failed < 1% of cells per arm), report written;
  stage 2 complete or running in tmux with resumable driver.
- `docs/results/0001_wsbench_chat_taxonomy.md`: headline table per arm (v5c rate, concept rate,
  detail rate, major rate, unsupported rate, strict rate, each with CI), the concept/detail
  share of contradicted spans, target mix, a handful of quoted examples per type, calibration
  numbers, spend, and caveats (judge differs from the WorkspaceBench pin; token arms judged
  through a summariser, so their labels describe the summary).
- README (taxonomy + quickstart + results pointer) and CLAUDE.md (layout, conventions,
  gotchas: keyless `ant` auth, readouts location, cache/resume, 32 GB box RAM cap) written.
