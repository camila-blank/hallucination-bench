# Plan 0001 — Bootstrap HallucinationBench v2; port `wsbench_chat`; add the taxonomy judge

Issue #1. Author: Camila Blank (planned with Claude), 2026-10-11. Revision 5 (after critiques 1–4).

## Goal

HallucinationBench v2 is a suite of evals that gives a comprehensive view of the **types and
severities** of hallucination produced by activation-reading lenses. This PR:

1. Bootstraps the repo with a slim shared core (LLM client, cache, readout contract, token-bag
   summariser, taxonomy) that later evals reuse.
2. Ports WorkspaceBench's `hallucination_chat` eval as the first eval, `wsbench_chat`. Both of
   its existing judge stages are reproduced **verbatim**: v5c span detection and v5c claim
   verify.
3. Adds a third stage, the **taxonomy classifier** (`taxonomy-v1`). It labels every
   hallucinated span that the first two stages found, by type (**concept vs detail**), target
   and severity, plus a grounding subtype. It only classifies; it never decides whether a span
   is a hallucination.
4. Calibrates the classifier on a hand-labelled set with a held-out gate. Then it runs all three
   stages over all 11 stored lens arms and writes per-lens hallucination profiles.

Non-goals: new banks or evals beyond `wsbench_chat` (later plans), producing new readouts (we
re-judge stored ones), and GPU code.

## Sources (read-only; port them, do not import them)

- Public WorkspaceBench, local clone `/workspace/root/wsbench-pub` @ `92d763e` (MIT, same
  authors):
  - `src/wsbench/evals/hallucination/{prompts,judge,score}.py`: the v5c span prompt
    (`JUDGE_*`, `v5c-chat`), the verify prompt (`VERIFY_*`, `v5c-chat-verify-v1`),
    `verify_quote`, `decode_token`, `parse_verdict`, `tally_claims`, `fully_judged`, the
    readout classes, the rates and the item bootstrap.
  - `evals/hallucination/{items.json,capture_rows.json,judge_prompt_v5c_chat.json,README.md}`.
    The README documents the cell grid, missing-cell and empty-cell rules, and the verify
    stage's tally rules. Port those rules as written.
  - `src/wsbench/{llm,cache,readouts,summarizer}.py`:
    - `llm.py`, the client: Anthropic / OpenRouter routing, retries, RPM pacer and spend tally;
    - `cache.py`, the JSONL cache with fingerprints;
    - `readouts.py`, the readout loader;
    - `summarizer.py`, the `interp-v1` summariser.
- Stored readouts are inputs and are not copied into git:
  `/workspace/wsbench-paper-data/readouts/<arm>/hallucination.jsonl`. Rows are
  `{id, layer, pos, token, samples}` (prose) or `{..., tokens, scores}` (token bag). Each
  arm-layer has 1,123 cells (149 items × their sites).

| arm | kind | layers | k |
|---|---|---|---|
| s3d-rl600 (O-lens RL), s3d-sft251 (O-lens SFT) | prose | 20,36,44,48,60 | 3 |
| nla-rl-L42, nla-sft-L42 | prose | 42 | 3 |
| patchscopes | prose (often `[""]`, which counts as an empty cell) | 20,36,44,48,60 | 1 |
| jlens, logit, tuned, rlens, template-v3 | tokens | 20,36,44,48,60 | 1 (summarised) |
| sae-l42 | tokens (feature descriptions) | 42 | 1 (summarised) |

No stored Opus-5.5 v5c caches exist for these exact readout files. The lens-agreement study
used a different pipeline, and the AR-ablation caches cover different arms. Everything is
judged fresh. `readouts/sae-l42-top20/` holds only a jailbreak file, so it is not an arm
here.

## Repo layout (new files)

```
hallucination-bench/
├── CLAUDE.md                      # agent runbook: layout, conventions, gotchas, run commands
├── README.md                      # updated: suite overview, taxonomy, quickstart, results pointer
├── pyproject.toml                 # uv project `hallucination-bench`, py>=3.12, deps: anthropic, openai, numpy; dev: pytest, ruff
├── plans/0001-bootstrap-wsbench-chat-taxonomy.md
├── data/
│   ├── calibration/
│   │   ├── README.md              # how the cases were made, any gold-label fixes and why
│   │   ├── dev.jsonl              # synthetic cases, used for prompt iteration
│   │   └── heldout.jsonl          # synthetic + real-readout spans, gate only
│   └── wsbench_chat/              # verbatim copies + README.md (provenance, source commit)
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
│   │   ├── v5c.py        # verbatim span + verify prompts/schemas/parsers (asserts == frozen JSON)
│   │   └── taxonomy_v1.py# classifier prompt (rendered from taxonomy.py), schema, parse
│   ├── evals/
│   │   ├── __init__.py   # registry: name -> module
│   │   └── wsbench_chat/
│   │       ├── __init__.py
│   │       ├── bank.py   # load items, marked response, expected cell grid
│   │       ├── run.py    # cells -> (summarise) -> span -> verify -> classify -> cells.jsonl
│   │       └── score.py  # headline + taxonomy metrics, item bootstrap CIs
│   ├── calibrate.py      # run classifier on calibration cases, score against gold, apply the gate
│   ├── report.py         # cross-arm tables (markdown + JSON), audit sample dump
│   └── cli.py            # `hbench run|calibrate|report|pilot-dump` (argparse)
├── scripts/run_all_arms.sh  # tmux driver (single process, all arms); logs/
├── docs/results/0001_wsbench_chat_taxonomy.md  # results write-up (filled in after the run)
└── tests/
    ├── test_v5c_frozen.py      # span + verify literals == frozen/source; parse_verdict + tally rules
    ├── test_taxonomy_judge.py  # schema, span-index matching, fail-closed rules, prompt render
    ├── test_score.py           # metrics on synthetic cells (denominators, CIs, decomposition identity)
    ├── test_readouts.py        # loader, decode_token, dedupe, token/prose kinds, empty cells
    └── test_cache.py           # resume, fingerprint mismatch -> re-query, partial cells re-queued
outputs/   (gitignored) outputs/wsbench_chat/<judge_model_slug>/<arm>/{cells.jsonl,summary.json}
logs/      (gitignored)
```

## Pipeline per cell (one cell = one arm × item × site × layer)

1. **Summarise** (token arms only): decode each token with `decode_token`, then summarise the
   bag with `interp-v1` (`render_bag` of the decoded tokens and scores). The interpretation is
   the single readout (k = 1). A failed summary leaves the cell unjudged; it is counted and
   never scored.
   - Caveat, matching the source: sae-l42 feature descriptions and template-v3 words are not
     byte-level BPE, so `decode_token` garbles any non-ASCII in them, and the prompt calls them
     "token readouts". Keep this as it is and note it in the README.
2. **Span stage (v5c, verbatim).** It returns, for each readout, `kind` plus `wrong` and
   `off_topic` spans, and gives each readout a class: `hallucinated` / `off_topic` /
   `consistent` / `generic`. All rules are unchanged:
   - A span must be verbatim in its own readout, or it is dropped and counted.
   - An unknown span type counts as wrong.
   - Revocation is reported.
   - `fully_judged` re-queues partially judged cells.
3. **Verify stage (v5c-chat-verify-v1, verbatim).** The span stage's verified wrong spans are
   passed in as established-false. The stage lists every other specific claim with status
   `true` / `unverifiable` / `disputed` / `off_topic`. The `tally_claims` rules are unchanged:
   - A quote must be verbatim in its readout, or it is dropped and counted.
   - A quote that overlaps an established-false span is skipped.
   - An unknown status becomes `unverifiable`.
   - A malformed answer is rejected and re-queued.

   We additionally keep the verified quotes themselves, because the classifier needs them. A
   wrapper around the frozen `tally_claims` returns per-status quote lists that are **disjoint
   by status**: a `disputed` claim is listed only under `disputed`, although the frozen tally
   also counts it as unverifiable. A test asserts that `len(unsupported) + len(disputed)` equals the
   frozen tally's `unverifiable`, and that the other statuses are equal. Routing is verbatim: verify runs on **every span-judged cell**, as in the source.
   Non-specific readouts are ignored downstream, so `n_cells_claims_unjudged` keeps its source
   meaning. Before classification, quotes are deduplicated within each readout, and any quote
   that overlaps one of that readout's v5c `off_topic` spans is dropped and counted
   (`n_offtopic_overlap_quotes`). Verify only sees `wrong` spans as established-false.
4. **Classify stage (taxonomy-v1, new).** One call per cell that has at least one span to
   classify on a specific-class readout. The spans to classify are:
   - every verified v5c `wrong` span (grounding `contradicted`);
   - every verify `unverifiable` claim (grounding `unsupported`);
   - every verify `disputed` claim (grounding `disputed`).

   A readout is specific-class when its class is in `SPECIFIC_CLASSES` (`hallucinated`,
   `off_topic`, `consistent`). This routes on the **class**, not the judge's `kind`, because
   v5c can call a readout `generic` and still mark a wrong span in it, which makes it
   `hallucinated`. The classifier sees the prompt, the marked response, the readouts, and a
   numbered list of the spans, each with its readout index and its **grounding label**
   (`contradicted` / `disputed` / `unsupported`). It needs the grounding because the type test
   and `unsupported_kind` depend on it. It does **not** see the earlier stages' `why` text, so
   it is not anchored on their reasoning. It labels each span; it never adds, removes or
   re-judges spans.
   - **Timing.** Classify runs only after the cell's verify call has tallied, or after verify
     has failed for good in this run. In that case it classifies the contradicted spans only.
   - **Re-classification.** A changed span list changes the rendered prompt, so the `:C`
     fingerprint no longer matches and the cell is re-classified on the next run (for example,
     after a later verify succeeds).
   - **Class source.** Verify's own per-readout `kind` is ignored; the class always comes from
     v5c.

Readouts of class `generic` (no wrong span) are not part of any rate, the same as the v5c
denominator, and the classifier never sees them. `off_topic` spans are never
classified. Verify's `off_topic` claims are dropped, as in the source.

## The taxonomy (`taxonomy.py`, v1)

Every hallucinated span gets the labels below. The definitions live only in `taxonomy.py` and
are rendered into the classifier prompt. Tests assert that the prompt contains every label and
definition.

### grounding (set in code from the stage that found the span, never by the classifier)

- `contradicted`: a verified v5c wrong span. The response, or the facts, rule it out.
- `disputed`: verify thinks the response rules the claim out, but v5c did not flag it.
- `unsupported`: verify `unverifiable`. The claim is specific, but the response neither states
  it nor clearly entails it, and nothing realises it after the marker. This follows the user's
  definition: only content that is stated or realised counts as grounded. Plausible inferences,
  unrealised predictions and fabricated specifics are therefore all hallucinations under the
  strict view. A claim that is true in the world but not stated in the conversation also counts
  as `unsupported`, matching verify's "never world knowledge" rule. The classifier records why
  in the subtype below, so analyses can choose a lenient view.

### unsupported_kind (classifier label; `unsupported` spans only)

- `prediction`: about what the response will say next, and the response never says it.
- `inference`: a plausible inference from the text that the text does not state or entail. This
  includes claims about the model's intent or mental state.
- `fabricated`: a specific with no basis in the text, such as an invented name, number, date,
  quotation, event or source.
- `world_knowledge`: a fact about the world, possibly true, that the conversation does not
  state.

The **lenient view** drops `inference` and `world_knowledge`. The **strict view** keeps all four.
`contradicted` and `disputed` spans have `unsupported_kind = null` and are always kept in both
views.

### type (the primary axis requested in issue #1)

The **main test** decides the type: if a reader believed the span, would they misunderstand
**what the text is about or what it claims**?

- `concept`: yes, the meaning is wrong. The span substitutes a different, meaningfully distinct
  entity, idea, event or situation **as the subject of a claim**, or it reverses a claim,
  stance or outcome. Examples:
  - "Picasso" when the text is about da Vinci;
  - Johnson where the text says Kennedy;
  - "argues remote work hurts productivity" where the text argues the opposite;
  - "a recipe for bread" where the response is a poem about bread;
  - "the user is angry" where the user is grateful;
  - "a 200 mg dose is safe" where the text says it is dangerous.
- `detail`: no, the gist is right. A **specific value** inside an otherwise correct
  understanding is wrong or invented. Examples:
  - "Steve" when the text is about an arbitrary character named Joe;
  - "October 24" where the text says October 29;
  - "over 700 died" where the text says over 140;
  - "3 tips" where the text lists 5;
  - "Quick Facts" quoted as a heading where it is "Basic Facts";
  - an incidental real-world name swapped in a passing mention (the text cites "a 2019 Stanford
    study" in an aside and the readout says Harvard).

Rules:

- **Names: apply this decision order.** What decides the type is whether the wrong name carries
  its own identity. Whether the referent is "the same" does not decide it.
  1. The wrong name denotes a **distinct identity** and is **the subject of the claim** →
     `concept`. A distinct identity is a real person, work, place or organisation, or another
     character or entity present in the text. Examples: Picasso for da Vinci; Johnson for
     Kennedy; confusing the protagonist with her rival; an arbitrary protagonist Joe called
     "Barack Obama".
  2. The wrong name denotes a distinct identity but appears only in a **passing mention** →
     `detail`. Example: the text cites "a 2019 Stanford study" in an aside and the readout says
     Harvard.
  3. Otherwise, the name is an arbitrary or invented label for the right referent → `detail`,
     even when the referent is the protagonist. Example: Steve for the story's Joe.
- **Wrong values:** a number, date, label or wording error is `detail`. The exception is a value
  that flips the claim (safe vs unsafe, won vs lost, increase vs decrease), which is `concept`.
  Magnitude alone never makes a span `concept`; it raises severity instead.
- **`contradicted` and `disputed` spans** use the main test above, since a correct value exists.
- **For `unsupported` spans** there is no "correct value", so the test becomes:
  does believing the span change the reader's understanding of what the text is about or what
  it claims?
  - An invented event, claim, topic, situation, intent or stance is `concept`. Examples: "the
    response goes on to recommend therapy", when it never does and nothing suggests it; "the
    user is a teacher".
  - An invented name, number, date, quotation or other specific value attached to something
    the text does discuss is `detail`. Examples: "Joe's sister Maria", when the text never
    names her; "about 30 minutes".

### target (what kind of thing is wrong; exactly one)

- `entity`: a named person, place, work, organisation or product.
- `quantity`: a number, date, time, count or measurement.
- `event`: what happened, or who did what.
- `attribute`: a property or relation of something, such as a cause, part-of, or the
  attribution of a quote.
- `stance`: an opinion, polarity, negation, sentiment or intent.
- `structure`: headings, list items, quoted wording, format, or what the response has already
  written.
- `topic`: the overall subject or situation of the conversation.

### severity (how much the span would mislead a reader about this conversation)

- `1 minor`: peripheral; it does not touch anything the response relies on.
- `2 moderate`: it misstates, or invents, a specific point the response makes or relies on.
- `3 major`: it misstates, or invents, the central subject, claim, answer or situation.

## taxonomy-v1 classifier details

- Input spans are numbered `S1..Sn`, each with its readout index.
- The output schema is enforced through the client's structured output:
  `{spans: [{span_id, type, target, severity, unsupported_kind|null, why}], rationale}`.
- Parsing fails closed:
  - A span id that is missing or duplicated gives that span `type = unknown`; it is counted in
    `n_unclassified`.
  - An extra id is ignored and counted.
  - An `unsupported_kind` given for a `contradicted` or `disputed` span is ignored (set to null).
  - A missing `unsupported_kind` for an `unsupported` span becomes `unknown`.
  - Enum values are validated in code too. A bad value makes the answer invalid. The answer is
    cached as a failure and re-queued on every later run, as `validate` does in wsbench. If it
    is still invalid at report time, the bad labels count as `unknown`.
- The prompt version is `taxonomy-v1`. The fingerprint includes the prompt version, the model
  and the rendered prompt, so prompt edits during calibration invalidate cached calls.

## Judge model and concurrency

- **Model.** The default is `claude-opus-5-5` for all four calls (summariser, span, verify,
  classify), using keyless first-party auth through the `ant` CLI profile. When
  `ANTHROPIC_API_KEY` is unset, the client constructs `anthropic.AsyncAnthropic()` with no
  `api_key`, so the SDK picks up the `ant` OAuth profile. It must not require the env var: the
  source's `api_key()` raises when the variable is unset (wsbench `llm.py` L63-67), so the port
  changes that function, not just the constructor.
  Overrides: `--judge-model`, `--summarizer-model`, `--classifier-model`.
- **Comparability note for the README.** The WorkspaceBench pin is Gemini 3.8 Flash, so our
  absolute v5c rates are not directly comparable to published WorkspaceBench numbers. The
  AR-ablation study also used Opus 5.5 for this family.
- **Call settings.** The Anthropic route sends no `temperature`, because newer Claude models
  reject the field; the source drops it too (wsbench `llm.py` L320-334). Labels are therefore
  single samples at the default temperature, and the README says so. Calls use structured
  JSON output with thinking off. A failed or refused call returns `None`; it is counted and never scored.
  Re-running resumes from the cache and retries only failed or partial cells.
- **One process for all arms.** `hbench run` takes several `--arm name=path` arguments, or
  `--readouts-root DIR`, and judges every arm in **one process**. That process has one shared
  RPM pacer (`--rpm`, default 1500) and one thread pool (`--concurrency`, default 64).
  Parallelism therefore never multiplies the rate limit. Stages pipeline per cell.

## Cache and outputs

- **Cache.** Each arm has its own cache file, `<arm>/cache.jsonl`, separate from the output.
  - Keys: `summ:{id}__L{layer:03d}__p{pos}` (summary), `{id}__L{layer:03d}__p{pos}` (span), `…:V`
    (verify) and `…:C` (classify).
  - Fingerprints include the prompt version, the model and the rendered prompt, so a classifier
    prompt bump re-runs only the `:C` entries.
- **Outputs.** `cells.jsonl` holds one merged record per cell: readouts, all stage outputs and
  labels.
  - `cells.jsonl` and `summary.json` are **regenerated from the cache over every judged layer**,
    at the end of each `hbench run` and by `hbench report`. Stage 2 therefore extends stage 1's
    outputs rather than writing a separate file.
  - `hbench run --rebuild` regenerates them with no API calls.
- **Short prose cells.** Some s3d rows have 0–2 non-empty samples, fewer than k. They are judged
  with what they have and counted in `n_short_cells`. All rates are readout-level, so they are
  unaffected.

## Metrics (`score.py`)

**Denominator.** Readout-level rates are over **specific readouts** (class in
`SPECIFIC_CLASSES`), as in v5c. Revoked readouts count, because the headline is unrevoked.
Every rate gets an item-level bootstrap 95% CI: 2000 draws, seed 0, resampling items.

Per arm, overall and per layer:

- **Headline:** `hallucination_rate` (v5c, unchanged) and `off_topic_rate`.
- **Contradicted decomposition.** Each hallucinated readout falls in exactly one bucket by the
  types of its contradicted spans: `concept_only`, `detail_only`, `both`, or `unclassified_only`
  (every contradicted span is `unknown`). Spans that are `unknown` alongside classified ones are
  ignored for bucketing. Also reported: `concept_rate = concept_only + both` and
  `detail_rate = detail_only + both`.
  - Identity, tested: `concept_only + detail_only + both + unclassified_only = hallucination_rate`.
  - **Coverage requirement:** if a hallucinated readout's cell has no classify result (failed
    call), all of its contradicted spans are `unknown`, so the identity still holds.
- **Severity.**
  - `major_rate` = specific readouts with at least one contradicted span of severity 3 ÷
    specific readouts.
  - `mean_max_severity` = the mean, over hallucinated readouts with at least one classified
    contradicted span, of the maximum severity. Readouts whose spans are all `unknown` are
    excluded and counted.
- **Unsupported.** Denominator **T** = verify-tallied specific readouts; the coverage T ÷
  specific is reported.
  - `unsupported_rate` = readouts with at least one `unsupported` span ÷ T.
  - Its concept and detail split: readouts with at least one unsupported concept (or detail)
    span ÷ T.
  - The split by `unsupported_kind`: readouts with at least one span of that kind ÷ T, plus the
    span-level share of each kind among unsupported spans.
  - `unsupported_rate_lenient`: the same as `unsupported_rate`, but excluding `inference` and
    `world_knowledge` spans.
- **Disputed.** `disputed_rate` = readouts with at least one disputed span ÷ T.
- **Strict and lenient rates.**
  - `strict_hallucination_rate` is the share of **verify-tallied** specific readouts with at
    least one contradicted, disputed or unsupported span (the user's definition).
  - `lenient_hallucination_rate` is the same with the lenient unsupported view. Spans with
    `unsupported_kind = unknown` are kept, which is conservative.
  - Both are reported next to their verify coverage (tallied ÷ specific), so a failed verify
    call cannot silently lower them.
  - `unsupported_rate` counts `unsupported` spans only; disputed spans are counted separately.
  - **Strict concept/detail decomposition.** This answers issue #1 under the user's definition.
    Readouts in T are bucketed exactly as in the contradicted decomposition (`concept_only` /
    `detail_only` / `both` / `unclassified_only`), using all their contradicted, disputed and
    unsupported spans. This gives `strict_concept_rate` and `strict_detail_rate`, with the
    identity `strict_concept_only + strict_detail_only + strict_both + strict_unclassified_only
    = strict_hallucination_rate`, which is tested. The same is done for the lenient view.
- **Span-level distributions.** Counts and shares by grounding × type × target × severity ×
  unsupported_kind.
- **Judge health.** These counts are reported:
  - `n_cells`, `n_missing`, `n_empty_cells`;
  - `n_failed` per stage;
  - `n_unverified_spans`, `n_unverified_claim_quotes`;
  - `n_unclassified`, `n_cells_claims_unjudged`;
  - spend in USD and tokens per stage.

**Token arms.** The summariser is told to "commit to the most specific reading", so it produces
unsupported specifics by design. The report shows token arms in a **separate table** with a
caveat, and does not rank their unsupported, strict or lenient rates alongside the prose arms.
Contradicted-span metrics are shown for both groups.

`summary.json` for each arm holds all of the above plus the config: judge models, prompt
versions, the source commit of the bank, and the readouts path with its sha256.

## Calibration (gate before the full run)

Calibration tests the **classifier only**, since stages 1 and 2 are the unchanged instrument.
A case is a prompt, a marked response, 1–3 readouts, and a list of spans with grounding and gold
labels (`type`, `target`, `severity`, `unsupported_kind`), exactly what the classifier sees in
production.

- **`dev.jsonl`** (synthetic, at least 40 spans) is written by the implementer from scratch and
  used for prompt iteration. Coverage:
  - The two anchor examples from issue #1: Picasso for da Vinci (concept) and Steve for an
    arbitrary Joe (detail).
  - At least 12 concept and 12 detail spans, covering all 7 targets.
  - At least 8 rule cases:
    - a real entity as the subject of the claim (rule 2) versus in a passing mention (rule 3);
    - an arbitrary protagonist renamed (detail) versus the protagonist confused with another
      character in the text (concept), and an arbitrary protagonist swapped for a famous real
      name (concept);
    - a value that flips the claim versus one that does not;
    - a magnitude error that stays detail.
  - At least 10 unsupported spans covering all 4 `unsupported_kind` values, with both
    concept-type and detail-type unsupported spans.
  - Severity 1, 2 and 3 each appear at least 5 times.
  - Readout styles that mimic real lenses: O-lens bullet lists with code-switched Chinese and
    English fragments, NLA `<explanation>` prose, and one-to-two-sentence summariser
    interpretations.
- **`heldout.jsonl`**, used only for the gate, has two parts:
  - (a) at least 30 further synthetic spans, written and gold-labelled by a **fresh-context
    subagent** that never sees the classifier prompt or the dev results. These labels are
    committed, frozen, before the first held-out check, together with the part (b) labels.
  - **Held-out minimums**, across parts (a) and (b): at least 20 unsupported spans with at least
    4 per `unsupported_kind`, at least 10 spans per severity level, and at least 4 disputed
    spans.
  - Every gate metric is reported with a Wilson 95% CI.
  - (b) at least 30 **real spans**. To produce these:
    1. Run a pilot of span + verify on 40 cells each from s3d-rl600 L44, nla-rl-L42 and jlens L44.
    2. Use `hbench pilot-dump` to sample contradicted / disputed / unsupported spans from that
       pilot, stratified by arm **and grounding**. Keep sampling until the real part has at
       least 8 concept and 8 detail gold spans.
    3. The main session hand-labels them **blind to classifier output, before writing the
       classifier prompt**. It commits the labels, frozen, before the first held-out check.

  Caveat to record in the calibration README: the gold labeller (Claude) is the same model
  family as the classifier, so agreement is optimistic. A 30-span sample goes into
  `audit_samples.md` for Camila to spot-check.
- **`hbench calibrate --split dev|heldout`** reports:
  - type accuracy, and recall for concept and for detail;
  - target accuracy;
  - severity: exact, within 1, and quadratic-weighted κ;
  - `unsupported_kind` accuracy;
  - a confusion matrix per label;
  - the run-to-run label flip rate across 2 runs. Gate metrics are the **mean of the 2 runs**;
  - on `dev` only, every disagreement, with the classifier's `why`. On `heldout`, aggregates
    only (no per-span lists), so held-out errors cannot leak into prompt tuning.
- **Gate, on held-out only:**

  | measure | threshold |
  |---|---|
  | type accuracy | ≥ 85% |
  | concept recall | ≥ 80% |
  | detail recall | ≥ 80% |
  | severity exact | ≥ 65% |
  | severity quadratic-weighted κ | ≥ 0.6 |
  | target accuracy | ≥ 70% |
  | `unsupported_kind` accuracy (unsupported spans only) | ≥ 70% |
  | `n_unclassified` | 0 |
  | type flip rate between the 2 runs | ≤ 5% |
  | type accuracy on the **real** part alone | ≥ 80% |

- **If the gate fails:**
  - Iterate on dev: revise the prompt and bump the version to `taxonomy-v1.1`, and so on.
  - Re-check held-out at most **3 times**. Every check is logged in the calibration README.
  - Gold labels change only when one is shown to be wrong. The fix and its reason are recorded.
  - If held-out still fails after 3 checks, stop and report rather than tune further.
- **Results doc.** Report the final calibration numbers there.

## Run

1. **Pilot** (as above), then calibration.
2. **Stage 1, layer-matched slice.** All 11 arms at one layer each: L44 for the 5-layer arms and
   L42 for the single-layer arms (nla-*, sae-l42). This is 11 × 1,123 ≈ 12.4k cells, with up
   to 4 calls per cell.
   - Before starting, run a 20-cell slice per arm. It prints prompts, checks parsing, and
     estimates the cost of stage 1 and stage 2 from measured spend. Report the estimate.
   - The slice also fails loudly, rather than counting every call as failed, if keyless `ant`
     auth or structured output does not work.
     There is no budget cap, but spend is always reported.
3. **Stage 2.** The remaining layers (20, 36, 48, 60) of the 5-layer arms, about 35.9k cells.
4. **Driver.** `scripts/run_all_arms.sh` runs stage 1, then `hbench report`, then stage 2, then
   `hbench report`.
   - It is one tmux session `hbench`, one `hbench run` process per stage, logging to
     `logs/run_<timestamp>.log`.
   - Re-running the script resumes from the caches.
   - It respects the 32 GB box RAM cap: it streams `cells.jsonl` and does not hold all prompts
     in memory.
5. **Report.** `hbench report` writes `outputs/wsbench_chat/<model>/report.{md,json}`:
   - one row per arm and per arm × layer, with every metric above, prose and token tables
     separate;
   - `audit_samples.md`, with 15 random spans per (type × grounding) across arms, with context,
     for human spot-checks.

## Deliverables / acceptance

- `uv run pytest`, `uv run ruff check` and `ruff format --check` all pass.
- `test_v5c_frozen.py` proves the span prompt is byte-identical to the frozen JSON and the
  verify prompt is identical to the wsbench-pub literal; port the source's verify literal test.
- The calibration gate is passed on held-out and reported.
- Stage 1 is complete for all 11 arms, with missing + failed cells under 1% per arm per stage,
  and the report is written. Stage 2 is either complete, or running in tmux with a resumable
  driver.
- `docs/results/0001_wsbench_chat_taxonomy.md` contains:
  - the headline tables per arm, each rate with a CI: v5c rate, concept rate, detail rate,
    strict concept rate, strict detail rate, major rate, unsupported rate (strict and lenient),
    and strict and lenient rates;
  - the concept/detail share of contradicted spans;
  - the target mix;
  - a handful of quoted examples per type;
  - the calibration numbers;
  - spend;
  - caveats: labels are single samples, since the Anthropic route sends no temperature; `concept`
    for unsupported spans (invented events and stances) extends the user's "entirely
    contradictory claims" wording; the judge differs from the WorkspaceBench pin; token arms are judged through a
    summariser, so their labels describe the summary; the gold labeller is the same model family
    as the classifier.
- README (taxonomy, quickstart, results pointer) and CLAUDE.md are written. CLAUDE.md covers
  layout, conventions, and gotchas: keyless `ant` auth, the readouts location, cache/resume, the
  single-process pacer, and the 32 GB RAM cap.
