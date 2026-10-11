# HallucinationBench v2

A suite of evals measuring the **types and severities** of hallucination produced by
activation-reading lenses (O-lens, NLA, J-lens, logit lens, SAE, Patchscopes, ...).

Supersedes the archived `camilablank/hallucination-bench-v1`. The WorkspaceBench
hallucination eval is one of many evals here; see `plans/` for the design history.

## Evals

| eval | what it measures | plan |
|---|---|---|
| `wsbench_chat` | WorkspaceBench's hallucination (chat) eval, ported verbatim (v5c span + verify stages), plus the taxonomy classifier | [0001](plans/0001-bootstrap-wsbench-chat-taxonomy.md) |
| `wsbench_chat` claim map | taxonomy v2: every character of every readout in exactly one of 22 leaves (8 groups), rolled up into 7 verdicts | [0002](plans/0002-claim-map-taxonomy-v2.md) |
| `sibling_probe` | are same-category "sibling" mentions (user Korean, readout says China) hallucinations? One DiffMean probe per concept checks whether the mentioned sibling is in the activation; per-arm probe-verified hallucination rate (s3d O-lens, NLA, J-lens) | [0003](plans/0003-sibling-concept-probes.md) |

Per cell (arm × item × read site × layer) `wsbench_chat` runs up to five calls:

1. **summarise** (token lenses only): the `interp-v1` summariser turns the decoded top-10 token
   bag into one interpretation, which is judged as the readout (k = 1);
2. **span** (`v5c-chat`, verbatim): wrong / off-topic spans per readout and a class per readout
   (`hallucinated` / `off_topic` / `consistent` / `generic`);
3. **verify** (`v5c-chat-verify-v1`, verbatim): every other specific claim as `true` /
   `unverifiable` / `disputed` / `off_topic`;
4. **classify** (`taxonomy-v1`, new): labels every hallucinated span found by 2–3. It never
   decides whether a span is a hallucination;
5. **map** (`claimmap-v2.3`, plan 0002 rev 13): code splits every readout into units; one
   judge call puts every unit, or verbatim sub-spans of it, into one taxonomy-v2 leaf; code
   rolls the leaves up. Single-call by default (needs only stage 1; no established spans, no
   verify); `--anchor v5c` restores rev 12 (the v5c verified wrong spans become their own
   "established" units; needs stage 2).

`hbench run --stages summary,span,verify,classify,map` picks the stages that may call the API
(default: all; dependencies run first if missing: `--stages map` runs summary, plus span on a
seeded 10% item subsample for the v5c comparison; under `--anchor v5c` span on every cell).

## The claim map (taxonomy v2)

Defined once in [`src/hallucination_bench/taxonomy_v2.py`](src/hallucination_bench/taxonomy_v2.py)
(leaves, groups, precedence, the Detail rule, tags, and the open decisions with their defaults in
`CHOSEN`) and rendered into the judge prompt. Three top branches (rev 13, option A):
non-claim / claim about the text / supplementary claim; the test is **settlement, not
subject** (an on-topic world claim the conversation does not state, entail or contradict is
supplementary). Eight groups: **hallucination** (`detail`, `false_claim`), **co-activation**
(`substitution_sibling`, `substitution_category`), **correct** (`restates_prefix`,
`stated_after`, `entailed`), **supplementary** (`supplementary_true`, `supplementary_false`,
`supplementary_unsettled`), **thinky** (`unrealised_option`, `unrealised_prediction`,
`model_state`, `unsupported_assertion`, `other_on_topic`), **unsettled** (`ambiguous_topic`), **junk** (`off_topic`), **non-claim**
(`generic`, `filler`, `formatting`, `junk_text`, `lens_meta`), plus two code-only rows:
`unknown_incorrect` (`--anchor v5c` only: an established v5c span the judge disputed or did
not label incorrect) and `failed`. Each leaf maps to one **verdict** (`taxonomy_v2.VERDICTS`):
hallucination, probe_candidate, alternative, correct, unresolved, junk, non_claim. The headline is the
character-weighted verdict share per readout (averaged over readouts) and the readout-level
verdict rates; groups and leaves are the breakdown. Probe-candidate segments carry a
`probe_target` (written to `probe_candidates.jsonl`). Rev 15 hallucination gate: a segment is
incorrect only if the conversation **obviously contradicts** it; uncontradicted invented content
is `unsupported_assertion` (unresolved); unrealised continuations that are not contradicted are
`unrealised_option` (same open-ended slot) or `unrealised_prediction` (a different move or
answer), both verdict `alternative`; content realised in any wording is `stated_after`. Invalid
`:M` answers are retried up to 2 times in-run before being cached as failed. Outputs are labelled **pre-gate** until
the held-out gate passes for the current prompt version.

## The taxonomy (v1)

Defined once in [`src/hallucination_bench/taxonomy.py`](src/hallucination_bench/taxonomy.py)
and rendered into the classifier prompt.

- **grounding** (set in code): `contradicted` (verified v5c wrong span), `disputed` (verify
  thinks the response rules it out, v5c did not flag it), `unsupported` (verify
  `unverifiable`: specific, but neither stated, entailed nor realised).
- **type** — would a reader who believed the span misunderstand what the text is about or
  claims? `concept` (yes: a different entity / idea / event as the subject of a claim, or a
  reversed claim; e.g. "Picasso" for da Vinci) vs `detail` (no: a wrong or invented specific
  value inside a correct understanding; e.g. "Steve" for an arbitrary character named Joe).
  Names follow an identity-first decision order; a value that flips the claim is `concept`;
  magnitude alone raises severity, not type.
- **target**: entity, quantity, event, attribute, stance, structure, topic.
- **severity**: 1 minor, 2 moderate, 3 major.
- **unsupported_kind** (unsupported spans only): prediction, inference, fabricated,
  world_knowledge. The **lenient view** drops inference and world_knowledge; the **strict
  view** (the default definition: only stated or realised content is grounded) keeps all four.

## Quickstart

```bash
uv sync                                   # Python >= 3.12; dev tools come with the dev group
uv run pytest && uv run ruff check && uv run ruff format --check

# smoke: 5 cells each of three arms (keyless Anthropic auth via the `ant` CLI profile)
R=/workspace/wsbench-paper-data/readouts
uv run hbench run --out outputs/smoke --readouts-root $R --arms s3d-rl600,nla-rl-L42,jlens \
    --layers 42,44 --limit 5 --show-prompts 5
uv run hbench report --out outputs/smoke

uv run hbench calibrate --split dev       # classifier vs hand-labelled dev cases
uv run hbench calibrate --split heldout   # the gate (held-out only)

# claim map (plan 0002): the :M stage on cached span results, then the v2 report
uv run hbench run --out outputs --readouts-root $R --arms s3d-rl600 --layers 44 --limit 10 \
    --stages map --show-prompts 3
uv run hbench report --out outputs --view v2   # report_v2.md, probe_candidates.jsonl, audit_samples_v2.md
uv run hbench calibrate --taxonomy v2 --split dev

scripts/run_all_arms.sh                   # full run in tmux session `hbench`; tail -f logs/run_*.log
```

Other commands: `hbench run --rebuild` (regenerate `cells.jsonl` / `summary.json` from the cache,
no API calls), `hbench run --skip-classify` (span + verify pilot), `hbench pilot-dump` (sample
pilot spans as unlabelled calibration cases), `hbench pilot-dump-v2` (real cells with their
production units, unlabelled, for v2 gold), `hbench units-v2` (fill a v2 case's `gold.units`
with the production unit split).

Outputs: `outputs/wsbench_chat/<judge model>/<arm>/{cache.jsonl,cells.jsonl,summary.json,summary_v2.json}`
and `outputs/wsbench_chat/<judge model>/{report.md,report.json,audit_samples.md}` (v2:
`report_v2.{md,json}`, `probe_candidates.jsonl`, `audit_samples_v2.md`).

## Judge and caveats

- All four calls default to `claude-opus-5-5` (overrides: `--judge-model`,
  `--summarizer-model`, `--classifier-model`). The WorkspaceBench pin is Gemini 3.8 Flash, so
  absolute v5c rates are **not comparable** to published WorkspaceBench numbers.
- The Anthropic route sends no `temperature` (newer Claude models reject it), so every label is
  a **single sample at the default temperature**. Opus 5.5 cannot disable thinking; calls omit
  `thinking` and send `output_config.effort = low` (`--effort`).
- Token lenses are judged through the summariser, which is told to "commit to the most specific
  reading": their labels describe the summary, and their unsupported / strict / lenient rates
  are reported in a separate table and not ranked against prose lenses.
- sae-l42 feature descriptions and template-v3 words are not byte-level BPE, so `decode_token`
  garbles any non-ASCII in them, and the summariser prompt calls them "token readouts"; this
  matches the source and is kept.
- `concept` for unsupported spans (invented events and stances) extends the issue's
  "entirely contradictory claims" wording.
- Calibration gold labels are written by Claude, the classifier's model family, so agreement
  is optimistic.

## Results

[`docs/results/0001_wsbench_chat_taxonomy.md`](docs/results/0001_wsbench_chat_taxonomy.md)
(filled in after the full run); claim map:
[`docs/results/0002_claim_map.md`](docs/results/0002_claim_map.md); sibling probes:
[`docs/results/0003_sibling_probes.md`](docs/results/0003_sibling_probes.md) (s3d sibling mentions 86% probe-unverified; real but small sibling signal, β₁ 0.27 z).
