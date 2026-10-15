# HallucinationBench v2

A suite of evals measuring the **types and severities** of hallucination produced by
activation-reading lenses (O-lens, NLA, J-lens, logit lens, SAE, Patchscopes, ...).

Supersedes the archived `camilablank/hallucination-bench-v1`. The WorkspaceBench
hallucination eval is one of many evals here; see `plans/` for the design history.

## Evals

| eval | what it measures | plan |
|---|---|---|
| `wsbench_chat` | WorkspaceBench's hallucination (chat) eval, ported verbatim (v5c span + verify stages), plus the taxonomy classifier | [0001](plans/0001-bootstrap-wsbench-chat-taxonomy.md) |
| `wsbench_chat` claim map | taxonomy v2: every character of every readout in exactly one of 11 scoring leaves (7 groups), rolled up into 6 verdicts | [0002](plans/0002-claim-map-taxonomy-v2.md) |
| sibling gate (rev 21) | `substitution_sibling` counts as hallucination only when unambiguous: 10-sample lens resample (sibling rarely returns while other items or the right answer do), then whitened-DiffMean probes for stable siblings | [0009](plans/0009-sibling-resample-probe-gate.md) |

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
`CHOSEN`) and rendered into the judge prompt. Rev 16 (the lean tree): **12 leaves** —
`non_claim` (tag `reason`), `correct` (tag `source`: prefix / after / entailed), `detail`,
`false_claim`, `substitution_sibling`, `substitution_category`, `model_state`, `unverifiable`
(tags `same_slot_alternative`, `world`, `prediction`), `supplementary_true`,
`supplementary_false`, `ambiguous_topic`, `off_topic` — in 8 groups, plus two code-only rows:
`unknown_incorrect` (`--anchor v5c` only) and `failed`. Tags are recorded but never change a
verdict. The hallucination gate: a segment is incorrect only if the conversation **obviously
contradicts** it; predictions get no category of their own (contradicted → incorrect, else
`unverifiable` with `prediction = true`); content realised in any wording is `correct`. The
test for world claims is **settlement, not subject** (text silent → `supplementary_*`).
Each leaf maps to one **verdict** (`taxonomy_v2.VERDICTS`): hallucination (detail,
false_claim), probe_candidate (substitutions, model_state, supplementary_false,
ambiguous_topic), correct, unresolved (unverifiable), junk, non_claim. The headline is the
character-weighted verdict share per readout and the readout-level verdict rates, with
**`false_claim` and `detail` always reported separately** next to the hallucination verdict;
groups and leaves are the breakdown. Probe-candidate segments carry a `probe_target`
(`probe_candidates.jsonl`). Invalid `:M` answers are retried up to 2 times in-run before
being cached as failed. Outputs are labelled **pre-gate** until the held-out gate passes for
the current prompt version. `hbench consistency-v2 --arms s3d-rl600:44,nla-rl-L42:42,jlens:44
--n 50 --seed 0` runs `:M` twice per sampled cell and reports κ, raw agreement, the between-run
leaf confusion, per-leaf self-agreement and the leaf pairs swapped in > 10% of a leaf's units
(`outputs/consistency/<model>/`). Rev 17: every incorrect segment must quote the
conversation span that contradicts it (`contradicted_by`) and every `correct` segment (except
`source = entailed`) the span that supports it (`supported_by`); a missing or unfindable quote
fails closed to `unverifiable` (counted). The quotes are carried into `cells.jsonl`,
`probe_candidates.jsonl`, the audit samples and the report's hallucination examples. Reports
include a judge-noise section and a "judge noise ±" column (half the r0 − r1 difference of each
headline metric in the consistency sample). Rev 18: `non_claim` and `off_topic` are merged into
one scoring leaf, `junk` (verdict junk, tag `reason` incl. `off_topic`), right after parsing;
the claimmap-v2.7 prompt and cached answers are unchanged, so re-scoring costs nothing. Reports
include a junk-by-reason table (off-topic sub-share per arm). Rev 19 (also after parsing, $0):
a segment the judge tags `hedged` (a list of options, "Germany or France or UK") is scored
`unverifiable` with `same_slot_alternative`, whatever its leaf; `model_state` is its own verdict,
`thinky` (verdicts: hallucination / probe_candidate / thinky / correct / unresolved / junk).
Rev 20 (scoring, $0): `substitution_category` is the third hallucination component, reported
separately next to `false_claim` and `detail`; `substitution_sibling` stays a probe candidate.

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
[`docs/results/0002_claim_map.md`](docs/results/0002_claim_map.md). Sibling gate (rev 21): [`docs/results/0009_sibling_gate.md`](docs/results/0009_sibling_gate.md),
re-scored report [`docs/results/0002_report_full_v2.8.md`](docs/results/0002_report_full_v2.8.md).
