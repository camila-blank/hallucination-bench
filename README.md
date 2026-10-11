# HallucinationBench v2

A suite of evals measuring the **types and severities** of hallucination produced by
activation-reading lenses (O-lens, NLA, J-lens, logit lens, SAE, Patchscopes, ...).

Supersedes the archived `camilablank/hallucination-bench-v1`. The WorkspaceBench
hallucination eval is one of many evals here; see `plans/` for the design history.

## Evals

| eval | what it measures | plan |
|---|---|---|
| `wsbench_chat` | WorkspaceBench's hallucination (chat) eval, ported verbatim (v5c span + verify stages), plus the taxonomy classifier | [0001](plans/0001-bootstrap-wsbench-chat-taxonomy.md) |

Per cell (arm × item × read site × layer) `wsbench_chat` runs up to four calls:

1. **summarise** (token lenses only): the `interp-v1` summariser turns the decoded top-10 token
   bag into one interpretation, which is judged as the readout (k = 1);
2. **span** (`v5c-chat`, verbatim): wrong / off-topic spans per readout and a class per readout
   (`hallucinated` / `off_topic` / `consistent` / `generic`);
3. **verify** (`v5c-chat-verify-v1`, verbatim): every other specific claim as `true` /
   `unverifiable` / `disputed` / `off_topic`;
4. **classify** (`taxonomy-v1`, new): labels every hallucinated span found by 2–3. It never
   decides whether a span is a hallucination.

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

scripts/run_all_arms.sh                   # full run in tmux session `hbench`; tail -f logs/run_*.log
```

Other commands: `hbench run --rebuild` (regenerate `cells.jsonl` / `summary.json` from the cache,
no API calls), `hbench run --skip-classify` (span + verify pilot), `hbench pilot-dump` (sample
pilot spans as unlabelled calibration cases).

Outputs: `outputs/wsbench_chat/<judge model>/<arm>/{cache.jsonl,cells.jsonl,summary.json}` and
`outputs/wsbench_chat/<judge model>/{report.md,report.json,audit_samples.md}`.

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
(filled in after the full run).
