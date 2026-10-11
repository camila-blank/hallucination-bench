# CLAUDE.md — HallucinationBench v2 agent runbook

Suite of evals for the **types and severities** of hallucination in activation-reading lenses.
First eval: `wsbench_chat` (plan `plans/0001-bootstrap-wsbench-chat-taxonomy.md` is the spec).

## Layout

```
src/hallucination_bench/
  llm.py          one judge client per process: Anthropic (claude-*) / OpenRouter, retries,
                  process-wide RPM pacer, one asyncio semaphore, per-call Usage (tokens, USD)
  cache.py        append-only JSONL cache {key, fingerprint, ts, result, meta}; failed = result None
  readouts.py     readout contract loader, decode_token, verify_quote, normalize_ws
  summarizer.py   interp-v1 token-bag -> prose prompt (verbatim)
  taxonomy.py     THE taxonomy (single source of truth; rendered into the classifier prompt)
  judges/v5c.py   verbatim v5c span + verify prompts/parsers; claim_lists (quotes, checked vs tally)
  judges/taxonomy_v1.py  classifier prompt, schema, validate, fail-closed parse
  evals/wsbench_chat/{bank,run,score}.py   bank + grid; pipeline; metrics + bootstrap
  calibrate.py    classifier vs gold cases, Wilson CIs, the held-out gate
  report.py       cross-arm report, audit samples, pilot-dump
  cli.py          hbench run|calibrate|report|pilot-dump
data/wsbench_chat/   verbatim copies from wsbench-pub @ 92d763e (never edit)
data/calibration/    dev.jsonl (implementer), heldout.jsonl (blind author only), README (schema + log)
scripts/run_all_arms.sh  tmux driver for the full run
```

## Conventions

- `uv` for everything: `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`.
- The v5c span/verify literals are frozen: `judges/v5c.py` asserts the span literals equal
  `data/wsbench_chat/judge_prompt_v5c_chat.json` at import; tests check the verify literals
  against the wsbench-pub source (`/workspace/root/wsbench-pub`, skipped if absent) and the
  data README. Never edit them; a classifier change bumps `taxonomy_v1.PROMPT_VERSION`
  (`taxonomy-v1.1`, ...), which re-queues only `:C` cache entries.
- Label definitions live only in `taxonomy.py`. Tests assert the prompt contains all of them.
- Held-out calibration: never read per-span held-out results into prompt tuning (the CLI only
  prints aggregates for `--split heldout`); at most 3 held-out checks, each logged in
  `data/calibration/README.md`. Do not create or edit `data/calibration/heldout.jsonl` unless
  you are the designated blind author.
- Modified pre-existing tests are a red flag in review: a test change must be sanctioned by the plan.

## Run commands

```bash
R=/workspace/wsbench-paper-data/readouts
uv run hbench run --out outputs/smoke --readouts-root $R --arms s3d-rl600,nla-rl-L42,jlens \
    --layers 42,44 --limit 5 --show-prompts 5          # smoke; prompts -> <arm>/prompts_sample.txt
uv run hbench run --rebuild --readouts-root $R --arms jlens --out outputs   # no API calls
uv run hbench report --out outputs
uv run hbench calibrate --split dev
uv run hbench run ... --skip-classify  &&  uv run hbench pilot-dump --arms a,b --dest x.jsonl
scripts/run_all_arms.sh        # tmux session `hbench`, log logs/run_<ts>.log; re-run resumes
```

## Gotchas

- **Keyless Anthropic auth.** With `ANTHROPIC_API_KEY` unset, `AsyncAnthropic()` picks up the
  `ant` CLI OAuth profile (`ant auth status`). Never require the env var. If the profile's
  credentials lapse mid-run, the 401 raises `JudgeConfigError` (exit 3): re-login with
  `ant auth login` and re-run (it resumes).
- **Opus 5.5 rejects `temperature` and `thinking: disabled`** (400). Calls send neither and set
  `output_config.effort` (default `low`) plus `output_config.format` json_schema. Labels are
  single samples.
- **Readouts** are inputs, not in git: `/workspace/wsbench-paper-data/readouts/<arm>/hallucination.jsonl`
  (11 arms; `sae-l42-top20` has no hallucination file). Each arm-layer = 1,123 cells.
  `--layers 42,44` makes each arm judge the layers it has (5-layer arms L44, nla/sae L42).
- **Cache / resume.** Per-arm `outputs/wsbench_chat/<model>/<arm>/cache.jsonl`. Keys
  `summ:{key}`, `{key}`, `{key}:V`, `{key}:C`, `key = {id}__L{layer:03d}__p{pos}`;
  fingerprint = (prompt version, model, effort, system, user). Failed / invalid answers are
  cached with `result: null` (+ `meta.raw`) and re-queued every run. `cells.jsonl` and
  `summary.json` are always regenerated from the cache (cache-only replay of the same
  pipeline), so a stale `:C` row is never attached to new S-ids.
- **`--limit`** slices cells per arm × layer; the rest show as `pending` in the summary.
- **One process for all arms**: one RPM pacer (`--rpm`, default 1500) and one semaphore
  (`--concurrency`, default 64). Do not launch parallel `hbench run` processes on the same arm
  (two writers on one cache file) or to "go faster" (they would each get the full rate).
- **32 GB RAM cap** (docker cgroup; `free` lies). Rebuild streams `cells.jsonl`; only slim
  label records are kept for scoring. OOM shows up as rc 137.
- Calibration cache: `outputs/calibration/<model>/cache.jsonl`, keys `{split}:{case_id}:r{n}`
  (each repeat is a separate call).
