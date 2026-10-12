# CLAUDE.md — HallucinationBench v2 agent runbook

Suite of evals for the **types and severities** of hallucination in activation-reading lenses.
First eval: `wsbench_chat` (plan `plans/0001-bootstrap-wsbench-chat-taxonomy.md` is the spec);
the claim map / taxonomy v2 is plan `plans/0002-claim-map-taxonomy-v2.md`.

## Layout

```
src/hallucination_bench/
  llm.py          one judge client per process: Anthropic (claude-*) / OpenRouter, retries,
                  process-wide RPM pacer, one asyncio semaphore, per-call Usage (tokens, USD)
  cache.py        append-only JSONL cache {key, fingerprint, ts, result, meta}; failed = result None
  readouts.py     readout contract loader, decode_token, verify_quote, normalize_ws
  summarizer.py   interp-v1 token-bag -> prose prompt (verbatim)
  taxonomy.py     THE taxonomy (single source of truth; rendered into the classifier prompt)
  taxonomy_v2.py  THE claim-map taxonomy (judge: 12 leaves / 8 groups; scoring, rev 18-19: 11 leaves /
                  7 groups / 6 VERDICTS (junk = non_claim + off_topic; model_state = thinky); leaf
                  tags, precedence, open
                  decisions in CHOSEN; render(anchored) for single-call vs --anchor v5c)
  units.py        deterministic readout splitter (units R{i}.U{n}) + offset-returning locate
  judges/v5c.py   verbatim v5c span + verify prompts/parsers; claim_lists (quotes, checked vs tally)
  judges/taxonomy_v1.py  classifier prompt, schema, validate, fail-closed parse
  judges/claimmap_v2.py  claim-map prompt, schema, validate, parse -> segments (forced/disputed)
  evals/wsbench_chat/{bank,run,score}.py   bank + grid; pipeline; metrics + bootstrap
  evals/wsbench_chat/score_v2.py   claim-map metrics (char-weighted shares, rates, comparisons)
  calibrate.py    classifier vs gold cases, Wilson CIs, the held-out gate
  calibrate_v2.py claim map vs unit-level gold (dev_v2 / heldout_v2), the v2 gate
  report.py       cross-arm report, audit samples, pilot-dump
  report_v2.py    `report --view v2`, probe_candidates.jsonl, audit_samples_v2.md, pilot-dump-v2
  consistency_v2.py  `consistency-v2`: two fresh :M calls per sampled cell, κ / confusion / flags
  cli.py          hbench run|calibrate|report|consistency-v2|pilot-dump|pilot-dump-v2|units-v2
data/wsbench_chat/   verbatim copies from wsbench-pub @ 92d763e (never edit)
data/calibration/    dev.jsonl (implementer), heldout.jsonl (blind author only), README (schema + log)
                     dev_v2.jsonl (implementer), heldout_v2.jsonl (blind author only)
scripts/run_all_arms.sh  tmux driver for the full run
scripts/build_dev_v2.py  writes dev_v2.jsonl (labels live here; re-run if the splitter changes)
```

## Conventions

- `uv` for everything: `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`.
- The v5c span/verify literals are frozen: `judges/v5c.py` asserts the span literals equal
  `data/wsbench_chat/judge_prompt_v5c_chat.json` at import; tests check the verify literals
  against the wsbench-pub source (`/workspace/root/wsbench-pub`, skipped if absent) and the
  data README. Never edit them; a classifier change bumps `taxonomy_v1.PROMPT_VERSION`
  (`taxonomy-v1.1`, ...), which re-queues only `:C` cache entries.
- Label definitions live only in `taxonomy.py` / `taxonomy_v2.py`. Tests assert the prompt
  contains all of them. A claim-map prompt change bumps `claimmap_v2.PROMPT_VERSION`
  (re-queues only `:M`); switching an open decision is one line in `taxonomy_v2.CHOSEN` (bump
  `TAXONOMY_VERSION` too).
- The v2 gold uses the production unit split: changing `units.py` invalidates gold (the loader
  re-splits and rejects mismatches). Re-run `scripts/build_dev_v2.py`, never hand-edit ids/texts.
  Do not create `data/calibration/heldout_v2*.jsonl` unless you are the blind author.
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
uv run hbench run ... --stages map     # claim map, single-call (summary first; span on 10% subsample)
uv run hbench run ... --stages map --anchor v5c   # plan rev 12: v5c spans as ESTABLISHED units
uv run hbench report --out outputs --view v2
uv run hbench calibrate --taxonomy v2 --split dev
uv run hbench consistency-v2 --arms s3d-rl600:44,nla-rl-L42:42,jlens:44 --n 50 --seed 0
uv run hbench pilot-dump-v2 --arms a,b --dest x.jsonl   # real cells + production units, unlabelled
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
  (v2: `v2:{split}:{case_id}:r{n}`; each repeat is a separate call).
- **`--stages`**: unselected stages are read from the cache only. The Python API default
  (`run_arms_async(stages=None)`) keeps plan 0001's stages (no map) so v1 tests and callers are
  unchanged; the CLI default is every stage. `hbench run` exits 4 if ANY stage has cached
  failures, including unselected ones (e.g. s3d's 5 old verify failures under `--stages map`).
- **`:M` cost** is output-heavy (NLA cells ~3.7k output tokens). The system prompt is ~6.6k
  tokens and cached; a burst of concurrent first calls all pay the cache write.
- `summary_v2.json` is written next to `summary.json` by every rebuild once any cell is mapped
  (only layers with `:M` rows enter it).
- **Claim-map mode** (plan rev 13) lives in `Models.map_anchor` (`none` = single-call default,
  `v5c` = rev 12) and is written to `arm.json`; `report` rebuilds with it (old `arm.json`
  without the key = `v5c`). Both modes share the `:M` key; the fingerprints differ, so their
  cache rows coexist. Single-call cells keep their map even when the span stage has no result
  (status `pending`); the v5c comparison covers only readouts with a span verdict.
- **Pre-gate**: every v2 output (`summary_v2.json` config, report, probe_candidates, audit
  samples) says pre-gate until `outputs/calibration/<model>/heldout_<PROMPT_VERSION>.json` has
  `gate_passed: true`.
- Old gold leaf names (every leaf of revs 12–15) are merged on load into the rev-16 leaves
  (`taxonomy_v2.LEAF_RENAMES`), with the carried tags under `tags` / `rest_tags`
  (`MERGE_TAGS`). `scripts/build_dev_v2.py` keeps the fine labels and writes merged gold.
- Consistency cache: `outputs/consistency/<model>/cache.jsonl`, keys `cons:{arm}:{cell}:r{n}`
  (the repeat index is in the fingerprint). Token arms need their `summ:` rows cached.
- **Quoted evidence** (rev 17): incorrect segments need `contradicted_by`, `correct` ones
  (except `source = entailed`) `supported_by`, located in prompt + response (marker removed,
  `claimmap_v2.conversation_text`). `parse(raw, units, conversation)` fails closed to
  `unverifiable` (`judge_leaf` keeps the judge's leaf; counters `n_unquoted_contradictions`,
  `n_unquoted_support`). Without `conversation` the check is skipped (unit tests only); every
  production caller (run, calibrate, consistency) passes it.
- **Judge noise** (rev 17): `report --view v2` reads
  `outputs/consistency/<model>/consistency_<PROMPT_VERSION>.json` for the "judge noise ±" column
  and the noise section; run `consistency-v2` on the run's arms first, else they show —.
- **Rev 18 junk merge is post-parse**: the judge enum/prompt (`JUDGE_LEAF_NAMES`, claimmap-v2.7)
  is unchanged; `claimmap_v2.parse` maps `non_claim` / `off_topic` to `junk` (reason kept /
  `off_topic`). Never render `LEAF_NAMES` into the prompt or schema (that would re-queue every
  `:M` answer). `consistency-v2 --rescore` re-parses the cached r0/r1 answers with the current parser
  (prompt rebuilt from stored unit texts + bank) with no API calls.
- **Rev 19, also post-parse**: `hedged` segments become `unverifiable` (+ `same_slot_alternative`,
  `raw_leaf` kept, `n_hedged_remapped`; health splits by origin `n_hedged_from_*`);
  `model_state` is the `thinky` verdict (not a probe candidate).
- **No git writes from agents** in this worktree (no add/rm/restore --staged/commit).
- **In-run retries** (rev 14): an invalid `:M` (or calibration) answer is re-asked up to 2 times
  (`run.MAP_INVALID_RETRIES`) before it is cached as failed; meta records `n_invalid_attempts`
  and the summed usage.
