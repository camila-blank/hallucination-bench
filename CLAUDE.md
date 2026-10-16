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
                  7 groups / 6 VERDICTS (junk = non_claim + off_topic; model_state = thinky;
                  rev 20: hallucination = false_claim + detail + substitution_category); leaf
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
  cli.py          hbench run|calibrate|report|consistency-v2|pilot-dump|pilot-dump-v2|units-v2|
                  sibling-gate <stage>|gate [import|snapshot|diff]
  evals/sibling_gate/   plan 0009 (rev 21): sibling gate. units (units, norm, item_sibling_names,
                  load_labels), resample (pools, trimmed manifests, job lines, collect, block
                  shape), fill_judge (sibfill-v1 + rung-1 rule), classes (sibclass-v1, dedupe,
                  sibconcepts-v1), texts (sibgen-chat-v2, checks, token_class key + per-class merge),
                  probes.py / um_probe.py (VENDORED verbatim from camila/um-ladder-v2; only the
                  um_probe import line differs), ladder (distractors, n/a, fire/nearby,
                  cross-method, matched null, labels), stages (the CLI stages)
data/wsbench_chat/   verbatim copies from wsbench-pub @ 92d763e (never edit)
data/calibration/    dev.jsonl (implementer), heldout.jsonl (blind author only), README (schema + log)
                     dev_v2.jsonl (implementer), heldout_v2.jsonl (blind author only)
scripts/run_all_arms.sh  tmux driver for the full run
scripts/build_dev_v2.py  writes dev_v2.jsonl (labels live here; re-run if the splitter changes)
scripts/sibling_gate/    run_all.sh (tmux `sibgate` driver, run from a /tmp copy), probes_modal.py
                         (Modal CPU pack/fit/score), plot_sibling_gate.py
data/sibling_gate/       fill_dev/gold, classes.json, concepts.json, probe_texts.jsonl, distractors.json
                         (plan 0010:) class_map.json (frozen raw class per (said, item) + raw ->
                         canonical names), probe_bank/<class>.jsonl.gz + cells.json (the text bank,
                         attempted cells), distractors/<class>.json (frozen per-unit sets + sha),
                         unit_costs.json (plan-0009 unit costs for the --plan estimate)
data/arms.json           plan 0010 arm registry: gated_arms, arms (cells/kind/family/layers/pool),
                         pools (generator + args, sampling, orig readouts, samples_per_run), env,
                         snapshots; arms.py reads it (units / resample constants are its views)
src/hallucination_bench/arms.py   the registry loader + views (`python -m ... arms bash`)
src/hallucination_bench/evals/sibling_gate/gate.py      `hbench gate` (plan / cost guard / stages /
                         exit codes / generation stamp / changes.md / diff / snapshot)
src/hallucination_bench/evals/sibling_gate/importer.py  `hbench gate import` (plan-0009 state)
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
uv run hbench gate --plan      # plan 0010: sibling gate work + cost; `hbench gate [--yes]` runs it
uv run hbench gate --add-arm mlnla-ck2000-all11 --plan
uv run hbench report --out outputs --view v2 --no-sibling-labels   # rev 20
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
- **Rev 20 (scoring)**: `substitution_category` is a hallucination component (verdict
  hallucination; its probe_target is dropped, `pair` kept, out of `probe_candidates.jsonl`).
  Every headline table, the v5c comparison, the judge-noise rows and the plots show the three
  components (false_claim / detail / substitution_category) separately.
- **Rev 21 (sibling gate, plan 0009)**: scoring-only. `taxonomy_v2.VERDICTS` / `VERDICT_ROWS` /
  `PROBE_CANDIDATE_LEAVES` / the leaf assert are UNCHANGED (rev 20); `SIBLING_GATED`
  (`substitution_sibling_gated`) lives only in `VERDICT_OF` / `GROUP_OF`, `score_v2.LEAF_ROWS`,
  `HALLUCINATION_LEAVES` / `HALLUCINATION_COMPONENTS` (4 components) and `FLAGS`. The join is in
  `score_v2.slim` (via `AccumulatorV2(sibling_labels=...)`, segment key `(cell key, readout,
  start, end)`): `ok` segments get `effective_leaf` on leaf AND alt view; disputed segments and
  agreement spans stay raw. Labels: `<out>/sibling_gate/labels/<arm>.jsonl` (`hbench run
  --rebuild` / `report` take `--sibling-labels`); no file = exact rev-20 scores
  (`tests/fixtures/score_v2_rev20.py` is the frozen rev-20 scorer the identity test compares
  against; never edit it). Raw-cell consumers (`probe_candidates`, `hallucination_examples`,
  audit samples, `scripts/plot_claim_hallucination_share.py`) must go through
  `sibling_gate.units.load_labels` + `seg_status` + `effective_leaf`. Judge side (consistency,
  calibrate, the claim-map parser, `hedged_origin`) reads raw leaves only. Units use the scored
  `leaf` (a hedged sibling remapped to `unverifiable` is not a unit). Import the unit module
  qualified (`from ...sibling_gate import units as sibling_units`): `hallucination_bench/units.py`
  is the readout splitter. `hbench sibling-gate units` streams the v2.7 cells
  (`--cells-root /workspace/hb-taxonomy-v2/outputs/wsbench_chat/claude-opus-5-5`); GPU / Modal
  stages only print their commands unless `--launch` (fit/score) or run from `run_all.sh`.
- **Rev 8.2 (sibling gate, full run)**: probes are fit per CLASS, not per layer
  (`evals/sibling_gate/probe_fit.py`, copied to the Modal image as `sibfit.py`; `probes_modal.py`
  `fit_class` = one container per class, `score_chunk` = (layer, 24 classes); `--phase pack` is a
  no-op). Only classes with >= 4 (said, item) pairs get probes (`concepts --min-class-pairs 4`);
  units in smaller classes are `probe_na` / `class_too_small`. The probe bank is captured in 32
  shards (`scripts/sibling_gate/capture_shards.sh`; `acts-sibgate-probe-sNN`, `fit --probe-acts`
  takes the comma list). `stages.probe_label` strips `/` from capture labels. The dedupe call
  returns only renamed classes (`classes.apply_merges`); `run_texts` runs classes concurrently.
  `scripts/sibling_gate/full_summary.py` writes `outputs/sibling_gate/full_summary.md`.
  The rev-20 rebuild for comparison: `hbench run --rebuild --sibling-labels <empty dir>`.
- **Plan 0010 gate pipeline (`hbench gate`)**: a GENERATION over the whole registered gate set
  (`data/arms.json` `gated_arms`, 14 plan-0009 arms; `--add-arm X` appends, `--arms` only
  narrows printing). Every run re-labels every gated arm; incrementality comes from frozen state:
  resample runs (`outputs/sibling_gate/resample/<pool>/runs/*.jsonl` + `state.json`; run0 =
  the plan-0009 `resample.jsonl`; new runs `gen-sibgate-<pool>-<sha8>-a<n>[-r<j>]`, legacy names
  never match `RUN_RE`; a key short after 2 attempts is `resample_failed`, s3d pools' 6 short
  keys count as exhausted), the LLM caches (the gate sets `fill_judge.RETRY_FAILED` False:
  cached failures stay final unless `--retry-failed`), `class_map.json` (never `dedupe:all`
  again; new raw names -> one `dedupe-incr-v1` call), frozen fillers, the text bank (a cell is
  generated once, `sibgen-chat-v2.1`, avoid list stored), the per-class `merge.json`
  (`--remerge`), `fit/<class>/inputs.sha256`, per-unit distractor freezes and per-class score
  `meta.json` (a unit whose sha differs -> `probe_na` `stale_score`). `--plan` writes nothing
  but the units cache; `--offline` turns any LLM call / Modal command into exit 3 (use it for
  idempotence checks). Exit codes 0 done / 2 needs `--yes` (estimate > `$GATE_AUTO_USD`, 300) /
  3 fatal / 4 partial (resample timeout: wait for the detached apps to finish, then re-run).
  Labels carry a stamp sidecar `labels/<arm>.gen.json` (generation + unit hash); the rebuild
  refuses a stamped file whose unit hash differs from the cells (rev 20 + warning,
  `summary_v2.json` `sibling_labels_guard`). Labels without a stamp (plan-0009 style) are used
  as before. `report --view v2` reads `<out>/sibling_gate/labels` by default
  (`--no-sibling-labels` = rev 20) and prints the generation + coverage table. The 11 script
  arms take cells from `--cells-root` (default `outputs/wsbench_chat/claude-opus-5-5`); the 3
  extras' registered cells live in hb-taxonomy-v2. `said_norm` can contain `|`: split
  `"<said>|<item>"` keys with `rsplit` in new code (`classes.canonical_map` keeps plan-0009's
  `split` so labels reproduce). Validation (§8 step 1): `gate import`, `gate --plan --offline`
  (zero work), `gate --offline` (0 calls, 0 launches), `gate diff --baseline ...` (0 diffs).
- **No git writes from agents** in this worktree (no add/rm/restore --staged/commit).
- **In-run retries** (rev 14): an invalid `:M` (or calibration) answer is re-asked up to 2 times
  (`run.MAP_INVALID_RETRIES`) before it is cached as failed; meta records `n_invalid_attempts`
  and the summed usage.
