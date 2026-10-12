# CLAUDE.md — HallucinationBench v2 agent runbook

Suite of evals for the **types and severities** of hallucination in activation-reading lenses.
First eval: `wsbench_chat` (plan `plans/0001-bootstrap-wsbench-chat-taxonomy.md` is the spec);
the claim map / taxonomy v2 is plan `plans/0002-claim-map-taxonomy-v2.md`; the sibling-concept
probe eval is plan `plans/0003-sibling-concept-probes.md` (stage 1: inventory + bank; stage 2:
probes, judge, score, report).

## Layout

```
src/hallucination_bench/
  llm.py          one judge client per process: Anthropic (claude-*) / OpenRouter, retries,
                  process-wide RPM pacer, one asyncio semaphore, per-call Usage (tokens, USD)
  cache.py        append-only JSONL cache {key, fingerprint, ts, result, meta}; failed = result None
  readouts.py     readout contract loader, decode_token, verify_quote, normalize_ws
  summarizer.py   interp-v1 token-bag -> prose prompt (verbatim)
  taxonomy.py     THE taxonomy (single source of truth; rendered into the classifier prompt)
  taxonomy_v2.py  THE claim-map taxonomy (12 leaves / 8 groups / 6 VERDICTS, leaf tags, precedence, open
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
  evals/sibling_probe/{concepts,bank}.py  plan 0003: inventory + alias matcher (text / token
                  bags), leak / 8-gram / window-identity checks, coverage rule, ReadSpec writers
  evals/sibling_probe/probes.py   cross-fit probes, recipes V2 (shrinkage-whitened DiffMean, primary) and V0
                  (raw DiffMean) (pure numpy; copied into the Modal job)
  evals/sibling_probe/probe_io.py inputs.json for the Modal job, loaders / merge of its outputs, check vs sweep
  evals/sibling_probe/judge.py    prefilter, 5% coverage sample, claim map (claimmap_v2 + the
                  wsbench_chat cached_call), mention -> segment, sibscope-v1 / sibcover-v1 calls
  evals/sibling_probe/{score,report}.py  M/U classes, gates, beta1/beta2/AUROC/Test 3, rates,
                  bootstrap, verdict table; report.md + plots/sibling_probe/
data/wsbench_chat/   verbatim copies from wsbench-pub @ 92d763e (never edit)
data/calibration/    dev.jsonl (implementer), heldout.jsonl (blind author only), README (schema + log)
                     dev_v2.jsonl (implementer), heldout_v2.jsonl (blind author only)
scripts/run_all_arms.sh  tmux driver for the full run
scripts/build_dev_v2.py  writes dev_v2.jsonl (labels live here; re-run if the splitter changes)
data/sibling_probe/  concepts.json, items.json, probe_texts.jsonl, pilot/ (README = schema + provenance)
scripts/sibling_probe/   count_siblings / build_concepts / gen_items / write_plans (local),
                         qwen_gen_modal.py + run_modal.sh (GPU; gw scripts run from $GW, never edited)
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
- **No git writes from agents** in this worktree (no add/rm/restore --staged/commit).
- **In-run retries** (rev 14): an invalid `:M` (or calibration) answer is re-asked up to 2 times
  (`run.MAP_INVALID_RETRIES`) before it is cached as failed; meta records `n_invalid_attempts`
  and the summed usage.

## sibling_probe (plan 0003)

```bash
uv run python scripts/sibling_probe/build_concepts.py        # concepts.json (tokenizer-checked)
uv run python scripts/sibling_probe/gen_items.py candidates  # Opus cues (cached) -> implication_in
scripts/sibling_probe/run_modal.sh qwen-implication          # then `candidates` again (top-up)
uv run python scripts/sibling_probe/gen_items.py select      # items.json + responses_in
scripts/sibling_probe/run_modal.sh qwen-responses            # merges responses into items.json
uv run python scripts/sibling_probe/gen_items.py probes      # probe_texts.jsonl
scripts/sibling_probe/run_modal.sh plans|smoke|capture|jlens|nla|s3d|fetch
# stage 2
scripts/sibling_probe/run_modal.sh probes        # prepare inputs + Modal probe job (both recipes) -> outputs/sibling/probes/{v2,v0}/
uv run python scripts/sibling_probe/check_v2_vs_sweep.py   # pipeline V2/V0 == sweep (max abs diff < 1e-3)
uv run python scripts/sibling_probe/arm_overlap.py         # -> outputs/sibling/overlap/{overlap.json,overlap.md}
scripts/sibling_probe/run_modal.sh probes-smoke  # acts-sibling-smoke + synthetic probe texts
uv run hbench sibling judge --arms s3d-rl600,nla-rl-L42,jlens   # one process; resumes from cache
uv run hbench sibling judge --rebuild            # cells.jsonl / run.json from the cache, no calls
uv run hbench sibling score                      # -> outputs/sibling/score/summary.json
uv run hbench sibling report                     # -> outputs/sibling/score/report.md + plots/sibling_probe/
```

- The `sibling` dependency group (transformers + jinja2, no torch) gives the Qwen3.6-27B
  tokenizer / chat template locally; it is a default uv group. Tokenizer tests skip without it.
- **Read window**: every message is `[system, cue + " " + request]`; `from_last_sentence_start`
  must start at the request sentence, so a cue ending in `."` (quote) breaks the window. The
  window-identity check (`bank.WindowChecker`) runs in `gen_items` and `write_plans`;
  `write_plans check-prompts` re-checks the `prompts` output and `check-manifest` the capture.
- **Changing `concepts.json` aliases** changes the Opus prompts (alias lists are in them), so
  the affected concepts' cached cues / probe texts are regenerated on the next run; re-run
  `candidates` -> implication -> `select` -> responses -> `probes` in that order.
- Opus 5.5 refuses every hacker-occupation prompt; `gen_items.MODEL_FOR` sends those to Sonnet 5.5.
- Implication / response caches: `outputs/sibling/gen/{implication,responses}_out.jsonl`
  (idempotent on key); Opus cache `outputs/sibling/gen/cache.jsonl`. Do not run two
  `gen_items` processes at once (one cache file).
- **Probe recipes (plan rev 11)**: `v2` = shrinkage-whitened DiffMean (primary, validity gate 0.85,
  lambda by inner leave-one-fold-out CV; `probes.Whitener` / `cross_fit_whitened`, a port of the
  sweep's V2) and `v0` = raw DiffMean (rev 10, gate 0.70). The job writes per-category pieces
  (`outputs/sibling/probes/pieces/<recipe>/<cat>/`, one container per category) and
  `hbench sibling probes --merge` merges them into `outputs/sibling/probes/{v2,v0}/`;
  `outputs/sibling/probes_v2_rawdm/` is the rev-10 run (legacy single-recipe layout, still
  loadable). `score` emits `recipes.{v2,v0}` + `recipe_sensitivity` (flag fires on different
  verdicts or non-overlapping CIs of the sensitivity-normalised rate 1 - det(M)/ceiling / the
  excess ratio, not on raw rates) + calibration + the wsbench-verbatim (`um-sd-`) transfer check.
- **Probe job** (`probes_modal.py`, CPU, 16 cores / 48 GB per category container): reads every acts file ONCE (all
  layers' eval positions, fp16): the `wsbench-acts` volume costs ~0.5 s per file open, so a
  per-layer re-read of the 5,064 probe files is far slower. Outputs also land on the
  `sibling-probes` volume (`/<run>/`); `hbench sibling probes --fetch --run <run>` re-gets them.
- **Folds** are per category, seeded, balanced by (concept, ending, kind) (`probes.assign_folds`, salt = category; run `sibling-v2` on; `outputs/sibling/probes_v1/` = the old (concept, ending) folds); eval cells are
  scored by all 4 fold models and indicators / z are averaged. `eval_scores.jsonl.gz` has one
  row per (id, layer, pos, concept); `det_*` are fold-averaged shares (0, .25, ... 1).
- **J-lens unmentioned** needs a summary, so `judge` summarises every J-lens cell by default
  (`--no-summarize-all` to judge-only); cells without a summary are `no_summary` (excluded).
- Judge outputs `outputs/sibling/judge/<arm>/{cache,cells}.jsonl` + `run.json` (claim-map
  PROMPT_VERSION, taxonomy version, git sha, spend). Keys: `summ:{key}`, `{key}:M`,
  `{key}:S:{readout}:{start}-{end}` (scope), `cover:{category}:{target}`.

## sibling_probe plan 0004 (11 layers, set B, verdict judge)

```bash
# set A readouts at 11 layers (GPU jobs: /tmp/sib_all11*.sh; ML-NLA from /workspace/gw-mlnla, the
# gw-wrong-bridge NLA reader lacks the mlnla lens); J-lens + ML-NLA gen dirs on wsbench-gen-sibling
uv run python scripts/sibling_probe/judge_all11.py --arms s3d-rl600-all11,jlens-all11   # claim map, new cells
uv run python scripts/sibling_probe/judge_all11.py --arms mlnla-rl600-all11
# set B (basic_readout): bank, inventory, claim map, brnorm, probe texts, captures, probes
uv run python scripts/sibling_probe/br_build.py bank-in|bank|inventory|judge|brnorm|probe-texts|capture-rows|probe-inputs
# claim units -> probe verdicts -> consensus -> sibverdict-v1 -> categories / summary / figures / calibration
uv run python scripts/sibling_probe/verdicts.py units|judge|label|figures|calib
uv run python scripts/sibling_probe/viewer/build_viewer.py      # -> outputs/sibling/viewer/ (deploy: scripts/serve/sibling_viewer.py)
```

- Arms: set A `s3d-rl600-all11`, `jlens-all11`, `mlnla-rl600-all11` (judge dirs; the first two
  seeded from the 4-layer caches) + `nla-rl-L42`; set B `br-<arm>` judge dirs, every non-empty
  cell judged, scope / coverage calls off. Methods in units: `s3d`, `jlens`, `nla-rl-L42`
  (primary), `mlnla-rl600` (side).
- Set B: `data/sibling_probe/basic_readout_bank.json` (67 in-pipeline items: entity + implicit;
  processing viewer-only), `br_concepts.json` (category-first: golds + 15 Opus members),
  `br_probe_texts.jsonl` (only categories with >= 2 golds: sport, country, musical_instrument;
  single-gold categories cannot pass the mandatory transfer gate). Opus caches per prompt version
  `outputs/sibling/br/cache_{brcat,brinv,brnorm,brtext}-v1.jsonl`. Captures `acts-br-eval`,
  `acts-br-probe` (12 layers incl. 42). Probes `outputs/sibling/br/probes/{v2,v0}`.
- Probes: set A 20..60 from `outputs/sibling/probes_all11/` (run `sibling-all11`, eval capture
  `acts-sibling-eval-all11`, probe capture `acts-sibling-probe-all11`); L42 from the plan-0003
  `outputs/sibling/probes/`. Set-B gates: validity + transfer per concept, category-layer rule
  over the category's GOLD concepts; never-gold concepts are `unvalidated_never_gold`.
- Verdict outputs `outputs/sibling/verdict/`: `units.jsonl`, `units_labeled.jsonl`, `cache.jsonl`
  (keys `v:<uid>` layer-visible, `b:<uid>` layer-blind), `blind_uids.json`, `gates*.json(l)`,
  `summary.json`, `report.md`. Figures `plots/sibling_probe/fig{1,2}_*.png`.
- Network on this box drops DNS intermittently: every Modal step in the drivers retries; check
  `getent hosts api.modal.com` before blaming a script.

## sibling_probe plan 0005 (basic-readout mention ladder)

```bash
S=scripts/sibling_probe/ladder.py
uv run python $S lexicons      # brlex-v1 -> data/sibling_probe/br_lexicons.json (Opus once + code checks/forced equivalences)
uv run python $S mentions      # units (method, item, layer, pos, concept) -> outputs/sibling/ladder/mentions.jsonl
uv run python $S judge         # brcommit-v1, one call per (unit, sample) -> outputs/sibling/ladder/commit.jsonl
uv run python $S todo          # data/sibling_probe/brcommit_{dev,gold}_todo.jsonl (dev 40 for Fable; gold 80 + all Ali units)
uv run python $S calib         # vs data/sibling_probe/brcommit_gold.jsonl -> outputs/sibling/ladder/calib.json
uv run python $S banks         # brbank-v1 -> data/sibling_probe/br_entity_banks.json (Ali bank seeds sport)
uv run python $S capture-rows  # -> outputs/sibling/ladder/capture_rows.json (bank x 3 templates + the 35 items)
MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run --env camila-blank scripts/sibling_probe/ladder_capture_modal.py --run ladder1
MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest volume get --env camila-blank sibling-probes ladder/ladder1 outputs/sibling/ladder/capture/
uv run python $S topup && uv run python $S capture-rows --exclude-runs ladder1   # entity top-up (< 40 kept) -> capture run ladder2
uv run --with scikit-learn python $S probes --run ladder1,ladder2 --modal   # filter, counts, probes on Modal CPU (~10 min) -> probes.json
uv run --with scikit-learn python $S probe-report [--fire-pair logreg,dom]  # AUROC tables, fire pair, capital pilot -> probe_quality.{md,json}
uv run python $S label         # -> outputs/sibling/ladder/ladder/{units.jsonl,summary.json,report.md}
uv run python $S figures       # -> plots/sibling_probe/ladder_*.png
uv run python scripts/sibling_probe/viewer/build_viewer.py  # viewer incl. the ladder panel + label filter
```

- Library: `evals/sibling_probe/ladder_lex.py` (lexicon matcher: word boundaries, longest match,
  block phrases, "football" resolved by the item's gold / `FOOTBALL_DEFAULT`, upper-case
  abbreviations case-sensitive), `ladder.py` (units, brcommit-v1 prompt / parse / fail-closed,
  roll-up, cross-method, fire rule, labels), `ladder_probe.py` (logreg PCA-k / DiffMean /
  whitened DiffMean, GroupKFold(5) by entity, inner-CV k and lambda). sklearn is NOT a project
  dependency: run the probe stages with `uv run --with scikit-learn`.
- Opus caches per prompt version: `outputs/sibling/ladder/cache_{brlex,brcommit,brbank}-v1.jsonl`.
  Changing the lexicon changes judge user texts (marked spans), so the affected calls re-run.
- Gold equivalences live in the lexicon (England / Britain / UK one concept; USSR = Russia;
  piano = keyboard; track and field = athletics = sprinting = running); the bank gold `england`
  maps to `united_kingdom` (`ladder.GOLD_MAP`).
- The probe job is CPU-heavy; keep BLAS at 2 threads per worker (`threadpool_limits(2)`), 16
  threads per fit is 4x slower than 2. 8 workers x 2 threads.
- Capture: 1 GPU (B300) HF transformers, no flash-linear-attention (slow fallback is fine for
  ~14k short prompts, ~1 h); items must tokenize to the bank `input_ids` or it aborts.
- Probe fitting: use `--modal` (one CPU container per bank x layer, acts read from the
  `sibling-probes` volume). Locally it is ~40 CPU-hours: lbfgs logreg (~300 iterations) dominates.
  Never `pkill -f` a pattern that is in your own shell's command line; use `pkill -f "ladder.py pro[b]es"`.
- Fire pair = logreg + raw DiffMean (the plan's probes). The whitened DiffMean (`wdm`) has the
  best AUROC but whitens away the within-gold variation the fire rule tests (Ali track/basketball
  stop firing); it is the `label_pair_alt` sensitivity row.
- Knowledge filter: Ali's best-category-word rule, plus the greedy continuation's first mention
  (multi-token names like " c|ello" are missing from the word list). Classes with fewer than 10
  known entities are dropped from the bank (tuba, oboe), so their units are probe n/a.
