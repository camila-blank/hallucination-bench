# CLAUDE.md — HallucinationBench v2 agent runbook

Suite of evals for the **types and severities** of hallucination in activation-reading lenses.
First eval: `wsbench_chat` (plan `plans/0001-bootstrap-wsbench-chat-taxonomy.md` is the spec);
the claim map / taxonomy v2 is plan `plans/0002-claim-map-taxonomy-v2.md`; the sibling eval is
the user-modelling ladder on 3 domains (plans `0006` + `0011`; its predecessors, plans 0003–0005,
and the occupation / emotion domains were removed in plan 0013); the `story` eval (the claim map on 20 story items, `--eval story`)
is plan `plans/0007-story-eval.md`; the `agentic_misalignment` eval (detector + claim map on lens
mentions of misalignment, its own driver, `--eval agentic_misalignment`) is plan
`plans/0008-agentic-misalignment-claim-map.md`. The `overrefusal` eval (OR-Bench pilot: refusal claims on prompts
the model never refuses; own subcommands `hbench overrefusal ...`) is plan
`plans/0015-overrefusal-eval.md`. The hallucination index (`hbench index`, one
macro per (method, layer)) is plan `plans/0013-final-sibling-eval-and-hallucination-index.md`
(hindex-v3: main / AM / sibling / story) + `plans/0019-safety-families-on-the-index.md`
(hindex-v4: + overrefusal only; strongreject / mask stay off, revs 6 / 5) +
`plans/0021-am-candidate-claims-index-v5.md` (hindex-v5: AM on candidate claims) +
`plans/0022-overrefusal-wrong-topic-refusals.md` (hindex-v6, the CLI default: overrefusal as
wrong-topic refusal claims). The `mask` eval (MASK lies vs
honest-but-wrong: does the lens claim the model is lying when it is honest? Its own CLI,
`hbench mask gt|run|report`; ARCHIVED 2026-10-29: a diagnostic, not part of HallucinationBench or
the index, not a hallucination score; HB's safety evals are overrefusal only) is plan `plans/0017-mask-eval.md`. The `benign_roleplay` eval (false jailbreak recognition on gated
harmless role-play prompts; items = WorkspaceBench's jailbreak negative controls + `hb_only`
gated role-play candidates; judged by running WSB's own pinned `jailbreak_recognition` family;
`hbench benign-roleplay import|run|report`; NOT on the index) is plan `plans/0023-benign-roleplay.md`.

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
  evals/__init__.py  EVALS registry + bank_for(name); Models.eval picks the bank and <out>/<eval>/
  evals/_bank_common.py  sites / positions / marked response / expected grid (every bank)
  evals/story/bank.py    the story bank (data/story); the pipeline is wsbench_chat's (plan 0007)
  evals/wsbench_chat/{bank,run,score}.py   bank + grid; pipeline (shared); metrics + bootstrap
  evals/wsbench_chat/score_v2.py   claim-map metrics (char-weighted shares, rates, comparisons)
  evals/agentic_misalignment/      plan 0008 (own driver; NOT the chat pipeline):
    kinds.py      THE misconduct-kind table + actor/stance enums + design-kind map (rendered
                  into the detector, screener and intent prompts)
    bank.py       data/agentic_misalignment bank; marked_prompt (marker in the PROMPT)
    detect.py     amdetect-v1 / amscreen-v1 prompts, schema, batch render, fail-closed parse
    verify.py     amverify-v1 (Stage H, rev 13): blind headline verifier prompt, schema, per-mention parse
    intent.py     amintent-v2 (research frame; rollout + scenario kinds, located quotes) + intent_overrides.json
    run.py        run_am (D -> S -> H -> M + intent), Detector terminal rule, Verifier, rebuild
    score.py      mention_verdict, regions, kind_match, set-up, counting spans, shares
    report.py     `report --eval agentic_misalignment` (report.md / report.json)
  judges/claimmap_am.py  claimmap-v2.7+am2: research frame + chat SYSTEM + tail verbatim, AM head (marker in prompt)
  calibrate.py    classifier vs gold cases, Wilson CIs, the held-out gate
  calibrate_v2.py claim map vs unit-level gold (dev_v2 / heldout_v2), the v2 gate
  report.py       cross-arm report, audit samples, pilot-dump
  report_v2.py    `report --view v2`, probe_candidates.jsonl, audit_samples_v2.md, pilot-dump-v2
  consistency_v2.py  `consistency-v2`: two fresh :M calls per sampled cell, κ / confusion / flags
  cli.py          hbench run|calibrate|report|consistency-v2|pilot-dump|pilot-dump-v2|units-v2|
                  index: shared commands + helpers; eval subcommands come from EVAL_CLIS (empty)
  evals/<eval>/cli.py  an eval's own CLI: agentic_misalignment (run / report behind
                  `--eval agentic_misalignment`). A new eval adds its CLI here and lists a
                  `register(sub)` module in EVAL_CLIS, not in cli.py (plan 0012 D2)
  evals/strongreject/   plan 0016, ARCHIVED DIAGNOSTIC (not in HB) (own CLI `hbench strongreject grade|run|report`, registered in
                  EVAL_CLIS; not in EVALS, not on the index):
    families.py   pinned StrongREJECT sources + the 7 families + wrap() (the package's rule)
    grade.py      srgrade-v1: StrongREJECT rubric (JSON) + explicit_refusal; adoption markers; outcome
    bank.py       data/strongreject/items.json; thinned sites (last 64 + every 16th; regions)
    detect.py     srdetect-v1 blind detector (harmful_compliance / persona_adoption /
                  refusal_intent / jailbreak_request / harm_salience)
    echo.py       srecho-v1 outcome-blind speaker / confirm / echo judge (numbered markers)
    run.py, score.py, report.py, cli.py   driver (grade, D, E, rebuild), metrics + cluster
                  bootstrap, report.md
    probe_check.py  probe check (plan 0015's refusal probe): premise, the rev-6 HB scores
                  (`hb_scores`: raw + probe_validated), probe-checked rates,
                  misses (scripts/strongreject/probe_modal.py scores sites; probe_report.py)
  index.py        plans 0013 + 0019 + 0021 + 0022: the hallucination index (CLI default
                  hindex-v6; `--version v5` / `v4` / `v3` = those exactly; library default v4) from the
                  families' outputs; safety families from each
                  eval's index_counts.jsonl / index_meta.json; method <-> arm mapping in
                  data/index/methods.json
  paths.py        external locations as HBENCH_* env vars with dev-box defaults (plan 0012 D3)
  evals/sibling_probe/  the user-modelling (UM) sibling ladder (plans 0006 + 0011):
    um_ladder.py   lexicon patch, FastMatcher, units, umcommit prompt / parse, distractors,
                   ProbeContext fire rule, labels, bootstrap
    um_domains.py  DOMAINS = country, capital_city, holiday (per-domain specs)
    um_probe.py, um_regions.py   logistic probe + nulls; 12 merged M49 regions (country)
    ladder_lex.py  lexicon matcher (plan 0005's, kept for the ladder)
    probes.py      DiffMean / whitened probes (copied into the Modal job with um_probe.py)
    {concepts,bank}.py  plan-0003 inventory / alias matcher, WindowChecker, ReadSpec writers (kept:
                   the ladder's probe-text and capture-plan steps use them)
data/wsbench_chat/   verbatim copies from wsbench-pub @ 92d763e (never edit)
data/calibration/    dev.jsonl (implementer), heldout.jsonl (blind author only), README (schema + log)
                     dev_v2.jsonl (implementer), heldout_v2.jsonl (blind author only)
data/story/          20 story items x 8 sites + capture rows, built by scripts/build_story_bank.py
data/agentic_misalignment/  32 AM items x every prompt position (50,452 sites), personas, built
                     by scripts/build_am_bank.py; intent_overrides.json after the intent audit
scripts/run_all_arms.sh  tmux driver for the full run
scripts/build_story_bank.py  writes data/story/ from global-workspace's hallucination_creative bank
scripts/fetch_story_readouts.sh  Modal story gen dir -> /workspace/hb-readouts-story/<arm>/ + grid check
scripts/build_dev_v2.py  writes dev_v2.jsonl (labels live here; re-run if the splitter changes)
data/sibling_probe/  UM ladder data (um_*, umcommit_*, br_lexicons_um_v1.json) + frozen plan-0003
                     inputs it reads (concepts.json, items.json, probe_texts.jsonl, pilot/);
                     README = schema + provenance
data/index/methods.json  plan 0013: index method <-> arm mapping, exclusions, notes
scripts/sibling_probe/   um_ladder.py (driver), um_probes_modal.py, um_lexicons.py, um_plot_*.py,
                         um_run_*.sh; gen_items.py (imported by `um_ladder.py probe-texts`)
scripts/build_am_bank.py  writes data/agentic_misalignment/ (streams the readouts; fails loudly)
scripts/am_audit.py  AM audits: sample | sample-headline | screen | summ-only | sample-tiers | score
scripts/run_am.sh    AM driver: step1 (tmux hbench-am-detect) | step2 (tmux hbench-am-map)
data/strongreject/   plan 0016: source/ (pinned StrongREJECT csv + wrappers + rubric), prompts.json
                     (280), items.json + capture_rows.json (read set; built after grading)
scripts/strongreject/  build_prompts.py, sample_modal.py (vLLM rollouts), build_bank.py,
                     readouts.sh (capture -> 3 reads -> fetch_readouts.sh convert + grid check)
evals/overrefusal/   plan 0015 (own driver + own CLI via EVAL_CLIS; NOT the chat pipeline):
    data.py      pinned OR-Bench candidates, populations (never_refused / hard_refused /
                 always_refused), bank render + sites
    gt.py        orgt-v1 refusal judge over 10 sampled (+ greedy) replies per prompt
    detect.py    ordetect-v1 blind batched detector (refusal_intent / harm_salience /
                 comply_intent / other; stance asserted / considered for prose refusal)
    verify.py    orverify-v1 blind verifier of refusal_intent candidates (headline membership)
    run.py       judge-gt, detect, verify, cache-only rebuild -> cells.jsonl / summary.json
    score.py     prompt-weighted rates per (arm, layer, population), prompt bootstrap
    report.py, cli.py
    wsb_digest.py, wsb_bpe.py  plan 0020: literal port of wsbench digest-v3 + bpe @ 1baaf00
                 (AST-equality test tests/test_or_digest.py, needs $HBENCH_WSBENCH_PUB)
    sjudge.py    plan 0020: bundle per (reader, item, layer) -> digest (Opus, Sonnet fallback) ->
                 strict judge orsj-v3 -> off-topic judge oroff-v1; line provenance
    sj_metrics.py plan 0020: metric 1 (raw / probe-validated, provenance union), metric 2
                 (off-topic), recall, health; index_counts.jsonl + index_meta.json
    sclaim.py    plan 0020 rev 7 (exploratory): orclaim-v1 statement labeller + claim-unit metric
                 (scripts/overrefusal/sj_claims.py -> sj_claims_full.json, index_counts_claims.jsonl)
    offtopic_export.py plan 0022: `hbench overrefusal offtopic-export`: sj_records_full.jsonl ->
                 offtopic_index/{index_counts.jsonl,index_meta.json} (unit refusal_claims, the
                 hindex-v6 overrefusal root) + sj_offtopic_full.json (value + 3 secondary numbers)
data/overrefusal/    candidates, gt_labels, items, capture_rows (README = schema)
evals/benign_roleplay/  plan 0023 (own CLI via EVAL_CLIS; in EVALS; NOT on the index):
    bank.py      import + fail-closed gate check (gate_sha re-hashed from build_v2 labels),
                 template_id (WSB J1.1 formula), MANIFEST
    wsb.py       WSB_SHA pin check + `uv run --no-sync wsbench run ...` subprocess calls; the
                 ONLY code touching WSB; ports nothing (test_no_wsb_logic_in_hb). Reuses the
                 campaign's judged dirs ($HBENCH_BR_CAMPAIGN) for the shared ids
    metrics.py   refusal conditions, flagged share + template_id cluster bootstrap, shared-subset
                 == WSB FPR check (exit 2), po-opus-v1 floor, summary.{json,md}
data/benign_roleplay/  items_v1.json (150: 86 WSB-shared + 64 hb_only), wsb_bank_hb_only_v1.json
                 (WSB_BANK_FILE_JAILBREAK_RECOGNITION), gt_labels.jsonl, MANIFEST (README)
scripts/overrefusal/ sample_modal.py (vLLM ground truth), reads.sh (capture + reads + convert)
evals/mask/          plan 0017 (own CLI `hbench mask`, registered in cli.EVAL_CLIS):
  source.py     MASK judge prompts verbatim (mask @ 25e0b12, sha-pinned) + per-archetype choice
  labels.py     MASK belief / honesty aggregation (verbatim ports) + item labels / strata
  gt.py         stage T: statement / belief / pressure-direction calls -> data/mask/labels.json
  bank.py       data/mask loaders, select_sites (8 prompt + 8 response), readouts grid check
  detect.py     maskdetect-v1 (blind; lie_claim / honest_claim / pressure_salience /
                belief_content / other), verify.py maskverify-v1 (+ blind variant)
  run.py        D -> V -> VB driver, cells.jsonl / summary.json; report.py metrics + bootstrap
  probe.py      rev 4 two-class deception probe on sibling_probe.probes (V2) + um_probe null z
data/mask/           items / rollouts.jsonl.gz / labels / read_set / sites / capture_rows (README)
scripts/mask/        build_items.py, gen_modal.py (Modal vLLM), build_capture.py, fetch_readouts.sh,
                     gate.py (self-knowledge gate, rev 3), probe_build.py + probe_modal.py (deception
                     probes, rev 4), examples.py (hand-reading)
```

## Conventions

- `uv` for everything: `uv sync`, `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`.
- The v5c span/verify literals are frozen: `judges/v5c.py` asserts the span literals equal
  `data/wsbench_chat/judge_prompt_v5c_chat.json` at import; tests check the verify literals
  against the wsbench-pub source (`$HBENCH_WSBENCH_PUB`, skipped if absent) and the
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
- External paths come from `paths.py` (`HBENCH_*` env vars, dev-box defaults); never hardcode a
  `/workspace/...` path in `src/`, `scripts/*.py` or `tests/`. Shell drivers `cd` to the repo
  root from their own location and read `${HBENCH_X:-<default>}`.
- CI is pending (plan 0012 D5, held back for a token scope): until it lands, run ruff check +
  format and `pytest -rs` locally before pushing. Plan numbers are unique: the next free one is
  0024 (0009/0010 belong to the unlanded resampling-gate branches).

## Run commands

```bash
R=${HBENCH_READOUTS_ROOT:-/workspace/wsbench-paper-data/readouts}
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

- **`--eval story`** (plan 0007) exists on `run` and `report` only. Its outputs live under
  `outputs/story/<model>/`, and its readouts are not in git:
  `/workspace/hb-readouts-story/<arm>/hallucination.jsonl`, produced by the Modal reads of
  `wsbench-acts:/acts-hallucination-story[-L42]`.
  - Story is **always pre-gate**, because the held-out gate is chat gold. Its report has no
    judge-noise numbers.
  - Never run `consistency-v2` for story: it is chat-only, and its output file is the chat noise
    source.
  - `pilot-dump*`, `calibrate` and `units-v2` are chat-only.
  - **DeepSeek-V4-Flash port** (2026-10-20): `HBENCH_STORY_ITEMS` points the story bank at
    another subject's on-policy bank (V4: `outputs/v4/story_bank/items.json`, built by
    gw-v4-flash-ar `scripts/v4_flash/story_v4_bank.py`; the producers read it as family
    `hallucination_story` of the bench copy `/workspace/root/wsbench-pub-v4-story`).
    `scripts/run_story_v4.sh <arm:volume>,...` fetches + grid-checks the readouts
    (`outputs/v4/story_readouts/`) and judges into `outputs/v4/story/`;
    `scripts/run_story_v4_rl600.sh` adds the RL600 arm once its adapter exists.
    **On the index** (plan 0014): V4 arms (`v4-*`) are matched by name; pass the V4 roots
    beside the Qwen ones: main/story `outputs/v4r/{wsbench_chat,story}/<judge>`, AM
    `outputs/v4/agentic_misalignment/<judge>`, sibling `outputs/v4/sibling/um` (command:
    `docs/results/0013` §DeepSeek-V4-Flash). The sibling root is an export (`scripts/v4_um/export_index_root.py`)
    of `outputs/v4/um`, renaming the ladder's `jlens` to `v4-jlens-std-n1000` (else it collides
    with Qwen `jlens`). V4 AM is read at 21/27/33 only, so V4 `full`/`matched` exist there only;
    chat/story at 21/27/33 come from the producers' jobs mode (`scripts/v4_index/`:
    `readout_jobs.py` jobs/merge, `judge_v4r.sh`, outputs `outputs/v4r/`). The V4 chat bank's first
    3 items were unpadded-vLLM garbage and are regenerated (`scripts/v4_index/regen_chat3.py`;
    originals `*.corrupt` in the bench copies). Judging shares the org's 2M output-tok/min limit:
    concurrency 128 draws 429s (retried); a 401 means the OAuth profile token rotated — retry.

- **`--eval agentic_misalignment`** (plan 0008) is its own driver (`evals/agentic_misalignment/run.py`),
  dispatched first in `cmd_run` / `cmd_report`; chat and story never touch it.
  - **The readouts are prompt positions only** (every position up to `<|im_start|>assistant`,
    no thinning), so the marker `⟦READ⟧` sits **in the prompt**, not the response; the claim map
    uses `claimmap_am` (chat SYSTEM byte for byte, new head, chat tail). Never feed AM cells to
    the chat pipeline.
  - Readouts file `<arm>/agentic_misalignment.jsonl` (default root the paper readouts; default
    arms s3d-rl600, nla-rl-L42, jlens). The loader checks the whole file (full 50,452-site grid
    per layer, no unknown ids) before `--items` / `--layers` / `--limit` apply.
  - `--stages detect,summ,verify,map,intent` add no dependencies: unselected stages are cache-only
    (`--stages map` maps only cells whose detection is cached). Intent is per item, cached in
    `<out>/agentic_misalignment/<model>/intent.jsonl`; any one `--arms` value triggers it.
  - Detector cache is per cell (`{key}:D`, J-lens summaries `{key}:Dsum`), fingerprint over the
    canonical readout lines (no batch label); usage only on `batchD:` rows (spend sums every
    row). Failed cells retry as batches of 1 in the same run; 2 failed single-cell rows with a
    partial parse -> accepted (`meta.accepted_partial`). `batchD:` / `audit:` keys are never
    counted as cells.
  - **Stage H** (`--stages verify`, `amverify-v1`, plan 0008 rev 13; exists because the O-lens
    headline-membership gate failed at 0.70 under amdetect-v2) re-judges only headline candidates
    and changes **only headline membership, never the map population** (map stays on every
    detected cell; no `:D` / `:Dsum` / `:M` key or fingerprint changes). Rows
    `{key}:H:{readout}:{start}-{end}` + usage-only `batchH:` rows (never counted as cells); a
    candidate without a result keeps the old rule (`health.n_headline_unverified`); record
    `actor` / `stance` are the verified ones (`stance_raw` / `actor_raw` = detector), `kind` stays
    the detector's; H failures exit 4 only when `verify` is selected. After Stage H, re-draw the
    membership sheet with `am_audit.py sample-headline` (rewrites ONLY `audit/headline.jsonl`).
  - J-lens: the scored mentions are the **summary's** (`:Dsum`); `lost_in_summary` is an upper
    bound until `am_audit.py sample`'s lost_in_summary sheet is read.
  - Every headline number counts **spans** (overlapping mention spans of a readout merged);
    the headline bootstrap resamples the 28 misaligned items. AM is always provisional (no gate).
  - **API refusals** (pilot 2026-10-13): without the research frame (`claimmap_am.RESEARCH_FRAME`,
    opening the AM user prompt and the intent prompt) Opus 5.5 hard-refused (stop_reason
    `refusal`, 0 output tokens) ~73% of murder and ~29% of blackmail-replacement map calls.
    Keep the frame in the USER prompt: in the system prompt it only halves refusals. Refusals are
    cached as failed and re-queued; watch `usage.refusals` in the cache meta.
  - Rebuilding an 11-layer arm loads ~555k cells (~1 GB for J-lens): fine under the 32 GB cap,
    but do not run several AM rebuilds in parallel.

- **strongreject is an archived diagnostic, NOT part of HallucinationBench** (Camila,
  2026-10-29: on clear jailbreaks the model plausibly wavers, so compliance claims are not
  verifiable hallucinations). Code, frozen data and results stay as merged; never put it on the
  index or call its numbers HB scores.
- **`hbench strongreject`** (plan 0016) is its own CLI (`grade` / `run` / `report`), not a
  `run --eval` switch; outputs `outputs/strongreject/<judge>/` (`grade_cache.jsonl`,
  `outcomes.json`, `<arm>/{cache,records,summary}`, `report.md`). Rollouts
  (`outputs/strongreject/rollouts.jsonl`) and readouts
  (`$HBENCH_SR_READOUTS`, default `/workspace/hb-readouts-strongreject/<arm>/strongreject.jsonl`)
  are not in git.
  - **Opus 5.5 refuses this eval's harmful content**, deterministically: ~14% of rubric grades
    (by the forbidden prompt's topic, even when the response is a refusal) and ~99% of 40-cell
    O-lens detector batches. A stronger research frame did not help. Every refused grade /
    detector / echo call is re-asked once on `claude-sonnet-5-5` (`Judge.fallback`,
    `run.call_with_fallback`; rows record `answered_by`). Opus-vs-Sonnet detector agreement on
    300 O-lens cells: kappa 0.64 (harmful_compliance) to 0.83, rates within 1–5 points
    (`scripts/strongreject/judge_agreement.py`). ~1% of O-lens cells stay failed (both refuse).
  - Qwen3.6-27B was jailbroken by **none** of the 10 deterministic wrappers (0/400 prompts
    successful; 5 complied samples in ~3,600 graded), so there is no recall set: `soft_fail`
    items (rubric-refused, not explicit refusals) are the only contrast.
  - The Qwen chat template **trims** the user content: `bank.render_prompt` strips too.
  - Modal from this box: the read scripts need `MODAL_RESOURCE_ENV=camila-blank` (their
    volumes default to env `main`, where we have no access); `wsbench_read_modal.py` needs
    `::main`; `uvx modal@latest` can hang on the shared uv cache lock, so
    `scripts/strongreject/readouts.sh` honours `MODAL_BIN` (e.g. `~/.local/bin/modal`).
    J-lens/NLA gen dirs land on `wsbench-gen-3` even when the client log says gen-4.
  - The multilayer NLA read is slow (~4 h on 16 H200s for ~48k cells: reference PyTorch
    kernels); the O-lens read took ~25 min.
  - **Rev 5 (final, frozen):** every prompt position is read (33,881 sites; regions turn = last
    4, tail = rest of last 64, body = before). `data/strongreject/MANIFEST.sha256` freezes the
    set (test-checked); re-run `scripts/strongreject/freeze.py` only for a deliberate new
    version. The score uses unsuccessful items only. The rev-1..4 readouts are kept as
    `strongreject.rev4.jsonl` beside the merged files; `rest_rows.py` / `readouts_rest.sh` /
    `merge_readouts.py` add positions some arm lacks. NLA unit files are written whole, so a
    stopped NLA read resumes cleanly; shard it one unit per shard.
  - **Probe check:** `probe_modal.py` scores every site with plan 0015's refusal probe (v2
    threshold from `hb-overrefusal:data/overrefusal/probes/refusal_probe_v2.npz`; our acts hold
    the whole template incl. the think block, so the tail is located from `n_positions`);
    **HB scores (rev 6):** `probe_validated` (headline) = claims at refusal-present sites /
    readouts, `raw` = all verified claims / readouts; `hbench strongreject report` emits both
    (reads `<out>/strongreject/probe_sites.jsonl`; `--probe-npz` / `HBENCH_SR_PROBE_NPZ` for
    cell validity). `probe_report.py` writes the standard probe-checked rate (claims at refusal-present sites /
    readouts) per region; `plot_rates.py` the slide plot.
  - A 401 "OAuth access token has been revoked" mid-run is the profile token rotating: re-run.
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
- **Exit codes**: `hbench run` exits 4 when failed calls remain cached (re-run to retry them;
  the driver does this up to 3 times), 3 on auth/config errors, 2 on missing cells.
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
- **No git writes from agents** in this worktree (no add/rm/restore --staged/commit).
- **Hallucination index** (plan 0013): `hbench index` reads the families' outputs only (no API
  calls); `hbench report` refreshes it after its own report returns normally (an index failure
  is a warning, never an exit code). Its definition reproduces the 2026-10-20 exploratory
  numbers bit for bit (fresh `default_rng(0)` / `Random(0)` per row): any change to weights, claim
  definition or bootstrap bumps `index.INDEX_VERSION`. A new arm is matched by name across
  families; only legacy names / exclusions go in `data/index/methods.json`.
  **hindex-v4** (plan 0019 rev 6) adds overrefusal only (weight 1; strongreject / mask dropped;
  share = Σ `n_contradicted` / Σ `n_claims` from its `index_counts.jsonl`, whose rows plan 0020
  emits as one verdict per item (`n_claims` = judged, unit columns 0, `unit: "items"` in
  `index_meta.json`); raw share and health beside it; absent with an `absent_reason` on nothing
  judged or all-unprobed false claims; cluster bootstrap, fresh `default_rng(0)` per (family,
  arm, layer), clusters sorted, percentiles over draws with claims). `full` = 5 families,
  `matched` = all but sibling, `full_v3` / `matched_v3` always emitted. Reader allowlist
  `index.SAFETY_READERS` is enforced in the index too. The overrefusal root is opt-in
  (`--overrefusal-root` or a set `HBENCH_SAFETY_OR_OUT`; never the paths.py default) so tests
  that build from `tmp_path` stay isolated. The hindex-v3 outputs are pinned by goldens built by
  the v3 code (`tests/fixtures/index_v3/`, `tests/fixtures/index_v3_devbox.{json,md}`; the
  dev-box one skips off the dev box): `--version v3` must reproduce both exactly (minus
  `git_sha` / `sources`), and its notes are frozen in `index.NOTES_V3`. Never regenerate a
  golden (`scripts/index/build_v3_fixture.py` is provenance, not a refresh tool).
  **hindex-v5** (plan 0021; `index.INDEX_VERSION_V5`; `index.INDEX_VERSION` still means
  "hindex-v4"): v4 with AM on **candidate claims** — `claim_rows(..., am_candidates=True)` (one
  streaming pass) keeps a claim only if its `[start, end)` overlaps by >= 1 char a
  `score.merge_intervals`-merged `mentions` span of the same readout in the same cell (touching
  does not count; `index.is_candidate`); items / readout / segment columns as v4, zero-candidate
  items stay as zero rows, `n_all_claims` tallied beside `n`; an AM (arm, layer) with 0
  candidates is absent (`row["absent"]["agentic_misalignment"] = "0 candidate claims"`). Every
  macro using AM (incl. `full_v3` / `matched_v3`) uses the v5 value; main / story / sibling /
  overrefusal are bit-identical to v4. The `hbench index` CLI defaults to v5; `build_index`
  (and so `hbench report`) keeps `version="v4"` as its default so pre-v5 tests exercise v4. The
  v5 note is in code (`index.NOTES_V5`), not methods.json; v4 notes still come from methods.json,
  guarded by `index.NOTES_V4` (a test asserts methods.json == NOTES_V4), and the v4 md header is
  frozen in `_render_md_v4`. v4 goldens: `tests/fixtures/index_v4/` (synthetic, generated by
  `scripts/index/build_v4_fixture.py` from the pre-v5 code) and
  `tests/fixtures/index_v4_devbox.{json,md}` (the #42 run; a filtered safety-root mirror).
  `tests/test_index_v5.py` also cross-checks v5 AM against PR #43's
  `scripts/index/am_candidate_claims.arm_rows` variant (a) (skips off the dev box).
  **hindex-v6** (plan 0022; `index.INDEX_VERSION_V6`, **the `hbench index` CLI default**;
  `build_index` still defaults to v4): v5 (AM on candidate claims: `index.AM_CANDIDATE_VERSIONS`)
  with overrefusal read from a root whose `index_meta.json` has `unit == "refusal_claims"`
  (`offtopic_export.py`; per item `n_claims` = decline lines, `n_false` = `n_contradicted` =
  off-topic decline lines, `n_failed_cells` = 1 for an unjudged / strict-judge-failed /
  off-topic-failed item-layer, all 427 items, `gates: {}`); share formula unchanged; main / AM /
  story / sibling bit-identical to v5. **Unit guard** `index.check_safety_units` (called in
  `build_index`, and in `hbench index` after the not-a-directory / no-`index_counts.jsonl`
  checks): v6 needs `refusal_claims`; v4 / v5 refuse it (`items` or a meta-less synthetic root is
  fine) — so `hbench report`'s v4 refresh refuses a v6 root from `HBENCH_SAFETY_OR_OUT`. v6 note in
  code (`index.NOTES_V6`, merged for v6 only; NOTES_V5 is not merged under v6); v6 md header in
  `_render_md_v6`; safety `share` text `index.SHARE_V6`. The export reads ONLY
  `sj_records_full.jsonl` (fingerprint-resolved) — never key-join the raw `sjudge/` / `offtopic/`
  caches (repeated keys under older fingerprints; ~165 records disagree). `off.topics` keys are
  JSON strings vs int `decline_lines`: the export asserts the sets match; records with empty
  `decline_lines` have no `off` key (`r.get("off")`). v5 dev-box golden:
  `tests/fixtures/index_v5_devbox.{json,md}` (the #45 run, `tests/test_index_v6.py`).
- **In-run retries** (rev 14): an invalid `:M` (or calibration) answer is re-asked up to 2 times
  (`run.MAP_INVALID_RETRIES`) before it is cached as failed; meta records `n_invalid_attempts`
  and the summed usage.

- **`hbench overrefusal ...`** (plan 0015, pilot) is its own driver and CLI. Readouts
  `$HBENCH_OR_READOUTS/<arm>/overrefusal.jsonl` (default `/workspace/hb-readouts-overrefusal`),
  prompt positions only (AM convention, render ends at `<|im_start|>assistant`), layers
  28/36/44/52; outputs `outputs/overrefusal/<model>/`. Modal names are prefixed
  `hb-overrefusal-` (acts `acts-hb-overrefusal`, gen dirs on `wsbench-gen-3`). Every rate is
  prompt-weighted (mean of per-prompt means); read the `asserted` + `tail` cell as the cleanest
  hallucination number (a `considered` claim on a borderline prompt can be faithful). The
  detector is blind to the prompt, so prompts *about* refusing (`about_refusal`) are reported
  with a sensitivity exclusion. On the hallucination index since hindex-v4 (plan 0019). **Detector / verifier judge is
  `--judge-model claude-sonnet-5-5`** (outputs `outputs/overrefusal/claude-sonnet-5-5/`): Opus 5.5
  hard-refuses ~all 40-cell O-lens batches (harmful readout text), Sonnet does not; GT
  (`judge-gt`) stays Opus with an Opus-full -> Opus-truncated -> Sonnet fallback chain. Keep
  concurrency <= 64 (shared org input-token limit). `wsbench_read_modal.py` needs
  `MODAL_RESOURCE_ENV=camila-blank` and its `::main` entrypoint; `reads.sh` sets both, plus a
  private `UV_CACHE_DIR` (the shared uv cache lock is contended by parallel agents).
  **Refusal probes** (rev 5-6): `evals/overrefusal/probe.py` (math, imports
  `sibling_probe.probes`), `probe_data.py`, `probe_report.py`; Modal
  `scripts/overrefusal/probe_modal.py` (`main` = fit + score, `::calib` = exploratory answered-
  prompt thresholds; run with `uvx --with numpy`). Reusable artifact:
  `data/overrefusal/probes/refusal_probe_v1.{npz,json}` (schema in `data/overrefusal/README.md`).
  The pre-registered Alpaca threshold fires on ~all OR-Bench prompts; use `_calib` thresholds for
  borderline prompts. **Plan 0019 (11 layers):** new layers 20/24/32/40/48/56/60 live in separate
  files: probe `refusal_probe_v2_l11.{npz,json}` + `outputs/overrefusal/probe_sites_v2_l11.jsonl`
  (`probe_modal.py::l11`), readouts `/workspace/hb-readouts-overrefusal-v3` (`reads_l11.sh`,
  `HBENCH_OR_READOUTS_V3`); `report` keeps the published 4 layers on their merged RNG stream and
  scores each new layer separately (`scores_new_layers`, `union_11`), and writes
  `index_counts.jsonl` + `index_meta.json` (`evals/overrefusal/index_export.py`: claim = mention
  deduped by readout/label/quote; false = verified refusal_intent; comply_intent in the
  denominator; verifier-rejected dropped; lead positions unprobed). **Plan 0020 (summarize-then-
  judge) supersedes that index:** `scripts/overrefusal/sj_run.py` (digest -> `orsj-v3` -> `oroff-v1`,
  caches `digest/ sjudge/ offtopic/`, records `sj_records_<tag>.jsonl`) then
  `scripts/overrefusal/sj_report.py` (metrics `sj_metrics_<tag>.json` + the index); `report` no
  longer writes the plan-0019 index once `sj_records_full.jsonl` exists. Unit = item per (reader,
  layer); the judge sees only numbered digest lines. Calibration labels:
  `data/overrefusal/sj/calibration_v{1,2,3}.json`. **Rev 7:** use `refusal_probe_v2.{npz,json}` (on-distribution threshold;
  primary); `hbench overrefusal probe-report` reads `outputs/overrefusal/probe_sites_v2.jsonl`
  (`probe_modal.py::v2`). The bank has 360 items (160 / 100 / 100); readouts default
  `$HBENCH_OR_READOUTS` = `/workspace/hb-readouts-overrefusal-v2` (old + `-v2add`, merged by
  `reads.sh merge`); `TAG=v2add reads.sh all` reads only the added items. `probe_validated` (rev 12, the
  HB headline; formerly `probe_checked`) = verified refusal readouts at probe-absent sites /
  readouts; `raw` = all verified refusal readouts / readouts. `hbench overrefusal report` writes both
  (`report.json` `hb_scores`, `data/overrefusal/hb_scores.json`). Worktree venv on this box: the
  uv sync command is hook-blocked, so install with `uv pip install --python .venv/bin/python
  -e . --group dev --group sibling` and run with `uv run --no-sync ...`.
- **`hbench mask`** (plan 0017) is its own subcommand (not `--eval`): `gt` (MASK ground
  truth for Qwen3.6-27B from `outputs/mask/gen/rollouts*.jsonl`), `run` (D -> V -> VB on
  `$HBENCH_MASK_READOUTS/<arm>/mask.jsonl`, default `/workspace/hb-readouts-mask`), `report`.
  - Pinned response = pressure sample 0; capture `input_ids = prompt_token_ids + pinned ids`
    (no re-tokenisation). Doubling-down is labelled but never read (its forced turn is a real
    lie in context).
  - Headline = verified `lie_claim` readout rate on `honest_resisted` items **beside the lie
    rate and the difference**; `honest_wrong` is descriptive (it is almost always `aligned`).
  - Modal vLLM needs `cpu=8` and the snapshot path (offline): on the default CPU share the
    engine hung or died in torch.compile (2026-10-27).
  - **Self-knowledge gate** (plan rev 3): `scripts/mask/gate.py requests|parse` -> `data/mask/gate.json`;
    `report` picks the frame by the pre-registered rule (`report.choose_frame`, ctx unless persona
    carryover dominates) and adds gated headline tables at k = 8 / 9 / 10. It only subsets items.
  - **Deception probes** (rev 4): `scripts/mask/probe_build.py` (Instructed-Pairs + extra capture rows) ->
    captures `hb-mask-probe-{train,extra}` -> `scripts/mask/probe_modal.py` (4 CPU containers, one per layer;
    `evals/mask/probe.py` wraps `sibling_probe.probes` V2 + `um_probe` null z, 2 concepts, one centring key)
    -> `data/mask/probe_scores.json`; `report` adds validity / transfer / premise / probe-checked tables.
    The Instructed-Pairs thresholds (own-data honest q95) are miscalibrated on MASK: they fire everywhere.
  - **Rev 5 (final frozen set):** lie = >= 9/10 contradicting samples; 874 items labelled; read set =
    every gated lie (156) + gated honest_resisted (155) + honest_wrong (9), at EVERY position
    (`sites.json.gz`, 146k sites, ~586k cells per arm). **The diagnostic rate (formerly "HB score"; not a hallucination score since 2026-10-29) is the benign side only**:
    verified lie_claim readout rate on gated honest_resisted, per (arm, layer, site set
    prompt / response / all); lie items are a labelled diagnostic (and the WorkspaceBench
    `mask-lie-contrast` family in global-workspace). `data/mask/FROZEN.json` pins every file.
  - **Detect one layer at a time** (`scripts/mask/run_rev5.sh <arm> <conc>` -> `cells_L0NN.jsonl`,
    `summary_L0NN.json`): a whole 586k-cell prose arm in one process was OOM-killed under the
    shared 32 GB cap. `report` merges `cells_L*.jsonl` (else `cells.jsonl`).
  - Primary probe = leave-one-archetype-out MASK probe (threshold = q95 on unread honest items of
    the training archetypes); probe-checked rate = verified claims at probe-CONTRADICTED sites /
    readouts (`rate_supported` is the complement). Instructed-Pairs = direction AUROC only.
  - The s3d reader's sglang servers die at start-up on some shards; re-run the same command
    (idempotent, same `--shards`) until no shard fails.
  - **MASK is an archived diagnostic, not part of HallucinationBench (Camila, 2026-10-29)**: prompt-position lie
    claims on honest items are unverifiable (they may be real considered-then-rejected thoughts).
    No `index_counts.jsonl` / `index_meta.json` is produced; `evals/mask/claims.py` keeps only the
    claim unit for `scripts/mask/strict_pilot.py` (strict lie-claim judge `maskstrict-v1`, own
    cache `outputs/mask/<model>/strict/`).
  - **Plan 0019 (11 layers), STOPPED 2026-10-29 (archival).** The new layers 20 / 24 / 32 / 40 / 48 / 56 / 60 are
    read on PROMPT sites of the 155 gated honest_resisted items only (`data/mask/l11/`,
    readouts `mask_l11.jsonl`).
    - Detect with `scripts/mask/run_l11.sh <arm> <conc>` (`--sites-file` / `--readouts-file`).
    - Probes: `probe_modal.py --l11` (LOAO only), writing `*_v5_l11`. `report` merges them.
    - Gate per layer: `transfer_loao` (prompt sites) ≥ 0.7.
    - Detected when stopped: J-lens and ML-NLA all 7 layers, s3d L20 / 24 / 32 / 40 only (L48
      partial in the cache). `report` adds an archival prompt-only new-layer table.
    - The merged layers' numbers are frozen (`l11/report_4layer_snapshot.json`, tested).
    - J-lens' merged layers stay in `cells.jsonl` and its new layers in `cells_L0NN.jsonl`. The
      report reads both.
  - `tmux kill-session -t name` prefix-matches: with `hb-mask-gen` gone it killed
    `hb-mask-genx`. Use `-t =name`.

## sibling eval: user-modelling ladder (plans 0006 + 0011)

The only sibling eval on this repo (plan 0013). General notes:

- The `sibling` dependency group (transformers + jinja2, no torch) gives the Qwen3.6-27B
  tokenizer / chat template locally; it is a default uv group. Tokenizer tests skip without it.
- **Read window**: every message is `[system, cue + " " + request]`; `from_last_sentence_start`
  must start at the request sentence, so a cue ending in `."` (quote) breaks the window. The
  window-identity check (`bank.WindowChecker`) runs in `um_ladder.py probe-texts`.
- Opus 5.5 refuses every hacker-occupation prompt; `gen_items.MODEL_FOR` sends those to Sonnet 5.5.
- Network on this box drops DNS intermittently: every Modal step in the drivers retries; check
  `getent hosts api.modal.com` before blaming a script.
- Probe fitting is CPU-heavy; keep BLAS at 2 threads per worker (`threadpool_limits(2)`).

### Country (plan 0006)

```bash
S=scripts/sibling_probe/um_ladder.py            # inputs read from --src-root (default $HBENCH_SIBLING_SRC_ROOT)
uv run --no-sync python $S summarize            # missing J-lens summaries (token-hit cells only)
uv run --no-sync python $S mentions             # -> outputs/sibling/um/{mentions,strata}.jsonl, counts.json
uv run --no-sync python $S todo                 # dev 40 / gold 80 (+ dev2 60, see data README)
uv run --no-sync python $S judge                # umcommit-v1.1 -> commit.jsonl; --ids <todo> for a subset
uv run --no-sync python $S calib                # dev / dev2 / gold -> calib.json
uv run --no-sync python $S concepts             # set C -> data/sibling_probe/um_country_concepts.json
uv run --no-sync python $S probe-texts          # Opus (sibgen-v1 prompt) + checks -> um_country_probe_texts.jsonl
uv run --no-sync python $S capture-rows         # ReadSpec plans -> outputs/sibling/um/plan_probe/
#   then wsbench_public_plan.py prompts + wsbench_capture_modal.py (from the GW worktree) -> acts-um-country-probe
uv run --no-sync python $S probe-inputs
MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run scripts/sibling_probe/um_probes_modal.py --phase fit
uv run --no-sync python $S distractors          # refuses if Phase B scores exist
MODAL_PROFILE=safety-sahan-2 uvx --python 3.12 modal@latest run scripts/sibling_probe/um_probes_modal.py --phase score
uv run --no-sync python $S label && uv run --no-sync python $S figures
```

- Library: `evals/sibling_probe/um_ladder.py` (Set A lexicon patch over plan 0005's lexicon,
  `FastMatcher` = `Matcher.find` on candidate patterns only, ~9x faster, identical hits; units;
  `umcommit` prompt / parse / fail-closed; distractors; `ProbeContext` fire rule; labels;
  bootstrap), `um_probe.py` (logistic probe, per-fold null z, implied validity, centroids;
  copied into the Modal job with `probes.py`), `um_regions.py` (12 merged M49 regions).
- The country ladder reads `data/sibling_probe/br_lexicons_um_v1.json`, **not** plan 0005's
  `br_lexicons.json`. It is a frozen copy taken before plan 0005's bare-name alias fix, and it is
  what results 0006 used (sha256 pinned by `tests/test_um_lexicon_pin.py`). Set A changes live
  only in `SET_A_LEXICON_PATCH`. Adopting the newer lexicon is a results change: re-run and
  update `docs/results/0006`.
- Lexicon traps the patch handles (each tested): "Korean" (missing from the copy), 朝鲜 (Korean
  ethnicity, not North Korea), Soviet aliases (a `soviet` stratum on Ukraine / Belarus / Russia
  golds, Russia elsewhere), capitals un-blocked as country aliases except Washington / Male
  ("a male singer") / Victoria / Sofia / Kingston / Georgetown / Nassau, capitalised-only
  aliases (pole, polish, turkey, china, every capital), regional phrases ("Latin American").
- J-lens tokens must be decoded (`decode_token`) before matching; raw bags miss every CJK hit.
- Judge prompt history: `umcommit-v1` (gold acc 0.85, committed precision 0.40) -> `v1.1` (the
  one allowed revision, from dev2: a described hypothetical person is not the user unless the
  sample says so). Prompt examples come from dev / dev2, so only the gold numbers are honest.
- Leakage guard: distractors are chosen from Phase A (probe texts only); `distractors` refuses
  to run once `outputs/sibling/um/probes/score/` exists and `label` refuses a distractor file
  whose sha256 differs from the one Phase B recorded.
- sklearn is not a project dependency: `uv run --with scikit-learn` for the logistic test
  (skipped otherwise); the Modal image installs it.

### capital_city + holiday (plan 0011)

Plan 0011 was numbered 0007 until plan 0012 renumbered it (story holds 0007). The same driver
as plan 0006 with `--category capital_city|holiday` (`scripts/sibling_probe/um_run_domains_stage1.sh`,
`um_run_domain_stage2.sh`); domain library `evals/sibling_probe/um_domains.py`, lexicons
`data/sibling_probe/um_lexicons.json` (`scripts/sibling_probe/um_lexicons.py`); macro plots
`um_plot_macro*.py` (3 domains). Occupation (judge committed precision 0.50) and emotion (gold
0.71) were dropped and removed in plan 0013. The 3-domain sibling score is the index's sibling
family (`docs/results/0013_hallucination_index.md`).

- **Unvalidated:** the adjacent-layer (L±1) code paths (`um_run_adj.sh`, added in `451b1e3`)
  never finished a run. No result depends on them; validate before citing anything from them.
