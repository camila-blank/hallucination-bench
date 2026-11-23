# Commands

Every pipeline's commands, as run for the results of record. Start with `hbench try` in the [README](../README.md#try-it-on-two-prompts).

Options are `key=value` and a boolean is `--key` (every command is a pydra config: `--show` prints
the resolved config, `--help` lists the keys). The `--some-flag value` spelling also works.

```bash
uv sync                                   # Python >= 3.12; dev tools come with the dev group
uv run pytest && uv run ruff check && uv run ruff format --check

# smoke: 5 cells each of three arms (keyless Anthropic auth via the `ant` CLI profile)
R=${HBENCH_READOUTS_ROOT:-/workspace/wsbench-paper-data/readouts}
uv run hbench run out=outputs/smoke readouts_root=$R arms=s3d-rl600,nla-rl-L42,jlens \
    layers=42,44 limit=5 show_prompts=5
uv run hbench report out=outputs/smoke

uv run hbench calibrate split=dev       # classifier vs hand-labelled dev cases
uv run hbench calibrate split=heldout   # the gate (held-out only)

# claim map: the map stage, then the claim-map report
uv run hbench run out=outputs readouts_root=$R arms=s3d-rl600 layers=44 limit=10 \
    stages=map show_prompts=3
uv run hbench report out=outputs view=claim_map   # report_v2.md, probe_candidates.jsonl, audit_samples_v2.md
uv run hbench calibrate taxonomy=claim_map split=dev


# story eval: the same pipeline on data/story, outputs under outputs/story/<model>/
scripts/fetch_story_readouts.sh jlens gen-hal-story-jlens tokens 20:61:4   # gen dir -> readouts + grid check
uv run hbench run eval=story readouts_root=/workspace/hb-readouts-story stages=map
uv run hbench report eval=story view=claim_map

# agentic_misalignment: own driver, stages detect,summ,verify,map,intent (verify = Stage H,
# the blind headline verifier); readouts default to
# /workspace/wsbench-paper-data/readouts/<arm>/agentic_misalignment.jsonl
uv run --no-sync python scripts/build_am_bank.py      # data/agentic_misalignment/ (offline)
uv run hbench run eval=agentic_misalignment arms=s3d-rl600,nla-rl-L42,jlens layers=32,42,48 \
    items=am-blackmail_none-none_replacement,rh-unit-tests,am-murder_explicit-america_none,am-blackmail_none-none_none \
    out=outputs/pilot show_prompts=2                # a pilot on four items (detect, summ, map + intent)
uv run hbench run eval=agentic_misalignment arms=s3d-rl600 stages=verify   # Stage H (headline membership)
uv run python scripts/am_audit.py sample-headline       # re-draw ONLY audit/headline.jsonl after Stage H
uv run python scripts/am_audit.py score                 # gates.json / tiers.json from the labelled sheets
uv run hbench report eval=agentic_misalignment        # outputs/agentic_misalignment/<model>/report.md

# the index: claim maps with the claimmap-v2.9 prompt and the continuation verifier (on by
# default with that prompt; the cverify stage runs after map), joint cells, the AM situation
# judge, then the index and the final plots
uv run hbench run out=outputs/v8 bank=x3 map_prompt=claimmap-v2.9 stages=map,cverify arms=s3d-rl600,...
uv run hbench run out=outputs/v8 bank=x3 map_prompt=claimmap-v2.9 --joint arms=s3d-rl600-all11 \
    joint_from=s3d-rl600,s3d-rl600-all11     # one judge call per site over all 11 layers
uv run --no-sync python scripts/v8/seed_am_root.py --src outputs/agentic_misalignment/claude-opus-5-5 \
    --dest outputs/v8/agentic_misalignment/claude-opus-5-5 --arms s3d-rl600,mlnla-rl600-all11,jlens
uv run hbench run eval=agentic_misalignment out=outputs/v8 map_prompt=claimmap-v2.9 \
    judge_model=claude-opus-5-5 map_model=claude-sonnet-5-5 stages=map,cverify,joint,situation
uv run hbench index version=v8 out=outputs/v8 ...
uv run hbench final manifest=data/index/v8_manifest.json out=outputs/final/

# overrefusal: own subcommands
uv run --no-sync hbench overrefusal candidates          # pinned OR-Bench -> data/overrefusal/candidates.jsonl
uvx --python 3.12 modal@latest run scripts/overrefusal/sample_modal.py --shards 4   # 10 samples/prompt
uvx --python 3.12 modal@latest run scripts/overrefusal/sample_modal.py --greedy --ids <hard ids>
uv run --no-sync hbench overrefusal judge-gt            # orgt-v1 refusal labels -> gt_labels.jsonl
uv run --no-sync hbench overrefusal bank                # populations, items.json, capture_rows.json
scripts/overrefusal/reads.sh all                        # Modal capture + 3 reads + convert + grid check
uv run --no-sync hbench overrefusal run arms=s3d-rl600,mlnla-rl600-all11,jlens   # detector + verifier
uv run --no-sync hbench overrefusal report              # outputs/overrefusal/<model>/report.md
uv run --no-sync python scripts/overrefusal/sj_run.py --tag full --concurrency 64   # digest -> orsj-v3 -> oroff-v1
uv run --no-sync python scripts/overrefusal/sj_report.py --tag full                 # metrics 1/2 + index_counts.jsonl
uv run --no-sync hbench overrefusal offtopic-export     # the index root offtopic_index/ + sj_offtopic_full.json
```

Other commands: `hbench run --rebuild` (regenerate `cells.jsonl` / `summary.json` from the cache,
no API calls), `hbench run --skip_classify` (span + verify pilot), `hbench pilot-dump` (sample
pilot spans as unlabelled calibration cases), `hbench pilot-dump-units` (real cells with their
production units, unlabelled, for claim-map gold), `hbench gold-units` (fill a claim-map case's
`gold.units` with the production unit split).

AM outputs: `outputs/agentic_misalignment/<judge model>/<arm>/{cache.jsonl,cells.jsonl,mentions.jsonl,summary.json}`,
`intent.jsonl`, `audit/` and `report.{md,json}` under `outputs/agentic_misalignment/<judge model>/`.

Outputs: `outputs/wsbench_chat/<judge model>/<arm>/{cache.jsonl,cells.jsonl,summary.json,summary_v2.json}`
and `outputs/wsbench_chat/<judge model>/{report.md,report.json,audit_samples.md}` (claim map:
`report_v2.{md,json}`, `probe_candidates.jsonl`, `audit_samples_v2.md`).
