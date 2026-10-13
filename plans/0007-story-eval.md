# Plan 0007: `story` eval: the claim map (taxonomy v2) on "tell me a story" readouts

Author: Camila Blank (planned with Claude), 2026-10-12. Revision 3 (critique 1: report contexts, pre-gate, consistency scope, summary invariance, bank commit, Modal job file; critique 2: pilot dumps stay chat, story-specific report notes, exact run.py sites; critique 3: defaulted signatures, literal layer lists; no substantive issues remain).
Builds on plan 0002 (claim map, rev 20) and plan 0001 (pipeline, cache, report). Everything not
restated here is unchanged.

## Why

WorkspaceBench once had a creative hallucination family (`hallucination_creative`, 20 story
items, judged by the Sonnet v5b on-topic judge). It was dropped when the chat family became the
headline. Camila (2026-10-12): bring it back as a subeval of HallucinationBench v2, judged with
the claim map (the "v2 judge", plan 0002) on **Opus 5.5**, and see what the results are.

Stories are a different regime from chat. Almost every specific in a story (names, places,
objects) is invented by the response itself. So a lens that reads a story has to track
arbitrary specifics, and plan 0002's `detail` leaf (a wrong arbitrary name or number) and the
`correct` source tags should carry most of the signal.

## Scope

- **Bank:** the 20 frozen `hallucination_creative` items, reused as they are (Camila's choice:
  this keeps item-level comparability with the old run). Each item is thinned to 8 read sites by
  the same rule the old judge used (below), so each layer has 160 cells.
- **Arms** (Camila's choice):

  | arm | lens | layers |
  |---|---|---|
  | `s3d-rl600-all11` | O-lens RL step 600 | 20:61:4 (11) |
  | `mlnla-rl600-all11` | multilayer NLA RL step 600 | 20:61:4 (11) |
  | `jlens` | J-lens n1000 top-10, through the interp-v1 summariser | 20:61:4 (11) |
  | `nla-rl-L42` | NLA-RL | 42 only (its capture layer) |

  These are the same arm names and read recipes as the plan-0002 full run, so story and chat
  results sit side by side.
- **Judge:** the claim map exactly as in plan 0002 rev 20: prompt `claimmap_v2.PROMPT_VERSION`
  unchanged, single-call mode (`map_anchor = none`), and `claude-opus-5-5` for every call
  (summary, span subsample, map) at effort `low`. Those are the existing CLI defaults, so they
  are restated here rather than changed.
  - No prompt edits for stories. The prompt already talks about "the conversation", and its
    rulings carry over. A reasonable continuation is consistent. An invented arbitrary specific
    with nothing in the conversation for it to replace is not a hallucination, but a wrong name
    for an established character is `detail`. Changing the prompt would make story and chat
    rates incomparable.

## Bank

New directory: `data/story/`, built by `scripts/build_story_bank.py` (deterministic, re-runnable).

- **Input:** WorkspaceBench `evals/workspace-bench/hillclimbing_evals/hallucination/creative/`
  `{items.json, capture_rows.json}` in `camilablank/global-workspace`. The path is a CLI
  argument, defaulting to `/workspace/global-workspace/...`. The builder records the source
  repo's `git rev-parse HEAD` and the source files' sha256 in `meta`.
- **Thinning:** exactly `judge.py`'s rule in the source repo: if `len(sites) > 8`,
  `step = len(sites) / 8` and keep `sites[int(i * step)] for i in range(8)`. The old creative
  judge and `chat_bank.py` use the same rule, so the story sites are the cells the old run judged.
- **`data/story/items.json`:** `{meta, items: [{id, stratum: "story", source:
  "hallucination_creative", prompt, response, response_start, sites: [{pos, token, kind,
  char}]}]}`. This is the same item schema `wsbench_chat` reads, so the pipeline needs no new
  fields. The source's other keys (`facts`, `rollout`, `screen`, ...) are dropped.
- **`data/story/capture_rows.json`:** the source capture rows verbatim, except that
  `eval_positions` is replaced by the item's thinned site positions (sorted). This is the capture
  record that produces the readouts. The judge never reads it.
- **Builder asserts** (fail loudly):
  - 20 items and 8 sites each;
  - every kept site's `pos` is in the source row's `eval_positions`;
  - `response[:char]` ends with the site token, after decoding the byte-level `Ġ`/`Ċ` forms;
  - no `eos` site is kept. A response's final `<|im_end|>` site follows a 512-token truncation,
    not a real ending, and the thinning never reaches it (checked: the kept kinds are clause 62,
    sentence 52, newline 30, markup 10, quote 6).
- **Known property, kept:** responses were capped at ~512 tokens, so most stories stop
  mid-plot. The text after the marker is still the reference, and the "reasonable continuation"
  ruling covers readouts that extrapolate past the cut.

## Code changes (hallucination-bench)

The eval becomes a parameter. **`wsbench_chat` behaviour is retained byte for byte**: the output
layout `outputs/wsbench_chat/<model>/<arm>/`, cache keys, fingerprints, prompts, `summary.json` /
`summary_v2.json` contents (`config.models` is unchanged because `eval` is NOT in
`Models.to_dict()`; `config.eval` already says `wsbench_chat`), the gate logic for chat, and every
existing test are unchanged. The only chat-visible change is a new `"eval"` key in `arm.json`.
**No pre-existing test may be modified.**

1. `src/hallucination_bench/evals/story/__init__.py` (`NAME = "story"`, docstring) and
   `evals/story/bank.py`, with the same public API as `wsbench_chat/bank.py`: `load_bank`,
   `load_items`, `sites_by_item`, `positions`, `marked_response`, `expected_grid`, `BANK_SOURCE`,
   plus `BANK_COMMIT` (the source commit read from `meta`). `wsbench_chat/bank.py` gains
   `BANK_COMMIT = WSBENCH_PUB_COMMIT` so both banks expose the same names. Share the pure helpers rather than
   copying them: move `sites_by_item`, `positions`, `marked_response` and `expected_grid` into
   `evals/_bank_common.py`, and have both bank modules re-export them. `wsbench_chat.bank` keeps
   its names. Each bank's `load_bank` is still `@cache`d.
2. `evals/__init__.py`: register `"story": "hallucination_bench.evals.story"` in `EVALS`, and
   add `bank_for(name) -> ModuleType`, which imports `f"{EVALS[name]}.bank"` and raises
   `KeyError` on an unknown name, as `get` does. `story` has only its own bank. The pipeline
   (`run`, `score`, `score_v2`) stays in `wsbench_chat/`, is shared, and is not moved: a move
   would churn every import for no behavioural gain. The registry docstring says so.
3. `wsbench_chat/run.py`:
   - `Models` gets `eval: str = "wsbench_chat"`. It is **not** added to `to_dict()` (that dict is
     `summary.json`'s `config.models`); `_write_arm_json` writes it as a top-level `"eval"` key.
     An `arm.json` without the key reads as `wsbench_chat`.
   - `arm_dir` uses `models.eval` instead of `NAME`.
   - `Arm` gets a `bank: ModuleType` field, set in `open_arm` from `bank_for(models.eval)`.
   - Every module-level `bank.` use in `run.py` goes through `arm.bank` (or the bank resolved in
     `open_arm`): `open_arm` (`load_items`, `positions`, `sites_by_item`), the classify-stage
     `tv1.render_prompt` (`:470`), `_map_stage`'s render (`:517`) and quote conversation
     (`:536`), `select` (`:577`) and `rebuild_arm_async` (`:765`). The span stage calls
     `v5c.judge_prompt(item, site, …)` and needs no change. `run.py` no longer imports `bank`.
   - The rebuild summary records `"eval": models.eval`, `"bank_source": arm.bank.BANK_SOURCE`
     and `"bank_commit": arm.bank.BANK_COMMIT` (the `WSBENCH_PUB_COMMIT` import in `run.py` goes).
   - **Pre-gate:** `pre_gate = True` whenever `models.eval != "wsbench_chat"`; for chat it stays
     `not gate_passed(out_root, mapper, anchor)` exactly. The held-out gate is chat gold, so a
     chat pass must never un-label story output.
   - The v5c subsample (`in_v5c_subsample`, seeded 10% by item hash) is unchanged. On 20 story
     items it selects about 2 items; `summary_v2.json` already reports the comparison's n.
   - No cache key or fingerprint includes the eval: story and chat caches are separate files
     because the arm directories differ.
4. `report.py`, `report_v2.py`:
   - `report()` takes `eval_name: str = "wsbench_chat"` (keyword, defaulted). The model root
     becomes `out_root / eval_name / slug` (`report.model_root`, `report_v2.model_root`), the
     bank comes from `bank_for(eval_name)`, and `Models` objects built from `arm.json` carry it.
   - The report path's only bank use is `report._context` (`report.py:164-167`), called by
     `report.audit_samples` and `report_v2._sample_lines`. `_context`, `audit_samples` and
     `_sample_lines` (and `report_v2.audit_samples`) gain a `bank` parameter, defaulted to the
     chat bank, threaded from `report()`.
   - `report.pilot_dump` and `report_v2.pilot_dump_v2` stay **chat-only** and unchanged in
     behaviour: they use `bank_for("wsbench_chat")` and `model_root(out_root, judge_model)`, whose
     new `eval_name` parameter defaults to `"wsbench_chat"`.
   - Signatures: `report_v2.build(summaries, judge_model, noise=None, *, eval_name="wsbench_chat")`,
     `noise_section(noise, eval_name="wsbench_chat")`, `report.build(summaries, judge_model, *,
     eval_name="wsbench_chat")`; the `Models` each report builds from `arm.json` gets
     `eval=eval_name` (the CLI argument), never a fallback read from `arm.json`.
   - Story-specific report text: for a
     non-chat eval the noise section says "judge-noise sample is chat-only; not estimated for
     story" (never "run `hbench consistency-v2`"), and the pre-gate note says "the held-out gate
     is chat gold; story has no gate of its own, so these numbers are provisional". Chat text is
     unchanged.
   - Report titles use the eval name ("# story claim map (taxonomy v2) — judge …").
   - The judge-noise column reads `outputs/consistency/<model>/consistency_<v>.json` only when
     `eval_name == "wsbench_chat"`, since the consistency sample is chat. For story it shows —,
     and the report notes that the noise sample is chat-only.
   - `consistency_v2.py` is **not** changed: it stays chat-only (its output file is the chat
     judge-noise source that `report_v2.load_noise` reads, so a story run there would overwrite
     chat noise).
5. `cli.py`: `--eval {wsbench_chat,story}` (default `wsbench_chat`, choices from `EVALS`) on
   `run` and `report` only. It sets `Models.eval` and is passed through as `eval_name`.
   `consistency-v2`, `calibrate`, `pilot-dump*` and `units-v2` stay chat-only.
6. Tests (new files only):
   - `tests/test_story_bank.py`: `data/story` has 20 items × 8 sites, `response[:char]` ends with
     the decoded token, `capture_rows` `eval_positions` equals the item's site positions, there
     are no `eos` sites, and `expected_grid(items, [42])` has 160 cells.
   - `tests/test_story_eval.py`, offline with no API: a tiny readouts file on two story sites. An
     `open_arm` with `Models(eval="story")` loads story items and writes under
     `out/story/<model>/<arm>/`; a `rebuild` with an empty cache reports the story grid as
     pending and `eval: story`, `bank_source` story, `pre_gate: true` even when a passing chat
     gate file exists; the same with default `Models` still resolves `wsbench_chat` paths and
     `summary.json` `config.models` has no `eval` key; a fixture story arm with a cached `:M`
     row renders through `report_v2.report(out, judge, eval_name="story")` (writes
     `report_v2.md` and `audit_samples_v2.md` under `out/story/<model>/`) and through
     `report.report(..., eval_name="story")`, with the story noise and gate strings asserted and
     no "consistency-v2" advice; `run.py`'s source has no remaining `bank.` attribute use outside
     `arm.bank` (an AST/regex check); and the CLI parser accepts `--eval story` on
     run and report and rejects an unknown eval.
7. Docs: README (evals table row, story quickstart), CLAUDE.md (layout lines for
   `evals/story`, `data/story`, `scripts/build_story_bank.py`, and the readouts root for story),
   and `data/story/README.md` (provenance, thinning, the 512-token truncation note).

## Readouts (global-workspace, Modal; no code changes there)

These are inputs, not in git: `/workspace/hb-readouts-story/<arm>/hallucination.jsonl`.

1. **Capture** (one H200; the 27B on 20 rows, cheap), from the global-workspace checkout:
   ```
   PYTHONPATH=src uvx --python 3.12 modal@latest run scripts/olens_suite/workspace_bench/wsbench_capture_modal.py \
     --prompts-json /workspace/hb-story/data/story/capture_rows.json \
     --acts-subdir acts-hallucination-story --layers 20:61:4 --max-positions 640
   # and --acts-subdir acts-hallucination-story-L42 --layers 42:43
   ```
   This uses new subdir names. The old `acts-hallucination-creative*` names are not reused,
   because their committed manifests list every site, and these captures target only the 8
   thinned sites.
2. **Reads** (k = 3 prose; tokens top-10):
   - `s3d-rl600-all11`: `wsbench_read_modal.py --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 --run hal-story-s3d-rl600-all11 --acts-subdir acts-hallucination-story --layers 20:61:4 --k 3 --shards 16 --merged-name s3d-rl600` (the chat run's merged dir, reused).
   - `nla-rl-L42`: `wsbench_nla_modal.py --run hal-story-nla-rl-L42 --adapter av_rl_adapters/iter_000400 --acts-subdir acts-hallucination-story-L42 --layers 42 --n-samples 3`.
     The chat arm read `acts-hallucination-chat-L42cap`, which is the same capture tool at
     `layers [42]` (verified: manifest `layers: [42]`, all positions), so the conventions match.
   - `jlens`: `wsbench_tokenlens_modal.py --lens jlens --jobs-json logs/jlens_story_jobs.json`,
     where the file holds a JSON **list**:
     `[{"acts": "acts-hallucination-story", "out": "gen-hal-story-jlens", "layers": [20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]}]`.
     The script's `jlens` artifact is the n1000 wikitext lens the chat `-x6` reads used; confirm
     against `gen-hal-chat-jlens-x6`'s config before reading.
   - `mlnla-rl600-all11`: from `/workspace/gw-mlnla` (branch `camila/multilayer-nla`; the
     `--lens` registry is not on main): `wsbench_nla_modal.py --lens mlnla-rl600 --run hal-story-mlnla-rl600-all11 --acts-subdirs acts-hallucination-story --layers 20,24,28,32,36,40,44,48,52,56,60 --n-samples 3 --shards 16`.
   - Shards follow the Modal fan-out rule (16 for the 220-unit multilayer reads).
3. **Convert:** `scripts/fetch_story_readouts.sh <arm> <gen> <prose|tokens> <20:61:4|42>` fetches each gen dir
   (volume `wsbench-gen-3`; every read sets `WSBENCH_GEN_VOLUME=wsbench-gen-3`) and runs
   `python -m wsbench convert-gen-dir gen_dir=… out=/workspace/hb-readouts-story/<arm>/hallucination.jsonl kind=prose|tokens`
   (wsbench-pub @ 92d763e).
   - Check: each file covers every (item, site, layer) of the expected grid: 160 × 11 rows, or
     160 for NLA.
   - `hbench run` reports missing cells as fatal unless `--allow-missing`; this run does not
     pass it.

## Run

In tmux session `hbench-story`, logging to `logs/story_<ts>.log`:

```
uv run --no-sync hbench run --eval story --readouts-root /workspace/hb-readouts-story \
  --arms s3d-rl600-all11,mlnla-rl600-all11,jlens,nla-rl-L42 --stages map --out outputs
# no --layers: each arm judges every layer in its file (11 or just 42)
uv run --no-sync hbench report --eval story --view v2 --out outputs
```

- **Calls:** 160 × 34 arm-layers = 5,440 `:M` calls, 1,760 J-lens summaries, and the span
  subsample.
- **Retries:** exit 4 (cached failures) triggers a re-run, which resumes.
- **Outputs:** `outputs/story/claude-opus-5-5/{report_v2.md,report_v2.json,…}`.

## Results

`docs/results/0007_story.md`:
- per-arm verdict shares and readout-level rates on stories, with the three hallucination
  components separate, beside the same arms' chat numbers from `0002_report_full_v2.7.md`;
- per-layer for the 11-layer arms;
- 10 hand-read examples per arm.

It also carries the same caveats as chat: pre-gate (the held-out gate is chat gold), no
judge-noise estimate on stories, and n = 20 items, so the item-bootstrap CIs are wide.

## Out of scope

- A larger story bank (fresh rollouts).
- Story calibration gold or consistency runs.
- Any prompt change.
- WorkspaceBench integration.
