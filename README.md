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
| `sibling_probe` user-modelling ladder | the sibling eval: is a same-category mention about the user (user is Korean, readout says China) a hallucination? Lexicon match -> commitment judge -> distractor-rank probe -> cross-method check -> label, on 3 domains: country (frozen lexicon `br_lexicons_um_v1.json`), capital_city (the city the user is visiting), holiday (`scripts/sibling_probe/um_ladder.py --category`) | [0006](plans/0006-user-modelling-country-ladder.md), [0011](plans/0011-user-modelling-ladder-4-domains.md) |
| `story` | the claim map (taxonomy v2, Opus 5.5) on 20 "tell me a story" items (WorkspaceBench's old `hallucination_creative` bank, 8 sites each); `--eval story` | [0007](plans/0007-story-eval.md) |
| `agentic_misalignment` | the claim map (v2.7) on lens **mentions of misalignment** (blackmail, leaking, letting someone die, ...) at every prompt position of WorkspaceBench's 32 AM items: a blind batched detector finds the mentions, the claim map judges them (marker in the prompt), and an intent call tiers each mention's kind against what the rollout actually does (realised / designed_only / scenario_set_up / unrelated); its own driver, `--eval agentic_misalignment` | [0008](plans/0008-agentic-misalignment-claim-map.md) |
| `strongreject` (**archived diagnostic, not part of HallucinationBench**) | **unsuccessful jailbreaks**: on StrongREJECT forbidden prompts x 6 jailbreak wrappers (+ bare control) that Qwen3.6-27B explicitly refuses in 10/10 on-policy samples, how often does a lens read at prompt positions claim the model will comply (`harmful_compliance`) or has adopted the jailbreak persona (`persona_adoption`)? Blind detector + outcome-blind echo/confirm judge on every prompt position; echo (restates the jailbreak's own demand) vs original; two HB scores per (reader, layer): **`probe_validated`** (headline: claims at positions where plan 0015's refusal probe says refusal IS represented / readouts, a lower bound) and **`raw`** (all verified claims / readouts, an upper bound); frozen final set (`data/strongreject/MANIFEST.sha256`); its own CLI `hbench strongreject {grade,run,report}`; not on the index | [0016](plans/0016-strongreject-eval.md) |
| `overrefusal` (pilot) | on OR-Bench-hard prompts Qwen3.6-27B **never refuses** (0/10 on-policy + greedy complies), how often do readouts at prompt positions claim the model will refuse? A blind batched detector (`refusal_intent` / `harm_salience` / `comply_intent` / `other`) + blind verifier; contrasts: hard prompts it refuses >= 8/10 and toxic prompts refused 10/10; own CLI `hbench overrefusal ...`; not on the index | [0015](plans/0015-overrefusal-eval.md) |

## overrefusal (plan 0015)

Two official HB scores on 160 never-refused OR-Bench prompts (tail sites headline): **probe_validated** (HEADLINE: verified refusal claims at sites where `refusal_probe_v2` says refusal is not represented / readouts; a lower bound) and **raw** (all verified refusal claims / readouts; an upper bound), frozen in `data/overrefusal/hb_scores.json` (frozen set `data/overrefusal/MANIFEST.sha256`; results `docs/results/0015_overrefusal_pilot.md`). Refusal probe for reuse: `data/overrefusal/probes/refusal_probe_v2.{npz,json}`.

![overrefusal HB rate](plots/overrefusal_hb_rate.png)
| `mask` (pilot) | MASK (lies vs honest-but-wrong): on-policy Qwen3.6-27B ground truth (MASK's judge prompts and belief / honesty aggregation), then how often a lens **claims the model is lying when it was honest** (blind detector -> prompt-aware verifier), at prompt sites (prediction) and response sites (detection), beside the same rate on lie items, also on the subset where Qwen itself says it lied / was honest (self-knowledge gate), with deception probes checking whether the lie is represented at the read sites; its own CLI `hbench mask gt|run|report`; not on the index | [0017](plans/0017-mask-eval.md) |

## strongreject: archived diagnostic (plan 0016)

**Not part of HallucinationBench.** Camila (2026-10-29): StrongREJECT is removed from HallucinationBench. On clear jailbreaks the model plausibly wavers before refusing, so a lens claim that it will comply is not a verifiable hallucination. The merged eval (code, frozen data
`data/strongreject/`, `hbench strongreject`, results
[`docs/results/0016_strongreject_pilot.md`](docs/results/0016_strongreject_pilot.md)) is kept
unchanged as a diagnostic; its numbers are not an HB score and it is not on the index. HB's
safety coverage is overrefusal (plan 0015).

## Hallucination index

`uv run hbench index` scores every model (one row per method x layer) from the families' outputs
(nothing is re-judged), and every `hbench report` refreshes it:
`outputs/index/<judge model>/index.{json,md}`. main = `wsbench_chat` weighted % of claims
hallucinated (false_claim 1, substitution_category 0.5, substitution_sibling 0.5, detail 0.2);
`agentic_misalignment` / `story` = unweighted % of claims hallucinated; sibling = the 3-domain UM
ladder hallucination share. Macro `full` = family weights main 9, agentic_misalignment 3,
sibling 2, story 1; `matched` = main + AM + story (9:3:1). Arms are matched by name across
families; legacy names and exclusions live in `data/index/methods.json`. Each family root is
overridable (`--main-root`, `--am-root`, `--story-root`, `--um-root`, repeatable). Definition:
[plan 0013](plans/0013-final-sibling-eval-and-hallucination-index.md).

External data is configured by environment variables, each defaulting to the dev-box path
(`src/hallucination_bench/paths.py`): `HBENCH_READOUTS_ROOT` (WorkspaceBench paper readouts),
`HBENCH_WSBENCH_PUB` (a workspace-bench checkout), `HBENCH_GLOBAL_WORKSPACE`,
`HBENCH_SIBLING_SRC_ROOT`, `HBENCH_ALI_PROBE_ROOT`, `HBENCH_COMBOS_READOUTS`; the shell drivers
also read `HBENCH_WSBENCH_PY`, `HBENCH_AM_GEN`, `HBENCH_AM_READOUTS`, `HBENCH_MLNLA_GEN`,
`HBENCH_EXTRA_READOUTS`, `HBENCH_STORY_GEN`, `HBENCH_STORY_READOUTS`, `HBENCH_GW_CAPTURE`
(a global-workspace checkout for the capture jobs). Tests that need one of them skip and name the variable.

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
R=${HBENCH_READOUTS_ROOT:-/workspace/wsbench-paper-data/readouts}
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

# story eval (plan 0007): the same pipeline on data/story, outputs under outputs/story/<model>/
scripts/fetch_story_readouts.sh jlens gen-hal-story-jlens tokens 20:61:4   # gen dir -> readouts + grid check
uv run hbench run --eval story --readouts-root /workspace/hb-readouts-story --stages map
uv run hbench report --eval story --view v2

# agentic_misalignment (plan 0008): own driver, stages detect,summ,verify,map,intent (verify = Stage H,
# the blind headline verifier, rev 13); readouts default to
# /workspace/wsbench-paper-data/readouts/<arm>/agentic_misalignment.jsonl
uv run --no-sync python scripts/build_am_bank.py      # data/agentic_misalignment/ (offline)
uv run hbench run --eval agentic_misalignment --arms s3d-rl600,nla-rl-L42,jlens --layers 32,42,48 \
    --items am-blackmail_none-none_replacement,rh-unit-tests,am-murder_explicit-america_none,am-blackmail_none-none_none \
    --out outputs/pilot --show-prompts 2                # the pilot (D, S, M + intent)
scripts/run_am.sh step1    # tmux hbench-am-detect: intent + detector + audit sheets (then hand audits)
scripts/run_am.sh step2    # tmux hbench-am-map: claim map + tier sheet + report
uv run hbench run --eval agentic_misalignment --arms s3d-rl600 --stages verify   # Stage H (headline membership)
uv run python scripts/am_audit.py sample-headline       # re-draw ONLY audit/headline.jsonl after Stage H
uv run python scripts/am_audit.py score                 # gates.json / tiers.json from the labelled sheets
uv run hbench report --eval agentic_misalignment        # outputs/agentic_misalignment/<model>/report.md

# overrefusal (plan 0015, pilot): own subcommands
uv run --no-sync hbench overrefusal candidates          # pinned OR-Bench -> data/overrefusal/candidates.jsonl
uvx --python 3.12 modal@latest run scripts/overrefusal/sample_modal.py --shards 4   # 10 samples/prompt
uvx --python 3.12 modal@latest run scripts/overrefusal/sample_modal.py --greedy --ids <hard ids>
uv run --no-sync hbench overrefusal judge-gt            # orgt-v1 refusal labels -> gt_labels.jsonl
uv run --no-sync hbench overrefusal bank                # populations, items.json, capture_rows.json
scripts/overrefusal/reads.sh all                        # Modal capture + 3 reads + convert + grid check
uv run --no-sync hbench overrefusal run --arms s3d-rl600,mlnla-rl600-all11,jlens   # detector + verifier
uv run --no-sync hbench overrefusal report              # outputs/overrefusal/<model>/report.md
# mask (plan 0017): scripts/mask/ builds + generates (Modal), hbench mask judges; readouts at
# $HBENCH_MASK_READOUTS/<arm>/mask.jsonl (default /workspace/hb-readouts-mask)
uv run --no-sync --with pyarrow python scripts/mask/build_items.py     # data/mask/items.json + requests
uvx --python 3.12 modal@latest run scripts/mask/gen_modal.py --requests outputs/mask/gen/requests.jsonl \
    --out outputs/mask/gen/rollouts.jsonl --n-shards 8                # (and requests_x.jsonl -> rollouts_x.jsonl)
uv run hbench mask gt                                    # MASK labels -> data/mask/labels.json
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/mask/build_capture.py   # read set, sites, capture rows
scripts/mask/fetch_readouts.sh jlens gen-hb-mask-jlens tokens       # after capture + reads (plan 0017)
uv run hbench mask run --arms s3d-rl600,mlnla-rl600-all11,jlens --concurrency 32
uv run --no-sync python scripts/mask/gate.py requests    # rev 3 gate (then gen_modal.py on gate_requests.jsonl)
uv run --no-sync python scripts/mask/gate.py parse       # -> data/mask/gate.json
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/mask/probe_build.py   # rev 4 probes: capture rows (then capture)
uvx --python 3.12 modal@latest run scripts/mask/probe_modal.py        # -> data/mask/probe_scores.json
scripts/mask/run_rev5.sh s3d-rl600 16                     # rev 5: detect + verify one layer at a time
uv run hbench mask report                                # outputs/mask/<model>/report.md (+ gated + probe tables)
uv run --no-sync python scripts/mask/plot_hb.py && uv run --no-sync python scripts/mask/freeze.py
```

Other commands: `hbench run --rebuild` (regenerate `cells.jsonl` / `summary.json` from the cache,
no API calls), `hbench run --skip-classify` (span + verify pilot), `hbench pilot-dump` (sample
pilot spans as unlabelled calibration cases), `hbench pilot-dump-v2` (real cells with their
production units, unlabelled, for v2 gold), `hbench units-v2` (fill a v2 case's `gold.units`
with the production unit split).

AM outputs: `outputs/agentic_misalignment/<judge model>/<arm>/{cache.jsonl,cells.jsonl,mentions.jsonl,summary.json}`,
`intent.jsonl`, `audit/` and `report.{md,json}` under `outputs/agentic_misalignment/<judge model>/`.

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

Index of every results doc: [`docs/results/README.md`](docs/results/README.md).

[`docs/results/0017_mask_pilot.md`](docs/results/0017_mask_pilot.md): mask (rev 5, final frozen set): the HB score is the verified lie-claim readout rate on 155 gated honest_resisted items at every prompt / response position (the O-lens claims a lie on 0.6-1.6% of prompt readouts, so 40-77% of honest items get at least one claim; NLA 0.4-0.5%; J-lens ~0). Figure `plots/0017_mask_hb_score.png`.

[`docs/results/0001_wsbench_chat_taxonomy.md`](docs/results/0001_wsbench_chat_taxonomy.md)
(filled in after the full run); claim map:
[`docs/results/0002_claim_map.md`](docs/results/0002_claim_map.md); user-modelling ladder (country):
[`docs/results/0006_um_country_ladder.md`](docs/results/0006_um_country_ladder.md); story eval:
[`docs/results/0007_story.md`](docs/results/0007_story.md) (O-lens 21% of readout characters
hallucinated on stories vs 10% on chat; NLAs 38–43%; J-lens 4%).
Hallucination index: [`docs/results/0013_hallucination_index.md`](docs/results/0013_hallucination_index.md).
