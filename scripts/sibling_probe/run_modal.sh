#!/usr/bin/env bash
# Plan 0003 Phase B/C GPU commands (sibling-concept probes). Run from the hallucination-bench
# root; the GPU scripts live in the global-workspace worktree $GW and are run from there
# (never edited). One stage per invocation:  scripts/sibling_probe/run_modal.sh <stage>
#
#   qwen-implication   implication check of the generated candidates (Qwen3.6-27B, greedy)
#   qwen-responses     greedy responses for every bank item (max 256 new tokens)
#   plans              ReadSpec plan files + s3d specs + jlens jobs, then `prompts` twice
#                      (eval / probe) + check-prompts (window identity, eval_positions)
#   smoke              2 eval items: plans -> prompts -> capture (acts-sibling-smoke) -> manifest check
#   capture            both captures (acts-sibling-eval, acts-sibling-probe), layers 28,36,42,44,52
#   jlens              J-lens over the EVAL capture only, layers 28,36,44,52, k=10
#   nla                NLA-RL (nla-rl400) over the EVAL capture only, layer 42
#   s3d                s3d-rl600 O-lens over the eval items (fanout_custom), layers 28,36,44,52
#   fetch              fetch gen dirs + s3d shards, fill gaps, convert, merge, cell-count checks
#   probes             DiffMean probes (probes_modal.py, CPU): prepare inputs -> fit/score on
#                      acts-sibling-probe + acts-sibling-eval -> outputs/sibling/probes/
#   probes-smoke       the same on acts-sibling-smoke with synthetic probe texts
#
# Stage 1 status (2026-10-11): qwen-implication, qwen-responses, plans and smoke have been run;
# capture / jlens / nla / s3d / fetch are documented, not run yet.
set -euo pipefail

HB=$(cd "$(dirname "$0")/../.." && pwd)
GW=${GW:-/workspace/gw-wrong-bridge}
WSB_SRC=${WSB_SRC:-${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}/src}
WB=$GW/scripts/olens_suite/workspace_bench
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}   # default environment: camila-blank
MODAL="uvx --python 3.12 modal@latest"
export WSBENCH_GEN_VOLUME=${WSBENCH_GEN_VOLUME:-wsbench-gen-sibling}  # gen-2/-3/-4 full (ENOSPC 2026-10-11)
S=$HB/outputs/sibling
GROUP=L28-36-42-44-52            # the one layer group `prompts` writes for both plans
MAXPOS=128                       # >= groups.json max_tokens (eval 78, probe 90)
cd "$HB"

case "${1:-}" in
qwen-implication)
  $MODAL run scripts/sibling_probe/qwen_gen_modal.py \
    --inp $S/gen/implication_in.jsonl --out $S/gen/implication_out.jsonl
  ;;
qwen-responses)
  $MODAL run scripts/sibling_probe/qwen_gen_modal.py \
    --inp $S/gen/responses_in.jsonl --out $S/gen/responses_out.jsonl
  uv run python scripts/sibling_probe/gen_items.py responses
  ;;
plans)
  uv run python scripts/sibling_probe/write_plans.py plans
  for t in eval probe; do
    uv run python $WB/wsbench_public_plan.py prompts --plan-dir $S/plan_$t \
      --out-dir $S/render_$t --wsbench-src $WSB_SRC
    uv run python scripts/sibling_probe/write_plans.py check-prompts --render-dir $S/render_$t
  done
  ;;
smoke)
  uv run python scripts/sibling_probe/write_plans.py plans \
    --ids um-sd-country-01,gen-tree-maple-a0-00 --tag smoke
  uv run python $WB/wsbench_public_plan.py prompts --plan-dir $S/plan_eval_smoke \
    --out-dir $S/render_eval_smoke --wsbench-src $WSB_SRC
  uv run python scripts/sibling_probe/write_plans.py check-prompts --render-dir $S/render_eval_smoke
  (cd "$GW" && PYTHONPATH=src $MODAL run $WB/wsbench_capture_modal.py \
    --prompts-json $S/render_eval_smoke/prompts/$GROUP.json --acts-subdir acts-sibling-smoke \
    --layers 28,36,42,44,52 --max-positions $MAXPOS --strict --force)
  mkdir -p $S/smoke
  $MODAL volume get --force wsbench-acts /acts-sibling-smoke/manifest.json $S/smoke/manifest.json
  uv run python scripts/sibling_probe/write_plans.py check-manifest \
    --render-dir $S/render_eval_smoke --manifest $S/smoke/manifest.json
  ;;
capture)
  for t in eval probe; do
    (cd "$GW" && PYTHONPATH=src $MODAL run $WB/wsbench_capture_modal.py \
      --prompts-json $S/render_$t/prompts/$GROUP.json --acts-subdir acts-sibling-$t \
      --layers 28,36,42,44,52 --max-positions $MAXPOS --strict)
    mkdir -p $S/acts_manifest
    $MODAL volume get --force wsbench-acts /acts-sibling-$t/manifest.json \
      $S/acts_manifest/$t.json
    uv run python scripts/sibling_probe/write_plans.py check-manifest \
      --render-dir $S/render_$t --manifest $S/acts_manifest/$t.json
  done
  ;;
jlens)
  # jobs: [{"acts": "acts-sibling-eval", "out": "gen-sibling-eval-jlens", "layers": [28,36,44,52]}]
  (cd "$GW" && PYTHONPATH=src $MODAL run --detach $WB/wsbench_tokenlens_modal.py \
    --lens jlens --jobs-json $S/jlens_jobs.json --k 10)
  ;;
nla)
  (cd "$GW" && PYTHONPATH=src $MODAL run --detach $WB/wsbench_nla_modal.py \
    --lens nla-rl400 --run sibling-eval-nla-rl400 --acts-subdirs acts-sibling-eval \
    --layers 42 --shards 4)
  ;;
s3d)
  # outputs wsbench-public-readouts:/s3d-rl600/custom_sibling_eval.<i>of8.jsonl
  (cd "$GW" && $MODAL run --detach $WB/wsbench_public_modal.py::fanout_custom \
    --arms s3d-rl600 --name sibling_eval --specs $S/s3d_specs.json --n-shards 8)
  ;;
fetch)
  mkdir -p $S/gens $S/readouts/{s3d-rl600,jlens,nla-rl-L42} $S/s3d_shards
  for g in gen-sibling-eval-jlens gen-sibling-eval-nla-rl400; do
    $MODAL volume get --force $WSBENCH_GEN_VOLUME /$g $S/gens/
  done
  # `modal volume get` can silently leave gaps: verify per item vs meta and re-get
  uv run python $WB/wsbench_combos_fill.py --gen-dir $S/gens/gen-sibling-eval-jlens \
    --volume $WSBENCH_GEN_VOLUME --meta-dir $S/render_eval/meta --layers 28,36,44,52
  uv run python $WB/wsbench_combos_fill.py --gen-dir $S/gens/gen-sibling-eval-nla-rl400 \
    --volume $WSBENCH_GEN_VOLUME --meta-dir $S/render_eval/meta --layers 42
  # convert exits non-zero on missing cells
  uv run python $WB/wsbench_public_plan.py convert --gen-dir $S/gens/gen-sibling-eval-jlens \
    --meta-dir $S/render_eval/meta --out-dir $S/readouts/jlens --kind tokens --layers 28,36,44,52
  uv run python $WB/wsbench_public_plan.py convert --gen-dir $S/gens/gen-sibling-eval-nla-rl400 \
    --meta-dir $S/render_eval/meta --out-dir $S/readouts/nla-rl-L42 --strip-nla --layers 42
  for i in 0 1 2 3 4 5 6 7; do
    $MODAL volume get --force wsbench-public-readouts \
      /s3d-rl600/custom_sibling_eval.${i}of8.jsonl $S/s3d_shards/
  done
  cat $S/s3d_shards/custom_sibling_eval.*of8.jsonl > $S/readouts/s3d-rl600/sibling_eval.jsonl
  # cell-count checks (exit 2 on any missing / duplicated cell or token mismatch)
  uv run python scripts/sibling_probe/write_plans.py check-cells --meta-dir $S/render_eval/meta \
    --readouts $S/readouts/s3d-rl600/sibling_eval.jsonl --layers 28,36,44,52
  uv run python scripts/sibling_probe/write_plans.py check-cells --meta-dir $S/render_eval/meta \
    --readouts $S/readouts/jlens/sibling_*.jsonl --layers 28,36,44,52
  uv run python scripts/sibling_probe/write_plans.py check-cells --meta-dir $S/render_eval/meta \
    --readouts $S/readouts/nla-rl-L42/sibling_*.jsonl --layers 42
  ;;
probes)
  uv run hbench sibling probes --prepare
  $MODAL run scripts/sibling_probe/probes_modal.py --inputs $S/probes/inputs.json \
    --eval-acts acts-sibling-eval --probe-acts acts-sibling-probe --run ${PROBES_RUN:-sibling-v3} \
    --out-dir $S/probes
  uv run hbench sibling probes --merge
  ;;
probes-smoke)
  uv run hbench sibling probes --prepare --out-dir $S/probes_smoke
  $MODAL run scripts/sibling_probe/probes_modal.py --inputs $S/probes_smoke/inputs.json \
    --eval-acts acts-sibling-smoke --run sibling-smoke --out-dir $S/probes_smoke \
    --synthetic-probes --n-boot 100
  uv run hbench sibling probes --merge --out-dir $S/probes_smoke
  ;;
*)
  sed -n '2,23p' "$0"
  exit 1
  ;;
esac
