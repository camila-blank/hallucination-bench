#!/usr/bin/env bash
# Plan 0007 stage 2 for ONE domain (after its judge): concept set -> probe texts -> capture ->
# probe fit -> distractors -> score -> label -> simple plot.   usage: um_run_domain_stage2.sh <domain>
set -uo pipefail
D=$1
HB=$(cd "$(dirname "$0")/../.." && pwd); cd "$HB"
GW=${GW:-/workspace/gw-wrong-bridge}; WB=$GW/scripts/olens_suite/workspace_bench
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
MODAL="uvx --python 3.12 modal@latest"
DS=${D//_/-}
O=$HB/outputs/sibling/um/$D
PY="env -u ANTHROPIC_API_KEY HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv/bin/python"
S="$PY scripts/sibling_probe/um_ladder.py --category $D"
L=20,24,28,32,36,40,42,44,48,52,56,60
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "[retry $i] $*"; sleep 30; done; return 1; }
set -e
echo "=== $D concepts $(date)"; $S concepts
echo "=== $D probe texts $(date)"; $S probe-texts
$S capture-rows
$PY $WB/wsbench_public_plan.py prompts --plan-dir $O/plan_probe --out-dir $O/render_probe \
  --wsbench-src /workspace/root/wsbench-pub/src
PJ=$(ls $O/render_probe/prompts/*.json | head -1)
set +e
echo "=== $D capture $(date)"
(cd "$GW" && retry env PYTHONPATH=src $MODAL run $WB/wsbench_capture_modal.py --prompts-json $PJ \
  --acts-subdir acts-um-$DS-probe --layers $L --max-positions 128 --strict ${CAPTURE_FLAGS:-}) || exit 1
retry $MODAL volume get --force wsbench-acts /acts-um-$DS-probe/manifest.json $O/probe_manifest.json || exit 1
set -e
$S probe-inputs
export RUN=um-$DS-v1 PDIR=outputs/sibling/um/$D/probes INP=outputs/sibling/um/$D/probe_inputs.json \
  DIST=data/sibling_probe/um_${D}_distractors.json NL=12
scripts/sibling_probe/um_run_probes.sh pack
$S distractors
scripts/sibling_probe/um_run_probes.sh score
$S label
$PY scripts/sibling_probe/um_plot_simple.py --category $D
echo "STAGE2DONE $D $(date)"
