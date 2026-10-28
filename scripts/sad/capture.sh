#!/usr/bin/env bash
# plan 0018 §Readouts step 1: capture the selected SAD plain prompts (layers 28,36,44,52) onto
# wsbench-acts:/hb-sad-acts with the global-workspace capture app (1 H200), then fetch the manifest
# and check its token grid against the local tokenisation. Run in tmux; log to logs/.
set -euo pipefail
export UV_CACHE_DIR=${HBENCH_SAD_UV_CACHE:-/root/.cache/uv-hbsad}  # never the shared cache
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
GW=${HBENCH_GLOBAL_WORKSPACE:-/workspace/global-workspace}
ROWS=${SAD_CAPTURE_ROWS:-$HERE/outputs/sad/capture_rows.json}
SUB=${SAD_ACTS_SUBDIR:-hb-sad-acts}
M="uvx --python 3.12 modal@latest"
export PYTHONPATH=src PYTHONUNBUFFERED=1 MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
cd "$GW"
$M run --env camila-blank scripts/olens_suite/workspace_bench/wsbench_capture_modal.py \
  --prompts-json "$ROWS" --acts-subdir "$SUB" --layers 28,36,44,52 --max-positions 2304 --strict ${EXTRA:-}
mkdir -p "$HERE/outputs/sad"
rm -f "$HERE/outputs/sad/acts_manifest.json"
MODAL_RESOURCE_ENV=camila-blank $M volume get wsbench-acts "/$SUB/manifest.json" "$HERE/outputs/sad/acts_manifest.json"
cd "$HERE" && HF_HUB_OFFLINE=1 .venv/bin/hbench sad check-manifest outputs/sad/acts_manifest.json
echo CAPTURE_OK
