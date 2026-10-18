#!/usr/bin/env bash
# Plan 0006 GPU/CPU chain: probe-text capture (1 GPU, from the GW worktree, never edited) ->
# manifest check -> probe inputs -> Phase A probe fit (CPU, one container per layer).
set -euo pipefail
HB=$(cd "$(dirname "$0")/../.." && pwd)
GW=${GW:-/workspace/gw-wrong-bridge}
WB=$GW/scripts/olens_suite/workspace_bench
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
MODAL="uvx --python 3.12 modal@latest"
S=$HB/outputs/sibling/um
L=20,24,28,32,36,40,42,44,48,52,56,60
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "[retry $i] $*"; sleep 30; done; return 1; }
echo "=== capture $(date)"
(cd "$GW" && retry env PYTHONPATH=src $MODAL run $WB/wsbench_capture_modal.py \
  --prompts-json $S/render_probe/prompts/L20-24-28-32-36-40-42-44-48-52-56-60.json \
  --acts-subdir acts-um-country-probe --layers $L --max-positions 128 --strict ${CAPTURE_FLAGS:-})
retry $MODAL volume get --force wsbench-acts /acts-um-country-probe/manifest.json $S/probe_manifest.json
python3 - "$S/probe_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
n = len(m["prompts"]); bad = [p["label"] for p in m["prompts"] if len(p["eval_positions"]) != 21]
print("manifest layers", m["layers"], "prompts", n, "bad windows", len(bad))
assert n == 3864 and not bad
PY
cd "$HB"
.venv/bin/python scripts/sibling_probe/um_ladder.py probe-inputs
echo "=== fit $(date)"
retry $MODAL run scripts/sibling_probe/um_probes_modal.py --phase fit
echo "ALLDONE $(date)"
