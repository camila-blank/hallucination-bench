#!/usr/bin/env bash
# Plan 0006 rev 9: the adjacent-layer (L +- 1) captures + probes. Probe texts and the 32 country
# eval items at the 22 layers next to the lens layers; then pack / fit / score into probes_adj/.
set -uo pipefail
HB=$(cd "$(dirname "$0")/../.." && pwd); cd "$HB"
GW=${GW:-${HBENCH_GW_CAPTURE:-/workspace/gw-wrong-bridge}}; WB=$GW/scripts/olens_suite/workspace_bench
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
MODAL="uvx --python 3.12 modal@latest"
S=$HB/outputs/sibling/um
L=19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "[retry $i] $*"; sleep 30; done; return 1; }
echo "=== capture eval adj $(date)"
(cd "$GW" && retry env PYTHONPATH=src $MODAL run $WB/wsbench_capture_modal.py \
  --prompts-json $S/render_eval/country_eval.json --acts-subdir acts-um-country-eval-adj \
  --layers $L --max-positions 128 --strict ${CAPTURE_FLAGS:-})
echo "=== capture probe adj $(date)"
(cd "$GW" && retry env PYTHONPATH=src $MODAL run $WB/wsbench_capture_modal.py \
  --prompts-json $S/render_probe/prompts/L20-24-28-32-36-40-42-44-48-52-56-60.json \
  --acts-subdir acts-um-country-probe-adj --layers $L --max-positions 128 --strict ${CAPTURE_FLAGS:-})
for sub in eval probe; do
  retry $MODAL volume get --force wsbench-acts /acts-um-country-$sub-adj/manifest.json $S/${sub}_adj_manifest.json
done
python3 - "$S" <<'PY'
import json, sys
for sub, n in (("eval", 32), ("probe", 3864)):
    m = json.load(open(f"{sys.argv[1]}/{sub}_adj_manifest.json"))
    bad = [p["label"] for p in m["prompts"] if len(p["eval_positions"]) != 21]
    print(sub, "layers", m["layers"], "prompts", len(m["prompts"]), "bad", len(bad))
    assert len(m["prompts"]) == n and not bad and len(m["layers"]) == 22
PY
export RUN=um-country-adj1 PDIR=outputs/sibling/um/probes_adj INP=outputs/sibling/um/probe_inputs_adj.json NL=22
scripts/sibling_probe/um_run_probes.sh pack
scripts/sibling_probe/um_run_probes.sh score
echo "ADJDONE $(date)"
