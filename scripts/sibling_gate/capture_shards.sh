# usage: capture.sh N  (after capture-manifest wrote outputs/sibling_gate/probes/capture_prompts.json)
set -u
N=${1:-32}
REPO=/workspace/hb-sibling-resample
SNAP_WB=/workspace/sibgate-snap/gw-wrong-bridge
export MODAL_RESOURCE_ENV=camila-blank
cd $REPO
TS=$(date +%Y%m%d_%H%M)
mkdir -p outputs/sibling_gate/probes/shards
python3 - <<PY
import json
rows=json.load(open("outputs/sibling_gate/probes/capture_prompts.json"))
n=$N
for s in range(n):
    json.dump(rows[s::n], open(f"outputs/sibling_gate/probes/shards/cp_{s:02d}.json","w"), ensure_ascii=False)
print(len(rows), "prompts ->", n, "shards")
PY
cd $SNAP_WB
for s in $(seq -f "%02g" 0 $((N-1))); do
  ( PYTHONPATH=src uvx --python 3.12 modal@latest run scripts/olens_suite/workspace_bench/wsbench_capture_modal.py \
      --prompts-json $REPO/outputs/sibling_gate/probes/shards/cp_$s.json --acts-subdir acts-sibgate-probe-s$s \
      --layers 20,24,28,32,36,40,42,44,48,52,56,60 > $REPO/logs/sibgate_capture_s${s}_$TS.log 2>&1
    echo "shard $s rc=$?" >> $REPO/logs/sibgate_capture_status_$TS.log ) &
done
wait
echo "[capture done $(date)]" >> $REPO/logs/sibgate_capture_status_$TS.log
