#!/usr/bin/env bash
# readouts_rest.sh <ts>   (plan 0016 rev 5): capture the positions some arm still lacks
# (outputs/strongreject/capture_rows_rest.json, from rest_rows.py) -> three reads in parallel
# (wide fan-out) -> fetch + convert into $HBENCH_SR_READOUTS_REST -> merge_readouts.py.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
TS=${1:-$(date +%Y%m%d_%H%M%S)}
L=$HERE/logs
GW=${HBENCH_GW_CAPTURE:-/workspace/global-workspace}
GWM=${HBENCH_GW_MLNLA:-/workspace/gw-mlnla}
ACTS=hb-strongreject-acts-rest
REST=${HBENCH_SR_READOUTS_REST:-/workspace/hb-readouts-strongreject-rest}
export PYTHONPATH=src PYTHONUNBUFFERED=1 WSBENCH_GEN_VOLUME=wsbench-gen-3 WSB_GEN_VOLUME=wsbench-gen-3
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2} MODAL_RESOURCE_ENV=${MODAL_RESOURCE_ENV:-camila-blank}
MB=${MODAL_BIN:-uvx --python 3.12 modal@latest}
M="$MB run"; W=scripts/olens_suite/workspace_bench
S3D_SHARDS=${S3D_SHARDS:-32}; NLA_SHARDS=${NLA_SHARDS:-64}

if [ "${SKIP_CAPTURE:-0}" != 1 ]; then
  echo "=== $(date -Is) capture"
  (cd "$GW" && $M $W/wsbench_capture_modal.py --prompts-json "$HERE/outputs/strongreject/capture_rows_rest.json" \
     --acts-subdir $ACTS --layers 28:53:8 --max-positions 1024) > "$L/sr_capture_rest_$TS.log" 2>&1 \
     || { echo "CAPTURE FAILED"; exit 1; }
fi
if [ "${SKIP_READS:-0}" != 1 ]; then
  echo "=== $(date -Is) reads"
  echo "[{\"acts\": \"$ACTS\", \"out\": \"gen-hb-strongreject-jlens-rest\", \"layers\": [28, 36, 44, 52]}]" > "$L/hb-strongreject-jlens-rest-jobs.json"
  ( cd "$GW" && $M $W/wsbench_read_modal.py::main --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 \
      --merged-name s3d-rl600 --run hb-strongreject-s3d-rl600-rest --acts-subdir $ACTS \
      --layers 28:53:8 --k 3 --shards $S3D_SHARDS > "$L/sr_read_rest_s3d_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/sr_read_rest_s3d_$TS.log" ) &
  ( cd "$GW" && $M $W/wsbench_tokenlens_modal.py --lens jlens --jobs-json "$L/hb-strongreject-jlens-rest-jobs.json" \
      > "$L/sr_read_rest_jlens_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/sr_read_rest_jlens_$TS.log" ) &
  ( cd "$GWM" && $M $W/wsbench_nla_modal.py --lens mlnla-rl600 --run hb-strongreject-mlnla-rl600-rest \
      --acts-subdirs $ACTS --layers 28,36,44,52 --n-samples 3 --shards $NLA_SHARDS \
      > "$L/sr_read_rest_mlnla_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/sr_read_rest_mlnla_$TS.log" ) &
  wait
  grep -H "EXIT=" "$L"/sr_read_rest_*_"$TS".log
fi
echo "=== $(date -Is) fetch + convert + merge"
rc=0
for spec in "s3d-rl600 prose" "jlens tokens" "mlnla-rl600 prose"; do
  set -- $spec
  GRID_CHECK=0 HBENCH_SR_READOUTS=$REST "$HERE/scripts/strongreject/fetch_readouts.sh" "$1" "gen-hb-strongreject-$1-rest" "$2" || rc=1
  "$HERE/.venv/bin/python" "$HERE/scripts/strongreject/merge_readouts.py" "$1" --rest-root "$REST" || rc=1
done
echo "READOUTS REST DONE rc=$rc"
