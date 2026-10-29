#!/usr/bin/env bash
# readouts_l11.sh <ts>   (plan 0019): read the 7 new layers (20/24/32/40/48/56/60) of every
# prompt position of the 173 read-set items (wsbench-acts:/hb-strongreject-L11-acts, captured
# from data/strongreject/capture_rows.json) for the 3 readers, wide fan-out, then fetch + convert
# into $HBENCH_SR_READOUTS_L11 and merge into the readouts files (old rows kept byte for byte;
# the pre-merge file is kept as strongreject.rev5.jsonl), checking the full 11-layer grid.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
TS=${1:-$(date +%Y%m%d_%H%M%S)}
L=$HERE/logs
GW=${HBENCH_GW_CAPTURE:-/workspace/global-workspace}
GWM=${HBENCH_GW_MLNLA:-/workspace/gw-mlnla}
ACTS=hb-strongreject-L11-acts
NEW=${HBENCH_SR_READOUTS_L11:-/workspace/hb-readouts-strongreject-l11}
LAYERS=20,24,32,40,48,56,60
export PYTHONPATH=src PYTHONUNBUFFERED=1 WSBENCH_GEN_VOLUME=wsbench-gen-3 WSB_GEN_VOLUME=wsbench-gen-3
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2} MODAL_RESOURCE_ENV=${MODAL_RESOURCE_ENV:-camila-blank}
M="${MODAL_BIN:-uvx --python 3.12 modal@latest} run"; W=scripts/olens_suite/workspace_bench
S3D_SHARDS=${S3D_SHARDS:-128}; NLA_SHARDS=${NLA_SHARDS:-1211}  # 173 items x 7 layers = 1211 units
if [ "${SKIP_READS:-0}" != 1 ]; then
  echo "[{\"acts\": \"$ACTS\", \"out\": \"gen-hb-strongreject-L11-jlens\", \"layers\": [20, 24, 32, 40, 48, 56, 60]}]" > "$L/hb-strongreject-L11-jlens-jobs.json"
  ( cd "$GW" && $M $W/wsbench_read_modal.py::main --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 \
      --merged-name s3d-rl600 --run hb-strongreject-L11-s3d-rl600 --acts-subdir $ACTS \
      --layers $LAYERS --k 3 --shards $S3D_SHARDS > "$L/L11_read_s3d_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/L11_read_s3d_$TS.log" ) &
  ( cd "$GW" && $M $W/wsbench_tokenlens_modal.py --lens jlens --jobs-json "$L/hb-strongreject-L11-jlens-jobs.json" \
      > "$L/L11_read_jlens_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/L11_read_jlens_$TS.log" ) &
  ( cd "$GWM" && $M $W/wsbench_nla_modal.py --lens mlnla-rl600 --run hb-strongreject-L11-mlnla-rl600 \
      --acts-subdirs $ACTS --layers $LAYERS --n-samples 3 --shards $NLA_SHARDS \
      > "$L/L11_read_mlnla_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/L11_read_mlnla_$TS.log" ) &
  wait
fi
echo "=== $(date -Is) fetch + convert + merge"
rc=0
for spec in "s3d-rl600 prose" "jlens tokens" "mlnla-rl600 prose"; do
  set -- $spec
  GRID_CHECK=0 HBENCH_SR_READOUTS=$NEW "$HERE/scripts/strongreject/fetch_readouts.sh" "$1" "gen-hb-strongreject-L11-$1" "$2" || rc=1
  "$HERE/.venv/bin/python" "$HERE/scripts/strongreject/merge_readouts.py" "$1" --rest-root "$NEW" \
    --layers 20,24,28,32,36,40,44,48,52,56,60 --backup strongreject.rev5.jsonl || rc=1
done
echo "READOUTS L11 DONE rc=$rc"
