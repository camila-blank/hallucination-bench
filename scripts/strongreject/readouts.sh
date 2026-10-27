#!/usr/bin/env bash
# readouts.sh <ts>   (plan 0016 §Readouts): capture -> three reads in parallel -> fetch + convert
# + grid check. Run in tmux; logs under logs/. Needs a global-workspace checkout
# ($HBENCH_GW_CAPTURE, default /workspace/global-workspace) and the multilayer-NLA checkout
# ($HBENCH_GW_MLNLA, default /workspace/gw-mlnla; its --lens registry is not on main).
#   SKIP_CAPTURE=1 / SKIP_READS=1 resume later steps.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
TS=${1:-$(date +%Y%m%d_%H%M%S)}
L=$HERE/logs
GW=${HBENCH_GW_CAPTURE:-/workspace/global-workspace}
GWM=${HBENCH_GW_MLNLA:-/workspace/gw-mlnla}
ACTS=hb-strongreject-acts
export PYTHONPATH=src PYTHONUNBUFFERED=1 WSBENCH_GEN_VOLUME=wsbench-gen-3 WSB_GEN_VOLUME=wsbench-gen-3
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
M="uvx --python 3.12 modal@latest run"; W=scripts/olens_suite/workspace_bench
retry() { local t; for t in 1 2 3; do "$@" && return 0; echo "retry $t: $*"; sleep 30; done; return 1; }

if [ "${SKIP_CAPTURE:-0}" != 1 ]; then
  echo "=== $(date -Is) capture"
  (cd "$GW" && retry $M $W/wsbench_capture_modal.py --prompts-json "$HERE/data/strongreject/capture_rows.json" \
     --acts-subdir $ACTS --layers 28:53:8 --max-positions 1024 ${CAPTURE_EXTRA:-}) > "$L/sr_capture_$TS.log" 2>&1 \
     || { echo "CAPTURE FAILED"; exit 1; }
fi

if [ "${SKIP_READS:-0}" != 1 ]; then
  echo "=== $(date -Is) reads"
  echo "[{\"acts\": \"$ACTS\", \"out\": \"gen-hb-strongreject-jlens\", \"layers\": [28, 36, 44, 52]}]" > "$L/hb-strongreject-jlens-jobs.json"
  ( cd "$GW" && $M $W/wsbench_read_modal.py --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 \
      --merged-name s3d-rl600 --run hb-strongreject-s3d-rl600 --acts-subdir $ACTS \
      --layers 28:53:8 --k 3 --shards 16 > "$L/sr_read_s3d_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/sr_read_s3d_$TS.log" ) &
  ( cd "$GW" && $M $W/wsbench_tokenlens_modal.py --lens jlens --jobs-json "$L/hb-strongreject-jlens-jobs.json" \
      > "$L/sr_read_jlens_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/sr_read_jlens_$TS.log" ) &
  ( cd "$GWM" && $M $W/wsbench_nla_modal.py --lens mlnla-rl600 --run hb-strongreject-mlnla-rl600 \
      --acts-subdirs $ACTS --layers 28,36,44,52 --n-samples 3 --shards 16 \
      > "$L/sr_read_mlnla_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/sr_read_mlnla_$TS.log" ) &
  wait
  grep -H "EXIT=" "$L"/sr_read_*_"$TS".log
fi

echo "=== $(date -Is) fetch + convert"
rc=0
"$HERE/scripts/strongreject/fetch_readouts.sh" s3d-rl600 gen-hb-strongreject-s3d-rl600 prose || rc=1
"$HERE/scripts/strongreject/fetch_readouts.sh" jlens gen-hb-strongreject-jlens tokens || rc=1
"$HERE/scripts/strongreject/fetch_readouts.sh" mlnla-rl600 gen-hb-strongreject-mlnla-rl600 prose || rc=1
echo "READOUTS DONE rc=$rc"
