#!/usr/bin/env bash
# plan 0018 §Readouts step 2: the three arms' reads of wsbench-acts:/hb-sad-acts, in parallel,
# every gen dir on ONE volume (wsbench-gen-3; the readers' own defaults differ). Arg: timestamp.
set -uo pipefail
export PYTHONPATH=src PYTHONUNBUFFERED=1 WSBENCH_GEN_VOLUME=wsbench-gen-3 WSB_GEN_VOLUME=wsbench-gen-3
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2} MODAL_RESOURCE_ENV=${MODAL_RESOURCE_ENV:-camila-blank}
# a private uv cache dir: the shared cache's lockfile wait times out (300 s) under sibling agents
export UV_CACHE_DIR=${HBENCH_SAD_UV_CACHE:-/root/.cache/uv-hbsad}  # never the shared cache
TS=${1:?timestamp}
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
GW=${HBENCH_GLOBAL_WORKSPACE:-/workspace/global-workspace}
L=$HERE/logs; M="uvx --python 3.12 modal@latest run --env camila-blank"; W=scripts/olens_suite/workspace_bench
SUB=${SAD_ACTS_SUBDIR:-hb-sad-acts}; LAYERS=28,36,44,52
echo "[{\"acts\": \"$SUB\", \"out\": \"gen-hb-sad-jlens\", \"layers\": [28, 36, 44, 52]}]" > "$L/sad_jlens_jobs.json"
cd "$GW"
( $M $W/wsbench_read_modal.py::main --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 \
    --merged-name s3d-rl600 --run hb-sad-s3d-rl600 --acts-subdir "$SUB" --layers $LAYERS --k 3 \
    --shards 16 > "$L/read_s3d_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/read_s3d_$TS.log" ) &
( $M $W/wsbench_nla_modal.py --lens mlnla-rl600 --run hb-sad-mlnla-rl600 --acts-subdirs "$SUB" \
    --layers $LAYERS --n-samples 3 --shards 16 > "$L/read_mlnla_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/read_mlnla_$TS.log" ) &
( $M $W/wsbench_tokenlens_modal.py --lens jlens --jobs-json "$L/sad_jlens_jobs.json" \
    > "$L/read_jlens_$TS.log" 2>&1; echo "EXIT=$?" >> "$L/read_jlens_$TS.log" ) &
wait; echo ALL_READS_DONE
