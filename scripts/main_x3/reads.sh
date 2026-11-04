#!/usr/bin/env bash
# Plan 0028: capture -> reads (s3d / mlnla / jlens, all 11 layers) -> fetch + convert + grid check.
#
#   TAG=a scripts/main_x3/reads.sh [capture|read-s3d|read-mlnla|read-jlens|fetch]
#
# TAG names one batch of new cells (acts-hb28-main-x3-$TAG, gen-hb28-main-x3-$TAG-<arm>), so the
# lmsys half can be captured and read later without touching a done batch. Rows to capture:
# $ROWS (default <bank>/capture_new_$TAG.json: build_bank.py's capture_new.json, possibly
# filtered to the batch's items). Runs the global-workspace launchers from $HBENCH_GW_CAPTURE
# (a LOCAL clone of GW main: /workspace imports hang). Merge: scripts/main_x3/merge_readouts.py.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
STEP=${1:?step}
TAG=${TAG:?set TAG}
GW=${HBENCH_GW_CAPTURE:-/root/tmp/hb28-gw}
BANK=${HBENCH_X3_BANK:-$HERE/data/wsbench_chat_x3}
ROWS=${ROWS:-$BANK/capture_new_$TAG.json}
GEN_ROOT=${HBENCH_X3_GEN:-/root/tmp/hb28-gen}
NEW=${HBENCH_X3_NEW_READOUTS:-/root/tmp/hb28-readouts-new}/$TAG
WSPUB=${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
M=${MODAL_BIN:-/root/.local/bin/modal}
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
export MODAL_RESOURCE_ENV=${MODAL_RESOURCE_ENV:-camila-blank}  # the launchers' volumes default to env main
export GEN_VOLUME=${GEN_VOLUME:-wsbench-gen-3}
export WSBENCH_GEN_VOLUME=$GEN_VOLUME WSB_GEN_VOLUME=$GEN_VOLUME PYTHONPATH=src PYTHONUNBUFFERED=1
W=scripts/olens_suite/workspace_bench
ACTS=acts-hb28-main-x3-$TAG
LAYERS=20:61:4
LAYER_LIST=20,24,28,32,36,40,44,48,52,56,60
G=gen-hb28-main-x3-$TAG
# reader  gen-dir  kind   (s3d feeds both s3d arms: merge_readouts.py)
ARMS=("s3d $G-s3d-rl600 prose" "mlnla-rl600-all11 $G-mlnla-rl600 prose" "jlens $G-jlens tokens")

retry3() { for t in 1 2 3; do "$@" && return 0; echo "retry $t: $*" >&2; sleep 60; done; return 1; }

case "$STEP" in
  capture)
    cd "$GW" && retry3 $M run $W/wsbench_capture_modal.py --prompts-json "$ROWS" \
      --acts-subdir $ACTS --layers $LAYERS --max-positions 2048 --strict ;;
  read-s3d)
    cd "$GW" && retry3 $M run $W/wsbench_read_modal.py::main \
      --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 --merged-name s3d-rl600 \
      --run hb28-main-x3-$TAG-s3d-rl600 --acts-subdir $ACTS --layers $LAYERS --k 3 \
      --shards ${S3D_SHARDS:-16} --prompt-kind concepts_raw --transform unit ;;
  read-mlnla)
    cd "$GW" && retry3 $M run $W/wsbench_nla_modal.py --lens mlnla-rl600 \
      --run hb28-main-x3-$TAG-mlnla-rl600 --acts-subdirs $ACTS --layers $LAYER_LIST \
      --n-samples 3 --shards ${MLNLA_SHARDS:-32} ;;
  read-jlens)
    jobs=$(mktemp --suffix=.json)
    echo "[{\"acts\": \"$ACTS\", \"out\": \"$G-jlens\", \"layers\": [$LAYER_LIST]}]" > "$jobs"
    cd "$GW" && retry3 $M run $W/wsbench_tokenlens_modal.py --lens jlens --jobs-json "$jobs" ;;
  fetch)
    for spec in "${ARMS[@]}"; do
      read -r arm gen kind <<< "$spec"
      [[ -z ${ONLY_ARMS:-} || ,$ONLY_ARMS, == *,$arm,* ]] || continue
      echo "=== $(date -Is) fetch + convert $arm ($gen)"
      mkdir -p "$GEN_ROOT" "$NEW/$arm"
      rm -rf "${GEN_ROOT:?}/${gen:?}"
      vol=""
      for v in "$GEN_VOLUME" wsbench-gen-2 wsbench-gen-4; do
        if $M volume ls "$v" "/$gen" >/dev/null 2>&1; then vol=$v; break; fi
      done
      [[ -n $vol ]] || { echo "no volume holds /$gen" >&2; exit 1; }
      (cd "$GEN_ROOT" && retry3 $M volume get "$vol" "/$gen" . --force >/dev/null) || exit 1
      (cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$GEN_ROOT/$gen" \
          out="$NEW/$arm/hallucination.jsonl" kind="$kind" | tail -1) || exit 1
      "$HERE/.venv/bin/python" scripts/main_x3/merge_readouts.py check-new \
          --rows "$ROWS" --readouts "$NEW/$arm/hallucination.jsonl" --kind "$kind" || exit 1
    done ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
