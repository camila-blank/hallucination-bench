#!/usr/bin/env bash
# Plan 0006 probe job, detached (the box's network drops; outputs land on the sibling-probes
# volume and are fetched afterwards): pack (16 shards) -> fit (12 layers) -> fetch.
set -uo pipefail
HB=$(cd "$(dirname "$0")/../.." && pwd); cd "$HB"
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
MODAL="uvx --python 3.12 modal@latest"
RUN=${RUN:-um-country-v2}
P=${PDIR:-outputs/sibling/um/probes}
INP=${INP:-outputs/sibling/um/probe_inputs.json}
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "[retry $i] $*"; sleep 30; done; return 1; }
wait_files() {  # wait_files <remote dir> <n>
  for _ in $(seq 1 720); do
    n=$($MODAL volume ls sibling-probes /$RUN/$1 2>/dev/null | grep -c "${3:-.}" || true)
    [ "$n" -ge "$2" ] && return 0; sleep 30; done; return 1; }
case "${1:-all}" in
pack|all)
  echo "=== pack $(date)"
  $MODAL run --detach scripts/sibling_probe/um_probes_modal.py --phase pack --inputs $INP --out-dir $P || echo "[client exited; waiting for volume outputs]"
  wait_files pack 16 'json' || { echo "pack timeout"; exit 1; }
  ;&
fit)
  echo "=== fit $(date)"
  $MODAL run --detach scripts/sibling_probe/um_probes_modal.py --phase fit --inputs $INP --out-dir $P || echo "[client exited; waiting for volume outputs]"
  wait_files fit ${NL:-12} 'fit_L' || { echo "fit timeout"; exit 1; }
  mkdir -p $P/fit
  for f in $($MODAL volume ls sibling-probes /$RUN/fit | grep -E "fit_L|centroids_L"); do
    retry $MODAL volume get --force sibling-probes /$RUN/fit/$(basename $f) $P/fit/
  done
  echo "FITDONE $(date)"
  ;;
score)
  echo "=== score $(date)"
  $MODAL run --detach scripts/sibling_probe/um_probes_modal.py --phase score --inputs $INP --out-dir $P || echo "[client exited; waiting for volume outputs]"
  wait_files score ${NL:-12} 'npz' || { echo "score timeout"; exit 1; }
  mkdir -p $P/score
  for f in $($MODAL volume ls sibling-probes /$RUN/score | grep npz); do
    retry $MODAL volume get --force sibling-probes /$RUN/score/$(basename $f) $P/score/
  done
  echo "SCOREDONE $(date)"
  ;;
esac
