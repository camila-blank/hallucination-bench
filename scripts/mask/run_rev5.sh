#!/usr/bin/env bash
# run_rev5.sh <arm> <concurrency>   (plan 0017 rev 5)
# Detect + verify one arm on every read site, one layer at a time (bounds memory: a full
# 586k-cell prose arm in one process was OOM-killed under the shared 32 GB cap). Each layer run
# writes outputs/mask/<model>/<arm>/cells_L0NN.jsonl; exit 4 (cached failures) is retried 3x.
set -uo pipefail
arm=$1; conc=${2:-16}
cd "$(dirname "$0")/../.."
for L in 28 36 44 52; do
  for try in 1 2 3 4; do
    env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 .venv/bin/hbench mask run --arms "$arm" \
      --layers "$L" --concurrency "$conc"
    rc=$?
    [[ $rc -eq 4 ]] || break
    echo "[run_rev5] $arm L$L: failed calls remain (try $try), retrying"
  done
  [[ $rc -eq 0 || $rc -eq 4 ]] || { echo "[run_rev5] $arm L$L exit $rc" >&2; exit $rc; }
done
echo "[run_rev5] $arm done"
