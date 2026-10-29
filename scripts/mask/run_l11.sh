#!/usr/bin/env bash
# run_l11.sh <arm> <concurrency>   (plan 0019, mask)
# Detect + verify one arm at the 7 new layers, one layer at a time, on the L11 read grid
# (honest_resisted prompt sites): readouts $HBENCH_MASK_READOUTS/<arm>/mask_l11.jsonl, sites
# data/mask/l11/sites_l11.json.gz; writes cells_L0NN.jsonl beside the merged layers' files.
# Same prompts / models / Opus -> Sonnet fallback as the merged run; exit 4 retried 3x.
set -uo pipefail
arm=$1; conc=${2:-16}
cd "$(dirname "$0")/../.."
for L in 20 24 32 40 48 56 60; do
  for try in 1 2 3 4; do
    env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 .venv/bin/hbench mask run --arms "$arm" \
      --layers "$L" --concurrency "$conc" --sites-file data/mask/l11/sites_l11.json.gz \
      --readouts-file mask_l11.jsonl
    rc=$?
    [[ $rc -eq 4 ]] || break
    echo "[run_l11] $arm L$L: failed calls remain (try $try), retrying"
  done
  [[ $rc -eq 0 || $rc -eq 4 ]] || { echo "[run_l11] $arm L$L exit $rc" >&2; exit $rc; }
done
echo "[run_l11] $arm done"
