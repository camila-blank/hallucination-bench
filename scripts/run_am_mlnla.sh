#!/bin/bash
# Multilayer-NLA RL600 arm for the agentic_misalignment eval (results 0008 addendum).
# Waits for the Modal read (gw-mlnla wsbench_nla_modal.py --lens mlnla-rl600, gen dir
# wsbench-gen-6:/gen-am-mlnla-rl600-all11), fetches + converts it to the readout contract, checks
# the full 32-item x 11-layer grid, then runs Stage D / H / M in layer chunks (memory: the box is
# shared) and rebuilds the report. Each hbench run resumes from cache.
#
#   scripts/run_am_mlnla.sh <read-driver-log>     # monitor: tail -f logs/am_mlnla_<ts>.log
set -uo pipefail
cd /workspace/hb-am-halluc
DRV=${1:?read driver log}
SKIP_FETCH=${SKIP_FETCH:-0}  # 1: readouts already converted, start at Stage D
ARM=mlnla-rl600-all11
GEN=gen-am-mlnla-rl600-all11
VOL=wsbench-gen-6
GROOT=/workspace/hb-am-gen
RROOT=/workspace/hb-readouts-am
LAYERS=(20,24,28 32,36,40 44,48,52 56,60)

if [ "$SKIP_FETCH" != 1 ]; then
echo "=== $(date -Is) waiting for the read ($DRV)"
until grep -q "READ DONE\|READ FAILED" "$DRV" 2>/dev/null; do sleep 120; done
grep -q "READ DONE" "$DRV" || { echo "=== read failed"; exit 1; }

echo "=== $(date -Is) fetch (per item dir: a bulk volume get can miss dirs)"
mkdir -p "$GROOT/$GEN" "$RROOT/$ARM"
M="env MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest volume"
for d in $($M ls $VOL /$GEN); do
  name=$(basename "$d")
  rm -rf "${GROOT:?}/$GEN/${name:?}"
  for t in 1 2 3; do $M get $VOL "/$GEN/$name" "$GROOT/$GEN/" --force >/dev/null 2>&1 && break; done
done

echo "=== $(date -Is) convert"
(cd /workspace/root/wsbench-pub && /workspace/root/wsbench-venv/bin/python -m wsbench convert-gen-dir \
  gen_dir="$GROOT/$GEN" out="$RROOT/$ARM/agentic_misalignment.jsonl" kind=prose | tail -1)
python3 - "$RROOT/$ARM/agentic_misalignment.jsonl" data/agentic_misalignment/items.json <<'PY' || exit 1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
items = json.load(open(sys.argv[2]))["items"]
layers = list(range(20, 61, 4))
want = {(it["id"], la, s["pos"]) for it in items for la in layers for s in it["sites"]}
keys = [(r["id"], r["layer"], r["pos"]) for r in rows]
have = set(keys)
miss, off, dup = want - have, have - want, len(keys) - len(have)
print(f"{len(rows)} rows, on-grid {len(have & want)}/{len(want)}, missing {len(miss)}, "
      f"off-grid {len(off)}, dup {dup}")
sys.exit(1 if (miss or off or dup) else 0)
PY
fi

A="--eval agentic_misalignment --readouts-root $RROOT --out outputs --arms $ARM"
# exit 4 (cached failures): retried 3x then continue; exit 3 (auth: the `ant` OAuth token lapsed)
# waits 10 min and retries for up to 12 h, so a re-login (`ant auth login`) resumes the chain
hb() {
  local rc a=0
  while :; do
    for t in 1 2 3; do uv run --no-sync hbench run $A "$@"; rc=$?; [ $rc -ne 4 ] && break; done
    [ $rc -ne 3 ] && break
    a=$((a+1)); [ $a -gt 72 ] && break
    echo "=== $(date -Is) auth error (exit 3), retry $a/72 in 10 min; re-login with: ant auth login"
    sleep 600
  done
  echo "=== $(date -Is) RC $rc for $*"; [ $rc -eq 0 ] || [ $rc -eq 4 ] || exit $rc
}
for L in "${LAYERS[@]}"; do hb --stages detect --layers $L --rpm 1200 --concurrency 300; done
for L in "${LAYERS[@]}"; do hb --stages verify --layers $L --rpm 400 --concurrency 200; done
for L in "${LAYERS[@]}"; do hb --stages map --layers $L --rpm 1000 --concurrency 700; done
echo "=== $(date -Is) report"
uv run --no-sync hbench report --eval agentic_misalignment --out outputs
echo MLNLA_DONE
