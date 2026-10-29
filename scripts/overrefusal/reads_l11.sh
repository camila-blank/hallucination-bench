#!/usr/bin/env bash
# Plan 0019 step 2: reads of the overrefusal bank at the 7 NEW layers (20/24/32/40/48/56/60),
# all three readers, from the capture wsbench-acts:/hb-overrefusal-L11-read (427 items, every
# prompt position); then fetch + convert + grid check, and merge with the 4-layer readouts into
# $HBENCH_OR_READOUTS_V3/<arm>/overrefusal.jsonl (all 11 layers).
#
#   scripts/overrefusal/reads_l11.sh [read-s3d|read-mlnla|read-jlens|fetch|merge]
# Same reader settings / artifacts / convert path as scripts/overrefusal/reads.sh (4-layer run).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
STEP=${1:?step}
GW=${HBENCH_GW_CAPTURE:-/workspace/global-workspace}
V2=${HBENCH_OR_READOUTS_V2:-/workspace/hb-readouts-overrefusal-v2}
NEW=${HBENCH_OR_READOUTS_L11:-/workspace/hb-readouts-overrefusal-L11}
V3=${HBENCH_OR_READOUTS_V3:-/workspace/hb-readouts-overrefusal-v3}
GEN_ROOT=${HBENCH_OR_GEN:-/workspace/hb-overrefusal-gen}
WSPUB=${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
export WSBENCH_GEN_VOLUME=${GEN_VOLUME:-wsbench-gen-3} PYTHONPATH=src PYTHONUNBUFFERED=1
export UV_CACHE_DIR=${HBENCH_OR_UV_CACHE:-$HOME/.cache/uv-hb-overrefusal} UV_LOCK_TIMEOUT=1800
export MODAL_RESOURCE_ENV=${MODAL_RESOURCE_ENV:-camila-blank}
M="uvx --python 3.12 modal@latest"
W=scripts/olens_suite/workspace_bench
ACTS=hb-overrefusal-L11-read
LAYERS=20,24,32,40,48,56,60
ROWS=$HERE/data/overrefusal/l11/capture_rows_all.json
ARMS=("s3d-rl600 gen-hb-overrefusal-L11-s3d-rl600 prose"
      "mlnla-rl600-all11 gen-hb-overrefusal-L11-mlnla-rl600 prose"
      "jlens gen-hb-overrefusal-L11-jlens tokens")

retry3() { for t in 1 2 3; do "$@" && return 0; echo "retry $t: $*" >&2; sleep 60; done; return 1; }

case "$STEP" in
  read-s3d)
    cd "$GW" && retry3 $M run $W/wsbench_read_modal.py::main \
      --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 --merged-name s3d-rl600 \
      --run hb-overrefusal-L11-s3d-rl600 --acts-subdir $ACTS --layers $LAYERS --k 3 \
      --shards ${S3D_SHARDS:-150} --prompt-kind concepts_raw --transform unit ;;
  read-mlnla)
    cd "$GW" && retry3 $M run $W/wsbench_nla_modal.py --lens mlnla-rl600 \
      --run hb-overrefusal-L11-mlnla-rl600 --acts-subdirs $ACTS --layers $LAYERS --n-samples 3 \
      --shards ${MLNLA_SHARDS:-300} ;;
  read-jlens)
    jobs=$(mktemp --suffix=.json)
    echo "[{\"acts\": \"$ACTS\", \"out\": \"gen-hb-overrefusal-L11-jlens\", \"layers\": [20, 24, 32, 40, 48, 56, 60]}]" > "$jobs"
    cd "$GW" && retry3 $M run $W/wsbench_tokenlens_modal.py --lens jlens --jobs-json "$jobs" ;;
  fetch)
    for spec in "${ARMS[@]}"; do
      read -r arm gen kind <<< "$spec"
      [[ -z ${ONLY_ARMS:-} || ,$ONLY_ARMS, == *,$arm,* ]] || continue
      echo "=== $(date -Is) fetch + convert $arm ($gen)"
      mkdir -p "$GEN_ROOT" "$NEW/$arm"
      rm -rf "${GEN_ROOT:?}/${gen:?}"
      (cd "$GEN_ROOT" && retry3 $M volume get "$WSBENCH_GEN_VOLUME" "/$gen" . --force >/dev/null) || exit 1
      (cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$GEN_ROOT/$gen" \
          out="$NEW/$arm/overrefusal.jsonl" kind="$kind" | tail -1) || exit 1
      python3 - "$NEW/$arm/overrefusal.jsonl" "$HERE/data/overrefusal/items.json" "$LAYERS" "$kind" <<'PY' || exit 1
import collections, json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
items = json.load(open(sys.argv[2]))["items"]
layers = [int(x) for x in sys.argv[3].split(",")]
want = {(it["id"], la, s["pos"]) for it in items for la in layers for s in it["sites"]}
keys = [(r["id"], r["layer"], r["pos"]) for r in rows]
have = set(keys)
miss, off, dup = want - have, have - want, len(keys) - len(have)
ks = collections.Counter(len(r.get("samples", [])) for r in rows if sys.argv[4] == "prose")
print(f"{sys.argv[1]}: {len(rows)} rows, on-grid {len(have & want)}/{len(want)}, missing {len(miss)}, "
      f"off-grid {len(off)}, dup {dup}, samples per prose row {dict(ks)}")
sys.exit(1 if (miss or off or dup or not rows) else 0)
PY
    done ;;
  merge)
    for spec in "${ARMS[@]}"; do
      read -r arm _ _ <<< "$spec"
      mkdir -p "$V3/$arm"
      cat "$V2/$arm/overrefusal.jsonl" "$NEW/$arm/overrefusal.jsonl" > "$V3/$arm/overrefusal.jsonl"
      echo "merged $arm: $(wc -l < "$V3/$arm/overrefusal.jsonl") rows"
    done ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
