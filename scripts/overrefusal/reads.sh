#!/usr/bin/env bash
# Plan 0015 Stage 4: capture -> three reads -> fetch + convert + grid check (Modal).
#
#   scripts/overrefusal/reads.sh [capture|reads|fetch|all]   (default all)
#   tmux new -s hb-overrefusal-reads "scripts/overrefusal/reads.sh all 2>&1 | tee logs/reads_<ts>.log"
#
# Runs the global-workspace Modal launchers from $HBENCH_GW_CAPTURE (a checkout on main; nothing
# there is modified). Every gen dir goes to $GEN_VOLUME (wsbench-gen-3: the launchers' own
# defaults differ) and every name is prefixed hb-overrefusal- so parallel evals never collide.
# Readouts land in $HBENCH_OR_READOUTS/<arm>/overrefusal.jsonl.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HERE"
STEP=${1:-all}
GW=${HBENCH_GW_CAPTURE:-/workspace/global-workspace}
DST=${HBENCH_OR_READOUTS:-/workspace/hb-readouts-overrefusal}
GEN_ROOT=${HBENCH_OR_GEN:-/workspace/hb-overrefusal-gen}
WSPUB=${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
export GEN_VOLUME=${GEN_VOLUME:-wsbench-gen-3}
export WSBENCH_GEN_VOLUME=$GEN_VOLUME WSB_GEN_VOLUME=$GEN_VOLUME PYTHONPATH=src PYTHONUNBUFFERED=1
# rev 7: TAG=v2add reads only the newly added items (acts / gen dirs / readouts suffixed)
TAG=${TAG:-}
ACTS=acts-hb-overrefusal${TAG:+-$TAG}
SHARDS=${SHARDS:-16}
LAYERS=28,36,44,52
LOGS=$HERE/logs
TS=$(date -u +%Y-%m-%d_%H%M)
M="uvx --python 3.12 modal@latest"
# a private uv cache: the shared one is lock-contended by parallel agents on this box
export UV_CACHE_DIR=${HBENCH_OR_UV_CACHE:-$HOME/.cache/uv-hb-overrefusal} UV_LOCK_TIMEOUT=1800
export MODAL_RESOURCE_ENV=${MODAL_RESOURCE_ENV:-camila-blank}  # wsbench_read_modal defaults to "main"
W=scripts/olens_suite/workspace_bench
ROWS=${ROWS:-$HERE/data/overrefusal/capture_rows.json}
G=gen-hb-overrefusal${TAG:+-$TAG}
DST=$DST${TAG:+-$TAG}
# arm  gen-dir  kind
ARMS=("s3d-rl600 $G-s3d-rl600 prose"
      "mlnla-rl600-all11 $G-mlnla-rl600 prose"
      "jlens $G-jlens tokens")

retry() { for t in 1 2 3; do "$@" && return 0; echo "retry $t: $*" >&2; sleep 30; done; return 1; }

capture() {
  echo "=== $(date -Is) capture -> wsbench-acts:/$ACTS"
  (cd "$GW" && $M run $W/wsbench_capture_modal.py --prompts-json "$ROWS" --acts-subdir $ACTS \
      --layers $LAYERS --max-positions 1024 --strict) || return 1
}

reads() {
  echo "=== $(date -Is) reads (logs $LOGS/read_*_$TS.log)"
  local jobs; jobs=$(mktemp --suffix=.json)
  echo "[{\"acts\": \"$ACTS\", \"out\": \"$G-jlens\", \"layers\": [28, 36, 44, 52]}]" > "$jobs"
  (cd "$GW" && $M run $W/wsbench_read_modal.py::main \
      --ckpt agu18dec/local-workspace:ckpts/ao/rl/s3d.ddp600.s0/iter_000600 --merged-name s3d-rl600 \
      --run hb-overrefusal${TAG:+-$TAG}-s3d-rl600 --acts-subdir $ACTS --layers $LAYERS --k 3 --shards $SHARDS \
      --prompt-kind concepts_raw --transform unit > "$LOGS/read_s3d_$TS.log" 2>&1; echo "EXIT=$?" >> "$LOGS/read_s3d_$TS.log") &
  (cd "$GW" && $M run $W/wsbench_nla_modal.py --lens mlnla-rl600 --run hb-overrefusal${TAG:+-$TAG}-mlnla-rl600 \
      --acts-subdirs $ACTS --layers $LAYERS --n-samples 3 --shards $((SHARDS * 2)) \
      > "$LOGS/read_mlnla_$TS.log" 2>&1; echo "EXIT=$?" >> "$LOGS/read_mlnla_$TS.log") &
  (cd "$GW" && $M run $W/wsbench_tokenlens_modal.py --lens jlens --jobs-json "$jobs" \
      > "$LOGS/read_jlens_$TS.log" 2>&1; echo "EXIT=$?" >> "$LOGS/read_jlens_$TS.log") &
  wait
  grep -H "EXIT=" "$LOGS"/read_*_"$TS".log
  ! grep -q "EXIT=[1-9]" "$LOGS"/read_*_"$TS".log
}

fetch() {
  for spec in "${ARMS[@]}"; do
    read -r arm gen kind <<< "$spec"
    [[ -z ${ONLY_ARMS:-} || ,$ONLY_ARMS, == *,$arm,* ]] || continue
    echo "=== $(date -Is) fetch + convert $arm ($GEN_VOLUME:/$gen, $kind)"
    mkdir -p "$GEN_ROOT" "$DST/$arm"
    rm -rf "${GEN_ROOT:?}/${gen:?}"
    # the launchers resolve their gen volume in-container, so the env var does not always reach
    # the writer (2026-10-27: the J-lens read wrote to its default wsbench-gen-4): find the dir
    vol=""
    for v in "$GEN_VOLUME" wsbench-gen-2 wsbench-gen-4; do
      if $M volume ls "$v" "/$gen" >/dev/null 2>&1; then vol=$v; break; fi
    done
    [[ -n $vol ]] || { echo "no volume holds /$gen" >&2; return 1; }
    echo "    found on $vol"
    (cd "$GEN_ROOT" && retry $M volume get "$vol" "/$gen" . --force >/dev/null) || return 1
    (cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$GEN_ROOT/$gen" \
        out="$DST/$arm/overrefusal.jsonl" kind="$kind" | tail -1) || return 1
    python3 - "$DST/$arm/overrefusal.jsonl" "$HERE/data/overrefusal/items.json" "$LAYERS" "$kind" "$ROWS" <<'PY' || return 1
import json, sys
rows = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
items = json.load(open(sys.argv[2]))["items"]
labels = {r["label"] for r in json.load(open(sys.argv[5]))}  # only the items this capture holds
items = [it for it in items if it["id"] in labels]
layers = [int(x) for x in sys.argv[3].split(",")]
want = {(it["id"], la, s["pos"]) for it in items for la in layers for s in it["sites"]}
keys = [(r["id"], r["layer"], r["pos"]) for r in rows]
have = set(keys)
miss, off, dup = want - have, have - want, len(keys) - len(have)
import collections
# the O-lens worker drops "contaminated" samples, so a prose cell may carry fewer than k = 3
ks = collections.Counter(len(r.get("samples", [])) for r in rows if sys.argv[4] == "prose")
bad_k = 0  # 0-sample rows load as empty cells (no mentions); reported, not fatal
print(f"{sys.argv[1]}: {len(rows)} rows, on-grid {len(have & want)}/{len(want)}, missing {len(miss)}, "
      f"off-grid {len(off)}, dup {dup}, samples per prose row {dict(ks)}")
sys.exit(1 if (miss or off or dup or bad_k or not rows) else 0)
PY
  done
}

merge() {  # rev 7: old readouts + the v2add batch -> $HBENCH_OR_READOUTS_V2/<arm>/overrefusal.jsonl
  local base=${HBENCH_OR_READOUTS:-/workspace/hb-readouts-overrefusal}
  local v2=${HBENCH_OR_READOUTS_V2:-/workspace/hb-readouts-overrefusal-v2}
  for spec in "${ARMS[@]}"; do
    read -r arm _ _ <<< "$spec"
    mkdir -p "$v2/$arm"
    local parts=("$base/$arm/overrefusal.jsonl")
    for t in v2add v2b; do  # rev 7 additions, rev 8 pair controls
      [[ -s "$base-$t/$arm/overrefusal.jsonl" ]] && parts+=("$base-$t/$arm/overrefusal.jsonl")
    done
    cat "${parts[@]}" > "$v2/$arm/overrefusal.jsonl"
    echo "merged $arm: $(wc -l < "$v2/$arm/overrefusal.jsonl") rows"
  done
}

case "$STEP" in
  merge) merge ;;
  capture) capture ;;
  reads) reads ;;
  fetch) fetch ;;
  all) capture && reads && fetch ;;
  *) echo "usage: $0 [capture|reads|fetch|all]" >&2; exit 2 ;;
esac
rc=$?
echo "=== $(date -Is) READS_${STEP^^} rc=$rc"
exit $rc
