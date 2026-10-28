#!/usr/bin/env bash
# fetch_readouts.sh <arm> <gen-dir-name> <prose|tokens> [volume] [rev5]   (plan 0018 §Readouts 3)
# With a 5th arg "rev5": write <arm>/sad_rev5.jsonl, check it covers only the rev-5 top-up items,
# then concatenate the rev-4 file (sad_rev4.jsonl, the original sad.jsonl) + it into sad.jsonl
# and check the full grid.
# Modal gen dir -> wsbench convert-gen-dir -> $HBENCH_SAD_READOUTS/<arm>/sad.jsonl, then a grid
# check against data/sad/selected.json x layers 28,36,44,52 (no missing / off-grid / duplicate).
set -euo pipefail
export UV_CACHE_DIR=${HBENCH_SAD_UV_CACHE:-/root/.cache/uv-hbsad}  # never the shared cache
arm=$1; gen=$2; kind=$3; vol=${4:-wsbench-gen-3}; part=${5:-}
OUTF=sad.jsonl; [[ $part == rev5 ]] && OUTF=sad_rev5.jsonl
[[ -n $arm && $arm != */* && -n $gen && $gen != */* ]] || { echo "bad args" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
G=${HBENCH_SAD_GEN:-/workspace/hb-sad-gen}
DST=${HBENCH_SAD_READOUTS:-/workspace/hb-readouts-sad}
WSPUB=${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
mkdir -p "$G" "$DST/$arm"
rm -rf "${G:?}/${gen:?}"
cd "$G" && MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest volume get "$vol" "/$gen" . --force >/dev/null
cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$G/$gen" out="$DST/$arm/$OUTF" kind="$kind" | tail -1
check() {
python3 - "$1" "$HERE/data/sad/selected.json" "$2" <<'PY'
import json, sys
items = json.load(open(sys.argv[2]))["items"]
if sys.argv[3] == "rev5":
    items = [it for it in items if it.get("batch") == "rev5"]
layers = [28, 36, 44, 52]
want = {(it["id"], la, s["pos"]) for it in items for la in layers for s in it["sites"]}
keys = []
with open(sys.argv[1]) as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            keys.append((r["id"], r["layer"], r["pos"]))
have = set(keys)
miss, off = want - have, have - want
print(f"{sys.argv[1]}: {len(keys)} rows, on-grid {len(have & want)}/{len(want)}, missing "
      f"{len(miss)}, off-grid {len(off)}, dup {len(keys) - len(have)}")
sys.exit(1 if (miss or off or len(keys) != len(have) or not keys) else 0)
PY
}
if [[ $part == rev5 ]]; then
  check "$DST/$arm/sad_rev5.jsonl" rev5
  [[ -f $DST/$arm/sad_rev4.jsonl ]] || mv "$DST/$arm/sad.jsonl" "$DST/$arm/sad_rev4.jsonl"
  cat "$DST/$arm/sad_rev4.jsonl" "$DST/$arm/sad_rev5.jsonl" > "$DST/$arm/sad.jsonl"
fi
check "$DST/$arm/sad.jsonl" all
