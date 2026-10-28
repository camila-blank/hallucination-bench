#!/usr/bin/env bash
# fetch_readouts.sh <arm> <gen-dir-name> <prose|tokens> [volume]   (plan 0018 §Readouts step 3)
# Modal gen dir -> wsbench convert-gen-dir -> $HBENCH_SAD_READOUTS/<arm>/sad.jsonl, then a grid
# check against data/sad/selected.json x layers 28,36,44,52 (no missing / off-grid / duplicate).
set -euo pipefail
export UV_CACHE_DIR=${HBENCH_SAD_UV_CACHE:-/root/.cache/uv-hbsad}  # never the shared cache
arm=$1; gen=$2; kind=$3; vol=${4:-wsbench-gen-3}
[[ -n $arm && $arm != */* && -n $gen && $gen != */* ]] || { echo "bad args" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
G=${HBENCH_SAD_GEN:-/workspace/hb-sad-gen}
DST=${HBENCH_SAD_READOUTS:-/workspace/hb-readouts-sad}
WSPUB=${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
mkdir -p "$G" "$DST/$arm"
rm -rf "${G:?}/${gen:?}"
cd "$G" && MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest volume get "$vol" "/$gen" . --force >/dev/null
cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$G/$gen" out="$DST/$arm/sad.jsonl" kind="$kind" | tail -1
python3 - "$DST/$arm/sad.jsonl" "$HERE/data/sad/selected.json" <<'PY'
import json, sys
items = json.load(open(sys.argv[2]))["items"]
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
