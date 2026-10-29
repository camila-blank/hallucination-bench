#!/usr/bin/env bash
# fetch_readouts.sh <arm> <gen-dir-name> <prose|tokens> [volume]   (plan 0017 §Readouts)
# Fetch a MASK gen dir from Modal, convert it to the readout contract (wsbench-pub
# convert-gen-dir) and write $HBENCH_MASK_READOUTS/<arm>/mask.jsonl, then check it is exactly the
# read-set grid (data/mask/sites.json.gz x the file's layers): no missing, off-grid or duplicate rows.
set -euo pipefail
arm=$1; gen=$2; kind=$3; vol=${4:-wsbench-gen-3}
[[ -n $arm && $arm != */* && -n $gen && $gen != */* ]] || { echo "need <arm> <gen> (no slashes)" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
G=${HBENCH_MASK_GEN:-/workspace/hb-mask-gen}
DST=${HBENCH_MASK_READOUTS:-/workspace/hb-readouts-mask}
OUTF=${HBENCH_MASK_READOUTS_FILE:-mask.jsonl}              # plan 0019: mask_l11.jsonl
SITES=${HBENCH_MASK_SITES:-$HERE/data/mask/sites.json.gz}  # plan 0019: data/mask/l11/sites_l11.json.gz
WSPUB=${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
mkdir -p "$G" "$DST/$arm"
rm -rf "${G:?}/${gen:?}"  # volume get --force never deletes stale shard files
cd "$G" && MODAL_RESOURCE_ENV=camila-blank UV_CACHE_DIR=${HBENCH_UV_CACHE:-/workspace/.cache/uv-hbmask} uvx --python 3.12 modal@latest volume get "$vol" "/$gen" . --force >/dev/null
cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$G/$gen" out="$DST/$arm/$OUTF" kind="$kind" | tail -1
python3 - "$DST/$arm/$OUTF" "$SITES" <<'PY'
import gzip, json, sys
want_pos = {(it["task_id"], s["pos"]) for it in json.load(gzip.open(sys.argv[2], "rt"))["items"] for s in it["sites"]}
keys = []
with open(sys.argv[1]) as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            keys.append((r["id"], r["layer"], r["pos"]))
layers = sorted({k[1] for k in keys})
want = {(t, la, p) for t, p in want_pos for la in layers}
have = set(keys)
miss, off, dup = want - have, have - want, len(keys) - len(have)
print(f"{sys.argv[1]}: {len(keys)} rows, layers {layers}, on-grid {len(have & want)}/{len(want)}, "
      f"missing {len(miss)}, off-grid {len(off)}, dup {dup}")
sys.exit(1 if (miss or off or dup or not keys) else 0)
PY
