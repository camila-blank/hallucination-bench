#!/usr/bin/env bash
# fetch_story_readouts.sh <arm> <gen-dir-name> <prose|tokens> <layers: 20:61:4 | 42 | csv> [volume]
#   (plan 0003 §Readouts)
# Fetch a story gen dir from Modal, convert it to the readout contract (wsbench-pub
# convert-gen-dir) and write /workspace/hb-readouts-story/<arm>/hallucination.jsonl, then check it
# is exactly the story bank's expected grid (data/story/items.json x <layers>): no missing,
# off-grid or duplicate (id, layer, pos) rows.
set -euo pipefail
arm=$1; gen=$2; kind=$3; layers=$4; vol=${5:-wsbench-gen-3}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
G=${STORY_GEN_ROOT:-/workspace/hb-story-gen}
DST=${STORY_READOUTS_ROOT:-/workspace/hb-readouts-story}
WSPUB=${WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
mkdir -p "$G" "$DST/$arm"
rm -rf "${G:?}/$gen"  # volume get --force never deletes stale shard files
cd "$G" && MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest volume get "$vol" "/$gen" . --force >/dev/null
cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$G/$gen" out="$DST/$arm/hallucination.jsonl" kind="$kind" | tail -1
python3 - "$DST/$arm/hallucination.jsonl" "$HERE/data/story/items.json" "$layers" <<'PY'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
items = json.load(open(sys.argv[2]))["items"]
spec = sys.argv[3]
if ":" in spec:
    a, b, c = (int(x) for x in spec.split(":"))
    layers = list(range(a, b, c))
else:
    layers = [int(x) for x in spec.split(",")]
want = {(it["id"], la, s["pos"]) for it in items for la in layers for s in it["sites"]}
keys = [(r["id"], r["layer"], r["pos"]) for r in rows]
have = set(keys)
n_dup = len(keys) - len(have)
miss, off = want - have, have - want
print(f"{sys.argv[1]}: {len(rows)} rows, expected layers {layers}, on-grid "
      f"{len(have & want)}/{len(want)}, missing {len(miss)}, off-grid {len(off)}, dup {n_dup}")
sys.exit(1 if (miss or off or n_dup or not rows) else 0)
PY
