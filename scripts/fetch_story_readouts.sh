#!/usr/bin/env bash
# fetch_story_readouts.sh <arm> <gen-dir-name> <prose|tokens> [volume]   (plan 0003 §Readouts)
# Fetch a story gen dir from Modal, convert it to the readout contract (wsbench-pub
# convert-gen-dir) and write /workspace/hb-readouts-story/<arm>/hallucination.jsonl, then check it
# covers the story bank's expected grid (data/story/items.json x the file's layers).
set -euo pipefail
arm=$1; gen=$2; kind=$3; vol=${4:-wsbench-gen-3}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
G=${STORY_GEN_ROOT:-/workspace/hb-story-gen}
DST=${STORY_READOUTS_ROOT:-/workspace/hb-readouts-story}
WSPUB=${WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
mkdir -p "$G" "$DST/$arm"
cd "$G" && MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest volume get "$vol" "/$gen" . --force >/dev/null
cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$G/$gen" out="$DST/$arm/hallucination.jsonl" kind="$kind" | tail -1
python3 - "$DST/$arm/hallucination.jsonl" "$HERE/data/story/items.json" <<'PY'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
items = json.load(open(sys.argv[2]))["items"]
layers = sorted({r["layer"] for r in rows})
want = {(it["id"], la, s["pos"]) for it in items for la in layers for s in it["sites"]}
have = {(r["id"], r["layer"], r["pos"]) for r in rows}
miss = want - have
print(f"{sys.argv[1]}: {len(rows)} rows, layers {layers}, on-grid {len(have & want)}/{len(want)}, "
      f"missing {len(miss)}, off-grid {len(have - want)}")
sys.exit(1 if miss else 0)
PY
