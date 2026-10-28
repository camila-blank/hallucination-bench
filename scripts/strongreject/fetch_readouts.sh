#!/usr/bin/env bash
# fetch_readouts.sh <arm> <gen-dir-name> <prose|tokens> [volume]     (plan 0016 §Readouts)
# Fetch a strongreject gen dir from Modal, convert it to the readout contract (wsbench-pub
# convert-gen-dir) into $HBENCH_SR_READOUTS/<arm>/strongreject.jsonl, then check it is exactly
# the bank's grid (data/strongreject/items.json sites x the layers present): no missing,
# off-grid or duplicate (id, layer, pos) rows, and every row's token equals the site token.
set -euo pipefail
arm=$1; gen=$2; kind=$3; vol=${4:-wsbench-gen-3}
[[ -n $arm && $arm != */* && -n $gen && $gen != */* ]] || { echo "bad <arm> <gen>" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
G=${HBENCH_SR_GEN:-/workspace/hb-strongreject-gen}
DST=${HBENCH_SR_READOUTS:-/workspace/hb-readouts-strongreject}
WSPUB=${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub}
WSPY=${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python}
mkdir -p "$G" "$DST/$arm"
rm -rf "${G:?}/${gen:?}"  # volume get --force never deletes stale shard files
export MODAL_RESOURCE_ENV=${MODAL_RESOURCE_ENV:-camila-blank}
M="${MODAL_BIN:-uvx --python 3.12 modal@latest} volume"
mkdir -p "$G/$gen"
for d in $($M ls "$vol" "/$gen"); do  # per item dir: a bulk volume get can miss dirs
  name=$(basename "$d")
  for t in 1 2 3; do $M get "$vol" "/$gen/$name" "$G/$gen/" --force >/dev/null 2>&1 && break; done
done
cd "$WSPUB" && "$WSPY" -m wsbench convert-gen-dir gen_dir="$G/$gen" out="$DST/$arm/strongreject.jsonl" kind="$kind" | tail -1
"$HERE/.venv/bin/python" - "$DST/$arm/strongreject.jsonl" "$HERE/data/strongreject/items.json" <<'PY'
import json, sys
from hallucination_bench.readouts import decode_token
items = json.load(open(sys.argv[2]))["items"]
sites = {(it["id"], s["pos"]): s for it in items for s in it["sites"]}
keys, bad_tok, ex = [], 0, []
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    r = json.loads(line)
    keys.append((r["id"], r["layer"], r["pos"]))
    s = sites.get((r["id"], r["pos"]))
    if s and "token" in r and not s.get("split_char") and r["token"] != s["token"] \
            and decode_token(r["token"]) != s["token"]:  # token lenses keep byte-level forms
        bad_tok += 1
        if len(ex) < 3:
            ex.append((r["id"], r["pos"], r["token"], s["token"]))
layers = sorted({k[1] for k in keys})
want = {(i, la, p) for (i, p) in sites for la in layers}
have = set(keys)
miss, off, dup = want - have, have - want, len(keys) - len(have)
print(f"{sys.argv[1]}: {len(keys)} rows, layers {layers}, on-grid {len(have & want)}/{len(want)}, "
      f"missing {len(miss)}, off-grid {len(off)}, dup {dup}, token mismatches {bad_tok} {ex}")
sys.exit(1 if (miss or off or dup or bad_tok or not keys) else 0)
PY
