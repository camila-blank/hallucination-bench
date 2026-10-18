#!/usr/bin/env bash
# add_layers.sh <arm> <gen-dir-name> <prose|tokens>
# Fetch a Modal gen dir of extra layers, convert it to the readout contract, and merge it with the
# arm's existing readouts into /workspace/hb-readouts/<arm>/hallucination.jsonl (same arm name, so
# already-judged layers are reused from the cache).
set -euo pipefail
arm=$1; gen=$2; kind=$3
G=${HBENCH_MLNLA_GEN:-/workspace/gw-mlnla/outputs/oracle_lens_evals/olens_sglang}
cd $G && MODAL_RESOURCE_ENV=camila-blank uvx --python 3.12 modal@latest volume get wsbench-gen-3 /$gen . --force >/dev/null
tmp=/tmp/hbx6_$arm.jsonl
cd ${HBENCH_WSBENCH_PUB:-/workspace/root/wsbench-pub} && ${HBENCH_WSBENCH_PY:-/workspace/root/wsbench-venv/bin/python} -m wsbench convert-gen-dir gen_dir=$G/$gen out=$tmp kind=$kind | tail -1
src=${HBENCH_READOUTS_ROOT:-/workspace/wsbench-paper-data/readouts}/$arm/hallucination.jsonl
dst=${HBENCH_EXTRA_READOUTS:-/workspace/hb-readouts}/$arm
[ -L $dst ] && rm $dst
mkdir -p $dst
python3 - "$src" "$tmp" "$dst/hallucination.jsonl" <<'PY'
import json, sys, collections
src, new, out = sys.argv[1:]
rows, seen = [], set()
for f in (src, new):
    for line in open(f):
        r = json.loads(line); k = (r["id"], r["layer"], r["pos"])
        if k in seen: continue
        seen.add(k); rows.append(r)
with open(out, "w") as fh:
    for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
c = collections.Counter(r["layer"] for r in rows)
print(out, len(rows), sorted(c.items()))
PY
