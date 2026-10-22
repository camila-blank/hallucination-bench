#!/usr/bin/env bash
# run_story_v4.sh <arm:volume>[,<arm:volume>...]   (plan 0007 story eval, DeepSeek-V4-Flash port)
#
# For each arm: fetch <arm>/hallucination_story.*of*.jsonl from the Modal volume (the V4 producers
# gw-v4-flash-ar scripts/v4_flash/wsbench_v4_modal.py -> wsbench-public-readouts, gw-v4-jlens
# wsbench_v4_jlens_modal.py -> wsbench-v4-readouts, both with WSB_BENCH=<bench copy with the
# hallucination_story family>), concatenate the shards to $R/<arm>/hallucination.jsonl and check
# the exact expected grid of the V4 story bank ($HBENCH_STORY_ITEMS x 5 layers). Then judge every
# fetched arm already in $R with `hbench run --eval story --stages map` and write both reports.
# Re-run resumes (cache). Run inside tmux; logs to logs/story_v4_<ts>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export HBENCH_STORY_ITEMS=${HBENCH_STORY_ITEMS:-$PWD/outputs/v4/story_bank/items.json}
R=${HBENCH_STORY_V4_READOUTS:-$PWD/outputs/v4/story_readouts}
OUT=${HBENCH_STORY_V4_OUT:-outputs/v4}
LAYERS=${STORY_V4_LAYERS:-15,24,30,36,39}
MODAL=(uvx --python 3.12 modal@latest)
arms=()
for av in ${1//,/ }; do
  arm=${av%%:*}; vol=${av#*:}
  tmp=$(mktemp -d)
  for f in $("${MODAL[@]}" volume ls "$vol" "/$arm" 2>/dev/null | grep -o 'hallucination_story\.[0-9]*of[0-9]*\.jsonl' | sort -u); do
    "${MODAL[@]}" volume get "$vol" "/$arm/$f" "$tmp/$f" --force >/dev/null
  done
  shopt -s nullglob
  shards=("$tmp"/hallucination_story.*of*.jsonl)
  shopt -u nullglob
  [[ ${#shards[@]} -gt 0 ]] || { echo "no hallucination_story shards for $arm on $vol" >&2; exit 2; }
  mkdir -p "$R/$arm"
  cat "${shards[@]}" > "$R/$arm/hallucination.jsonl"
  rm -rf "$tmp"
  python3 - "$R/$arm/hallucination.jsonl" "$HBENCH_STORY_ITEMS" "$LAYERS" "${#shards[@]}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
items = json.load(open(sys.argv[2]))["items"]
layers = [int(x) for x in sys.argv[3].split(",")]
want = {(it["id"], la, s["pos"]) for it in items for la in layers for s in it["sites"]}
keys = [(r["id"], r["layer"], r["pos"]) for r in rows]
have = set(keys)
miss, off, dup = want - have, have - want, len(keys) - len(have)
print(f"{sys.argv[1]}: {sys.argv[4]} shards, {len(rows)} rows, on-grid {len(have & want)}/"
      f"{len(want)}, missing {len(miss)}, off-grid {len(off)}, dup {dup}")
sys.exit(1 if (miss or off or dup or not rows) else 0)
PY
  arms+=("$arm")
done
ARMS=$(IFS=,; echo "${arms[*]}")
echo "=== $(date -Is) judging $ARMS (bank $HBENCH_STORY_ITEMS)"
for a in 1 2 3 4; do
  rc=0
  env -u ANTHROPIC_API_KEY uv run --no-sync hbench run --eval story --out "$OUT" \
    --readouts-root "$R" --arms "$ARMS" --stages map --concurrency 48 || rc=$?
  [[ $rc -eq 0 ]] && break
  [[ $rc -eq 4 ]] || { echo "hbench run rc=$rc (not retryable)" >&2; exit $rc; }
  echo "=== $(date -Is) rc=4 attempt $a; resuming"
done
env -u ANTHROPIC_API_KEY uv run --no-sync hbench report --eval story --out "$OUT"
env -u ANTHROPIC_API_KEY uv run --no-sync hbench report --eval story --out "$OUT" --view v2
echo "=== $(date -Is) DONE"
