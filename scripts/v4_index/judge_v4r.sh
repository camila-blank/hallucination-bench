#!/usr/bin/env bash
# judge_v4r.sh chat|story   (plan 0014) — judge the merged V4 readouts (readout_jobs.py merge) into
# $OUT/{wsbench_chat,story}/<judge>/<arm>/ with the existing V4 runs' recipe: chat = every stage on
# the regenerated V4 chat bank, story = --stages map on the V4 story bank. Seed $OUT's arm dirs with
# the old runs' cache.jsonl first so already-judged cells are cache hits (content fingerprints:
# a changed readout or bank item is re-judged). Resumes on rc=4. Run in tmux; logs to logs/.
set -euo pipefail
cd "$(dirname "$0")/../.."
FAM=${1:?chat|story}
OUT=${HBENCH_V4R_OUT:-/workspace/hb-v4/outputs/v4r}
ARMS=${ARMS:-v4-jlens-std-n1000,v4-s3don-sft,v4-s3don-rl600}
CONC=${CONC:-64}  # the org output-token limit (2M/min, shared) 429s at 128
PY=${PY:-/workspace/hb-v4/.venv/bin/python}
export PYTHONPATH=src PYTHONUNBUFFERED=1
case "$FAM" in
  chat)
    export HBENCH_WSBENCH_CHAT_ITEMS=${HBENCH_WSBENCH_CHAT_ITEMS:-/workspace/root/wsbench-pub-v4-opus/evals/hallucination/items.json}
    ARGS=(--eval wsbench_chat --readouts-root "$OUT/chat_readouts") ;;
  story)
    export HBENCH_STORY_ITEMS=${HBENCH_STORY_ITEMS:-/workspace/hb-v4/outputs/v4/story_bank/items.json}
    ARGS=(--eval story --readouts-root "$OUT/story_readouts" --stages map) ;;
  *) echo "usage: $0 chat|story" >&2; exit 2 ;;
esac
for a in 1 2 3 4 5 6; do
  rc=0
  env -u ANTHROPIC_API_KEY "$PY" -m hallucination_bench.cli run --out "$OUT" "${ARGS[@]}" \
    --arms "$ARMS" --concurrency "$CONC" || rc=$?
  [[ $rc -eq 0 ]] && break
  # rc 3 = judge config error; seen once as a 401 while the shared OAuth profile token rotated
  # (2026-10-25), so it gets a pause and the same bounded retries
  [[ $rc -eq 4 || $rc -eq 3 ]] || { echo "hbench run rc=$rc (not retryable)" >&2; exit $rc; }
  [[ $rc -eq 3 ]] && sleep 120
  echo "=== $(date -Is) rc=$rc attempt $a; resuming"
done
EV=${ARGS[1]}
env -u ANTHROPIC_API_KEY "$PY" -m hallucination_bench.cli report --eval "$EV" --out "$OUT"
env -u ANTHROPIC_API_KEY "$PY" -m hallucination_bench.cli report --eval "$EV" --out "$OUT" --view v2
echo "=== $(date -Is) DONE $FAM"
