#!/usr/bin/env bash
# Full wsbench_chat run over all 11 stored lens arms (plan 0001, "Run"), in ONE tmux session.
#
#   scripts/run_all_arms.sh [extra `hbench run` args, e.g. --rpm 1000]
#
# Stage 1 (layer-matched slice: L44 for the 5-layer arms, L42 for nla-* and sae-l42) ->
# hbench report -> stage 2 (layers 20,36,48,60 of the 5-layer arms) -> hbench report.
# One `hbench run` process per stage judges every arm (one RPM pacer, one concurrency limit).
# Re-running the script resumes from the per-arm caches; only failed / partial / missing calls
# are re-queued. cells.jsonl is streamed, so memory stays well under the box's 32 GB cap.
#
# Monitor:  tail -f logs/run_<timestamp>.log      Attach:  tmux attach -t hbench
set -euo pipefail
cd "$(dirname "$0")/.."

READOUTS_ROOT=${READOUTS_ROOT:-/workspace/wsbench-paper-data/readouts}
OUT=${OUT:-outputs}
ARMS=${ARMS:-s3d-rl600,s3d-sft251,nla-rl-L42,nla-sft-L42,patchscopes,jlens,logit,tuned,rlens,template-v3,sae-l42}
SESSION=${SESSION:-hbench}

if [[ -z "${HBENCH_IN_TMUX:-}" ]]; then
  mkdir -p logs
  LOG="logs/run_$(date +%Y-%m-%d_%H%M).log"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "HBENCH_IN_TMUX=1 bash $(printf '%q ' "$0" "$@") 2>&1 | tee -a $LOG"
  echo "started tmux session '$SESSION'; monitor with: tail -f $LOG"
  exit 0
fi

echo "=== $(date -Is) stage 1: layer-matched slice (L44 / L42)"
uv run hbench run --readouts-root "$READOUTS_ROOT" --arms "$ARMS" --layers 42,44 --out "$OUT" "$@"
echo "=== $(date -Is) report after stage 1"
uv run hbench report --out "$OUT"
echo "=== $(date -Is) stage 2: layers 20,36,48,60 of the 5-layer arms"
uv run hbench run --readouts-root "$READOUTS_ROOT" --arms "$ARMS" --layers 20,36,48,60 --out "$OUT" "$@"
echo "=== $(date -Is) report after stage 2"
uv run hbench report --out "$OUT"
echo "=== $(date -Is) done"
