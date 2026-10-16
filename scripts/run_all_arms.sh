#!/usr/bin/env bash
# Full wsbench_chat run over all 11 stored lens arms (plan 0001, "Run"), in ONE tmux session.
#
#   scripts/run_all_arms.sh [extra `hbench run` args, e.g. --rpm 1000]
#
# Stage 1 (layer-matched slice: L44 for the 5-layer arms, L42 for nla-* and sae-l42) ->
# hbench report -> stage 2 (layers 20,36,48,60 of the 5-layer arms) -> hbench report ->
# hbench gate (plan 0010; exit 2 = needs --yes, 3 = fatal, 4 = partial) -> hbench report --view v2.
# One `hbench run` process per stage judges every arm (one RPM pacer, one concurrency limit).
# Re-running the script resumes from the per-arm caches; only failed / partial / missing calls
# are re-queued. cells.jsonl is streamed, so memory stays well under the box's 32 GB cap.
#
# Monitor:  tail -f logs/run_<timestamp>.log      Attach:  tmux attach -t hbench
set -uo pipefail
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

# A stage that ends with failed calls exits non-zero; re-run it (cache resume) up to 3 times,
# then carry on so the report still shows what failed.
stage() {
  for attempt in 1 2 3; do
    uv run hbench run --readouts-root "$READOUTS_ROOT" --arms "$ARMS" --out "$OUT" "$@" && return 0
    echo "=== $(date -Is) stage exited non-zero (attempt $attempt); resuming"
  done
}

echo "=== $(date -Is) stage 1: layer-matched slice (L44 / L42)"
stage --layers 42,44 "$@"
echo "=== $(date -Is) report after stage 1"
uv run hbench report --out "$OUT"
echo "=== $(date -Is) stage 2: layers 20,36,48,60 of the 5-layer arms"
stage --layers 20,36,48,60 "$@"
echo "=== $(date -Is) report after stage 2"
uv run hbench report --out "$OUT"

# Plan 0010: the sibling gate, once (NOT wrapped in stage(): it must not be retried 3 times).
# It gates every arm of data/arms.json gated_arms (this script's ARMS from $OUT; the extra arms
# from their registered cells). Unattended runs never pass --yes: an estimate over
# $GATE_AUTO_USD (default $300) is skipped and the old labels generation still applies.
echo "=== $(date -Is) sibling gate (hbench gate)"
uv run hbench gate --out "$OUT/sibling_gate" --cells-root "$OUT/wsbench_chat/claude-opus-5-5" \
  ${GATE_ARGS:-}
rc=$?
case $rc in
  0) echo "=== gate exit 0: done" ;;
  2) echo "=== gate exit 2: needs --yes (estimate over \$GATE_AUTO_USD); the previous labels apply only to arms whose unit set is unchanged (the rebuild refuses the rest)" ;;
  3) echo "=== gate exit 3: fatal (config / auth / spend limit / unregistered arm); see above" ;;
  4) echo "=== gate exit 4: partial (resample timeout / job failure); re-run resumes" ;;
  *) echo "=== gate exit $rc: unexpected" ;;
esac
echo "=== $(date -Is) final report (v2, sibling labels from $OUT/sibling_gate/labels)"
uv run hbench report --out "$OUT" --view v2
# the rebuild refuses a stamped labels file whose unit hash differs from the cells (rev 20)
refused=$(grep -lE '"sibling_labels_guard": *"mismatch"' "$OUT"/wsbench_chat/*/*/summary_v2.json 2>/dev/null)
if [[ -n "$refused" ]]; then
  echo "=== sibling labels REFUSED (unit hash mismatch; re-run hbench gate) for:"
  for f in $refused; do echo "    $(basename "$(dirname "$f")")"; done
fi
echo "=== $(date -Is) done"
if [[ $rc -eq 3 ]]; then
  echo "=== exiting 3: the gate failed fatally (see above)"
  exit 3
fi
