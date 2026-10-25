#!/usr/bin/env bash
# agentic_misalignment on DeepSeek-V4-Flash lenses: scripts/run_am.sh's stages, adapted.
#
#   scripts/run_am_v4.sh step1   # tmux hbench-am-v4-detect: intent + Stage D (+ J-lens S) + sheets
#   scripts/run_am_v4.sh step2   # tmux hbench-am-v4-map: Stage S + H + M, sheets, report
#
# Adaptations (vs run_am.sh): the V4 bank (HBENCH_AM_ITEMS, scripts/build_am_bank_v4.py), V4 arms
# (PROSE_ARMS, JLENS_ARM), layers 21,27,33 (depth-matched to Qwen 32,42,48), the readouts root
# outputs/v4/am_readouts (<arm>/agentic_misalignment.jsonl symlinks), out outputs/v4, judge
# commands under `env -u ANTHROPIC_API_KEY` (ant CLI OAuth), Stage H (verify, plan rev 13) in
# step 2 + the headline sheet. The intent overrides are the Qwen bank's (same prompts + prefills).
#
# Monitor: tail -f logs/am_v4_<step>_<timestamp>.log
set -uo pipefail
cd "$(dirname "$0")/.."

STEP=${1:-}
OUT=${OUT:-outputs/v4}
READOUTS_ROOT=${READOUTS_ROOT:-outputs/v4/am_readouts}
PROSE_ARMS=${PROSE_ARMS-v4-s3don-sft}
JLENS_ARM=${JLENS_ARM:-v4-jlens-std-n1000}
LAYERS=${LAYERS:-21,27,33}
CONCURRENCY=${CONCURRENCY:-128}
RPM_DETECT=${RPM_DETECT:-600}
RPM_MAP=${RPM_MAP:-300}
export HBENCH_AM_ITEMS=${HBENCH_AM_ITEMS:-$PWD/outputs/v4/am_bank/items.json}
case "$STEP" in
  step1) SESSION=${SESSION:-hbench-am-v4-detect} ;;
  step2) SESSION=${SESSION:-hbench-am-v4-map} ;;
  *) echo "usage: $0 step1|step2" >&2; exit 2 ;;
esac

if [[ -z "${HBENCH_IN_TMUX:-}" ]]; then
  mkdir -p logs
  LOG="logs/am_v4_${STEP}_$(date +%Y-%m-%d_%H%M).log"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "HBENCH_IN_TMUX=1 PROSE_ARMS=$PROSE_ARMS JLENS_ARM=$JLENS_ARM bash $(printf '%q ' "$0" "$@") 2>&1 | tee -a $LOG"
  echo "started tmux session '$SESSION'; monitor with: tail -f $LOG"
  exit 0
fi

AM=(--eval agentic_misalignment --readouts-root "$READOUTS_ROOT" --out "$OUT" --layers "$LAYERS")
ALL_ARMS=${PROSE_ARMS:+$PROSE_ARMS,}$JLENS_ARM  # PROSE_ARMS="" = J-lens only (O-lens readouts pending)

# exit 3 (auth) is retried too, up to 12 times a minute apart: the ant OAuth token is shared by
# every judge on this box and its rotation briefly revokes a running process's token (2026-10-20)
hb() {
  local rc n_auth=0 attempt=1
  while [[ $attempt -le 3 ]]; do
    env -u ANTHROPIC_API_KEY uv run --no-sync hbench run "${AM[@]}" "$@"
    rc=$?
    [[ $rc -eq 0 ]] && return 0
    if [[ $rc -eq 3 && $n_auth -lt 12 ]]; then
      n_auth=$((n_auth + 1)); echo "=== $(date -Is) auth error ($n_auth); retrying in 60 s"; sleep 60; continue
    fi
    attempt=$((attempt + 1))
    if [[ $rc -ne 4 ]]; then
      echo "=== $(date -Is) hbench run exited $rc (not retried)"
      return $rc
    fi
    echo "=== $(date -Is) failed calls remain (attempt $attempt); resuming"
  done
  return 4
}

go_on() {
  if [[ $1 -eq 4 ]]; then echo "=== $(date -Is) continuing with cached failures (reported in the summary)"; return 0; fi
  exit "$1"
}

audit() {
  local rc
  for i in 1 2 3 4 5 6; do
    env -u ANTHROPIC_API_KEY uv run --no-sync python scripts/am_audit.py "$@" --out "$OUT" \
      --readouts-root "$READOUTS_ROOT" --arms "$ALL_ARMS" && return 0
    rc=$?
    [[ $rc -eq 4 ]] && return 0  # cached failures (refusals): reported, not retried
    echo "=== $(date -Is) am_audit $1 exited $rc (try $i); retrying in 60 s"; sleep 60
  done
  return $rc
}

if [[ "$STEP" == step1 ]]; then
  echo "=== $(date -Is) step 1: intent"
  hb --arms "$JLENS_ARM" --stages intent --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" || go_on $?
  if [[ -n "$PROSE_ARMS" ]]; then
    echo "=== $(date -Is) step 1: Stage D, prose arms ($PROSE_ARMS)"
    hb --arms "$PROSE_ARMS" --stages detect --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" || go_on $?
  fi
  echo "=== $(date -Is) step 1: Stage D + S, J-lens"
  hb --arms "$JLENS_ARM" --stages detect,summ --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" || go_on $?
  echo "=== $(date -Is) step 1: audit sheets, screener, summary-only estimate"
  audit sample && audit screen --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" \
    && audit summ-only --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY"
  echo "=== $(date -Is) step 1 done"
else
  echo "=== $(date -Is) step 2: Stage S + H + M ($ALL_ARMS)"
  hb --arms "$ALL_ARMS" --stages summ,verify,map --rpm "$RPM_MAP" --concurrency "$CONCURRENCY" || go_on $?
  echo "=== $(date -Is) step 2: headline + tier sheets, report"
  audit sample-headline
  audit sample-tiers
  env -u ANTHROPIC_API_KEY uv run --no-sync hbench report --eval agentic_misalignment --out "$OUT" \
    --readouts-root "$READOUTS_ROOT"
  echo "=== $(date -Is) step 2 done"
fi
