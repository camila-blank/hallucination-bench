#!/usr/bin/env bash
# agentic_misalignment run (plan 0008, "Run"): two steps with hand audits in between.
#
#   scripts/run_am.sh step1    # tmux session hbench-am-detect: intent + Stage D (+ J-lens Stage S)
#                              # + am_audit.py sample / screen / summ-only
#   scripts/run_am.sh step2    # tmux session hbench-am-map: Stage M (+ summ) + sample-tiers + report
#
# Between the steps: hand-label the step-1 sheets and the 32 intent results (overrides in
# data/agentic_misalignment/intent_overrides.json), run `am_audit.py score`, check the gates.
# After step 2: label audit/tiers.jsonl, then `am_audit.py score` and `hbench report --eval
# agentic_misalignment` again (the report labels the unrelated rate "uncorrected" until then).
#
# Each `hbench run` that exits 4 (failed calls cached) is re-run (cache resume) up to 3 times;
# exit 3 (auth / config) stops the step. CONCURRENCY must be >= rpm x p50 latency (Little's
# law): set it from the pilot (default 600 ~ 60 s map / 25 s detect calls).
#
# Monitor:  tail -f logs/am_<step>_<timestamp>.log      Attach:  tmux attach -t hbench-am-<detect|map>
set -uo pipefail
cd "$(dirname "$0")/.."

STEP=${1:-}
OUT=${OUT:-outputs}
READOUTS_ROOT=${READOUTS_ROOT:-/workspace/wsbench-paper-data/readouts}
CONCURRENCY=${CONCURRENCY:-600}
RPM_DETECT=${RPM_DETECT:-1200}
RPM_MAP=${RPM_MAP:-450}
case "$STEP" in
  step1) SESSION=${SESSION:-hbench-am-detect} ;;
  step2) SESSION=${SESSION:-hbench-am-map} ;;
  *) echo "usage: $0 step1|step2" >&2; exit 2 ;;
esac

if [[ -z "${HBENCH_IN_TMUX:-}" ]]; then
  mkdir -p logs
  LOG="logs/am_${STEP}_$(date +%Y-%m-%d_%H%M).log"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "HBENCH_IN_TMUX=1 bash $(printf '%q ' "$0" "$@") 2>&1 | tee -a $LOG"
  echo "started tmux session '$SESSION'; monitor with: tail -f $LOG"
  exit 0
fi

AM=(--eval agentic_misalignment --readouts-root "$READOUTS_ROOT" --out "$OUT")

# run `hbench run ...`, re-running it on exit 4 (failed calls remain) up to 3 times
hb() {
  local rc
  for attempt in 1 2 3; do
    uv run --no-sync hbench run "${AM[@]}" "$@"
    rc=$?
    [[ $rc -eq 0 ]] && return 0
    if [[ $rc -ne 4 ]]; then
      echo "=== $(date -Is) hbench run exited $rc (not retried)"
      return $rc
    fi
    echo "=== $(date -Is) failed calls remain (attempt $attempt); resuming"
  done
  return 4
}

audit() { uv run --no-sync python scripts/am_audit.py "$@" --out "$OUT" --readouts-root "$READOUTS_ROOT"; }

if [[ "$STEP" == step1 ]]; then
  echo "=== $(date -Is) step 1: intent (per item; any one arm triggers it)"
  hb --arms s3d-rl600 --stages intent --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" || exit $?
  echo "=== $(date -Is) step 1: Stage D, prose arms"
  hb --arms s3d-rl600,nla-rl-L42 --stages detect --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" || exit $?
  echo "=== $(date -Is) step 1: Stage D + S, J-lens"
  hb --arms jlens --stages detect,summ --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" || exit $?
  echo "=== $(date -Is) step 1: audit sheets, screener, summary-only estimate"
  audit sample && audit screen --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY" \
    && audit summ-only --rpm "$RPM_DETECT" --concurrency "$CONCURRENCY"
  echo "=== $(date -Is) step 1 done: hand-label audit/*.jsonl + the intent audit, then am_audit.py score"
else
  echo "=== $(date -Is) step 2: Stage S + M (arms in the order nla-rl-L42, s3d-rl600, jlens)"
  hb --arms s3d-rl600,nla-rl-L42,jlens --stages summ,map --rpm "$RPM_MAP" --concurrency "$CONCURRENCY" || exit $?
  echo "=== $(date -Is) step 2: tier sheet + report"
  audit sample-tiers
  uv run --no-sync hbench report --eval agentic_misalignment --out "$OUT" --readouts-root "$READOUTS_ROOT"
  echo "=== $(date -Is) step 2 done: label audit/tiers.jsonl, then am_audit.py score and hbench report again"
fi
