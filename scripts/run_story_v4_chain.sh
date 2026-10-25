#!/usr/bin/env bash
# run_story_v4_chain.sh: one serial driver for the rest of the V4 story eval (one hbench writer at
# a time): wait for both v4-s3don-sft story shards (hallucination_story.{0,1}of2 on
# wsbench-public-readouts, from the fanout launched 2026-10-20), judge SFT + J-lens, then hand
# off to run_story_v4_rl600.sh (waits for RL600, reads it, judges all three, writes reports).
#   tmux new -d -s hbstory-chain "scripts/run_story_v4_chain.sh > logs/story_v4_chain_<ts>.log 2>&1"
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL=(uvx --python 3.12 modal@latest)
until [[ $("${MODAL[@]}" volume ls wsbench-public-readouts /v4-s3don-sft 2>/dev/null \
  | grep -c 'hallucination_story\.[01]of2\.jsonl' || true) -eq 2 ]]; do
  echo "=== $(date -Is) waiting for v4-s3don-sft story shards"; sleep 300
done
scripts/run_story_v4.sh v4-s3don-sft:wsbench-public-readouts,v4-jlens-std-n1000:wsbench-v4-readouts
exec scripts/run_story_v4_rl600.sh
