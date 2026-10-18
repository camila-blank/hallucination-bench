#!/usr/bin/env bash
# Plan 0011 stage 1 per domain: J-lens summaries -> mentions -> calibration sets.
set -euo pipefail
HB=$(cd "$(dirname "$0")/../.." && pwd); cd "$HB"
S="env -u ANTHROPIC_API_KEY HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv/bin/python scripts/sibling_probe/um_ladder.py"
for D in ${DOMAINS:-occupation emotion capital_city holiday}; do
  echo "=== $D $(date)"
  $S --category $D summarize
  $S --category $D mentions
  $S --category $D todo
done
echo STAGE1DONE
