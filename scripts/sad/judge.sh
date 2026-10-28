#!/usr/bin/env bash
# plan 0018: detector + verifier for the given arms (csv); exit 4 (failed calls cached) -> retry.
# Calls .venv/bin/hbench directly (going through uv would wait on the shared uv cache lockfile).
ARMS=${1:?arms}; CONC=${2:-32}
cd "$(dirname "$0")/../.."
for i in 1 2; do  # refusals are deterministic: more attempts recover ~nothing
  env -u ANTHROPIC_API_KEY PYTHONUNBUFFERED=1 .venv/bin/hbench sad run --arms "$ARMS" \
    --concurrency "$CONC" --out outputs
  rc=$?; echo "[attempt $i] rc=$rc"; [ $rc = 4 ] || break
done
echo "JUDGE_EXIT=$rc"
