#!/bin/bash
# Memory-lean step 2 (plan 0008): Stage S + M per arm, 11-layer arms in layer chunks (each resumes
# from cache), then the tier sheet and the report. Order nla-rl-L42, s3d-rl600, jlens.
cd "$(cd "$(dirname "$0")/.." && pwd)"  # the repo root (plan 0012 D3)
R=${HBENCH_READOUTS_ROOT:-/workspace/wsbench-paper-data/readouts}
A="--eval agentic_misalignment --readouts-root $R --out outputs --rpm ${RPM_MAP:-450} --concurrency ${CONCURRENCY:-400}"
hb() { for t in 1 2 3; do uv run --no-sync hbench run $A "$@"; rc=$?; [ $rc -ne 4 ] && break; done; echo "=== RC $rc for $*"; [ $rc -eq 0 ] || [ $rc -eq 4 ] || exit $rc; }
hb --arms nla-rl-L42 --stages map
for L in 20,24,28 32,36,40 44,48,52 56,60; do hb --arms s3d-rl600 --stages map --layers $L; done
for L in 20,24,28,32 36,40,44,48 52,56,60; do hb --arms jlens --stages summ,map --layers $L; done
echo STEP2DONE
