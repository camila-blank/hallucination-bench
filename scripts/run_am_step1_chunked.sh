#!/bin/bash
# Memory-lean step 1 (plan 0008): the box is shared and other sessions can hold ~20 GB of the
# 32 GB cgroup cap, so arms run alone and 11-layer arms in layer chunks (each resumes from cache).
cd /workspace/hb-am-halluc
A="--eval agentic_misalignment --readouts-root /workspace/wsbench-paper-data/readouts --out outputs --rpm 1200 --concurrency 200"
hb() { for t in 1 2 3; do uv run --no-sync hbench run $A "$@"; rc=$?; [ $rc -ne 4 ] && break; done; echo "=== RC $rc for $*"; [ $rc -eq 0 ] || [ $rc -eq 4 ] || exit $rc; }
hb --arms nla-rl-L42 --stages detect
hb --arms s3d-rl600 --stages detect --layers 20,24,28
hb --arms s3d-rl600 --stages detect --layers 32,36,40
hb --arms s3d-rl600 --stages detect --layers 44,48,52
hb --arms s3d-rl600 --stages detect --layers 56,60
hb --arms jlens --stages detect,summ --layers 20,24,28,32
hb --arms jlens --stages detect,summ --layers 36,40,44,48
hb --arms jlens --stages detect,summ --layers 52,56,60
AU="--out outputs --readouts-root /workspace/wsbench-paper-data/readouts"
uv run --no-sync python scripts/am_audit.py sample $AU; echo "=== sample $?"
uv run --no-sync python scripts/am_audit.py screen $AU --rpm 1200 --concurrency 200; echo "=== screen $?"
uv run --no-sync python scripts/am_audit.py summ-only $AU --rpm 1200 --concurrency 200; echo "=== summ-only $?"
echo STEP1DONE
