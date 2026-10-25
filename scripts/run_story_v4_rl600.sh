#!/usr/bin/env bash
# run_story_v4_rl600.sh: add the V4 O-lens RL step-600 arm (v4-s3don-rl600) to the V4 story eval.
#   tmux new -d -s hbstory-rl600 "scripts/run_story_v4_rl600.sh > logs/story_v4_rl600_$(date -u +%Y%m%d_%H%M).log 2>&1"
# 1. waits until runs/s3don/rl/iter_000600 exists on the v4-flash volume (adapter_model.safetensors);
# 2. reads the story bank with it (gw-v4-flash-ar wsbench_v4_modal.py fanout, WSB_BENCH = the
#    bench copy with the hallucination_story family, 2 shards; B300:8 / B200:8 / B300:4);
# 3. re-runs run_story_v4.sh on all three arms (the SFT / J-lens caches make those no-ops) and
#    rewrites both reports.
set -euo pipefail
cd "$(dirname "$0")/.."
GW=${HBENCH_GW_V4_FLASH_AR:-/workspace/gw-v4-flash-ar}
BENCH=${HBENCH_STORY_V4_BENCH:-/workspace/root/wsbench-pub-v4-story}
GPUS=${STORY_V4_GPUS:-B300:8,B200:8,B300:4}
MODAL=(uvx --python 3.12 modal@latest)
until "${MODAL[@]}" volume ls v4-flash runs/s3don/rl/iter_000600 2>/dev/null | grep -q adapter_model.safetensors; do
  echo "=== $(date -Is) waiting for runs/s3don/rl/iter_000600"; sleep 600
done
sleep 300  # let the checkpoint commit settle
echo "=== $(date -Is) RL600 adapter present; reading the story bank"
# GPU scarcity (2026-10-20: B300:8 / B200:8 / B300:4 each pended for hours): launch one detached
# fanout per allowed GPU shape (never H200: OOMs loading V4); produce() skips a shard whose file
# already exists, so a late-starting duplicate exits before loading the model
for gpu in ${GPUS//,/ }; do
  (cd "$GW/scripts/v4_flash" && WSB_BENCH=$BENCH PYTHONUNBUFFERED=1 "${MODAL[@]}" run --detach \
    wsbench_v4_modal.py::fanout --arm v4-s3don-rl600 --families hallucination_story \
    --n-shards 2 --gpu "$gpu") > "logs/story_v4_olens_rl600_${gpu/:/x}_$(date -u +%Y%m%d_%H%M).log" 2>&1 &
done  # not waited on: `modal run` blocks while its function pends; poll the volume instead
until [[ $("${MODAL[@]}" volume ls wsbench-public-readouts /v4-s3don-rl600 2>/dev/null \
  | grep -c 'hallucination_story\.[01]of2\.jsonl' || true) -eq 2 ]]; do
  echo "=== $(date -Is) waiting for v4-s3don-rl600 story shards"; sleep 300
done
echo "=== $(date -Is) RL600 shards present; stop any still-pending duplicate fanout apps by hand"
exec scripts/run_story_v4.sh \
  v4-s3don-sft:wsbench-public-readouts,v4-jlens-std-n1000:wsbench-v4-readouts,v4-s3don-rl600:wsbench-public-readouts
