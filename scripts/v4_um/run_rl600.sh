#!/usr/bin/env bash
# Add the V4 O-lens RL600 arm to the UM ladder (3-arm run). Tmux-launchable:
#   tmux new -d -s umv4rl600 "scripts/v4_um/run_rl600.sh 2>&1 | tee logs/umv4rl600_$(date -u +%Y-%m-%d_%H%M).log"
# 1. waits for runs/s3don/rl/iter_000600 on the v4-flash volume (adapter_config.json present);
# 2. snapshots the 2-arm results (label dirs -> label_2arm, um3_v4.json -> um3_v4_2arm.json);
# 3. O-lens RL600 readouts of the same eval acts; fetch;
# 4. the full downstream with 3 arms (cross-method pool and concept set / distractors include
#    RL600's commits, as the method requires) under probe run suffix -3arm -> um3_v4.json.
set -uo pipefail
HB=$(cd "$(dirname "$0")/../.." && pwd); cd "$HB"
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
MODAL="uvx --python 3.12 modal@latest"
O=outputs/v4/um
echo "=== waiting for RL600 checkpoint $(date -u)"
for _ in $(seq 1 1440); do
  $MODAL volume ls v4-flash /runs/s3don/rl/iter_000600 2>/dev/null | grep -q adapter_config && break
  sleep 60
done
$MODAL volume ls v4-flash /runs/s3don/rl/iter_000600 | grep -q adapter_config || { echo "no checkpoint"; exit 1; }
if [ ! -f $O/um3_v4_2arm.json ] && [ -f $O/um3_v4.json ]; then
  cp $O/um3_v4.json $O/um3_v4_2arm.json
  for d in "" capital_city/ holiday/; do
    [ -d $O/${d}label ] && cp -r $O/${d}label $O/${d}label_2arm
    [ -d $O/${d}data ] && cp -r $O/${d}data $O/${d}data_2arm
    [ -d $O/${d}probes ] && cp -r $O/${d}probes $O/${d}probes_2arm
  done
fi
scripts/v4_um/run_v4_um.sh olens v4-s3don-rl600 4 || exit 1
export V4_ARMS=v4-s3don-sft,jlens,v4-s3don-rl600 V4_RUN_SUFFIX=-3arm PROBE_TAG=r
scripts/v4_um/run_v4_um.sh all
