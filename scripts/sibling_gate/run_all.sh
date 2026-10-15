#!/usr/bin/env bash
# Plan 0009 §5 step 3: the chained sibling-gate driver (stages = `hbench sibling-gate <stage>`):
#   units -> resample-manifests -> (GPU resample jobs) -> resample-collect -> fill ->
#   fill-calib (dev) -> classes -> concepts -> probe-texts -> capture-manifest -> (GPU capture)
#   -> fit -> score -> label -> report -> figures
#
# Run it from a /tmp snapshot (editing a running bash script breaks it), in tmux `sibgate`:
#   cp scripts/sibling_gate/run_all.sh /tmp/sibgate_run_all.sh
#   tmux new -s sibgate "bash /tmp/sibgate_run_all.sh"
#   tail -f logs/sibgate_<ts>.log
#
# Generators run from SNAPSHOT copies (never edited in place; commits recorded in the log):
#   SNAP    = a copy of /workspace/gw-mlnla at its current HEAD (wsbench_read / nla / patchscopes)
#   SNAP_WB = a copy of /workspace/gw-wrong-bridge at its current HEAD (wsbench_capture_modal.py)
# MERGED_S3D_RL600 / MERGED_S3D_SFT251 = the original runs' merged dir names (run_config.json).
# GPU: every resample generator runs on a single H200 per container (Camila 2026-10-15), fanned
# out as --shards 16 containers (resample.py job lines). wsbench_nla_modal.py reads
# WSBENCH_READ_GPU (default H200); wsbench_read_modal.py pins H200; the snapshot's
# modal_lensagree.py has GPU=H200 (original H100). B300 fails (torch cu128 has no sm_103).
# Re-running resumes: caches are append-only and the generators skip finished unit files.
set -euo pipefail
REPO=${REPO:-/workspace/hb-sibling-resample}
cd "$REPO"
TS=$(date +%Y%m%d_%H%M)
mkdir -p logs
LOG=logs/sibgate_$TS.log
exec > >(tee -a "$LOG") 2>&1
echo "[sibgate] start $TS (log $LOG)"

: "${SNAP:?set SNAP to a snapshot copy of /workspace/gw-mlnla}"
: "${SNAP_WB:?set SNAP_WB to a snapshot copy of /workspace/gw-wrong-bridge}"
: "${MERGED_S3D_RL600:?original merged dir of the s3d-rl600 readouts}"
: "${MERGED_S3D_SFT251:?original merged dir of the s3d-sft251 readouts}"
export MERGED_S3D_RL600 MERGED_S3D_SFT251
export WSBENCH_GEN_VOLUME=${WSBENCH_GEN_VOLUME:-sibgate-gen} WSBENCH_READ_GPU=${WSBENCH_READ_GPU:-H200}
echo "[sibgate] generator snapshot $(git -C "$SNAP" rev-parse HEAD 2>/dev/null || echo '?')"
echo "[sibgate] capture snapshot $(git -C "$SNAP_WB" rev-parse HEAD 2>/dev/null || echo '?')"

CELLS=${CELLS:-/workspace/hb-taxonomy-v2/outputs/wsbench_chat/claude-opus-5-5}
ACTS_LOCAL=${ACTS_LOCAL:-/workspace/gw-ar-probe-deception-v2/evals/workspace-bench/hillclimbing_evals/hallucination/chat}
READOUTS=${READOUTS:-/workspace/hb-readouts}
OUT=outputs/sibling_gate
MOD="uvx --python 3.12 modal@latest"
SG="uv run hbench sibling-gate"
POOLS="s3d-rl600 s3d-sft251 mlnla-rl600 mlnla-rl2-800 nla-rl nla-sft patchscopes"
declare -A ARM_OF=([s3d-rl600]=s3d-rl600-all11 [s3d-sft251]=s3d-sft251
  [mlnla-rl600]=mlnla-rl600-all11 [mlnla-rl2-800]=mlnla-rl2-800-all11 [nla-rl]=nla-rl-L42
  [nla-sft]=nla-sft-L42 [patchscopes]=patchscopes)

$SG units --cells-root "$CELLS"
$SG resample-manifests --dry
$SG resample-manifests --manifest-11 "$ACTS_LOCAL/acts-hallucination-chat/manifest.json" \
  --manifest-l42 "$ACTS_LOCAL/acts-hallucination-chat-L42/manifest.json"

FAILED=""
for p in $POOLS; do  # GPU resample jobs, then collect (keeps only the requested keys)
  if ! bash "$OUT/resample/$p/jobs.sh"; then
    echo "[sibgate] pool $p: generator failed -> resample_failed"; FAILED="$FAILED,$p"; continue
  fi
  G="$OUT/resample/$p/gen"; mkdir -p "$G"
  if [ "$p" = patchscopes ]; then RUNS=$(for s in $(seq 0 9); do echo "gen-sibgate-$p-s$s"; done)
  else RUNS="gen-sibgate-$p"; fi
  for r in $RUNS; do $MOD volume get --force "$WSBENCH_GEN_VOLUME" "/$r" "$G/" ; done
  if ! $SG resample-collect --pool "$p" --src "$G" \
      --orig-readouts "$READOUTS/${ARM_OF[$p]}/hallucination.jsonl"; then
    echo "[sibgate] pool $p: collect / block-shape check failed -> resample_failed"
    FAILED="$FAILED,$p"
  fi
done

$SG fill --failed-pools "${FAILED#,}"
$SG fill-calib --split dev
$SG classes
$SG concepts
$SG probe-texts
uv run --with transformers --with jinja2 $SG capture-manifest
(cd "$SNAP_WB" && PYTHONPATH=src $MOD run scripts/olens_suite/workspace_bench/wsbench_capture_modal.py \
  --prompts-json "$REPO/$OUT/probes/capture_prompts.json" --acts-subdir acts-sibgate-probe \
  --layers 20,24,28,32,36,40,42,44,48,52,56,60)
$SG fit
$MOD run scripts/sibling_gate/probes_modal.py --phase pack --inputs $OUT/probes/probe_inputs.json
$MOD run scripts/sibling_gate/probes_modal.py --phase fit --inputs $OUT/probes/probe_inputs.json \
  --out-dir $OUT/probes
$SG score --launch
$SG label
$SG report
# report v2.8: rebuild + re-score every arm under rev 21 ($0 judge side; cache only)
uv run hbench report --out "${HB_OUT:-outputs}" --view v2 --sibling-labels "$REPO/$OUT/labels"
$SG figures
echo "[sibgate] done $(date +%Y%m%d_%H%M)"
