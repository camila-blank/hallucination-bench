#!/usr/bin/env bash
# V4 O-lens agentic_misalignment readouts at layers 21,27,33 (HallucinationBench AM port):
# Modal fanout (gw-v4-flash-ar wsbench_v4_modal.py, cell-level shards) -> fetch shards ->
# merge (cat in shard order, the existing flow) -> grid check vs the V4 bank -> symlink into
# outputs/v4/am_readouts/<hb arm>/.
#
#   scripts/am_v4_olens.sh sft      # Modal arm v4-s3don-sft-am3l   -> HB arm v4-s3don-sft
#   scripts/am_v4_olens.sh rl600    # Modal arm v4-s3don-rl600-am3l -> HB arm v4-s3don-rl600
#                                   # (waits until runs/s3don/rl/iter_000600 is on the volume)
# Env: N_SHARDS (20), BATCH (128), GPU (B300:8), RUN_HB=1 also runs scripts/run_am_v4.sh step1+2
# for that arm afterwards (blocking, in this shell).
set -uo pipefail
cd "$(dirname "$0")/.."
HB=$PWD
WHICH=${1:?sft|rl600}
case "$WHICH" in
  sft) MARM=v4-s3don-sft-am3l; HARM=v4-s3don-sft ;;
  rl600) MARM=v4-s3don-rl600-am3l; HARM=v4-s3don-rl600 ;;
  *) echo "usage: $0 sft|rl600" >&2; exit 2 ;;
esac
N_SHARDS=${N_SHARDS:-20}
BATCH=${BATCH:-64}  # 128+ cells (512+ sequences) OOM in the MoE prefill on B300:8 (smoke 2026-10-20: 256 OOM)
GPU=${GPU:-B300:8}
SHARDS=${SHARDS:-}         # shard order for the launch ("19,18,..."; default 0..N-1)
POLL_ONLY=${POLL_ONLY:-}   # 1 = launch nothing: other fanouts produce the shards, just wait for them
GW=${GW:-/workspace/gw-v4-flash-ar/scripts/v4_flash}
PUB=${HBENCH_WSBENCH_PUB_V4:-/workspace/root/wsbench-pub-v4-opus}
MODAL="uvx --python 3.12 modal@latest"
SH=$PUB/outputs/shards/$MARM
RD=$PUB/outputs/readouts/$MARM

if [[ "$WHICH" == rl600 ]]; then
  # the chain prints "RL done" after the run (rl_eval_after.sh waits on the same line)
  until grep -q "RL done" "$GW/../../logs/v4s3don_ao_chain.log" 2>/dev/null \
      && (cd "$GW" && $MODAL volume ls v4-flash runs/s3don/rl/iter_000600 2>/dev/null | grep -q .); do
    echo "$(date -Is) waiting for runs/s3don/rl/iter_000600"; sleep 600
  done
  sleep 300  # let the checkpoint finish writing
fi

missing() {
  local n=0
  for s in $(seq 0 $((N_SHARDS - 1))); do
    [[ -s $SH/agentic_misalignment.${s}of${N_SHARDS}.jsonl ]] || n=$((n + 1))
  done
  echo $n
}

fetch() {
  for f in $(cd "$GW" && $MODAL volume ls wsbench-public-readouts "$MARM" 2>/dev/null \
      | grep -o "agentic_misalignment\.[0-9]*of${N_SHARDS}\.jsonl" | sort -u); do
    [[ -s $SH/$f ]] || (cd "$GW" && $MODAL volume get --force wsbench-public-readouts "$MARM/$f" "$SH/$f")
  done
}

# app state from `modal app list --json` (running while any of its tasks are)
app_state() {
  (cd "$GW" && $MODAL app list --json 2>/dev/null) | python3 -c "
import json, sys
try:
    apps = json.load(sys.stdin)
except Exception:
    print('unknown'); sys.exit()
st = [str(a.get('State', a.get('state', ''))) for a in apps if a.get('App ID', a.get('app_id')) == sys.argv[1]]
print(st[0] if st else 'gone')" "$1"
}

mkdir -p "$SH" "$RD"
LOGDIR=$HB/logs
if [[ "$POLL_ONLY" == 1 ]]; then
  until fetch; [[ $(missing) -eq 0 ]]; do
    echo "$(date -Is) $(missing) shards missing (poll only)"; sleep 300
  done
fi
for attempt in 1 2 3; do
  fetch
  [[ $(missing) -eq 0 ]] && break
  FLOG=$LOGDIR/am_olens_fanout_${MARM}_$(date -u +%Y-%m-%d_%H%M).log
  echo "=== $(date -Is) fanout $MARM attempt $attempt ($N_SHARDS shards, batch $BATCH, $GPU) -> $FLOG"
  # --detach: a dropped client connection (this box loses DNS) does not stop the app
  (cd "$GW" && PYTHONUNBUFFERED=1 $MODAL run --detach wsbench_v4_modal.py::fanout --arm "$MARM" \
    --families agentic_misalignment --n-shards "$N_SHARDS" --layers 21,27,33 --cell-shard \
    --batch "$BATCH" --gpu "$GPU" ${SHARDS:+--shards "$SHARDS"} > "$FLOG" 2>&1) &
  until APP=$(grep -o "ap-[A-Za-z0-9]*" "$FLOG" | head -n 1) && [[ -n $APP ]]; do sleep 10; done
  echo "app $APP"
  # EXTRA_GPUS ("B300:4 B200:8"): untracked helper fanouts on other GPU types, first attempt only,
  # each walking the shards in a different order (descending / from the middle) so they rarely
  # start a shard another app is on; finished shards are skipped at container start
  if [[ $attempt -eq 1 && -n "${EXTRA_GPUS:-}" ]]; then
    k=0
    for g in $EXTRA_GPUS; do
      if [[ $k -eq 0 ]]; then ord=$(seq $((N_SHARDS - 1)) -1 0 | paste -sd,)
      else ord=$( (seq $((N_SHARDS / 2)) $((N_SHARDS - 1)); seq 0 $((N_SHARDS / 2 - 1))) | paste -sd,); fi
      XLOG=$LOGDIR/am_olens_fanout_${MARM}_$(echo "$g" | tr -d :)_$(date -u +%Y-%m-%d_%H%M).log
      (cd "$GW" && PYTHONUNBUFFERED=1 $MODAL run --detach wsbench_v4_modal.py::fanout --arm "$MARM" \
        --families agentic_misalignment --n-shards "$N_SHARDS" --layers 21,27,33 --cell-shard \
        --batch "$BATCH" --gpu "$g" --shards "$ord" > "$XLOG" 2>&1) &
      echo "helper fanout $g order $ord -> $XLOG"
      k=$((k + 1))
    done
  fi
  while :; do
    sleep 300
    fetch
    [[ $(missing) -eq 0 ]] && break
    st=$(app_state "$APP")
    echo "$(date -Is) $(missing) shards missing; app $APP $st"
    case "$st" in *stopped*|*gone*) break ;; esac
  done
done
[[ $(missing) -eq 0 ]] || { echo "shards still missing" >&2; exit 1; }

# the producer writes ensure_ascii=False JSON; HB's loader splits lines with str.splitlines(), which
# also breaks at raw U+0085 / U+2028 / U+2029 inside strings -> escape those (same JSON values)
for s in $(seq 0 $((N_SHARDS - 1))); do cat "$SH/agentic_misalignment.${s}of${N_SHARDS}.jsonl"; done \
  | python3 -c "
import sys
for line in sys.stdin.buffer:
    t = line.decode('utf-8')
    sys.stdout.buffer.write(t.replace('\u0085', '\\\\u0085').replace('\u2028', '\\\\u2028')
                            .replace('\u2029', '\\\\u2029').encode('utf-8'))" \
  > "$RD/agentic_misalignment.jsonl.tmp" && mv "$RD/agentic_misalignment.jsonl.tmp" "$RD/agentic_misalignment.jsonl"
wc -l "$RD/agentic_misalignment.jsonl"
uv run --no-sync python scripts/build_am_bank_v4.py --out outputs/v4/am_bank/items.check.json \
  --check-readouts "$RD/agentic_misalignment.jsonl" || exit 1
cmp -s outputs/v4/am_bank/items.check.json outputs/v4/am_bank/items.json || {
  echo "rebuilt bank differs from outputs/v4/am_bank/items.json" >&2; exit 1; }
mkdir -p "outputs/v4/am_readouts/$HARM"
ln -sfn "$RD/agentic_misalignment.jsonl" "outputs/v4/am_readouts/$HARM/agentic_misalignment.jsonl"
echo "=== $(date -Is) readouts ready: outputs/v4/am_readouts/$HARM"

if [[ "${RUN_HB:-}" == 1 ]]; then
  # one hbench writer per arm cache: the RL600 driver waits for the SFT driver's session to end
  if [[ "$WHICH" == rl600 ]]; then
    while tmux has-session -t hbv4am-olens-sft 2>/dev/null || tmux has-session -t hbv4am-hb-sft 2>/dev/null; do sleep 120; done
  fi
  HBENCH_IN_TMUX=1 PROSE_ARMS=$HARM bash scripts/run_am_v4.sh step1
  HBENCH_IN_TMUX=1 PROSE_ARMS=${ALL_PROSE:-$HARM} bash scripts/run_am_v4.sh step2
fi
