#!/usr/bin/env bash
# V4 UM ladder driver (plans 0006 + 0011 ported to DeepSeek-V4-Flash; outputs/v4/um/PORT_PLAN.md).
# Stages (each resumable; run in a named tmux session, log to logs/):
#   olens <arm>   O-lens readouts of the saved eval acts (4 x B300:8), waits for the outputs
#   fetch         um-v4 volume -> outputs/v4/um/modal_out (responses, readouts, pack metas) + src
#   judge         per domain: J-lens summaries -> mentions -> commitment judge (Opus 5.5, ant profile)
#   texts         per domain: concept set -> probe texts; capture the new ones (B300:8)
#   probes        per domain: probe inputs -> fit (GPU eig) -> distractors -> score
#   label         per domain: label; then um3_v4.json
#   all           fetch judge texts probes label   (after `olens v4-s3don-sft` and the items job)
# env: V4_ARMS (default v4-s3don-sft,jlens), DOMAINS, UM3_OUT
set -uo pipefail
HB=$(cd "$(dirname "$0")/../.." && pwd); cd "$HB"
V4DIR=${V4DIR:-/workspace/gw-v4-flash-ar/scripts/v4_flash}
export MODAL_PROFILE=${MODAL_PROFILE:-safety-sahan-2}
MODAL="uvx --python 3.12 modal@latest"
PY="env -u ANTHROPIC_API_KEY HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 .venv/bin/python"
export V4_ARMS=${V4_ARMS:-v4-s3don-sft,jlens}
S="$PY scripts/v4_um/um_ladder_v4.py"
O=outputs/v4/um
MO=$O/modal_out
DOMS=${DOMAINS:-country capital_city holiday}
retry() { for i in 1 2 3 4 5 6; do "$@" && return 0; echo "[retry $i] $*"; sleep 30; done; return 1; }
vcount() { $MODAL volume ls um-v4 "$1" 2>/dev/null | grep -c "${2:-.}" || true; }
wait_n() { for _ in $(seq 1 1440); do [ "$(vcount "$1" "$3")" -ge "$2" ] && return 0; sleep 30; done; return 1; }
odir() { if [ "$1" = country ]; then echo $O; else echo $O/$1; fi; }

st_olens() {
  local arm=$1 n=${2:-4}
  echo "=== olens $arm $(date -u)"
  (cd "$V4DIR" && $MODAL run --detach um_v4_modal.py::run_olens --arm "$arm" --n-shards "$n") \
    || echo "[client exited; waiting for volume outputs]"
  wait_n /readouts/$arm "$n" "sibling\." || { echo "olens timeout"; return 1; }
  echo "OLENSDONE $arm $(date -u)"
}

st_fetch() {
  echo "=== fetch $(date -u)"
  mkdir -p $MO/readouts $MO/pack
  retry $MODAL volume get --force um-v4 /responses.jsonl $MO/responses.jsonl || return 1
  retry $MODAL volume get --force um-v4 /eval_acts/manifest.json $MO/eval_manifest.json || return 1
  for arm in v4-jlens-std-n1000 v4-s3don-sft v4-s3don-rl600; do
    for f in $($MODAL volume ls um-v4 /readouts/$arm 2>/dev/null | grep jsonl); do
      mkdir -p $MO/readouts/$arm
      retry $MODAL volume get --force um-v4 /readouts/$arm/$(basename $f) $MO/readouts/$arm/ || return 1
    done
  done
  for d in country capital_city holiday; do
    mkdir -p $MO/pack/$d
    for f in $($MODAL volume ls um-v4 /pack/$d 2>/dev/null | grep "\.json"); do
      retry $MODAL volume get --force um-v4 /pack/$d/$(basename $f) $MO/pack/$d/ || return 1
    done
  done
  $PY scripts/v4_um/build_inputs.py src || return 1
}

st_judge() {
  for D in $DOMS; do
    echo "=== judge $D $(date -u)"
    $S --category $D summarize || return 1
    $S --category $D mentions || return 1
    for k in 1 2 3; do $S --category $D judge && break; echo "[judge retry $k]"; sleep 60; done
  done
}

st_texts() {
  for D in $DOMS; do
    echo "=== concepts + probe texts $D $(date -u)"
    $S --category $D concepts || return 1
    $S --category $D probe-texts || return 1
  done
  st_fetch_packs
  $PY scripts/v4_um/build_inputs.py probe-job --only-new --tag ${PROBE_TAG:-n} --per-shard 800 || return 1
  local n
  n=$(python3 -c "import json;j=json.load(open('$O/modal_in/probe_${PROBE_TAG:-n}.json'));print(sum(len(v) for v in j.values()))")
  if [ "$n" -gt 0 ]; then
    echo "=== capture $n new shards $(date -u)"
    (cd "$V4DIR" && $MODAL run --detach um_v4_modal.py::run_probe --texts-json "$HB/$O/modal_in/probe_${PROBE_TAG:-n}.json" --tag ${PROBE_TAG:-n}) \
      || echo "[client exited; waiting for volume outputs]"
    for D in $DOMS; do
      local k
      k=$(python3 -c "import json;j=json.load(open('$O/modal_in/probe_${PROBE_TAG:-n}.json'));print(len(j.get('$D',[])))")
      [ "$k" -gt 0 ] && { wait_n /pack/$D $((k)) "${PROBE_TAG:-n}[0-9]*\.json" || { echo "capture timeout $D"; return 1; }; }
    done
  fi
  echo "TEXTSDONE $(date -u)"
}

st_fetch_packs() {
  for d in $DOMS; do
    mkdir -p $MO/pack/$d
    for f in $($MODAL volume ls um-v4 /pack/$d 2>/dev/null | grep "\.json"); do
      retry $MODAL volume get --force um-v4 /pack/$d/$(basename $f) $MO/pack/$d/ || return 1
    done
  done
}

st_probes() {
  for D in $DOMS; do
    local OD RUN; OD=$(odir $D); RUN=v4-um-${D//_/-}${V4_RUN_SUFFIX:-}
    echo "=== probes $D $(date -u)"
    $S --category $D probe-inputs || return 1
    if [ "$(vcount /probes/$RUN/fit fit_L)" -lt 11 ]; then
      $MODAL run --detach scripts/v4_um/um_probes_v4_modal.py --phase fit --inputs $OD/probe_inputs.json \
        --out-dir $OD/probes --domain $D || echo "[client exited; waiting for volume outputs]"
      wait_n /probes/$RUN/fit 11 fit_L || { echo "fit timeout $D"; return 1; }
    fi
    mkdir -p $OD/probes/fit
    for f in $($MODAL volume ls um-v4 /probes/$RUN/fit | grep -E "fit_L|centroids_L"); do
      retry $MODAL volume get --force um-v4 /probes/$RUN/fit/$(basename $f) $OD/probes/fit/ || return 1
    done
    rm -rf $OD/probes/score
    $S --category $D distractors || return 1
    retry $MODAL run scripts/v4_um/um_probes_v4_modal.py --phase score --inputs $OD/probe_inputs.json \
      --out-dir $OD/probes --domain $D --distractors $OD/data/um_${D}_distractors.json || return 1
    echo "PROBESDONE $D $(date -u)"
  done
}

st_label() {
  for D in $DOMS; do
    echo "=== label $D $(date -u)"
    $S --category $D label || return 1
  done
  $PY scripts/v4_um/um3_v4.py --out ${UM3_OUT:-$O/um3_v4.json}
}

case "${1:-all}" in
  olens) st_olens "${2:-v4-s3don-sft}" "${3:-4}" ;;
  fetch) st_fetch ;;
  judge) st_judge ;;
  texts) st_texts ;;
  probes) st_probes ;;
  label) st_label ;;
  all) st_fetch && st_judge && st_texts && st_probes && st_label && echo "ALLDONE $(date -u)" ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
