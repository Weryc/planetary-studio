#!/bin/zsh
# Run rows FROM..TO (1-based, inclusive) of a dataset plan with sim_stack, gently: low priority,
# a pause between captures. Rows already done (their folder exists with a grezza.tif) are
# skipped, so a run can be stopped and resumed.
#
#   crates/ps-train/scripts/dataset_run.sh Immagini_test/dataset_giove/piano.tsv 1 44
#
# Environment: RAYON_NUM_THREADS (6), PAUSE between captures in s (30), ROOT the folder holding
# Immagini_test (the repository by default), SIM_STACK the binary (./target/release/sim_stack),
# WORK_MIN / REST_MIN: after WORK_MIN minutes of simulating, rest REST_MIN minutes (off by default;
# 30 / 30 keeps the machine busy at most half of every hour).
set -u
PLAN=$1; FROM=$2; TO=$3
OUT=$(dirname "$PLAN")
cd "${ROOT:-$(dirname "$0")/../../..}"
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-6}
PAUSE=${PAUSE:-30}
SIM_STACK=${SIM_STACK:-./target/release/sim_stack}
WORK_MIN=${WORK_MIN:-0}; REST_MIN=${REST_MIN:-0}
busy=0
tail -n +2 "$PLAN" | sed -n "${FROM},${TO}p" | while IFS=$'\t' read id split group ref scope camera night r0 dr0 gain bits frames keep diam ground seed; do
  if ls -d "$OUT"/*/*/"${id}"_*/grezza.tif >/dev/null 2>&1; then echo "== $id già fatto"; continue; fi
  if (( WORK_MIN > 0 && busy >= WORK_MIN * 60 )); then
    echo "== riposo ${REST_MIN} min $(date +%H:%M:%S)"; sleep $(( REST_MIN * 60 )); busy=0
  fi
  echo "== $id $split $group $ref $scope $camera D/r0=$dr0 $gain ${bits}bit $frames×$keep $(date +%H:%M:%S)"
  t0=$SECONDS
  nice -n 10 $SIM_STACK --ref "Immagini_test/riferimenti/$ref" --scope $scope --camera $camera \
    --d-r0 $dr0 --gain $gain --bits $bits --frames $frames --keep $keep --diam $diam --ground $ground \
    --seed $seed --label $id --out "$OUT" 2>&1 | grep -E "stack:|fatto|panic|error"
  busy=$(( busy + SECONDS - t0 + PAUSE ))
  sleep $PAUSE
done
echo "== FINE $(date +%H:%M:%S)"
