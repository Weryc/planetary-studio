#!/bin/zsh
# Run rows FROM..TO (1-based, inclusive) of a dataset plan with sim_stack, gently: half the
# cores, low priority, a pause between captures. Rows already done (their folder exists with a
# grezza.tif) are skipped, so a run can be stopped and resumed.
#
#   crates/ps-train/scripts/dataset_run.sh Immagini_test/dataset_giove/piano.tsv 1 44
set -u
PLAN=$1; FROM=$2; TO=$3
OUT=$(dirname "$PLAN")
cd "$(dirname "$0")/../../.."
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-6}
PAUSE=${PAUSE:-30}
tail -n +2 "$PLAN" | sed -n "${FROM},${TO}p" | while IFS=$'\t' read id split group ref scope camera night r0 dr0 gain bits frames keep diam ground seed; do
  if ls -d "$OUT"/*/*/"${id}"_*/grezza.tif >/dev/null 2>&1; then echo "== $id già fatto"; continue; fi
  echo "== $id $split $group $ref $scope $camera D/r0=$dr0 $gain ${bits}bit $frames×$keep $(date +%H:%M:%S)"
  nice -n 10 ./target/release/sim_stack --ref "Immagini_test/riferimenti/$ref" --scope $scope --camera $camera \
    --d-r0 $dr0 --gain $gain --bits $bits --frames $frames --keep $keep --diam $diam --ground $ground \
    --seed $seed --label $id --out "$OUT" 2>&1 | grep -E "stack:|fatto|panic|error"
  sleep $PAUSE
done
echo "== FINE $(date +%H:%M:%S)"
