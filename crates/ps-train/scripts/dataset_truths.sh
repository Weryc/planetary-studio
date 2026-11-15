#!/bin/zsh
# Write each finished run's own truths (verita_campionata.tif, limite_diffrazione.tif) into its
# folder, for runs simulated before sim_stack kept them there: the shared copy one level up was
# overwritten whenever the same telescope and camera came back with another disk size.
#   dataset_truths.sh PLAN
set -u
PLAN=$1; OUT=$(dirname "$PLAN")
cd "$(dirname "$0")/../../.."
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-3}
tail -n +2 "$PLAN" | while IFS=$'\t' read id split group ref scope camera night r0 dr0 gain bits frames keep diam ground seed; do
  run=( $OUT/*/*/${id}_*(N/) )
  (( ${#run} )) || continue
  [[ -f $run[1]/grezza.tif ]] || continue
  [[ -f $run[1]/verita_campionata.tif ]] && continue
  nice -n 12 ./target/release/sim_stack --ref "Immagini_test/riferimenti/$ref" --scope $scope --camera $camera \
    --d-r0 $dr0 --gain $gain --bits $bits --frames $frames --keep $keep --diam $diam --ground $ground \
    --seed $seed --label $id --out "$OUT" --truth-only 1 2>&1 | grep -E "verità scritte|panic|error" | sed "s#^#$id #"
done
