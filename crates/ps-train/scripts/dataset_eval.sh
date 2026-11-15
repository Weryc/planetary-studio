#!/bin/zsh
# Evaluate runs listed in RUNS (one folder per line) in chunks of 11, half load, with pauses.
# usage: dataset_eval.sh RUNS OUT [extra sim_eval args…]
cd /Volumes/SSD_1Tb/Programmazione/planetary_studio
RUNS=$1; OUT=$2; shift 2
export RAYON_NUM_THREADS=6
split -l 11 $RUNS $OUT.chunk.
for c in $OUT.chunk.*; do
  echo "== $c $(date +%H:%M:%S)"
  nice -n 10 ./target/release/sim_eval $OUT/${c##*.} $(cat $c) "$@" 2>&1 | grep -vE '^  (cielo|buio)'
  rm $c
  sleep 30
done
echo "== FINE $(date +%H:%M:%S)"
