#!/bin/zsh
# Evaluate every simulated raw of a dataset as it appears, a few at a time, until the
# simulations are over and nothing is left. Threads follow the load: 3 while sim_stack runs
# (so both together stay at 9 of 12 cores), 9 once it is done.
#   dataset_eval_follow.sh DATASET_DIR OUT_DIR
set -u
cd /Volumes/SSD_1Tb/Programmazione/planetary_studio
DATA=$1; OUT=$2; CHUNK=${CHUNK:-6}
mkdir -p $OUT
n=0
while true; do
  done_list=$(find $DATA -mindepth 4 -maxdepth 4 -name grezza.tif | sed 's#/grezza.tif##' | sort -t/ -k5)
  todo=()
  for r in ${(f)done_list}; do
    name=${r:t}
    q=( $OUT/*/quad/$name.json(N) ); (( ${#q} )) || todo+=$r
  done
  running=$(pgrep -x sim_stack >/dev/null && echo 1 || echo 0)
  if (( ${#todo} == 0 )); then
    pgrep -f dataset_run.sh >/dev/null || break
    sleep 60; continue
  fi
  threads=$(( running ? 3 : 9 ))
  n=$((n + 1)); chunk=$(printf "c%03d" $n)
  echo "== $chunk ${#todo[@]} da fare, $threads thread $(date +%H:%M:%S)"
  RAYON_NUM_THREADS=$threads nice -n 12 ./target/release/sim_eval $OUT/$chunk ${todo[1,$CHUNK]} 2>&1 | grep -vE '^  (cielo|buio|bianchi)'
  sleep 15
done
echo "== FINE $(date +%H:%M:%S)"
