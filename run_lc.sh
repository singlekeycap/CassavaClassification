#!/usr/bin/env bash
# Learning-curve runs (same list as `run_step3.sh lc`), safe to start several times in parallel:
# each run is claimed with an atomic lock directory before it starts, so two lanes never train
# the same run. Start one lane per GPU you want to use, e.g. in separate tmux windows:
#
#   ./run_lc.sh            # lane 1
#   ./run_lc.sh            # lane 2 (another window, a minute later so it picks another GPU)
#
# Resumable: finished runs are skipped. After a crash, delete the stale lock of the run that
# was interrupted (Deep_Learning/runs/.locks/<name>) and start the lane again.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PY=venv/bin/python
RUNS=Deep_Learning/runs
LOCKS=$RUNS/.locks
LOGS=logs/step3
mkdir -p "$LOGS" "$LOCKS"

run() {
  local name=$1; shift
  [[ -f "$RUNS/$name/test_metrics.json" ]] && return
  mkdir "$LOCKS/$name" 2>/dev/null || return          # claimed by another lane
  local t0=$SECONDS
  echo "[$(date '+%F %T')] start $name  (log: $LOGS/$name.log)"
  $PY gpu_run.py --wait --min-free-gb 12 -- \
    $PY -m Deep_Learning.train --model resnet50 --seed 42 --run-name "$name" "$@" \
    > "$LOGS/$name.log" 2>&1
  local status=$? elapsed=$(( SECONDS - t0 ))
  if [[ $status -eq 0 ]]; then
    echo "[$(date '+%F %T')] done  $name  $(grep -o '"test_macro_f1": [0-9.]*' "$RUNS/$name/test_metrics.json")"
  else
    echo "[$(date '+%F %T')] FAIL  $name (exit $status) - see $LOGS/$name.log"
    rmdir "$LOCKS/$name"                               # let a later lane retry it
    if (( elapsed < 180 )); then echo "[stop] setup error"; tail -20 "$LOGS/$name.log"; exit 1; fi
  fi
}

# fraction  epochs  patience  (smaller subsets get more epochs so they are not undertrained)
for cfg in "0.05 60 15" "0.10 40 10" "0.25 30 8" "0.50 25 6"; do
  read -r frac epochs patience <<< "$cfg"
  pct=$(printf "%02d" "$(echo "$frac * 100 / 1" | bc)")
  for fold in 0 1 2; do
    for method in none weights; do
      run "lc${pct}_${method}_f${fold}" --train-frac "$frac" --val-fold "$fold" --imbalance "$method" \
          --epochs "$epochs" --patience "$patience"
    done
  done
done
echo "[$(date '+%F %T')] lane finished"
