#!/usr/bin/env bash
# Step 3 training runs (ResNet50, same settings as the r50_* runs unless stated):
#
#   random : naive random split vs. our grouped split (course rule 5b).
#            Baseline + class-weighted loss, 3 different random splits  -> rand_<method>_s<seed>
#   lc     : learning curve. Train on 5/10/25/50% of the training split (100% = the r50_* runs),
#            baseline + class-weighted loss, validation folds 0-2       -> lc<pct>_<method>_f<fold>
#            Smaller subsets get more epochs (and patience) so they are not undertrained.
#
#   ./run_step3.sh random      # 6 runs   (~2.5 h)
#   ./run_step3.sh lc          # DON'T: use ./run_lc.sh instead (it has locks, so several lanes can run)
#   ./run_step3.sh all         # DON'T: same reason; run `random` here and the learning curve with run_lc.sh
#
# The two parts can run at the same time in two tmux windows (each picks its own GPU).
# Resumable: finished runs (test_metrics.json exists) are skipped. A run that fails within
# 3 minutes stops the queue (setup error); a later failure is logged and the queue moves on.
set -u
cd "$(dirname "$0")"

PART=${1:-all}
export PYTHONUNBUFFERED=1
PY=venv/bin/python
RUNS=Deep_Learning/runs
LOGS=logs/step3
mkdir -p "$LOGS"

run() {   # run <name> <train.py args...>
  local name=$1; shift
  if [[ -f "$RUNS/$name/test_metrics.json" ]]; then echo "[skip] $name already finished"; return; fi
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
    if (( elapsed < 180 )); then
      echo "[stop] failed after ${elapsed}s - looks like a setup error. Fix it and rerun."
      tail -20 "$LOGS/$name.log"; exit 1
    fi
  fi
}

if [[ $PART == random || $PART == all ]]; then
  for seed in 0 1 2; do
    for method in none weights; do
      run "rand_${method}_s${seed}" --split random --split-seed "$seed" --imbalance "$method"
    done
  done
fi

if [[ $PART == lc || $PART == all ]]; then
  # fraction  epochs  patience
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
fi
echo "[$(date '+%F %T')] queue '$PART' finished"
