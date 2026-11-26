#!/usr/bin/env bash
# Step 2: retrain every imbalance strategy on validation folds 1-4 (fold 0 already done),
# giving 5 runs per method (full 5-fold cross-validation).
# Same settings as the r50_* runs; only --val-fold changes. Runs are named r50_<method>_f<fold>.
#
#   ./run_step2.sh                 # all 16 runs, one after another
#   ./run_step2.sh 1               # only fold 1 (use to split the work over several GPUs)
#   ./run_step2.sh "2 3"           # folds 2 and 3
#
# Resumable: a run whose test_metrics.json already exists is skipped, so after a crash or
# Ctrl-C just start the script again. A run that fails after training for a while is logged
# and the queue moves on; one that fails within 3 minutes (setup error) stops the queue.
# Never start two copies with the same folds: they would train the same run twice.
set -u
cd "$(dirname "$0")"

FOLDS=${1:-"1 2 3 4"}
export PYTHONUNBUFFERED=1   # so `tail -f` on a log shows epochs as they finish
METHODS="none sampler weights oversample"
PY=venv/bin/python
RUNS=Deep_Learning/runs
LOGS=logs/step2
mkdir -p "$LOGS"

for fold in $FOLDS; do
  for method in $METHODS; do
    name="r50_${method}_f${fold}"
    if [[ -f "$RUNS/$name/test_metrics.json" ]]; then
      echo "[skip] $name already finished"
      continue
    fi
    t0=$SECONDS
    echo "[$(date '+%F %T')] start $name  (log: $LOGS/$name.log)"
    $PY gpu_run.py --wait --min-free-gb 12 -- \
      $PY -m Deep_Learning.train --model resnet50 --imbalance "$method" \
          --val-fold "$fold" --seed 42 --run-name "$name" \
      > "$LOGS/$name.log" 2>&1
    status=$?
    elapsed=$(( SECONDS - t0 ))
    if [[ $status -eq 0 ]]; then
      echo "[$(date '+%F %T')] done  $name  $(grep -o '"test_macro_f1": [0-9.]*' "$RUNS/$name/test_metrics.json")"
    else
      echo "[$(date '+%F %T')] FAIL  $name (exit $status) - see $LOGS/$name.log"
      if (( elapsed < 180 )); then
        echo "[stop] failed after ${elapsed}s - looks like a setup error, not a training problem. Fix it and rerun."
        tail -20 "$LOGS/$name.log"
        exit 1
      fi
    fi
  done
done
echo "[$(date '+%F %T')] queue finished"
