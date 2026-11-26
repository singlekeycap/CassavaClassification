# Workflow: deep-learning part of the Cassava project

Every step we took, in order: what it does, the command, what it produces, and what we decided.
All commands run from the project root with the project environment (`venv/`). To reproduce the
results from scratch, follow the steps in order; steps 5–8 can run in parallel once step 4 is done.

```bash
cd computer_vision_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
venv/bin/python -m ipykernel install --user --name cassava-venv --display-name "cassava (venv)"
```

Data: the Kaggle "Cassava Leaf Disease Classification" training set in `dataset/`
(`train.csv`, `train_images/`, `label_num_to_disease_map.json`). Do not redistribute it.

---

## Overview

| Step | What | Code | Output | GPU time |
|---|---|---|---|---|
| 0 | Find near-duplicate groups | `src/01_hashes.py`, `src/02_embeddings.py` | `features/` | minutes |
| 1 | Grouped train/val/test split | `src/03_group_and_split.py` | `splits/splits.csv` | none |
| 2 | Dataset and imbalance analysis | `src/04_eda.py`, `python -m Deep_Learning.analysis` | `reports/figures/imbalance/` | none |
| 3 | Training code | `Deep_Learning/` | — | — |
| 4 | Train 4 imbalance strategies on fold 0 | `Deep_Learning/train.py` | `Deep_Learning/runs/r50_*` | 1.5 h |
| 5 | Step 1 analysis: compare the 4 runs | `notebooks/step1_compare_runs.ipynb` | `reports/figures/comparison/` | none |
| 6 | Step 2: repeat on folds 1–4 | `run_step2.sh` | `runs/r50_<method>_f<k>` | 6.8 h |
| 7 | Step 2 analysis: 5-fold comparison | `notebooks/step2_cross_validation.ipynb` | `reports/figures/cross_validation/` | none |
| 8a | Error analysis | `notebooks/step3c_error_analysis.ipynb` | `reports/figures/error_analysis/` | minutes |
| 8b | Robustness study | `Deep_Learning/evaluation/robustness.py` + `notebooks/step3_robustness.ipynb` | `reports/robustness_eval/`, `reports/figures/robustness/` | 0.7 h |
| 8c | Random vs grouped split | `run_step3.sh random` | `runs/rand_*` | 3.8 h |
| 8d | Learning curve | `run_lc.sh` (several lanes) | `runs/lc<pct>_*` | 10.3 h |
| 8e | Analysis of 8c + 8d | `notebooks/step3_split_and_learning_curve.ipynb` | `reports/figures/split_and_learning_curve/` | none |
| 9 | Report draft, docs | `docs/` | this file, `report_deep_learning_draft.md` | — |

Total: about 23 GPU-hours on shared NVIDIA A100 80 GB GPUs. [TODO: the first four runs (step 4) were
launched from a WSL machine according to their log; confirm which GPU they used.]

---

## Step 0 – Near-duplicate detection

**Why.** The course requires splitting along the "unit of correlation". Cassava has no plant, field
or farmer IDs, so we estimate it: photos of the same plant or photo session look nearly identical,
and a model could memorise them if they appear in both training and test.

```bash
venv/bin/python src/01_hashes.py        # perceptual hashes (pHash, dHash)  -> features/hashes.csv
venv/bin/python src/02_embeddings.py    # DINOv2 ViT-S/14 embeddings       -> features/embeddings.npy
```

## Step 1 – Grouped split

```bash
venv/bin/python src/03_group_and_split.py     # -> splits/splits.csv, splits/split_stats.json
```

- Two images are linked if they are mutual 3-nearest neighbours with cosine similarity ≥ 0.80
  (mean-centred DINOv2 embeddings), or if their pHash distance is ≤ 24 bits. Connected components = groups.
- Result: 230 multi-image groups, 840 images (3.9%); 96 of these groups contain more than one label.
  The pHash check found no exact copies, so every link comes from visual similarity (same plant/session).
- `StratifiedGroupKFold`: 1/7 of the data is the **fixed test set** (3,078 images); the rest is split
  into **5 grouped, stratified folds** (≈3,670 each). No group crosses a split.
- `splits.csv` columns: `image_id, label, group_id, group_size, split (trainval/test), fold (0–4)`.
- Decision: declared at the interim checkpoint (rule R8). The test set is only used for final reporting.

## Step 2 – Dataset and imbalance analysis

```bash
venv/bin/python src/04_eda.py                 # interim-report figures -> reports/
venv/bin/python -m Deep_Learning.analysis     # imbalance figures/tables -> reports/figures/imbalance/
```

Key facts: 21,397 images; CMD 61.5%, CBB 5.1%; imbalance ratio 12.1 : 1; always predicting CMD gives
61.5% accuracy but 0.15 macro-F1, hence **macro-F1 is the main metric** (plus balanced accuracy).

## Step 3 – Training code (`Deep_Learning/`)

| Module | Role |
|---|---|
| `data/splits.py` | loads the grouped split; `--split random` builds a naive split of the same sizes; `--train-frac` takes a nested stratified subset |
| `data/imbalance.py` | class weights, weighted sampler, oversampling |
| `data/transforms.py` | augmentation (train) / resize + center crop (test) |
| `models/resnet.py` | ResNet with a dropout + linear head, backbone freezing, two learning-rate groups |
| `training/` | train/eval loops, warm-up + cosine schedule, seeding |
| `train.py` | command-line entry point; one folder per run in `Deep_Learning/runs/<name>/` |

Every run folder contains: `config.json` (all settings), `history.csv` (per epoch), `best.pt`
(best-validation checkpoint), `test_metrics.json`, `test_report.txt`, `test_probs.npy` (softmax on the
test set, used by all later analyses), `confusion_matrix.png`, `history.png` and, from step 8 on,
`split_ids.csv` (which image went where).

Settings (defaults): ResNet-50 (ImageNet V2 weights), 384 px, batch 32, AdamW (head 1e-3, backbone 1e-4,
weight decay 1e-4), 1 warm-up epoch + cosine, backbone frozen for epoch 1, label smoothing 0.1,
dropout 0.3, 20 epochs, early stopping patience 5 on validation macro-F1, mixed precision, seed 42.

## Step 4 – Four imbalance strategies on fold 0

Equivalent command (the original four runs were launched before the queue scripts existed):

```bash
for m in none sampler weights oversample; do
  venv/bin/python gpu_run.py -- venv/bin/python -m Deep_Learning.train --imbalance $m --run-name r50_$m
done
```

`gpu_run.py` picks the least busy GPU with enough free memory. Result: all four at 86.5–88% accuracy and
76–79% macro-F1, about 1 point apart, which raised the question whether the differences are real.

## Step 5 – Step 1 analysis: are the differences real? (single run)

`notebooks/step1_compare_runs.ipynb` (no GPU, ~30 s). It:
1. rebuilds the test labels and checks each run against its saved metrics;
2. writes the results table (`comparison_table.tex/.csv`) and a per-class F1 chart;
3. plots the 4 confusion matrices;
4. computes 95% bootstrap confidence intervals (10,000 resamples of the test set, by near-duplicate
   group, paired across methods) and McNemar tests (Holm-corrected).

Finding: only class-weighted loss improved balanced accuracy significantly (+2.3, CI [+0.7, +4.0]);
the bootstrap only measures test-set noise, not training noise, so we did step 2.

## Step 6 – Step 2: repeat on validation folds 1–4

```bash
tmux new -s step2
./run_step2.sh 2>&1 | tee logs/step2/queue.log      # 16 runs, resumable; logs in logs/step2/
```

Each strategy is retrained with each other fold as validation: 4 methods × 5 folds = 20 runs, all scored
on the same test set. How the queue scripts work (logs, tmux, resume, locks) is explained in
`docs/experiment_queue_guide.md`.

## Step 7 – Step 2 analysis: 5-fold comparison

`notebooks/step2_cross_validation.ipynb`: per-fold scores, mean ± std (`cv_summary.tex`), fold-by-fold
paired differences with paired t-tests, sign counts and Holm correction (`cv_paired_tests.csv`,
`cv_paired_diff.pdf`), per-class F1.

Findings: class-weighted loss improves balanced accuracy on 5/5 folds (+1.9, p = 0.002); its macro-F1
gain (+0.6, 4/5 folds) is not significant after correction; sampler and oversampling do not help;
the fold-0 "sampler lowers accuracy" result does not hold across folds.

## Step 8 – Remaining course components

### 8a – Error analysis (component 5)

`notebooks/step3c_error_analysis.ipynb` (CPU; uses a GPU for two small checks if one is free):
error consistency over all 20 models, the most common consistent confusions, image grids of failures
(with the center-crop region marked), images fixed by class weighting, reliability diagrams and ECE,
a test of scoring the full uncropped image, and Grad-CAM on failures.

### 8b – Robustness (component 4)

```bash
venv/bin/python gpu_run.py -- venv/bin/python -m Deep_Learning.evaluation.robustness   # ~40 min
```

Re-scores the 20 models on 19 versions of the test set (clean + 6 corruptions × 3 severities), after
checking that the clean predictions match each run's `test_probs.npy`. Writes
`reports/robustness_eval/metrics.csv`, `probs.npz` and a cache of the test images
(`test_uint8_384.npy`, 1.4 GB, git-ignored). Then `notebooks/step3_robustness.ipynb`: degradation
curves, robustness per method, which classes break and into what, and natural capture conditions
(brightness and sharpness thirds of the clean test photos).

### 8c – Random vs grouped split (rule 5b) and 8d – Learning curve (component 3)

New `train.py` options (defaults reproduce the earlier runs exactly):
`--split random --split-seed S` and `--train-frac F`.

```bash
tmux new -s step3
./run_step3.sh random 2>&1 | tee -a logs/step3/queue_random.log   # 6 runs: 2 methods x 3 random splits
# in other tmux windows, one per free GPU (they share the work through lock folders):
./run_lc.sh 2>&1 | tee -a logs/step3/queue_lc.log                 # 24 runs: 4 fractions x 2 methods x 3 folds
```

Do not use `run_step3.sh lc`: it has no locks. Learning-curve fractions get more epochs
(5%: 60, 10%: 40, 25%: 30, 50%: 25) and longer patience.

### 8e – Analysis of 8c and 8d

`notebooks/step3_split_and_learning_curve.ipynb`: random vs grouped headline numbers, a direct leakage
measurement (accuracy on test images whose near-duplicate was in training vs unseen groups, with class
mix), an undertraining check, the learning curve, and gain per doubling of the data.

## Step 9 – Documentation

- `docs/report_deep_learning_draft.md`: draft text of the deep-learning report sections, with all numbers.
- `docs/workflow.md`: this file.
- `docs/experiment_queue_guide.md`: how the experiment queues work (reusable for other projects).
- `Deep_Learning/README.md`: code structure, flags, scripts and notebooks.

---

## Where everything is

| Path | Contents |
|---|---|
| `splits/splits.csv` | the grouped split (all experiments use it unless `--split random`) |
| `Deep_Learning/runs/` | one folder per training run (50 runs: `r50_*`, `rand_*`, `lc*`) |
| `logs/step2/`, `logs/step3/` | one log per run + queue summaries |
| `reports/figures/imbalance/` | dataset imbalance figures and tables |
| `reports/figures/comparison/` | step 1 (single run) |
| `reports/figures/cross_validation/` | step 2 (5 folds), the main results |
| `reports/figures/error_analysis/` | error analysis |
| `reports/figures/robustness/` | robustness study |
| `reports/figures/split_and_learning_curve/` | random vs grouped split, learning curve |
| `reports/robustness_eval/` | raw robustness predictions (input of the robustness notebook) |

## Run naming

| Pattern | Meaning |
|---|---|
| `r50_<method>` | fold 0, grouped split, full data |
| `r50_<method>_f<k>` | fold k (1–4), grouped split, full data |
| `rand_<method>_s<seed>` | random split with seed 0–2, full data |
| `lc<pct>_<method>_f<k>` | learning curve: `pct`% of the training data, fold k (0–2) |

`<method>` is `none`, `sampler`, `weights` or `oversample`. Notebooks identify runs from `config.json`
(`imbalance`, `val_fold`, `split`, `train_frac`), not from the names.

## Rules we followed

- The test set is never used for choosing a model, epoch or setting. Note: unlike the interim plan
  ("touched once"), every run and analysis was scored on it; the report states this deviation (rule 5c).
- Every analysis re-checks predictions against the saved metrics before using them.
- Claims are made on 5-fold paired comparisons with multiple-comparison correction; single-run results
  are reported as such.
- Images and the image cache are not committed; clear image outputs from notebooks before any public push (rule R12).
