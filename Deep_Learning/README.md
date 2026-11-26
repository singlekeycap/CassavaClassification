# Deep Learning – ResNet on Cassava (imbalanced, 5 classes)

```
Deep_Learning/
├── config.py            paths, class names, constants
├── train.py             CLI entry point (wires everything together)
├── data/                splitting, imbalance handling, loading
│   ├── splits.py        train/val/test split (grouped, or stratified fallback)
│   ├── imbalance.py     class weights, WeightedRandomSampler, oversampling
│   ├── transforms.py    augmentation / preprocessing
│   ├── dataset.py       CassavaDataset
│   └── loaders.py       build_dataloaders() -> DataBundle
├── losses/              cross-entropy / focal loss (factory.py, focal.py)
├── models/              resnet.py: ResNet18-152, new head, freezing, param groups
├── training/            engine.py (train/eval loops), scheduler.py, utils.py (seeding)
├── evaluation/          metrics.py (macro-F1, report), plots.py (confusion matrix, curves),
│                        robustness.py (re-score models on corrupted test images)
├── analysis/            imbalance_report.py: paper figures/tables on class imbalance
└── runs/                outputs, one folder per run
```
Each folder is a Python package; import from it directly, e.g.
`from Deep_Learning.data import build_dataloaders`, `from Deep_Learning.models import build_resnet`.

## Splits
Reuses the grouped split in `splits/splits.csv` (near-duplicate groups never cross splits):
`test` = held-out split (3,078), `val` = CV fold `--val-fold` (~3,670), `train` = other 4 folds (~14,640).
Falls back to a plain stratified 70/15/15 split if that file is missing.

## Imbalance strategies (`--imbalance`)
- `sampler` (default): `WeightedRandomSampler`, class-balanced-ish batches
- `weights`: class-weighted loss
- `oversample`: duplicate minority rows up to `--oversample-target` (`median`/`max`/int)
- `none`
`--weight-power` 1.0 = full inverse frequency, 0.5 = sqrt (softer). Combine with `--loss focal` if desired.
Val/test are never resampled; model selection uses **macro-F1**.

## Run (from the project root)
```bash
source venv/bin/activate        # project venv (see requirements.txt)
python -m Deep_Learning.data                                   # print split / class stats
python -m Deep_Learning.train --model resnet50 --imbalance sampler
python -m Deep_Learning.train --model resnet50 --imbalance weights --loss focal
python -m Deep_Learning.train --help   # `python Deep_Learning/train.py` also works
# train.py picks the best free GPU itself (see gpu_run.py in the project root).
# To choose one yourself:  CUDA_VISIBLE_DEVICES=1 python -m Deep_Learning.train ...
# For any other script:    python gpu_run.py [--min-free-gb 12] [--wait] -- python -m <module> ...
```
Outputs: `Deep_Learning/runs/<run>/` → `best.pt`, `history.csv`, `config.json`, `test_report.txt`,
`test_metrics.json`, `test_probs.npy`, `history.png`, `confusion_matrix.png`.

On a 6 GB GPU, resnet50 @ 384px fits batch 32 with AMP; lower `--batch-size` or `--img-size` if you hit OOM.

## Imbalance figures for the paper
```bash
python -m Deep_Learning.analysis      # -> reports/figures/imbalance/
```
Writes `fig1_class_distribution`, `fig2_split_distribution`, `fig3_resampling_effect`,
`fig4_class_weights` (each as 300-dpi PNG + vector PDF), `imbalance_table.tex` (booktabs),
`imbalance_table.csv` and `imbalance_stats.json`.

## Splits for the extra studies (`train.py` flags, defaults reproduce the main runs)
- `--split random --split-seed S`: naive stratified random split with the same sizes as the grouped
  split (course rule 5b, leakage comparison).
- `--train-frac F`: train on a nested stratified fraction of the training split (learning curve);
  validation and test are unchanged.
Every run also writes `split_ids.csv` (which image went to train / val / test).

## Experiment queues (project root, run inside tmux)
| script | runs | used by |
|---|---|---|
| `run_step2.sh` | 4 methods x validation folds 1-4 (`r50_<method>_f<k>`) | step 2 notebook |
| `run_step3.sh random` | baseline + class weights on 3 random splits (`rand_*`) | split / learning-curve notebook |
| `run_lc.sh` (start one per GPU) | learning curve, 5/10/25/50% x folds 0-2 (`lc<pct>_*`) | split / learning-curve notebook |
| `gpu_run.py -- venv/bin/python -m Deep_Learning.evaluation.robustness` | 20 models x 19 corrupted test sets | robustness notebook |

Use `run_lc.sh` for the learning curve, not `run_step3.sh lc`: only `run_lc.sh` uses locks, so
several lanes can run in parallel without training the same run twice.

## Analysis notebooks (`notebooks/`, kernel "cassava (venv)")
| notebook | content | figures/tables in |
|---|---|---|
| `step1_compare_runs` | 4 methods on fold 0: table, confusion matrices, bootstrap CIs, McNemar | `reports/figures/comparison/` |
| `step2_cross_validation` | 4 methods x 5 folds: mean +- std, paired tests | `reports/figures/cross_validation/` |
| `step3c_error_analysis` | error consistency over 20 models, failure images, calibration, crop test, Grad-CAM | `reports/figures/error_analysis/` |
| `step3_robustness` | corruptions and natural capture conditions | `reports/figures/robustness/` |
| `step3_split_and_learning_curve` | random vs grouped split, leakage, learning curve | `reports/figures/split_and_learning_curve/` |

The test set is never used to choose a model, epoch or setting: best epochs come from the validation
fold, and all test-set analyses are reported as-is.

`reports/robustness_eval/test_uint8_384.npy` is a cache of the test images: do not commit it (dataset
licence, 1.4 GB). Notebooks with image outputs should be cleared before pushing to a public repository.
