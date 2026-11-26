"""Train / validation / test splitting.

By default we reuse ``splits/splits.csv`` produced by ``src/03_group_and_split.py``:
near-duplicate / same-session images share a ``group_id`` and never cross splits, so
validation and test scores are not inflated by leakage.

    test  = rows with split == "test"            (~14%, fixed hold-out)
    val   = trainval rows with fold == val_fold  (~17%)
    train = remaining trainval rows              (~69%)

If ``splits.csv`` is missing, ``make_stratified_split`` builds a plain stratified
train/val/test split from ``dataset/train.csv`` instead (no grouping).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import NUM_CLASSES, SPLITS_CSV, TRAIN_CSV, short_class_names


def load_grouped_split(val_fold: int = 0, splits_csv: Path = SPLITS_CSV):
    """train / val / test DataFrames from the grouped split file."""
    df = pd.read_csv(splits_csv)
    test = df[df.split == "test"]
    tv = df[df.split == "trainval"]
    val = tv[tv.fold == val_fold]
    train = tv[tv.fold != val_fold]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_stratified_split(val_size=0.15, test_size=0.15, seed=42, csv: Path = TRAIN_CSV):
    """Fallback: stratified (not grouped) train / val / test split."""
    df = pd.read_csv(csv)
    trainval, test = train_test_split(df, test_size=test_size, stratify=df.label, random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(trainval, test_size=rel_val, stratify=trainval.label, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_random_split(seed: int = 0, val_fold: int = 0, splits_csv: Path = SPLITS_CSV):
    """Naive (ungrouped) stratified random split with the same sizes as the grouped split.

    Test size = grouped test size, val size = grouped fold ``val_fold`` size, train = the rest,
    so a random-vs-grouped comparison is not confounded by training-set size. Near-duplicate
    groups are ignored on purpose: this is the leaky baseline that rule 5b asks for.
    """
    df = pd.read_csv(splits_csv)
    n_test = int((df.split == "test").sum())
    n_val = int(((df.split == "trainval") & (df.fold == val_fold)).sum())
    trainval, test = train_test_split(df, test_size=n_test, stratify=df.label, random_state=seed)
    train, val = train_test_split(trainval, test_size=n_val, stratify=trainval.label, random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def subsample_train(train: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    """Stratified subset of the training rows, nested across fractions.

    Each row gets one fixed random key (depends only on ``seed``); within each class the rows
    with the smallest keys are kept, so the 5% subset is inside the 10% subset, and so on.
    """
    if frac >= 1.0:
        return train
    key = np.random.default_rng(seed).random(len(train))
    keep = []
    for _, idx in train.groupby("label").indices.items():
        n = max(1, int(round(frac * len(idx))))
        keep.extend(idx[np.argsort(key[idx])[:n]])
    return train.iloc[sorted(keep)].reset_index(drop=True)


def get_splits(val_fold=0, seed=42, split="grouped", split_seed=0):
    """split="grouped" (default, near-duplicates never cross splits) or "random" (naive)."""
    if split == "random":
        return make_random_split(split_seed, val_fold)
    if split != "grouped":
        raise ValueError(f"unknown split: {split}")
    if SPLITS_CSV.exists():
        return load_grouped_split(val_fold)
    print(f"[data] {SPLITS_CSV} not found -> using plain stratified split")
    return make_stratified_split(seed=seed)


def describe(train, val, test) -> pd.DataFrame:
    """Class counts per split (rows)."""
    out = pd.DataFrame({
        name: d.label.value_counts().reindex(range(NUM_CLASSES), fill_value=0).values
        for name, d in (("train", train), ("val", val), ("test", test))
    }, index=short_class_names()).T
    out["total"] = out.sum(1)
    return out
