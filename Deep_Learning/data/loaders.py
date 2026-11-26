"""Build train / val / test DataLoaders with the chosen imbalance strategy.

imbalance: "none" | "weights" (weighted loss) | "sampler" (WeightedRandomSampler)
           | "oversample" (duplicate minority rows).
Validation / test are never resampled, so their metrics reflect the true distribution.
"""
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .dataset import CassavaDataset
from .imbalance import class_counts, class_weights, make_sampler, oversample
from .splits import describe, get_splits, subsample_train
from .transforms import build_transforms


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    loss_weights: torch.Tensor | None   # pass to the loss when imbalance == "weights"
    frames: tuple                        # (train_df, val_df, test_df) before resampling


def build_dataloaders(img_size=384, batch_size=32, num_workers=8, val_fold=0,
                      imbalance="sampler", weight_power=0.5, oversample_target="median",
                      seed=42, split="grouped", split_seed=0, train_frac=1.0) -> DataBundle:
    train_df, val_df, test_df = get_splits(val_fold, seed, split, split_seed)
    if train_frac < 1.0:   # learning curve: shrink only the training set (val/test unchanged)
        train_df = subsample_train(train_df, train_frac, seed=split_seed)
        print(f"[data] training on {train_frac:.0%} of the train split: {len(train_df)} images")
    print(describe(train_df, val_df, test_df).to_string(), "\n")

    loss_weights, sampler, fit_df = None, None, train_df
    if imbalance == "weights":
        loss_weights = class_weights(train_df.label, weight_power)
    elif imbalance == "sampler":
        sampler = make_sampler(train_df.label, weight_power, seed=seed)
    elif imbalance == "oversample":
        fit_df = oversample(train_df, oversample_target, seed)
        print("[data] after oversampling:", class_counts(fit_df.label).tolist())
    elif imbalance != "none":
        raise ValueError(f"unknown imbalance strategy: {imbalance}")

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_dl = DataLoader(CassavaDataset(fit_df, build_transforms(img_size, True)),
                          sampler=sampler, shuffle=sampler is None, drop_last=True, **common)
    val_dl = DataLoader(CassavaDataset(val_df, build_transforms(img_size, False)), shuffle=False, **common)
    test_dl = DataLoader(CassavaDataset(test_df, build_transforms(img_size, False)), shuffle=False, **common)
    return DataBundle(train_dl, val_dl, test_dl, loss_weights, (train_df, val_df, test_df))
