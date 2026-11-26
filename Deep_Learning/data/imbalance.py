"""Class-imbalance handling (CMD ~61% vs CBB ~5%). Applied to the training split only.

    class_weights  inverse-frequency weights for a weighted loss
    make_sampler   WeightedRandomSampler -> roughly class-balanced mini-batches
    oversample     duplicate minority-class rows up to a target count (static)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from ..config import NUM_CLASSES


def class_counts(labels) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=NUM_CLASSES)


def class_weights(labels, power: float = 1.0) -> torch.Tensor:
    """Inverse-frequency weights, normalised so the mean weight is 1.

    power=1 -> full inverse frequency, power=0.5 -> "sqrt" (softer) re-weighting.
    """
    counts = class_counts(labels).astype(np.float64)
    w = (counts.sum() / (len(counts) * np.maximum(counts, 1))) ** power
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def make_sampler(labels, power: float = 1.0, num_samples: int | None = None, seed: int = 42):
    """WeightedRandomSampler whose per-sample weight = class weight of its label."""
    w_cls = class_weights(labels, power).numpy()
    sample_w = w_cls[np.asarray(labels)]
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                 num_samples=num_samples or len(labels), replacement=True, generator=g)


def oversample(df: pd.DataFrame, target: int | str = "max", seed: int = 42) -> pd.DataFrame:
    """Duplicate minority-class rows (random with replacement) up to ``target`` per class.

    target="max" -> every class matches the majority class; "median" -> the median count;
    an int -> that count. Classes already above the target are left untouched.
    """
    counts = df.label.value_counts()
    tgt = {"max": counts.max(), "median": int(counts.median())}.get(target, target)
    rng = np.random.default_rng(seed)
    parts = [df]
    for label, n in counts.items():
        if n < tgt:
            idx = rng.choice(df.index[df.label == label], size=int(tgt - n), replace=True)
            parts.append(df.loc[idx])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
