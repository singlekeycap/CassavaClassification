import torch
import torch.nn as nn

from .focal import FocalLoss


def build_loss(name: str = "ce", weights: torch.Tensor | None = None,
               label_smoothing: float = 0.1, gamma: float = 2.0) -> nn.Module:
    """name: "ce" (cross-entropy, optionally class-weighted) or "focal"."""
    if name == "ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=weights, label_smoothing=label_smoothing)
    raise ValueError(f"unknown loss: {name}")
