import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017): down-weights easy, well-classified
    examples so training focuses on hard / minority-class ones.

    alpha: optional per-class weight tensor (e.g. ``data.class_weights``).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.softmax(logits, 1).gather(1, target[:, None]).squeeze(1)
        return ((1 - pt) ** self.gamma * ce).mean()
