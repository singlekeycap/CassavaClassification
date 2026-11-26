"""ResNet classifiers (torchvision) with ImageNet-pretrained backbones."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

_RESNETS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
}
AVAILABLE = tuple(_RESNETS)


def build_resnet(name: str = "resnet50", num_classes: int = 5, pretrained: bool = True,
                 dropout: float = 0.3, freeze_backbone: bool = False) -> nn.Module:
    """ResNet with its final FC replaced by Dropout -> Linear(num_classes)."""
    if name not in _RESNETS:
        raise ValueError(f"{name} not in {AVAILABLE}")
    ctor, weights = _RESNETS[name]
    model = ctor(weights=weights if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    if freeze_backbone:
        set_backbone_trainable(model, False)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool):
    """Freeze / unfreeze everything except the classification head."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float = 0.1, weight_decay: float = 1e-4):
    """Lower LR for the pretrained backbone, full LR for the new head."""
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    return [
        {"params": body, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
        {"params": head, "lr": lr, "weight_decay": weight_decay},
    ]
