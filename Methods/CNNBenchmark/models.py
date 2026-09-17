"""Model constructors for the RGB benchmark."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from Methods.BaselineCNN.model import BaselineCNN


def build_report_cnn(num_classes: int = 8, dropout: float = 0.35) -> nn.Module:
    """Build the scratch CNN used consistently across benchmark sizes."""
    return BaselineCNN(
        in_channels=3,
        num_classes=num_classes,
        dropout=dropout,
    )


def build_resnet18(
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build an RGB ResNet-18 transfer-learning benchmark."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def initialize_classifier(module: nn.Module) -> None:
    """Initialize newly attached linear layers without changing pretrained blocks."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=5**0.5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
