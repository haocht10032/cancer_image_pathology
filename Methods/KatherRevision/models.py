"""Matched frozen-ResNet model and deterministic feature cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from Methods.UNIAttribution.data import UNIImageDataset


class FrozenResNet18Classifier(nn.Module):
    """Frozen ImageNet ResNet18 with the same lightweight head shape as the ViTs."""

    def __init__(self, num_classes: int, dropout: float = 0.20) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = backbone
        self.feature_dim = int(backbone.fc.in_features)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )
        self.freeze_encoder()

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenResNet18Classifier":
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_cls(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_from_features(self.encode_cls(images))


def build_frozen_resnet18_classifier(
    num_classes: int,
    device: torch.device,
    dropout: float = 0.20,
) -> FrozenResNet18Classifier:
    return FrozenResNet18Classifier(num_classes, dropout=dropout).to(device)


def resnet_target_layer(model: nn.Module, resolution: str = "standard") -> nn.Module:
    """Return layer4 for standard CAM or layer3 for a higher-resolution CAM."""
    backbone = getattr(model, "encoder", model)
    if resolution == "standard":
        return backbone.layer4[-1]
    if resolution == "higher":
        return backbone.layer3[-1]
    raise ValueError(f"Unsupported ResNet CAM resolution: {resolution}")


def _load_cache(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def extract_frozen_resnet_features(
    model: FrozenResNet18Classifier,
    manifest: pd.DataFrame,
    device: torch.device,
    cache_path: Path | str,
    batch_size: int = 64,
    num_workers: int = 4,
    overwrite: bool = False,
    preprocessing_id: str = "imagenet_normalized_150px_v1",
) -> dict[str, object]:
    """Cache deterministic 512-dimensional GAP features for all tiles."""
    cache_path = Path(cache_path)
    expected_paths = manifest["image_path"].astype(str).tolist()
    if cache_path.is_file() and not overwrite:
        cache = _load_cache(cache_path)
        if list(cache.get("paths", [])) != expected_paths:
            raise ValueError("Frozen-ResNet cache paths do not match the manifest")
        if cache.get("preprocessing_id") != preprocessing_id:
            raise ValueError("Frozen-ResNet cache uses different preprocessing")
        return cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = UNIImageDataset(manifest, augment=False, image_size=150)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    paths: list[str] = []
    case_ids: list[str] = []
    model.eval()
    for batch in loader:
        features.append(model.encode_cls(batch["image"].to(device)).cpu())
        labels.append(torch.as_tensor(batch["label"]).cpu())
        paths.extend(batch["path"])
        case_ids.extend(batch["case_id"])
    cache = {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "paths": paths,
        "case_ids": case_ids,
        "feature_dim": model.feature_dim,
        "preprocessing_id": preprocessing_id,
    }
    torch.save(cache, cache_path)
    return cache

