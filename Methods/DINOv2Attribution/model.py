"""DINOv2 ViT-L/14 frozen-backbone classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from Methods.UNIAttribution.model import UNIClassifier


DINO_MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"
DINO_IMAGE_SIZE = 224
DINO_PATCH_SIZE = 14
DINO_NATIVE_GRID_SIZE = DINO_IMAGE_SIZE // DINO_PATCH_SIZE


class DINOv2Classifier(UNIClassifier):
    """The same lightweight head used for UNI, attached to DINOv2."""


def _load_dinov2_encoder(
    device: torch.device,
    model_name: str = DINO_MODEL_NAME,
    checkpoint_path: Path | str | None = None,
) -> nn.Module:
    import timm

    create_kwargs = {
        "img_size": DINO_IMAGE_SIZE,
        "dynamic_img_size": True,
        "pretrained_cfg_overlay": {
            "input_size": (3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
            "crop_pct": 1.0,
        },
    }
    if checkpoint_path is None:
        encoder = timm.create_model(
            model_name,
            pretrained=True,
            **create_kwargs,
        )
    else:
        encoder = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=str(Path(checkpoint_path).expanduser().resolve()),
            **create_kwargs,
        )
    encoder.to(device)
    encoder.eval()
    return encoder


def build_dinov2_classifier(
    num_classes: int,
    device: torch.device,
    model_name: str = DINO_MODEL_NAME,
    checkpoint_path: Path | str | None = None,
    dropout: float = 0.20,
) -> DINOv2Classifier:
    """Build frozen DINOv2 with the UNI-equivalent classifier head."""
    encoder = _load_dinov2_encoder(
        device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )
    model = DINOv2Classifier(
        encoder,
        num_classes=num_classes,
        dropout=dropout,
    )
    model.freeze_encoder()
    return model.to(device)


def resolve_dinov2_transform(
    encoder: nn.Module,
) -> tuple[object, dict[str, object]]:
    """Resolve the 224-pixel model transform after config overlay."""
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    data_config = resolve_data_config(encoder.pretrained_cfg, model=encoder)
    data_config["input_size"] = (3, DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)
    data_config["crop_pct"] = 1.0
    transform = create_transform(**data_config, is_training=False)
    return transform, data_config
