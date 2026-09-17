"""UNI encoder loading and frozen-backbone classifier definitions."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class UNIClassifier(nn.Module):
    """Attach a lightweight classifier to a pathology-pretrained UNI encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        feature_dim = int(getattr(encoder, "num_features", 1024))
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.feature_dim = feature_dim
        self.encoder_frozen = False

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encoder_frozen = True

    def train(self, mode: bool = True) -> "UNIClassifier":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.forward_features(images)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError(
                "Expected UNI forward_features() to return [batch, tokens, channels]"
            )
        return tokens

    def encode_cls(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(images)[:, 0]

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_from_features(self.encode_cls(images))


def load_uni_encoder(
    project_root: Path | str,
    device: torch.device,
    model_name: str = "hf-hub:MahmoodLab/uni",
    assets_dir: Path | str | None = None,
) -> nn.Module:
    """Load UNI from Hugging Face through timm, or from an explicit local cache."""
    project_root = Path(project_root).resolve()
    if assets_dir is None:
        import timm

        try:
            encoder = timm.create_model(
                model_name,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
        except Exception as error:
            error_name = type(error).__name__
            error_message = str(error)
            if "403" in error_message or "authorized list" in error_message:
                raise RuntimeError(
                    "Hugging Face authentication succeeded, but this account "
                    "has not been granted access to MahmoodLab/UNI. Visit "
                    "https://huggingface.co/MahmoodLab/UNI while signed into "
                    "the same account, request or accept access, and rerun the "
                    "model-loading cell after approval."
                ) from error
            if error_name in {
                "GatedRepoError",
                "RepositoryNotFoundError",
                "HfHubHTTPError",
            } or "401" in error_message:
                raise RuntimeError(
                    "UNI could not be downloaded from Hugging Face. "
                    "Request or accept access to MahmoodLab/UNI, then run "
                    "huggingface_hub.login() in this notebook environment "
                    "before building the model."
                ) from error
            raise
        encoder.to(device)
    else:
        from uni.get_encoder import get_encoder

        encoder, _ = get_encoder(
            enc_name="uni",
            img_resize=224,
            center_crop=True,
            device=device,
            assets_dir=str(Path(assets_dir)),
        )
    if encoder is None:
        raise RuntimeError("The UNI encoder loader returned no model")
    encoder.eval()
    return encoder


def resolve_uni_transform(
    encoder: nn.Module,
) -> tuple[object, dict[str, object]]:
    """Resolve the evaluation transform embedded in UNI's timm configuration."""
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    data_config = resolve_data_config(encoder.pretrained_cfg, model=encoder)
    transform = create_transform(**data_config, is_training=False)
    return transform, data_config


def build_uni_classifier(
    project_root: Path | str,
    num_classes: int,
    device: torch.device,
    model_name: str = "hf-hub:MahmoodLab/uni",
    assets_dir: Path | str | None = None,
    dropout: float = 0.20,
) -> UNIClassifier:
    encoder = load_uni_encoder(
        project_root,
        device,
        model_name=model_name,
        assets_dir=assets_dir,
    )
    model = UNIClassifier(encoder, num_classes=num_classes, dropout=dropout)
    model.freeze_encoder()
    return model.to(device)


def save_classifier_head(
    model: UNIClassifier,
    checkpoint_path: Path | str,
    **metadata: object,
) -> None:
    """Save only the small trained head; UNI weights remain in their own checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "feature_dim": model.feature_dim,
            **metadata,
        },
        checkpoint_path,
    )


def load_classifier_head(
    model: UNIClassifier,
    checkpoint_path: Path | str,
    map_location: torch.device | str = "cpu",
) -> dict[str, object]:
    try:
        payload = torch.load(
            Path(checkpoint_path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(Path(checkpoint_path), map_location=map_location)
    model.classifier.load_state_dict(payload["classifier_state_dict"])
    return payload
