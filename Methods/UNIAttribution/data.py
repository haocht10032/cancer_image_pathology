"""Fixed-split image and cached-feature data utilities for UNI experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from Methods.BaselineCNN.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    JointTileTransform,
    discover_images,
    make_grouped_split,
)


UNI_IMAGE_SIZE = 224
UNI_PATCH_SIZE = 16
UNI_GRID_SIZE = UNI_IMAGE_SIZE // UNI_PATCH_SIZE


def load_fixed_manifest(
    dataset_dir: Path | str,
    previous_manifest_path: Path | str | None = None,
    random_state: int = 41,
) -> pd.DataFrame:
    """Reuse an earlier split by relative path, or recreate it deterministically."""
    current = discover_images(dataset_dir)
    if previous_manifest_path is None:
        return make_grouped_split(current, random_state=random_state)

    previous_manifest_path = Path(previous_manifest_path)
    if not previous_manifest_path.is_file():
        return make_grouped_split(current, random_state=random_state)

    previous = pd.read_csv(previous_manifest_path)
    required = {"relative_path", "split"}
    missing = required.difference(previous.columns)
    if missing:
        raise ValueError(
            f"Previous manifest is missing required columns: {sorted(missing)}"
        )

    split_table = previous[["relative_path", "split"]].drop_duplicates()
    if split_table["relative_path"].duplicated().any():
        raise ValueError("Previous manifest has conflicting duplicate relative paths")

    merged = current.merge(
        split_table,
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(current) or merged["split"].isna().any():
        missing_count = int(merged["split"].isna().sum())
        raise ValueError(
            "The previous split does not match the current dataset "
            f"({missing_count} images have no split assignment)"
        )

    case_membership = merged.groupby("case_id")["split"].nunique()
    if case_membership.max() != 1:
        raise ValueError("Source-case leakage found in the reused split")
    return merged.sort_values(["class_name", "image_path"]).reset_index(drop=True)


class UNIImageDataset(Dataset):
    """Load Kather tiles with UNI's 224-pixel ImageNet-normalized input."""

    def __init__(
        self,
        frame: pd.DataFrame,
        augment: bool = False,
        image_size: int = UNI_IMAGE_SIZE,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        if "image_path" not in self.frame.columns:
            if "path" not in self.frame.columns:
                raise ValueError(
                    "UNIImageDataset requires an 'image_path' or 'path' column"
                )
            self.frame["image_path"] = self.frame["path"]
        self.transform = JointTileTransform(augment)
        self.image_size = image_size
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        image, _ = self.transform(image, None)
        if self.image_transform is not None:
            image_tensor = self.image_transform(image)
        else:
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BICUBIC,
            )
            image_tensor = TF.to_tensor(image)
            image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        return {
            "image": image_tensor,
            "label": int(row["label"]),
            "path": str(image_path),
            "case_id": str(row["case_id"]),
        }


class CachedFeatureDataset(Dataset):
    """Expose cached frozen-encoder CLS features to a lightweight head."""

    def __init__(self, cache: dict[str, object]) -> None:
        self.features = torch.as_tensor(cache["features"]).float()
        self.labels = torch.as_tensor(cache["labels"]).long()
        self.paths = list(cache["paths"])
        self.case_ids = list(cache["case_ids"])
        if len(self.features) != len(self.labels):
            raise ValueError("Cached feature and label counts do not match")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "features": self.features[index],
            "label": self.labels[index],
            "path": self.paths[index],
            "case_id": self.case_ids[index],
        }


def build_image_loaders(
    manifest: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_training: bool = False,
    image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
) -> dict[str, DataLoader]:
    """Build deterministic UNI loaders; augmentation is opt-in for feature caching."""
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        split_frame = manifest[manifest["split"] == split]
        if split_frame.empty:
            raise ValueError(f"No samples found for split: {split}")
        dataset = UNIImageDataset(
            split_frame,
            augment=augment_training and split == "train",
            image_transform=image_transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders


def build_feature_loaders(
    caches: dict[str, dict[str, object]],
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build loaders for cached CLS features."""
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        dataset = CachedFeatureDataset(caches[split])
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Convert an ImageNet-normalized tensor back to displayable RGB."""
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)
