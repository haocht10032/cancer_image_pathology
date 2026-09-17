"""Dataset discovery, case-grouped splitting, and aligned image/mask loading."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


InputMode = Literal["rgb", "masked_rgb", "rgb_mask"]
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CASE_PATTERN = re.compile(r"CRC-Prim-HE-(\d+)", re.IGNORECASE)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _case_id(path: Path) -> str:
    match = CASE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot recover source case from filename: {path.name}")
    return f"CRC-Prim-HE-{int(match.group(1)):02d}"


def discover_images(dataset_dir: Path | str) -> pd.DataFrame:
    """Build a manifest from the eight class folders in Kather-2016."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class folders found under {dataset_dir}")

    class_to_index = {path.name: index for index, path in enumerate(class_dirs)}
    rows: list[dict[str, object]] = []
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "relative_path": str(image_path.relative_to(dataset_dir)),
                    "class_name": class_dir.name,
                    "label": class_to_index[class_dir.name],
                    "case_id": _case_id(image_path),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No supported images found under {dataset_dir}")
    return frame.sort_values(["class_name", "image_path"]).reset_index(drop=True)


def make_grouped_split(
    frame: pd.DataFrame,
    random_state: int = 41,
    test_fold: int = 0,
    validation_fold: int = 0,
) -> pd.DataFrame:
    """Create train/validation/test splits without sharing source cases."""
    required = {"label", "case_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    result = frame.copy().reset_index(drop=True)
    result["split"] = ""

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    outer_splits = list(
        outer.split(result, y=result["label"], groups=result["case_id"])
    )
    train_val_idx, test_idx = outer_splits[test_fold % len(outer_splits)]
    result.loc[test_idx, "split"] = "test"

    train_val = result.iloc[train_val_idx].reset_index()
    inner_group_count = train_val["case_id"].nunique()
    inner_splits = min(4, inner_group_count)
    if inner_splits < 2:
        raise ValueError("At least two non-test source cases are required")

    inner = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state + 1,
    )
    folds = list(
        inner.split(
            train_val,
            y=train_val["label"],
            groups=train_val["case_id"],
        )
    )
    inner_train_idx, validation_idx = folds[validation_fold % len(folds)]
    result.loc[train_val.loc[inner_train_idx, "index"], "split"] = "train"
    result.loc[train_val.loc[validation_idx, "index"], "split"] = "validation"

    if (result["split"] == "").any():
        raise RuntimeError("Some samples were not assigned to a split")

    case_membership = result.groupby("case_id")["split"].nunique()
    if case_membership.max() != 1:
        raise RuntimeError("Source-case leakage detected across splits")
    return result


class JointTileTransform:
    """Apply matching geometry to an image and mask, with image-only color jitter."""

    def __init__(self, augment: bool) -> None:
        self.augment = augment
        self.color_jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image | None]:
        if not self.augment:
            return image, mask

        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if mask is not None:
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        angle = random.choice((0, 90, 180, 270))
        if angle:
            image = image.rotate(angle)
            if mask is not None:
                mask = mask.rotate(angle)
        return self.color_jitter(image), mask


class HistologyTileDataset(Dataset):
    """Load RGB, masked RGB, or RGB plus a binary MaskCut channel."""

    def __init__(
        self,
        frame: pd.DataFrame,
        input_mode: InputMode = "rgb_mask",
        augment: bool = False,
        normalize: bool = True,
        invert_mask: bool = False,
    ) -> None:
        if input_mode not in {"rgb", "masked_rgb", "rgb_mask"}:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        if input_mode != "rgb" and "mask_path" not in frame.columns:
            raise ValueError("A mask_path column is required for mask-based inputs")

        self.frame = frame.reset_index(drop=True).copy()
        self.input_mode = input_mode
        self.normalize = normalize
        self.invert_mask = invert_mask
        self.transform = JointTileTransform(augment)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        mask = None
        if self.input_mode != "rgb":
            mask_path = Path(str(row["mask_path"]))
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing pseudo-mask: {mask_path}")
            with Image.open(mask_path) as source:
                mask = source.convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)

        image, mask = self.transform(image, mask)
        image_tensor = TF.to_tensor(image)

        mask_tensor = None
        if mask is not None:
            mask_tensor = (TF.pil_to_tensor(mask).float() >= 127.5).float()
            if self.invert_mask:
                mask_tensor = 1.0 - mask_tensor
            if self.input_mode == "masked_rgb":
                image_tensor = image_tensor * mask_tensor

        if self.normalize:
            image_tensor = TF.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)

        if self.input_mode == "rgb_mask":
            assert mask_tensor is not None
            model_input = torch.cat((image_tensor, mask_tensor), dim=0)
        else:
            model_input = image_tensor

        return {
            "image": model_input,
            "label": int(row["label"]),
            "path": str(image_path),
            "case_id": str(row["case_id"]),
        }
