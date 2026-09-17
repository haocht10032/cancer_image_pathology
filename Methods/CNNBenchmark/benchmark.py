"""Reusable orchestration for RGB-only CNN benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from Methods.BaselineCNN.data import HistologyTileDataset
from Methods.BaselineCNN.training import evaluate_model, fit_model, set_seed


def balanced_training_subset(
    manifest: pd.DataFrame,
    sample_count: int | None,
    random_state: int = 41,
) -> pd.DataFrame:
    """Subsample only the training split while preserving all validation/test rows."""
    if "split" not in manifest.columns:
        raise ValueError("Manifest must contain a split column")

    training = manifest[manifest["split"] == "train"]
    held_out = manifest[manifest["split"] != "train"]
    if sample_count is None or sample_count >= len(training):
        return manifest.copy().reset_index(drop=True)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    labels = sorted(training["label"].unique())
    base_count, remainder = divmod(sample_count, len(labels))
    sampled_groups: list[pd.DataFrame] = []
    for position, label in enumerate(labels):
        class_rows = training[training["label"] == label]
        requested = base_count + int(position < remainder)
        if requested > len(class_rows):
            raise ValueError(
                f"Class {label} has {len(class_rows)} training rows, "
                f"but {requested} were requested"
            )
        sampled_groups.append(
            class_rows.sample(
                n=requested,
                random_state=random_state + int(label),
            )
        )

    sampled_training = pd.concat(sampled_groups, ignore_index=True)
    result = pd.concat((sampled_training, held_out), ignore_index=True)
    return result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_rgb_loaders(
    manifest: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Build RGB loaders with augmentation restricted to the training set."""
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        split_frame = manifest[manifest["split"] == split]
        if split_frame.empty:
            raise ValueError(f"No samples found for split: {split}")
        dataset = HistologyTileDataset(
            split_frame,
            input_mode="rgb",
            augment=split == "train",
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    return loaders


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def run_benchmark(
    model_builder: Callable[[], nn.Module],
    manifest: pd.DataFrame,
    class_names: list[str],
    device: torch.device,
    checkpoint_path: Path | str,
    random_state: int = 41,
    batch_size: int = 64,
    num_workers: int = 4,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
) -> tuple[nn.Module, pd.DataFrame, dict[str, object]]:
    """Train one model configuration and evaluate its best checkpoint."""
    set_seed(random_state)
    loaders = build_rgb_loaders(
        manifest,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = model_builder()
    history = fit_model(
        model,
        loaders["train"],
        loaders["validation"],
        device,
        checkpoint_path,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
    )
    evaluation = evaluate_model(
        model,
        loaders["test"],
        device,
        class_names,
    )
    return model, history, evaluation
