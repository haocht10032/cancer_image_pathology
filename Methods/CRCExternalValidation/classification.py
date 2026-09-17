"""Full-data common-seven training and CRC-VAL external inference."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from Methods.BaselineCNN import discover_images, set_seed
from Methods.BaselineCNN.data import HistologyTileDataset
from Methods.CNNBenchmark import build_resnet18
from Methods.UNIAttribution.data import CachedFeatureDataset, UNIImageDataset
from Methods.UNIAttribution.model import (
    UNIClassifier,
    load_classifier_head,
    save_classifier_head,
)


COMMON_CLASS_NAMES = (
    "adipose",
    "background",
    "debris_mucus",
    "lymphocytes",
    "normal_mucosa",
    "stroma",
    "tumor",
)
COMMON_TO_LABEL = {name: index for index, name in enumerate(COMMON_CLASS_NAMES)}
KATHER_TO_COMMON = {
    "01_TUMOR": "tumor",
    "02_STROMA": "stroma",
    "04_LYMPHO": "lymphocytes",
    "05_DEBRIS": "debris_mucus",
    "06_MUCOSA": "normal_mucosa",
    "07_ADIPOSE": "adipose",
    "08_EMPTY": "background",
}
CRC_TO_COMMON = {
    "ADI": "adipose",
    "BACK": "background",
    "DEB": "debris_mucus",
    "LYM": "lymphocytes",
    "MUC": "debris_mucus",
    "NORM": "normal_mucosa",
    "STR": "stroma",
    "TUM": "tumor",
}


def build_kather_common_manifest(dataset_dir: Path | str) -> pd.DataFrame:
    """Remove complex stroma and relabel Kather into the locked common taxonomy."""
    frame = discover_images(dataset_dir)
    frame = frame[frame["class_name"].isin(KATHER_TO_COMMON)].copy()
    frame["source_class"] = frame["class_name"]
    frame["class_name"] = frame["source_class"].map(KATHER_TO_COMMON)
    frame["label"] = frame["class_name"].map(COMMON_TO_LABEL).astype(int)
    frame["domain"] = "Kather-5K"
    frame["patient_id"] = pd.NA
    frame["slide_id"] = frame["case_id"]
    return frame.sort_values(["label", "case_id", "relative_path"]).reset_index(
        drop=True
    )


def build_crc_common_manifest(
    dataset_dir: Path | str,
    preflight_inventory: Path | str | None = None,
) -> pd.DataFrame:
    """Build the eligible CRC manifest without treating filenames as patients."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if preflight_inventory is not None and Path(preflight_inventory).is_file():
        frame = pd.read_csv(preflight_inventory)
        required = {"image_path", "relative_path", "crc_class"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Preflight inventory lacks columns: {sorted(missing)}")
    else:
        rows = []
        for source_class in sorted(CRC_TO_COMMON):
            for image_path in sorted((dataset_dir / source_class).glob("*.tif")):
                rows.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "relative_path": str(image_path.relative_to(dataset_dir)),
                        "crc_class": source_class,
                    }
                )
        frame = pd.DataFrame(rows)
    frame = frame[frame["crc_class"].isin(CRC_TO_COMMON)].copy()
    frame["source_class"] = frame["crc_class"]
    frame["class_name"] = frame["source_class"].map(CRC_TO_COMMON)
    frame["label"] = frame["class_name"].map(COMMON_TO_LABEL).astype(int)
    frame["domain"] = "CRC-VAL-HE-7K"
    for column in ("patient_id", "slide_id"):
        if column not in frame:
            frame[column] = pd.NA
    frame["case_id"] = frame["patient_id"].astype("string")
    return frame.sort_values(["label", "source_class", "relative_path"]).reset_index(
        drop=True
    )


def _load_torch(path: Path, device: torch.device | str = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.inference_mode()
def extract_transformer_cache(
    model: UNIClassifier,
    manifest: pd.DataFrame,
    device: torch.device,
    image_transform: Callable,
    cache_path: Path | str,
    preprocessing_id: str,
    batch_size: int = 24,
    num_workers: int = 4,
    overwrite: bool = False,
) -> dict[str, object]:
    """Extract deterministic CLS features once for full-data head training."""
    cache_path = Path(cache_path)
    expected_paths = manifest["image_path"].astype(str).tolist()
    if cache_path.is_file() and not overwrite:
        cache = _load_torch(cache_path)
        if list(cache.get("paths", [])) != expected_paths:
            raise ValueError(f"Cached paths do not match manifest: {cache_path}")
        if cache.get("preprocessing_id") != preprocessing_id:
            raise ValueError(f"Cached preprocessing does not match: {cache_path}")
        return cache

    dataset = UNIImageDataset(
        manifest,
        augment=False,
        image_transform=image_transform,
    )
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
        images = batch["image"].to(device, non_blocking=True)
        features.append(model.encode_cls(images).detach().cpu())
        labels.append(torch.as_tensor(batch["label"]).cpu())
        paths.extend(batch["path"])
        case_ids.extend(batch["case_id"])
    cache = {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "paths": paths,
        "case_ids": case_ids,
        "feature_dim": int(model.feature_dim),
        "preprocessing_id": preprocessing_id,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _feature_loader(
    cache: dict[str, object], batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        CachedFeatureDataset(cache),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _fixed_feature_fit(
    model: UNIClassifier,
    cache: dict[str, object],
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
) -> pd.DataFrame:
    set_seed(seed)
    for module in model.classifier.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    loader = _feature_loader(cache, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.classifier.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum())
            count += len(targets)
        scheduler.step()
        history.append(
            {
                "model": type(model).__name__,
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": total_loss / max(count, 1),
                "train_accuracy": correct / max(count, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    save_classifier_head(
        model,
        checkpoint_path,
        seed=int(seed),
        epochs=int(epochs),
        training_protocol="full_kather_fixed_epoch_common_seven",
    )
    model.eval()
    return pd.DataFrame(history)


@torch.inference_mode()
def _predict_feature_cache(
    model: UNIClassifier,
    cache: dict[str, object],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = _feature_loader(cache, batch_size=batch_size, shuffle=False)
    logits = []
    labels = []
    paths = []
    model.eval()
    for batch in loader:
        logits.append(model.classifier(batch["features"].to(device)).cpu().numpy())
        labels.append(torch.as_tensor(batch["label"]).numpy())
        paths.extend(batch["path"])
    return np.concatenate(logits), np.concatenate(labels), paths


def _prediction_frame(
    logits: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    manifest: pd.DataFrame,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "image_path": paths,
            "label": labels.astype(int),
            "prediction": predictions.astype(int),
            "confidence": probabilities.max(axis=1),
            "correct": labels == predictions,
            "model": model_name,
            "seed": int(seed),
        }
    )
    metadata = manifest[
        [
            "image_path",
            "relative_path",
            "source_class",
            "class_name",
            "patient_id",
            "slide_id",
        ]
    ].copy()
    frame = frame.merge(metadata, on="image_path", how="left", validate="one_to_one")
    frame["predicted_class_name"] = [COMMON_CLASS_NAMES[i] for i in predictions]
    for index, class_name in enumerate(COMMON_CLASS_NAMES):
        frame[f"logit_{index}_{class_name}"] = logits[:, index]
        frame[f"probability_{index}_{class_name}"] = probabilities[:, index]
    return frame


def train_evaluate_transformer_heads(
    model: UNIClassifier,
    train_cache: dict[str, object],
    external_cache: dict[str, object],
    external_manifest: pd.DataFrame,
    device: torch.device,
    checkpoint_dir: Path | str,
    model_name: str,
    seeds: Iterable[int] = (11, 89, 181),
    epochs: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    force_retrain: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir = Path(checkpoint_dir)
    predictions = []
    histories = []
    for seed in seeds:
        checkpoint = checkpoint_dir / f"seed_{int(seed)}.pt"
        if force_retrain or not checkpoint.is_file():
            history = _fixed_feature_fit(
                model,
                train_cache,
                device,
                checkpoint,
                int(seed),
                epochs,
                learning_rate,
                weight_decay,
                batch_size,
            )
            history["model"] = model_name
            histories.append(history)
        else:
            load_classifier_head(model, checkpoint, map_location=device)
        logits, labels, paths = _predict_feature_cache(
            model, external_cache, device, batch_size
        )
        predictions.append(
            _prediction_frame(
                logits, labels, paths, external_manifest, model_name, int(seed)
            )
        )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(),
    )


def _resnet_loader(
    manifest: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    if training:
        dataset = HistologyTileDataset(manifest, input_mode="rgb", augment=True)
    else:
        dataset = UNIImageDataset(manifest, augment=False, image_size=150)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def _fixed_resnet_fit(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    checkpoint: Path,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> pd.DataFrame:
    set_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    history = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum())
            count += len(targets)
        scheduler.step()
        history.append(
            {
                "model": "ResNet18",
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss": total_loss / max(count, 1),
                "train_accuracy": correct / max(count, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return pd.DataFrame(history)


@torch.inference_mode()
def _predict_image_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logits = []
    labels = []
    paths = []
    model.eval()
    for batch in loader:
        logits.append(model(batch["image"].to(device)).cpu().numpy())
        labels.append(torch.as_tensor(batch["label"]).numpy())
        paths.extend(batch["path"])
    return np.concatenate(logits), np.concatenate(labels), paths


def train_evaluate_resnet(
    train_manifest: pd.DataFrame,
    external_manifest: pd.DataFrame,
    device: torch.device,
    checkpoint_dir: Path | str,
    seeds: Iterable[int] = (11, 89, 181),
    epochs: int = 12,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    num_workers: int = 4,
    force_retrain: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir = Path(checkpoint_dir)
    external_loader = _resnet_loader(
        external_manifest, batch_size, num_workers, training=False
    )
    predictions = []
    histories = []
    for seed in seeds:
        set_seed(int(seed))
        train_loader = _resnet_loader(
            train_manifest, batch_size, num_workers, training=True
        )
        model = build_resnet18(
            num_classes=len(COMMON_CLASS_NAMES),
            pretrained=True,
            freeze_backbone=False,
        ).to(device)
        checkpoint = checkpoint_dir / f"seed_{int(seed)}.pt"
        if force_retrain or not checkpoint.is_file():
            histories.append(
                _fixed_resnet_fit(
                    model,
                    train_loader,
                    device,
                    checkpoint,
                    int(seed),
                    epochs,
                    learning_rate,
                    weight_decay,
                )
            )
        else:
            model.load_state_dict(_load_torch(checkpoint, device))
        logits, labels, paths = _predict_image_model(model, external_loader, device)
        predictions.append(
            _prediction_frame(
                logits, labels, paths, external_manifest, "ResNet18", int(seed)
            )
        )
        del train_loader
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(),
    )


def classification_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return seed-level, per-class, and long-form confusion tables."""
    labels = list(range(len(COMMON_CLASS_NAMES)))
    overall = []
    per_class = []
    confusion = []
    for (model_name, seed), frame in predictions.groupby(["model", "seed"]):
        y_true = frame["label"].to_numpy(dtype=int)
        y_pred = frame["prediction"].to_numpy(dtype=int)
        overall.append(
            {
                "model": model_name,
                "seed": int(seed),
                "images": int(len(frame)),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                ),
            }
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        for index, class_name in enumerate(COMMON_CLASS_NAMES):
            per_class.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "class_name": class_name,
                    "precision": precision[index],
                    "recall": recall[index],
                    "f1": f1[index],
                    "support": int(support[index]),
                }
            )
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        for true_index, true_name in enumerate(COMMON_CLASS_NAMES):
            for predicted_index, predicted_name in enumerate(COMMON_CLASS_NAMES):
                confusion.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[true_index, predicted_index]),
                        "row_proportion": matrix[true_index, predicted_index]
                        / max(matrix[true_index].sum(), 1),
                    }
                )
    return pd.DataFrame(overall), pd.DataFrame(per_class), pd.DataFrame(confusion)
