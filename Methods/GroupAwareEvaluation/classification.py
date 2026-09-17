"""Out-of-fold training, prediction, and class-complete metrics."""

from __future__ import annotations

import json
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
    log_loss,
)
from torch import nn
from torch.utils.data import DataLoader

from Methods.BaselineCNN.data import HistologyTileDataset
from Methods.BaselineCNN.model import BaselineCNN
from Methods.BaselineCNN.training import fit_model, set_seed
from Methods.UNIAttribution.data import CachedFeatureDataset, UNIImageDataset
from Methods.UNIAttribution.model import (
    UNIClassifier,
    load_classifier_head,
)
from Methods.UNIAttribution.training import fit_frozen_head


def _represented_class_balanced_accuracy(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> float:
    recalls = [
        float((predictions[labels == label] == label).mean())
        for label in np.unique(labels)
    ]
    return float(np.mean(recalls))


def _load_torch(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.inference_mode()
def extract_all_uni_features(
    model: UNIClassifier,
    manifest: pd.DataFrame,
    device: torch.device,
    cache_path: Path | str,
    image_transform: Callable,
    preprocessing_id: str,
    batch_size: int = 32,
    num_workers: int = 4,
    overwrite: bool = False,
) -> dict[str, object]:
    """Cache one deterministic frozen-UNI CLS vector for every image."""
    cache_path = Path(cache_path)
    expected_paths = manifest["image_path"].astype(str).tolist()
    if cache_path.is_file() and not overwrite:
        cache = _load_torch(cache_path)
        if list(cache.get("paths", [])) != expected_paths:
            raise ValueError("Cached UNI paths do not match the current manifest")
        if cache.get("preprocessing_id") != preprocessing_id:
            raise ValueError("Cached UNI features use a different preprocessing setup")
        return cache

    cache_path.parent.mkdir(parents=True, exist_ok=True)
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
        "feature_dim": model.feature_dim,
        "preprocessing_id": preprocessing_id,
    }
    torch.save(cache, cache_path)
    return cache


extract_all_transformer_features = extract_all_uni_features


def _feature_loader(
    cache: dict[str, object],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    subset = {
        "features": torch.as_tensor(cache["features"])[indices],
        "labels": torch.as_tensor(cache["labels"])[indices],
        "paths": [cache["paths"][int(index)] for index in indices],
        "case_ids": [cache["case_ids"][int(index)] for index in indices],
    }
    return DataLoader(
        CachedFeatureDataset(subset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def _predict_feature_head(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    case_ids: list[str] = []
    classifier.eval()
    for batch in loader:
        values = classifier(batch["features"].to(device, non_blocking=True))
        logits.append(values.cpu().numpy())
        labels.append(torch.as_tensor(batch["label"]).numpy())
        paths.extend(batch["path"])
        case_ids.extend(batch["case_id"])
    return np.concatenate(logits), np.concatenate(labels), paths, case_ids


@torch.inference_mode()
def _predict_image_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    case_ids: list[str] = []
    model.eval()
    for batch in loader:
        values = model(batch["image"].to(device, non_blocking=True))
        logits.append(values.cpu().numpy())
        labels.append(torch.as_tensor(batch["label"]).numpy())
        paths.extend(batch["path"])
        case_ids.extend(batch["case_id"])
    return np.concatenate(logits), np.concatenate(labels), paths, case_ids


def _prediction_frame(
    logits: np.ndarray,
    labels: np.ndarray,
    paths: list[str],
    case_ids: list[str],
    lookup: pd.DataFrame,
    model_name: str,
    fold: int,
    seed: int,
    class_names: list[str],
) -> pd.DataFrame:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "path": paths,
            "case_id": case_ids,
            "label": labels.astype(int),
            "prediction": predictions.astype(int),
            "confidence": probabilities.max(axis=1),
            "correct": labels == predictions,
            "model": model_name,
            "fold": int(fold),
            "seed": int(seed),
        }
    )
    frame = frame.merge(
        lookup[["image_path", "relative_path", "class_name"]],
        left_on="path",
        right_on="image_path",
        how="left",
        validate="one_to_one",
    ).drop(columns="image_path")
    frame["predicted_class_name"] = [
        class_names[int(index)] for index in predictions
    ]
    for class_index, class_name in enumerate(class_names):
        safe_name = class_name.lower().replace(" ", "_")
        frame[f"logit_{class_index}_{safe_name}"] = logits[:, class_index]
        frame[f"probability_{class_index}_{safe_name}"] = probabilities[:, class_index]
    return frame


def _fold_rows(assignments: pd.DataFrame, fold: int) -> pd.DataFrame:
    frame = assignments[assignments["fold"] == fold].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No assignment rows found for fold {fold}")
    return frame


def run_uni_oof(
    model: UNIClassifier,
    feature_cache: dict[str, object],
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    class_names: list[str],
    device: torch.device,
    checkpoint_dir: Path | str,
    seeds: Iterable[int],
    batch_size: int = 128,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    force_retrain: bool = False,
    model_name: str = "UNI",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train frozen Transformer heads and emit held-out predictions."""
    checkpoint_dir = Path(checkpoint_dir)
    path_to_index = {
        str(path): index for index, path in enumerate(feature_cache["paths"])
    }
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    for seed in seeds:
        for fold in sorted(assignments["fold"].unique()):
            fold_frame = _fold_rows(assignments, int(fold))
            split_indices = {
                split: np.asarray(
                    [
                        path_to_index[path]
                        for path in fold_frame.loc[
                            fold_frame["split"] == split,
                            "image_path",
                        ].astype(str)
                    ],
                    dtype=int,
                )
                for split in ("train", "validation", "test")
            }
            loaders = {
                split: _feature_loader(
                    feature_cache,
                    indices,
                    batch_size=batch_size,
                    shuffle=split == "train",
                )
                for split, indices in split_indices.items()
            }
            checkpoint = checkpoint_dir / f"seed_{seed}" / f"fold_{fold}.pt"
            if force_retrain or not checkpoint.is_file():
                history = fit_frozen_head(
                    model,
                    loaders["train"],
                    loaders["validation"],
                    device,
                    checkpoint,
                    seed=int(seed),
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    patience=patience,
                )
                history.insert(0, "fold", int(fold))
                history.insert(0, "seed", int(seed))
                history_frames.append(history)
            else:
                load_classifier_head(model, checkpoint, map_location=device)

            logits, labels, paths, case_ids = _predict_feature_head(
                model.classifier,
                loaders["test"],
                device,
            )
            prediction_frames.append(
                _prediction_frame(
                    logits,
                    labels,
                    paths,
                    case_ids,
                    manifest,
                    model_name,
                    int(fold),
                    int(seed),
                    class_names,
                )
            )
    histories = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )
    return pd.concat(prediction_frames, ignore_index=True), histories


run_transformer_oof = run_uni_oof


def _cnn_loaders(
    fold_frame: pd.DataFrame,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        dataset = HistologyTileDataset(
            fold_frame[fold_frame["split"] == split],
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


def run_cnn_oof(
    assignments: pd.DataFrame,
    class_names: list[str],
    device: torch.device,
    checkpoint_dir: Path | str,
    seeds: Iterable[int],
    batch_size: int = 64,
    num_workers: int = 4,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    force_retrain: bool = False,
    model_builder: Callable[[], nn.Module] | None = None,
    model_name: str = "CNN",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the existing RGB CNN in each grouped fold and save OOF logits."""
    checkpoint_dir = Path(checkpoint_dir)
    if model_builder is None:
        model_builder = lambda: BaselineCNN(
            in_channels=3,
            num_classes=len(class_names),
        )
    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    for seed in seeds:
        for fold in sorted(assignments["fold"].unique()):
            fold_frame = _fold_rows(assignments, int(fold))
            loaders = _cnn_loaders(fold_frame, batch_size, num_workers)
            checkpoint = checkpoint_dir / f"seed_{seed}" / f"fold_{fold}.pt"
            set_seed(int(seed))
            model = model_builder().to(device)
            if force_retrain or not checkpoint.is_file():
                history = fit_model(
                    model,
                    loaders["train"],
                    loaders["validation"],
                    device,
                    checkpoint,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    patience=patience,
                )
                history.insert(0, "fold", int(fold))
                history.insert(0, "seed", int(seed))
                history_frames.append(history)
            else:
                model.load_state_dict(torch.load(checkpoint, map_location=device))
            logits, labels, paths, case_ids = _predict_image_model(
                model,
                loaders["test"],
                device,
            )
            prediction_frames.append(
                _prediction_frame(
                    logits,
                    labels,
                    paths,
                    case_ids,
                    fold_frame,
                    model_name,
                    int(fold),
                    int(seed),
                    class_names,
                )
            )
            del model
    histories = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )
    return pd.concat(prediction_frames, ignore_index=True), histories


def classification_metrics(
    predictions: pd.DataFrame,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    """Calculate class-complete overall, fold, and per-class OOF metrics."""
    labels = list(range(len(class_names)))
    overall_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    matrices: dict[tuple[str, int], np.ndarray] = {}
    for (model_name, seed), frame in predictions.groupby(["model", "seed"]):
        y_true = frame["label"].to_numpy()
        y_pred = frame["prediction"].to_numpy()
        probability_columns = [
            column
            for column in frame.columns
            if column.startswith("probability_")
        ]
        y_probability = frame[probability_columns].to_numpy(dtype=float)
        # CSV round-tripping introduces row-sum errors around 1e-7. Renormalize
        # before sklearn.log_loss so valid softmax probabilities do not emit a
        # misleading "values do not sum to one" warning.
        y_probability = np.clip(y_probability, 0.0, 1.0)
        y_probability /= y_probability.sum(axis=1, keepdims=True)
        source_accuracies = (
            frame.assign(_correct=frame["label"] == frame["prediction"])
            .groupby("case_id")["_correct"]
            .mean()
        )
        matrices[(str(model_name), int(seed))] = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )
        aggregate_loss = log_loss(
            y_true,
            y_probability,
            labels=labels,
        )
        overall_rows.append(
            {
                "scope": "aggregate_oof",
                "model": model_name,
                "seed": int(seed),
                "fold": -1,
                "image_count": len(frame),
                "class_coverage": int(frame["label"].nunique()),
                "source_group_count": int(frame["case_id"].nunique()),
                "accuracy": accuracy_score(y_true, y_pred),
                "source_macro_accuracy": float(source_accuracies.mean()),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
                "cross_entropy_loss": aggregate_loss,
                "test_loss": aggregate_loss,
            }
        )
        for fold, fold_frame in frame.groupby("fold"):
            fold_true = fold_frame["label"].to_numpy()
            fold_pred = fold_frame["prediction"].to_numpy()
            fold_probability = fold_frame[probability_columns].to_numpy(dtype=float)
            fold_probability = np.clip(fold_probability, 0.0, 1.0)
            fold_probability /= fold_probability.sum(axis=1, keepdims=True)
            fold_loss = log_loss(
                fold_true,
                fold_probability,
                labels=labels,
            )
            overall_rows.append(
                {
                    "scope": "fold",
                    "model": model_name,
                    "seed": int(seed),
                    "fold": int(fold),
                    "image_count": len(fold_frame),
                    "class_coverage": int(fold_frame["label"].nunique()),
                    "source_group_count": int(fold_frame["case_id"].nunique()),
                    "accuracy": accuracy_score(fold_true, fold_pred),
                    "source_macro_accuracy": accuracy_score(
                        fold_true,
                        fold_pred,
                    ),
                    "balanced_accuracy": _represented_class_balanced_accuracy(
                        fold_true,
                        fold_pred,
                    ),
                    "macro_f1": f1_score(
                        fold_true,
                        fold_pred,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    ),
                    "cross_entropy_loss": fold_loss,
                    "test_loss": fold_loss,
                }
            )
        for class_index, class_name in enumerate(class_names):
            class_frame = frame[frame["label"] == class_index]
            per_class_rows.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "label": class_index,
                    "class_name": class_name,
                    "image_count": len(class_frame),
                    "source_group_count": int(class_frame["case_id"].nunique()),
                    "accuracy": float(
                        (class_frame["prediction"] == class_index).mean()
                    ),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(per_class_rows), matrices


def save_classification_artifacts(
    predictions: pd.DataFrame,
    class_names: list[str],
    output_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist OOF predictions, logits, metrics, and confusion matrices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions_and_logits.csv", index=False)
    overall, per_class, matrices = classification_metrics(predictions, class_names)
    overall.to_csv(output_dir / "classification_metrics.csv", index=False)
    per_class.to_csv(output_dir / "per_class_accuracy.csv", index=False)
    per_source = (
        predictions.assign(
            _correct=predictions["label"] == predictions["prediction"]
        )
        .groupby(["model", "seed", "fold", "case_id"])
        .agg(
            image_count=("label", "size"),
            class_coverage=("label", "nunique"),
            accuracy=("_correct", "mean"),
        )
        .reset_index()
    )
    per_source.to_csv(output_dir / "per_source_performance.csv", index=False)
    for (model_name, seed), matrix in matrices.items():
        np.save(
            output_dir / f"confusion_{model_name.lower()}_seed_{seed}.npy",
            matrix,
        )
        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names,
        ).to_csv(
            output_dir / f"confusion_{model_name.lower()}_seed_{seed}.csv"
        )
    metadata = {
        "models": sorted(predictions["model"].unique().tolist()),
        "seeds": sorted(int(value) for value in predictions["seed"].unique()),
        "class_names": class_names,
        "aggregate_oof_images_per_model_seed": int(
            predictions.groupby(["model", "seed"]).size().min()
        ),
    }
    (output_dir / "classification_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return overall, per_class
