"""Frozen-UNI feature extraction, head training, and classification evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader

from Methods.BaselineCNN.training import set_seed
from Methods.UNIAttribution.model import (
    UNIClassifier,
    load_classifier_head,
    save_classifier_head,
)


@torch.inference_mode()
def extract_cls_feature_cache(
    model: UNIClassifier,
    image_loaders: dict[str, DataLoader],
    device: torch.device,
    cache_dir: Path | str,
    overwrite: bool = False,
    preprocessing_id: str = "uni_timm_pretrained_cfg_v1",
) -> dict[str, dict[str, object]]:
    """Extract one deterministic frozen UNI CLS embedding per image."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    caches: dict[str, dict[str, object]] = {}

    for split, loader in image_loaders.items():
        cache_path = cache_dir / f"uni_cls_{split}.pt"
        if cache_path.is_file() and not overwrite:
            try:
                cached = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                cached = torch.load(cache_path, map_location="cpu")
            expected_paths = loader.dataset.frame["image_path"].astype(str).tolist()
            if list(cached["paths"]) != expected_paths:
                raise ValueError(
                    f"Cached UNI features do not match the fixed {split} manifest. "
                    "Set overwrite=True to rebuild them."
                )
            if int(cached["feature_dim"]) != model.feature_dim:
                raise ValueError(
                    f"Cached {split} features have the wrong embedding dimension"
                )
            if cached.get("preprocessing_id") != preprocessing_id:
                raise ValueError(
                    f"Cached {split} features use different UNI preprocessing. "
                    "Set overwrite=True to rebuild them."
                )
            caches[split] = cached
            continue

        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        paths: list[str] = []
        case_ids: list[str] = []
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
        caches[split] = cache
    return caches


def reset_classifier_head(model: UNIClassifier, seed: int) -> None:
    """Reinitialize only the lightweight head for an independent training seed."""
    set_seed(seed)
    for module in model.classifier.modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()


def _run_feature_epoch(
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    training = optimizer is not None
    classifier.train(training)
    total_loss = 0.0
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    paths: list[str] = []

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = classifier(features)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(targets)
        batch_probabilities = torch.softmax(logits.detach(), dim=1)
        labels.append(targets.detach().cpu().numpy())
        predictions.append(batch_probabilities.argmax(dim=1).cpu().numpy())
        probabilities.append(batch_probabilities.cpu().numpy())
        paths.extend(batch["path"])

    sample_count = len(loader.dataset)
    return (
        total_loss / max(sample_count, 1),
        np.concatenate(labels),
        np.concatenate(predictions),
        np.concatenate(probabilities),
        paths,
    )


def fit_frozen_head(
    model: UNIClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path | str,
    seed: int = 41,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
) -> pd.DataFrame:
    """Train the lightweight head on cached CLS features and restore its best state."""
    reset_classifier_head(model, seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_f1 = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss, train_y, train_pred, _, _ = _run_feature_epoch(
            model.classifier,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_y, validation_pred, _, _ = _run_feature_epoch(
            model.classifier,
            validation_loader,
            criterion,
            device,
        )
        train_f1 = f1_score(train_y, train_pred, average="macro")
        validation_f1 = f1_score(validation_y, validation_pred, average="macro")
        scheduler.step(validation_f1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": accuracy_score(train_y, train_pred),
                "validation_accuracy": accuracy_score(validation_y, validation_pred),
                "train_macro_f1": train_f1,
                "validation_macro_f1": validation_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            epochs_without_improvement = 0
            save_classifier_head(
                model,
                checkpoint_path,
                seed=seed,
                validation_macro_f1=float(validation_f1),
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    load_classifier_head(model, checkpoint_path, map_location=device)
    model.eval()
    return pd.DataFrame(history)


def evaluate_frozen_head(
    model: UNIClassifier,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, object]:
    """Evaluate the trained head while preserving predictions for attribution sampling."""
    criterion = nn.CrossEntropyLoss()
    loss, labels, predictions, probabilities, paths = _run_feature_epoch(
        model.classifier,
        loader,
        criterion,
        device,
    )
    label_indices = np.arange(len(class_names))
    return {
        "loss": loss,
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=label_indices,
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "paths": paths,
    }


def classification_row(
    evaluation: dict[str, object],
    seed: int,
    model_name: str = "UNI frozen encoder",
) -> dict[str, object]:
    return {
        "seed": seed,
        "model": model_name,
        "accuracy": evaluation["accuracy"],
        "balanced_accuracy": evaluation["balanced_accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "test_loss": evaluation["loss"],
    }
