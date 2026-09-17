"""Training and evaluation helpers shared by the trial notebook."""

from __future__ import annotations

import random
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


def set_seed(seed: int = 41) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for batch in loader:
        inputs = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        batch_probabilities = torch.softmax(logits.detach(), dim=1)
        labels.append(targets.detach().cpu().numpy())
        predictions.append(batch_probabilities.argmax(dim=1).cpu().numpy())
        probabilities.append(batch_probabilities.cpu().numpy())

    sample_count = len(loader.dataset)
    return (
        total_loss / max(sample_count, 1),
        np.concatenate(labels),
        np.concatenate(predictions),
        np.concatenate(probabilities),
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Path | str,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
) -> pd.DataFrame:
    """Train with early stopping on validation macro-F1 and restore the best state."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    model.to(device)
    best_f1 = -np.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_y, train_pred, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss, validation_y, validation_pred, _ = _run_epoch(
            model,
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
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return pd.DataFrame(history)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    loss, labels, predictions, probabilities = _run_epoch(
        model,
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
    }
