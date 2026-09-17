"""Classification tables for the main and matched-training Kather analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Methods.GroupAwareEvaluation.classification import classification_metrics
from Methods.KatherRevision.statistics import (
    exact_source_sign_flip,
    hierarchical_bootstrap_ci,
    leave_one_source_out,
)


def _confusion_long(
    matrices: dict[tuple[str, int], object],
    class_names: list[str],
) -> pd.DataFrame:
    rows = []
    for (model, seed), matrix in matrices.items():
        for true_index, true_name in enumerate(class_names):
            for predicted_index, predicted_name in enumerate(class_names):
                rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "true_class": true_name,
                        "predicted_class": predicted_name,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    return pd.DataFrame(rows)


def paired_accuracy_source_inference(
    predictions: pd.DataFrame,
    reference_model: str = "UNI",
    bootstrap_iterations: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare image correctness after averaging seeds, clustered by source."""
    image = (
        predictions.groupby(
            ["relative_path", "case_id", "model"], as_index=False
        )["correct"]
        .mean()
    )
    pivot = image.pivot(
        index=["relative_path", "case_id"], columns="model", values="correct"
    ).reset_index()
    rows = []
    loso_frames = []
    for comparator in sorted(set(predictions["model"]) - {reference_model}):
        if not {reference_model, comparator}.issubset(pivot.columns):
            continue
        paired = pivot.dropna(subset=[reference_model, comparator]).copy()
        paired["difference"] = paired[reference_model] - paired[comparator]
        estimate, low, high = hierarchical_bootstrap_ci(
            paired,
            "difference",
            image_column="relative_path",
            iterations=bootstrap_iterations,
        )
        sign_estimate, sign_p, groups, inference = exact_source_sign_flip(
            paired, "difference", alternative="two-sided"
        )
        rows.append(
            {
                "model_a": reference_model,
                "model_b": comparator,
                "metric": "accuracy",
                "difference_definition": "model_a_minus_model_b",
                "images": int(len(paired)),
                "source_groups": groups,
                "mean_difference": estimate,
                "hierarchical_ci_low": low,
                "hierarchical_ci_high": high,
                "source_sign_flip_estimate": sign_estimate,
                "source_sign_flip_p": sign_p,
                "source_sign_flip_type": inference,
            }
        )
        loso = leave_one_source_out(paired, "difference")
        loso["model_a"] = reference_model
        loso["model_b"] = comparator
        loso_frames.append(loso)
    return (
        pd.DataFrame(rows),
        pd.concat(loso_frames, ignore_index=True) if loso_frames else pd.DataFrame(),
    )


def save_classification_revision(
    predictions: pd.DataFrame,
    class_names: list[str],
    output_dir: Path | str,
    bootstrap_iterations: int = 5000,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall, per_class, matrices = classification_metrics(predictions, class_names)
    per_seed = overall[overall["scope"].eq("aggregate_oof")].copy()
    per_fold = overall[overall["scope"].eq("fold")].copy()
    confusion = _confusion_long(matrices, class_names)
    source_seed = (
        predictions.groupby(["model", "seed", "case_id"], as_index=False)
        .agg(images=("relative_path", "nunique"), accuracy=("correct", "mean"))
    )
    counts = (
        predictions.drop_duplicates(["model", "seed", "relative_path"])
        .groupby(["model", "seed", "case_id", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "image_count"})
    )
    paired, loso = paired_accuracy_source_inference(
        predictions, bootstrap_iterations=bootstrap_iterations
    )
    tables = {
        "oof_predictions_and_logits": predictions,
        "classification_per_seed": per_seed,
        "classification_per_fold": per_fold,
        "classification_per_class": per_class,
        "classification_confusion_long": confusion,
        "classification_source_seed": source_seed,
        "classification_class_source_counts": counts,
        "classification_paired_source_inference": paired,
        "classification_leave_one_source_out": loso,
    }
    paths = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
