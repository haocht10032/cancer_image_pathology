"""Quantitative attribution evaluation and representative-example selection."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

from Methods.BaselineCNN.data import IMAGENET_MEAN, IMAGENET_STD
from Methods.UNIAttribution.attribution import (
    GradCAM,
    attribution_occlusion_correlation,
    deletion_auc,
    deletion_curves,
    normalize_map,
    pairwise_attribution_stability,
    patch_occlusion_scores,
    save_attribution_figure,
    transformer_attributions,
)
from Methods.UNIAttribution.data import UNI_GRID_SIZE, UNI_IMAGE_SIZE
from Methods.UNIAttribution.model import UNIClassifier, load_classifier_head


ATTRIBUTION_METHODS = (
    "raw_attention",
    "attention_rollout",
    "gradient_attention_rollout",
)


def select_representative_examples(
    evaluation: dict[str, object],
    correct_count: int = 4,
    incorrect_count: int = 4,
) -> pd.DataFrame:
    """Select confident correct and incorrect test predictions."""
    labels = np.asarray(evaluation["labels"])
    predictions = np.asarray(evaluation["predictions"])
    probabilities = np.asarray(evaluation["probabilities"])
    paths = list(evaluation["paths"])
    confidence = probabilities[np.arange(len(predictions)), predictions]
    frame = pd.DataFrame(
        {
            "path": paths,
            "label": labels,
            "prediction": predictions,
            "confidence": confidence,
            "correct": labels == predictions,
        }
    )
    correct = (
        frame[frame["correct"]]
        .sort_values("confidence", ascending=False)
        .head(correct_count)
    )
    incorrect = (
        frame[~frame["correct"]]
        .sort_values("confidence", ascending=False)
        .head(incorrect_count)
    )
    return pd.concat((correct, incorrect), ignore_index=True)


def load_normalized_image(
    image_path: Path | str,
    image_size: int,
    device: torch.device,
    image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Load an RGB image with the normalization used by both project methods."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if image_transform is not None:
        tensor = image_transform(image)
    else:
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0).to(device)


def _normalized_deletion_curves(curves: pd.DataFrame) -> pd.DataFrame:
    result = curves.copy()
    for strategy, indices in result.groupby("strategy").groups.items():
        baseline = result.loc[
            indices,
            ["fraction_removed", "target_probability", "target_logit"],
        ].sort_values("fraction_removed").iloc[0]
        probability_scale = max(abs(float(baseline["target_probability"])), 1e-8)
        logit_scale = max(abs(float(baseline["target_logit"])), 1e-8)
        result.loc[indices, "normalized_probability"] = (
            result.loc[indices, "target_probability"] / probability_scale
        )
        result.loc[indices, "normalized_logit"] = (
            result.loc[indices, "target_logit"] / logit_scale
        )
        result.loc[indices, "probability_drop"] = (
            float(baseline["target_probability"])
            - result.loc[indices, "target_probability"]
        )
        result.loc[indices, "logit_drop"] = (
            float(baseline["target_logit"]) - result.loc[indices, "target_logit"]
        )
    return result


def evaluate_uni_examples(
    model: UNIClassifier,
    examples: pd.DataFrame,
    device: torch.device,
    class_names: list[str],
    output_dir: Path | str,
    random_repeats: int = 20,
    random_seed: int = 41,
    image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate UNI maps against patch occlusion and top-k deletion."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps" / "uni"
    metric_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    occlusion_frames: list[pd.DataFrame] = []

    for example_index, row in examples.reset_index(drop=True).iterrows():
        image = load_normalized_image(
            row["path"],
            UNI_IMAGE_SIZE,
            device,
            image_transform=image_transform,
        )
        attributions = transformer_attributions(model, image)
        target_class = int(attributions["predicted_class"])
        occlusion = patch_occlusion_scores(
            model,
            image,
            target_class=target_class,
        )
        occlusion_map = normalize_map(occlusion["logit_drop"])
        occlusion_frame = pd.DataFrame(
            {
                "patch_row": np.repeat(np.arange(UNI_GRID_SIZE), UNI_GRID_SIZE),
                "patch_column": np.tile(np.arange(UNI_GRID_SIZE), UNI_GRID_SIZE),
                "logit_drop": occlusion["logit_drop"].flatten().numpy(),
                "probability_drop": occlusion["probability_drop"].flatten().numpy(),
            }
        )
        occlusion_frame.insert(0, "predicted_class", target_class)
        occlusion_frame.insert(0, "correct", bool(row["correct"]))
        occlusion_frame.insert(0, "path", row["path"])
        occlusion_frame["logit_drop_rank"] = occlusion_frame["logit_drop"].rank(
            method="min",
            ascending=False,
        )
        occlusion_frames.append(occlusion_frame)
        maps = {
            method: attributions[method]
            for method in ATTRIBUTION_METHODS
        }
        maps["patch_occlusion"] = occlusion_map

        status = "correct" if bool(row["correct"]) else "incorrect"
        title = (
            f"{status}: true={class_names[int(row['label'])]}, "
            f"pred={class_names[target_class]}"
        )
        save_attribution_figure(
            image,
            maps,
            heatmap_dir / f"{status}_{example_index:02d}.png",
            title,
        )

        for method in ATTRIBUTION_METHODS:
            attribution = attributions[method]
            correlation, correlation_p = attribution_occlusion_correlation(
                attribution,
                occlusion["logit_drop"],
            )
            curves = deletion_curves(
                model,
                image,
                attribution,
                target_class=target_class,
                random_repeats=random_repeats,
                random_seed=random_seed + example_index,
            )
            curves = _normalized_deletion_curves(curves)
            curves.insert(0, "method", method)
            curves.insert(0, "correct", bool(row["correct"]))
            curves.insert(0, "path", row["path"])
            curve_frames.append(curves)
            aucs = deletion_auc(curves, value_column="normalized_probability")
            metric_rows.append(
                {
                    "model": "UNI",
                    "path": row["path"],
                    "correct": bool(row["correct"]),
                    "true_class": int(row["label"]),
                    "predicted_class": target_class,
                    "attribution_method": method,
                    "attribution_occlusion_spearman": correlation,
                    "attribution_occlusion_p": correlation_p,
                    "top_deletion_auc": aucs["highest"],
                    "random_deletion_auc": aucs["random"],
                    "low_deletion_auc": aucs["lowest"],
                    "top_minus_random_auc": aucs["highest"] - aucs["random"],
                    "original_probability": occlusion["original_probability"],
                    "mean_occlusion_logit_drop": float(
                        occlusion["logit_drop"].mean()
                    ),
                    "max_occlusion_logit_drop": float(
                        occlusion["logit_drop"].max()
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    occlusion_scores = pd.concat(occlusion_frames, ignore_index=True)
    metrics.to_csv(output_dir / "uni_attribution_metrics.csv", index=False)
    curves.to_csv(output_dir / "uni_deletion_curves.csv", index=False)
    occlusion_scores.to_csv(
        output_dir / "uni_patch_occlusion_scores.csv",
        index=False,
    )
    return metrics, curves


def evaluate_cnn_examples(
    model: nn.Module,
    target_layer: nn.Module,
    examples: pd.DataFrame,
    device: torch.device,
    class_names: list[str],
    output_dir: Path | str,
    random_repeats: int = 20,
    random_seed: int = 41,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate CNN Grad-CAM with the same 14 x 14 occlusion protocol."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps" / "cnn"
    metric_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    occlusion_frames: list[pd.DataFrame] = []

    with GradCAM(model, target_layer) as gradcam:
        for example_index, row in examples.reset_index(drop=True).iterrows():
            image = load_normalized_image(row["path"], 150, device)
            attribution_result = gradcam.attribute(image)
            target_class = int(attribution_result["predicted_class"])
            attribution = attribution_result["gradcam"]
            occlusion = patch_occlusion_scores(
                model,
                image,
                target_class=target_class,
            )
            occlusion_map = normalize_map(occlusion["logit_drop"])
            occlusion_frame = pd.DataFrame(
                {
                    "patch_row": np.repeat(
                        np.arange(UNI_GRID_SIZE),
                        UNI_GRID_SIZE,
                    ),
                    "patch_column": np.tile(
                        np.arange(UNI_GRID_SIZE),
                        UNI_GRID_SIZE,
                    ),
                    "logit_drop": occlusion["logit_drop"].flatten().numpy(),
                    "probability_drop": occlusion[
                        "probability_drop"
                    ].flatten().numpy(),
                }
            )
            occlusion_frame.insert(0, "predicted_class", target_class)
            occlusion_frame.insert(
                0,
                "correct",
                int(row["label"]) == target_class,
            )
            occlusion_frame.insert(0, "path", row["path"])
            occlusion_frame["logit_drop_rank"] = occlusion_frame[
                "logit_drop"
            ].rank(method="min", ascending=False)
            occlusion_frames.append(occlusion_frame)
            status = "correct" if int(row["label"]) == target_class else "incorrect"
            title = (
                f"{status}: true={class_names[int(row['label'])]}, "
                f"pred={class_names[target_class]}"
            )
            save_attribution_figure(
                image,
                {"Grad-CAM": attribution, "patch_occlusion": occlusion_map},
                heatmap_dir / f"{status}_{example_index:02d}.png",
                title,
            )

            correlation, correlation_p = attribution_occlusion_correlation(
                attribution,
                occlusion["logit_drop"],
            )
            curves = deletion_curves(
                model,
                image,
                attribution,
                target_class=target_class,
                random_repeats=random_repeats,
                random_seed=random_seed + example_index,
            )
            curves = _normalized_deletion_curves(curves)
            curves.insert(0, "method", "gradcam")
            curves.insert(0, "correct", int(row["label"]) == target_class)
            curves.insert(0, "path", row["path"])
            curve_frames.append(curves)
            aucs = deletion_auc(curves, value_column="normalized_probability")
            metric_rows.append(
                {
                    "model": "CNN",
                    "path": row["path"],
                    "correct": int(row["label"]) == target_class,
                    "true_class": int(row["label"]),
                    "predicted_class": target_class,
                    "attribution_method": "gradcam",
                    "attribution_occlusion_spearman": correlation,
                    "attribution_occlusion_p": correlation_p,
                    "top_deletion_auc": aucs["highest"],
                    "random_deletion_auc": aucs["random"],
                    "low_deletion_auc": aucs["lowest"],
                    "top_minus_random_auc": aucs["highest"] - aucs["random"],
                    "original_probability": occlusion["original_probability"],
                    "mean_occlusion_logit_drop": float(
                        occlusion["logit_drop"].mean()
                    ),
                    "max_occlusion_logit_drop": float(
                        occlusion["logit_drop"].max()
                    ),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    curves = pd.concat(curve_frames, ignore_index=True)
    occlusion_scores = pd.concat(occlusion_frames, ignore_index=True)
    metrics.to_csv(output_dir / "cnn_attribution_metrics.csv", index=False)
    curves.to_csv(output_dir / "cnn_deletion_curves.csv", index=False)
    occlusion_scores.to_csv(
        output_dir / "cnn_patch_occlusion_scores.csv",
        index=False,
    )
    return metrics, curves


def evaluate_uni_seed_stability(
    model: UNIClassifier,
    image: torch.Tensor,
    checkpoints_by_seed: dict[int, Path | str],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Measure prediction-conditioned gradient-attribution stability across heads."""
    maps_by_seed: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for seed, checkpoint_path in sorted(checkpoints_by_seed.items()):
        load_classifier_head(model, checkpoint_path, map_location=device)
        result = transformer_attributions(model, image)
        maps_by_seed[seed] = result["gradient_attention_rollout"].cpu()
        rows.append(
            {
                "seed": seed,
                "predicted_class": result["predicted_class"],
                "target_probability": result["target_probability"],
            }
        )

    predictions = pd.DataFrame(rows)
    majority_class = int(predictions["predicted_class"].mode().iloc[0])
    stability = pairwise_attribution_stability(maps_by_seed)
    stability["majority_predicted_class"] = majority_class
    stability["prediction_agreement"] = float(
        (predictions["predicted_class"] == majority_class).mean()
    )
    return predictions, stability
