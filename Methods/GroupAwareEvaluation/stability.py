"""Cross-seed prediction and attribution stability on the frozen cohort."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

from Methods.BaselineCNN.model import BaselineCNN
from Methods.UNIAttribution.attribution import (
    GradCAM,
    standardize_attribution_grid,
    transformer_attributions,
)
from Methods.UNIAttribution.data import UNI_IMAGE_SIZE
from Methods.UNIAttribution.evaluation import load_normalized_image
from Methods.UNIAttribution.model import UNIClassifier, load_classifier_head


def _flatten_map(attribution: torch.Tensor) -> np.ndarray:
    return attribution.detach().cpu().flatten().numpy()


def _map_rows(
    cohort_id: str,
    path: str,
    case_id: str,
    label: int,
    model_name: str,
    fold: int,
    seed: int,
    map_target: str,
    target_class: int,
    attribution: torch.Tensor,
) -> pd.DataFrame:
    values = attribution.detach().cpu().numpy()
    grid_size = values.shape[-1]
    return pd.DataFrame(
        {
            "cohort_id": cohort_id,
            "path": path,
            "case_id": case_id,
            "true_class": label,
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "map_target": map_target,
            "target_class": target_class,
            "patch_row": np.repeat(np.arange(grid_size), grid_size),
            "patch_column": np.tile(np.arange(grid_size), grid_size),
            "attribution": values.flatten(),
        }
    )


def _summarize_stability(
    predictions: pd.DataFrame,
    maps: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    for cohort_id, prediction_frame in predictions.groupby("cohort_id"):
        prediction_frame = prediction_frame.sort_values("seed")
        model_name = str(prediction_frame["model"].iloc[0])
        seed_pairs = list(combinations(prediction_frame["seed"].tolist(), 2))
        predictions_by_seed = prediction_frame.set_index("seed")
        image_pair_rows: list[dict[str, object]] = []
        for first_seed, second_seed in seed_pairs:
            first_prediction = int(
                predictions_by_seed.loc[first_seed, "predicted_class"]
            )
            second_prediction = int(
                predictions_by_seed.loc[second_seed, "predicted_class"]
            )
            same_prediction = first_prediction == second_prediction

            common_maps = maps[
                (maps["cohort_id"] == cohort_id)
                & (maps["map_target"] == "common_true_class")
                & (maps["seed"].isin((first_seed, second_seed)))
            ]
            common_pivot = common_maps.pivot(
                index=["patch_row", "patch_column"],
                columns="seed",
                values="attribution",
            )
            common_correlation = float(
                spearmanr(
                    common_pivot[first_seed],
                    common_pivot[second_seed],
                ).statistic
            )
            predicted_correlation = np.nan
            if same_prediction:
                predicted_maps = maps[
                    (maps["cohort_id"] == cohort_id)
                    & (maps["map_target"] == "predicted_class")
                    & (maps["seed"].isin((first_seed, second_seed)))
                ]
                predicted_pivot = predicted_maps.pivot(
                    index=["patch_row", "patch_column"],
                    columns="seed",
                    values="attribution",
                )
                predicted_correlation = float(
                    spearmanr(
                        predicted_pivot[first_seed],
                        predicted_pivot[second_seed],
                    ).statistic
                )
            item = {
                "cohort_id": cohort_id,
                "path": prediction_frame["path"].iloc[0],
                "case_id": prediction_frame["case_id"].iloc[0],
                "class_name": prediction_frame["class_name"].iloc[0],
                "true_class": int(prediction_frame["true_class"].iloc[0]),
                "model": model_name,
                "fold": int(prediction_frame["fold"].iloc[0]),
                "cohort_stratum": prediction_frame["cohort_stratum"].iloc[0],
                "seed_a": int(first_seed),
                "seed_b": int(second_seed),
                "same_predicted_class": same_prediction,
                "predicted_class_spearman": predicted_correlation,
                "common_true_class_spearman": common_correlation,
            }
            pair_rows.append(item)
            image_pair_rows.append(item)

        predicted_values = prediction_frame["predicted_class"]
        mode_count = int(predicted_values.value_counts().iloc[0])
        pair_frame = pd.DataFrame(image_pair_rows)
        image_rows.append(
            {
                "cohort_id": cohort_id,
                "path": prediction_frame["path"].iloc[0],
                "case_id": prediction_frame["case_id"].iloc[0],
                "class_name": prediction_frame["class_name"].iloc[0],
                "true_class": int(prediction_frame["true_class"].iloc[0]),
                "model": model_name,
                "fold": int(prediction_frame["fold"].iloc[0]),
                "cohort_stratum": prediction_frame["cohort_stratum"].iloc[0],
                "seed_count": len(prediction_frame),
                "correct_rate_across_seeds": float(
                    prediction_frame["correct"].mean()
                ),
                "all_seeds_correct": bool(prediction_frame["correct"].all()),
                "any_seed_incorrect": bool((~prediction_frame["correct"]).any()),
                "modal_prediction_agreement": mode_count / len(prediction_frame),
                "pairwise_prediction_agreement": float(
                    pair_frame["same_predicted_class"].mean()
                ),
                "mean_predicted_class_spearman_same_prediction": float(
                    pair_frame.loc[
                        pair_frame["same_predicted_class"],
                        "predicted_class_spearman",
                    ].mean()
                ),
                "mean_common_true_class_spearman": float(
                    pair_frame["common_true_class_spearman"].mean()
                ),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(image_rows)


def _save_stability(
    output_dir: Path,
    prefix: str,
    predictions: list[dict[str, object]],
    maps: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frame = pd.DataFrame(predictions)
    map_frame = pd.concat(maps, ignore_index=True)
    pair_frame, image_frame = _summarize_stability(
        prediction_frame,
        map_frame,
    )
    prediction_frame.to_csv(
        output_dir / f"{prefix}_stability_predictions.csv",
        index=False,
    )
    map_frame.to_csv(
        output_dir / f"{prefix}_stability_attribution_maps.csv",
        index=False,
    )
    pair_frame.to_csv(
        output_dir / f"{prefix}_stability_seed_pairs.csv",
        index=False,
    )
    image_frame.to_csv(
        output_dir / f"{prefix}_stability_per_image.csv",
        index=False,
    )
    return prediction_frame, pair_frame, image_frame


def evaluate_transformer_stability(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seeds: Iterable[int],
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
    model_name: str = "UNI",
    output_prefix: str = "uni",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure Transformer gradient-rollout stability without mixing targets."""
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2:
        raise ValueError("Stability evaluation requires at least two seeds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, object]] = []
    map_frames: list[pd.DataFrame] = []
    for fold, fold_cohort in cohort.groupby("fold"):
        images = {
            row["cohort_id"]: load_normalized_image(
                row["path"],
                UNI_IMAGE_SIZE,
                device,
                image_transform=image_transform,
            )
            for _, row in fold_cohort.iterrows()
        }
        for seed in seeds:
            checkpoint = Path(checkpoint_dir) / f"seed_{seed}" / f"fold_{fold}.pt"
            load_classifier_head(model, checkpoint, map_location=device)
            for _, row in fold_cohort.iterrows():
                image = images[row["cohort_id"]]
                predicted_result = transformer_attributions(model, image)
                predicted_class = int(predicted_result["predicted_class"])
                true_class = int(row["label"])
                if predicted_class == true_class:
                    common_result = predicted_result
                else:
                    common_result = transformer_attributions(
                        model,
                        image,
                        target_class=true_class,
                    )
                prediction_rows.append(
                    {
                        "cohort_id": row["cohort_id"],
                        "path": row["path"],
                        "case_id": row["case_id"],
                        "class_name": row["class_name"],
                        "true_class": true_class,
                        "model": model_name,
                        "fold": int(fold),
                        "seed": int(seed),
                        "predicted_class": predicted_class,
                        "correct": predicted_class == true_class,
                        "confidence": predicted_result["target_probability"],
                        "cohort_stratum": row["cohort_stratum"],
                    }
                )
                map_frames.append(
                    _map_rows(
                        row["cohort_id"],
                        row["path"],
                        row["case_id"],
                        true_class,
                        model_name,
                        int(fold),
                        int(seed),
                        "predicted_class",
                        predicted_class,
                        standardize_attribution_grid(
                            predicted_result["gradient_attention_rollout"],
                            image_size=tuple(image.shape[-2:]),
                        ),
                    )
                )
                map_frames.append(
                    _map_rows(
                        row["cohort_id"],
                        row["path"],
                        row["case_id"],
                        true_class,
                        model_name,
                        int(fold),
                        int(seed),
                        "common_true_class",
                        true_class,
                        standardize_attribution_grid(
                            common_result["gradient_attention_rollout"],
                            image_size=tuple(image.shape[-2:]),
                        ),
                    )
                )
    return _save_stability(
        output_dir,
        output_prefix,
        prediction_rows,
        map_frames,
    )


def evaluate_uni_stability(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seeds: Iterable[int],
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return evaluate_transformer_stability(
        model,
        cohort,
        checkpoint_dir,
        seeds,
        device,
        output_dir,
        image_transform,
        model_name="UNI",
        output_prefix="uni",
    )


def evaluate_dinov2_stability(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seeds: Iterable[int],
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return evaluate_transformer_stability(
        model,
        cohort,
        checkpoint_dir,
        seeds,
        device,
        output_dir,
        image_transform,
        model_name="DINOv2",
        output_prefix="dinov2",
    )


def evaluate_cnn_stability(
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seeds: Iterable[int],
    device: torch.device,
    output_dir: Path | str,
    num_classes: int = 8,
    model_builder: Callable[[], nn.Module] | None = None,
    target_layer_getter: Callable[[nn.Module], nn.Module] | None = None,
    model_name: str = "CNN",
    output_prefix: str = "cnn",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Measure CNN Grad-CAM stability with conditional predicted-class maps."""
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2:
        raise ValueError("Stability evaluation requires at least two seeds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_builder is None:
        model_builder = lambda: BaselineCNN(in_channels=3, num_classes=num_classes)
    if target_layer_getter is None:
        target_layer_getter = lambda model: model.features[-1][3]
    prediction_rows: list[dict[str, object]] = []
    map_frames: list[pd.DataFrame] = []
    for fold, fold_cohort in cohort.groupby("fold"):
        images = {
            row["cohort_id"]: load_normalized_image(row["path"], 150, device)
            for _, row in fold_cohort.iterrows()
        }
        for seed in seeds:
            model = model_builder().to(device)
            checkpoint = Path(checkpoint_dir) / f"seed_{seed}" / f"fold_{fold}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.eval()
            with GradCAM(model, target_layer_getter(model)) as gradcam:
                for _, row in fold_cohort.iterrows():
                    image = images[row["cohort_id"]]
                    predicted_result = gradcam.attribute(
                        image,
                        output_grid_size=int(image.shape[-1]),
                    )
                    predicted_class = int(predicted_result["predicted_class"])
                    true_class = int(row["label"])
                    if predicted_class == true_class:
                        common_result = predicted_result
                    else:
                        common_result = gradcam.attribute(
                            image,
                            target_class=true_class,
                            output_grid_size=int(image.shape[-1]),
                        )
                    prediction_rows.append(
                        {
                            "cohort_id": row["cohort_id"],
                            "path": row["path"],
                            "case_id": row["case_id"],
                            "class_name": row["class_name"],
                            "true_class": true_class,
                            "model": model_name,
                            "fold": int(fold),
                            "seed": int(seed),
                            "predicted_class": predicted_class,
                            "correct": predicted_class == true_class,
                            "confidence": predicted_result["target_probability"],
                            "cohort_stratum": row["cohort_stratum"],
                        }
                    )
                    map_frames.append(
                        _map_rows(
                            row["cohort_id"],
                            row["path"],
                            row["case_id"],
                            true_class,
                            model_name,
                            int(fold),
                            int(seed),
                            "predicted_class",
                            predicted_class,
                            standardize_attribution_grid(
                                predicted_result["gradcam"],
                                image_size=tuple(image.shape[-2:]),
                            ),
                        )
                    )
                    map_frames.append(
                        _map_rows(
                            row["cohort_id"],
                            row["path"],
                            row["case_id"],
                            true_class,
                            model_name,
                            int(fold),
                            int(seed),
                            "common_true_class",
                            true_class,
                            standardize_attribution_grid(
                                common_result["gradcam"],
                                image_size=tuple(image.shape[-2:]),
                            ),
                        )
                    )
            del model
    return _save_stability(
        output_dir,
        output_prefix,
        prediction_rows,
        map_frames,
    )
