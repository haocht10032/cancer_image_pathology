"""Intervention-versus-gradient attribution on the frozen UNI OOF cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from Methods.GroupAwareEvaluation.faithfulness import (
    _decorate,
    _faithfulness_row,
    _logit_deletion_curves,
    _map_frame,
    _patch_occlusion_reference,
    _representative_heatmap_ids,
    _seed_for_image,
    _target_roles,
)
from Methods.GroupAwareEvaluation.stability import (
    _map_rows,
    _summarize_stability,
)
from Methods.UNIAttribution.attribution import (
    normalize_map,
    save_attribution_figure,
    standardize_attribution_grid,
    transformer_attributions,
)
from Methods.UNIAttribution.data import UNI_GRID_SIZE, UNI_IMAGE_SIZE
from Methods.UNIAttribution.evaluation import load_normalized_image
from Methods.UNIAttribution.intervention import intervention_patch_attribution
from Methods.UNIAttribution.model import UNIClassifier, load_classifier_head


INTERVENTION_METHOD = "intervention_activation_patch"
GRADIENT_METHOD = "gradient_attention_rollout"
COMPARISON_METHODS = (GRADIENT_METHOD, INTERVENTION_METHOD)


def _intervention_effect_frame(
    result: dict[str, object],
    row: pd.Series,
    target_role: str,
    target_class: int,
    seed: int,
) -> pd.DataFrame:
    effects = result["intervention_effect"].detach().cpu().numpy()
    margin_effects = result["margin_intervention_effect"].detach().cpu().numpy()
    target_logits = result["intervened_target_logits"].detach().cpu().numpy()
    target_margins = result["intervened_target_margins"].detach().cpu().numpy()
    grid_size = effects.shape[-1]
    frame = pd.DataFrame(
        {
            "patch_index": np.arange(grid_size**2),
            "patch_row": np.repeat(np.arange(grid_size), grid_size),
            "patch_column": np.tile(np.arange(grid_size), grid_size),
            "target_logit_intervention_effect": effects.flatten(),
            "target_margin_intervention_effect": margin_effects.flatten(),
            "intervened_target_logit": target_logits,
            "intervened_target_margin": target_margins,
            "baseline_target_logit": result["baseline_target_logit"],
            "baseline_target_margin": result["baseline_target_margin"],
            "intervention_layer_index": result["layer_index"],
            "intervention_layer_number": result["layer_number"],
            "transformer_block_count": result["block_count"],
            "replacement": result["replacement"],
            "effect_definition": result["effect_definition"],
            "interventions_independent": result["interventions_independent"],
        }
    )
    return _decorate(
        frame,
        row,
        "UNI",
        INTERVENTION_METHOD,
        target_role,
        target_class,
        seed,
    )


def _save_intervention_outputs(
    output_dir: Path,
    metrics: list[dict[str, object]],
    curves: list[pd.DataFrame],
    maps: list[pd.DataFrame],
    occlusions: list[pd.DataFrame],
    effects: list[pd.DataFrame],
    metadata: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frame = pd.DataFrame(metrics)
    curve_frame = pd.concat(curves, ignore_index=True)
    pd.concat(maps, ignore_index=True).to_csv(
        output_dir / "uni_intervention_attribution_maps.csv", index=False
    )
    pd.concat(occlusions, ignore_index=True).to_csv(
        output_dir / "uni_input_patch_occlusion_scores.csv", index=False
    )
    pd.concat(effects, ignore_index=True).to_csv(
        output_dir / "uni_intervention_effects.csv", index=False
    )
    metric_frame.to_csv(
        output_dir / "uni_intervention_faithfulness_metrics.csv", index=False
    )
    curve_frame.to_csv(
        output_dir / "uni_intervention_deletion_curves.csv", index=False
    )
    with (output_dir / "intervention_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    return metric_frame, curve_frame


def evaluate_uni_intervention_faithfulness(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seed: int,
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
    intervention_layer: int = -2,
    replacement: str = "image_patch_mean",
    intervention_batch_size: int = 64,
    random_repeats: int = 20,
    heatmap_limit: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare UNI gradient rollout and activation patching on one cohort."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    curves_out: list[pd.DataFrame] = []
    maps_out: list[pd.DataFrame] = []
    occlusions_out: list[pd.DataFrame] = []
    effects_out: list[pd.DataFrame] = []
    heatmap_ids = _representative_heatmap_ids(cohort, heatmap_limit)
    resolved_layer: int | None = None
    block_count: int | None = None

    for fold, fold_cohort in cohort.groupby("fold"):
        checkpoint = Path(checkpoint_dir) / f"seed_{seed}" / f"fold_{fold}.pt"
        load_classifier_head(model, checkpoint, map_location=device)
        for _, row in fold_cohort.iterrows():
            image = load_normalized_image(
                row["path"],
                UNI_IMAGE_SIZE,
                device,
                image_transform=image_transform,
            )
            expected_prediction = int(row["uni_prediction"])
            true_class = int(row["label"])
            display_maps: dict[str, torch.Tensor] = {}
            display_occlusion: torch.Tensor | None = None

            for target_role, target_class in _target_roles(
                true_class, expected_prediction
            ):
                gradient_result = transformer_attributions(
                    model, image, target_class=target_class
                )
                intervention_result = intervention_patch_attribution(
                    model,
                    image,
                    target_class=target_class,
                    layer=intervention_layer,
                    replacement=replacement,
                    intervention_batch_size=intervention_batch_size,
                )
                if int(gradient_result["predicted_class"]) != expected_prediction:
                    raise RuntimeError(
                        "UNI checkpoint prediction does not match the frozen cohort "
                        f"for {row['relative_path']}"
                    )
                if int(intervention_result["predicted_class"]) != expected_prediction:
                    raise RuntimeError(
                        "UNI intervention baseline prediction does not match the frozen "
                        f"cohort for {row['relative_path']}"
                    )
                resolved_layer = int(intervention_result["layer_index"])
                block_count = int(intervention_result["block_count"])

                occlusion = _patch_occlusion_reference(
                    model,
                    image,
                    target_class,
                    expected_prediction,
                    true_class,
                )
                occlusion = _decorate(
                    occlusion,
                    row,
                    "UNI",
                    "input_patch_occlusion",
                    target_role,
                    target_class,
                    seed,
                )
                occlusions_out.append(occlusion)
                effects_out.append(
                    _intervention_effect_frame(
                        intervention_result,
                        row,
                        target_role,
                        target_class,
                        seed,
                    )
                )
                if target_role == "predicted":
                    occlusion_grid = torch.from_numpy(
                        occlusion["target_class_logit_drop"]
                        .to_numpy()
                        .reshape(UNI_GRID_SIZE, UNI_GRID_SIZE)
                    )
                    display_occlusion = normalize_map(occlusion_grid)

                native_maps = {
                    GRADIENT_METHOD: gradient_result[GRADIENT_METHOD],
                    INTERVENTION_METHOD: intervention_result["intervention_effect"],
                }
                for method, native_attribution in native_maps.items():
                    attribution = standardize_attribution_grid(
                        native_attribution,
                        image_size=tuple(image.shape[-2:]),
                        output_grid_size=UNI_GRID_SIZE,
                    )
                    curves = _logit_deletion_curves(
                        model,
                        image,
                        attribution,
                        target_class,
                        expected_prediction,
                        true_class,
                        random_seed=_seed_for_image(
                            seed, row["cohort_id"], target_role
                        ),
                        random_repeats=random_repeats,
                    )
                    curves = _decorate(
                        curves,
                        row,
                        "UNI",
                        method,
                        target_role,
                        target_class,
                        seed,
                    )
                    curves_out.append(curves)
                    maps_out.append(
                        _map_frame(
                            attribution,
                            row["cohort_id"],
                            row["path"],
                            "UNI",
                            method,
                            target_role,
                            target_class,
                            int(fold),
                            seed,
                            native_grid_size=int(native_attribution.shape[-1]),
                        )
                    )
                    metric = _faithfulness_row(curves, attribution, occlusion)
                    metrics.append(
                        {
                            key: curves.iloc[0][key]
                            for key in (
                                "cohort_id",
                                "path",
                                "relative_path",
                                "case_id",
                                "class_name",
                                "true_class",
                                "model",
                                "method",
                                "target_role",
                                "target_class",
                                "correct",
                                "confidence_group",
                                "fold",
                                "seed",
                            )
                        }
                        | metric
                    )
                    if target_role == "predicted":
                        display_maps[method] = attribution

            if str(row["cohort_id"]) in heatmap_ids and display_occlusion is not None:
                display_maps["input_patch_occlusion"] = display_occlusion
                status = "correct" if bool(row["uni_correct"]) else "incorrect"
                display_maps[INTERVENTION_METHOD] = normalize_map(
                    display_maps[INTERVENTION_METHOD]
                )
                save_attribution_figure(
                    image,
                    display_maps,
                    output_dir / "heatmaps" / f"{row['cohort_id']}_{status}.png",
                    (
                        f"{status}: true={row['class_name']}, "
                        f"pred={row['uni_predicted_class_name']}"
                    ),
                )

    metadata = {
        "model": "UNI",
        "classifier_training_changed": False,
        "cohort_changed": False,
        "intervention_layer_requested": intervention_layer,
        "intervention_layer_index": resolved_layer,
        "intervention_layer_number_one_based": (
            None if resolved_layer is None else resolved_layer + 1
        ),
        "transformer_block_count": block_count,
        "replacement": replacement,
        "replacement_definition": (
            "mean contextualized patch-token activation from the same image at "
            "the intervention layer"
        ),
        "interventions_independent": True,
        "effect_definition": (
            "baseline target-class logit minus target-class logit after replacing "
            "one contextualized patch token"
        ),
        "positive_effect_interpretation": (
            "the original patch token supports the model target score relative to "
            "the within-image mean replacement"
        ),
        "negative_effect_interpretation": (
            "the original patch token suppresses the model target score relative to "
            "the within-image mean replacement"
        ),
        "biological_causality_claimed": False,
    }
    return _save_intervention_outputs(
        output_dir,
        metrics,
        curves_out,
        maps_out,
        occlusions_out,
        effects_out,
        metadata,
    )


def _stability_map(
    row: pd.Series,
    seed: int,
    fold: int,
    method: str,
    map_target: str,
    target_class: int,
    attribution: torch.Tensor,
) -> pd.DataFrame:
    frame = _map_rows(
        row["cohort_id"],
        row["path"],
        row["case_id"],
        int(row["label"]),
        "UNI",
        fold,
        seed,
        map_target,
        target_class,
        attribution,
    )
    frame.insert(5, "method", method)
    return frame


def evaluate_uni_intervention_stability(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seeds: Iterable[int],
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
    intervention_layer: int = -2,
    replacement: str = "image_patch_mean",
    intervention_batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare gradient and intervention map stability without mixing targets."""
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2:
        raise ValueError("Stability evaluation requires at least two seeds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, object]] = []
    maps: list[pd.DataFrame] = []

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
                true_class = int(row["label"])
                predicted_gradient = transformer_attributions(model, image)
                predicted_class = int(predicted_gradient["predicted_class"])
                predicted_intervention = intervention_patch_attribution(
                    model,
                    image,
                    target_class=predicted_class,
                    layer=intervention_layer,
                    replacement=replacement,
                    intervention_batch_size=intervention_batch_size,
                )
                if predicted_class == true_class:
                    common_gradient = predicted_gradient
                    common_intervention = predicted_intervention
                else:
                    common_gradient = transformer_attributions(
                        model, image, target_class=true_class
                    )
                    common_intervention = intervention_patch_attribution(
                        model,
                        image,
                        target_class=true_class,
                        layer=intervention_layer,
                        replacement=replacement,
                        intervention_batch_size=intervention_batch_size,
                    )

                predictions.append(
                    {
                        "cohort_id": row["cohort_id"],
                        "path": row["path"],
                        "case_id": row["case_id"],
                        "class_name": row["class_name"],
                        "true_class": true_class,
                        "model": "UNI",
                        "fold": int(fold),
                        "seed": seed,
                        "predicted_class": predicted_class,
                        "correct": predicted_class == true_class,
                        "confidence": predicted_gradient["target_probability"],
                        "cohort_stratum": row["cohort_stratum"],
                    }
                )
                method_maps = {
                    GRADIENT_METHOD: (
                        predicted_gradient[GRADIENT_METHOD],
                        common_gradient[GRADIENT_METHOD],
                    ),
                    INTERVENTION_METHOD: (
                        predicted_intervention["intervention_effect"],
                        common_intervention["intervention_effect"],
                    ),
                }
                for method, (predicted_map, common_map) in method_maps.items():
                    maps.append(
                        _stability_map(
                            row,
                            seed,
                            int(fold),
                            method,
                            "predicted_class",
                            predicted_class,
                            standardize_attribution_grid(
                                predicted_map,
                                image_size=tuple(image.shape[-2:]),
                            ),
                        )
                    )
                    maps.append(
                        _stability_map(
                            row,
                            seed,
                            int(fold),
                            method,
                            "common_true_class",
                            true_class,
                            standardize_attribution_grid(
                                common_map,
                                image_size=tuple(image.shape[-2:]),
                            ),
                        )
                    )

    prediction_frame = pd.DataFrame(predictions)
    map_frame = pd.concat(maps, ignore_index=True)
    pair_frames: list[pd.DataFrame] = []
    image_frames: list[pd.DataFrame] = []
    for method, method_maps in map_frame.groupby("method"):
        pair_frame, image_frame = _summarize_stability(
            prediction_frame, method_maps
        )
        pair_frame.insert(6, "method", method)
        image_frame.insert(6, "method", method)
        pair_frames.append(pair_frame)
        image_frames.append(image_frame)
    pair_frame = pd.concat(pair_frames, ignore_index=True)
    image_frame = pd.concat(image_frames, ignore_index=True)
    prediction_frame.to_csv(
        output_dir / "uni_intervention_stability_predictions.csv", index=False
    )
    map_frame.to_csv(
        output_dir / "uni_intervention_stability_attribution_maps.csv", index=False
    )
    pair_frame.to_csv(
        output_dir / "uni_intervention_stability_seed_pairs.csv", index=False
    )
    image_frame.to_csv(
        output_dir / "uni_intervention_stability_per_image.csv", index=False
    )
    return prediction_frame, pair_frame, image_frame
