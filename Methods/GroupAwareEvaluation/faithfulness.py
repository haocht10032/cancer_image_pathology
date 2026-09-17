"""Paired, logit-based faithfulness evaluation on frozen OOF models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

from Methods.BaselineCNN.model import BaselineCNN
from Methods.UNIAttribution.attribution import (
    GradCAM,
    _mask_patch_indices,
    normalize_map,
    save_attribution_figure,
    standardize_attribution_grid,
    transformer_attributions,
)
from Methods.UNIAttribution.data import UNI_GRID_SIZE, UNI_IMAGE_SIZE
from Methods.UNIAttribution.evaluation import load_normalized_image
from Methods.UNIAttribution.model import UNIClassifier, load_classifier_head


DELETION_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
UNI_METHODS = (
    "raw_attention",
    "attention_rollout",
    "gradient_attention_rollout",
)


def _target_margin(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    competitors = logits.clone()
    competitors[:, target_class] = -torch.inf
    return logits[:, target_class] - competitors.max(dim=1).values


def _score_batch(
    logits: torch.Tensor,
    target_class: int,
    predicted_class: int,
    true_class: int,
) -> dict[str, torch.Tensor]:
    return {
        "target_class_logit": logits[:, target_class],
        "predicted_class_logit": logits[:, predicted_class],
        "true_class_logit": logits[:, true_class],
        "target_margin": _target_margin(logits, target_class),
    }


@torch.inference_mode()
def _patch_occlusion_reference(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    predicted_class: int,
    true_class: int,
    grid_size: int = UNI_GRID_SIZE,
    batch_size: int = 64,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    baseline_logits = model(image)
    baseline = {
        name: float(value[0].cpu())
        for name, value in _score_batch(
            baseline_logits,
            target_class,
            predicted_class,
            true_class,
        ).items()
    }
    batches: list[torch.Tensor] = []
    patch_indices = list(range(grid_size**2))
    for start in range(0, len(patch_indices), batch_size):
        indices = patch_indices[start : start + batch_size]
        masked = torch.cat(
            [
                _mask_patch_indices(
                    image,
                    [index],
                    grid_size=grid_size,
                    fill_value=fill_value,
                )
                for index in indices
            ],
            dim=0,
        )
        batches.append(model(masked).cpu())
    logits = torch.cat(batches)
    scores = _score_batch(logits, target_class, predicted_class, true_class)
    frame = pd.DataFrame(
        {
            "patch_index": patch_indices,
            "patch_row": np.repeat(np.arange(grid_size), grid_size),
            "patch_column": np.tile(np.arange(grid_size), grid_size),
        }
    )
    for name, values in scores.items():
        frame[name] = values.numpy()
        frame[f"{name}_drop"] = baseline[name] - frame[name]
    return frame


@torch.inference_mode()
def _logit_deletion_curves(
    model: nn.Module,
    image: torch.Tensor,
    attribution: torch.Tensor,
    target_class: int,
    predicted_class: int,
    true_class: int,
    random_seed: int,
    random_repeats: int,
    fractions: tuple[float, ...] = DELETION_FRACTIONS,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    grid_size = int(attribution.shape[-1])
    patch_count = grid_size**2
    values = attribution.detach().cpu().flatten().numpy()
    highest = np.argsort(-values)
    lowest = np.argsort(values)
    rng = np.random.default_rng(random_seed)
    strategies = {
        "top": [highest],
        "bottom": [lowest],
        "random": [rng.permutation(patch_count) for _ in range(random_repeats)],
    }
    rows: list[dict[str, object]] = []
    for strategy, orders in strategies.items():
        for fraction in fractions:
            remove_count = min(
                patch_count,
                int(round(float(fraction) * patch_count)),
            )
            masked = torch.cat(
                [
                    _mask_patch_indices(
                        image,
                        order[:remove_count],
                        grid_size=grid_size,
                        fill_value=fill_value,
                    )
                    for order in orders
                ],
                dim=0,
            )
            score_columns = _score_batch(
                model(masked),
                target_class,
                predicted_class,
                true_class,
            )
            row: dict[str, object] = {
                "strategy": strategy,
                "fraction_removed": float(fraction),
                "patches_removed": remove_count,
                "repeat_count": len(orders),
            }
            for name, score in score_columns.items():
                row[name] = float(score.mean().cpu())
                row[f"{name}_std"] = float(score.std(unbiased=False).cpu())
            rows.append(row)
    curves = pd.DataFrame(rows)
    for strategy, indices in curves.groupby("strategy").groups.items():
        baseline = curves.loc[indices].sort_values("fraction_removed").iloc[0]
        for name in (
            "target_class_logit",
            "predicted_class_logit",
            "true_class_logit",
            "target_margin",
        ):
            curves.loc[indices, f"{name}_drop"] = (
                float(baseline[name]) - curves.loc[indices, name]
            )
    return curves


def _auc(frame: pd.DataFrame, value_column: str) -> float:
    ordered = frame.sort_values("fraction_removed")
    return float(
        np.trapz(
            ordered[value_column].to_numpy(),
            ordered["fraction_removed"].to_numpy(),
        )
    )


def _faithfulness_row(
    curves: pd.DataFrame,
    attribution: torch.Tensor,
    occlusion: pd.DataFrame,
) -> dict[str, float]:
    correlation = spearmanr(
        attribution.detach().cpu().flatten().numpy(),
        occlusion["target_class_logit_drop"].to_numpy(),
    )
    by_strategy = {
        strategy: frame
        for strategy, frame in curves.groupby("strategy")
    }
    logit_auc = {
        strategy: _auc(frame, "target_class_logit")
        for strategy, frame in by_strategy.items()
    }
    margin_auc = {
        strategy: _auc(frame, "target_margin")
        for strategy, frame in by_strategy.items()
    }
    result = {
        "attribution_occlusion_spearman": float(correlation.statistic),
        "attribution_occlusion_p": float(correlation.pvalue),
        "top_target_logit_auc": logit_auc["top"],
        "random_target_logit_auc": logit_auc["random"],
        "bottom_target_logit_auc": logit_auc["bottom"],
        "top_minus_random_target_logit_auc": (
            logit_auc["top"] - logit_auc["random"]
        ),
        "top_margin_auc": margin_auc["top"],
        "random_margin_auc": margin_auc["random"],
        "bottom_margin_auc": margin_auc["bottom"],
        "top_minus_random_margin_auc": margin_auc["top"] - margin_auc["random"],
    }
    top = by_strategy["top"].set_index("fraction_removed")
    for fraction in (0.05, 0.10, 0.20, 0.30):
        suffix = int(round(fraction * 100))
        row = top.loc[fraction]
        result[f"top_{suffix}_target_logit_drop"] = float(
            row["target_class_logit_drop"]
        )
        result[f"top_{suffix}_margin_drop"] = float(row["target_margin_drop"])
    return result


def _map_frame(
    attribution: torch.Tensor,
    cohort_id: str,
    path: str,
    model_name: str,
    method: str,
    target_role: str,
    target_class: int,
    fold: int,
    seed: int,
    native_grid_size: int | None = None,
) -> pd.DataFrame:
    values = attribution.detach().cpu().numpy()
    grid_size = values.shape[-1]
    return pd.DataFrame(
        {
            "cohort_id": cohort_id,
            "path": path,
            "model": model_name,
            "method": method,
            "target_role": target_role,
            "target_class": target_class,
            "fold": fold,
            "seed": seed,
            "map_space": "shared_image_grid",
            "native_grid_size": (
                grid_size if native_grid_size is None else native_grid_size
            ),
            "evaluation_grid_size": grid_size,
            "patch_row": np.repeat(np.arange(grid_size), grid_size),
            "patch_column": np.tile(np.arange(grid_size), grid_size),
            "attribution": values.flatten(),
        }
    )


def _target_roles(true_class: int, predicted_class: int) -> list[tuple[str, int]]:
    roles = [("predicted", predicted_class)]
    if true_class != predicted_class:
        roles.append(("true", true_class))
    return roles


def _seed_for_image(base_seed: int, cohort_id: str, target_role: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{cohort_id}:{target_role}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def _representative_heatmap_ids(
    cohort: pd.DataFrame,
    limit: int,
) -> set[str]:
    if limit <= 0:
        return set()
    ordered = cohort.sort_values(
        ["label", "cohort_stratum", "case_id", "relative_path"]
    ).copy()
    ordered["_stratum_rank"] = ordered.groupby(
        ["label", "cohort_stratum"]
    ).cumcount()
    ordered = ordered.sort_values(
        ["_stratum_rank", "label", "cohort_stratum", "case_id"]
    )
    return set(ordered.head(limit)["cohort_id"].astype(str))


def _decorate(
    frame: pd.DataFrame,
    row: pd.Series,
    model_name: str,
    method: str,
    target_role: str,
    target_class: int,
    seed: int,
) -> pd.DataFrame:
    result = frame.copy()
    values = {
        "cohort_id": row["cohort_id"],
        "path": row["path"],
        "relative_path": row["relative_path"],
        "case_id": row["case_id"],
        "class_name": row["class_name"],
        "true_class": int(row["label"]),
        "model": model_name,
        "method": method,
        "target_role": target_role,
        "target_class": target_class,
        "correct": bool(row[f"{model_name.lower()}_correct"]),
        "confidence_group": row["cohort_stratum"],
        "fold": int(row["fold"]),
        "seed": int(seed),
    }
    for key, value in reversed(list(values.items())):
        result.insert(0, key, value)
    return result


def _save_frames(
    output_dir: Path,
    prefix: str,
    metrics: list[dict[str, object]],
    curves: list[pd.DataFrame],
    maps: list[pd.DataFrame],
    occlusions: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frame = pd.DataFrame(metrics)
    curve_frame = pd.concat(curves, ignore_index=True)
    map_frame = pd.concat(maps, ignore_index=True)
    occlusion_frame = pd.concat(occlusions, ignore_index=True)
    metric_frame.to_csv(output_dir / f"{prefix}_faithfulness_metrics.csv", index=False)
    curve_frame.to_csv(output_dir / f"{prefix}_deletion_curves.csv", index=False)
    map_frame.to_csv(output_dir / f"{prefix}_attribution_maps.csv", index=False)
    occlusion_frame.to_csv(
        output_dir / f"{prefix}_patch_occlusion_scores.csv",
        index=False,
    )
    return metric_frame, curve_frame


def evaluate_transformer_faithfulness(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seed: int,
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
    random_repeats: int = 20,
    heatmap_limit: int = 32,
    model_name: str = "UNI",
    column_prefix: str = "uni",
    output_prefix: str = "uni",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate matched Transformer attributions on held-out images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    curves_out: list[pd.DataFrame] = []
    maps_out: list[pd.DataFrame] = []
    occlusions_out: list[pd.DataFrame] = []
    heatmap_ids = _representative_heatmap_ids(cohort, heatmap_limit)

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
            expected_prediction = int(row[f"{column_prefix}_prediction"])
            true_class = int(row["label"])
            display_maps: dict[str, torch.Tensor] = {}
            display_occlusion: torch.Tensor | None = None
            for target_role, target_class in _target_roles(
                true_class,
                expected_prediction,
            ):
                result = transformer_attributions(
                    model,
                    image,
                    target_class=target_class,
                )
                if int(result["predicted_class"]) != expected_prediction:
                    raise RuntimeError(
                        f"{model_name} checkpoint prediction does not match frozen OOF "
                        f"manifest for {row['relative_path']}"
                    )
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
                    model_name,
                    "patch_occlusion",
                    target_role,
                    target_class,
                    seed,
                )
                occlusions_out.append(occlusion)
                if target_role == "predicted":
                    grid = torch.from_numpy(
                        occlusion["target_class_logit_drop"].to_numpy().reshape(
                            UNI_GRID_SIZE,
                            UNI_GRID_SIZE,
                        )
                    )
                    display_occlusion = normalize_map(grid)

                for method in UNI_METHODS:
                    native_attribution = result[method]
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
                        random_seed=_seed_for_image(seed, row["cohort_id"], target_role),
                        random_repeats=random_repeats,
                    )
                    curves = _decorate(
                        curves,
                        row,
                        model_name,
                        method,
                        target_role,
                        target_class,
                        seed,
                    )
                    curves_out.append(curves)
                    map_frame = _map_frame(
                        attribution,
                        row["cohort_id"],
                        row["path"],
                        model_name,
                        method,
                        target_role,
                        target_class,
                        int(fold),
                        seed,
                        native_grid_size=int(native_attribution.shape[-1]),
                    )
                    maps_out.append(map_frame)
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

            if (
                str(row["cohort_id"]) in heatmap_ids
                and display_occlusion is not None
            ):
                display_maps["patch_occlusion"] = display_occlusion
                status = (
                    "correct"
                    if bool(row[f"{column_prefix}_correct"])
                    else "incorrect"
                )
                save_attribution_figure(
                    image,
                    display_maps,
                    output_dir
                    / "heatmaps"
                    / output_prefix
                    / f"{row['cohort_id']}_{status}.png",
                    (
                        f"{status}: true={row['class_name']}, "
                        f"pred={row[f'{column_prefix}_predicted_class_name']}"
                    ),
                )
    return _save_frames(
        output_dir,
        output_prefix,
        metrics,
        curves_out,
        maps_out,
        occlusions_out,
    )


def evaluate_uni_faithfulness(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seed: int,
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
    random_repeats: int = 20,
    heatmap_limit: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_transformer_faithfulness(
        model,
        cohort,
        checkpoint_dir,
        seed,
        device,
        output_dir,
        image_transform,
        random_repeats=random_repeats,
        heatmap_limit=heatmap_limit,
        model_name="UNI",
        column_prefix="uni",
        output_prefix="uni",
    )


def evaluate_dinov2_faithfulness(
    model: UNIClassifier,
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seed: int,
    device: torch.device,
    output_dir: Path | str,
    image_transform: Callable,
    random_repeats: int = 20,
    heatmap_limit: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluate_transformer_faithfulness(
        model,
        cohort,
        checkpoint_dir,
        seed,
        device,
        output_dir,
        image_transform,
        random_repeats=random_repeats,
        heatmap_limit=heatmap_limit,
        model_name="DINOv2",
        column_prefix="dinov2",
        output_prefix="dinov2",
    )


def evaluate_cnn_faithfulness(
    cohort: pd.DataFrame,
    checkpoint_dir: Path | str,
    seed: int,
    device: torch.device,
    output_dir: Path | str,
    num_classes: int = 8,
    random_repeats: int = 20,
    heatmap_limit: int = 32,
    model_builder: Callable[[], nn.Module] | None = None,
    target_layer_getter: Callable[[nn.Module], nn.Module] | None = None,
    model_name: str = "CNN",
    column_prefix: str = "cnn",
    output_prefix: str = "cnn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fold-matched CNN Grad-CAM on the identical frozen cohort."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_builder is None:
        model_builder = lambda: BaselineCNN(in_channels=3, num_classes=num_classes)
    if target_layer_getter is None:
        target_layer_getter = lambda model: model.features[-1][3]
    metrics: list[dict[str, object]] = []
    curves_out: list[pd.DataFrame] = []
    maps_out: list[pd.DataFrame] = []
    occlusions_out: list[pd.DataFrame] = []
    heatmap_ids = _representative_heatmap_ids(cohort, heatmap_limit)

    for fold, fold_cohort in cohort.groupby("fold"):
        model = model_builder().to(device)
        checkpoint = Path(checkpoint_dir) / f"seed_{seed}" / f"fold_{fold}.pt"
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        target_layer = target_layer_getter(model)
        with GradCAM(model, target_layer) as gradcam:
            for _, row in fold_cohort.iterrows():
                image = load_normalized_image(row["path"], 150, device)
                expected_prediction = int(row[f"{column_prefix}_prediction"])
                true_class = int(row["label"])
                display_map: torch.Tensor | None = None
                display_occlusion: torch.Tensor | None = None
                for target_role, target_class in _target_roles(
                    true_class,
                    expected_prediction,
                ):
                    result = gradcam.attribute(
                        image,
                        target_class=target_class,
                        output_grid_size=int(image.shape[-1]),
                    )
                    if int(result["predicted_class"]) != expected_prediction:
                        raise RuntimeError(
                            f"{model_name} checkpoint prediction does not match frozen OOF "
                            f"manifest for {row['relative_path']}"
                        )
                    native_attribution = result["gradcam"]
                    attribution = standardize_attribution_grid(
                        native_attribution,
                        image_size=tuple(image.shape[-2:]),
                        output_grid_size=UNI_GRID_SIZE,
                    )
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
                        model_name,
                        "patch_occlusion",
                        target_role,
                        target_class,
                        seed,
                    )
                    occlusions_out.append(occlusion)
                    curves = _logit_deletion_curves(
                        model,
                        image,
                        attribution,
                        target_class,
                        expected_prediction,
                        true_class,
                        random_seed=_seed_for_image(seed, row["cohort_id"], target_role),
                        random_repeats=random_repeats,
                    )
                    curves = _decorate(
                        curves,
                        row,
                        model_name,
                        "gradcam",
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
                            model_name,
                            "gradcam",
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
                        display_map = attribution
                        grid = torch.from_numpy(
                            occlusion["target_class_logit_drop"].to_numpy().reshape(
                                UNI_GRID_SIZE,
                                UNI_GRID_SIZE,
                            )
                        )
                        display_occlusion = normalize_map(grid)
                if (
                    str(row["cohort_id"]) in heatmap_ids
                    and display_map is not None
                    and display_occlusion is not None
                ):
                    status = (
                        "correct"
                        if bool(row[f"{column_prefix}_correct"])
                        else "incorrect"
                    )
                    save_attribution_figure(
                        image,
                        {
                            "Grad-CAM": display_map,
                            "patch_occlusion": display_occlusion,
                        },
                        output_dir
                        / "heatmaps"
                        / output_prefix
                        / f"{row['cohort_id']}_{status}.png",
                        (
                            f"{status}: true={row['class_name']}, "
                            f"pred={row[f'{column_prefix}_predicted_class_name']}"
                        ),
                    )
        del model
    return _save_frames(
        output_dir,
        output_prefix,
        metrics,
        curves_out,
        maps_out,
        occlusions_out,
    )
