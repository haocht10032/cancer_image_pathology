"""Partitioned image x seed faithfulness evaluation for the Kather revision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

from Methods.KatherRevision.attribution import integrated_gradients, resize_map
from Methods.KatherRevision.models import resnet_target_layer
from Methods.KatherRevision.perturbations import (
    PERTURBATIONS,
    replace_patch_indices,
    replacement_reference,
)
from Methods.UNIAttribution.attribution import GradCAM, transformer_attributions
from Methods.UNIAttribution.evaluation import load_normalized_image
from Methods.UNIAttribution.model import load_classifier_head


DELETION_FRACTIONS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
EPSILON = 1e-8


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(":".join(map(str, values)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _margin(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    competitors = logits.clone()
    competitors[:, int(target_class)] = -torch.inf
    return logits[:, int(target_class)] - competitors.max(dim=1).values


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if np.unique(first).size < 2 or np.unique(second).size < 2:
        return np.nan
    return float(spearmanr(first, second).statistic)


def _auc(frame: pd.DataFrame, column: str) -> float:
    ordered = frame.sort_values("fraction_removed")
    return float(
        np.trapz(
            ordered[column].to_numpy(dtype=float),
            ordered["fraction_removed"].to_numpy(dtype=float),
        )
    )


@dataclass
class ModelAdapter:
    model_name: str
    checkpoint_dir: Path
    device: torch.device
    image_size: int
    native_grid_size: int
    primary_method: str
    model: nn.Module | None = None

    def load_checkpoint(self, fold: int, seed: int) -> nn.Module:
        raise NotImplementedError

    def load_image(self, path: str) -> torch.Tensor:
        raise NotImplementedError

    def attribution_maps(
        self,
        image: torch.Tensor,
        target_class: int,
        integrated_gradient_steps: int,
        include_higher_res_cam: bool,
        requested_methods: Iterable[str] | None = None,
    ) -> dict[str, tuple[torch.Tensor, int]]:
        raise NotImplementedError


class TransformerAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        checkpoint_dir: Path | str,
        device: torch.device,
        image_transform: Callable,
        native_grid_size: int,
    ) -> None:
        super().__init__(
            model_name=model_name,
            checkpoint_dir=Path(checkpoint_dir),
            device=device,
            image_size=224,
            native_grid_size=native_grid_size,
            primary_method="gradient_attention_rollout",
            model=model,
        )
        self.image_transform = image_transform

    def load_checkpoint(self, fold: int, seed: int) -> nn.Module:
        checkpoint = self.checkpoint_dir / f"seed_{seed}" / f"fold_{fold}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        assert self.model is not None
        load_classifier_head(self.model, checkpoint, map_location=self.device)
        self.model.eval()
        return self.model

    def load_image(self, path: str) -> torch.Tensor:
        return load_normalized_image(
            path,
            self.image_size,
            self.device,
            image_transform=self.image_transform,
        )

    def attribution_maps(
        self,
        image: torch.Tensor,
        target_class: int,
        integrated_gradient_steps: int,
        include_higher_res_cam: bool,
        requested_methods: Iterable[str] | None = None,
    ) -> dict[str, tuple[torch.Tensor, int]]:
        del include_higher_res_cam
        assert self.model is not None
        requested = set(requested_methods or (
            "gradient_attention_rollout",
            "integrated_gradients",
        ))
        maps: dict[str, tuple[torch.Tensor, int]] = {}
        if "gradient_attention_rollout" in requested:
            rollout = transformer_attributions(
                self.model, image, target_class=int(target_class)
            )["gradient_attention_rollout"]
            maps["gradient_attention_rollout"] = (
                rollout.detach(),
                self.native_grid_size,
            )
        if "integrated_gradients" in requested:
            ig = integrated_gradients(
                self.model,
                image,
                int(target_class),
                steps=integrated_gradient_steps,
            )["integrated_gradients"]
            maps["integrated_gradients"] = (ig.detach(), self.native_grid_size)
        return maps


class CNNAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        model_builder: Callable[[], nn.Module],
        checkpoint_dir: Path | str,
        device: torch.device,
        checkpoint_type: str = "state_dict",
        image_size: int = 150,
        native_grid_size: int = 5,
    ) -> None:
        super().__init__(
            model_name=model_name,
            checkpoint_dir=Path(checkpoint_dir),
            device=device,
            image_size=image_size,
            native_grid_size=native_grid_size,
            primary_method="gradcam",
        )
        self.model_builder = model_builder
        self.checkpoint_type = checkpoint_type

    def load_checkpoint(self, fold: int, seed: int) -> nn.Module:
        checkpoint = self.checkpoint_dir / f"seed_{seed}" / f"fold_{fold}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.model = self.model_builder().to(self.device)
        if self.checkpoint_type == "state_dict":
            self.model.load_state_dict(_load_state_dict(checkpoint, self.device))
        elif self.checkpoint_type == "classifier_head":
            load_classifier_head(self.model, checkpoint, map_location=self.device)
        else:
            raise ValueError(f"Unsupported checkpoint type: {self.checkpoint_type}")
        self.model.eval()
        return self.model

    def load_image(self, path: str) -> torch.Tensor:
        return load_normalized_image(path, self.image_size, self.device)

    def attribution_maps(
        self,
        image: torch.Tensor,
        target_class: int,
        integrated_gradient_steps: int,
        include_higher_res_cam: bool,
        requested_methods: Iterable[str] | None = None,
    ) -> dict[str, tuple[torch.Tensor, int]]:
        assert self.model is not None
        requested = set(requested_methods or (
            "gradcam",
            "gradcam_higher",
            "gradcam_multilevel",
            "integrated_gradients",
        ))
        # A frozen backbone has no trainable convolutional parameters. Make the
        # input a gradient source so intermediate feature maps still participate
        # in autograd and Grad-CAM can capture their gradients. This does not
        # unfreeze or update any model parameter.
        cam_image = image.detach().clone().requires_grad_(True)
        maps: dict[str, tuple[torch.Tensor, int]] = {}
        needs_standard = bool(
            requested.intersection(("gradcam", "gradcam_multilevel"))
        )
        standard: torch.Tensor | None = None
        if needs_standard:
            with GradCAM(
                self.model, resnet_target_layer(self.model, "standard")
            ) as cam:
                standard = cam.attribute(
                    cam_image,
                    target_class=int(target_class),
                    output_grid_size=self.native_grid_size,
                )["gradcam"]
            if "gradcam" in requested:
                maps["gradcam"] = (standard.detach(), self.native_grid_size)
        needs_higher = include_higher_res_cam and bool(
            requested.intersection(("gradcam_higher", "gradcam_multilevel"))
        )
        if needs_higher:
            with GradCAM(self.model, resnet_target_layer(self.model, "higher")) as cam:
                higher = cam.attribute(
                    cam_image,
                    target_class=int(target_class),
                    output_grid_size=10,
                )["gradcam"]
            if "gradcam_higher" in requested:
                maps["gradcam_higher"] = (higher.detach(), 10)
            if "gradcam_multilevel" in requested:
                if standard is None:
                    raise RuntimeError("Multilevel Grad-CAM requires standard CAM")
                maps["gradcam_multilevel"] = (
                    0.5 * resize_map(standard, 10) + 0.5 * higher,
                    10,
                )
        if "integrated_gradients" in requested:
            ig = integrated_gradients(
                self.model,
                image,
                int(target_class),
                steps=integrated_gradient_steps,
            )["integrated_gradients"]
            maps["integrated_gradients"] = (ig.detach(), self.native_grid_size)
        return maps


@torch.inference_mode()
def _batched_logits(
    model: nn.Module,
    images: list[torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(images), batch_size):
        outputs.append(model(torch.cat(images[start : start + batch_size], dim=0)).cpu())
    return torch.cat(outputs)


def _occlusion_frame(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    grid_size: int,
    perturbation: str,
    batch_size: int,
) -> pd.DataFrame:
    with torch.inference_mode():
        baseline_logits = model(image).cpu()
    baseline_logit = float(baseline_logits[0, target_class])
    baseline_margin = float(_margin(baseline_logits, target_class)[0])
    reference = replacement_reference(image, perturbation, grid_size)
    masked = [
        replace_patch_indices(image, [index], grid_size, reference)
        for index in range(grid_size**2)
    ]
    logits = _batched_logits(model, masked, batch_size)
    target_logits = logits[:, target_class]
    margins = _margin(logits, target_class)
    drops = baseline_logit - target_logits.numpy()
    margin_drops = baseline_margin - margins.numpy()
    return pd.DataFrame(
        {
            "patch_index": np.arange(grid_size**2),
            "patch_row": np.repeat(np.arange(grid_size), grid_size),
            "patch_column": np.tile(np.arange(grid_size), grid_size),
            "target_logit": target_logits.numpy(),
            "target_margin": margins.numpy(),
            "target_logit_drop": drops,
            "target_margin_drop": margin_drops,
            "supportive_logit_effect": np.clip(drops, 0, None),
            "inhibitory_logit_effect": np.clip(-drops, 0, None),
            "absolute_logit_effect": np.abs(drops),
        }
    )


def _deletion_curves(
    model: nn.Module,
    image: torch.Tensor,
    attribution: torch.Tensor,
    target_class: int,
    grid_size: int,
    perturbation: str,
    random_seed: int,
    random_repeats: int,
    batch_size: int,
) -> pd.DataFrame:
    values = resize_map(attribution, grid_size).cpu().flatten().numpy()
    highest = np.argsort(-values)
    lowest = np.argsort(values)
    rng = np.random.default_rng(random_seed)
    strategies = {
        "top": [highest],
        "bottom": [lowest],
        "random": [rng.permutation(grid_size**2) for _ in range(random_repeats)],
    }
    reference = replacement_reference(image, perturbation, grid_size)
    with torch.inference_mode():
        baseline_logits = model(image).cpu()
    baseline_logit = float(baseline_logits[0, target_class])
    baseline_margin = float(_margin(baseline_logits, target_class)[0])
    rows = []
    for strategy, orders in strategies.items():
        for fraction in DELETION_FRACTIONS:
            remove_count = min(grid_size**2, int(round(fraction * grid_size**2)))
            images = [
                replace_patch_indices(image, order[:remove_count], grid_size, reference)
                for order in orders
            ]
            logits = _batched_logits(model, images, batch_size)
            target_logits = logits[:, target_class]
            margins = _margin(logits, target_class)
            mean_logit = float(target_logits.mean())
            mean_margin = float(margins.mean())
            rows.append(
                {
                    "strategy": strategy,
                    "fraction_removed": float(fraction),
                    "patches_removed": int(remove_count),
                    "repeat_count": len(orders),
                    "unperturbed_target_logit": baseline_logit,
                    "unperturbed_target_margin": baseline_margin,
                    "target_logit": mean_logit,
                    "target_margin": mean_margin,
                    "target_logit_std": float(target_logits.std(unbiased=False)),
                    "target_margin_std": float(margins.std(unbiased=False)),
                    "target_logit_drop": baseline_logit - mean_logit,
                    "target_margin_drop": baseline_margin - mean_margin,
                    "relative_target_logit_reduction": (
                        baseline_logit - mean_logit
                    ) / max(abs(baseline_logit), EPSILON),
                    "relative_target_margin_reduction": (
                        baseline_margin - mean_margin
                    ) / max(abs(baseline_margin), EPSILON),
                }
            )
    return pd.DataFrame(rows)


def _metric_row(
    attribution: torch.Tensor,
    occlusion: pd.DataFrame,
    curves: pd.DataFrame,
) -> dict[str, object]:
    values = attribution.cpu().flatten().numpy()
    signed = occlusion["target_logit_drop"].to_numpy()
    supportive = occlusion["supportive_logit_effect"].to_numpy()
    absolute = occlusion["absolute_logit_effect"].to_numpy()
    by_strategy = {
        strategy: frame for strategy, frame in curves.groupby("strategy")
    }
    baseline_logit = float(curves["unperturbed_target_logit"].iloc[0])
    baseline_margin = float(curves["unperturbed_target_margin"].iloc[0])

    result: dict[str, object] = {
        "unperturbed_target_logit": baseline_logit,
        "unperturbed_target_margin": baseline_margin,
        "near_zero_logit_denominator": bool(abs(baseline_logit) < 0.25),
        "near_zero_margin_denominator": bool(abs(baseline_margin) < 0.25),
        "attribution_occlusion_spearman": _safe_spearman(values, signed),
        "attribution_supportive_occlusion_spearman": _safe_spearman(
            values, supportive
        ),
        "attribution_absolute_occlusion_spearman": _safe_spearman(values, absolute),
        "positive_occlusion_fraction": float((signed > 0).mean()),
        "negative_occlusion_fraction": float((signed < 0).mean()),
        "mean_supportive_occlusion_effect": float(supportive.mean()),
        "mean_absolute_occlusion_effect": float(absolute.mean()),
        "mean_inhibitory_occlusion_effect": float(
            occlusion["inhibitory_logit_effect"].mean()
        ),
    }
    for score in ("target_logit", "target_margin"):
        raw_auc = {
            strategy: _auc(frame, score) for strategy, frame in by_strategy.items()
        }
        relative_column = f"relative_{score}_reduction"
        relative_auc = {
            strategy: _auc(frame, relative_column)
            for strategy, frame in by_strategy.items()
        }
        raw_difference = raw_auc["top"] - raw_auc["random"]
        relative_difference = relative_auc["top"] - relative_auc["random"]
        result.update(
            {
                f"top_{score}_auc": raw_auc["top"],
                f"random_{score}_auc": raw_auc["random"],
                f"bottom_{score}_auc": raw_auc["bottom"],
                f"top_minus_random_{score}_auc": raw_difference,
                f"normalized_top_minus_random_{score}_auc": raw_difference
                / max(abs(raw_auc["random"]), EPSILON),
                f"top_relative_{score}_reduction_auc": relative_auc["top"],
                f"random_relative_{score}_reduction_auc": relative_auc["random"],
                f"bottom_relative_{score}_reduction_auc": relative_auc["bottom"],
                f"top_minus_random_relative_{score}_reduction_auc": (
                    relative_difference
                ),
                f"top_beats_random_{score}": bool(relative_difference > 0),
            }
        )

    top = by_strategy["top"].set_index("fraction_removed")
    random = by_strategy["random"].set_index("fraction_removed")
    for fraction in (0.05, 0.10, 0.20, 0.30):
        suffix = int(round(100 * fraction))
        for score in ("target_logit", "target_margin"):
            result[f"top_{suffix}_{score}_drop"] = float(
                top.loc[fraction, f"{score}_drop"]
            )
            result[f"top_{suffix}_relative_{score}_reduction"] = float(
                top.loc[fraction, f"relative_{score}_reduction"]
            )
            result[f"top_minus_random_{suffix}_relative_{score}_reduction"] = float(
                top.loc[fraction, f"relative_{score}_reduction"]
                - random.loc[fraction, f"relative_{score}_reduction"]
            )
    return result


def _metadata(row: pd.Series, adapter: ModelAdapter) -> dict[str, object]:
    return {
        "cohort_id": row["cohort_id"],
        "cohort_name": row["cohort_name"],
        "path": row["path"],
        "relative_path": row["relative_path"],
        "case_id": row["case_id"],
        "class_name": row["class_name"],
        "true_class": int(row["label"]),
        "model": adapter.model_name,
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
        "prediction": int(row["prediction"]),
        "correct": bool(row["correct"]),
        "confidence": float(row["confidence"]),
        "confidence_group": row["confidence_group"],
        "all_models_correct": bool(row["all_models_correct"]),
        "target_role": row["target_role"],
        "target_class": int(row["target_class"]),
        "analysis_family": row["analysis_family"],
    }


def _map_frame(
    attribution: torch.Tensor,
    metadata: dict[str, object],
    method: str,
    grid_label: str,
    grid_size: int,
    native_grid_size: int,
    null_repetition: int | None,
) -> pd.DataFrame:
    values = resize_map(attribution, grid_size).cpu().numpy()
    frame = pd.DataFrame(
        {
            "patch_index": np.arange(grid_size**2),
            "patch_row": np.repeat(np.arange(grid_size), grid_size),
            "patch_column": np.tile(np.arange(grid_size), grid_size),
            "attribution": values.flatten(),
        }
    )
    for key, value in reversed(list(metadata.items())):
        frame.insert(0, key, value)
    frame["method"] = method
    frame["grid_label"] = grid_label
    frame["evaluation_grid_size"] = grid_size
    frame["native_grid_size"] = native_grid_size
    frame["null_repetition"] = null_repetition
    return frame


def _decorate(
    frame: pd.DataFrame,
    metadata: dict[str, object],
    method: str | None,
    perturbation: str,
    grid_label: str,
    grid_size: int,
) -> pd.DataFrame:
    output = frame.copy()
    for key, value in reversed(list(metadata.items())):
        output.insert(0, key, value)
    output["method"] = method
    output["perturbation"] = perturbation
    output["grid_label"] = grid_label
    output["evaluation_grid_size"] = grid_size
    return output


def _partition_paths(
    output_dir: Path,
    cohort_name: str,
    model_name: str,
    seed: int,
    fold: int,
) -> dict[str, Path]:
    base = output_dir / "partitions" / cohort_name / model_name / f"seed_{seed}"
    base.mkdir(parents=True, exist_ok=True)
    stem = f"fold_{fold}"
    return {
        "metrics": base / f"{stem}_metrics.csv",
        "curves": base / f"{stem}_deletion_curves.csv",
        "maps": base / f"{stem}_attribution_maps.csv",
        "occlusion": base / f"{stem}_occlusion_scores.csv",
    }


def evaluate_image_seed_partitions(
    adapter: ModelAdapter,
    target_index: pd.DataFrame,
    output_dir: Path | str,
    perturbations: Iterable[str] = PERTURBATIONS,
    common_grid_sizes: Iterable[int] = (14, 7),
    integrated_gradient_steps: int = 32,
    random_repeats: int = 20,
    random_null_repeats: int = 5,
    random_null_seed: int | None = 41,
    batch_size: int = 32,
    include_higher_res_cam: bool = True,
    requested_methods: Iterable[str] | None = None,
    include_native_grid: bool = True,
    overwrite: bool = False,
    max_images_per_partition: int | None = None,
) -> list[Path]:
    """Evaluate and save each model/seed/fold partition independently."""
    output_dir = Path(output_dir)
    selected = target_index[target_index["model"].eq(adapter.model_name)].copy()
    written: list[Path] = []
    for (cohort_name, seed, fold), partition in selected.groupby(
        ["cohort_name", "seed", "fold"], sort=True
    ):
        paths = _partition_paths(
            output_dir, str(cohort_name), adapter.model_name, int(seed), int(fold)
        )
        if not overwrite and all(path.is_file() for path in paths.values()):
            written.extend(paths.values())
            continue
        adapter.load_checkpoint(int(fold), int(seed))
        assert adapter.model is not None
        metrics_out: list[dict[str, object]] = []
        curves_out: list[pd.DataFrame] = []
        maps_out: list[pd.DataFrame] = []
        occlusion_out: list[pd.DataFrame] = []
        image_groups = list(partition.groupby("cohort_id", sort=True))
        if max_images_per_partition is not None:
            image_groups = image_groups[:max_images_per_partition]

        for cohort_id, image_rows in image_groups:
            image = adapter.load_image(str(image_rows["path"].iloc[0]))
            with torch.inference_mode():
                checkpoint_prediction = int(adapter.model(image).argmax(dim=1).item())
            expected_prediction = int(image_rows["prediction"].iloc[0])
            if checkpoint_prediction != expected_prediction:
                raise RuntimeError(
                    f"{adapter.model_name} prediction mismatch for {cohort_id}: "
                    f"checkpoint={checkpoint_prediction}, OOF={expected_prediction}"
                )

            for target_class, target_rows in image_rows.groupby("target_class"):
                maps = adapter.attribution_maps(
                    image,
                    int(target_class),
                    integrated_gradient_steps,
                    include_higher_res_cam,
                    requested_methods=requested_methods,
                )
                run_random_null = random_null_repeats > 0 and (
                    random_null_seed is None or int(seed) == int(random_null_seed)
                )
                if run_random_null:
                    for repetition in range(random_null_repeats):
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(
                            _stable_seed(cohort_id, seed, target_class, repetition)
                        )
                        random_map = torch.rand(
                            adapter.native_grid_size,
                            adapter.native_grid_size,
                            generator=generator,
                        ).to(image.device)
                        maps[f"random_attribution_{repetition}"] = (
                            random_map,
                            adapter.native_grid_size,
                        )

                archive_target_rows = target_rows.drop_duplicates(
                    ["target_role", "target_class"]
                )
                for _, target_row in archive_target_rows.iterrows():
                    metadata = _metadata(target_row, adapter)
                    for method, (attribution, native_grid) in maps.items():
                        null_repetition = (
                            int(method.rsplit("_", 1)[1])
                            if method.startswith("random_attribution_")
                            else None
                        )
                        method_name = (
                            "random_attribution"
                            if null_repetition is not None
                            else method
                        )
                        # Random maps are a compact null analysis, not a map archive.
                        if null_repetition is not None:
                            continue
                        native_evaluation = (
                            native_grid
                            if method in {"gradcam_higher", "gradcam_multilevel"}
                            else adapter.native_grid_size
                        )
                        grid_specs = []
                        if include_native_grid:
                            grid_specs.append(
                                (f"native_{native_evaluation}", native_evaluation)
                            )
                        grid_specs.extend(
                            (f"common_{int(size)}", int(size))
                            for size in common_grid_sizes
                        )
                        for grid_label, grid_size in grid_specs:
                            maps_out.append(
                                _map_frame(
                                    attribution,
                                    metadata,
                                    method_name,
                                    grid_label,
                                    grid_size,
                                    native_grid,
                                    null_repetition,
                                )
                            )

                # Occlusion and deletion are cached by target/grid/perturbation.
                unique_grid_sizes = {int(size) for size in common_grid_sizes}
                if include_native_grid:
                    unique_grid_sizes.add(adapter.native_grid_size)
                    if any(
                        method in maps
                        for method in ("gradcam_higher", "gradcam_multilevel")
                    ):
                        unique_grid_sizes.add(10)
                unique_grid_sizes = sorted(unique_grid_sizes)
                for grid_size in unique_grid_sizes:
                    for perturbation in perturbations:
                        occlusion = _occlusion_frame(
                            adapter.model,
                            image,
                            int(target_class),
                            grid_size,
                            str(perturbation),
                            batch_size,
                        )
                        metadata = _metadata(target_rows.iloc[0], adapter)
                        metadata["target_role"] = "unique_target"
                        metadata["analysis_family"] = "occlusion_reference"
                        occlusion_out.append(
                            _decorate(
                                occlusion,
                                metadata,
                                method=None,
                                perturbation=str(perturbation),
                                grid_label=f"image_space_{grid_size}",
                                grid_size=grid_size,
                            )
                        )
                        for method, (attribution, native_grid) in maps.items():
                            method_native = (
                                native_grid
                                if method in {"gradcam_higher", "gradcam_multilevel"}
                                else adapter.native_grid_size
                            )
                            eligible_grids = {
                                *[int(size) for size in common_grid_sizes],
                            }
                            if include_native_grid:
                                eligible_grids.add(method_native)
                            if grid_size not in eligible_grids:
                                continue
                            grid_labels = []
                            if include_native_grid and grid_size == method_native:
                                grid_labels.append(f"native_{grid_size}")
                            if grid_size in {int(size) for size in common_grid_sizes}:
                                grid_labels.append(f"common_{grid_size}")
                            null_repetition = (
                                int(method.rsplit("_", 1)[1])
                                if method.startswith("random_attribution_")
                                else None
                            )
                            method_name = (
                                "random_attribution"
                                if null_repetition is not None
                                else method
                            )
                            # Restrict the optional random-map null to the canonical
                            # perturbation/grid and one prespecified training seed.
                            if null_repetition is not None and not (
                                str(perturbation) == "normalized_zero"
                                and grid_size == 14
                            ):
                                continue
                            if null_repetition is not None:
                                grid_labels = ["common_14"]
                            curves = _deletion_curves(
                                adapter.model,
                                image,
                                attribution,
                                int(target_class),
                                grid_size,
                                str(perturbation),
                                _stable_seed(
                                    cohort_id,
                                    seed,
                                    target_class,
                                    perturbation,
                                    grid_size,
                                ),
                                random_repeats,
                                batch_size,
                            )
                            curves["null_repetition"] = null_repetition
                            resized = resize_map(attribution, grid_size)
                            metric_values = _metric_row(resized, occlusion, curves)
                            for grid_label in dict.fromkeys(grid_labels):
                                for _, target_row in archive_target_rows.iterrows():
                                    metadata = _metadata(target_row, adapter)
                                    archived = metadata.copy()
                                    archived["analysis_family"] = "curve_archive"
                                    curves_out.append(
                                        _decorate(
                                            curves,
                                            archived,
                                            method_name,
                                            str(perturbation),
                                            grid_label,
                                            grid_size,
                                        )
                                    )
                                for _, target_row in target_rows.iterrows():
                                    metadata = _metadata(target_row, adapter)
                                    metric = {
                                        **metadata,
                                        "method": method_name,
                                        "null_repetition": null_repetition,
                                        "perturbation": str(perturbation),
                                        "grid_label": grid_label,
                                        "evaluation_grid_size": grid_size,
                                        "native_grid_size": native_grid,
                                        **metric_values,
                                    }
                                    metrics_out.append(metric)

        pd.DataFrame(metrics_out).to_csv(paths["metrics"], index=False)
        pd.concat(curves_out, ignore_index=True).to_csv(paths["curves"], index=False)
        pd.concat(maps_out, ignore_index=True).to_csv(paths["maps"], index=False)
        pd.concat(occlusion_out, ignore_index=True).to_csv(
            paths["occlusion"], index=False
        )
        written.extend(paths.values())
    return written


def consolidate_partitions(
    output_dir: Path | str,
    artifact_name: str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Combine all completed partition files of one artifact type."""
    output_dir = Path(output_dir)
    files = sorted((output_dir / "partitions").glob(f"**/*_{artifact_name}.csv"))
    if not files:
        raise FileNotFoundError(f"No partition files found for {artifact_name}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def evaluate_map_only_partitions(
    adapter: ModelAdapter,
    target_index: pd.DataFrame,
    output_dir: Path | str,
    requested_methods: Iterable[str],
    grid_size: int = 14,
    integrated_gradient_steps: int = 32,
    overwrite: bool = False,
) -> list[Path]:
    """Save attribution maps across seeds without occlusion or deletion.

    This is the computationally efficient stability path. It deliberately stores
    one map per image, seed, model, target role, and requested method, while the
    expensive faithfulness interventions remain confined to prespecified seeds.
    """
    output_dir = Path(output_dir)
    selected = target_index[target_index["model"].eq(adapter.model_name)].copy()
    written: list[Path] = []
    for (cohort_name, seed, fold), partition in selected.groupby(
        ["cohort_name", "seed", "fold"], sort=True
    ):
        path = (
            output_dir
            / "map_partitions"
            / str(cohort_name)
            / adapter.model_name
            / f"seed_{int(seed)}"
            / f"fold_{int(fold)}_attribution_maps.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and not overwrite:
            written.append(path)
            continue
        adapter.load_checkpoint(int(fold), int(seed))
        assert adapter.model is not None
        output: list[pd.DataFrame] = []
        for cohort_id, image_rows in partition.groupby("cohort_id", sort=True):
            image = adapter.load_image(str(image_rows["path"].iloc[0]))
            with torch.inference_mode():
                checkpoint_prediction = int(adapter.model(image).argmax(dim=1).item())
            expected_prediction = int(image_rows["prediction"].iloc[0])
            if checkpoint_prediction != expected_prediction:
                raise RuntimeError(
                    f"{adapter.model_name} prediction mismatch for {cohort_id}: "
                    f"checkpoint={checkpoint_prediction}, OOF={expected_prediction}"
                )
            for target_class, target_rows in image_rows.groupby("target_class"):
                maps = adapter.attribution_maps(
                    image,
                    int(target_class),
                    integrated_gradient_steps,
                    include_higher_res_cam=False,
                    requested_methods=requested_methods,
                )
                archive_rows = target_rows.drop_duplicates(
                    ["target_role", "target_class"]
                )
                for _, target_row in archive_rows.iterrows():
                    metadata = _metadata(target_row, adapter)
                    metadata["analysis_family"] = "stability_map_archive"
                    for method, (attribution, native_grid) in maps.items():
                        output.append(
                            _map_frame(
                                attribution,
                                metadata,
                                method,
                                f"common_{int(grid_size)}",
                                int(grid_size),
                                native_grid,
                                null_repetition=None,
                            )
                        )
        if output:
            pd.concat(output, ignore_index=True).to_csv(path, index=False)
        else:
            pd.DataFrame(
                columns=[
                    "cohort_id",
                    "cohort_name",
                    "model",
                    "seed",
                    "method",
                    "target_role",
                    "target_class",
                    "patch_index",
                    "attribution",
                ]
            ).to_csv(path, index=False)
        written.append(path)
    return written


def consolidate_map_only_partitions(
    output_dir: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    files = sorted(
        (output_dir / "map_partitions").glob("**/*_attribution_maps.csv")
    )
    if not files:
        raise FileNotFoundError("No map-only stability partitions found")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame
