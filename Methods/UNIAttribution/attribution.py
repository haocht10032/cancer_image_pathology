"""Transformer and CNN attribution methods with patch-level faithfulness tests."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader

from Methods.UNIAttribution.data import UNI_GRID_SIZE, denormalize_image
from Methods.UNIAttribution.model import UNIClassifier


def normalize_map(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each map independently."""
    flat = values.flatten(start_dim=-2)
    minimum = flat.min(dim=-1).values[..., None, None]
    maximum = flat.max(dim=-1).values[..., None, None]
    return (values - minimum) / (maximum - minimum).clamp_min(epsilon)


def standardize_attribution_grid(
    attribution: torch.Tensor,
    image_size: tuple[int, int],
    output_grid_size: int = UNI_GRID_SIZE,
) -> torch.Tensor:
    """Map native attribution through image space onto a shared evaluation grid."""
    if attribution.ndim != 2:
        raise ValueError("Attribution must be a two-dimensional spatial map")
    if attribution.shape == (output_grid_size, output_grid_size):
        return attribution
    image_map = F.interpolate(
        attribution[None, None],
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    evaluation_map = F.adaptive_avg_pool2d(
        image_map,
        output_size=(output_grid_size, output_grid_size),
    )[0, 0]
    return normalize_map(evaluation_map)


def _rollout(attention_matrices: list[torch.Tensor]) -> torch.Tensor:
    """Multiply residual-augmented, row-normalized attention across layers."""
    if not attention_matrices:
        raise ValueError("At least one attention matrix is required")
    token_count = attention_matrices[0].shape[-1]
    identity = torch.eye(
        token_count,
        device=attention_matrices[0].device,
        dtype=attention_matrices[0].dtype,
    ).unsqueeze(0)
    joint = identity.expand(attention_matrices[0].shape[0], -1, -1)
    for attention in attention_matrices:
        augmented = attention + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = augmented @ joint
    return joint


class TimmAttentionCapture(AbstractContextManager["TimmAttentionCapture"]):
    """Capture attention tensors from timm ViT blocks on the active graph."""

    def __init__(self, encoder: nn.Module) -> None:
        self.encoder = encoder
        self.attentions: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._fused_states: list[bool] = []

    def _capture(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if output.requires_grad:
            output.retain_grad()
        self.attentions.append(output)

    def __enter__(self) -> "TimmAttentionCapture":
        blocks = getattr(self.encoder, "blocks", None)
        if blocks is None:
            raise TypeError("The encoder does not expose timm-style Transformer blocks")
        self.attentions.clear()
        for block in blocks:
            attention = block.attn
            self._fused_states.append(bool(getattr(attention, "fused_attn", False)))
            if hasattr(attention, "fused_attn"):
                attention.fused_attn = False
            self._handles.append(attention.attn_drop.register_forward_hook(self._capture))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        for block, fused_state in zip(self.encoder.blocks, self._fused_states):
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = fused_state
        self._handles.clear()
        self._fused_states.clear()


def transformer_attributions(
    model: UNIClassifier,
    image: torch.Tensor,
    target_class: int | None = None,
) -> dict[str, object]:
    """Compute attention baselines and class-conditioned gradient-weighted rollout."""
    if image.shape[0] != 1:
        raise ValueError("Attribution currently expects a single image")

    model.eval()
    model.zero_grad(set_to_none=True)
    attributed_image = image.detach().clone().requires_grad_(True)
    with TimmAttentionCapture(model.encoder) as capture:
        logits = model(attributed_image)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
        target_class = predicted_class if target_class is None else int(target_class)
        logits[0, target_class].backward()

    attentions = capture.attentions
    if not attentions:
        raise RuntimeError("No attention matrices were captured")
    token_count = attentions[-1].shape[-1]
    patch_count = token_count - 1
    grid_size = int(round(patch_count**0.5))
    if grid_size * grid_size != patch_count:
        raise RuntimeError(
            f"Expected square patch tokens after CLS, found {patch_count}"
        )

    mean_attentions = [attention.detach().mean(dim=1) for attention in attentions]
    raw_cls_attention = mean_attentions[-1][0, 0, 1:].reshape(grid_size, grid_size)
    rollout = _rollout(mean_attentions)[0, 0, 1:].reshape(grid_size, grid_size)

    gradient_attentions: list[torch.Tensor] = []
    for attention in attentions:
        if attention.grad is None:
            raise RuntimeError("Attention gradients were not retained")
        weighted = (attention.detach() * attention.grad.detach()).clamp_min(0)
        gradient_attentions.append(weighted.mean(dim=1))
    gradient_rollout = _rollout(gradient_attentions)[
        0, 0, 1:
    ].reshape(grid_size, grid_size)

    return {
        "raw_attention": normalize_map(raw_cls_attention),
        "attention_rollout": normalize_map(rollout),
        "gradient_attention_rollout": normalize_map(gradient_rollout),
        "target_class": target_class,
        "predicted_class": predicted_class,
        "target_logit": float(logits[0, target_class].detach().cpu()),
        "target_probability": float(
            probabilities[0, target_class].detach().cpu()
        ),
    }


@torch.inference_mode()
def cache_patch_tokens(
    model: UNIClassifier,
    loader: DataLoader,
    token_path: Path | str,
    index_path: Path | str,
    device: torch.device,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Stream all 14 x 14 UNI patch tokens to a float16 NumPy memmap."""
    token_path = Path(token_path)
    index_path = Path(index_path)
    if token_path.is_file() and index_path.is_file() and not overwrite:
        return token_path, index_path

    token_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_count = len(loader.dataset)
    token_store = np.lib.format.open_memmap(
        token_path,
        mode="w+",
        dtype=np.float16,
        shape=(sample_count, UNI_GRID_SIZE, UNI_GRID_SIZE, model.feature_dim),
    )
    index_rows: list[dict[str, object]] = []
    offset = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = model.forward_tokens(images)
        patch_tokens = tokens[:, 1:]
        if patch_tokens.shape[1] != UNI_GRID_SIZE**2:
            raise RuntimeError(
                f"Expected {UNI_GRID_SIZE**2} UNI patch tokens, "
                f"found {patch_tokens.shape[1]}"
            )
        patch_tokens = patch_tokens.reshape(
            len(images),
            UNI_GRID_SIZE,
            UNI_GRID_SIZE,
            model.feature_dim,
        )
        batch_size = len(images)
        token_store[offset : offset + batch_size] = (
            patch_tokens.detach().cpu().numpy().astype(np.float16)
        )
        for position in range(batch_size):
            index_rows.append(
                {
                    "token_index": offset + position,
                    "path": batch["path"][position],
                    "case_id": batch["case_id"][position],
                    "label": int(batch["label"][position]),
                }
            )
        offset += batch_size
    token_store.flush()
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return token_path, index_path


def _patch_bounds(length: int, grid_size: int) -> np.ndarray:
    return np.rint(np.linspace(0, length, grid_size + 1)).astype(int)


def _mask_patch_indices(
    image: torch.Tensor,
    patch_indices: Iterable[int],
    grid_size: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    masked = image.clone()
    height, width = image.shape[-2:]
    y_bounds = _patch_bounds(height, grid_size)
    x_bounds = _patch_bounds(width, grid_size)
    for patch_index in patch_indices:
        row, column = divmod(int(patch_index), grid_size)
        masked[
            ...,
            y_bounds[row] : y_bounds[row + 1],
            x_bounds[column] : x_bounds[column + 1],
        ] = fill_value
    return masked


@torch.inference_mode()
def patch_occlusion_scores(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int | None = None,
    grid_size: int = UNI_GRID_SIZE,
    occlusion_batch_size: int = 64,
    group_size: int = 1,
    fill_value: float = 0.0,
) -> dict[str, object]:
    """Measure predicted-class drops when each patch or patch group is removed."""
    if image.shape[0] != 1:
        raise ValueError("Patch occlusion expects a single image")
    model.eval()
    original_logits = model(image)
    original_probabilities = torch.softmax(original_logits, dim=1)
    predicted_class = int(original_probabilities.argmax(dim=1).item())
    target_class = predicted_class if target_class is None else int(target_class)
    original_logit = float(original_logits[0, target_class])
    original_probability = float(original_probabilities[0, target_class])

    regions: list[list[int]] = []
    for row in range(0, grid_size, group_size):
        for column in range(0, grid_size, group_size):
            region = [
                patch_row * grid_size + patch_column
                for patch_row in range(row, min(row + group_size, grid_size))
                for patch_column in range(
                    column, min(column + group_size, grid_size)
                )
            ]
            regions.append(region)

    occluded_logits: list[torch.Tensor] = []
    occluded_probabilities: list[torch.Tensor] = []
    for start in range(0, len(regions), occlusion_batch_size):
        batch_regions = regions[start : start + occlusion_batch_size]
        occluded_batch = torch.cat(
            [
                _mask_patch_indices(
                    image,
                    region,
                    grid_size=grid_size,
                    fill_value=fill_value,
                )
                for region in batch_regions
            ],
            dim=0,
        )
        logits = model(occluded_batch)
        probabilities = torch.softmax(logits, dim=1)
        occluded_logits.append(logits[:, target_class].cpu())
        occluded_probabilities.append(probabilities[:, target_class].cpu())

    logit_drops = original_logit - torch.cat(occluded_logits)
    probability_drops = original_probability - torch.cat(occluded_probabilities)
    region_grid = int(np.ceil(grid_size / group_size))
    return {
        "logit_drop": logit_drops.reshape(region_grid, region_grid),
        "probability_drop": probability_drops.reshape(region_grid, region_grid),
        "regions": regions,
        "target_class": target_class,
        "predicted_class": predicted_class,
        "original_logit": original_logit,
        "original_probability": original_probability,
    }


@torch.inference_mode()
def deletion_curves(
    model: nn.Module,
    image: torch.Tensor,
    attribution: torch.Tensor,
    target_class: int,
    fractions: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    random_repeats: int = 20,
    random_seed: int = 41,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Compare top-, random-, and lowest-attribution patch deletion."""
    model.eval()
    grid_size = attribution.shape[-1]
    patch_count = grid_size**2
    flat_attribution = attribution.detach().cpu().flatten().numpy()
    highest_order = np.argsort(-flat_attribution)
    lowest_order = np.argsort(flat_attribution)
    rng = np.random.default_rng(random_seed)

    rows: list[dict[str, object]] = []
    strategies = {
        "highest": [highest_order],
        "lowest": [lowest_order],
        "random": [rng.permutation(patch_count) for _ in range(random_repeats)],
    }
    for strategy, orders in strategies.items():
        for fraction in fractions:
            remove_count = min(
                patch_count,
                int(round(float(fraction) * patch_count)),
            )
            masked_images = [
                _mask_patch_indices(
                    image,
                    order[:remove_count],
                    grid_size=grid_size,
                    fill_value=fill_value,
                )
                for order in orders
            ]
            batch = torch.cat(masked_images, dim=0)
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            target_logits = logits[:, target_class]
            target_probabilities = probabilities[:, target_class]
            rows.append(
                {
                    "strategy": strategy,
                    "fraction_removed": float(fraction),
                    "patches_removed": remove_count,
                    "target_logit": float(target_logits.mean().cpu()),
                    "target_probability": float(target_probabilities.mean().cpu()),
                    "target_logit_std": float(target_logits.std(unbiased=False).cpu()),
                    "target_probability_std": float(
                        target_probabilities.std(unbiased=False).cpu()
                    ),
                }
            )
    return pd.DataFrame(rows)


def deletion_auc(
    curves: pd.DataFrame,
    value_column: str = "target_probability",
) -> dict[str, float]:
    """Return trapezoidal deletion AUC; lower values mean faster evidence removal."""
    aucs: dict[str, float] = {}
    for strategy, group in curves.groupby("strategy"):
        ordered = group.sort_values("fraction_removed")
        aucs[strategy] = float(
            np.trapz(
                ordered[value_column].to_numpy(),
                ordered["fraction_removed"].to_numpy(),
            )
        )
    return aucs


def attribution_occlusion_correlation(
    attribution: torch.Tensor,
    occlusion_scores: torch.Tensor,
) -> tuple[float, float]:
    """Spearman correlation between attribution ranking and occlusion logit drops."""
    result = spearmanr(
        attribution.detach().cpu().flatten().numpy(),
        occlusion_scores.detach().cpu().flatten().numpy(),
    )
    return float(result.statistic), float(result.pvalue)


class GradCAM(AbstractContextManager["GradCAM"]):
    """Minimal Grad-CAM implementation for the existing CNN baseline."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _forward_hook(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def __enter__(self) -> "GradCAM":
        self._handle = self.target_layer.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def attribute(
        self,
        image: torch.Tensor,
        target_class: int | None = None,
        output_grid_size: int = UNI_GRID_SIZE,
    ) -> dict[str, object]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        logits = self.model(image)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
        target_class = predicted_class if target_class is None else int(target_class)
        logits[0, target_class].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).clamp_min(0)
        cam = F.interpolate(
            cam,
            size=(output_grid_size, output_grid_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return {
            "gradcam": normalize_map(cam).detach(),
            "target_class": target_class,
            "predicted_class": predicted_class,
            "target_logit": float(logits[0, target_class].detach().cpu()),
            "target_probability": float(
                probabilities[0, target_class].detach().cpu()
            ),
        }


def topk_jaccard(
    first: torch.Tensor,
    second: torch.Tensor,
    fraction: float = 0.10,
) -> float:
    """Top-k set overlap for attribution stability."""
    count = max(1, int(round(first.numel() * fraction)))
    first_top = set(torch.topk(first.flatten(), count).indices.cpu().tolist())
    second_top = set(torch.topk(second.flatten(), count).indices.cpu().tolist())
    return len(first_top & second_top) / len(first_top | second_top)


def pairwise_attribution_stability(
    maps_by_seed: dict[int, torch.Tensor],
    top_fraction: float = 0.10,
) -> dict[str, float]:
    """Summarize map stability across all seed pairs."""
    seeds = sorted(maps_by_seed)
    correlations: list[float] = []
    jaccards: list[float] = []
    for first_position, first_seed in enumerate(seeds):
        for second_seed in seeds[first_position + 1 :]:
            correlation, _ = attribution_occlusion_correlation(
                maps_by_seed[first_seed],
                maps_by_seed[second_seed],
            )
            correlations.append(correlation)
            jaccards.append(
                topk_jaccard(
                    maps_by_seed[first_seed],
                    maps_by_seed[second_seed],
                    fraction=top_fraction,
                )
            )
    return {
        "seed_count": len(seeds),
        "pair_count": len(correlations),
        "mean_spearman": float(np.nanmean(correlations)),
        "std_spearman": float(np.nanstd(correlations)),
        "mean_topk_jaccard": float(np.mean(jaccards)),
        "std_topk_jaccard": float(np.std(jaccards)),
    }


def save_attribution_figure(
    image: torch.Tensor,
    maps: dict[str, torch.Tensor],
    output_path: Path | str,
    title: str,
    alpha: float = 0.45,
) -> None:
    """Save the image and attribution overlays as a publication-ready row."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_image = denormalize_image(image[0].detach().cpu()).permute(1, 2, 0).numpy()
    figure, axes = plt.subplots(1, len(maps) + 1, figsize=(3.2 * (len(maps) + 1), 3.2))
    axes[0].imshow(display_image)
    axes[0].set_title(title)
    axes[0].axis("off")
    for axis, (name, attribution) in zip(axes[1:], maps.items()):
        resized = F.interpolate(
            attribution.detach().cpu()[None, None],
            size=display_image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        axis.imshow(display_image)
        axis.imshow(resized.numpy(), cmap="inferno", alpha=alpha, vmin=0, vmax=1)
        axis.set_title(name)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
