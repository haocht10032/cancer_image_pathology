"""Common input attribution and spatial map utilities."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def normalize_map(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    values = values.detach()
    minimum = values.min()
    maximum = values.max()
    return (values - minimum) / (maximum - minimum + epsilon)


def resize_map(values: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Map an attribution into a square image-space evaluation grid."""
    if values.ndim != 2:
        raise ValueError("Attribution map must be two-dimensional")
    resized = F.interpolate(
        values[None, None].float(),
        size=(grid_size, grid_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return normalize_map(resized)


def integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    steps: int = 32,
    internal_batch_size: int = 8,
    baseline: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """Input-space Integrated Gradients with a normalized-zero baseline.

    All models use the same alpha path, target-logit objective, baseline definition,
    trapezoidal integration, and RGB-channel reduction.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected one image with shape [1, channels, height, width]")
    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two steps")
    baseline = torch.zeros_like(image) if baseline is None else baseline
    if baseline.shape != image.shape:
        raise ValueError("Integrated Gradients baseline shape does not match image")

    model.eval()
    with torch.no_grad():
        logits = model(image)
        predicted_class = int(logits.argmax(dim=1).item())
        target_logit = float(logits[0, target_class].cpu())

    difference = image - baseline
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=image.device)
    gradient_sum = torch.zeros_like(image)
    for start in range(0, len(alphas), internal_batch_size):
        chunk = alphas[start : start + internal_batch_size]
        interpolated = baseline + chunk[:, None, None, None] * difference
        interpolated = interpolated.detach().requires_grad_(True)
        scores = model(interpolated)[:, int(target_class)]
        gradients = torch.autograd.grad(scores.sum(), interpolated)[0]
        weights = torch.ones(len(chunk), device=image.device)
        global_indices = torch.arange(start, start + len(chunk), device=image.device)
        weights[(global_indices == 0) | (global_indices == steps)] = 0.5
        gradient_sum += (gradients * weights[:, None, None, None]).sum(
            dim=0, keepdim=True
        )

    average_gradient = gradient_sum / float(steps)
    channel_attribution = difference * average_gradient
    signed = channel_attribution.sum(dim=1)[0]
    positive = signed.clamp_min(0)
    absolute = channel_attribution.abs().sum(dim=1)[0]
    return {
        "integrated_gradients": normalize_map(positive),
        "integrated_gradients_absolute": normalize_map(absolute),
        "integrated_gradients_signed": signed.detach(),
        "predicted_class": predicted_class,
        "target_class": int(target_class),
        "target_logit": target_logit,
    }


def attribution_for_target(
    provider: Callable[[int], torch.Tensor],
    target_class: int,
    grid_size: int,
) -> torch.Tensor:
    """Resolve and standardize a provider's target-conditioned map."""
    return resize_map(provider(int(target_class)), grid_size)

