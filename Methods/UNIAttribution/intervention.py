"""Activation-patching attribution for frozen UNI classifiers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

from Methods.UNIAttribution.data import UNI_GRID_SIZE
from Methods.UNIAttribution.model import UNIClassifier


def resolve_intervention_layer(encoder: nn.Module, layer: int = -2) -> int:
    """Resolve a block index and require at least one downstream block."""
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise TypeError("The encoder does not expose timm-style Transformer blocks")
    block_count = len(blocks)
    resolved = layer if layer >= 0 else block_count + layer
    if resolved < 0 or resolved >= block_count - 1:
        raise ValueError(
            "Intervention layer must precede at least one Transformer block; "
            f"received {layer} for {block_count} blocks"
        )
    return int(resolved)


@contextmanager
def _capture_block_output(
    block: nn.Module,
) -> Iterator[list[torch.Tensor]]:
    captured: list[torch.Tensor] = []

    def hook(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("UNI block output must be a tensor")
        captured.append(output)

    handle = block.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _continue_encoder(
    encoder: nn.Module,
    tokens: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    output = tokens
    for block in encoder.blocks[layer_index + 1 :]:
        output = block(output)
    norm = getattr(encoder, "norm", None)
    if norm is not None:
        output = norm(output)
    return output


def _target_margin(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    competitors = logits.clone()
    competitors[:, target_class] = -torch.inf
    return logits[:, target_class] - competitors.max(dim=1).values


def _replacement_vector(
    layer_tokens: torch.Tensor,
    prefix_tokens: int,
    replacement: str,
    reference_activation: torch.Tensor | None,
) -> torch.Tensor:
    patch_tokens = layer_tokens[:, prefix_tokens:]
    if replacement == "image_patch_mean":
        return patch_tokens.mean(dim=1)
    if replacement == "zero":
        return torch.zeros_like(patch_tokens[:, 0])
    if replacement == "reference_activation":
        if reference_activation is None:
            raise ValueError(
                "reference_activation is required for replacement='reference_activation'"
            )
        reference = reference_activation.to(
            device=layer_tokens.device,
            dtype=layer_tokens.dtype,
        )
        if reference.ndim == 1:
            reference = reference.unsqueeze(0)
        expected = (1, layer_tokens.shape[-1])
        if tuple(reference.shape) != expected:
            raise ValueError(
                f"Expected reference activation shape {expected}, found {tuple(reference.shape)}"
            )
        return reference
    raise ValueError(
        "replacement must be 'image_patch_mean', 'zero', or 'reference_activation'"
    )


@torch.inference_mode()
def intervention_patch_attribution(
    model: UNIClassifier,
    image: torch.Tensor,
    target_class: int | None = None,
    layer: int = -2,
    replacement: str = "image_patch_mean",
    reference_activation: torch.Tensor | None = None,
    intervention_batch_size: int = 64,
    consistency_atol: float = 1e-4,
    consistency_rtol: float = 1e-4,
) -> dict[str, object]:
    """Measure independent patch-token intervention effects on a class logit.

    The intervention is applied to the output of one UNI Transformer block.
    One contextualized patch token at a time is replaced, and the modified
    token sequence is propagated through all remaining blocks and final norm.
    The signed effect is baseline target logit minus intervened target logit.
    """
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Intervention attribution expects one batched image")
    if intervention_batch_size <= 0:
        raise ValueError("intervention_batch_size must be positive")

    model.eval()
    layer_index = resolve_intervention_layer(model.encoder, layer)
    block = model.encoder.blocks[layer_index]
    with _capture_block_output(block) as captured:
        final_tokens = model.forward_tokens(image)
    if len(captured) != 1:
        raise RuntimeError(
            f"Expected one activation from block {layer_index}, captured {len(captured)}"
        )
    layer_tokens = captured[0]
    prefix_tokens = int(getattr(model.encoder, "num_prefix_tokens", 1))
    patch_count = layer_tokens.shape[1] - prefix_tokens
    grid_size = int(round(patch_count**0.5))
    if grid_size * grid_size != patch_count:
        raise RuntimeError(f"Expected square patch tokens, found {patch_count}")
    if grid_size != UNI_GRID_SIZE:
        raise RuntimeError(
            f"Expected UNI {UNI_GRID_SIZE} x {UNI_GRID_SIZE} patches, found {grid_size} x {grid_size}"
        )

    baseline_logits = model.forward_from_features(final_tokens[:, 0])
    predicted_class = int(baseline_logits.argmax(dim=1).item())
    target_class = predicted_class if target_class is None else int(target_class)

    continued_tokens = _continue_encoder(model.encoder, layer_tokens, layer_index)
    continued_logits = model.forward_from_features(continued_tokens[:, 0])
    if not torch.allclose(
        baseline_logits,
        continued_logits,
        atol=consistency_atol,
        rtol=consistency_rtol,
    ):
        max_error = float((baseline_logits - continued_logits).abs().max().cpu())
        raise RuntimeError(
            "Continuing from the captured UNI layer did not reproduce the normal "
            f"forward pass (maximum logit error {max_error:.6g})"
        )

    replacement_vector = _replacement_vector(
        layer_tokens,
        prefix_tokens,
        replacement,
        reference_activation,
    )
    baseline_target_logit = baseline_logits[:, target_class]
    baseline_target_margin = _target_margin(baseline_logits, target_class)
    intervened_logits: list[torch.Tensor] = []
    for start in range(0, patch_count, intervention_batch_size):
        patch_indices = torch.arange(
            start,
            min(start + intervention_batch_size, patch_count),
            device=layer_tokens.device,
        )
        patched = layer_tokens.expand(len(patch_indices), -1, -1).clone()
        patched[
            torch.arange(len(patch_indices), device=layer_tokens.device),
            prefix_tokens + patch_indices,
        ] = replacement_vector.expand(len(patch_indices), -1)
        continued = _continue_encoder(model.encoder, patched, layer_index)
        intervened_logits.append(model.forward_from_features(continued[:, 0]))

    intervention_logits = torch.cat(intervened_logits, dim=0)
    target_logits = intervention_logits[:, target_class]
    target_margins = _target_margin(intervention_logits, target_class)
    logit_effects = baseline_target_logit - target_logits
    margin_effects = baseline_target_margin - target_margins

    return {
        "intervention_effect": logit_effects.reshape(grid_size, grid_size),
        "margin_intervention_effect": margin_effects.reshape(grid_size, grid_size),
        "intervened_target_logits": target_logits,
        "intervened_target_margins": target_margins,
        "baseline_logits": baseline_logits[0],
        "baseline_target_logit": float(baseline_target_logit.item()),
        "baseline_target_margin": float(baseline_target_margin.item()),
        "predicted_class": predicted_class,
        "target_class": target_class,
        "layer_index": layer_index,
        "layer_number": layer_index + 1,
        "block_count": len(model.encoder.blocks),
        "replacement": replacement,
        "interventions_independent": True,
        "effect_definition": "baseline_target_logit_minus_intervened_target_logit",
    }
