"""Image-space perturbations shared by all Kather-5K revision experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur


PERTURBATIONS = ("normalized_zero", "gaussian_blur", "local_mean")


@dataclass(frozen=True)
class PatchGeometry:
    height: int
    width: int
    grid_size: int

    @property
    def row_bounds(self) -> np.ndarray:
        return np.rint(np.linspace(0, self.height, self.grid_size + 1)).astype(int)

    @property
    def column_bounds(self) -> np.ndarray:
        return np.rint(np.linspace(0, self.width, self.grid_size + 1)).astype(int)

    def bounds(self, index: int) -> tuple[int, int, int, int]:
        row, column = divmod(int(index), self.grid_size)
        return (
            int(self.row_bounds[row]),
            int(self.row_bounds[row + 1]),
            int(self.column_bounds[column]),
            int(self.column_bounds[column + 1]),
        )


def _local_mean_reference(image: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Fill each region with the mean of its available 8-neighborhood regions."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected one image with shape [1, channels, height, width]")
    geometry = PatchGeometry(int(image.shape[-2]), int(image.shape[-1]), grid_size)
    patch_means: dict[tuple[int, int], torch.Tensor] = {}
    for row in range(grid_size):
        for column in range(grid_size):
            y0, y1, x0, x1 = geometry.bounds(row * grid_size + column)
            patch_means[(row, column)] = image[:, :, y0:y1, x0:x1].mean(
                dim=(-2, -1), keepdim=True
            )

    reference = image.clone()
    for row in range(grid_size):
        for column in range(grid_size):
            neighbors = [
                patch_means[(neighbor_row, neighbor_column)]
                for neighbor_row in range(max(0, row - 1), min(grid_size, row + 2))
                for neighbor_column in range(
                    max(0, column - 1), min(grid_size, column + 2)
                )
                if (neighbor_row, neighbor_column) != (row, column)
            ]
            replacement = (
                torch.stack(neighbors).mean(dim=0)
                if neighbors
                else patch_means[(row, column)]
            )
            y0, y1, x0, x1 = geometry.bounds(row * grid_size + column)
            reference[:, :, y0:y1, x0:x1] = replacement
    return reference


def replacement_reference(
    image: torch.Tensor,
    perturbation: str,
    grid_size: int,
    blur_sigma: float | None = None,
) -> torch.Tensor:
    """Return a full-image source from which intervened regions are copied."""
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"Unsupported perturbation: {perturbation}")
    if perturbation == "normalized_zero":
        return torch.zeros_like(image)
    if perturbation == "gaussian_blur":
        minimum_side = max(3, min(int(image.shape[-2]), int(image.shape[-1])))
        # Match blur scale in image space despite the 150 px CNN and 224 px ViT
        # inputs. The sigma spans 2% of the shorter image dimension.
        sigma = float(blur_sigma) if blur_sigma is not None else minimum_side * 0.02
        proposed_kernel = 2 * int(np.ceil(3.0 * sigma)) + 1
        kernel = min(
            proposed_kernel,
            minimum_side if minimum_side % 2 else minimum_side - 1,
        )
        kernel = max(kernel, 3)
        return gaussian_blur(image, [kernel, kernel], [sigma, sigma])
    return _local_mean_reference(image, grid_size)


def replace_patch_indices(
    image: torch.Tensor,
    indices: Iterable[int],
    grid_size: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Replace selected image-space regions using a precomputed reference tensor."""
    if image.shape != reference.shape:
        raise ValueError("Image and perturbation reference must have identical shapes")
    output = image.clone()
    geometry = PatchGeometry(int(image.shape[-2]), int(image.shape[-1]), grid_size)
    for index in indices:
        y0, y1, x0, x1 = geometry.bounds(int(index))
        output[:, :, y0:y1, x0:x1] = reference[:, :, y0:y1, x0:x1]
    return output
