"""Generate and cache MaskCut pseudo-masks for Kather-2016 tiles."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.linalg import eigh
from tqdm.auto import tqdm


DINO_WEIGHTS = {
    ("small", 8): (
        "https://dl.fbaipublicfiles.com/dino/"
        "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    ),
    ("base", 8): (
        "https://dl.fbaipublicfiles.com/dino/"
        "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    ),
}
DINO_DIMS = {"small": 384, "base": 768}


def _load_dino_and_crf(maskcut_dir: Path) -> tuple[object, object | None]:
    """Import DINO from the repository and CRF when its dependency is available."""
    maskcut_dir = maskcut_dir.resolve()
    if not (maskcut_dir / "dino.py").is_file():
        raise FileNotFoundError(f"dino.py not found under {maskcut_dir}")

    maskcut_dir_text = str(maskcut_dir)
    if maskcut_dir_text not in sys.path:
        sys.path.insert(0, maskcut_dir_text)

    dino_module = importlib.import_module("dino")
    try:
        crf_module = importlib.import_module("crf")
    except ImportError:
        crf_module = None
    return dino_module, crf_module


def build_maskcut_backbone(
    maskcut_dir: Path | str,
    architecture: str = "small",
    patch_size: int = 8,
    device: str | torch.device = "cuda",
) -> tuple[object, object | None]:
    """Load one DINO backbone and reuse it for all images."""
    maskcut_dir = Path(maskcut_dir)
    key = (architecture, patch_size)
    if key not in DINO_WEIGHTS:
        raise ValueError(f"Unsupported DINO configuration: {key}")

    dino_module, crf_module = _load_dino_and_crf(maskcut_dir)
    backbone = dino_module.ViTFeat(
        DINO_WEIGHTS[key],
        DINO_DIMS[architecture],
        architecture,
        "k",
        patch_size,
    )
    backbone.eval()
    if torch.device(device).type == "cuda":
        backbone.cuda()
    return backbone, crf_module


def _affinity_matrix(features: torch.Tensor, tau: float) -> tuple[np.ndarray, np.ndarray]:
    features = F.normalize(features, p=2, dim=0)
    affinity = (features.transpose(0, 1) @ features).cpu().numpy()
    affinity = np.where(affinity > tau, 1.0, 1e-5)
    degree = np.diag(np.sum(affinity, axis=1))
    return affinity, degree


def _seed_component(
    bipartition: np.ndarray,
    seed: int,
    dimensions: tuple[int, int],
) -> np.ndarray:
    grid = bipartition.reshape(dimensions)
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3), dtype=np.uint8))
    seed_row, seed_column = np.unravel_index(seed, dimensions)
    component_id = labels[seed_row, seed_column]
    if component_id == 0:
        return np.zeros(dimensions, dtype=np.float32)
    return (labels == component_id).astype(np.float32)


def _maskcut_forward(
    features: torch.Tensor,
    dimensions: tuple[int, int],
    image_size: tuple[int, int],
    tau: float,
    number_of_masks: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compact MaskCut graph partition matching the repository's core flow."""
    masks: list[np.ndarray] = []
    eigenvectors: list[np.ndarray] = []
    painted = torch.zeros(dimensions, dtype=features.dtype, device=features.device)
    base_features = features.clone()

    for iteration in range(number_of_masks):
        if iteration == 0:
            active_features = base_features
        else:
            feature_grid = base_features.view(
                base_features.shape[0],
                dimensions[0],
                dimensions[1],
            )
            active_features = (feature_grid * (1 - painted).unsqueeze(0)).flatten(1)

        affinity, degree = _affinity_matrix(active_features, tau)
        _, vectors = eigh(
            degree - affinity,
            degree,
            subset_by_index=[1, 2],
        )
        second_vector = vectors[:, 0]
        partition = second_vector > second_vector.mean()
        seed = int(np.argmax(np.abs(second_vector)))

        partition_grid = partition.reshape(dimensions)
        foreground_corners = sum(
            int(value)
            for value in (
                partition_grid[0, 0],
                partition_grid[0, -1],
                partition_grid[-1, 0],
                partition_grid[-1, -1],
            )
        )
        should_reverse = foreground_corners >= 3 or not partition[seed]
        if should_reverse:
            partition = np.logical_not(partition)
            oriented_vector = -second_vector
        else:
            oriented_vector = second_vector
        seed = int(np.argmax(oriented_vector))

        patch_mask = _seed_component(partition, seed, dimensions)
        patch_mask_tensor = torch.from_numpy(patch_mask).to(features.device)
        area_fraction = patch_mask_tensor.float().mean().item()
        if iteration > 0:
            intersection = (patch_mask_tensor * painted).sum()
            union = ((patch_mask_tensor + painted) > 0).sum().clamp_min(1)
            if (intersection / union).item() > 0.5 or area_fraction <= 0.01:
                patch_mask_tensor.zero_()

        painted = torch.clamp(painted + patch_mask_tensor, 0, 1)
        image_mask = F.interpolate(
            patch_mask_tensor[None, None],
            size=image_size,
            mode="nearest",
        )[0, 0]
        previous = 0 if not masks else np.sum(masks, axis=0)
        non_overlapping = image_mask.cpu().numpy() - previous
        non_overlapping[non_overlapping <= 0] = 0
        masks.append(non_overlapping.astype(np.float32))

        eigen_grid = torch.from_numpy(second_vector.reshape(dimensions)).to(
            features.device
        )
        eigen_image = F.interpolate(
            eigen_grid[None, None],
            size=image_size,
            mode="nearest",
        )[0, 0]
        eigenvectors.append(eigen_image.cpu().numpy())
    return masks, eigenvectors


def _run_maskcut(
    image_path: Path,
    backbone: object,
    patch_size: int,
    tau: float,
    number_of_masks: int,
    fixed_size: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], Image.Image]:
    with Image.open(image_path) as source:
        resized_image = source.convert("RGB").resize(
            (fixed_size, fixed_size),
            Image.Resampling.LANCZOS,
        )

    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    standard_deviation = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    tensor = ((tensor - mean) / standard_deviation).unsqueeze(0).to(device)

    with torch.inference_mode():
        features = backbone(tensor)[0]
    feature_height = fixed_size // patch_size
    feature_width = fixed_size // patch_size
    masks, eigenvectors = _maskcut_forward(
        features,
        (feature_height, feature_width),
        (fixed_size, fixed_size),
        tau,
        number_of_masks,
    )
    return masks, eigenvectors, resized_image


def _refine_mask(
    image: Image.Image,
    binary_mask: np.ndarray,
    crf_module: object | None,
) -> np.ndarray:
    if crf_module is None:
        raise ImportError("CRF refinement requested, but pydensecrf is unavailable")
    refined = crf_module.densecrf(np.asarray(image), binary_mask)
    return ndimage.binary_fill_holes(refined >= 0.5).astype(np.uint8)


def generate_pseudo_masks(
    frame: pd.DataFrame,
    maskcut_dir: Path | str,
    mask_root: Path | str,
    architecture: str = "small",
    patch_size: int = 8,
    tau: float = 0.15,
    number_of_masks: int = 1,
    fixed_size: int = 160,
    use_crf: bool = False,
    overwrite: bool = False,
    device: str | torch.device | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Generate masks once, save PNG files, and add mask_path to the manifest."""
    if "image_path" not in frame.columns or "class_name" not in frame.columns:
        raise ValueError("Manifest requires image_path and class_name columns")

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    mask_root = Path(mask_root).expanduser().resolve()
    mask_root.mkdir(parents=True, exist_ok=True)

    result = frame.copy().reset_index(drop=True)
    failures: list[dict[str, str]] = []
    selected_indices = set(result.index[:limit]) if limit is not None else None
    mask_paths = [
        str(
            mask_root
            / str(row["class_name"])
            / f"{Path(str(row['image_path'])).stem}_mask.png"
        )
        for _, row in result.iterrows()
    ]
    result["mask_path"] = mask_paths
    pending_indices = [
        index
        for index, output_path in enumerate(mask_paths)
        if (selected_indices is None or index in selected_indices)
        and (overwrite or not Path(output_path).is_file())
    ]
    if not pending_indices:
        return result, failures

    backbone, crf_module = build_maskcut_backbone(
        maskcut_dir,
        architecture=architecture,
        patch_size=patch_size,
        device=resolved_device,
    )

    for index in tqdm(
        pending_indices,
        total=len(pending_indices),
        desc="Generating MaskCut masks",
    ):
        row = result.iloc[index]
        image_path = Path(str(row["image_path"]))
        output_path = Path(mask_paths[index])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            bipartitions, _, resized_image = _run_maskcut(
                image_path,
                backbone,
                patch_size,
                tau,
                number_of_masks,
                fixed_size,
                resolved_device,
            )
            binary_mask = (
                np.sum(np.asarray(bipartitions), axis=0) > 0.5
            ).astype(np.uint8)
            if use_crf:
                binary_mask = _refine_mask(
                    resized_image,
                    binary_mask,
                    crf_module,
                )

            with Image.open(image_path) as source:
                original_size = source.size
            mask_image = Image.fromarray(binary_mask * 255, mode="L")
            mask_image = mask_image.resize(
                original_size,
                Image.Resampling.NEAREST,
            )
            mask_image.save(output_path)
        except Exception as error:
            failures.append(
                {
                    "image_path": str(image_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return result, failures
