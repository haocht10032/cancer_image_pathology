from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image


def _grid(frame: pd.DataFrame, value: str, size: int = 14) -> np.ndarray:
    values = np.full((size, size), np.nan, dtype=float)
    for row in frame.itertuples(index=False):
        values[int(row.patch_row), int(row.patch_column)] = float(getattr(row, value))
    return values


def _normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def save_uni_region_example(
    project_root: Path | str,
    cohort_id: str,
    output_path: Path | str,
    *,
    target_role: str = "predicted",
    top_fraction: float = 0.10,
) -> Path:
    """Show UNI prediction-contributing regions and matched input occlusion evidence."""
    if target_role not in {"predicted", "true"}:
        raise ValueError("target_role must be 'predicted' or 'true'")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    root = Path(project_root)
    grouped = root / "artifacts" / "grouped_oof_faithfulness"
    three_model = root / "artifacts" / "dinov2_three_model"

    manifest = pd.read_csv(grouped / "faithfulness_cohort_manifest.csv")
    sample = manifest.loc[manifest["cohort_id"].astype(str).eq(str(cohort_id))]
    if len(sample) != 1:
        raise ValueError(f"Expected one cohort row for {cohort_id}, found {len(sample)}")
    sample = sample.iloc[0]

    image_path = (
        root
        / "Colorectal Histology MNIST"
        / "Kather_texture_2016_image_tiles_5000"
        / "Kather_texture_2016_image_tiles_5000"
        / sample["relative_path"]
    )
    image = np.asarray(Image.open(image_path).convert("RGB"))

    attribution_rows = pd.read_csv(
        grouped / "faithfulness" / "uni_attribution_maps.csv"
    )
    attribution_rows = attribution_rows.loc[
        attribution_rows["cohort_id"].astype(str).eq(str(cohort_id))
        & attribution_rows["method"].eq("gradient_attention_rollout")
        & attribution_rows["target_role"].eq(target_role)
    ]
    if attribution_rows.empty:
        raise ValueError(
            f"No UNI gradient-rollout map for cohort={cohort_id}, target_role={target_role}"
        )
    attribution = _grid(attribution_rows, "attribution")

    occlusion_rows = pd.read_csv(
        grouped / "faithfulness" / "uni_patch_occlusion_scores.csv"
    )
    occlusion_rows = occlusion_rows.loc[
        occlusion_rows["cohort_id"].astype(str).eq(str(cohort_id))
        & occlusion_rows["target_role"].eq(target_role)
    ]
    occlusion = _grid(occlusion_rows, "target_class_logit_drop")

    metrics = pd.read_csv(
        three_model / "faithfulness" / "three_model_faithfulness_metrics.csv"
    )
    metrics = metrics.loc[
        metrics["cohort_id"].astype(str).eq(str(cohort_id))
        & metrics["model"].eq("UNI")
        & metrics["method"].eq("gradient_attention_rollout")
        & metrics["target_role"].eq(target_role)
    ].iloc[0]

    attr_norm = _normalize(attribution)
    occ_norm = _normalize(occlusion)
    extent = (0, image.shape[1], image.shape[0], 0)
    patch_width = image.shape[1] / attribution.shape[1]
    patch_height = image.shape[0] / attribution.shape[0]
    top_count = max(1, int(np.ceil(attribution.size * top_fraction)))
    top_indices = np.argpartition(attribution.ravel(), -top_count)[-top_count:]

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.65), constrained_layout=True)
    for axis in axes:
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

    axes[0].set_title("Original tile", fontsize=11)

    axes[1].imshow(
        attr_norm,
        cmap="magma",
        interpolation="bilinear",
        alpha=0.66,
        extent=extent,
        norm=Normalize(0, 1),
    )
    axes[1].set_title("UNI gradient-weighted rollout", fontsize=11)

    for index in top_indices:
        row, column = np.unravel_index(index, attribution.shape)
        axes[2].add_patch(
            Rectangle(
                (column * patch_width, row * patch_height),
                patch_width,
                patch_height,
                facecolor="#ffd54f",
                edgecolor="#00e5ff",
                linewidth=1.0,
                alpha=0.42,
            )
        )
    axes[2].set_title(f"Top {top_fraction:.0%} attributed patches", fontsize=11)

    axes[3].imshow(
        occ_norm,
        cmap="magma",
        interpolation="bilinear",
        alpha=0.66,
        extent=extent,
        norm=Normalize(0, 1),
    )
    axes[3].set_title("Input occlusion logit drop", fontsize=11)

    fig.suptitle(
        "Prediction-contributing regions: "
        f"true={sample['class_name']}, UNI prediction={sample['uni_predicted_class_name']} "
        f"({sample['uni_confidence']:.1%})\n"
        f"Attribution target={target_role} class; "
        f"Attribution-occlusion Spearman r={metrics['attribution_occlusion_spearman']:.3f}; "
        "warm colors indicate stronger model contribution",
        fontsize=12,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output
