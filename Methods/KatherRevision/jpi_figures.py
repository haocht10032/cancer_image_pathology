"""Focused figures for the compute-bounded JPI revision."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Methods.KatherRevision.statistics import architecture_primary_rows


COLORS = {
    "ResNet18": "#3266A8",
    "FrozenResNet18": "#7456A6",
    "DINOv2": "#D17A22",
    "UNI": "#23866B",
}
ORDER = ["ResNet18", "DINOv2", "UNI"]


def _style() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.08)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def create_jpi_figures(
    metrics: pd.DataFrame,
    classification: pd.DataFrame,
    repeat_calibration: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Create the essential classification and robustness figures."""
    _style()
    output_dir = Path(output_dir)
    outputs: dict[str, Path] = {}

    classification_order = [name for name in [*ORDER, "FrozenResNet18"] if name in set(classification["model"])]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for axis, metric, label in zip(
        axes,
        ("accuracy", "balanced_accuracy", "macro_f1"),
        ("Accuracy", "Balanced accuracy", "Macro-F1"),
    ):
        sns.stripplot(
            data=classification,
            x="model",
            y=metric,
            order=classification_order,
            hue="model",
            palette=COLORS,
            jitter=0.10,
            alpha=0.75,
            legend=False,
            ax=axis,
        )
        sns.pointplot(
            data=classification,
            x="model",
            y=metric,
            order=classification_order,
            color="#171717",
            markers="D",
            linestyles="none",
            errorbar=None,
            ax=axis,
        )
        axis.set(xlabel="", ylabel=label)
        axis.tick_params(axis="x", rotation=24)
    outputs["classification"] = _save(
        figure, output_dir / "01_classification_seeds.png"
    )

    all_primary = architecture_primary_rows(metrics)
    primary = all_primary[
        all_primary["analysis_family"].eq("primary_jointly_correct_true_class")
    ]
    selected14 = primary[
        primary["cohort_name"].eq("selected_faithfulness")
        & primary["grid_label"].eq("common_14")
    ]
    source = (
        selected14.groupby(["model", "perturbation", "case_id"], as_index=False)
        .agg(
            spearman=("attribution_occlusion_spearman", "mean"),
            relative_auc=(
                "top_minus_random_relative_target_logit_reduction_auc",
                "mean",
            ),
        )
    )
    source["perturbation_label"] = source["perturbation"].map(
        {
            "normalized_zero": "Normalized zero",
            "gaussian_blur": "Gaussian blur",
            "local_mean": "Neighborhood mean",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for axis, metric, label in (
        (axes[0], "spearman", "Attribution-occlusion Spearman"),
        (axes[1], "relative_auc", "Top - random relative reduction AUC"),
    ):
        sns.pointplot(
            data=source,
            x="perturbation_label",
            y=metric,
            hue="model",
            hue_order=ORDER,
            palette=COLORS,
            dodge=0.28,
            errorbar=("ci", 95),
            ax=axis,
        )
        axis.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set(xlabel="", ylabel=label)
        axis.tick_params(axis="x", rotation=18)
    axes[1].legend_.remove()
    outputs["perturbation"] = _save(
        figure, output_dir / "02_perturbation_sensitivity.png"
    )

    grid = primary[
        primary["cohort_name"].eq("selected_faithfulness")
        & primary["perturbation"].eq("normalized_zero")
    ].copy()
    grid["grid"] = (
        grid["grid_label"]
        .str.replace("common_", "", regex=False)
        .str.replace("native_", "native ", regex=False)
    )
    grid_source = grid.groupby(["model", "grid", "case_id"], as_index=False)[
        "attribution_occlusion_spearman"
    ].mean()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    common_grid = grid_source[grid_source["grid"].isin(("7", "14"))]
    sns.pointplot(
        data=common_grid,
        x="grid",
        y="attribution_occlusion_spearman",
        hue="model",
        hue_order=ORDER,
        palette=COLORS,
        dodge=0.30,
        errorbar=("ci", 95),
        ax=axes[0],
    )
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel="Common image-space grid",
        ylabel="Source-level mean Spearman",
        title="Cross-model comparison",
    )
    native_grid = grid_source[grid_source["grid"].str.startswith("native")]
    sns.pointplot(
        data=native_grid,
        x="model",
        y="attribution_occlusion_spearman",
        order=ORDER,
        hue="model",
        palette=COLORS,
        errorbar=("ci", 95),
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set(
        xlabel="Native grid (5, 16, and 14)",
        ylabel="Source-level mean Spearman",
        title="Descriptive within-pipeline sensitivity",
    )
    outputs["grid"] = _save(figure, output_dir / "03_grid_sensitivity.png")

    random_cohort = primary[
        primary["cohort_name"].eq("random_prediction_independent")
        & primary["perturbation"].eq("normalized_zero")
        & primary["grid_label"].eq("common_14")
    ]
    random_image = random_cohort.groupby(["cohort_id", "model"], as_index=False)[
        [
            "attribution_occlusion_spearman",
            "top_minus_random_relative_target_logit_reduction_auc",
        ]
    ].mean()
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for axis, metric, label in (
        (axes[0], "attribution_occlusion_spearman", "Spearman"),
        (
            axes[1],
            "top_minus_random_relative_target_logit_reduction_auc",
            "Top - random relative reduction AUC",
        ),
    ):
        sns.boxplot(
            data=random_image,
            x="model",
            y=metric,
            order=ORDER,
            hue="model",
            palette=COLORS,
            showfliers=False,
            legend=False,
            ax=axis,
        )
        axis.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set(xlabel="", ylabel=label)
    outputs["random_cohort"] = _save(
        figure, output_dir / "04_random_cohort_validation.png"
    )

    frozen = all_primary[
        all_primary["analysis_family"].eq(
            "training_regime_jointly_correct_true_class"
        )
        & all_primary["perturbation"].eq("normalized_zero")
        & all_primary["grid_label"].eq("common_14")
    ]
    if not frozen.empty:
        frozen_image = frozen.groupby(["cohort_id", "model"], as_index=False)[
            [
                "attribution_occlusion_spearman",
                "top_minus_random_relative_target_logit_reduction_auc",
            ]
        ].mean()
        figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
        panels = (
            ("attribution_occlusion_spearman", "Attribution-occlusion Spearman"),
            (
                "top_minus_random_relative_target_logit_reduction_auc",
                "Top - random relative logit AUC",
            ),
        )
        for axis, (metric, label) in zip(axes, panels):
            sns.boxplot(
                data=frozen_image,
                x="model",
                y=metric,
                order=["ResNet18", "FrozenResNet18"],
                hue="model",
                palette=COLORS,
                showfliers=False,
                legend=False,
                ax=axis,
            )
            axis.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
            axis.set(xlabel="", ylabel=label)
            axis.tick_params(axis="x", rotation=18)
        outputs["training_regime"] = _save(
            figure, output_dir / "05_frozen_resnet_control.png"
        )

    if not repeat_calibration.empty:
        figure, axis = plt.subplots(figsize=(4.7, 3.8))
        sns.scatterplot(
            data=repeat_calibration,
            x="five_repeat_auc",
            y="twenty_repeat_auc",
            hue="model",
            hue_order=ORDER,
            palette=COLORS,
            s=35,
            alpha=0.75,
            ax=axis,
        )
        limits = [
            min(repeat_calibration["five_repeat_auc"].min(), repeat_calibration["twenty_repeat_auc"].min()),
            max(repeat_calibration["five_repeat_auc"].max(), repeat_calibration["twenty_repeat_auc"].max()),
        ]
        axis.plot(limits, limits, color="#555555", linestyle="--", linewidth=0.9)
        axis.set(
            xlabel="Random deletion AUC, 5 repeats",
            ylabel="Random deletion AUC, 20 repeats",
        )
        outputs["repeat_calibration"] = _save(
            figure, output_dir / "06_random_repeat_calibration.png"
        )
    return outputs
