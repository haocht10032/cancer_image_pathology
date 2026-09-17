"""Publication-quality figures for the Kather-5K robustness revision."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from Methods.KatherRevision.statistics import architecture_primary_rows


MODEL_ORDER = ["ResNet18", "DINOv2", "UNI"]
MODEL_COLORS = {
    "ResNet18": "#3B6FB6",
    "DINOv2": "#D17A22",
    "UNI": "#2D8C6F",
    "FrozenResNet18": "#7A5AA6",
}
PERTURBATION_LABELS = {
    "normalized_zero": "Normalized zero",
    "gaussian_blur": "Gaussian blur",
    "local_mean": "Neighborhood mean",
}


def _style() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#20252B",
            "text.color": "#20252B",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def classification_figure(classification: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro-F1"),
    ]
    available = [(column, label) for column, label in metrics if column in classification]
    fig, axes = plt.subplots(1, len(available), figsize=(4.0 * len(available), 3.5))
    axes = np.atleast_1d(axes)
    for axis, (column, label) in zip(axes, available):
        sns.stripplot(
            data=classification,
            x="model",
            y=column,
            order=[name for name in [*MODEL_ORDER, "FrozenResNet18"] if name in set(classification["model"])],
            hue="model",
            palette={**MODEL_COLORS, "FrozenResNet18": "#7A5AA6"},
            jitter=0.12,
            size=5,
            alpha=0.75,
            legend=False,
            ax=axis,
        )
        sns.pointplot(
            data=classification,
            x="model",
            y=column,
            order=[name for name in [*MODEL_ORDER, "FrozenResNet18"] if name in set(classification["model"])],
            color="#111111",
            errorbar=("ci", 95),
            markers="D",
            linestyles="none",
            ax=axis,
        )
        axis.set(xlabel="", ylabel=label, title=label)
        axis.tick_params(axis="x", rotation=25)
    _save(fig, output)


def primary_faithfulness_figure(metrics: pd.DataFrame, output: Path) -> None:
    frame = architecture_primary_rows(metrics)
    frame = frame[
        frame["analysis_family"].eq("primary_jointly_correct_true_class")
        & frame["perturbation"].eq("normalized_zero")
        & frame["grid_label"].eq("common_14")
    ]
    panels = [
        ("attribution_occlusion_spearman", "Attribution-occlusion Spearman"),
        (
            "top_minus_random_relative_target_logit_reduction_auc",
            "Top - random relative logit-reduction AUC",
        ),
        (
            "top_minus_random_relative_target_margin_reduction_auc",
            "Top - random relative margin-reduction AUC",
        ),
    ]
    image = frame.groupby(["cohort_name", "cohort_id", "model"], as_index=False)[
        [column for column, _ in panels]
    ].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    for axis, (column, label) in zip(axes, panels):
        sns.boxplot(
            data=image,
            x="model",
            y=column,
            hue="model",
            order=MODEL_ORDER,
            palette=MODEL_COLORS,
            showfliers=False,
            legend=False,
            ax=axis,
        )
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set(xlabel="", ylabel=label)
    _save(fig, output)


def perturbation_grid_figure(metrics: pd.DataFrame, output: Path) -> None:
    frame = architecture_primary_rows(metrics)
    frame = frame[frame["analysis_family"].eq("primary_jointly_correct_true_class")]
    summary = (
        frame.groupby(["model", "perturbation", "grid_label", "case_id"], as_index=False)[
            "attribution_occlusion_spearman"
        ].mean()
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    perturbation = summary[summary["grid_label"].eq("common_14")].copy()
    perturbation["Perturbation"] = perturbation["perturbation"].map(PERTURBATION_LABELS)
    sns.pointplot(
        data=perturbation,
        x="Perturbation",
        y="attribution_occlusion_spearman",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        errorbar=("ci", 95),
        dodge=0.25,
        markers="o",
        ax=axes[0],
    )
    axes[0].set(xlabel="", ylabel="Source-level mean Spearman", title="Perturbation sensitivity")
    axes[0].tick_params(axis="x", rotation=18)
    grid = summary[summary["perturbation"].eq("normalized_zero")].copy()
    grid = grid[grid["grid_label"].isin(("common_7", "common_14"))]
    grid["Grid"] = grid["grid_label"].map({"common_7": "7 x 7", "common_14": "14 x 14"})
    sns.pointplot(
        data=grid,
        x="Grid",
        y="attribution_occlusion_spearman",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        errorbar=("ci", 95),
        dodge=0.25,
        markers="o",
        ax=axes[1],
    )
    axes[1].set(xlabel="", ylabel="Source-level mean Spearman", title="Shared-grid sensitivity")
    axes[1].legend_.remove()
    _save(figure, output)


def common_explainer_figure(metrics: pd.DataFrame, output: Path) -> None:
    frame = metrics[
        metrics["method"].eq("integrated_gradients")
        & metrics["model"].isin(MODEL_ORDER)
        & metrics["analysis_family"].eq("primary_jointly_correct_true_class")
        & metrics["perturbation"].eq("normalized_zero")
        & metrics["grid_label"].eq("common_14")
    ]
    image = frame.groupby(["cohort_name", "cohort_id", "model"], as_index=False)[
        "attribution_occlusion_spearman"
    ].mean()
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    if image.empty:
        axis.text(
            0.5,
            0.5,
            "No jointly correct proof examples available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set(
            xlabel="",
            ylabel="Attribution-occlusion Spearman",
            title="Common explainer: Integrated Gradients",
            xticks=[],
            yticks=[],
        )
        _save(fig, output)
        return
    sns.violinplot(
        data=image,
        x="model",
        y="attribution_occlusion_spearman",
        order=MODEL_ORDER,
        hue="model",
        palette=MODEL_COLORS,
        inner="quartile",
        cut=0,
        legend=False,
        ax=axis,
    )
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    axis.set(xlabel="", ylabel="Attribution-occlusion Spearman", title="Common explainer: Integrated Gradients")
    _save(fig, output)


def heterogeneity_figure(metrics: pd.DataFrame, output: Path) -> None:
    frame = architecture_primary_rows(metrics)
    frame = frame[
        frame["perturbation"].eq("normalized_zero")
        & frame["grid_label"].eq("common_14")
    ]
    true_target = frame[
        frame["analysis_family"].eq("all_image_true_class_sensitivity")
    ]
    decision = frame[frame["analysis_family"].eq("actual_decision")]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    sns.pointplot(
        data=true_target,
        x="class_name",
        y="attribution_occlusion_spearman",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        errorbar=("ci", 95),
        dodge=0.35,
        ax=axes[0, 0],
    )
    axes[0, 0].set(xlabel="", ylabel="Spearman", title="Tissue-class heterogeneity")
    axes[0, 0].tick_params(axis="x", rotation=40)
    sns.pointplot(
        data=true_target,
        x="case_id",
        y="attribution_occlusion_spearman",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        errorbar=None,
        dodge=0.35,
        ax=axes[0, 1],
    )
    axes[0, 1].set(xlabel="Source group", ylabel="Spearman", title="Source-group heterogeneity")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].legend_.remove()
    decision = decision.copy()
    decision["Prediction"] = np.where(decision["correct"], "Correct", "Incorrect")
    sns.pointplot(
        data=decision,
        x="Prediction",
        y="attribution_occlusion_spearman",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        errorbar=("ci", 95),
        dodge=0.25,
        ax=axes[1, 0],
    )
    axes[1, 0].set(xlabel="", ylabel="Spearman", title="Actual-decision faithfulness")
    primary = frame[
        frame["analysis_family"].eq("primary_jointly_correct_true_class")
    ].copy()
    primary["Cohort"] = primary["cohort_name"].map(
        {
            "selected_faithfulness": "Selected cohort",
            "random_prediction_independent": "Random cohort",
        }
    ).fillna(primary["cohort_name"])
    sns.pointplot(
        data=primary,
        x="Cohort",
        y="attribution_occlusion_spearman",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        errorbar=("ci", 95),
        dodge=0.25,
        ax=axes[1, 1],
    )
    axes[1, 1].set(xlabel="", ylabel="Spearman", title="Cohort sensitivity")
    axes[1, 1].legend_.remove()
    _save(figure, output)


def deletion_curve_figure(curves: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    keys = architecture_primary_rows(metrics)[
        ["cohort_name", "cohort_id", "model", "seed", "method"]
    ].drop_duplicates()
    frame = curves.merge(keys, how="inner")
    frame = frame[
        frame["target_role"].eq("true_class")
        & frame["all_models_correct"]
        & frame["perturbation"].eq("normalized_zero")
        & frame["grid_label"].eq("common_14")
    ]
    image = frame.groupby(
        ["model", "strategy", "fraction_removed", "case_id"], as_index=False
    )["relative_target_logit_reduction"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), sharey=True)
    strategy_style = {"top": "-", "random": "--", "bottom": ":"}
    for axis, model in zip(axes, MODEL_ORDER):
        subset = image[image["model"].eq(model)]
        for strategy in ("top", "random", "bottom"):
            values = subset[subset["strategy"].eq(strategy)]
            summary = values.groupby("fraction_removed")[
                "relative_target_logit_reduction"
            ].agg(["mean", "sem"])
            x = summary.index.to_numpy()
            axis.plot(x, summary["mean"], strategy_style[strategy], label=strategy.title(), color=MODEL_COLORS[model])
            axis.fill_between(x, summary["mean"] - 1.96 * summary["sem"], summary["mean"] + 1.96 * summary["sem"], color=MODEL_COLORS[model], alpha=0.12)
        axis.set(title=model, xlabel="Fraction removed")
        axis.axhline(0, color="#888888", linewidth=0.7)
    axes[0].set_ylabel("Relative target-logit reduction")
    axes[-1].legend(frameon=False)
    _save(fig, output)


def stability_figure(
    attribution_stability: pd.DataFrame,
    prediction_stability: pd.DataFrame,
    output: Path,
) -> None:
    attribution_stability = attribution_stability[
        attribution_stability["model"].isin(MODEL_ORDER)
    ].copy()
    prediction_stability = prediction_stability[
        prediction_stability["model"].isin(MODEL_ORDER)
    ].copy()
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    if attribution_stability.empty:
        axes[0].text(
            0.5,
            0.5,
            "Requires at least two training seeds",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set(
            xlabel="",
            ylabel="Cross-seed map Spearman",
            title="Common true-class maps",
            xticks=[],
            yticks=[],
        )
    else:
        true_maps = attribution_stability[
            attribution_stability["target_mode"].eq("common_true_class")
            & attribution_stability["grid_label"].eq("common_14")
        ]
        sns.boxplot(
            data=true_maps,
            x="model",
            y="mean_attribution_spearman",
            order=MODEL_ORDER,
            hue="model",
            palette=MODEL_COLORS,
            showfliers=False,
            legend=False,
            ax=axes[0],
        )
        axes[0].set(
            xlabel="",
            ylabel="Cross-seed map Spearman",
            title="Common true-class maps",
        )
    if (
        prediction_stability.empty
        or prediction_stability["pairwise_prediction_agreement"].notna().sum() == 0
    ):
        axes[1].text(
            0.5,
            0.5,
            "Requires at least two training seeds",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set(
            xlabel="",
            ylabel="Pairwise agreement",
            title="Prediction agreement",
            xticks=[],
            yticks=[],
        )
    else:
        sns.boxplot(
            data=prediction_stability,
            x="model",
            y="pairwise_prediction_agreement",
            order=MODEL_ORDER,
            hue="model",
            palette=MODEL_COLORS,
            showfliers=False,
            legend=False,
            ax=axes[1],
        )
        axes[1].set(
            xlabel="", ylabel="Pairwise agreement", title="Prediction agreement"
        )
    _save(figure, output)


def representative_maps_figure(
    maps: pd.DataFrame,
    metrics: pd.DataFrame,
    output: Path,
    seed: int = 41,
    examples: int = 4,
) -> None:
    metric = architecture_primary_rows(metrics)
    metric = metric[
        metric["analysis_family"].eq("primary_jointly_correct_true_class")
        & metric["perturbation"].eq("normalized_zero")
        & metric["grid_label"].eq("common_14")
        & metric["seed"].eq(seed)
    ]
    candidates = (
        metric.groupby(["cohort_name", "cohort_id", "class_name", "path"])["model"]
        .nunique()
        .reset_index(name="model_count")
    )
    candidates = candidates[candidates["model_count"].eq(3)]
    candidates = candidates.groupby("class_name", as_index=False).head(1).head(examples)
    selected = maps[
        maps["target_role"].eq("true_class")
        & maps["all_models_correct"]
        & maps["grid_label"].eq("common_14")
        & maps["seed"].eq(seed)
    ]
    if candidates.empty:
        figure, axis = plt.subplots(figsize=(8.0, 2.5))
        axis.text(
            0.5,
            0.5,
            "No jointly correct three-model proof examples available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        _save(figure, output)
        return
    figure, axes = plt.subplots(
        len(candidates), 4, figsize=(10.5, 2.55 * len(candidates))
    )
    axes = np.atleast_2d(axes)
    for row_index, candidate in candidates.reset_index(drop=True).iterrows():
        image = np.asarray(Image.open(candidate["path"]).convert("RGB"))
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(str(candidate["class_name"]))
        for column, model in enumerate(MODEL_ORDER, start=1):
            method = "gradcam" if model == "ResNet18" else "gradient_attention_rollout"
            values = selected[
                selected["cohort_id"].eq(candidate["cohort_id"])
                & selected["model"].eq(model)
                & selected["method"].eq(method)
            ].sort_values("patch_index")["attribution"].to_numpy()
            axes[row_index, column].imshow(image)
            if len(values) == 196:
                axes[row_index, column].imshow(
                    values.reshape(14, 14),
                    cmap="magma",
                    alpha=0.55,
                    interpolation="bilinear",
                    extent=(0, image.shape[1], image.shape[0], 0),
                )
            axes[row_index, column].set_title(model)
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    _save(figure, output)


def robustness_figure(table: pd.DataFrame, output: Path) -> None:
    columns = [column for column in table if column.startswith("robust_")]
    values = table[columns].astype(float).to_numpy()
    fig, axis = plt.subplots(figsize=(8.6, 2.4 + 0.5 * len(table)))
    sns.heatmap(
        values,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette(["#C94C4C", "#E8B949", "#2D8C6F"], as_cmap=True),
        annot=np.where(values == 1, "Yes", "No"),
        fmt="",
        cbar=False,
        xticklabels=[column.replace("robust_", "").replace("_", " ") for column in columns],
        yticklabels=table["conclusion"],
        ax=axis,
    )
    axis.set(xlabel="", ylabel="")
    axis.tick_params(axis="x", rotation=25)
    _save(fig, output)


def create_revision_figures(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    classification: pd.DataFrame,
    attribution_stability: pd.DataFrame,
    prediction_stability: pd.DataFrame,
    maps: pd.DataFrame,
    robustness: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    _style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "classification": output_dir / "01_classification_across_seeds.png",
        "primary_faithfulness": output_dir / "02_scale_free_primary_faithfulness.png",
        "perturbation_grid": output_dir / "03_perturbation_and_grid_sensitivity.png",
        "integrated_gradients": output_dir / "04_common_integrated_gradients.png",
        "deletion_curves": output_dir / "05_scale_free_deletion_curves.png",
        "stability": output_dir / "06_cross_seed_stability.png",
        "representative_maps": output_dir / "07_representative_attribution_maps.png",
        "robustness": output_dir / "08_conclusion_robustness.png",
        "heterogeneity": output_dir / "09_heterogeneity_and_random_cohort.png",
    }
    classification_figure(classification, outputs["classification"])
    primary_faithfulness_figure(metrics, outputs["primary_faithfulness"])
    perturbation_grid_figure(metrics, outputs["perturbation_grid"])
    common_explainer_figure(metrics, outputs["integrated_gradients"])
    deletion_curve_figure(curves, metrics, outputs["deletion_curves"])
    stability_figure(attribution_stability, prediction_stability, outputs["stability"])
    representative_maps_figure(maps, metrics, outputs["representative_maps"])
    robustness_figure(robustness, outputs["robustness"])
    heterogeneity_figure(metrics, outputs["heterogeneity"])
    return outputs
