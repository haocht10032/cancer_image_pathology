"""Publication-oriented figures for the final three-model comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw

from .analysis import (
    CLASSIFICATION_METRICS,
    FAITHFULNESS_METRICS,
    MODEL_ORDER,
    PRIMARY_METHODS,
    STABILITY_METRICS,
    primary_faithfulness_rows,
)


MODEL_COLORS = {
    "ResNet18": "#C44E52",
    "DINOv2": "#2A9D8F",
    "UNI": "#3A6EA5",
}
METHOD_COLORS = {
    "gradient_attention_rollout": "#3A6EA5",
    "intervention_activation_patch": "#D17A22",
}
STRATEGY_COLORS = {"top": "#C44E52", "random": "#777777", "bottom": "#2A9D8F"}


def _prepare_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )


def _save(figure: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _model_boxplot(axis: plt.Axes, frame: pd.DataFrame, value: str) -> None:
    sns.boxplot(
        data=frame,
        x="model",
        y=value,
        hue="model",
        order=MODEL_ORDER,
        palette=MODEL_COLORS,
        legend=False,
        showfliers=False,
        width=0.62,
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x="model",
        y=value,
        order=MODEL_ORDER,
        color="#222222",
        alpha=0.45,
        size=2.0,
        jitter=0.18,
        ax=axis,
    )
    axis.set_xlabel("")


def _classification_figure(classification: pd.DataFrame, output: Path) -> None:
    frame = classification[classification["scope"] == "aggregate_oof"].copy()
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_f1": "Macro-F1",
    }
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for axis, metric in zip(axes, labels):
        _model_boxplot(axis, frame, metric)
        axis.set_title(labels[metric])
        axis.set_ylabel("Score")
        axis.set_ylim(0.65, 0.9)
    _save(figure, output)


def _faithfulness_figure(primary: pd.DataFrame, output: Path) -> None:
    labels = {
        "attribution_occlusion_spearman": "Attribution-occlusion Spearman",
        "top_minus_random_target_logit_auc": "Top minus random logit AUC",
        "top_minus_random_margin_auc": "Top minus random margin AUC",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for axis, metric in zip(axes, FAITHFULNESS_METRICS):
        _model_boxplot(axis, primary, metric)
        axis.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
        axis.set_title(labels[metric])
        axis.set_ylabel("Per-image value")
    axes[1].text(
        0.02,
        0.03,
        "More negative favors top-ranked deletion",
        transform=axes[1].transAxes,
        fontsize=7,
    )
    _save(figure, output)


def _deletion_figure(deletion: pd.DataFrame, output: Path) -> None:
    primary = deletion[
        deletion.apply(
            lambda row: PRIMARY_METHODS.get(str(row["model"])) == row["method"],
            axis=1,
        )
        & deletion["target_class"].eq(deletion["true_class"])
    ].copy()
    figure, axes = plt.subplots(
        3, 2, figsize=(11, 10), sharex=True, constrained_layout=True
    )
    values = (
        ("target_class_logit_drop", "True-class logit decrease"),
        ("target_margin_drop", "True-class margin decrease"),
    )
    for row_index, model in enumerate(MODEL_ORDER):
        model_rows = primary[primary["model"] == model]
        for column_index, (value, label) in enumerate(values):
            axis = axes[row_index, column_index]
            for strategy in ("top", "random", "bottom"):
                rows = model_rows[model_rows["strategy"] == strategy]
                summary = rows.groupby("fraction_removed")[value].agg(["mean", "sem"])
                x = summary.index.to_numpy(dtype=float)
                mean = summary["mean"].to_numpy(dtype=float)
                error = 1.96 * summary["sem"].fillna(0).to_numpy(dtype=float)
                axis.plot(
                    x,
                    mean,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    color=STRATEGY_COLORS[strategy],
                    label=strategy.title(),
                )
                axis.fill_between(
                    x,
                    mean - error,
                    mean + error,
                    color=STRATEGY_COLORS[strategy],
                    alpha=0.12,
                    linewidth=0,
                )
            axis.axhline(0, color="#333333", linewidth=0.7)
            axis.set_title(f"{model}: {label}")
            axis.set_ylabel("Decrease from baseline")
            if row_index == 2:
                axis.set_xlabel("Fraction of input patches deleted")
    axes[0, 1].legend(frameon=False, loc="best")
    _save(figure, output)


def _fixed_deletion_figure(primary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, suffix, title in (
        (axes[0], "target_logit", "Top-patch true-class logit decrease"),
        (axes[1], "margin", "Top-patch true-class margin decrease"),
    ):
        for model in MODEL_ORDER:
            rows = primary[primary["model"] == model]
            means = []
            errors = []
            for fraction in (5, 10, 20, 30):
                values = rows[f"top_{fraction}_{suffix}_drop"]
                means.append(float(values.mean()))
                errors.append(float(1.96 * values.sem()))
            axis.errorbar(
                (5, 10, 20, 30),
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.5,
                color=MODEL_COLORS[model],
                label=model,
            )
        axis.axhline(0, color="#333333", linewidth=0.7)
        axis.set_title(title)
        axis.set_xlabel("Top-ranked patches deleted (%)")
        axis.set_ylabel("Mean decrease (95% image CI)")
    axes[1].legend(frameon=False)
    _save(figure, output)


def _stability_figure(stability: pd.DataFrame, output: Path) -> None:
    labels = {
        "pairwise_prediction_agreement": "Prediction agreement",
        "mean_predicted_class_spearman_same_prediction": "Conditional predicted-class map stability",
        "mean_common_true_class_spearman": "Common true-class map stability",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    for axis, metric in zip(axes, STABILITY_METRICS):
        _model_boxplot(axis, stability, metric)
        axis.set_title(labels[metric])
        axis.set_ylabel("Across-seed score")
        axis.set_ylim(-0.1, 1.05)
    _save(figure, output)


def _per_class_figure(primary: pd.DataFrame, output: Path) -> None:
    metrics = (
        "attribution_occlusion_spearman",
        "top_minus_random_target_logit_auc",
        "top_minus_random_margin_auc",
    )
    labels = ("Spearman", "Logit AUC", "Margin AUC")
    classes = list(dict.fromkeys(primary.sort_values("true_class")["class_name"]))
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), constrained_layout=True)
    for axis, metric, label in zip(axes, metrics, labels):
        table = (
            primary.groupby(["class_name", "model"])[metric]
            .mean()
            .unstack("model")
            .reindex(index=classes, columns=MODEL_ORDER)
        )
        sns.heatmap(
            table,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.7},
            ax=axis,
        )
        axis.set_title(f"Per-class {label}")
        axis.set_xlabel("")
        axis.set_ylabel("")
    _save(figure, output)


def _representative_data(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    three_model = project_root / "artifacts" / "dinov2_three_model"
    grouped = project_root / "artifacts" / "grouped_oof_faithfulness"
    cohort = pd.read_csv(three_model / "three_model_faithfulness_cohort.csv")
    dataset = (
        project_root
        / "Colorectal Histology MNIST"
        / "Kather_texture_2016_image_tiles_5000"
        / "Kather_texture_2016_image_tiles_5000"
    )
    missing = ~cohort["path"].map(lambda value: Path(value).is_file())
    cohort.loc[missing, "path"] = cohort.loc[missing, "relative_path"].map(
        lambda value: str(dataset / value)
    )
    maps = pd.concat(
        [
            pd.read_csv(three_model / "faithfulness" / "resnet18_attribution_maps.csv"),
            pd.read_csv(three_model / "faithfulness" / "dinov2_attribution_maps.csv"),
            pd.read_csv(grouped / "faithfulness" / "uni_attribution_maps.csv"),
        ],
        ignore_index=True,
    )
    occlusion = pd.concat(
        [
            pd.read_csv(three_model / "faithfulness" / "resnet18_patch_occlusion_scores.csv"),
            pd.read_csv(three_model / "faithfulness" / "dinov2_patch_occlusion_scores.csv"),
            pd.read_csv(grouped / "faithfulness" / "uni_patch_occlusion_scores.csv"),
        ],
        ignore_index=True,
    )
    return cohort, maps, occlusion


def _render_representative_panel(
    row: pd.Series,
    maps: pd.DataFrame,
    occlusion: pd.DataFrame,
    output: Path,
) -> None:
    with Image.open(row["path"]) as source:
        image = np.asarray(source.convert("RGB"))
    figure, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"True: {row['class_name']}\n{row['cohort_stratum']}")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    prediction_columns = {
        "ResNet18": "resnet18_predicted_class_name",
        "DINOv2": "dinov2_predicted_class_name",
        "UNI": "uni_predicted_class_name",
    }
    for column, model in enumerate(MODEL_ORDER, start=1):
        attribution_rows = maps[
            maps["cohort_id"].eq(row["cohort_id"])
            & maps["model"].eq(model)
            & maps["method"].eq(PRIMARY_METHODS[model])
            & maps["target_class"].eq(int(row["label"]))
        ]
        occlusion_rows = occlusion[
            occlusion["cohort_id"].eq(row["cohort_id"])
            & occlusion["model"].eq(model)
            & occlusion["target_class"].eq(int(row["label"]))
        ]
        if attribution_rows.empty or occlusion_rows.empty:
            raise RuntimeError(
                f"Missing representative true-class map for {row['cohort_id']} {model}"
            )
        attribution = attribution_rows.pivot(
            index="patch_row", columns="patch_column", values="attribution"
        ).to_numpy()
        occlusion_map = occlusion_rows.pivot(
            index="patch_row",
            columns="patch_column",
            values="target_class_logit_drop",
        ).to_numpy()
        for axis, values, title in (
            (
                axes[0, column],
                attribution,
                f"{model}\npred={row[prediction_columns[model]]}",
            ),
            (axes[1, column], occlusion_map, f"{model} input occlusion"),
        ):
            resized = np.asarray(
                Image.fromarray(values.astype(np.float32), mode="F").resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
                )
            )
            axis.imshow(image)
            axis.imshow(resized, cmap="inferno", alpha=0.45)
            axis.set_title(title)
            axis.axis("off")
    figure.tight_layout()
    _save(figure, output)


def _representative_montage(
    project_root: Path,
    cohort: pd.DataFrame,
    maps: pd.DataFrame,
    occlusion: pd.DataFrame,
    stratum: str,
    output: Path,
) -> None:
    selected = (
        cohort[cohort["cohort_stratum"].eq(stratum)]
        .sort_values(["label", "case_id", "relative_path"])
        .groupby("label", group_keys=False)
        .head(1)
    )
    if selected["label"].nunique() != 8:
        raise RuntimeError(f"Representative stratum {stratum} does not cover all classes")
    panel_dir = output.parent / "representative_panels" / stratum
    paths: list[Path] = []
    for _, row in selected.iterrows():
        panel_path = panel_dir / f"{row['cohort_id']}.png"
        _render_representative_panel(row, maps, occlusion, panel_path)
        paths.append(panel_path)
    images = [Image.open(path).convert("RGB") for path in paths]
    target_width = 1400
    resized = []
    for image in images:
        height = int(image.height * target_width / image.width)
        resized.append(image.resize((target_width, height), Image.Resampling.LANCZOS))
    tile_height = max(image.height for image in resized)
    margin = 24
    canvas = Image.new(
        "RGB",
        (2 * target_width + 3 * margin, 4 * tile_height + 5 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (path, image) in enumerate(zip(paths, resized)):
        column = index % 2
        row = index // 2
        x = margin + column * (target_width + margin)
        y = margin + row * (tile_height + margin)
        canvas.paste(image, (x, y))
        draw.text((x + 8, y + 8), path.stem, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _intervention_figure(
    intervention: Mapping[str, pd.DataFrame],
    incorrect_tests: pd.DataFrame,
    output: Path,
) -> None:
    metrics = intervention["metrics"]
    true_rows = metrics[metrics["target_class"].eq(metrics["true_class"])].copy()
    labels = {
        "gradient_attention_rollout": "Gradient rollout",
        "intervention_activation_patch": "Activation patching",
    }
    true_rows["method_label"] = true_rows["method"].map(labels)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, metric, title in (
        (axes[0, 0], "attribution_occlusion_spearman", "True-class occlusion correlation"),
        (axes[0, 1], "top_minus_random_target_logit_auc", "True-class logit AUC"),
    ):
        sns.boxplot(
            data=true_rows,
            x="method_label",
            y=metric,
            hue="method_label",
            palette=dict(zip(labels.values(), METHOD_COLORS.values())),
            legend=False,
            showfliers=False,
            ax=axis,
        )
        axis.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
        axis.set(title=title, xlabel="", ylabel="Per-image value")

    stability = intervention["stability"].copy()
    stability["method_label"] = stability["method"].map(labels)
    sns.boxplot(
        data=stability,
        x="method_label",
        y="mean_common_true_class_spearman",
        hue="method_label",
        palette=dict(zip(labels.values(), METHOD_COLORS.values())),
        legend=False,
        showfliers=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set(
        title="Common true-class stability",
        xlabel="",
        ylabel="Across-seed Spearman",
    )

    forest = incorrect_tests.copy().iloc[::-1]
    y = np.arange(len(forest))
    axes[1, 1].errorbar(
        forest["mean_oriented_improvement"],
        y,
        xerr=np.vstack(
            [
                forest["mean_oriented_improvement"] - forest["oriented_hierarchical_ci_low"],
                forest["oriented_hierarchical_ci_high"] - forest["mean_oriented_improvement"],
            ]
        ),
        fmt="o",
        color=METHOD_COLORS["intervention_activation_patch"],
        capsize=3,
    )
    axes[1, 1].axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    axes[1, 1].set_yticks(y, forest["metric"].str.replace("_", " "))
    axes[1, 1].set(
        title="Incorrect UNI predictions: intervention improvement",
        xlabel="Oriented difference (positive favors intervention)",
    )
    _save(figure, output)


def create_final_figures(
    project_root: Path | str,
    artifacts: Mapping[str, pd.DataFrame],
    output_dir: Path | str,
    intervention: Mapping[str, pd.DataFrame] | None = None,
    incorrect_tests: pd.DataFrame | None = None,
) -> list[Path]:
    """Create all final figures and return their output paths."""
    _prepare_style()
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = primary_faithfulness_rows(artifacts["faithfulness"])
    outputs = [
        output_dir / "01_classification_performance.png",
        output_dir / "02_primary_faithfulness.png",
        output_dir / "03_logit_margin_deletion_curves.png",
        output_dir / "04_fixed_deletion_effects.png",
        output_dir / "05_cross_seed_stability.png",
        output_dir / "06_per_class_faithfulness.png",
        output_dir / "07_representative_correct_attribution_maps.png",
        output_dir / "08_representative_incorrect_attribution_maps.png",
    ]
    _classification_figure(artifacts["classification"], outputs[0])
    _faithfulness_figure(primary, outputs[1])
    _deletion_figure(artifacts["deletion"], outputs[2])
    _fixed_deletion_figure(primary, outputs[3])
    _stability_figure(artifacts["stability"], outputs[4])
    _per_class_figure(primary, outputs[5])
    cohort, maps, occlusion = _representative_data(project_root)
    _representative_montage(
        project_root,
        cohort,
        maps,
        occlusion,
        "correct_high_confidence",
        outputs[6],
    )
    _representative_montage(
        project_root,
        cohort,
        maps,
        occlusion,
        "incorrect_any_model",
        outputs[7],
    )
    if intervention is not None and incorrect_tests is not None:
        ablation_path = output_dir / "09_intervention_negative_ablation.png"
        _intervention_figure(intervention, incorrect_tests, ablation_path)
        outputs.append(ablation_path)
    return outputs
