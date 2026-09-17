"""Publication-oriented figures for the group-aware evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_METHODS = {
    "CNN": "gradcam",
    "ResNet18": "gradcam",
    "DINOv2": "gradient_attention_rollout",
    "UNI": "gradient_attention_rollout",
}


def _save(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def _mean_sem(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    return (
        frame.groupby(group_columns)[value_column]
        .agg(["mean", "sem"])
        .reset_index()
        .fillna({"sem": 0.0})
    )


def create_main_figures(
    classification: pd.DataFrame,
    faithfulness: pd.DataFrame,
    deletion_curves: pd.DataFrame,
    stability: pd.DataFrame,
    class_names: list[str],
    output_dir: Path | str,
) -> list[Path]:
    """Create the six requested main comparison figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[Path] = []

    fold_metrics = classification[classification["scope"] == "fold"].copy()
    if not fold_metrics.empty:
        long = fold_metrics.melt(
            id_vars=["model", "seed", "fold"],
            value_vars=["accuracy", "balanced_accuracy", "macro_f1"],
            var_name="metric",
            value_name="score",
        )
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for axis, metric in zip(
            axes,
            ("accuracy", "balanced_accuracy", "macro_f1"),
        ):
            sns.boxplot(
                data=long[long["metric"] == metric],
                x="model",
                y="score",
                color="white",
                fliersize=0,
                ax=axis,
            )
            sns.stripplot(
                data=long[long["metric"] == metric],
                x="model",
                y="score",
                hue="seed",
                dodge=False,
                alpha=0.65,
                size=3,
                ax=axis,
            )
            axis.set_title(metric.replace("_", " ").title())
            axis.set_xlabel("")
            if axis.legend_ is not None:
                axis.legend_.remove()
        axes[0].set_ylabel("Held-out fold score")
        path = output_dir / "01_classification_across_folds_and_seeds.png"
        _save(figure, path)
        paths.append(path)

    primary = faithfulness[
        faithfulness["method"]
        == faithfulness["model"].map(PRIMARY_METHODS)
    ].copy()
    primary = primary[primary["target_class"] == primary["true_class"]]
    if not primary.empty:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(
            data=primary,
            x="model",
            y="attribution_occlusion_spearman",
            hue="model",
            dodge=False,
            showfliers=False,
            ax=axis,
        )
        sns.stripplot(
            data=primary,
            x="model",
            y="attribution_occlusion_spearman",
            color="0.25",
            alpha=0.35,
            size=2.5,
            ax=axis,
        )
        axis.axhline(0, color="0.3", linewidth=1)
        axis.set(
            title="Attribution agreement with patch occlusion",
            xlabel="",
            ylabel="Spearman correlation",
        )
        if axis.legend_ is not None:
            axis.legend_.remove()
        path = output_dir / "02_three_model_occlusion_correlation.png"
        _save(figure, path)
        paths.append(path)

    primary_curves = deletion_curves[
        (
            deletion_curves["method"]
            == deletion_curves["model"].map(PRIMARY_METHODS)
        )
        & (deletion_curves["target_class"] == deletion_curves["true_class"])
    ]
    if not primary_curves.empty:
        summary = _mean_sem(
            primary_curves,
            ["model", "strategy", "fraction_removed"],
            "target_class_logit_drop",
        )
        model_names = [
            model_name
            for model_name in ("ResNet18", "CNN", "DINOv2", "UNI")
            if model_name in set(primary_curves["model"])
        ]
        figure, axes = plt.subplots(
            1,
            len(model_names),
            figsize=(5.2 * len(model_names), 4),
            sharey=True,
        )
        axes = np.atleast_1d(axes)
        for axis, model_name in zip(axes, model_names):
            for strategy, values in summary[
                summary["model"] == model_name
            ].groupby("strategy"):
                values = values.sort_values("fraction_removed")
                x = values["fraction_removed"].to_numpy()
                mean = values["mean"].to_numpy()
                sem = values["sem"].to_numpy()
                axis.plot(x, mean, marker="o", label=strategy)
                axis.fill_between(x, mean - sem, mean + sem, alpha=0.18)
            axis.set(
                title=model_name,
                xlabel="Fraction of patches removed",
                ylabel="Target-class logit decrease",
            )
            axis.legend(title="Deletion")
        path = output_dir / "03_logit_deletion_curves.png"
        _save(figure, path)
        paths.append(path)

    if not primary.empty:
        class_order = list(range(len(class_names)))
        per_class = (
            primary.groupby(["model", "true_class"])[
                "attribution_occlusion_spearman"
            ]
            .mean()
            .reset_index()
        )
        figure, axis = plt.subplots(figsize=(11, 4.5))
        sns.barplot(
            data=per_class,
            x="true_class",
            y="attribution_occlusion_spearman",
            hue="model",
            order=class_order,
            ax=axis,
        )
        axis.axhline(0, color="0.3", linewidth=1)
        axis.set_xticks(class_order, class_names, rotation=45, ha="right")
        axis.set(
            title="Per-class faithfulness",
            xlabel="True tissue class",
            ylabel="Mean occlusion Spearman correlation",
        )
        path = output_dir / "04_per_class_faithfulness.png"
        _save(figure, path)
        paths.append(path)

        status = primary.copy()
        status["prediction_status"] = np.where(
            status["correct"],
            "correct",
            "incorrect",
        )
        figure, axis = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(
            data=status,
            x="prediction_status",
            y="attribution_occlusion_spearman",
            hue="model",
            showfliers=False,
            ax=axis,
        )
        axis.axhline(0, color="0.3", linewidth=1)
        axis.set(
            title="Faithfulness by prediction correctness",
            xlabel="",
            ylabel="Occlusion Spearman correlation",
        )
        path = output_dir / "05_correct_vs_incorrect_faithfulness.png"
        _save(figure, path)
        paths.append(path)

        auc_long = primary.melt(
            id_vars=["model", "cohort_id", "class_name", "correct"],
            value_vars=[
                "top_minus_random_target_logit_auc",
                "top_minus_random_margin_auc",
            ],
            var_name="outcome",
            value_name="top_minus_random_auc",
        )
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
        outcome_titles = {
            "top_minus_random_target_logit_auc": "Target-logit AUC",
            "top_minus_random_margin_auc": "True-class-margin AUC",
        }
        for axis, outcome in zip(axes, outcome_titles):
            subset = auc_long[auc_long["outcome"] == outcome]
            sns.boxplot(
                data=subset,
                x="model",
                y="top_minus_random_auc",
                hue="model",
                dodge=False,
                showfliers=False,
                ax=axis,
            )
            axis.axhline(0, color="0.3", linewidth=1)
            axis.set(
                title=outcome_titles[outcome],
                xlabel="",
                ylabel="Top AUC - random AUC",
            )
            if axis.legend_ is not None:
                axis.legend_.remove()
        path = output_dir / "06_top_minus_random_auc.png"
        _save(figure, path)
        paths.append(path)

    if not stability.empty:
        long = stability.melt(
            id_vars=["model", "class_name", "cohort_stratum"],
            value_vars=[
                "pairwise_prediction_agreement",
                "mean_predicted_class_spearman_same_prediction",
                "mean_common_true_class_spearman",
            ],
            var_name="stability_metric",
            value_name="value",
        )
        figure, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        metric_titles = {
            "pairwise_prediction_agreement": "Prediction agreement",
            "mean_predicted_class_spearman_same_prediction": (
                "Predicted-map stability\n(same prediction only)"
            ),
            "mean_common_true_class_spearman": "Common-target map stability",
        }
        for axis, metric in zip(axes, metric_titles):
            subset = long[long["stability_metric"] == metric]
            sns.boxplot(
                data=subset,
                x="model",
                y="value",
                hue="cohort_stratum",
                showfliers=False,
                ax=axis,
            )
            axis.set(title=metric_titles[metric], xlabel="", ylabel="")
            if axis is not axes[-1] and axis.legend_ is not None:
                axis.legend_.remove()
        if axes[-1].legend_ is not None:
            axes[-1].legend(
                title="Cohort stratum",
                fontsize=8,
                title_fontsize=8,
            )
        path = output_dir / "07_attribution_stability_across_seeds.png"
        _save(figure, path)
        paths.append(path)
    return paths


def create_three_model_heatmaps(
    cohort: pd.DataFrame,
    attribution_maps: pd.DataFrame,
    occlusion_scores: pd.DataFrame,
    output_dir: Path | str,
    images_per_class: int = 1,
) -> list[Path]:
    """Save identical-image attribution and occlusion panels for three models."""
    import matplotlib.pyplot as plt
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        cohort.sort_values(
            ["label", "cohort_stratum", "case_id", "relative_path"]
        )
        .groupby("label", group_keys=False)
        .head(images_per_class)
    )
    model_methods = PRIMARY_METHODS
    paths: list[Path] = []
    for _, row in selected.iterrows():
        with Image.open(row["path"]) as source:
            image = np.asarray(source.convert("RGB"))
        figure, axes = plt.subplots(2, 4, figsize=(14, 7))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(
            f"{row['class_name']}\n{row['cohort_stratum']}"
        )
        axes[1, 0].axis("off")
        model_order = (
            ("ResNet18", "DINOv2", "UNI")
            if "ResNet18" in set(attribution_maps["model"])
            else ("CNN", "DINOv2", "UNI")
        )
        for column, model_name in enumerate(model_order, start=1):
            method = model_methods[model_name]
            maps = attribution_maps[
                (attribution_maps["cohort_id"] == row["cohort_id"])
                & (attribution_maps["model"] == model_name)
                & (attribution_maps["method"] == method)
                & (
                    attribution_maps["target_class"]
                    == int(row["label"])
                )
            ]
            scores = occlusion_scores[
                (occlusion_scores["cohort_id"] == row["cohort_id"])
                & (occlusion_scores["model"] == model_name)
                & (
                    occlusion_scores["target_class"]
                    == int(row["label"])
                )
            ]
            if maps.empty or scores.empty:
                axes[0, column].text(
                    0.5,
                    0.5,
                    "Not available",
                    ha="center",
                    va="center",
                )
                axes[1, column].text(
                    0.5,
                    0.5,
                    "Not available",
                    ha="center",
                    va="center",
                )
            else:
                if (
                    "evaluation_grid_size" in maps
                    and pd.notna(maps["evaluation_grid_size"].iloc[0])
                ):
                    grid_size = int(maps["evaluation_grid_size"].iloc[0])
                else:
                    grid_size = int(round(len(maps) ** 0.5))
                attribution = maps.pivot(
                    index="patch_row",
                    columns="patch_column",
                    values="attribution",
                ).to_numpy()
                occlusion = scores.pivot(
                    index="patch_row",
                    columns="patch_column",
                    values="target_class_logit_drop",
                ).to_numpy()
                for axis, values, title in (
                    (axes[0, column], attribution, model_name),
                    (axes[1, column], occlusion, f"{model_name} occlusion"),
                ):
                    resized = np.asarray(
                        Image.fromarray(
                            values.astype(np.float32, copy=False),
                            mode="F",
                        ).resize(
                            (image.shape[1], image.shape[0]),
                            Image.Resampling.BILINEAR,
                        )
                    )
                    axis.imshow(image)
                    axis.imshow(resized, cmap="inferno", alpha=0.45)
                    axis.set_title(title)
                if attribution.shape != (grid_size, grid_size):
                    raise RuntimeError("Saved attribution grid metadata mismatch")
            for axis in (axes[0, column], axes[1, column]):
                axis.axis("off")
        axes[0, 0].axis("off")
        figure.tight_layout()
        path = output_dir / f"{row['cohort_id']}_three_model_heatmap.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
