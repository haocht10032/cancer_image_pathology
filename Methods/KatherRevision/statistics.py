"""Source-cluster inference and heterogeneity summaries for Kather-5K."""

from __future__ import annotations

import hashlib
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PRIMARY_METHODS = {
    "ResNet18": "gradcam",
    "FrozenResNet18": "gradcam",
    "DINOv2": "gradient_attention_rollout",
    "UNI": "gradient_attention_rollout",
}
SCALE_FREE_METRICS = (
    "attribution_occlusion_spearman",
    "normalized_top_minus_random_target_logit_auc",
    "normalized_top_minus_random_target_margin_auc",
    "top_minus_random_relative_target_logit_reduction_auc",
    "top_minus_random_relative_target_margin_reduction_auc",
    "top_minus_random_5_relative_target_logit_reduction",
    "top_minus_random_10_relative_target_logit_reduction",
    "top_minus_random_20_relative_target_logit_reduction",
    "top_minus_random_30_relative_target_logit_reduction",
    "top_minus_random_5_relative_target_margin_reduction",
    "top_minus_random_10_relative_target_margin_reduction",
    "top_minus_random_20_relative_target_margin_reduction",
    "top_minus_random_30_relative_target_margin_reduction",
)


def architecture_primary_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    expected = metrics["model"].map(PRIMARY_METHODS)
    return metrics[
        metrics["method"].eq(expected) & metrics["null_repetition"].isna()
    ].copy()


def integrated_gradients_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[
        metrics["method"].eq("integrated_gradients")
        & metrics["null_repetition"].isna()
    ].copy()


def _holm(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 0.0
    count = len(array)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * array[index])
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=values.index)


def seed_image_source_summaries(
    metrics: pd.DataFrame,
    value_columns: Iterable[str] = SCALE_FREE_METRICS,
) -> dict[str, pd.DataFrame]:
    """Separate optimization, image, and source-group variability."""
    values = [column for column in value_columns if column in metrics]
    grouping = [
        "cohort_name",
        "analysis_family",
        "target_role",
        "model",
        "method",
        "perturbation",
        "grid_label",
    ]
    seed_summary = (
        metrics.groupby([*grouping, "seed"], dropna=False)[values]
        .mean()
        .reset_index()
    )
    image_summary = (
        metrics.groupby(
            [*grouping, "cohort_id", "case_id", "class_name", "correct"],
            dropna=False,
        )[values]
        .mean()
        .reset_index()
    )
    source_summary = (
        image_summary.groupby([*grouping, "case_id"], dropna=False)[values]
        .mean()
        .reset_index()
    )
    return {
        "seed": seed_summary,
        "image": image_summary,
        "source": source_summary,
    }


def hierarchical_bootstrap_ci(
    values: pd.DataFrame,
    value_column: str,
    source_column: str = "case_id",
    image_column: str = "cohort_id",
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 2027,
) -> tuple[float, float, float]:
    """Resample source groups first and image means second."""
    image_values = (
        values.groupby([source_column, image_column], as_index=False)[value_column]
        .mean()
        .dropna(subset=[value_column])
    )
    sources = image_values[source_column].unique()
    if len(sources) < 2:
        return float(image_values[value_column].mean()), np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    source_frames = {
        source: image_values[image_values[source_column].eq(source)]
        for source in sources
    }
    for iteration in range(iterations):
        sampled_sources = rng.choice(sources, size=len(sources), replace=True)
        sampled_values = []
        for source in sampled_sources:
            frame = source_frames[source]
            sampled_values.extend(
                rng.choice(
                    frame[value_column].to_numpy(dtype=float),
                    size=len(frame),
                    replace=True,
                )
            )
        estimates[iteration] = float(np.mean(sampled_values))
    alpha = (1.0 - confidence) / 2.0
    return (
        float(image_values[value_column].mean()),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def exact_source_sign_flip(
    differences: pd.DataFrame,
    value_column: str,
    source_column: str = "case_id",
    alternative: str = "two-sided",
    monte_carlo_iterations: int = 100000,
    seed: int = 2027,
) -> tuple[float, float, int, str]:
    """Sign-flip source-level paired differences, exactly for 10 groups."""
    source_values = (
        differences.groupby(source_column)[value_column].mean().dropna().to_numpy()
    )
    observed = float(source_values.mean())
    group_count = len(source_values)
    if group_count == 0:
        return np.nan, np.nan, 0, "none"
    if group_count <= 15:
        signs = np.asarray(list(product((-1.0, 1.0), repeat=group_count)))
        null = (signs * source_values).mean(axis=1)
        inference = "exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(
            (-1.0, 1.0), size=(monte_carlo_iterations, group_count), replace=True
        )
        null = (signs * source_values).mean(axis=1)
        inference = "monte_carlo"
    if alternative == "greater":
        tail_count = np.sum(null >= observed)
    elif alternative == "less":
        tail_count = np.sum(null <= observed)
    else:
        tail_count = np.sum(np.abs(null) >= abs(observed))
    p_value = float(
        tail_count / len(null)
        if inference == "exact"
        else (tail_count + 1) / (len(null) + 1)
    )
    return observed, p_value, group_count, inference


def leave_one_source_out(
    values: pd.DataFrame,
    value_column: str,
    source_column: str = "case_id",
) -> pd.DataFrame:
    rows = []
    for source in sorted(values[source_column].dropna().unique()):
        retained = values[~values[source_column].eq(source)]
        rows.append(
            {
                "excluded_source_group": source,
                "retained_source_groups": int(retained[source_column].nunique()),
                "estimate": float(retained[value_column].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_model_inference(
    metrics: pd.DataFrame,
    metric_columns: Iterable[str] = SCALE_FREE_METRICS,
    model_pairs: tuple[tuple[str, str], ...] = (
        ("UNI", "ResNet18"),
        ("UNI", "DINOv2"),
        ("DINOv2", "ResNet18"),
    ),
    bootstrap_iterations: int = 5000,
    seed: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paired image inference after averaging repeated training seeds."""
    identity = ["cohort_id", "case_id"]
    family = [
        "cohort_name",
        "analysis_family",
        "target_role",
        "method_family",
        "perturbation",
        "grid_label",
    ]
    working = metrics.copy()
    working["method_family"] = np.where(
        working["method"].eq("integrated_gradients"),
        "integrated_gradients",
        "architecture_primary",
    )
    rows = []
    paired_out = []
    loso_out = []
    for keys, frame in working.groupby(family, dropna=False):
        labels = dict(zip(family, keys if isinstance(keys, tuple) else (keys,)))
        for metric in metric_columns:
            if metric not in frame:
                continue
            image_means = (
                frame.groupby([*identity, "model"], as_index=False)[metric].mean()
            )
            pivot = image_means.pivot(
                index=identity, columns="model", values=metric
            ).reset_index()
            for first, second in model_pairs:
                if first not in pivot or second not in pivot:
                    continue
                paired = pivot.dropna(subset=[first, second]).copy()
                paired["difference"] = paired[first] - paired[second]
                if paired.empty:
                    continue
                try:
                    statistic, wilcoxon_p = wilcoxon(
                        paired["difference"], alternative="two-sided"
                    )
                except ValueError:
                    statistic, wilcoxon_p = 0.0, 1.0
                estimate, ci_low, ci_high = hierarchical_bootstrap_ci(
                    paired,
                    "difference",
                    iterations=bootstrap_iterations,
                    seed=_seed_for_test(seed, labels, metric, first, second),
                )
                sign_estimate, sign_p, groups, sign_type = exact_source_sign_flip(
                    paired,
                    "difference",
                    seed=_seed_for_test(seed + 1, labels, metric, first, second),
                )
                rows.append(
                    {
                        **labels,
                        "metric": metric,
                        "model_a": first,
                        "model_b": second,
                        "difference_definition": "model_a_minus_model_b",
                        "image_count": int(len(paired)),
                        "source_group_count": groups,
                        "mean_difference": estimate,
                        "hierarchical_ci_low": ci_low,
                        "hierarchical_ci_high": ci_high,
                        "wilcoxon_statistic": float(statistic),
                        "wilcoxon_p": float(wilcoxon_p),
                        "source_sign_flip_estimate": sign_estimate,
                        "source_sign_flip_p": sign_p,
                        "source_sign_flip_type": sign_type,
                    }
                )
                paired_long = paired[[*identity, first, second, "difference"]].copy()
                for key, value in labels.items():
                    paired_long[key] = value
                paired_long["metric"] = metric
                paired_long["model_a"] = first
                paired_long["model_b"] = second
                paired_out.append(paired_long)
                loso = leave_one_source_out(paired, "difference")
                for key, value in labels.items():
                    loso[key] = value
                loso["metric"] = metric
                loso["model_a"] = first
                loso["model_b"] = second
                loso_out.append(loso)
    tests = pd.DataFrame(rows)
    if not tests.empty:
        holm_family = [
            "cohort_name",
            "analysis_family",
            "method_family",
            "perturbation",
            "grid_label",
            "metric",
        ]
        tests["wilcoxon_p_holm"] = tests.groupby(holm_family, dropna=False)[
            "wilcoxon_p"
        ].transform(_holm)
        tests["source_sign_flip_p_holm"] = tests.groupby(
            holm_family, dropna=False
        )["source_sign_flip_p"].transform(_holm)
    return (
        tests,
        pd.concat(paired_out, ignore_index=True) if paired_out else pd.DataFrame(),
        pd.concat(loso_out, ignore_index=True) if loso_out else pd.DataFrame(),
    )


def _seed_for_test(seed: int, labels: dict[str, object], *values: object) -> int:
    text = ":".join([str(seed), *map(str, labels.values()), *map(str, values)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def paired_top_vs_random(
    metrics: pd.DataFrame,
    bootstrap_iterations: int = 5000,
) -> pd.DataFrame:
    """Rank-based paired top-vs-random tests using image means, not seed rows."""
    columns = (
        "top_minus_random_relative_target_logit_reduction_auc",
        "top_minus_random_relative_target_margin_reduction_auc",
    )
    grouping = [
        "cohort_name",
        "analysis_family",
        "model",
        "method",
        "perturbation",
        "grid_label",
    ]
    rows = []
    for keys, frame in metrics.groupby(grouping, dropna=False):
        labels = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        for column in columns:
            image_values = (
                frame.groupby(["cohort_id", "case_id"], as_index=False)[column]
                .mean()
                .dropna(subset=[column])
            )
            try:
                statistic, p_value = wilcoxon(
                    image_values[column], alternative="greater"
                )
            except ValueError:
                statistic, p_value = 0.0, 1.0
            estimate, ci_low, ci_high = hierarchical_bootstrap_ci(
                image_values, column, iterations=bootstrap_iterations
            )
            sign_estimate, sign_p, groups, sign_type = exact_source_sign_flip(
                image_values, column, alternative="greater"
            )
            rows.append(
                {
                    **labels,
                    "metric": column,
                    "alternative": "top deletion more damaging than random",
                    "images": int(len(image_values)),
                    "source_groups": groups,
                    "mean_difference": estimate,
                    "median_difference": float(image_values[column].median()),
                    "hierarchical_ci_low": ci_low,
                    "hierarchical_ci_high": ci_high,
                    "proportion_top_beats_random": float(
                        (image_values[column] > 0).mean()
                    ),
                    "wilcoxon_statistic": float(statistic),
                    "wilcoxon_p": float(p_value),
                    "source_sign_flip_estimate": sign_estimate,
                    "source_sign_flip_p": sign_p,
                    "source_sign_flip_type": sign_type,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_p_holm"] = result.groupby(
            ["cohort_name", "analysis_family", "perturbation", "grid_label"],
            dropna=False,
        )["wilcoxon_p"].transform(_holm)
        result["source_sign_flip_p_holm"] = result.groupby(
            ["cohort_name", "analysis_family", "perturbation", "grid_label"],
            dropna=False,
        )["source_sign_flip_p"].transform(_holm)
    return result


def heterogeneity_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Class/source/correctness/confidence summaries with robust location statistics."""
    frames = []
    subgroup_definitions = {
        "overall": [],
        "class": ["class_name"],
        "source_group": ["case_id"],
        "correctness": ["correct"],
        "confidence": ["confidence_group"],
    }
    base = [
        "cohort_name",
        "analysis_family",
        "model",
        "method",
        "perturbation",
        "grid_label",
    ]
    for subgroup, columns in subgroup_definitions.items():
        grouped = metrics.groupby([*base, *columns], dropna=False)
        summary = grouped.agg(
            image_seed_rows=("cohort_id", "size"),
            images=("cohort_id", "nunique"),
            seeds=("seed", "nunique"),
            source_groups=("case_id", "nunique"),
            spearman_mean=("attribution_occlusion_spearman", "mean"),
            spearman_median=("attribution_occlusion_spearman", "median"),
            spearman_q1=("attribution_occlusion_spearman", lambda x: x.quantile(0.25)),
            spearman_q3=("attribution_occlusion_spearman", lambda x: x.quantile(0.75)),
            positive_spearman_fraction=(
                "attribution_occlusion_spearman",
                lambda x: float((x > 0).mean()),
            ),
            top_beats_random_fraction=(
                "top_beats_random_target_logit",
                "mean",
            ),
        ).reset_index()
        summary["subgroup_type"] = subgroup
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def adipose_empty_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    excluded = {"07_ADIPOSE", "08_EMPTY"}
    frames = []
    for label, frame in (
        ("all_classes", metrics),
        ("exclude_adipose_empty", metrics[~metrics["class_name"].isin(excluded)]),
    ):
        summary = (
            frame.groupby(
                [
                    "cohort_name",
                    "analysis_family",
                    "model",
                    "method",
                    "perturbation",
                    "grid_label",
                ],
                dropna=False,
            )
            .agg(
                images=("cohort_id", "nunique"),
                mean_spearman=("attribution_occlusion_spearman", "mean"),
                median_spearman=("attribution_occlusion_spearman", "median"),
                top_beats_random_fraction=("top_beats_random_target_logit", "mean"),
            )
            .reset_index()
        )
        summary["class_sensitivity"] = label
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def sensitivity_summary(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create prespecified perturbation, grid, explainer, target, and null tables."""
    grouping = [
        "cohort_name",
        "analysis_family",
        "model",
        "method",
        "perturbation",
        "grid_label",
    ]
    aggregations = dict(
        image_seed_rows=("cohort_id", "size"),
        images=("cohort_id", "nunique"),
        seeds=("seed", "nunique"),
        source_groups=("case_id", "nunique"),
        mean_spearman=("attribution_occlusion_spearman", "mean"),
        median_spearman=("attribution_occlusion_spearman", "median"),
        positive_spearman_fraction=(
            "attribution_occlusion_spearman",
            lambda values: float((values > 0).mean()),
        ),
        top_beats_random_fraction=("top_beats_random_target_logit", "mean"),
        mean_scale_free_logit_auc=(
            "top_minus_random_relative_target_logit_reduction_auc",
            "mean",
        ),
        mean_scale_free_margin_auc=(
            "top_minus_random_relative_target_margin_reduction_auc",
            "mean",
        ),
        supportive_spearman=("attribution_supportive_occlusion_spearman", "mean"),
        absolute_spearman=("attribution_absolute_occlusion_spearman", "mean"),
        supportive_effect=("mean_supportive_occlusion_effect", "mean"),
        absolute_effect=("mean_absolute_occlusion_effect", "mean"),
        inhibitory_effect=("mean_inhibitory_occlusion_effect", "mean"),
    )
    nonnull = metrics[metrics["null_repetition"].isna()].copy()
    overall = nonnull.groupby(grouping, dropna=False).agg(**aggregations).reset_index()
    return {
        "perturbation_sensitivity": overall,
        "grid_resolution_sensitivity": overall,
        "integrated_gradients_summary": overall[
            overall["method"].eq("integrated_gradients")
        ].copy(),
        "architecture_explainer_summary": overall[
            overall["method"].isin(set(PRIMARY_METHODS.values()))
        ].copy(),
        "higher_resolution_cam_summary": overall[
            overall["method"].isin(("gradcam_higher", "gradcam_multilevel"))
        ].copy(),
        "random_attribution_null": (
            metrics[metrics["method"].eq("random_attribution")]
            .groupby(grouping, dropna=False)
            .agg(**aggregations)
            .reset_index()
        ),
    }


def conclusion_robustness_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Audit the conservative primary conclusion across prespecified axes."""
    primary = architecture_primary_rows(metrics)
    primary = primary[
        primary["analysis_family"].eq("primary_jointly_correct_true_class")
    ]
    selected_primary = primary[
        primary["cohort_name"].eq("selected_faithfulness")
    ]

    def winner(frame: pd.DataFrame, metric: str, higher_is_better: bool = True) -> str | None:
        means = frame.groupby("model")[metric].mean().dropna()
        if len(means) != 3:
            return None
        return str(means.idxmax() if higher_is_better else means.idxmin())

    zero14 = selected_primary[
        selected_primary["perturbation"].eq("normalized_zero")
        & selected_primary["grid_label"].eq("common_14")
    ]
    ranking_metrics = (
        ("attribution_occlusion_spearman", True),
        ("top_minus_random_relative_target_logit_reduction_auc", True),
        ("top_minus_random_relative_target_margin_reduction_auc", True),
    )

    def uni_wins_all(frame: pd.DataFrame) -> bool:
        return all(
            winner(frame, metric, higher_is_better) == "UNI"
            for metric, higher_is_better in ranking_metrics
        )

    score_robust = all(
        (
            winner(zero14, metric, higher_is_better) == "UNI"
        )
        for metric, higher_is_better in (
            ("attribution_occlusion_spearman", True),
            ("normalized_top_minus_random_target_logit_auc", False),
            ("normalized_top_minus_random_target_margin_auc", False),
            ("top_minus_random_relative_target_logit_reduction_auc", True),
            ("top_minus_random_relative_target_margin_reduction_auc", True),
        )
    )
    perturbation_frames = list(
        selected_primary[selected_primary["grid_label"].eq("common_14")].groupby(
            "perturbation"
        )
    )
    perturbation_robust = bool(perturbation_frames) and all(
        uni_wins_all(frame) for _, frame in perturbation_frames
    )
    resolution_frames = list(
        selected_primary[
            selected_primary["perturbation"].eq("normalized_zero")
            & selected_primary["grid_label"].isin(("common_14", "common_7"))
        ].groupby("grid_label")
    )
    resolution_robust = bool(resolution_frames) and all(
        uni_wins_all(frame) for _, frame in resolution_frames
    )
    cohort_frames = list(
        primary[
            primary["perturbation"].eq("normalized_zero")
            & primary["grid_label"].eq("common_14")
        ].groupby("cohort_name")
    )
    cohort_robust = len(cohort_frames) >= 2 and all(
        uni_wins_all(frame) for _, frame in cohort_frames
    )
    source_supported = []
    for metric, _ in ranking_metrics:
        image_means = (
            zero14.groupby(["cohort_id", "case_id", "model"], as_index=False)[
                metric
            ].mean()
        )
        source_pivot = image_means.pivot(
            index=["cohort_id", "case_id"], columns="model", values=metric
        ).reset_index()
        for comparator in ("ResNet18", "DINOv2"):
            if not {"UNI", comparator}.issubset(source_pivot.columns):
                source_supported.append(False)
                continue
            paired = source_pivot.dropna(subset=["UNI", comparator]).copy()
            paired["difference"] = paired["UNI"] - paired[comparator]
            _, ci_low, _ = hierarchical_bootstrap_ci(
                paired, "difference", iterations=2000
            )
            _, p_value, _, _ = exact_source_sign_flip(
                paired, "difference", alternative="greater"
            )
            source_supported.append(bool(ci_low > 0 and p_value <= 0.05))
    source_robust = bool(source_supported and all(source_supported))
    absolute_modest = bool(
        zero14.groupby("model")["attribution_occlusion_spearman"].mean().max()
        < 0.30
    )
    return pd.DataFrame(
        [
            {
                "conclusion": (
                    "UNI + gradient-weighted rollout ranks highest under the "
                    "primary evaluation"
                ),
                "robust_across_score_normalization": score_robust,
                "robust_across_perturbation_type": perturbation_robust,
                "robust_across_spatial_resolution": resolution_robust,
                "robust_on_random_cohort": cohort_robust,
                "robust_to_source_group_inference": source_robust,
            },
            {
                "conclusion": "Absolute attribution-occlusion agreement remains modest",
                "robust_across_score_normalization": absolute_modest,
                "robust_across_perturbation_type": absolute_modest,
                "robust_across_spatial_resolution": absolute_modest,
                "robust_on_random_cohort": absolute_modest,
                "robust_to_source_group_inference": absolute_modest,
            },
        ]
    )


def save_revision_tables(
    metrics: pd.DataFrame,
    output_dir: Path | str,
    bootstrap_iterations: int = 5000,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = architecture_primary_rows(metrics)
    ig = integrated_gradients_rows(metrics)
    analysis_rows = pd.concat((primary, ig), ignore_index=True)
    summaries = seed_image_source_summaries(analysis_rows)
    sensitivities = sensitivity_summary(metrics)
    tests, paired, loso = paired_model_inference(
        analysis_rows, bootstrap_iterations=bootstrap_iterations
    )
    tables = {
        "image_seed_metrics": metrics,
        "seed_summaries": summaries["seed"],
        "image_summaries": summaries["image"],
        "source_group_summaries": summaries["source"],
        "paired_model_tests": tests,
        "paired_model_values": paired,
        "leave_one_source_out": loso,
        "paired_top_vs_random": paired_top_vs_random(
            analysis_rows, bootstrap_iterations=bootstrap_iterations
        ),
        "heterogeneity": heterogeneity_summary(analysis_rows),
        "adipose_empty_sensitivity": adipose_empty_sensitivity(analysis_rows),
        "final_conclusion_robustness": conclusion_robustness_table(metrics),
        **sensitivities,
    }
    paths = {}
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
