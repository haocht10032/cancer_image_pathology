"""Paired Wilcoxon tests and image/source-group bootstrap intervals."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_METRICS = (
    "attribution_occlusion_spearman",
    "top_target_logit_auc",
    "random_target_logit_auc",
    "bottom_target_logit_auc",
    "top_minus_random_target_logit_auc",
    "top_margin_auc",
    "random_margin_auc",
    "bottom_margin_auc",
    "top_minus_random_margin_auc",
    "top_5_target_logit_drop",
    "top_10_target_logit_drop",
    "top_20_target_logit_drop",
    "top_30_target_logit_drop",
    "top_5_margin_drop",
    "top_10_margin_drop",
    "top_20_margin_drop",
    "top_30_margin_drop",
)

PRIMARY_METHODS = {
    "CNN": "gradcam",
    "ResNet18": "gradcam",
    "DINOv2": "gradient_attention_rollout",
    "UNI": "gradient_attention_rollout",
}

WITHIN_MODEL_METRICS = (
    "top_minus_random_target_logit_auc",
    "top_minus_random_margin_auc",
)

# Positive oriented differences always favor the first attribution method.
# Random deletion is a shared diagnostic and has no faithfulness direction.
ATTRIBUTION_METRIC_DIRECTIONS = {
    "attribution_occlusion_spearman": 1,
    "top_target_logit_auc": -1,
    "random_target_logit_auc": 0,
    "bottom_target_logit_auc": 1,
    "top_minus_random_target_logit_auc": -1,
    "top_margin_auc": -1,
    "random_margin_auc": 0,
    "bottom_margin_auc": 1,
    "top_minus_random_margin_auc": -1,
    "top_5_target_logit_drop": 1,
    "top_10_target_logit_drop": 1,
    "top_20_target_logit_drop": 1,
    "top_30_target_logit_drop": 1,
    "top_5_margin_drop": 1,
    "top_10_margin_drop": 1,
    "top_20_margin_drop": 1,
    "top_30_margin_drop": 1,
}


def _bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    estimates = values[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1 - alpha)),
    )


def _hierarchical_bootstrap_mean_ci(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    unit_column: str,
    iterations: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    groups = frame[group_column].drop_duplicates().to_numpy()
    if len(groups) == 0:
        return np.nan, np.nan
    grouped = {
        group: rows.groupby(unit_column, as_index=False)[value_column].mean()
        for group, rows in frame.groupby(group_column)
    }
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_values: list[float] = []
        for group in sampled_groups:
            units = grouped[group][value_column].to_numpy()
            sampled_values.extend(
                rng.choice(units, size=len(units), replace=True).tolist()
            )
        estimates[iteration] = float(np.mean(sampled_values))
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1 - alpha)),
    )


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def _format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.0001:
        return "p < 0.0001"
    return f"p = {value:.4f}"


def _model_rows(
    metrics: pd.DataFrame,
    model_name: str,
    comparison_target: str,
) -> pd.DataFrame:
    if model_name not in PRIMARY_METHODS:
        raise ValueError(f"No primary explanation configured for {model_name}")
    frame = metrics[
        (metrics["model"] == model_name)
        & (metrics["method"] == PRIMARY_METHODS[model_name])
    ].copy()
    if comparison_target == "true_class":
        return frame[frame["target_class"] == frame["true_class"]]
    elif comparison_target == "shared_prediction":
        return frame[frame["target_role"] == "predicted"]
    else:
        raise ValueError(
            "comparison_target must be 'true_class' or 'shared_prediction'"
        )


def _paired_rows(
    metrics: pd.DataFrame,
    model_a: str,
    model_b: str,
    comparison_target: str,
) -> pd.DataFrame:
    first = _model_rows(metrics, model_a, comparison_target)
    second = _model_rows(metrics, model_b, comparison_target)
    if comparison_target == "true_class":
        join = ["cohort_id", "seed", "true_class"]
    else:
        join = ["cohort_id", "seed", "target_class"]
    identity = [
        "path",
        "relative_path",
        "case_id",
        "class_name",
        "fold",
        "correct",
        "confidence_group",
    ]
    metric_columns = [column for column in DEFAULT_METRICS if column in metrics]
    paired = first[join + identity + metric_columns].merge(
        second[join + identity + metric_columns],
        on=join,
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    paired["prediction_status"] = np.select(
        [
            paired["correct_a"] & paired["correct_b"],
            (~paired["correct_a"]) & (~paired["correct_b"]),
        ],
        ["both_correct", "both_incorrect"],
        default="mixed",
    )
    paired["model_a"] = model_a
    paired["model_b"] = model_b
    paired["comparison_target"] = comparison_target
    return paired


def pairwise_model_comparisons(
    metrics: pd.DataFrame,
    model_pairs: Iterable[tuple[str, str]] = (
        ("ResNet18", "DINOv2"),
        ("ResNet18", "UNI"),
        ("DINOv2", "UNI"),
    ),
    metric_columns: Iterable[str] = DEFAULT_METRICS,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired comparisons for prespecified model-explanation pipelines."""
    all_paired: list[pd.DataFrame] = []
    result_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for model_a, model_b in model_pairs:
        for target in ("true_class", "shared_prediction"):
            paired = _paired_rows(
                metrics,
                model_a,
                model_b,
                target,
            )
            all_paired.append(paired)
            subgroups = {"all": paired}
            subgroups.update(
                {
                    str(status): rows
                    for status, rows in paired.groupby("prediction_status")
                }
            )
            for subgroup, subset in subgroups.items():
                for metric in metric_columns:
                    first_column = f"{metric}_a"
                    second_column = f"{metric}_b"
                    if first_column not in subset or second_column not in subset:
                        continue
                    complete = subset.dropna(
                        subset=[first_column, second_column]
                    ).copy()
                    if complete.empty:
                        continue
                    complete["paired_difference"] = (
                        complete[first_column] - complete[second_column]
                    )
                    differences = complete["paired_difference"].to_numpy(
                        dtype=float
                    )
                    if np.allclose(differences, 0):
                        statistic, p_value = 0.0, 1.0
                    else:
                        test = wilcoxon(
                            differences,
                            alternative="two-sided",
                            zero_method="wilcox",
                            method="auto",
                        )
                        statistic = float(test.statistic)
                        p_value = float(test.pvalue)
                    image_low, image_high = _bootstrap_mean_ci(
                        differences,
                        bootstrap_iterations,
                        confidence,
                        rng,
                    )
                    hierarchical_low, hierarchical_high = (
                        _hierarchical_bootstrap_mean_ci(
                            complete,
                            "paired_difference",
                            "case_id_a",
                            "cohort_id",
                            bootstrap_iterations,
                            confidence,
                            rng,
                        )
                    )
                    result_rows.append(
                        {
                            "comparison": f"{model_a} - {model_b}",
                            "model_a": model_a,
                            "model_b": model_b,
                            "comparison_target": target,
                            "subgroup": subgroup,
                            "metric": metric,
                            "pair_count": len(complete),
                            "source_group_count": int(
                                complete["case_id_a"].nunique()
                            ),
                            "model_a_mean": float(
                                complete[first_column].mean()
                            ),
                            "model_b_mean": float(
                                complete[second_column].mean()
                            ),
                            "mean_paired_difference": float(differences.mean()),
                            "wilcoxon_statistic": statistic,
                            "wilcoxon_p_value": p_value,
                            "paired_bootstrap_ci_low": image_low,
                            "paired_bootstrap_ci_high": image_high,
                            "hierarchical_bootstrap_ci_low": hierarchical_low,
                            "hierarchical_bootstrap_ci_high": hierarchical_high,
                            "confidence_level": confidence,
                            "bootstrap_iterations": bootstrap_iterations,
                        }
                    )
    results = pd.DataFrame(result_rows)
    if not results.empty:
        results["wilcoxon_p_holm"] = results.groupby(
            ["comparison_target", "subgroup", "metric"],
            group_keys=False,
        )["wilcoxon_p_value"].apply(_holm_adjust)
        results["significant_holm_0.05"] = (
            results["wilcoxon_p_holm"] < 0.05
        )
        results["wilcoxon_p_display"] = results[
            "wilcoxon_p_value"
        ].map(_format_p_value)
        results["wilcoxon_p_holm_display"] = results[
            "wilcoxon_p_holm"
        ].map(_format_p_value)
    return results, pd.concat(all_paired, ignore_index=True)


def within_model_random_deletion_tests(
    metrics: pd.DataFrame,
    model_names: Iterable[str] = ("ResNet18", "DINOv2", "UNI"),
    metric_columns: Iterable[str] = WITHIN_MODEL_METRICS,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> pd.DataFrame:
    """Test whether top-ranked deletion AUC is lower than random within model."""
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        frame = _model_rows(metrics, model_name, "true_class")
        subgroups = {
            "all": frame,
            "correct": frame[frame["correct"]],
            "incorrect": frame[~frame["correct"]],
        }
        for subgroup, subset in subgroups.items():
            for metric in metric_columns:
                if metric not in subset:
                    continue
                complete = subset.dropna(subset=[metric]).copy()
                if complete.empty:
                    continue
                values = complete[metric].to_numpy(dtype=float)
                if np.allclose(values, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    test = wilcoxon(
                        values,
                        alternative="less",
                        zero_method="wilcox",
                        method="auto",
                    )
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)
                image_low, image_high = _bootstrap_mean_ci(
                    values,
                    bootstrap_iterations,
                    confidence,
                    rng,
                )
                hierarchical_low, hierarchical_high = (
                    _hierarchical_bootstrap_mean_ci(
                        complete,
                        metric,
                        "case_id",
                        "cohort_id",
                        bootstrap_iterations,
                        confidence,
                        rng,
                    )
                )
                rows.append(
                    {
                        "model": model_name,
                        "method": PRIMARY_METHODS[model_name],
                        "comparison": "top AUC - random AUC < 0",
                        "subgroup": subgroup,
                        "metric": metric,
                        "image_count": len(complete),
                        "source_group_count": int(
                            complete["case_id"].nunique()
                        ),
                        "mean_difference": float(values.mean()),
                        "median_difference": float(np.median(values)),
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_value": p_value,
                        "paired_bootstrap_ci_low": image_low,
                        "paired_bootstrap_ci_high": image_high,
                        "hierarchical_bootstrap_ci_low": hierarchical_low,
                        "hierarchical_bootstrap_ci_high": hierarchical_high,
                        "confidence_level": confidence,
                        "bootstrap_iterations": bootstrap_iterations,
                    }
                )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["wilcoxon_p_holm"] = results.groupby(
            ["subgroup", "metric"],
            group_keys=False,
        )["wilcoxon_p_value"].apply(_holm_adjust)
        results["significant_holm_0.05"] = (
            results["wilcoxon_p_holm"] < 0.05
        )
        results["wilcoxon_p_display"] = results[
            "wilcoxon_p_value"
        ].map(_format_p_value)
        results["wilcoxon_p_holm_display"] = results[
            "wilcoxon_p_holm"
        ].map(_format_p_value)
    return results


def paired_method_comparisons(
    metrics: pd.DataFrame,
    metric_columns: Iterable[str] = DEFAULT_METRICS,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible UNI-versus-CNN paired comparison."""
    return pairwise_model_comparisons(
        metrics,
        model_pairs=(("UNI", "CNN"),),
        metric_columns=metric_columns,
        bootstrap_iterations=bootstrap_iterations,
        confidence=confidence,
        random_seed=random_seed,
    )


def paired_attribution_method_comparison(
    metrics: pd.DataFrame,
    model_name: str = "UNI",
    method_a: str = "intervention_activation_patch",
    method_b: str = "gradient_attention_rollout",
    metric_columns: Iterable[str] = DEFAULT_METRICS,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare two attribution methods on identical model-image-target rows."""
    required = {
        "cohort_id",
        "seed",
        "case_id",
        "model",
        "method",
        "target_role",
        "target_class",
        "true_class",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Faithfulness metrics are missing columns: {sorted(missing)}")
    frame = metrics[metrics["model"] == model_name].copy()
    result_rows: list[dict[str, object]] = []
    paired_outputs: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_seed)

    for comparison_target in ("true_class", "predicted_class"):
        if comparison_target == "true_class":
            target_frame = frame[frame["target_class"] == frame["true_class"]]
            join = ["cohort_id", "seed", "true_class", "target_class"]
        else:
            target_frame = frame[frame["target_role"] == "predicted"]
            join = ["cohort_id", "seed", "target_class"]
        first = target_frame[target_frame["method"] == method_a]
        second = target_frame[target_frame["method"] == method_b]
        identity = [
            "path",
            "relative_path",
            "case_id",
            "class_name",
            "fold",
            "correct",
            "confidence_group",
        ]
        available_metrics = [
            column for column in metric_columns if column in target_frame.columns
        ]
        paired = first[join + identity + available_metrics].merge(
            second[join + available_metrics],
            on=join,
            how="inner",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        paired["model"] = model_name
        paired["method_a"] = method_a
        paired["method_b"] = method_b
        paired["comparison_target"] = comparison_target
        paired_outputs.append(paired)
        subgroups = {
            "all": paired,
            "correct": paired[paired["correct"]],
            "incorrect": paired[~paired["correct"]],
        }
        subgroups.update(
            {
                f"confidence:{group}": rows
                for group, rows in paired.groupby("confidence_group")
            }
        )

        for subgroup, subset in subgroups.items():
            for metric in available_metrics:
                first_column = f"{metric}_a"
                second_column = f"{metric}_b"
                complete = subset.dropna(
                    subset=[first_column, second_column]
                ).copy()
                if complete.empty:
                    continue
                complete["raw_difference"] = (
                    complete[first_column] - complete[second_column]
                )
                raw = complete["raw_difference"].to_numpy(dtype=float)
                direction = ATTRIBUTION_METRIC_DIRECTIONS.get(metric, 1)
                if np.allclose(raw, 0):
                    statistic, p_two_sided = 0.0, 1.0
                else:
                    test = wilcoxon(
                        raw,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    )
                    statistic = float(test.statistic)
                    p_two_sided = float(test.pvalue)
                raw_low, raw_high = _bootstrap_mean_ci(
                    raw, bootstrap_iterations, confidence, rng
                )
                hierarchical_raw_low, hierarchical_raw_high = (
                    _hierarchical_bootstrap_mean_ci(
                        complete,
                        "raw_difference",
                        "case_id",
                        "cohort_id",
                        bootstrap_iterations,
                        confidence,
                        rng,
                    )
                )

                if direction == 0:
                    oriented = np.full_like(raw, np.nan)
                    p_greater = np.nan
                    oriented_low = oriented_high = np.nan
                    hierarchical_low = hierarchical_high = np.nan
                    interpretation = "shared random-deletion diagnostic"
                else:
                    complete["oriented_improvement"] = raw * direction
                    oriented = complete["oriented_improvement"].to_numpy(
                        dtype=float
                    )
                    if np.allclose(oriented, 0):
                        p_greater = 1.0
                    else:
                        p_greater = float(
                            wilcoxon(
                                oriented,
                                alternative="greater",
                                zero_method="wilcox",
                                method="auto",
                            ).pvalue
                        )
                    oriented_low, oriented_high = _bootstrap_mean_ci(
                        oriented, bootstrap_iterations, confidence, rng
                    )
                    hierarchical_low, hierarchical_high = (
                        _hierarchical_bootstrap_mean_ci(
                            complete,
                            "oriented_improvement",
                            "case_id",
                            "cohort_id",
                            bootstrap_iterations,
                            confidence,
                            rng,
                        )
                    )
                    interpretation = (
                        "positive oriented improvement favors intervention"
                    )

                result_rows.append(
                    {
                        "comparison": f"{method_a} - {method_b}",
                        "model": model_name,
                        "method_a": method_a,
                        "method_b": method_b,
                        "comparison_target": comparison_target,
                        "subgroup": subgroup,
                        "metric": metric,
                        "faithfulness_direction": int(direction),
                        "interpretation": interpretation,
                        "pair_count": int(len(complete)),
                        "source_group_count": int(complete["case_id"].nunique()),
                        "method_a_mean": float(complete[first_column].mean()),
                        "method_b_mean": float(complete[second_column].mean()),
                        "mean_raw_difference_a_minus_b": float(raw.mean()),
                        "raw_paired_bootstrap_ci_low": raw_low,
                        "raw_paired_bootstrap_ci_high": raw_high,
                        "raw_hierarchical_bootstrap_ci_low": hierarchical_raw_low,
                        "raw_hierarchical_bootstrap_ci_high": hierarchical_raw_high,
                        "mean_oriented_improvement": (
                            np.nan if direction == 0 else float(oriented.mean())
                        ),
                        "oriented_paired_bootstrap_ci_low": oriented_low,
                        "oriented_paired_bootstrap_ci_high": oriented_high,
                        "oriented_hierarchical_bootstrap_ci_low": hierarchical_low,
                        "oriented_hierarchical_bootstrap_ci_high": hierarchical_high,
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_two_sided": p_two_sided,
                        "wilcoxon_p_intervention_better": p_greater,
                        "confidence_level": confidence,
                        "bootstrap_iterations": bootstrap_iterations,
                    }
                )

    results = pd.DataFrame(result_rows)
    if not results.empty:
        results["wilcoxon_p_two_sided_holm"] = results.groupby(
            ["comparison_target", "subgroup"], group_keys=False
        )["wilcoxon_p_two_sided"].apply(_holm_adjust)
        directed = results["faithfulness_direction"] != 0
        results["wilcoxon_p_intervention_better_holm"] = np.nan
        results.loc[directed, "wilcoxon_p_intervention_better_holm"] = (
            results.loc[directed]
            .groupby(["comparison_target", "subgroup"], group_keys=False)[
                "wilcoxon_p_intervention_better"
            ]
            .apply(_holm_adjust)
        )
        results["intervention_better_holm_0.05"] = (
            results["wilcoxon_p_intervention_better_holm"] < 0.05
        )
    paired_values = pd.concat(paired_outputs, ignore_index=True)
    return results, paired_values


def paired_stability_method_comparison(
    stability_metrics: pd.DataFrame,
    method_a: str = "intervention_activation_patch",
    method_b: str = "gradient_attention_rollout",
    metric_columns: Iterable[str] = (
        "mean_predicted_class_spearman_same_prediction",
        "mean_common_true_class_spearman",
    ),
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare cross-seed map stability on the same images and source cases."""
    required = {
        "cohort_id",
        "case_id",
        "method",
        "class_name",
        "cohort_stratum",
        "all_seeds_correct",
        "any_seed_incorrect",
    }
    missing = required.difference(stability_metrics.columns)
    if missing:
        raise ValueError(f"Stability metrics are missing columns: {sorted(missing)}")
    available_metrics = [
        column for column in metric_columns if column in stability_metrics.columns
    ]
    if not available_metrics:
        raise ValueError("No requested stability metrics are available")

    first = stability_metrics[stability_metrics["method"] == method_a]
    second = stability_metrics[stability_metrics["method"] == method_b]
    identity = [
        "case_id",
        "class_name",
        "true_class",
        "fold",
        "cohort_stratum",
        "all_seeds_correct",
        "any_seed_incorrect",
        "pairwise_prediction_agreement",
    ]
    paired = first[["cohort_id", *identity, *available_metrics]].merge(
        second[["cohort_id", *available_metrics]],
        on="cohort_id",
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    paired["method_a"] = method_a
    paired["method_b"] = method_b

    subgroups = {
        "all": paired,
        "all_seeds_correct": paired[paired["all_seeds_correct"]],
        "any_seed_incorrect": paired[paired["any_seed_incorrect"]],
    }
    subgroups.update(
        {
            f"confidence:{group}": rows
            for group, rows in paired.groupby("cohort_stratum")
        }
    )
    subgroups.update(
        {
            f"class:{class_name}": rows
            for class_name, rows in paired.groupby("class_name")
        }
    )

    rng = np.random.default_rng(random_seed)
    result_rows: list[dict[str, object]] = []
    for subgroup, subset in subgroups.items():
        for metric in available_metrics:
            first_column = f"{metric}_a"
            second_column = f"{metric}_b"
            complete = subset.dropna(
                subset=[first_column, second_column]
            ).copy()
            if complete.empty:
                continue
            complete["paired_difference"] = (
                complete[first_column] - complete[second_column]
            )
            differences = complete["paired_difference"].to_numpy(dtype=float)
            if np.allclose(differences, 0):
                statistic = 0.0
                p_two_sided = 1.0
                p_method_a_more_stable = 1.0
            else:
                two_sided = wilcoxon(
                    differences,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(two_sided.statistic)
                p_two_sided = float(two_sided.pvalue)
                p_method_a_more_stable = float(
                    wilcoxon(
                        differences,
                        alternative="greater",
                        zero_method="wilcox",
                        method="auto",
                    ).pvalue
                )
            image_low, image_high = _bootstrap_mean_ci(
                differences, bootstrap_iterations, confidence, rng
            )
            hierarchical_low, hierarchical_high = (
                _hierarchical_bootstrap_mean_ci(
                    complete,
                    "paired_difference",
                    "case_id",
                    "cohort_id",
                    bootstrap_iterations,
                    confidence,
                    rng,
                )
            )
            result_rows.append(
                {
                    "comparison": f"{method_a} - {method_b}",
                    "subgroup": subgroup,
                    "metric": metric,
                    "pair_count": int(len(complete)),
                    "source_group_count": int(complete["case_id"].nunique()),
                    "method_a_mean": float(complete[first_column].mean()),
                    "method_b_mean": float(complete[second_column].mean()),
                    "mean_paired_difference": float(differences.mean()),
                    "paired_bootstrap_ci_low": image_low,
                    "paired_bootstrap_ci_high": image_high,
                    "hierarchical_bootstrap_ci_low": hierarchical_low,
                    "hierarchical_bootstrap_ci_high": hierarchical_high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_two_sided": p_two_sided,
                    "wilcoxon_p_method_a_more_stable": p_method_a_more_stable,
                    "confidence_level": confidence,
                    "bootstrap_iterations": bootstrap_iterations,
                }
            )

    results = pd.DataFrame(result_rows)
    if not results.empty:
        results["wilcoxon_p_two_sided_holm"] = results.groupby(
            "subgroup", group_keys=False
        )["wilcoxon_p_two_sided"].apply(_holm_adjust)
        results["wilcoxon_p_method_a_more_stable_holm"] = results.groupby(
            "subgroup", group_keys=False
        )["wilcoxon_p_method_a_more_stable"].apply(_holm_adjust)
        results["method_a_more_stable_holm_0.05"] = (
            results["wilcoxon_p_method_a_more_stable_holm"] < 0.05
        )
    return results, paired
