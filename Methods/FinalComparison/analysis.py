"""Focused final statistics for the three-model attribution study."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MODEL_ORDER = ("ResNet18", "DINOv2", "UNI")
PRIMARY_METHODS = {
    "ResNet18": "gradcam",
    "DINOv2": "gradient_attention_rollout",
    "UNI": "gradient_attention_rollout",
}
CLASSIFICATION_METRICS = (
    "accuracy",
    "source_macro_accuracy",
    "balanced_accuracy",
    "macro_f1",
)
FAITHFULNESS_METRICS = (
    "attribution_occlusion_spearman",
    "top_minus_random_target_logit_auc",
    "top_minus_random_margin_auc",
)
FIXED_DELETION_METRICS = tuple(
    f"top_{fraction}_{score}_drop"
    for fraction in (5, 10, 20, 30)
    for score in ("target_logit", "margin")
)
INTERVENTION_EXPLORATORY_DIRECTIONS = {
    "top_minus_random_target_logit_auc": -1,
    "top_minus_random_margin_auc": -1,
    **{
        f"top_{fraction}_{score}_drop": 1
        for fraction in (5, 10, 20, 30)
        for score in ("target_logit", "margin")
    },
}
STABILITY_METRICS = (
    "pairwise_prediction_agreement",
    "mean_predicted_class_spearman_same_prediction",
    "mean_common_true_class_spearman",
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _holm_adjust(values: pd.Series) -> pd.Series:
    raw = values.to_numpy(dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(raw) - rank) * raw[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def _add_holm(
    frame: pd.DataFrame,
    p_column: str,
    output_column: str,
    family_columns: Iterable[str],
) -> None:
    frame[output_column] = np.nan
    for _, indices in frame.groupby(list(family_columns), dropna=False).groups.items():
        valid = frame.loc[indices, p_column].dropna()
        if not valid.empty:
            frame.loc[valid.index, output_column] = _holm_adjust(valid)


def _bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    estimates = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
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
    grouped = {
        group: rows.groupby(unit_column, as_index=False)[value_column].mean()[
            value_column
        ].to_numpy()
        for group, rows in frame.groupby(group_column)
    }
    groups = np.asarray(list(grouped), dtype=object)
    if len(groups) == 0:
        return np.nan, np.nan
    estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_values: list[float] = []
        for group in sampled_groups:
            units = grouped[group]
            sampled_values.extend(
                rng.choice(units, size=len(units), replace=True).tolist()
            )
        estimates[iteration] = float(np.mean(sampled_values))
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def _wilcoxon(values: np.ndarray, alternative: str) -> tuple[float, float]:
    if len(values) == 0 or np.allclose(values, 0):
        return 0.0, 1.0
    result = wilcoxon(
        values,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def load_three_model_artifacts(project_root: Path | str) -> dict[str, pd.DataFrame]:
    """Load and validate the frozen three-model result artifacts."""
    root = Path(project_root)
    three_model = root / "artifacts" / "dinov2_three_model"
    grouped = root / "artifacts" / "grouped_oof_faithfulness"
    paths = {
        "classification": three_model / "classification" / "classification_metrics.csv",
        "per_class": three_model / "classification" / "per_class_accuracy.csv",
        "per_source": three_model / "classification" / "per_source_performance.csv",
        "faithfulness": three_model / "faithfulness" / "three_model_faithfulness_metrics.csv",
        "deletion": three_model / "faithfulness" / "three_model_deletion_curves.csv",
        "faithfulness_tests": three_model / "faithfulness" / "three_model_paired_comparisons.csv",
        "within_model_tests": three_model / "faithfulness" / "within_model_tests_against_random.csv",
        "cohort": three_model / "three_model_faithfulness_cohort.csv",
        "resnet18_stability": three_model / "stability" / "resnet18_stability_per_image.csv",
        "dinov2_stability": three_model / "stability" / "dinov2_stability_per_image.csv",
        "uni_stability": grouped / "stability" / "uni_stability_per_image.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing final-analysis artifacts:\n" + "\n".join(missing))
    artifacts = {name: pd.read_csv(path) for name, path in paths.items()}
    artifacts["stability"] = pd.concat(
        [
            artifacts.pop("resnet18_stability"),
            artifacts.pop("dinov2_stability"),
            artifacts.pop("uni_stability"),
        ],
        ignore_index=True,
    )

    cohort = artifacts["cohort"]
    if len(cohort) != 272 or cohort["cohort_id"].nunique() != 272:
        raise RuntimeError("Expected the frozen 272-image faithfulness cohort")
    if cohort["class_name"].nunique() != 8 or cohort["case_id"].nunique() != 10:
        raise RuntimeError("Final cohort must contain eight classes and ten source cases")
    if not cohort["selected_without_attribution"].astype(bool).all():
        raise RuntimeError("Cohort was not frozen independently of attribution results")
    for model, rows in artifacts["stability"].groupby("model"):
        if set(rows["cohort_id"]) != set(cohort["cohort_id"]):
            raise RuntimeError(f"{model} stability rows do not match the frozen cohort")
    return artifacts


def load_intervention_artifacts(project_root: Path | str) -> dict[str, pd.DataFrame]:
    """Load the completed UNI activation-patching ablation."""
    root = Path(project_root) / "artifacts" / "uni_intervention_causal"
    paths = {
        "metrics": root / "faithfulness" / "uni_intervention_faithfulness_metrics.csv",
        "curves": root / "faithfulness" / "uni_intervention_deletion_curves.csv",
        "stability": root / "stability" / "uni_intervention_stability_per_image.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The completed online intervention CSVs are required for the 55-error "
            "analysis. Missing:\n" + "\n".join(missing)
        )
    artifacts = {name: pd.read_csv(path) for name, path in paths.items()}
    methods = set(artifacts["metrics"]["method"])
    expected = {"gradient_attention_rollout", "intervention_activation_patch"}
    if methods != expected:
        raise RuntimeError(f"Unexpected intervention methods: {sorted(methods)}")
    return artifacts


def primary_faithfulness_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select each model's primary explanation for the common true-class target."""
    _require_columns(
        metrics,
        ["model", "method", "target_class", "true_class", "cohort_id"],
        "faithfulness metrics",
    )
    primary = metrics[
        metrics.apply(
            lambda row: PRIMARY_METHODS.get(str(row["model"])) == row["method"],
            axis=1,
        )
        & metrics["target_class"].eq(metrics["true_class"])
    ].copy()
    counts = primary.groupby("model")["cohort_id"].nunique().to_dict()
    if counts != {model: 272 for model in MODEL_ORDER}:
        raise RuntimeError(f"Primary true-class pairing is incomplete: {counts}")
    return primary


def classification_summary(classification: pd.DataFrame) -> pd.DataFrame:
    rows = classification[classification["scope"] == "aggregate_oof"].copy()
    summary = rows.groupby("model")[list(CLASSIFICATION_METRICS)].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reindex(MODEL_ORDER).reset_index()


def classification_seed_tests(
    classification: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = classification[classification["scope"] == "aggregate_oof"].copy()
    tests: list[dict[str, object]] = []
    pairs: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_seed)
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        first = rows[rows["model"] == model_a]
        second = rows[rows["model"] == model_b]
        paired = first[["seed", *CLASSIFICATION_METRICS]].merge(
            second[["seed", *CLASSIFICATION_METRICS]],
            on="seed",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        paired["model_a"] = model_a
        paired["model_b"] = model_b
        pairs.append(paired)
        for metric in CLASSIFICATION_METRICS:
            differences = (
                paired[f"{metric}_a"] - paired[f"{metric}_b"]
            ).to_numpy(dtype=float)
            statistic, p_value = _wilcoxon(differences, "two-sided")
            low, high = _bootstrap_mean_ci(
                differences, bootstrap_iterations, confidence, rng
            )
            tests.append(
                {
                    "comparison": f"{model_a} - {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "seed_pairs": len(differences),
                    "model_a_mean": float(paired[f"{metric}_a"].mean()),
                    "model_b_mean": float(paired[f"{metric}_b"].mean()),
                    "mean_paired_difference": float(differences.mean()),
                    "paired_bootstrap_ci_low": low,
                    "paired_bootstrap_ci_high": high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                }
            )
    result = pd.DataFrame(tests)
    _add_holm(result, "wilcoxon_p_value", "wilcoxon_p_holm", ["metric"])
    result["significant_holm_0.05"] = result["wilcoxon_p_holm"] < 0.05
    return result, pd.concat(pairs, ignore_index=True)


def classification_source_tests(
    per_source: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> pd.DataFrame:
    """Compare source-level accuracy with cases resampled before seeds."""
    tests: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        first = per_source[per_source["model"] == model_a]
        second = per_source[per_source["model"] == model_b]
        paired = first[["seed", "case_id", "accuracy"]].merge(
            second[["seed", "case_id", "accuracy"]],
            on=["seed", "case_id"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        paired["difference"] = paired["accuracy_a"] - paired["accuracy_b"]
        values = paired["difference"].to_numpy(dtype=float)
        statistic, p_value = _wilcoxon(values, "two-sided")
        low, high = _hierarchical_bootstrap_mean_ci(
            paired,
            "difference",
            "case_id",
            "seed",
            bootstrap_iterations,
            confidence,
            rng,
        )
        tests.append(
            {
                "comparison": f"{model_a} - {model_b}",
                "model_a": model_a,
                "model_b": model_b,
                "metric": "source_accuracy",
                "paired_source_seed_rows": len(paired),
                "source_group_count": int(paired["case_id"].nunique()),
                "mean_paired_difference": float(values.mean()),
                "hierarchical_bootstrap_ci_low": low,
                "hierarchical_bootstrap_ci_high": high,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
            }
        )
    result = pd.DataFrame(tests)
    result["wilcoxon_p_holm"] = _holm_adjust(result["wilcoxon_p_value"])
    result["significant_holm_0.05"] = result["wilcoxon_p_holm"] < 0.05
    result["hierarchical_ci_excludes_zero"] = (
        result["hierarchical_bootstrap_ci_low"]
        * result["hierarchical_bootstrap_ci_high"]
        > 0
    )
    result["supported_by_paired_and_hierarchical"] = (
        result["significant_holm_0.05"]
        & result["hierarchical_ci_excludes_zero"]
    )
    return result


def faithfulness_summary(primary: pd.DataFrame) -> pd.DataFrame:
    metrics = [*FAITHFULNESS_METRICS, *FIXED_DELETION_METRICS]
    summary = primary.groupby("model")[metrics].agg(["mean", "std", "median"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    counts = primary.groupby("model").agg(
        image_count=("cohort_id", "nunique"),
        source_group_count=("case_id", "nunique"),
    )
    return counts.join(summary).reindex(MODEL_ORDER).reset_index()


def focused_faithfulness_tests(tests: pd.DataFrame) -> pd.DataFrame:
    """Select final endpoints and apply Holm correction within endpoint families."""
    metrics = set((*FAITHFULNESS_METRICS, *FIXED_DELETION_METRICS))
    result = tests[
        tests["comparison_target"].eq("true_class")
        & tests["subgroup"].eq("all")
        & tests["metric"].isin(metrics)
    ].copy()
    result["final_test_family"] = np.select(
        [
            result["metric"].isin(FAITHFULNESS_METRICS),
            result["metric"].str.contains("target_logit_drop"),
            result["metric"].str.contains("margin_drop"),
        ],
        ["primary_faithfulness", "fixed_logit_deletion", "fixed_margin_deletion"],
        default="other",
    )
    _add_holm(
        result,
        "wilcoxon_p_value",
        "wilcoxon_p_holm_final_family",
        ["final_test_family"],
    )
    result["significant_final_family_0.05"] = (
        result["wilcoxon_p_holm_final_family"] < 0.05
    )
    result["hierarchical_ci_excludes_zero"] = (
        result["hierarchical_bootstrap_ci_low"]
        * result["hierarchical_bootstrap_ci_high"]
        > 0
    )
    result["supported_by_paired_and_hierarchical"] = (
        result["significant_final_family_0.05"]
        & result["hierarchical_ci_excludes_zero"]
    )
    return result.sort_values(["final_test_family", "metric", "comparison"])


def stability_summary(stability: pd.DataFrame) -> pd.DataFrame:
    summary = stability.groupby("model")[list(STABILITY_METRICS)].agg(
        ["mean", "std", "median"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reindex(MODEL_ORDER).reset_index()


def stability_pairwise_tests(
    stability: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
) -> pd.DataFrame:
    tests: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for model_a, model_b in combinations(MODEL_ORDER, 2):
        first = stability[stability["model"] == model_a]
        second = stability[stability["model"] == model_b]
        identity = ["cohort_id", "case_id", "class_name", "cohort_stratum"]
        paired = first[[*identity, *STABILITY_METRICS]].merge(
            second[["cohort_id", *STABILITY_METRICS]],
            on="cohort_id",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        for metric in STABILITY_METRICS:
            complete = paired.dropna(subset=[f"{metric}_a", f"{metric}_b"]).copy()
            complete["difference"] = complete[f"{metric}_a"] - complete[f"{metric}_b"]
            values = complete["difference"].to_numpy(dtype=float)
            statistic, p_value = _wilcoxon(values, "two-sided")
            low, high = _hierarchical_bootstrap_mean_ci(
                complete,
                "difference",
                "case_id",
                "cohort_id",
                bootstrap_iterations,
                confidence,
                rng,
            )
            tests.append(
                {
                    "comparison": f"{model_a} - {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "pair_count": len(complete),
                    "source_group_count": int(complete["case_id"].nunique()),
                    "model_a_mean": float(complete[f"{metric}_a"].mean()),
                    "model_b_mean": float(complete[f"{metric}_b"].mean()),
                    "mean_paired_difference": float(values.mean()),
                    "hierarchical_bootstrap_ci_low": low,
                    "hierarchical_bootstrap_ci_high": high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                }
            )
    result = pd.DataFrame(tests)
    _add_holm(result, "wilcoxon_p_value", "wilcoxon_p_holm", ["metric"])
    result["significant_holm_0.05"] = result["wilcoxon_p_holm"] < 0.05
    result["hierarchical_ci_excludes_zero"] = (
        result["hierarchical_bootstrap_ci_low"]
        * result["hierarchical_bootstrap_ci_high"]
        > 0
    )
    result["supported_by_paired_and_hierarchical"] = (
        result["significant_holm_0.05"]
        & result["hierarchical_ci_excludes_zero"]
    )
    return result


def incorrect_intervention_exploratory(
    metrics: pd.DataFrame,
    bootstrap_iterations: int = 5000,
    confidence: float = 0.95,
    random_seed: int = 2027,
    expected_images: int = 55,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paired true-class deletion analysis for the 55 incorrect UNI predictions."""
    _require_columns(
        metrics,
        [
            "cohort_id",
            "case_id",
            "class_name",
            "seed",
            "model",
            "method",
            "correct",
            "target_class",
            "true_class",
            *INTERVENTION_EXPLORATORY_DIRECTIONS,
        ],
        "intervention metrics",
    )
    subset = metrics[
        metrics["model"].eq("UNI")
        & ~metrics["correct"].astype(bool)
        & metrics["target_class"].eq(metrics["true_class"])
        & metrics["method"].isin(
            ("intervention_activation_patch", "gradient_attention_rollout")
        )
    ].copy()
    method_counts = subset.groupby("method")["cohort_id"].nunique().to_dict()
    expected_counts = {
        "gradient_attention_rollout": expected_images,
        "intervention_activation_patch": expected_images,
    }
    if method_counts != expected_counts:
        raise RuntimeError(
            f"Expected {expected_images} paired incorrect UNI images, found {method_counts}"
        )
    intervention = subset[subset["method"] == "intervention_activation_patch"]
    gradient = subset[subset["method"] == "gradient_attention_rollout"]
    identity = ["cohort_id", "case_id", "class_name", "seed", "true_class"]
    metrics_to_pair = list(INTERVENTION_EXPLORATORY_DIRECTIONS)
    paired = intervention[[*identity, *metrics_to_pair]].merge(
        gradient[["cohort_id", "seed", "true_class", *metrics_to_pair]],
        on=["cohort_id", "seed", "true_class"],
        suffixes=("_intervention", "_gradient"),
        validate="one_to_one",
    )
    rng = np.random.default_rng(random_seed)
    tests: list[dict[str, object]] = []
    for metric, direction in INTERVENTION_EXPLORATORY_DIRECTIONS.items():
        paired[f"{metric}_raw_difference"] = (
            paired[f"{metric}_intervention"] - paired[f"{metric}_gradient"]
        )
        paired[f"{metric}_oriented_improvement"] = (
            paired[f"{metric}_raw_difference"] * direction
        )
        raw = paired[f"{metric}_raw_difference"].to_numpy(dtype=float)
        oriented = paired[f"{metric}_oriented_improvement"].to_numpy(dtype=float)
        statistic, p_two_sided = _wilcoxon(raw, "two-sided")
        _, p_intervention_better = _wilcoxon(oriented, "greater")
        raw_low, raw_high = _hierarchical_bootstrap_mean_ci(
            paired,
            f"{metric}_raw_difference",
            "case_id",
            "cohort_id",
            bootstrap_iterations,
            confidence,
            rng,
        )
        oriented_low, oriented_high = _hierarchical_bootstrap_mean_ci(
            paired,
            f"{metric}_oriented_improvement",
            "case_id",
            "cohort_id",
            bootstrap_iterations,
            confidence,
            rng,
        )
        tests.append(
            {
                "analysis": "exploratory_incorrect_uni_true_class",
                "metric": metric,
                "faithfulness_direction": direction,
                "image_pairs": len(paired),
                "source_group_count": int(paired["case_id"].nunique()),
                "intervention_mean": float(paired[f"{metric}_intervention"].mean()),
                "gradient_mean": float(paired[f"{metric}_gradient"].mean()),
                "mean_raw_difference": float(raw.mean()),
                "raw_hierarchical_ci_low": raw_low,
                "raw_hierarchical_ci_high": raw_high,
                "mean_oriented_improvement": float(oriented.mean()),
                "oriented_hierarchical_ci_low": oriented_low,
                "oriented_hierarchical_ci_high": oriented_high,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_two_sided": p_two_sided,
                "wilcoxon_p_intervention_better": p_intervention_better,
            }
        )
    result = pd.DataFrame(tests)
    result["wilcoxon_p_two_sided_holm"] = _holm_adjust(
        result["wilcoxon_p_two_sided"]
    )
    result["wilcoxon_p_intervention_better_holm"] = _holm_adjust(
        result["wilcoxon_p_intervention_better"]
    )
    result["intervention_better_holm_0.05"] = (
        result["wilcoxon_p_intervention_better_holm"] < 0.05
    ) & (result["oriented_hierarchical_ci_low"] > 0)
    by_class = subset.groupby(["class_name", "method"])[metrics_to_pair].agg(
        ["count", "mean"]
    )
    by_class.columns = [f"{metric}_{stat}" for metric, stat in by_class.columns]
    return result, paired, by_class.reset_index()
