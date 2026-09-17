"""Focused common-seven attribution evaluation and paired external inference."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn

from Methods.KatherRevision.evaluation import CNNAdapter, TransformerAdapter
from Methods.UNIAttribution.model import load_classifier_head


PRIMARY_METHODS = {
    "ResNet18": "gradcam",
    "DINOv2": "gradient_attention_rollout",
    "UNI": "gradient_attention_rollout",
}
PRIMARY_METRICS = (
    "attribution_occlusion_spearman",
    "top_minus_random_relative_target_logit_reduction_auc",
    "top_beats_random_target_logit",
)


@torch.inference_mode()
def recompute_live_cohort_predictions(
    adapter: CNNAdapter | TransformerAdapter,
    cohort: pd.DataFrame,
    classification_predictions: pd.DataFrame,
    class_names: Iterable[str],
) -> pd.DataFrame:
    """Reconcile cached-feature predictions with live attribution forwards.

    Earlier external-classification runs used half-precision cached Transformer
    features. Attribution recomputes encoder features in full precision. This
    function freezes the live prediction used by the attribution model and records
    whether it differs from the classification CSV, without rerunning full-dataset
    inference or retraining a model.
    """
    names = tuple(class_names)
    selected = classification_predictions[
        classification_predictions["model"].eq(adapter.model_name)
        & classification_predictions["relative_path"].isin(cohort["relative_path"])
    ].copy()
    expected = len(cohort) * selected["seed"].nunique()
    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} cached cohort predictions for {adapter.model_name}, "
            f"found {len(selected)}"
        )
    rows = []
    cohort_lookup = cohort.set_index("relative_path", drop=False)
    for seed, seed_predictions in selected.groupby("seed", sort=True):
        adapter.load_checkpoint(0, int(seed))
        assert adapter.model is not None
        for _, cached_row in seed_predictions.sort_values("relative_path").iterrows():
            cohort_row = cohort_lookup.loc[cached_row["relative_path"]]
            image_path = str(
                cohort_row.get("path", cohort_row.get("image_path"))
            )
            image = adapter.load_image(image_path)
            logits = adapter.model(image).detach().cpu()[0]
            probabilities = torch.softmax(logits, dim=0)
            prediction = int(probabilities.argmax().item())
            true_class = int(
                cohort_row.get("common_label", cohort_row.get("label"))
            )
            row = cached_row.to_dict()
            row["classification_cached_prediction"] = int(cached_row["prediction"])
            row["classification_cached_confidence"] = float(cached_row["confidence"])
            row["prediction"] = prediction
            row["confidence"] = float(probabilities[prediction])
            row["correct"] = bool(prediction == true_class)
            row["predicted_class_name"] = names[prediction]
            row["prediction_changed_from_classification"] = bool(
                prediction != int(cached_row["prediction"])
            )
            for index, class_name in enumerate(names):
                row[f"logit_{index}_{class_name}"] = float(logits[index])
                row[f"probability_{index}_{class_name}"] = float(
                    probabilities[index]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _load_state_dict(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class ExternalCNNAdapter(CNNAdapter):
    """CNN adapter for one full-data checkpoint per training seed."""

    def load_checkpoint(self, fold: int, seed: int) -> nn.Module:
        del fold
        checkpoint = self.checkpoint_dir / f"seed_{int(seed)}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.model = self.model_builder().to(self.device)
        self.model.load_state_dict(_load_state_dict(checkpoint, self.device))
        self.model.eval()
        return self.model


class ExternalTransformerAdapter(TransformerAdapter):
    """Transformer adapter for one full-data classifier head per seed."""

    def load_checkpoint(self, fold: int, seed: int) -> nn.Module:
        del fold
        checkpoint = self.checkpoint_dir / f"seed_{int(seed)}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        assert self.model is not None
        load_classifier_head(self.model, checkpoint, map_location=self.device)
        self.model.eval()
        return self.model


def build_external_target_index(
    cohort: pd.DataFrame,
    predictions: pd.DataFrame,
    model_name: str,
    include_error_predicted_targets: bool = False,
) -> pd.DataFrame:
    """Create true-class targets plus predicted-class targets for errors."""
    model_predictions = predictions[predictions["model"].eq(model_name)].copy()
    required_seeds = set(model_predictions["seed"].unique())
    if not required_seeds:
        raise ValueError(f"No predictions found for {model_name}")
    cohort_frame = cohort.copy()
    if "path" not in cohort_frame:
        cohort_frame["path"] = cohort_frame["image_path"]
    if "common_class" not in cohort_frame:
        cohort_frame["common_class"] = cohort_frame["class_name"]
    if "common_label" not in cohort_frame:
        cohort_frame["common_label"] = cohort_frame["label"]
    merge_columns = [
        "relative_path",
        "model",
        "seed",
        "prediction",
        "confidence",
        "correct",
        "predicted_class_name",
    ]
    merged = cohort_frame.merge(
        model_predictions[merge_columns],
        on="relative_path",
        how="inner",
        validate="one_to_many",
    )
    expected = len(cohort_frame) * len(required_seeds)
    if len(merged) != expected:
        raise RuntimeError(
            f"Expected {expected} cohort prediction rows for {model_name}, "
            f"found {len(merged)}"
        )
    merged["fold"] = 0
    merged["class_name"] = merged["common_class"]
    merged["label"] = merged["common_label"].astype(int)
    merged["cohort_name"] = "crc_common_seven_external"
    merged["all_models_correct"] = False
    merged["case_id"] = merged.get("patient_id", pd.Series(pd.NA, index=merged.index))
    merged["case_id"] = merged["case_id"].fillna("unavailable").astype(str)
    median_confidence = merged.groupby(["model", "seed"])["confidence"].transform(
        "median"
    )
    merged["confidence_group"] = np.where(
        merged["confidence"] <= median_confidence, "low", "high"
    )

    true_rows = merged.copy()
    true_rows["target_role"] = "common_true_class"
    true_rows["target_class"] = true_rows["label"]
    true_rows["analysis_family"] = "all_image_true_class_sensitivity"

    if not include_error_predicted_targets:
        return true_rows.reset_index(drop=True)
    error_rows = merged[~merged["correct"]].copy()
    error_rows["target_role"] = "error_predicted_class"
    error_rows["target_class"] = error_rows["prediction"].astype(int)
    error_rows["analysis_family"] = "error_predicted_class"
    return pd.concat((true_rows, error_rows), ignore_index=True)


def _joint_correct_table(predictions: pd.DataFrame) -> pd.DataFrame:
    expected_models = set(PRIMARY_METHODS)
    available_models = set(predictions["model"].unique())
    if not expected_models.issubset(available_models):
        raise ValueError(
            f"Joint correctness requires {sorted(expected_models)}, found "
            f"{sorted(available_models)}"
        )
    working = predictions[predictions["model"].isin(expected_models)].copy()
    joint = (
        working.groupby(["relative_path", "seed"])
        .agg(
            model_count=("model", "nunique"),
            all_models_correct=("correct", "all"),
        )
        .reset_index()
    )
    joint["all_models_correct"] &= joint["model_count"].eq(len(expected_models))
    return joint[["relative_path", "seed", "all_models_correct"]]


def finalize_analysis_families(
    frame: pd.DataFrame,
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach joint correctness and duplicate the prespecified primary family."""
    joint = _joint_correct_table(all_predictions)
    output = frame.drop(columns="all_models_correct", errors="ignore").merge(
        joint,
        on=["relative_path", "seed"],
        how="left",
        validate="many_to_one",
    )
    output["all_models_correct"] = output["all_models_correct"].fillna(False)
    primary = output[
        output["target_role"].eq("common_true_class")
        & output["all_models_correct"]
    ].copy()
    primary["analysis_family"] = "primary_jointly_correct_true_class"
    return pd.concat((output, primary), ignore_index=True)


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


def _paired_bootstrap(
    paired: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, str, int]:
    rng = np.random.default_rng(seed)
    patient_available = (
        "case_id" in paired
        and not paired["case_id"].isin(["unavailable", "nan", "<NA>"]).all()
        and paired.loc[
            ~paired["case_id"].isin(["unavailable", "nan", "<NA>"]),
            "case_id",
        ].nunique()
        > 1
    )
    image_values = paired[["cohort_id", "case_id", "difference"]].dropna().copy()
    estimates = np.empty(iterations, dtype=float)
    if patient_available:
        image_values = image_values[
            ~image_values["case_id"].isin(["unavailable", "nan", "<NA>"])
        ]
        patients = image_values["case_id"].unique()
        patient_frames = {
            patient: image_values[image_values["case_id"].eq(patient)]
            for patient in patients
        }
        for iteration in range(iterations):
            sampled_patients = rng.choice(patients, size=len(patients), replace=True)
            values = []
            for patient in sampled_patients:
                patient_frame = patient_frames[patient]
                values.extend(
                    rng.choice(
                        patient_frame["difference"].to_numpy(dtype=float),
                        size=len(patient_frame),
                        replace=True,
                    )
                )
            estimates[iteration] = float(np.mean(values))
        inference_level = "patient_first_hierarchical_bootstrap"
        cluster_count = len(patients)
    else:
        values = image_values["difference"].to_numpy(dtype=float)
        for iteration in range(iterations):
            estimates[iteration] = float(
                rng.choice(values, size=len(values), replace=True).mean()
            )
        inference_level = "paired_tile_bootstrap_exploratory"
        cluster_count = len(values)
    return (
        float(image_values["difference"].mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        inference_level,
        int(cluster_count),
    )


def paired_external_inference(
    metrics: pd.DataFrame,
    metric_columns: Iterable[str] = PRIMARY_METRICS,
    model_pairs: tuple[tuple[str, str], ...] = (
        ("UNI", "ResNet18"),
        ("UNI", "DINOv2"),
        ("DINOv2", "ResNet18"),
    ),
    bootstrap_iterations: int = 5000,
    seed: int = 2027,
    analysis_family: str = "primary_jointly_correct_true_class",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average seeds within images, then compare models on paired images."""
    working = metrics[
        metrics["analysis_family"].eq(analysis_family)
        & metrics["perturbation"].eq("normalized_zero")
        & metrics["grid_label"].eq("common_14")
    ].copy()
    working = working[
        working["method"].eq(working["model"].map(PRIMARY_METHODS))
    ]
    tests = []
    paired_values = []
    for metric in metric_columns:
        if metric not in working:
            continue
        image_means = (
            working.groupby(
                ["cohort_id", "relative_path", "case_id", "class_name", "model"],
                as_index=False,
            )[metric]
            .mean()
        )
        pivot = image_means.pivot(
            index=["cohort_id", "relative_path", "case_id", "class_name"],
            columns="model",
            values=metric,
        ).reset_index()
        for first, second in model_pairs:
            if first not in pivot or second not in pivot:
                continue
            paired = pivot.dropna(subset=[first, second]).copy()
            paired["difference"] = paired[first] - paired[second]
            if paired.empty:
                continue
            try:
                statistic, p_value = wilcoxon(
                    paired["difference"], alternative="two-sided"
                )
            except ValueError:
                statistic, p_value = 0.0, 1.0
            estimate, ci_low, ci_high, inference_level, cluster_count = (
                _paired_bootstrap(
                    paired,
                    iterations=bootstrap_iterations,
                    seed=seed + len(tests),
                )
            )
            tests.append(
                {
                    "analysis_family": analysis_family,
                    "metric": metric,
                    "model_a": first,
                    "model_b": second,
                    "difference_definition": "model_a_minus_model_b",
                    "paired_images": int(len(paired)),
                    "cluster_count": cluster_count,
                    "mean_difference": estimate,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_inference_level": inference_level,
                    "wilcoxon_statistic": float(statistic),
                    "wilcoxon_p": float(p_value),
                }
            )
            paired["metric"] = metric
            paired["model_a"] = first
            paired["model_b"] = second
            paired_values.append(paired)
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty:
        test_frame["wilcoxon_p_holm"] = test_frame.groupby("metric")[
            "wilcoxon_p"
        ].transform(_holm)
    return (
        test_frame,
        pd.concat(paired_values, ignore_index=True)
        if paired_values
        else pd.DataFrame(),
    )
