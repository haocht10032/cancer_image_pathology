"""Analysis-only upgrades for the completed reference-seed faithfulness run."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EPSILON = 1e-6
IDENTITY_CANDIDATES = (
    "cohort_id",
    "path",
    "relative_path",
    "case_id",
    "class_name",
    "true_class",
    "model",
    "method",
    "target_role",
    "target_class",
    "correct",
    "confidence_group",
    "fold",
    "seed",
)


def _trapz(frame: pd.DataFrame, column: str) -> float:
    ordered = frame.sort_values("fraction_removed")
    return float(np.trapz(ordered[column], ordered["fraction_removed"]))


def add_scale_free_deletion_metrics(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    identity_columns: Iterable[str] = IDENTITY_CANDIDATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive scale-free deletion metrics without rerunning a model.

    The completed three-model curves use ``target_class_logit`` and
    ``target_margin``. This function adds relative reductions from each image's
    unperturbed score and returns one enriched row per existing metric row.
    """
    identity = [column for column in identity_columns if column in curves.columns]
    required = {
        "strategy",
        "fraction_removed",
        "target_class_logit",
        "target_margin",
    }
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Legacy deletion curves are missing columns: {sorted(missing)}")

    upgraded_curves = curves.copy()
    baseline = (
        upgraded_curves[upgraded_curves["fraction_removed"].eq(0)]
        .groupby(identity, dropna=False)[["target_class_logit", "target_margin"]]
        .first()
        .rename(
            columns={
                "target_class_logit": "unperturbed_target_logit",
                "target_margin": "unperturbed_target_margin",
            }
        )
        .reset_index()
    )
    upgraded_curves = upgraded_curves.merge(
        baseline, on=identity, how="left", validate="many_to_one"
    )
    upgraded_curves["target_logit_drop"] = (
        upgraded_curves["unperturbed_target_logit"]
        - upgraded_curves["target_class_logit"]
    )
    upgraded_curves["target_margin_drop"] = (
        upgraded_curves["unperturbed_target_margin"]
        - upgraded_curves["target_margin"]
    )
    for score in ("target_logit", "target_margin"):
        baseline_column = f"unperturbed_{score}"
        upgraded_curves[f"relative_{score}_reduction"] = (
            upgraded_curves[f"{score}_drop"]
            / upgraded_curves[baseline_column].abs().clip(lower=EPSILON)
        )

    rows: list[dict[str, object]] = []
    for keys, frame in upgraded_curves.groupby(identity, dropna=False):
        labels = dict(zip(identity, keys if isinstance(keys, tuple) else (keys,)))
        strategies = {name: values for name, values in frame.groupby("strategy")}
        if not {"top", "random", "bottom"} <= set(strategies):
            continue
        row: dict[str, object] = {**labels}
        row["unperturbed_target_logit"] = float(
            frame["unperturbed_target_logit"].iloc[0]
        )
        row["unperturbed_target_margin"] = float(
            frame["unperturbed_target_margin"].iloc[0]
        )
        row["near_zero_logit_denominator"] = bool(
            abs(row["unperturbed_target_logit"]) < 0.25
        )
        row["near_zero_margin_denominator"] = bool(
            abs(row["unperturbed_target_margin"]) < 0.25
        )
        for score, legacy_column in (
            ("target_logit", "target_class_logit"),
            ("target_margin", "target_margin"),
        ):
            score_auc = {
                name: _trapz(values, legacy_column)
                for name, values in strategies.items()
            }
            relative_column = f"relative_{score}_reduction"
            relative_auc = {
                name: _trapz(values, relative_column)
                for name, values in strategies.items()
            }
            score_difference = score_auc["top"] - score_auc["random"]
            relative_difference = relative_auc["top"] - relative_auc["random"]
            row.update(
                {
                    f"top_{score}_auc": score_auc["top"],
                    f"random_{score}_auc": score_auc["random"],
                    f"bottom_{score}_auc": score_auc["bottom"],
                    f"top_minus_random_{score}_auc": score_difference,
                    f"normalized_top_minus_random_{score}_auc": (
                        score_difference / max(abs(score_auc["random"]), EPSILON)
                    ),
                    f"top_relative_{score}_reduction_auc": relative_auc["top"],
                    f"random_relative_{score}_reduction_auc": relative_auc["random"],
                    f"bottom_relative_{score}_reduction_auc": relative_auc["bottom"],
                    f"top_minus_random_relative_{score}_reduction_auc": (
                        relative_difference
                    ),
                    f"top_beats_random_{score}": bool(relative_difference > 0),
                }
            )
            top = strategies["top"].set_index("fraction_removed")
            random = strategies["random"].set_index("fraction_removed")
            for fraction in (0.05, 0.10, 0.20, 0.30):
                suffix = int(round(100 * fraction))
                row[f"top_{suffix}_{score}_drop"] = float(
                    top.loc[fraction, f"{score}_drop"]
                )
                row[f"top_{suffix}_relative_{score}_reduction"] = float(
                    top.loc[fraction, relative_column]
                )
                row[
                    f"top_minus_random_{suffix}_relative_{score}_reduction"
                ] = float(
                    top.loc[fraction, relative_column]
                    - random.loc[fraction, relative_column]
                )
        rows.append(row)

    derived = pd.DataFrame(rows)
    metric_identity = [column for column in identity if column in metrics.columns]
    replace_columns = [column for column in derived if column not in metric_identity]
    enriched = metrics.drop(columns=replace_columns, errors="ignore").merge(
        derived,
        on=metric_identity,
        how="left",
        validate="one_to_one",
    )
    return enriched, upgraded_curves


def attach_reference_seed_analysis_families(metrics: pd.DataFrame) -> pd.DataFrame:
    """Map the completed reference-seed targets to explicit analysis families."""
    frame = metrics.copy()
    frame["cohort_name"] = "selected_faithfulness"
    normalized_role = frame["target_role"].replace(
        {"predicted": "predicted_class", "true": "true_class"}
    )
    frame["target_role"] = normalized_role

    primary_methods = {
        "ResNet18": "gradcam",
        "DINOv2": "gradient_attention_rollout",
        "UNI": "gradient_attention_rollout",
    }
    architecture_rows = frame[frame["method"].eq(frame["model"].map(primary_methods))]
    prediction_rows = architecture_rows[
        architecture_rows["target_role"].eq("predicted_class")
    ]
    jointly_correct = (
        prediction_rows.groupby("cohort_id")["correct"].all().rename("all_models_correct")
    )
    frame = frame.merge(jointly_correct, on="cohort_id", how="left")
    frame["all_models_correct"] = frame["all_models_correct"].fillna(False)

    families = []

    def emit(selection: pd.Series, family: str) -> None:
        selected = frame[selection].copy()
        selected["analysis_family"] = family
        families.append(selected)

    predicted = frame["target_role"].eq("predicted_class")
    true_target = frame["target_role"].eq("true_class")
    emit(predicted, "actual_decision")
    emit(predicted & frame["all_models_correct"], "primary_jointly_correct_true_class")
    emit(predicted & ~frame["correct"].astype(bool), "error_predicted_class")
    emit(true_target & ~frame["correct"].astype(bool), "error_true_class")
    emit(
        (predicted & frame["correct"].astype(bool))
        | (true_target & ~frame["correct"].astype(bool)),
        "all_image_true_class_sensitivity",
    )
    output = pd.concat(families, ignore_index=True)
    output["perturbation"] = "normalized_zero"
    output["grid_label"] = "common_14"
    output["evaluation_grid_size"] = 14
    output["native_grid_size"] = output["model"].map(
        {"ResNet18": 5, "DINOv2": 16, "UNI": 14}
    )
    output["null_repetition"] = np.nan
    return output
