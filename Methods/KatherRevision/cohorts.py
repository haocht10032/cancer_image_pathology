"""Prediction-independent cohort selection and explicit explanation targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _balanced_random_take(
    frame: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    working = frame.copy()
    working["_random"] = rng.random(len(working))
    working = working.sort_values(["case_id", "_random"])
    working["_source_rank"] = working.groupby("case_id").cumcount()
    working = working.sort_values(["_source_rank", "_random", "case_id"])
    return working.head(count).drop(columns=["_random", "_source_rank"])


def build_random_cohort(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    images_per_class: int = 34,
    seed: int = 314159,
) -> pd.DataFrame:
    """Select a class/source-stratified cohort without model predictions."""
    test_rows = assignments.loc[
        assignments["split"].eq("test"),
        ["relative_path", "fold", "case_id", "label", "class_name"],
    ].drop_duplicates("relative_path")
    frame = manifest.merge(
        test_rows,
        on=["relative_path", "case_id", "label", "class_name"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(manifest):
        raise RuntimeError("OOF test assignments do not cover the full image manifest")

    rng = np.random.default_rng(seed)
    selected = []
    for label, class_frame in frame.groupby("label", sort=True):
        if len(class_frame) < images_per_class:
            raise ValueError(
                f"Class {label} has only {len(class_frame)} images; "
                f"cannot sample {images_per_class}"
            )
        selected.append(_balanced_random_take(class_frame, images_per_class, rng))
    cohort = pd.concat(selected, ignore_index=True)
    cohort.insert(
        0,
        "cohort_id",
        [
            hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            for value in cohort["relative_path"].astype(str)
        ],
    )
    cohort["path"] = cohort["image_path"]
    cohort["cohort_name"] = "random_prediction_independent"
    cohort["selection_seed"] = int(seed)
    cohort["selected_using_predictions"] = False
    cohort["selected_using_attributions"] = False
    return cohort.sort_values(
        ["label", "case_id", "relative_path"]
    ).reset_index(drop=True)


def freeze_random_cohort(
    cohort: pd.DataFrame,
    output_path: Path | str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write the prediction-independent cohort once, with a content hash."""
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.is_file() and not overwrite:
        existing = pd.read_csv(output_path)
        if existing["cohort_id"].astype(str).tolist() != cohort[
            "cohort_id"
        ].astype(str).tolist():
            raise RuntimeError(
                "The frozen random cohort differs from the proposed cohort. "
                "Use a new output directory or explicitly allow overwrite."
            )
        return existing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_path, index=False)
    metadata = {
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "image_count": int(len(cohort)),
        "class_count": int(cohort["label"].nunique()),
        "source_group_count": int(cohort["case_id"].nunique()),
        "selection_seed": int(cohort["selection_seed"].iloc[0]),
        "selection_used_predictions": False,
        "selection_used_attributions": False,
        "selection_variables": ["class", "source_group", "random_seed"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cohort


def build_budgeted_cohort(
    cohort: pd.DataFrame,
    images_per_class: int,
    seed: int,
    cohort_name: str,
    preserve_stratum: str | None = None,
) -> pd.DataFrame:
    """Create a deterministic class/source-balanced cohort for bounded compute.

    When a selection-stratum column is supplied, every available stratum receives
    at least one image per class before remaining slots are allocated
    proportionally. Selection within each class/stratum is round-robin by source.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for _, class_frame in cohort.groupby("label", sort=True):
        if len(class_frame) < images_per_class:
            raise ValueError(
                f"Class has {len(class_frame)} images; requested {images_per_class}"
            )
        if preserve_stratum is None or preserve_stratum not in class_frame:
            selected.append(_balanced_random_take(class_frame, images_per_class, rng))
            continue

        strata = list(class_frame[preserve_stratum].dropna().unique())
        if len(strata) > images_per_class:
            strata = strata[:images_per_class]
        allocation = {value: 1 for value in strata}
        remaining = images_per_class - len(strata)
        sizes = class_frame[preserve_stratum].value_counts()
        while remaining > 0:
            candidates = [
                value
                for value in strata
                if allocation[value] < int(sizes.get(value, 0))
            ]
            if not candidates:
                break
            value = max(
                candidates,
                key=lambda item: float(sizes[item]) / (allocation[item] + 1),
            )
            allocation[value] += 1
            remaining -= 1
        pieces = []
        for value, count in allocation.items():
            stratum_frame = class_frame[class_frame[preserve_stratum].eq(value)]
            pieces.append(_balanced_random_take(stratum_frame, count, rng))
        class_selected = pd.concat(pieces, ignore_index=True)
        if len(class_selected) < images_per_class:
            unused = class_frame[
                ~class_frame["cohort_id"].isin(class_selected["cohort_id"])
            ]
            class_selected = pd.concat(
                (
                    class_selected,
                    _balanced_random_take(
                        unused, images_per_class - len(class_selected), rng
                    ),
                ),
                ignore_index=True,
            )
        selected.append(class_selected)

    result = pd.concat(selected, ignore_index=True)
    result["cohort_name"] = cohort_name
    result["budget_selection_seed"] = int(seed)
    result["budget_images_per_class"] = int(images_per_class)
    return result.sort_values(
        ["label", "case_id", "relative_path"]
    ).reset_index(drop=True)


def attach_prediction_hierarchy(
    cohort: pd.DataFrame,
    predictions: pd.DataFrame,
    model_names: tuple[str, ...] = ("ResNet18", "DINOv2", "UNI"),
) -> pd.DataFrame:
    """Create one row per image, seed, and model with joint-correct indicators."""
    selected = predictions[predictions["model"].isin(model_names)].copy()
    identity = [
        "path",
        "relative_path",
        "case_id",
        "label",
        "class_name",
        "fold",
    ]
    cohort_identity = cohort.copy()
    if "path" not in cohort_identity:
        cohort_identity["path"] = cohort_identity["image_path"]
    cohort_columns = ["cohort_id", *identity]
    if "cohort_name" in cohort_identity:
        cohort_columns.append("cohort_name")
    else:
        cohort_identity["cohort_name"] = "selected_faithfulness"
        cohort_columns.append("cohort_name")
    merged = cohort_identity[cohort_columns].merge(
        selected,
        on=identity,
        how="inner",
        validate="one_to_many",
    )
    expected = len(cohort_identity) * selected["seed"].nunique() * len(model_names)
    if len(merged) != expected:
        raise RuntimeError(
            f"Expected {expected} image-seed-model prediction rows, found {len(merged)}"
        )

    joint = (
        merged.pivot_table(
            index=["cohort_id", "seed"],
            columns="model",
            values="correct",
            aggfunc="first",
        )
        .reindex(columns=model_names)
        .all(axis=1)
        .rename("all_models_correct")
        .reset_index()
    )
    merged = merged.merge(joint, on=["cohort_id", "seed"], validate="many_to_one")
    quantiles = merged.groupby(["model", "seed"])["confidence"].transform(
        lambda values: values.quantile([0.25, 0.75]).to_numpy()[0]
    )
    upper = merged.groupby(["model", "seed"])["confidence"].transform(
        lambda values: values.quantile(0.75)
    )
    merged["confidence_group"] = np.where(
        merged["confidence"] <= quantiles,
        "low",
        np.where(merged["confidence"] >= upper, "high", "middle"),
    )
    return merged.sort_values(["cohort_id", "seed", "model"]).reset_index(drop=True)


def explanation_targets(prediction_index: pd.DataFrame) -> pd.DataFrame:
    """Emit explicit, intentionally overlapping target-analysis families.

    The overlap is necessary: an incorrect prediction belongs to the actual-decision
    analysis and to the narrower error analysis, while every image also belongs to
    the all-image true-class sensitivity analysis. Downstream evaluation caches the
    unique image/target computation before attaching these analysis labels.
    """

    def target_rows(
        frame: pd.DataFrame,
        role: str,
        target_column: str,
        family: str,
    ) -> pd.DataFrame:
        output = frame.copy()
        output["target_role"] = role
        output["target_class"] = output[target_column].astype(int)
        output["analysis_family"] = family
        return output

    families = [
        target_rows(
            prediction_index[prediction_index["all_models_correct"]],
            "true_class",
            "label",
            "primary_jointly_correct_true_class",
        ),
        target_rows(
            prediction_index,
            "predicted_class",
            "prediction",
            "actual_decision",
        ),
        target_rows(
            prediction_index[~prediction_index["correct"]],
            "predicted_class",
            "prediction",
            "error_predicted_class",
        ),
        target_rows(
            prediction_index[~prediction_index["correct"]],
            "true_class",
            "label",
            "error_true_class",
        ),
        target_rows(
            prediction_index,
            "true_class",
            "label",
            "all_image_true_class_sensitivity",
        ),
    ]
    return pd.concat(families, ignore_index=True).sort_values(
        ["cohort_id", "seed", "model", "analysis_family"]
    ).reset_index(drop=True)
