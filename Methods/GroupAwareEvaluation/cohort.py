"""Pre-attribution, class-stratified faithfulness cohort construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _source_balanced_take(
    frame: pd.DataFrame,
    count: int,
    score_column: str,
    ascending: bool,
    seed: int,
) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    working = frame.copy()
    rng = np.random.default_rng(seed)
    working["_tie"] = rng.random(len(working))
    working = working.sort_values(
        [score_column, "_tie"],
        ascending=[ascending, True],
    )
    working["_source_rank"] = working.groupby("case_id").cumcount()
    working = working.sort_values(
        ["_source_rank", score_column, "_tie"],
        ascending=[True, ascending, True],
    )
    return working.head(count).drop(columns=["_tie", "_source_rank"])


def _reference_predictions(
    predictions: pd.DataFrame,
    reference_seed: int,
) -> pd.DataFrame:
    selected = predictions[predictions["seed"] == reference_seed].copy()
    required_models = {"UNI", "CNN"}
    available = set(selected["model"])
    if not required_models.issubset(available):
        raise ValueError(
            f"Reference seed {reference_seed} is missing models "
            f"{sorted(required_models - available)}"
        )
    identity = [
        "path",
        "relative_path",
        "case_id",
        "label",
        "class_name",
        "fold",
    ]
    frames = []
    for model_name in ("UNI", "CNN"):
        model_frame = selected[selected["model"] == model_name].copy()
        rename = {
            "prediction": f"{model_name.lower()}_prediction",
            "predicted_class_name": f"{model_name.lower()}_predicted_class_name",
            "confidence": f"{model_name.lower()}_confidence",
            "correct": f"{model_name.lower()}_correct",
        }
        frames.append(
            model_frame[
                identity + list(rename)
            ].rename(columns=rename)
        )
    merged = frames[0].merge(
        frames[1],
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    expected = selected.groupby("model").size().min()
    if len(merged) != expected:
        raise RuntimeError("UNI and CNN OOF predictions do not cover identical images")
    merged["joint_confidence"] = merged[
        ["uni_confidence", "cnn_confidence"]
    ].mean(axis=1)
    merged["both_correct"] = merged["uni_correct"] & merged["cnn_correct"]
    merged["either_incorrect"] = ~merged["both_correct"]
    return merged


def build_faithfulness_cohort(
    predictions: pd.DataFrame,
    reference_seed: int,
    correct_per_class: int = 24,
    incorrect_per_class: int = 10,
    low_confidence_fraction: float = 1 / 3,
    seed: int = 2027,
) -> pd.DataFrame:
    """Select a paired, source-balanced cohort before attribution is examined."""
    if correct_per_class < 1:
        raise ValueError("correct_per_class must be positive")
    if not 0 < low_confidence_fraction <= 1:
        raise ValueError("low_confidence_fraction must be in (0, 1]")
    frame = _reference_predictions(predictions, reference_seed)
    rows: list[pd.DataFrame] = []
    labels = sorted(int(value) for value in frame["label"].unique())

    for label in labels:
        class_frame = frame[frame["label"] == label]
        correct = class_frame[class_frame["both_correct"]]
        if correct.empty:
            raise RuntimeError(
                f"No jointly correct UNI/CNN OOF image is available for class {label}"
            )
        low_count = max(1, int(round(correct_per_class * low_confidence_fraction)))
        low = _source_balanced_take(
            correct,
            min(low_count, len(correct)),
            "joint_confidence",
            ascending=True,
            seed=seed + label * 101,
        )
        low = low.assign(cohort_stratum="correct_low_confidence")
        remaining = correct[~correct["path"].isin(low["path"])]
        high = _source_balanced_take(
            remaining,
            min(correct_per_class - len(low), len(remaining)),
            "joint_confidence",
            ascending=False,
            seed=seed + label * 101 + 1,
        )
        high = high.assign(cohort_stratum="correct_high_confidence")
        selected_correct = pd.concat((low, high), ignore_index=True)
        if selected_correct.empty:
            raise RuntimeError(f"Could not select a correct image for class {label}")
        rows.append(selected_correct)

        incorrect = class_frame[class_frame["either_incorrect"]].copy()
        if not incorrect.empty and incorrect_per_class > 0:
            incorrect["error_priority"] = (
                (~incorrect["uni_correct"]).astype(int)
                + (~incorrect["cnn_correct"]).astype(int)
            )
            incorrect["selection_score"] = (
                incorrect["error_priority"] * 10 + incorrect["joint_confidence"]
            )
            selected_incorrect = _source_balanced_take(
                incorrect,
                min(incorrect_per_class, len(incorrect)),
                "selection_score",
                ascending=False,
                seed=seed + label * 101 + 2,
            ).assign(cohort_stratum="incorrect_any_model")
            rows.append(selected_incorrect.drop(columns=["error_priority", "selection_score"]))

    cohort = pd.concat(rows, ignore_index=True)
    cohort = cohort.drop_duplicates("path").reset_index(drop=True)
    cohort.insert(
        0,
        "cohort_id",
        [
            hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
            for path in cohort["relative_path"].astype(str)
        ],
    )
    cohort["reference_seed"] = int(reference_seed)
    cohort["selection_seed"] = int(seed)
    cohort["selected_without_attribution"] = True
    correct_coverage = cohort.loc[
        cohort["both_correct"],
        "label",
    ].nunique()
    if correct_coverage != len(labels):
        raise RuntimeError(
            f"Correct-image cohort covers {correct_coverage} of {len(labels)} classes"
        )
    return cohort.sort_values(
        ["label", "cohort_stratum", "case_id", "relative_path"]
    ).reset_index(drop=True)


def freeze_cohort_manifest(
    cohort: pd.DataFrame,
    output_path: Path | str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Write the cohort once and protect it from post-attribution reselection."""
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.is_file() and not overwrite:
        existing = pd.read_csv(output_path)
        if existing["cohort_id"].astype(str).tolist() != cohort[
            "cohort_id"
        ].astype(str).tolist():
            raise RuntimeError(
                "A different frozen cohort already exists. Use a new artifact "
                "directory or explicitly set overwrite=True before attribution."
            )
        return existing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_path, index=False)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "sha256": digest,
        "image_count": len(cohort),
        "class_count": int(cohort["label"].nunique()),
        "source_group_count": int(cohort["case_id"].nunique()),
        "selection_used_attribution_results": False,
        "note": (
            "This manifest was selected from OOF labels, predictions, confidence, "
            "class, and source group only."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cohort


def attach_model_predictions(
    cohort: pd.DataFrame,
    predictions: pd.DataFrame,
    model_name: str,
    reference_seed: int,
    column_prefix: str,
) -> pd.DataFrame:
    """Attach a new model's OOF outputs without changing frozen image selection."""
    selected = predictions[
        (predictions["model"] == model_name)
        & (predictions["seed"] == reference_seed)
    ].copy()
    columns = [
        "path",
        "prediction",
        "predicted_class_name",
        "confidence",
        "correct",
    ]
    missing = set(columns).difference(selected.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    selected = selected[columns].rename(
        columns={
            "prediction": f"{column_prefix}_prediction",
            "predicted_class_name": f"{column_prefix}_predicted_class_name",
            "confidence": f"{column_prefix}_confidence",
            "correct": f"{column_prefix}_correct",
        }
    )
    result = cohort.merge(
        selected,
        on="path",
        how="left",
        validate="one_to_one",
    )
    prediction_column = f"{column_prefix}_prediction"
    if result[prediction_column].isna().any():
        missing_count = int(result[prediction_column].isna().sum())
        raise ValueError(
            f"{missing_count} frozen cohort images lack {model_name} OOF predictions"
        )
    result[prediction_column] = result[prediction_column].astype(int)
    result[f"{column_prefix}_correct"] = result[
        f"{column_prefix}_correct"
    ].astype(bool)
    return result
