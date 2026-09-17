"""Cross-seed prediction and attribution stability from saved revision maps."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


STABILITY_PAIR_COLUMNS = [
    "cohort_name",
    "cohort_id",
    "case_id",
    "class_name",
    "model",
    "method",
    "grid_label",
    "target_mode",
    "seed_a",
    "seed_b",
    "prediction_a",
    "prediction_b",
    "same_prediction",
    "attribution_spearman",
]
STABILITY_IMAGE_COLUMNS = [
    "cohort_name",
    "cohort_id",
    "case_id",
    "class_name",
    "model",
    "method",
    "grid_label",
    "target_mode",
    "seed_pairs",
    "mean_attribution_spearman",
    "median_attribution_spearman",
]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.unique(first).size < 2 or np.unique(second).size < 2:
        return np.nan
    return float(spearmanr(first, second).statistic)


def prediction_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed variability without treating seeds as biological samples."""
    rows = []
    grouping = ["cohort_name", "cohort_id", "case_id", "class_name", "model"]
    for keys, frame in predictions.groupby(grouping, dropna=False):
        labels = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        values = frame.drop_duplicates("seed").sort_values("seed")
        seed_predictions = values["prediction"].astype(int).to_numpy()
        pairs = list(combinations(range(len(seed_predictions)), 2))
        pairwise_agreement = (
            float(
                np.mean(
                    [
                        seed_predictions[first] == seed_predictions[second]
                        for first, second in pairs
                    ]
                )
            )
            if pairs
            else np.nan
        )
        counts = pd.Series(seed_predictions).value_counts()
        rows.append(
            {
                **labels,
                "seed_count": int(len(seed_predictions)),
                "pairwise_prediction_agreement": pairwise_agreement,
                "majority_prediction_agreement": float(counts.iloc[0] / len(values)),
                "unique_predictions": int(counts.size),
            }
        )
    return pd.DataFrame(rows)


def attribution_stability(
    maps: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare common-target maps and conditional predicted-target maps by seed.

    Predicted-class maps are compared only when both seeds predict the same class.
    The all-image true-class maps provide the common-target comparison when seed
    predictions differ.
    """
    maps = maps[maps["null_repetition"].isna()].copy()
    target_sets = {
        "common_true_class": maps[maps["target_role"].eq("true_class")],
        "predicted_class_same_prediction": maps[
            maps["target_role"].eq("predicted_class")
        ],
    }
    prediction_lookup = predictions.set_index(
        ["cohort_name", "cohort_id", "model", "seed"]
    )["prediction"]
    rows = []
    grouping = [
        "cohort_name",
        "cohort_id",
        "case_id",
        "class_name",
        "model",
        "method",
        "grid_label",
    ]
    for target_mode, target_maps in target_sets.items():
        for keys, frame in target_maps.groupby(grouping, dropna=False):
            labels = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
            seed_maps = {
                int(seed): values.sort_values("patch_index")["attribution"].to_numpy()
                for seed, values in frame.groupby("seed")
            }
            for first_seed, second_seed in combinations(sorted(seed_maps), 2):
                prediction_key = (
                    labels["cohort_name"],
                    labels["cohort_id"],
                    labels["model"],
                )
                first_prediction = int(prediction_lookup.loc[(*prediction_key, first_seed)])
                second_prediction = int(
                    prediction_lookup.loc[(*prediction_key, second_seed)]
                )
                same_prediction = first_prediction == second_prediction
                if target_mode == "predicted_class_same_prediction" and not same_prediction:
                    continue
                rows.append(
                    {
                        **labels,
                        "target_mode": target_mode,
                        "seed_a": first_seed,
                        "seed_b": second_seed,
                        "prediction_a": first_prediction,
                        "prediction_b": second_prediction,
                        "same_prediction": same_prediction,
                        "attribution_spearman": _correlation(
                            seed_maps[first_seed], seed_maps[second_seed]
                        ),
                    }
                )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return (
            pd.DataFrame(columns=STABILITY_PAIR_COLUMNS),
            pd.DataFrame(columns=STABILITY_IMAGE_COLUMNS),
        )
    per_image = (
        pairs.groupby([*grouping, "target_mode"], dropna=False)
        .agg(
            seed_pairs=("attribution_spearman", "size"),
            mean_attribution_spearman=("attribution_spearman", "mean"),
            median_attribution_spearman=("attribution_spearman", "median"),
        )
        .reset_index()
    )
    return pairs, per_image


def save_stability_tables(
    maps: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = prediction_stability(predictions)
    pairs, per_image = attribution_stability(maps, predictions)
    source = (
        per_image.groupby(
            ["cohort_name", "case_id", "model", "method", "grid_label", "target_mode"],
            dropna=False,
        )["mean_attribution_spearman"]
        .mean()
        .reset_index()
        if not per_image.empty
        else pd.DataFrame(
            columns=[
                "cohort_name",
                "case_id",
                "model",
                "method",
                "grid_label",
                "target_mode",
                "mean_attribution_spearman",
            ]
        )
    )
    frames = {
        "prediction_stability_per_image": prediction,
        "attribution_stability_seed_pairs": pairs,
        "attribution_stability_per_image": per_image,
        "attribution_stability_source_groups": source,
    }
    paths = {}
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
