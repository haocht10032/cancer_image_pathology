"""Leakage-checked grouped out-of-fold assignments."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "image_path",
    "relative_path",
    "class_name",
    "label",
    "case_id",
}


def _choose_validation_group(
    manifest: pd.DataFrame,
    test_group: str,
    groups: list[str],
    labels: list[int],
    seed: int,
) -> str:
    """Choose a validation group while retaining every class in training."""
    candidates: list[tuple[tuple[float, ...], str]] = []
    global_counts = manifest.groupby("label").size().reindex(labels, fill_value=0)
    test_position = groups.index(test_group)
    for validation_group in groups:
        if validation_group == test_group:
            continue
        train = manifest[
            ~manifest["case_id"].isin((test_group, validation_group))
        ]
        train_counts = train.groupby("label").size().reindex(labels, fill_value=0)
        if (train_counts == 0).any():
            continue

        validation = manifest[manifest["case_id"] == validation_group]
        validation_counts = (
            validation.groupby("label").size().reindex(labels, fill_value=0)
        )
        coverage = float((validation_counts > 0).sum())
        distribution = validation_counts / max(int(validation_counts.sum()), 1)
        target_distribution = global_counts / int(global_counts.sum())
        distribution_error = float(np.abs(distribution - target_distribution).sum())
        size_error = abs(len(validation) - len(manifest) / len(groups))
        cyclic_distance = (
            groups.index(validation_group) - test_position
        ) % len(groups)
        tie = int(
            hashlib.sha256(
                f"{seed}:{test_group}:{validation_group}".encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        candidates.append(
            (
                (
                    float(cyclic_distance),
                    -coverage,
                    distribution_error,
                    float(size_error),
                    float(tie),
                ),
                validation_group,
            )
        )

    if not candidates:
        raise ValueError(
            f"No validation group leaves all classes in training for test group {test_group}"
        )
    return min(candidates, key=lambda item: item[0])[1]


def build_grouped_oof_assignments(
    manifest: pd.DataFrame,
    seed: int = 41,
    group_column: str = "case_id",
) -> pd.DataFrame:
    """Create leave-one-source-group-out folds with a disjoint validation group."""
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if group_column != "case_id":
        manifest = manifest.rename(columns={group_column: "case_id"})

    source = manifest.copy().reset_index(drop=True)
    groups = sorted(source["case_id"].astype(str).unique())
    labels = sorted(int(value) for value in source["label"].unique())
    if len(groups) < 3:
        raise ValueError("Grouped OOF evaluation requires at least three source groups")

    frames: list[pd.DataFrame] = []
    for fold, test_group in enumerate(groups):
        validation_group = _choose_validation_group(
            source,
            test_group,
            groups,
            labels,
            seed,
        )
        fold_frame = source.copy()
        fold_frame["fold"] = fold
        fold_frame["split"] = "train"
        fold_frame.loc[
            fold_frame["case_id"].astype(str) == validation_group,
            "split",
        ] = "validation"
        fold_frame.loc[
            fold_frame["case_id"].astype(str) == test_group,
            "split",
        ] = "test"
        fold_frame["test_group"] = test_group
        fold_frame["validation_group"] = validation_group
        frames.append(fold_frame)

    assignments = pd.concat(frames, ignore_index=True)
    validate_oof_assignments(assignments, expected_image_count=len(source))
    return assignments


def validate_oof_assignments(
    assignments: pd.DataFrame,
    expected_image_count: int | None = None,
) -> None:
    """Raise if a source leaks or an image lacks exactly one OOF test prediction."""
    required = {"relative_path", "case_id", "label", "fold", "split"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Fold assignments are missing columns: {sorted(missing)}")

    for fold, frame in assignments.groupby("fold"):
        memberships = frame.groupby("case_id")["split"].nunique()
        if int(memberships.max()) != 1:
            raise RuntimeError(f"Source-group leakage detected in fold {fold}")
        split_groups = {
            split: set(rows["case_id"])
            for split, rows in frame.groupby("split")
        }
        if set(split_groups) != {"train", "validation", "test"}:
            raise RuntimeError(f"Fold {fold} does not contain all three splits")
        if (
            split_groups["train"] & split_groups["validation"]
            or split_groups["train"] & split_groups["test"]
            or split_groups["validation"] & split_groups["test"]
        ):
            raise RuntimeError(f"Overlapping source groups detected in fold {fold}")
        train_labels = set(frame.loc[frame["split"] == "train", "label"])
        all_labels = set(frame["label"])
        if train_labels != all_labels:
            raise RuntimeError(
                f"Fold {fold} training data omit labels {sorted(all_labels - train_labels)}"
            )

    test_rows = assignments[assignments["split"] == "test"]
    test_counts = test_rows.groupby("relative_path").size()
    if not test_counts.eq(1).all():
        raise RuntimeError("Each image must occur in exactly one held-out test fold")
    if expected_image_count is not None and len(test_counts) != expected_image_count:
        raise RuntimeError(
            f"Expected {expected_image_count} OOF images, found {len(test_counts)}"
        )
    if test_rows["label"].nunique() != assignments["label"].nunique():
        raise RuntimeError("Aggregate OOF test predictions do not cover every class")


def fold_class_group_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    """Return explicit image and source-group counts for every fold/split/class."""
    image_counts = (
        assignments.groupby(["fold", "split", "label", "class_name"])
        .size()
        .rename("image_count")
        .reset_index()
    )
    group_counts = (
        assignments.groupby(["fold", "split", "label", "class_name"])["case_id"]
        .nunique()
        .rename("source_group_count")
        .reset_index()
    )
    return image_counts.merge(
        group_counts,
        on=["fold", "split", "label", "class_name"],
        validate="one_to_one",
    )
