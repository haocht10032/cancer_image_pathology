"""CPU-only integrity and arithmetic audit of downloaded v5 outputs."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from Methods.Feedback2.probability_deletion import (
    KEY, OUTPUT_SUBDIR, METHODS, inputs, select_metrics, digest, auc,
    archive_curve_path,
)


def audit(root):
    root = Path(root)
    base = root / OUTPUT_SUBDIR / "full"
    output = root / "artifacts/feedback2_revision/reporting"
    output.mkdir(parents=True, exist_ok=True)
    counts, tie_rows = [], []
    hashes = {}

    def sha(path):
        path = str(path)
        if path not in hashes:
            hashes[path] = digest(path)
        return hashes[path]

    for dataset in ("kather", "crc"):
        expected = select_metrics(inputs(root, dataset)[0])
        combined = pd.read_csv(base / dataset / "probability_image_seed_metrics.csv")
        identity = KEY + ["analysis_family"]
        assert not combined.duplicated(identity).any()
        assert set(combined[identity].itertuples(index=False, name=None)) == set(
            expected[identity].itertuples(index=False, name=None))
        assert combined.probability_protocol.eq("deterministic_ties_v5_sensitivity").all()
        assert combined.tie_policy.eq("score_then_patch_index_ascending").all()
        collected = []
        markers = sorted((base / dataset).glob("*/seed_*_fold_*.json"))
        assert len(markers) == (90 if dataset == "kather" else 9)
        for marker in markers:
            meta = json.loads(marker.read_text())
            assert meta["version"] == 5 and meta["proof"] is False
            assert meta["tie_policy"] == "score_then_patch_index_ascending"
            assert meta["implementation_sha256"] == sha(root / "Methods/Feedback2/probability_deletion.py")
            metric_path, map_path = inputs(root, dataset)
            for field, path in (("metrics_sha256", metric_path), ("maps_sha256", map_path),
                                ("curves_sha256", archive_curve_path(root, dataset))):
                assert meta[field] == sha(path), (field, marker)
            model = marker.parent.name
            seed, fold = map(int, marker.stem.replace("seed_", "").split("_fold_"))
            folder = {"ResNet18": "resnet18", "DINOv2": "dinov2", "UNI": "uni"}[model]
            if dataset == "crc":
                checkpoint = root / "artifacts/crc_val_external/common_seven/checkpoints" / folder / f"seed_{seed}.pt"
            else:
                run = "grouped_oof_faithfulness" if model == "UNI" else "dinov2_three_model"
                checkpoint = root / "artifacts" / run / "checkpoints" / folder / f"seed_{seed}/fold_{fold}.pt"
            assert meta["checkpoint_sha256"] == sha(checkpoint), checkpoint
            stem = marker.with_suffix("")
            for suffix, checksum in meta["output_hashes"].items():
                assert sha(Path(str(stem) + suffix)) == checksum, (marker, suffix)
            metrics = pd.read_csv(Path(str(stem) + "_metrics.csv"))
            curves = pd.read_csv(Path(str(stem) + "_curves.csv"))
            collected.append(metrics)
            assert len(metrics.drop_duplicates(KEY)) == meta["rows"]
            assert len(curves) == 49 * meta["rows"]
            assert not curves.duplicated(KEY + ["strategy", "fraction_removed", "repetition"]).any()
            assert curves.loc[~curves.tie_boundary, "archived_replay_verified"].all()
            assert curves.loc[curves.strategy.eq("random"), "archived_replay_verified"].all()
            logits = curves[[f"logit_{i}" for i in range(7 if dataset == "crc" else 8)]].to_numpy()
            assert np.isfinite(logits).all()
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            target = curves.target_class.to_numpy(int)
            assert np.allclose(probabilities[np.arange(len(curves)), target], curves.target_probability, atol=1e-6)
            assert np.allclose(curves.baseline_probability-curves.target_probability, curves.probability_drop, atol=1e-8)
            for row in curves.itertuples():
                indices = json.loads(row.deleted_patch_indices)
                assert len(indices) == row.patches_removed == round(196 * row.fraction_removed)
                assert len(set(indices)) == len(indices)
                assert all(0 <= i < 196 for i in indices)
            lookup = metrics.drop_duplicates(KEY).set_index(KEY)
            for key, frame in curves.groupby(KEY):
                reference = lookup.loc[key]
                mean = frame.groupby(["strategy", "fraction_removed"], as_index=False).probability_drop.mean()
                areas = {s: auc(f, "probability_drop") for s, f in mean.groupby("strategy")}
                for s, value in areas.items():
                    assert np.isclose(value, reference[f"{s}_probability_reduction_auc"], atol=1e-10)
                difference = areas["top"] - areas["random"]
                assert np.isclose(difference, reference.top_minus_random_probability_reduction_auc, atol=1e-10)
                assert reference.top_beats_random_probability == float(difference > 0)
            steps = curves.drop_duplicates(KEY + ["strategy", "fraction_removed"])
            for strategy, group in steps.groupby("strategy"):
                tie_rows.append(dict(dataset=dataset, model=model, seed=seed, fold=fold,
                    strategy=strategy, steps=len(group), tied_steps=int(group.tie_boundary.sum()),
                    archive_mismatches=int((~group.archived_replay_verified).sum()),
                    affected_targets=group.loc[~group.archived_replay_verified, KEY].drop_duplicates().shape[0]))
        pd.testing.assert_frame_equal(
            combined.sort_values(identity).reset_index(drop=True),
            pd.concat(collected).sort_values(identity).reset_index(drop=True), check_dtype=False)
        counts.append(dict(dataset=dataset, partitions=len(markers), metric_rows=len(combined),
            unique_image_seed_model_targets=len(combined.drop_duplicates(KEY)),
            images=combined.cohort_id.nunique(), seeds=sorted(map(int, combined.seed.unique())),
            models=sorted(combined.model.unique())))
        print(dataset, counts[-1], flush=True)
    pd.DataFrame(tie_rows).to_csv(output / "probability_tie_audit.csv", index=False)
    report = dict(status="passed", counts=counts, checks="input/checkpoint/output hashes; manifest; probabilities; AUC; tie-free/random validation")
    (output / "probability_download_audit.json").write_text(json.dumps(report, indent=2))
    print("All download checks passed.", flush=True)


if __name__ == "__main__":
    audit(Path(__file__).resolve().parents[2])
