"""Probability sensitivity with explicit ties, without new maps or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Methods.KatherRevision.evaluation import DELETION_FRACTIONS, _stable_seed
from Methods.KatherRevision.perturbations import replacement_reference, replace_patch_indices

METHODS = {"ResNet18": "gradcam", "DINOv2": "gradient_attention_rollout", "UNI": "gradient_attention_rollout"}
KEY = ["cohort_name", "cohort_id", "model", "seed", "fold", "target_class"]
VERSION = 5
OUTPUT_SUBDIR = "artifacts/feedback2_revision/probability_deletion/deterministic_ties_v5"


def compatible_fingerprint(actual, expected):
    return actual == expected


def validate_curve_steps(curves, archived, scores, key):
    """Require replication wherever a tied boundary cannot change the patch set."""
    columns = ["strategy", "fraction_removed"]
    old = archived[columns + ["target_logit"]].drop_duplicates()
    if old.duplicated(columns).any():
        raise RuntimeError(f"Conflicting archived deletion curves: {key}")
    steps = curves.groupby(columns, as_index=False).target_logit.mean().merge(
        old.rename(columns={"target_logit": "archived_target_logit"}),
        on=columns, how="left", validate="one_to_one")
    if steps.archived_target_logit.isna().any():
        raise RuntimeError(f"Missing archived deletion steps: {key}")
    values = np.sort(np.asarray(scores, dtype=np.float32))
    ambiguous = []
    for row in steps.itertuples():
        count = int(round(row.fraction_removed * len(values)))
        ordered = values[::-1] if row.strategy == "top" else values
        ambiguous.append(row.strategy != "random" and 0 < count < len(values)
                         and ordered[count-1] == ordered[count])
    steps["tie_boundary"] = ambiguous
    steps["archived_replay_verified"] = np.isclose(
        steps.target_logit, steps.archived_target_logit, atol=2e-3, rtol=2e-3)
    failed = steps[~steps.tie_boundary & ~steps.archived_replay_verified]
    if not failed.empty:
        raise RuntimeError(f"Tie-free/random deletion mismatch: {key}; "
                           f"{failed.to_dict('records')}. Do not relax tolerances.")
    steps["replay_logit_difference"] = steps.target_logit - steps.archived_target_logit
    return steps.drop(columns="target_logit")


def patch_orders(scores, random_seed):
    # Explicit secondary key makes equal-score interventions reproducible.
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (196,) or not np.isfinite(scores).all():
        raise ValueError("Expected 196 finite archived attribution scores")
    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(scores))
    return {"top": [np.lexsort((indices, -scores))],
            "bottom": [np.lexsort((indices, scores))],
            "random": [rng.permutation(196) for _ in range(5)]}


def configure_replay_backend():
    """Match the original notebooks' set_seed and default CUDA precision policy."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return dict(matmul_tf32=False, cudnn_tf32=True, cudnn_deterministic=True,
                cudnn_benchmark=False, torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version())


def validate_baseline(baseline, row, key):
    prediction = int(baseline.argmax())
    actual = float(baseline[int(row.target_class)])
    expected = float(row.unperturbed_target_logit)
    if prediction != int(row.prediction) or not np.isclose(
            actual, expected, atol=2e-3, rtol=2e-4):
        raise RuntimeError(
            f"Checkpoint/preprocessing/precision mismatch: {key}; "
            f"prediction replay={prediction}, archived={int(row.prediction)}; "
            f"target logit replay={actual:.9g}, archived={expected:.9g}, "
            f"absolute difference={abs(actual-expected):.9g}; "
            f"cudnn_tf32={torch.backends.cudnn.allow_tf32}, "
            f"matmul_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"deterministic={torch.backends.cudnn.deterministic}. "
            "Do not relax the tolerance; check the checkpoint, image, transform, "
            "and original CUDA/PyTorch environment.")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inputs(root, dataset):
    root = Path(root)
    if dataset == "kather":
        base = root / "artifacts/kather5k_jpi_revision"
        return (base / "statistics/image_seed_metrics.csv",
                base / "evaluation/consolidated/attribution_maps.csv")
    if dataset == "crc":
        base = root / "artifacts/crc_val_external/common_seven/faithfulness"
        return base / "external_faithfulness_metrics.csv", base / "external_attribution_maps.csv"
    raise ValueError(dataset)


def archive_curve_path(root, dataset):
    if dataset == "kather":
        return Path(root) / "artifacts/kather5k_jpi_revision/evaluation/consolidated/deletion_curves.csv"
    if dataset == "crc":
        return Path(root) / "artifacts/crc_val_external/common_seven/faithfulness/external_deletion_curves.csv"
    raise ValueError(dataset)


def select_metrics(path, model_name=None):
    frame = pd.read_csv(path)
    mask = (frame.perturbation.eq("normalized_zero") & frame.grid_label.eq("common_14")
            & frame.seed.isin([11, 89, 181]) & frame.model.isin(METHODS)
            & frame.method.eq(frame.model.map(METHODS))
            & frame.target_class.eq(frame.true_class)
            & frame.analysis_family.isin(["primary_jointly_correct_true_class", "all_image_true_class_sensitivity"]))
    if model_name:
        mask &= frame.model.eq(model_name)
    result = frame.loc[mask].copy()
    if result.empty:
        raise ValueError(f"No eligible archived metrics: {path}, {model_name}")
    return result


def make_adapter(root, dataset, name, device, uni_assets=None):
    from Methods.CNNBenchmark import build_resnet18
    from Methods.DINOv2Attribution import build_dinov2_classifier, resolve_dinov2_transform
    from Methods.UNIAttribution import build_uni_classifier, resolve_uni_transform
    from Methods.KatherRevision.evaluation import CNNAdapter, TransformerAdapter
    from Methods.CRCExternalValidation.faithfulness import ExternalCNNAdapter, ExternalTransformerAdapter
    root = Path(root)
    external = dataset == "crc"
    classes = 7 if external else 8
    folder = {"ResNet18": "resnet18", "DINOv2": "dinov2", "UNI": "uni"}[name]
    if external:
        checkpoint = root / "artifacts/crc_val_external/common_seven/checkpoints" / folder
    else:
        run = "grouped_oof_faithfulness" if name == "UNI" else "dinov2_three_model"
        checkpoint = root / "artifacts" / run / "checkpoints" / folder
    if name == "ResNet18":
        cls = ExternalCNNAdapter if external else CNNAdapter
        return cls(name, lambda: build_resnet18(num_classes=classes, pretrained=False,
                   freeze_backbone=False), checkpoint, device, image_size=150, native_grid_size=5)
    if name == "DINOv2":
        model = build_dinov2_classifier(classes, device)
        transform, _ = resolve_dinov2_transform(model.encoder)
        grid = 16
    else:
        model = build_uni_classifier(root, classes, device, assets_dir=uni_assets)
        transform, _ = resolve_uni_transform(model.encoder)
        grid = 14
    cls = ExternalTransformerAdapter if external else TransformerAdapter
    return cls(name, model, checkpoint, device, transform, native_grid_size=grid)


def image_path(root, dataset, row):
    roots = [Path(root) / "CRC-VAL-HE-7K"] if dataset == "crc" else [
        Path(root) / "Colorectal Histology MNIST/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles_5000"]
    candidates = [r / row.relative_path for r in roots] + [Path(row.path)]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found: {candidates}")


@torch.inference_mode()
def probability_curves(model, image, scores, target, random_seed, batch_size=8):
    """Softmax each repeat before averaging; preserve the original random orders."""
    orders = patch_orders(scores, random_seed)
    baseline = model(image).float().cpu()[0]
    base_probability = float(baseline.softmax(0)[target])
    reference = replacement_reference(image, "normalized_zero", 14)
    rows = []
    for strategy, permutations in orders.items():
        for fraction in DELETION_FRACTIONS:
            count = int(round(fraction * 196))
            variants = [replace_patch_indices(image, order[:count], 14, reference)
                        for order in permutations]
            logits = torch.cat([model(torch.cat(variants[i:i+batch_size])).float().cpu()
                                for i in range(0, len(variants), batch_size)])
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite deletion logits")
            for repetition, values in enumerate(logits):
                probability = float(values.softmax(0)[target])
                result = dict(strategy=strategy, fraction_removed=fraction,
                              repetition=repetition, patches_removed=count,
                              baseline_probability=base_probability,
                              target_probability=probability,
                              probability_drop=base_probability-probability,
                              target_logit=float(values[target]),
                              baseline_target_logit=float(baseline[target]))
                result.update({f"logit_{i}": float(v) for i, v in enumerate(values)})
                rows.append(result)
    return pd.DataFrame(rows), baseline


def auc(frame, column):
    frame = frame.sort_values("fraction_removed")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(frame[column], frame.fraction_removed))


def run_dataset(root, dataset, name, device, *, proof=True, batch_size=8, uni_assets=None):
    backend = configure_replay_backend()
    root = Path(root)
    metric_path, map_path = inputs(root, dataset)
    targets = select_metrics(metric_path, name)
    unique = targets.drop_duplicates(KEY).sort_values(KEY)
    if proof:
        regressions = unique.cohort_id.isin(["dc1d4e475494529d", "167d6917f8661d4c", "b21bcf660439087b"])
        unique = pd.concat([unique.head(1), unique[regressions]]).drop_duplicates(KEY)
    output = root / OUTPUT_SUBDIR / ("proof" if proof else "full") / dataset / name
    output.mkdir(parents=True, exist_ok=True)
    # Chunk the large map archive rather than loading all pipelines into memory.
    keep = []
    for chunk in pd.read_csv(map_path, chunksize=200000):
        selected = chunk[chunk.model.eq(name) & chunk.method.eq(METHODS[name])
                         & chunk.grid_label.eq("common_14")]
        selected = selected.merge(unique[KEY], on=KEY, how="inner", validate="many_to_one")
        if not selected.empty:
            keep.append(selected)
    if not keep:
        raise RuntimeError("No saved maps match the frozen target manifest")
    maps = pd.concat(keep, ignore_index=True)
    map_groups = dict(tuple(maps.groupby(KEY, sort=False)))
    curve_path = archive_curve_path(root, dataset)
    archived_parts = []
    for chunk in pd.read_csv(curve_path, chunksize=200000):
        selected = chunk[chunk.model.eq(name) & chunk.method.eq(METHODS[name])
                         & chunk.grid_label.eq("common_14") & chunk.perturbation.eq("normalized_zero")
                         & (~chunk.strategy.eq("random") | chunk.repeat_count.eq(5))]
        selected = selected.merge(unique[KEY], on=KEY, how="inner", validate="many_to_one")
        if not selected.empty:
            archived_parts.append(selected)
    if not archived_parts:
        raise RuntimeError("No archived curves for validation")
    archive_groups = dict(tuple(pd.concat(archived_parts).groupby(KEY, sort=False)))
    unique.to_csv(output / "frozen_targets.csv", index=False)
    fingerprints = {"metrics_sha256": digest(metric_path), "maps_sha256": digest(map_path),
                    "version": VERSION, "dataset": dataset, "model": name,
                    "seed_policy": "11,89,181", "fractions": list(DELETION_FRACTIONS),
                    "random_repeats": 5, "proof": proof, "batch_size": batch_size,
                    "implementation_sha256": digest(__file__), "backend": backend,
                    "numpy_version": np.__version__, "ranking_dtype": "float32",
                    "curves_sha256": digest(curve_path),
                    "tie_policy": "score_then_patch_index_ascending",
                    "validation": "baseline_and_all_random_and_tie_free_steps"}
    adapter = make_adapter(root, dataset, name, device, uni_assets)
    fingerprints["transform"] = repr(getattr(adapter, "image_transform", "RGB; bicubic 150x150; ImageNet normalization"))
    try:
        for (seed, fold), partition in unique.groupby(["seed", "fold"], sort=True):
            stem = output / f"seed_{seed}_fold_{fold}"
            checkpoint = adapter.checkpoint_dir / (f"seed_{seed}.pt" if dataset == "crc" else f"seed_{seed}/fold_{fold}.pt")
            fingerprint = dict(fingerprints, checkpoint_sha256=digest(checkpoint), rows=len(partition))
            marker = stem.with_suffix(".json")
            if marker.is_file():
                previous = json.loads(marker.read_text())
                expected = dict(fingerprint)
                actual = {k: v for k, v in previous.items() if k != "output_hashes"}
                if not compatible_fingerprint(actual, expected):
                    raise RuntimeError(f"Resume configuration changed: {marker}; use a new output directory")
                for suffix, checksum in previous["output_hashes"].items():
                    if digest(Path(str(stem)+suffix)) != checksum:
                        raise RuntimeError(f"Incomplete or modified partition: {stem}")
                print("Cached", stem.name, "validated implementation version", actual["version"])
                continue
            adapter.load_checkpoint(int(fold), int(seed))
            metrics_out, curves_out = [], []
            for _, row in partition.iterrows():
                key = tuple(row[k] for k in KEY)
                group = map_groups[key]
                # Duplicate analysis-family rows must encode exactly the same map.
                if group.groupby("patch_index").attribution.nunique().max() != 1:
                    raise RuntimeError(f"Conflicting archived maps: {key}")
                points = group.drop_duplicates("patch_index").sort_values("patch_index")
                if points.patch_index.tolist() != list(range(196)):
                    raise RuntimeError(f"Missing patches: {key}")
                image = adapter.load_image(str(image_path(root, dataset, row)))
                # Fail before the expensive deletion replay, not after 49 forwards.
                with torch.inference_mode():
                    baseline = adapter.model(image).float().cpu()[0]
                validate_baseline(baseline, row, key)
                curves, baseline = probability_curves(adapter.model, image, points.attribution,
                    int(row.target_class), _stable_seed(row.cohort_id, int(seed), int(row.target_class),
                    "normalized_zero", 14), batch_size)
                validate_baseline(baseline, row, key)
                mean_curves = curves.groupby(["strategy", "fraction_removed"], as_index=False).mean(numeric_only=True)
                probabilities = {s: auc(f, "probability_drop") for s, f in mean_curves.groupby("strategy")}
                old_logit_auc = {s: auc(f, "target_logit") for s, f in mean_curves.groupby("strategy")}
                archived_aucs = {s: float(row[f"{s}_target_logit_auc"]) for s in old_logit_auc}
                checks = validate_curve_steps(curves, archive_groups[key], points.attribution, key)
                curves = curves.merge(checks, on=["strategy", "fraction_removed"],
                                      how="left", validate="many_to_one")
                verified = checks.groupby("strategy").archived_replay_verified.all().to_dict()
                curves["diagnostic_only"] = curves.strategy.eq("bottom")
                curves["archived_target_logit_auc"] = curves.strategy.map(archived_aucs)
                curves["replay_target_logit_auc"] = curves.strategy.map(old_logit_auc)
                # Preserve the actual intervention sets for future audits, including ties.
                orders = patch_orders(points.attribution, _stable_seed(
                    row.cohort_id, int(seed), int(row.target_class), "normalized_zero", 14))
                curves["deleted_patch_indices"] = [json.dumps(
                    orders[r.strategy][int(r.repetition)][:int(r.patches_removed)].tolist())
                    for r in curves.itertuples()]
                metadata = {k: row[k] for k in KEY}
                for k, v in metadata.items():
                    curves[k] = v
                curves_out.append(curves)
                families = targets.merge(pd.DataFrame([metadata]), on=KEY, how="inner")
                for _, original in families.drop_duplicates([*KEY, "analysis_family"]).iterrows():
                    result = original.to_dict()
                    result.update({f"{s}_probability_reduction_auc": v for s, v in probabilities.items()})
                    result["tie_policy"] = "score_then_patch_index_ascending"
                    result["probability_protocol"] = "deterministic_ties_v5_sensitivity"
                    for s in verified:
                        result[f"{s}_archived_replay_verified"] = verified[s]
                        result[f"{s}_replay_logit_auc_difference"] = old_logit_auc[s] - archived_aucs[s]
                        result[f"{s}_deterministic_target_logit_auc"] = old_logit_auc[s]
                        result[f"{s}_tie_boundary_steps"] = int(checks.loc[checks.strategy.eq(s), "tie_boundary"].sum())
                    result["top_minus_random_deterministic_target_logit_auc"] = old_logit_auc["top"] - old_logit_auc["random"]
                    difference = probabilities["top"] - probabilities["random"]
                    result.update(top_minus_random_probability_reduction_auc=difference,
                                  top_beats_random_probability=float(difference > 0))
                    metrics_out.append(result)
            for suffix, frame in (("_metrics.csv", pd.DataFrame(metrics_out)), ("_curves.csv", pd.concat(curves_out))):
                path = Path(str(stem)+suffix)
                temporary = path.with_suffix(".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(path)
            fingerprint["output_hashes"] = {s: digest(Path(str(stem)+s)) for s in ("_metrics.csv", "_curves.csv")}
            marker.write_text(json.dumps(fingerprint, indent=2))
            print("Completed", dataset, name, seed, fold, len(partition), "targets")
    finally:
        adapter.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output
