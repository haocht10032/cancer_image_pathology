"""Taxonomy-harmonized Kather-to-CRC external validation."""

from .classification import (
    COMMON_CLASS_NAMES,
    build_crc_common_manifest,
    build_kather_common_manifest,
    classification_tables,
    extract_transformer_cache,
    train_evaluate_resnet,
    train_evaluate_transformer_heads,
)
from .faithfulness import (
    ExternalCNNAdapter,
    ExternalTransformerAdapter,
    build_external_target_index,
    finalize_analysis_families,
    paired_external_inference,
    recompute_live_cohort_predictions,
)

__all__ = [
    "COMMON_CLASS_NAMES",
    "build_crc_common_manifest",
    "build_kather_common_manifest",
    "classification_tables",
    "extract_transformer_cache",
    "train_evaluate_resnet",
    "train_evaluate_transformer_heads",
    "ExternalCNNAdapter",
    "ExternalTransformerAdapter",
    "build_external_target_index",
    "finalize_analysis_families",
    "paired_external_inference",
    "recompute_live_cohort_predictions",
]
