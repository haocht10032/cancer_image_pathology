"""UNI linear probing and patch-level attribution workflow."""

from .attribution import (
    GradCAM,
    cache_patch_tokens,
    deletion_auc,
    deletion_curves,
    patch_occlusion_scores,
    standardize_attribution_grid,
    transformer_attributions,
)
from .data import (
    UNI_GRID_SIZE,
    UNI_IMAGE_SIZE,
    UNI_PATCH_SIZE,
    UNIImageDataset,
    build_feature_loaders,
    build_image_loaders,
    load_fixed_manifest,
)
from .evaluation import (
    evaluate_cnn_examples,
    evaluate_uni_examples,
    evaluate_uni_seed_stability,
    load_normalized_image,
    select_representative_examples,
)
from .intervention import (
    intervention_patch_attribution,
    resolve_intervention_layer,
)
from .model import (
    UNIClassifier,
    build_uni_classifier,
    load_classifier_head,
    resolve_uni_transform,
)
from .training import (
    classification_row,
    evaluate_frozen_head,
    extract_cls_feature_cache,
    fit_frozen_head,
)

__all__ = [
    "GradCAM",
    "UNIClassifier",
    "UNI_GRID_SIZE",
    "UNI_IMAGE_SIZE",
    "UNI_PATCH_SIZE",
    "UNIImageDataset",
    "build_feature_loaders",
    "build_image_loaders",
    "build_uni_classifier",
    "cache_patch_tokens",
    "classification_row",
    "deletion_auc",
    "deletion_curves",
    "evaluate_cnn_examples",
    "evaluate_frozen_head",
    "evaluate_uni_examples",
    "evaluate_uni_seed_stability",
    "extract_cls_feature_cache",
    "fit_frozen_head",
    "load_classifier_head",
    "load_fixed_manifest",
    "load_normalized_image",
    "intervention_patch_attribution",
    "patch_occlusion_scores",
    "resolve_uni_transform",
    "resolve_intervention_layer",
    "select_representative_examples",
    "standardize_attribution_grid",
    "transformer_attributions",
]
