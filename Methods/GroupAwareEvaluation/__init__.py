"""Group-aware out-of-fold classification and attribution evaluation."""

from .classification import (
    classification_metrics,
    extract_all_transformer_features,
    extract_all_uni_features,
    run_cnn_oof,
    run_transformer_oof,
    run_uni_oof,
    save_classification_artifacts,
)
from .cohort import (
    attach_model_predictions,
    build_faithfulness_cohort,
    freeze_cohort_manifest,
)
from .faithfulness import (
    evaluate_cnn_faithfulness,
    evaluate_dinov2_faithfulness,
    evaluate_transformer_faithfulness,
    evaluate_uni_faithfulness,
)
from .figures import create_main_figures, create_three_model_heatmaps
from .intervention import (
    evaluate_uni_intervention_faithfulness,
    evaluate_uni_intervention_stability,
)
from .folds import (
    build_grouped_oof_assignments,
    fold_class_group_counts,
    validate_oof_assignments,
)
from .stability import (
    evaluate_cnn_stability,
    evaluate_dinov2_stability,
    evaluate_transformer_stability,
    evaluate_uni_stability,
)
from .statistics import (
    paired_attribution_method_comparison,
    paired_method_comparisons,
    paired_stability_method_comparison,
    pairwise_model_comparisons,
    within_model_random_deletion_tests,
)

__all__ = [
    "build_faithfulness_cohort",
    "attach_model_predictions",
    "build_grouped_oof_assignments",
    "classification_metrics",
    "create_main_figures",
    "create_three_model_heatmaps",
    "evaluate_cnn_faithfulness",
    "evaluate_dinov2_faithfulness",
    "evaluate_dinov2_stability",
    "evaluate_cnn_stability",
    "evaluate_uni_faithfulness",
    "evaluate_uni_intervention_faithfulness",
    "evaluate_uni_intervention_stability",
    "evaluate_transformer_faithfulness",
    "evaluate_transformer_stability",
    "evaluate_uni_stability",
    "extract_all_transformer_features",
    "extract_all_uni_features",
    "fold_class_group_counts",
    "freeze_cohort_manifest",
    "paired_method_comparisons",
    "paired_attribution_method_comparison",
    "paired_stability_method_comparison",
    "pairwise_model_comparisons",
    "run_cnn_oof",
    "run_transformer_oof",
    "run_uni_oof",
    "save_classification_artifacts",
    "validate_oof_assignments",
    "within_model_random_deletion_tests",
]
