"""Final three-model classification and attribution analysis."""

from .analysis import (
    CLASSIFICATION_METRICS,
    FAITHFULNESS_METRICS,
    FIXED_DELETION_METRICS,
    MODEL_ORDER,
    PRIMARY_METHODS,
    classification_seed_tests,
    classification_source_tests,
    classification_summary,
    faithfulness_summary,
    focused_faithfulness_tests,
    incorrect_intervention_exploratory,
    load_intervention_artifacts,
    load_three_model_artifacts,
    primary_faithfulness_rows,
    stability_pairwise_tests,
    stability_summary,
)
from .figures import create_final_figures

__all__ = [
    "CLASSIFICATION_METRICS",
    "FAITHFULNESS_METRICS",
    "FIXED_DELETION_METRICS",
    "MODEL_ORDER",
    "PRIMARY_METHODS",
    "classification_seed_tests",
    "classification_source_tests",
    "classification_summary",
    "create_final_figures",
    "faithfulness_summary",
    "focused_faithfulness_tests",
    "incorrect_intervention_exploratory",
    "load_intervention_artifacts",
    "load_three_model_artifacts",
    "primary_faithfulness_rows",
    "stability_pairwise_tests",
    "stability_summary",
]
