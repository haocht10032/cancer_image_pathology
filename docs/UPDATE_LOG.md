# Publication update log

## 2026-10-02

This update synchronizes the public code and compact outputs with the finalized
manuscript reporting. It is not a new model-training experiment or a tagged
archival release.

- Updated reporting scripts and generated-table wording to distinguish
  probability-drop AUC from deletion-success consistency.
- Clarified that source-weighted Kather tests support UNI's greater consistency
  over DINOv2 in the selected and jointly correct random subsets. Probability-drop
  AUC does not establish a UNI-over-DINOv2 difference in the four analyses.
- Retained the external analysis as exploratory tile-level inference because
  patient identifiers are unavailable. The conditional cohort is not a
  population-wide external faithfulness estimate.
- Re-exported the compact numerical outputs and updated script checksum mappings;
  the frozen scientific values are unchanged.
- Kept original images, restricted pretrained weights, private correspondence,
  manuscript drafts, and large per-patch/per-step files out of GitHub.

The repository and educational demonstration are public. No archive DOI or public
download for the full perturbation files has been issued by this update. Those
files require a separate, checked archive. See [release tasks](RELEASE_CHECKLIST.md)
and [large-output contents](LARGE_ARTIFACTS.md).

Validation results for this snapshot are recorded in
[PACKAGE_VALIDATION.md](PACKAGE_VALIDATION.md). No classification, attribution, or
GPU deletion experiment was rerun.
