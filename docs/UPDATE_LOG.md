# Publication update log

## 2026-10-02 scoped licensing update

The maintainer authorized using MIT for original project software and CC BY 4.0
for original documentation and authorized compact numerical outputs while final
ownership/coauthor confirmation is recorded. Added the license texts, exact scope,
and third-party notices. The compact MaskCut implementation is excluded from the
MIT grant pending provenance review. No written coauthor approval is asserted.

This is a current-repository licensing/documentation update only. No scientific
code, trained models, or frozen results were changed. The historical `v1.0.0`
tag and Zenodo record remain unchanged, with their existing CC BY 4.0 metadata.
The separate full perturbation archive remains unpublished.

## 2026-10-02 software DOI follow-up

The published Zenodo software record for `v1.0.0` is
https://doi.org/10.5281/zenodo.23105039, corresponding to GitHub commit
`e248c59a4094913d150b8a3652e57700d35ba38e`. Its version-specific DOI is now in the
README, citation metadata and manuscript. The following reporting update was
archived in that release; this DOI documentation is a post-release change and
does not move or replace the tag.

The full perturbation archive remains unpublished. Zenodo currently lists CC BY
4.0 for the software record; author confirmation of that license remains pending.
No LICENSE was added and no archive license was changed in this follow-up.

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

The repository and educational demonstration are public. This initial reporting
update did not issue an archive DOI or a public download for the full perturbation files. Those
files require a separate, checked archive. See [release tasks](RELEASE_CHECKLIST.md)
and [large-output contents](LARGE_ARTIFACTS.md).

Validation results for this snapshot are recorded in
[PACKAGE_VALIDATION.md](PACKAGE_VALIDATION.md). No classification, attribution, or
GPU deletion experiment was rerun.
