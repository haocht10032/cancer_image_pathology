# Publication package validation

## 2026-10-02 public update

Validation was repeated in a disposable copy without original images or pretrained
weights before publishing the updated repository snapshot.

- All 350 export checksums, 13 clean notebooks, Python syntax, and
  credential/private-path/excluded-file/size scans passed.
- All 52 compressed CSVs expanded with their decoded checksums verified.
- The current `analysis/feedback2_analysis.R` completed, skipping the optional
  original-tile gallery. Five headline tables (108 rows) matched the frozen
  exports at relative tolerance 1e-10 and absolute tolerance 1e-12, including
  confidence intervals and Holm-adjusted p-values. The fifth table was
  `external_source_class_means.csv`, in addition to the four listed below.
- Both probability table/figure scripts completed. The independent exact
  sign-flip/missingness verifier passed.
- Dataset-free toy-model tests passed for five seeds, checking intervention
  arithmetic, tied ranks, shared random controls, and deletion ordering.
- `git diff --check` passed before publication.

The PyTorch unit tests could not be repeated in this local runtime: the system
Python had an incompatible-architecture PyTorch installation and the bundled
Python did not include PyTorch. The earlier six-test result below is historical,
not a newly verified result. Model/GPU workflows and a fresh dependency install
were not rerun. R emitted locale/package-build warnings but completed.

## Original 2026-09-16 package check

Local validation completed on 2026-09-16. This is a packaging and CPU-reporting
validation, not a new model experiment or approval to distribute third-party data.

## Passed checks

- All 350 exported-file checksums matched the export manifest.
- All 13 notebooks had empty outputs and execution counts; Python source and
  notebook code cells passed syntax checks.
- Credential-pattern, private-home-path, excluded-file-type and file-size checks
  passed. Pattern scans are not a substitute for final human review.
- All 52 compressed CSV exports expanded with verified decoded checksums in a
  disposable copy without original images or pretrained weights.
- `analysis/feedback2_analysis.R` completed in that copy. The optional original-tile
  gallery was skipped because dataset images were absent.
- Regenerated headline CSVs matched the frozen outputs, with no differences outside
  numeric tolerance (relative 1e-10, absolute 1e-12):
  `image_weighted_model_means.csv` (12 rows), `paired_inference.csv` (36 rows),
  `probability_model_means.csv` (12 rows), and
  `probability_paired_inference.csv` (24 rows).
- Both `build_probability_tables.R` and `plot_probability_results.R` completed.
- All six `Methods.Feedback2.test_probability_deletion` unit tests passed on CPU.

R reported locale/package-build-version warnings but completed successfully.
The observed reporting versions are recorded in
`environment/observed_reporting_runtime.tsv`.

## Not certified by this check

GPU training, checkpoint replay, dataset downloading and a fresh installation of
all model dependencies were not rerun. The recorded model requirements are not a
complete environment lock. Full per-patch archives and checkpoints are not bundled.
License selection, redistribution rights and author approval remain release tasks.
No files were uploaded during the original package preparation; this statement
does not describe subsequent public repository updates.
