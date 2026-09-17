# Publication package validation

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
No files were uploaded to GitHub during preparation.
