# Public numerical results archive

The full recorded numerical perturbation and stability outputs are publicly
available as dataset version `1.0.0` at
https://doi.org/10.5281/zenodo.23143934 (published 2026-10-04).
This is separate from the code/compact-output DOI
https://doi.org/10.5281/zenodo.23105039.

## Contents

- `kather_final_perturbation_outputs.zip`: final Kather revision, completed
  sensitivities, classification, folds/cohorts, and historical supporting inputs.
- `crc_final_perturbation_outputs.zip`: final common-seven external outputs,
  corrected consolidated maps/curves/occlusion scores, predictions and manifests.
- `feedback2_final_probability_deletion.zip`: final `deterministic_ties_v5/full/`
  partition outputs, including per-step logits/probabilities, deleted patch indices,
  provenance and frozen targets; earlier probability/proof runs are excluded.
- `cross_seed_stability.zip`: recorded predictions, target classes, per-patch
  maps, seed-pair metrics and final stability summaries.
- `secondary_activation_patching_ablation.zip`: penultimate-layer,
  within-image-mean intervention and stability outputs, not a new training loss.
- `reproduction_context.zip`: observed runtimes, requirements and export provenance,
  not a complete dependency lock.

The record also contains a README, CC BY 4.0 license, per-file manifest,
`SHA256SUMS.txt`, and validation report. All 11 uploaded file sizes and Zenodo MD5
checksums matched the prepared local package on 2026-10-04. The export contains
591 numerical files and 15,979,144 CSV data rows. This is packaging validation,
not an independent GPU reproduction or a complete rights audit.

## Restore and verify

Download the required ZIPs and supporting files. Verify the upload-file checksums
with `shasum -a 256 -c SHA256SUMS.txt`. Numerical ZIP members preserve `artifacts/`
paths; extract into a separate analysis checkout at its root. The manifest's
`published_sha256` fields validate extracted files. Inspect reproduction context
separately rather than overwriting the repository environment records.

Machine-specific paths were sanitized and numeric CSV strings were preserved
without recomputation or rounding. Original source artifacts were not changed.
Nested experiment hashes refer to original unsanitized bytes; use the archive
manifest for exported-file validation. Do not disable original byte-hash replay
guards to force these sanitized files to match an original experiment.

Five existing summary tables have duplicate column labels, retained and documented
in the manifest/report. Use image-/patch-level tables for fresh aggregation. Different
stages have different cohorts, targets, seeds and class encodings; do not pool
historical and final snapshots as independent samples. See the archive README.

## Exclusions and limits

Original images, pretrained weights, trained checkpoints, embeddings/patch tokens,
private documents and obsolete/proof outputs are excluded. The archive supports
reanalysis of saved outputs, not standalone GPU inference. Obtain images and
authorized models separately; trained checkpoints must be retained or regenerated.

`provenance/large_artifacts_inventory.csv` is an older candidate inventory and can
list excluded or duplicate materials. It is not this record's content manifest or
permission to redistribute each entry. Use the published `file_manifest.csv`.
