# Provenance and sanitization

The source working directory and online experiment outputs are unchanged. This
folder is a publication copy. Machine-specific path prefixes are converted to
repository-relative paths; unrelated personal paths are replaced with an explicit
configuration marker. Notebook outputs, execution counts, attachments, and transient
metadata are removed. Root discovery is made portable. Scientific numeric CSV
fields are preserved as strings during the CSV roundtrip, not rounded or recomputed.

`provenance/export_manifest.json` records the original file checksum, published
checksum and decoded checksum for each exported file. Sanitized/compressed files
will not have the original experiment's byte hash. Old metadata hashes remain
historical provenance, not validation signatures for the new export.

`scripts/verify_publication.py` validates this public export. The original
`Methods/Feedback2/audit_downloads.py` is for byte-identical original archives,
with original inputs and checkpoints; it is not the verifier for sanitized files.
Notebook 14 outputs included here are final aggregate image-seed metrics. Full
curves with deleted indices and original partition signatures belong in the
separate archive after its own privacy and rights review.

The online notebook 14 completed 99 full partitions and passed source/checkpoint/
output hash, manifest, probability arithmetic, AUC and tie-free/random validation
checks before packaging. Public verification does not imply the GPU workflows
were rerun from this cleaned package.

Raw executed notebooks are not included; their outputs can contain paths, images
and environment information. Only cleaned workflow notebooks are public candidates.
Never upload the private manuscript feedback or correspondence as provenance.

Restoring original GPU archives may require path reconciliation or a clean new run:
sanitized manifests cannot be assumed compatible with original byte-hash resume
checks. Do not disable replay or provenance guards to force compatibility. See
`PACKAGE_VALIDATION.md` for the specific reporting checks completed on this package.
