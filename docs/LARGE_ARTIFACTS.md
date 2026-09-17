# Separate research archive

This repository is deliberately not a full checkpoint/data mirror. The inventory
`provenance/large_artifacts_inventory.csv` lists excluded derived files by relative
path, size and category. It is an inventory, not permission to redistribute each
entry and not a downloadable archive. No DOI/download URL exists yet.

Before an archival release, select and sanitize the required final outputs:

- Original grouped OOF and external trained checkpoints, only if redistribution
  terms permit. Checkpoints/features containing restricted model information
  need separate review; do not assume UNI weights are redistributable.
- Kather consolidated attribution maps, occlusion scores and deletion curves.
- Final CRC attribution maps, occlusion scores and deletion curves.
- Cross-seed maps needed to recompute stability from scratch.
- `deterministic_ties_v5/full/` partition curves with all logits, deletion indices,
  metrics and provenance. Exclude earlier probability versions and proof runs.
- Original run configurations and environment export, sanitized as necessary.

The inventory may include intermediate or overlapping files. Deduplicate and freeze
a minimal approved archive; do not blindly upload every inventory entry. Keep
numeric values and image/source identifiers consistent with the compact exports.
Publish an archive README, file checksums, and its relationship to a tagged code
release. Restore files at their documented project-relative paths to rerun the
GPU replay/audit; read PROVENANCE.md before using historical hashes.

No large artifacts were moved, deleted, uploaded or repackaged by this preparation.
