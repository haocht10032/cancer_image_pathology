# Workflow index

## Final study

| Notebook | Purpose | Requirements |
|---|---|---|
| 04 | Source-grouped folds, UNI/CNN OOF and cohort | Images; GPU for training/explanation |
| 05 | DINOv2 three-model comparison | 04 outputs; images, encoders, GPU |
| 06 | Activation-patching negative ablation | Frozen UNI and saved cohort; GPU |
| 07 | Historical three-model reporting | Earlier experiment outputs; mostly CPU |
| 08 | Representative region visualization | Images and saved maps/checkpoints |
| 10 | Final bounded Kather revision | OOF outputs, maps/checkpoints; GPU for new perturbations |
| 11 | External dataset and taxonomy preflight | CRC and Kather images; CPU |
| 12 | Refit seven-class Kather models; external inference | Both datasets and authorized weights; GPU |
| 13 | External attribution evaluation | 11/12 outputs; GPU |
| 14 | Final probability sensitivity, deterministic ties v5 | Saved maps/curves/checkpoints; GPU |

Notebooks 01-03 are historical undergraduate reproduction and initial baselines,
not the final statistical evidence. Their original fixed-split results must not be
conflated with grouped OOF results. The main sequence is not an instruction to
rerun expensive completed experiments: use the compact reporting path first.
Notebook 09 is intentionally omitted as superseded by notebook 10.

Three-model primary explanation methods remain Grad-CAM for ResNet18 and
gradient-weighted rollout for DINOv2/UNI. Integrated Gradients code exists but
was not part of the finalized bounded comparison. Do not claim it was completed.
No causal-ranking loss exists or is justified by the activation-patching ablation.

Frozen folds and manifests are under `artifacts/grouped_oof_faithfulness/folds/`,
the grouped cohort manifest, `artifacts/kather5k_jpi_revision/cohorts/`, and
`artifacts/crc_val_external/preflight/`. CSV paths are repository-relative. Notebook
bootstrap cells change the working directory to the repository root.

Seeds: original main classification 10 seeds; bounded revision and probability
sensitivity 11/89/181. Do not pool across protocol versions. Actual common
class mappings are in `Methods/KatherRevision/CRC_VAL_STUDY_CONFIG.json` and the
external preflight artifacts, rather than inferred from matching folder names.
