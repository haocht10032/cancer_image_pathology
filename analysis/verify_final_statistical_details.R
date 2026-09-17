#!/usr/bin/env Rscript

# CPU-only independent verification of frozen sign-flip and missing-value results.
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
file_arg <- gsub("~+~", " ", file_arg, fixed = TRUE)
script_dir <- dirname(normalizePath(file_arg, mustWork = TRUE))
project_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
output_dir <- file.path(script_dir, "manuscript_revision_outputs")
statistics_dir <- file.path(project_root, "artifacts/kather5k_jpi_revision/statistics")
tests <- read.csv(file.path(statistics_dir, "paired_model_tests.csv"))
values <- read.csv(file.path(statistics_dir, "paired_model_values.csv"))

primary_rows <- function(frame) {
  subset(frame,
    cohort_name == "selected_faithfulness" &
    analysis_family == "primary_jointly_correct_true_class" &
    method_family == "architecture_primary" &
    perturbation == "normalized_zero" & grid_label == "common_14" &
    metric %in% c("attribution_occlusion_spearman",
                  "top_minus_random_relative_target_logit_reduction_auc"))
}
tests <- primary_rows(tests)
values <- primary_rows(values)
stopifnot(nrow(tests) == 6L)

sign_audit <- do.call(rbind, lapply(seq_len(nrow(tests)), function(i) {
  test <- tests[i, ]
  paired <- subset(values, metric == test$metric & model_a == test$model_a &
                    model_b == test$model_b)
  stopifnot(!anyDuplicated(paired[c("case_id", "cohort_id")]))
  source_effects <- tapply(paired$difference, paired$case_id, mean, na.rm = TRUE)
  source_effects <- source_effects[is.finite(source_effects)]
  n_groups <- length(source_effects)
  stopifnot(n_groups == 10L)
  signs <- as.matrix(expand.grid(rep(list(c(-1, 1)), n_groups)))
  stopifnot(nrow(signs) == 1024L, nrow(unique(signs)) == 1024L)
  observed <- mean(source_effects)
  null_statistics <- as.vector(signs %*% source_effects) / n_groups
  # Protect equality at the two observed sign configurations from rounding.
  tolerance <- 1e-12 * max(1, abs(observed))
  extreme <- abs(null_statistics) >= abs(observed) - tolerance
  p <- mean(extreme)
  stopifnot(abs(observed - test$source_sign_flip_estimate) < 1e-10,
            abs(p - test$source_sign_flip_p) < 1e-10,
            test$source_sign_flip_type == "exact")
  data.frame(metric = test$metric, model_a = test$model_a, model_b = test$model_b,
    images = nrow(paired), source_groups = n_groups,
    sign_configurations = nrow(signs), extreme_configurations = sum(extreme),
    source_mean_difference = observed, recomputed_p = p,
    saved_p = test$source_sign_flip_p,
    saved_holm_p = test$source_sign_flip_p_holm)
}))
sign_audit$recomputed_holm_p <- ave(sign_audit$recomputed_p, sign_audit$metric,
                                  FUN = function(p) p.adjust(p, "holm"))
stopifnot(all(abs(sign_audit$recomputed_holm_p - sign_audit$saved_holm_p) < 1e-10))

external <- read.csv(file.path(project_root,
  "artifacts/crc_val_external/common_seven/faithfulness/external_faithfulness_metrics.csv"))
missing <- external[!is.finite(external$attribution_occlusion_spearman), ]
stopifnot(nrow(missing) == 1L, missing$model == "ResNet18",
  missing$cohort_id == "ee8856c731ec4e6f", missing$seed == 181L,
  missing$analysis_family == "all_image_true_class_sensitivity",
  tolower(as.character(missing$all_models_correct)) == "false")
conditional <- subset(external, analysis_family == "primary_jointly_correct_true_class")
counts <- do.call(rbind, lapply(split(conditional, conditional$model), function(d) {
  data.frame(model = unique(d$model), image_seed_observations = nrow(d),
    unique_images = length(unique(d$cohort_id)),
    finite_correlations = sum(is.finite(d$attribution_occlusion_spearman)))
}))
stopifnot(nrow(counts) == 3L, all(counts$image_seed_observations == 106L),
  all(counts$unique_images == 48L), all(counts$finite_correlations == 106L))
resnet <- subset(external, model == "ResNet18")
stopifnot(nrow(resnet) == 498L,
  sum(is.finite(resnet$attribution_occlusion_spearman)) == 497L)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(sign_audit, file.path(output_dir, "exact_sign_flip_verification.csv"), row.names = FALSE)
write.csv(counts, file.path(output_dir, "external_conditional_missingness_audit.csv"), row.names = FALSE)
write.csv(missing[c("cohort_id", "model", "seed", "class_name", "target_role",
                   "analysis_family", "all_models_correct")],
  file.path(output_dir, "external_undefined_correlation_audit.csv"), row.names = FALSE)
cat("Verified six primary contrasts: all 1,024 sign patterns; saved raw and Holm p-values match.\n")
cat("The undefined correlation is outside Table 5: all models retain 106 observations / 48 images.\n")
