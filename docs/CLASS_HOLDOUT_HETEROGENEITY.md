# Frozen class-holdout heterogeneity diagnostic v0.1 — 2026-09-28

## Question and boundary

Describe how the **already completed** CIFAR-100 80-ID/20-held-out-fine-class OOD result varies across all 20 held-out fine classes. This is a post-hoc analysis of the same official test images and the same three frozen models, not a new confirmation set or a basis for detector, temperature, threshold, class, or checkpoint selection. The three-seed aggregate in `report/cifar100-class-holdout/` remains unchanged.

This declaration is made before inspecting or computing any held-out fine-class performance. The inputs are the audited 160-file class-holdout completion, its three seed-specific ID-test and OOD-test scores/acceptance masks, and the frozen official 8,000-ID/2,000-OOD row/class mapping. No training, GPU extraction, detector fitting, or threshold fitting occurs here.

## Full population and fixed calculations

- Include exactly all 20 held-out fine classes in ascending original CIFAR-100 fine ID; each supplies its complete 100 official test images. Preserve original row indices and fine/coarse class IDs/names. Never select classes by score.
- For every seed 0/1/2, both `T1` and `calibrated` variants, and all six fixed methods (`msp`, `energy`, `maxlogit`, `entropy`, `knn`, `mahalanobis`), calculate per-class AUROC against the **same full 8,000 ID-test scores**. Higher score means more OOD; tied ID/OOD pairs count one half. The 20 equal-sized class AUROCs must average to the saved overall AUROC for each seed/method/variant, up to floating-point tolerance.
- For each of the four existing ID-validation targets 80/90/95/99, count each class's accepted OOD images directly from the saved fixed-threshold OOD mask. Save integer counts and rates (`count/100`). The 20 class counts must sum exactly to the saved full-2,000 OOD count for the same seed/method/variant/target. Also show the actual full 8,000 ID-test acceptance from the original score manifest, not an inferred 80/90/95/99% test coverage.
- Retain the full `[seed, variant, method, held-out class]` AUROC array and `[seed, variant, method, target, held-out class]` acceptance-count array with axes and source hashes. Publish all classes and methods in CSV, with per-seed values, mean and **sample SD across the three training seeds**. A class has only 100 OOD test images, so a single-seed acceptance rate changes in 1% steps. Report class-median and 25th/75th percentiles as descriptive distributions of the 20 class means, never as confidence intervals.
- AUPR is omitted from the per-class diagnostic because restricting OOD to 100 images while retaining 8,000 ID changes prevalence. The complete-population AUPR remains in the already audited aggregate report. No per-class method winner, p-value, causal semantic-distance statement, macro comparison with another study, or new hyperparameter is produced.

## Verification

Before writing results, bind the declaration to this protocol, the completed 160-file manifest, frozen test-population audit, every seed's independent score audit and score manifest. A separate CPU implementation recomputes all per-class AUROCs with a different ranking/metric reference, all class acceptance counts directly from raw scores and frozen thresholds/masks, the equal-class aggregate identities, every CSV/statistical field, and source/raw hashes. Keep all outputs local and write-once; reject incomplete or modified baseline evidence.
