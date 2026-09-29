# Places365-Standard frozen OOD check

All 36,500 official Places365-Standard validation images were scored as one previously unscored natural-OOD population against the same 10,000 CIFAR-100 ID-test images. Three frozen isolated classifiers (selected epochs 92/96/98) and their ID-only temperatures and thresholds were reused. The protocol was frozen before download; [amendment 1](../../docs/PLACES365_AMENDMENT_1.md) changed only the category metadata retrieval path before archive download. No OOD image was used to select a model, fit a detector or choose a threshold.

Each table value is the mean ± sample SD across three training seeds, in percent. Individual seed points and exact threshold counts are in [summary.json](summary.json). AUROC and AUPR use 10,000 ID + 36,500 OOD samples, so OOD prevalence is 78.49%. FPR95-ID uses the ID **test** order statistic as a benchmark operating point; fixed95 OOD/ID acceptance uses the pre-existing 1,500-row ID-validation threshold.

## T1

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 75.52 ± 0.59 | 90.06 ± 0.38 | 51.08 ± 0.95 | 81.73 ± 1.61 | 82.42 ± 0.22 | 95.20 ± 0.51 |
| Energy | 75.93 ± 0.85 | 90.09 ± 0.43 | 49.89 ± 1.62 | 82.55 ± 1.55 | 83.43 ± 1.08 | 95.18 ± 0.75 |
| MaxLogit | 76.15 ± 0.84 | 90.30 ± 0.44 | 50.02 ± 1.59 | 81.60 ± 1.75 | 82.97 ± 0.87 | 95.38 ± 0.68 |
| Entropy | 76.43 ± 0.67 | 90.42 ± 0.42 | 51.59 ± 1.03 | 81.18 ± 1.76 | 82.18 ± 0.58 | 95.25 ± 0.70 |
| KNN-50 | 75.56 ± 0.84 | 90.39 ± 0.48 | 48.54 ± 0.84 | 79.93 ± 1.68 | 81.62 ± 1.70 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 59.45 ± 0.48 | 81.77 ± 0.22 | 33.68 ± 0.79 | 94.36 ± 0.27 | 94.69 ± 0.56 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 50.36 ± 0.41 | 69.72 ± 0.59 | 82.42 ± 0.22 | 95.70 ± 0.42 |
| Energy | 47.74 ± 0.88 | 70.30 ± 0.79 | 83.43 ± 1.08 | 96.75 ± 0.37 |
| MaxLogit | 47.68 ± 0.91 | 69.81 ± 0.75 | 82.97 ± 0.87 | 96.24 ± 1.01 |
| Entropy | 46.61 ± 0.75 | 67.87 ± 0.79 | 82.18 ± 0.58 | 95.94 ± 0.68 |
| KNN-50 | 48.95 ± 1.34 | 68.75 ± 1.14 | 81.62 ± 1.70 | 94.70 ± 1.00 |
| Pooled Mahalanobis | 73.84 ± 0.42 | 87.87 ± 0.81 | 94.69 ± 0.56 | 99.31 ± 0.14 |

## calibrated

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 76.14 ± 0.66 | 90.33 ± 0.41 | 51.33 ± 1.04 | 81.42 ± 1.75 | 81.65 ± 0.43 | 95.06 ± 0.67 |
| Energy | 75.68 ± 0.84 | 89.92 ± 0.42 | 49.71 ± 1.62 | 83.18 ± 1.53 | 84.08 ± 1.12 | 95.27 ± 0.74 |
| MaxLogit | 76.15 ± 0.84 | 90.30 ± 0.44 | 50.02 ± 1.59 | 81.60 ± 1.75 | 82.97 ± 0.87 | 95.38 ± 0.68 |
| Entropy | 76.73 ± 0.73 | 90.46 ± 0.42 | 51.72 ± 1.14 | 81.53 ± 1.46 | 82.54 ± 0.44 | 95.26 ± 0.60 |
| KNN-50 | 75.56 ± 0.84 | 90.39 ± 0.48 | 48.54 ± 0.84 | 79.93 ± 1.68 | 81.62 ± 1.70 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 59.45 ± 0.48 | 81.77 ± 0.22 | 33.68 ± 0.79 | 94.36 ± 0.27 | 94.69 ± 0.56 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 47.77 ± 0.77 | 68.70 ± 0.38 | 81.65 ± 0.43 | 96.05 ± 0.53 |
| Energy | 48.15 ± 2.15 | 71.19 ± 0.87 | 84.08 ± 1.12 | 96.88 ± 0.33 |
| MaxLogit | 47.68 ± 0.91 | 69.81 ± 0.75 | 82.97 ± 0.87 | 96.24 ± 1.01 |
| Entropy | 46.87 ± 1.24 | 69.26 ± 0.59 | 82.54 ± 0.44 | 96.24 ± 0.60 |
| KNN-50 | 48.95 ± 1.34 | 68.75 ± 1.14 | 81.62 ± 1.70 | 94.70 ± 1.00 |
| Pooled Mahalanobis | 73.84 ± 0.42 | 87.87 ± 0.81 | 94.69 ± 0.56 | 99.31 ± 0.14 |

## Evidence and scope

Per-image raw logits/features, predictions, source filenames, twelve score vectors and all 80/90/95/99 ID/OOD acceptance masks are retained locally for each seed. The image cache and each seed's feature scores were independently audited against decoded archive pixels and refitted training-ID feature banks. Score audits independently recomputed all formulas, 468 metric fields, 432 masks, and fixed threshold counts across the three seeds. [Completion manifest](completion.json) binds the report, figures, audits and regression evidence.

This one sequentially chosen natural-OOD dataset contains potential scene/object overlap with CIFAR-100 and a 256→32 source/resolution shift. These measurements do not isolate those factors, establish universal generalization or reproduce an OpenOOD leaderboard. All six methods and both temperature variants are reported in their frozen order; there is no post-test winner selection. Sample SD reflects only three trained classifier seeds, not image-level confidence intervals.

Figures: [AUROC](auroc.png) and [fixed95 OOD acceptance](fixed95-acceptance.png).
