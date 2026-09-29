# Oxford-IIIT Pet frozen OOD check

All 3,669 official Oxford-IIIT Pet test images were scored as one previously unscored natural-OOD population against the same 10,000 CIFAR-100 ID-test images. Three frozen isolated classifiers (selected epochs 92/96/98) and their ID-only temperatures and thresholds were reused. The protocol was frozen before archive download. No OOD image was used to select a model, fit a detector or choose a threshold.

Each table value is the mean ± sample SD across three training seeds, in percent. Individual seed points and exact threshold counts are in [summary.json](summary.json). AUROC and AUPR use 10,000 ID + 3,669 OOD samples, so OOD prevalence is 26.84%. FPR95-ID uses the ID **test** order statistic as a benchmark operating point; fixed95 OOD/ID acceptance uses the pre-existing 1,500-row ID-validation threshold.

## T1

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 80.84 ± 0.67 | 56.16 ± 1.46 | 92.37 ± 0.37 | 75.45 ± 2.09 | 76.45 ± 0.61 | 95.20 ± 0.51 |
| Energy | 81.60 ± 0.78 | 56.51 ± 1.37 | 92.52 ± 0.43 | 76.02 ± 1.50 | 76.76 ± 2.01 | 95.18 ± 0.75 |
| MaxLogit | 81.87 ± 0.77 | 57.39 ± 1.47 | 92.61 ± 0.42 | 74.86 ± 1.74 | 76.39 ± 1.10 | 95.38 ± 0.68 |
| Entropy | 82.10 ± 0.76 | 57.82 ± 1.54 | 92.82 ± 0.40 | 74.46 ± 1.77 | 75.59 ± 0.93 | 95.25 ± 0.70 |
| KNN-50 | 83.56 ± 0.09 | 56.82 ± 0.84 | 93.96 ± 0.02 | 77.33 ± 1.26 | 79.52 ± 1.82 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 56.56 ± 0.63 | 29.08 ± 0.52 | 80.10 ± 0.55 | 95.56 ± 0.78 | 95.88 ± 0.49 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 40.73 ± 0.44 | 61.02 ± 0.55 | 76.45 ± 0.61 | 93.65 ± 0.85 |
| Energy | 37.51 ± 1.48 | 61.54 ± 0.43 | 76.76 ± 2.01 | 94.89 ± 0.39 |
| MaxLogit | 37.18 ± 1.86 | 60.75 ± 0.74 | 76.39 ± 1.10 | 94.51 ± 1.23 |
| Entropy | 35.87 ± 1.15 | 58.43 ± 1.19 | 75.59 ± 0.93 | 93.95 ± 0.95 |
| KNN-50 | 35.56 ± 0.26 | 62.30 ± 1.98 | 79.52 ± 1.82 | 95.86 ± 0.88 |
| Pooled Mahalanobis | 77.58 ± 1.08 | 89.87 ± 0.71 | 95.88 ± 0.49 | 99.48 ± 0.11 |

## calibrated

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 81.70 ± 0.71 | 57.37 ± 1.54 | 92.68 ± 0.38 | 74.52 ± 2.00 | 75.07 ± 1.05 | 95.06 ± 0.67 |
| Energy | 81.29 ± 0.78 | 55.86 ± 1.36 | 92.41 ± 0.43 | 76.74 ± 1.46 | 77.89 ± 2.00 | 95.27 ± 0.74 |
| MaxLogit | 81.87 ± 0.77 | 57.39 ± 1.47 | 92.61 ± 0.42 | 74.86 ± 1.74 | 76.39 ± 1.10 | 95.38 ± 0.68 |
| Entropy | 82.53 ± 0.76 | 57.88 ± 1.43 | 93.03 ± 0.41 | 74.66 ± 1.32 | 75.90 ± 0.89 | 95.26 ± 0.60 |
| KNN-50 | 83.56 ± 0.09 | 56.82 ± 0.84 | 93.96 ± 0.02 | 77.33 ± 1.26 | 79.52 ± 1.82 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 56.56 ± 0.63 | 29.08 ± 0.52 | 80.10 ± 0.55 | 95.56 ± 0.78 | 95.88 ± 0.49 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 37.47 ± 1.20 | 59.42 ± 0.90 | 75.07 ± 1.05 | 94.16 ± 0.81 |
| Energy | 38.14 ± 2.82 | 62.51 ± 1.28 | 77.89 ± 2.00 | 95.02 ± 0.15 |
| MaxLogit | 37.18 ± 1.86 | 60.75 ± 0.74 | 76.39 ± 1.10 | 94.51 ± 1.23 |
| Entropy | 35.81 ± 1.86 | 59.76 ± 0.51 | 75.90 ± 0.89 | 94.21 ± 0.94 |
| KNN-50 | 35.56 ± 0.26 | 62.30 ± 1.98 | 79.52 ± 1.82 | 95.86 ± 0.88 |
| Pooled Mahalanobis | 77.58 ± 1.08 | 89.87 ± 0.71 | 95.88 ± 0.49 | 99.48 ± 0.11 |

## Evidence and scope

Per-image raw logits/features, predictions, source filenames, twelve score vectors and all 80/90/95/99 ID/OOD acceptance masks are retained locally for each seed. The image cache and each seed's feature scores were independently audited against decoded archive pixels and refitted training-ID feature banks. Score audits independently recomputed all formulas, 468 metric fields, 432 masks, and fixed threshold counts across the three seeds. [Completion manifest](completion.json) binds the report, figures, audits and regression evidence.

This one sequentially chosen natural-OOD dataset contains potential cat/dog semantic overlap with CIFAR-100 and a large-source-to-32 resize. These measurements do not isolate those factors, establish universal generalization or reproduce an OpenOOD leaderboard. All six methods and both temperature variants are reported in their frozen order; there is no post-test winner selection. Sample SD reflects only three trained classifier seeds, not image-level confidence intervals.

Figures: [AUROC](auroc.png) and [fixed95 OOD acceptance](fixed95-acceptance.png).
