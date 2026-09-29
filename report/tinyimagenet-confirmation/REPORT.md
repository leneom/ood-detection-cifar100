# Tiny ImageNet-200 frozen OOD check

All 10,000 official Tiny ImageNet validation images were scored as one previously unscored natural-OOD population against the same 10,000 CIFAR-100 ID-test images. Three frozen isolated classifiers (selected epochs 92/96/98) and their ID-only temperatures and thresholds were reused. The protocol was frozen before download; [amendment 1](../../docs/TINYIMAGENET_CONFIRMATION_AMENDMENT_1.md) corrected only the input-array layout before image decoding. No OOD image was used to select a model, fit a detector or choose a threshold.

Each table value is the mean ± sample SD across three training seeds, in percent. Individual seed points and exact threshold counts are in [summary.json](summary.json). AUROC and AUPR use 10,000 ID + 10,000 OOD samples, so OOD prevalence is 50%. FPR95-ID uses the ID **test** order statistic as a benchmark operating point; fixed95 OOD/ID acceptance uses the pre-existing 1,500-row ID-validation threshold.

## T1

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 78.02 ± 0.19 | 74.56 ± 0.50 | 79.27 ± 0.23 | 78.15 ± 1.23 | 78.88 ± 1.11 | 95.20 ± 0.51 |
| Energy | 78.95 ± 0.33 | 75.41 ± 0.54 | 79.34 ± 0.41 | 77.86 ± 0.98 | 78.80 ± 2.25 | 95.18 ± 0.75 |
| MaxLogit | 79.12 ± 0.32 | 75.75 ± 0.58 | 79.45 ± 0.40 | 77.07 ± 1.31 | 78.60 ± 1.63 | 95.38 ± 0.68 |
| Entropy | 79.19 ± 0.23 | 75.86 ± 0.52 | 79.94 ± 0.26 | 76.86 ± 1.32 | 77.85 ± 1.66 | 95.25 ± 0.70 |
| KNN-50 | 80.17 ± 0.28 | 76.46 ± 0.55 | 81.42 ± 0.29 | 76.82 ± 0.98 | 78.65 ± 2.75 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 58.81 ± 0.45 | 54.69 ± 0.45 | 62.35 ± 0.45 | 94.49 ± 0.21 | 94.82 ± 0.67 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 45.87 ± 1.17 | 65.33 ± 0.90 | 78.88 ± 1.11 | 94.58 ± 0.42 |
| Energy | 41.36 ± 0.08 | 64.36 ± 1.33 | 78.80 ± 2.25 | 95.28 ± 0.66 |
| MaxLogit | 41.45 ± 0.46 | 64.06 ± 1.78 | 78.60 ± 1.63 | 94.75 ± 1.44 |
| Entropy | 41.01 ± 0.75 | 62.43 ± 2.56 | 77.85 ± 1.66 | 94.32 ± 1.07 |
| KNN-50 | 39.98 ± 0.71 | 62.81 ± 2.37 | 78.65 ± 2.75 | 94.55 ± 0.84 |
| Pooled Mahalanobis | 74.61 ± 0.22 | 88.24 ± 0.47 | 94.82 ± 0.67 | 99.33 ± 0.13 |

## calibrated

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 78.82 ± 0.23 | 75.42 ± 0.54 | 79.75 ± 0.26 | 77.35 ± 1.37 | 77.52 ± 1.35 | 95.06 ± 0.67 |
| Energy | 78.72 ± 0.33 | 75.07 ± 0.53 | 79.19 ± 0.41 | 78.50 ± 0.97 | 79.45 ± 2.09 | 95.27 ± 0.74 |
| MaxLogit | 79.12 ± 0.32 | 75.75 ± 0.58 | 79.45 ± 0.40 | 77.07 ± 1.31 | 78.60 ± 1.63 | 95.38 ± 0.68 |
| Entropy | 79.63 ± 0.26 | 76.10 ± 0.51 | 80.30 ± 0.30 | 76.95 ± 0.68 | 78.00 ± 1.44 | 95.26 ± 0.60 |
| KNN-50 | 80.17 ± 0.28 | 76.46 ± 0.55 | 81.42 ± 0.29 | 76.82 ± 0.98 | 78.65 ± 2.75 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 58.81 ± 0.45 | 54.69 ± 0.45 | 62.35 ± 0.45 | 94.49 ± 0.21 | 94.82 ± 0.67 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 42.56 ± 0.57 | 63.65 ± 1.98 | 77.52 ± 1.35 | 94.81 ± 0.61 |
| Energy | 41.56 ± 1.12 | 65.39 ± 2.01 | 79.45 ± 2.09 | 95.36 ± 0.76 |
| MaxLogit | 41.45 ± 0.46 | 64.06 ± 1.78 | 78.60 ± 1.63 | 94.75 ± 1.44 |
| Entropy | 40.92 ± 0.51 | 63.31 ± 1.88 | 78.00 ± 1.44 | 94.61 ± 0.95 |
| KNN-50 | 39.98 ± 0.71 | 62.81 ± 2.37 | 78.65 ± 2.75 | 94.55 ± 0.84 |
| Pooled Mahalanobis | 74.61 ± 0.22 | 88.24 ± 0.47 | 94.82 ± 0.67 | 99.33 ± 0.13 |

## Evidence and scope

Per-image raw logits/features, predictions, source filenames, twelve score vectors and all 80/90/95/99 ID/OOD acceptance masks are retained locally for each seed. The image cache and each seed's feature scores were independently audited against decoded archive pixels and refitted training-ID feature banks. Score audits independently recomputed all formulas, 468 metric fields, 432 masks, and fixed threshold counts across the three seeds. [Completion manifest](completion.json) binds the report, figures, audits and regression evidence.

This one new near-OOD dataset contains potential semantic overlap with CIFAR-100 and a 64→32 source/resolution shift. These measurements do not isolate those factors, establish universal generalization or reproduce an OpenOOD leaderboard. All six methods and both temperature variants are reported in their frozen order; there is no post-test winner selection. Sample SD reflects only three trained classifier seeds, not image-level confidence intervals.

Figures: [AUROC](auroc.png) and [fixed95 OOD acceptance](fixed95-acceptance.png).
