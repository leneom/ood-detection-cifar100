# Fashion-MNIST frozen population extension

All 10,000 official Fashion-MNIST test images were scored against the same 10,000 CIFAR-100 ID-test images. This population was selected sequentially after viewing the Tiny ImageNet results, then its full protocol was frozen before any Fashion image access. Three isolated classifiers (selected epochs 92/96/98), their ID-only temperatures and thresholds, and all six original methods were reused. No Fashion image was used to select a model, fit a detector or choose a threshold.

Each table value is the mean ± sample SD across three training seeds, in percent. Individual seed points and exact threshold counts are in [summary.json](summary.json). AUROC and AUPR use 10,000 ID + 10,000 OOD samples, so OOD prevalence is 50%. FPR95-ID uses the ID **test** order statistic as a benchmark operating point; fixed95 OOD/ID acceptance uses the pre-existing 1,500-row ID-validation threshold.

## T1

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 86.10 ± 0.31 | 83.15 ± 0.71 | 88.47 ± 0.29 | 65.18 ± 1.88 | 66.17 ± 2.01 | 95.20 ± 0.51 |
| Energy | 90.42 ± 0.16 | 87.21 ± 0.30 | 92.28 ± 0.09 | 57.13 ± 0.64 | 58.76 ± 5.65 | 95.18 ± 0.75 |
| MaxLogit | 90.13 ± 0.08 | 87.01 ± 0.33 | 92.04 ± 0.03 | 57.94 ± 0.41 | 60.22 ± 4.23 | 95.38 ± 0.68 |
| Entropy | 88.49 ± 0.22 | 85.83 ± 0.51 | 90.21 ± 0.23 | 59.32 ± 1.33 | 60.71 ± 3.67 | 95.25 ± 0.70 |
| KNN-50 | 90.48 ± 0.44 | 88.48 ± 0.60 | 92.05 ± 0.34 | 51.28 ± 2.28 | 54.00 ± 1.20 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 60.51 ± 2.72 | 53.31 ± 2.59 | 67.49 ± 1.65 | 97.51 ± 1.05 | 97.72 ± 0.78 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 28.90 ± 1.47 | 49.09 ± 1.59 | 66.17 ± 2.01 | 89.37 ± 2.05 |
| Energy | 14.45 ± 1.74 | 37.51 ± 2.64 | 58.76 ± 5.65 | 89.04 ± 2.87 |
| MaxLogit | 15.69 ± 1.39 | 39.37 ± 2.49 | 60.22 ± 4.23 | 88.47 ± 3.27 |
| Entropy | 20.27 ± 0.62 | 40.73 ± 3.32 | 60.71 ± 3.67 | 87.60 ± 3.28 |
| KNN-50 | 16.78 ± 0.52 | 36.24 ± 0.26 | 54.00 ± 1.20 | 81.83 ± 1.45 |
| Pooled Mahalanobis | 78.32 ± 5.90 | 92.52 ± 3.04 | 97.72 ± 0.78 | 99.91 ± 0.08 |

## calibrated

| Method | AUROC | AUPR-out | AUPR-in | FPR95-ID | Fixed95 OOD accepted | Fixed95 ID-test accepted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 87.70 ± 0.26 | 84.81 ± 0.65 | 89.75 ± 0.24 | 62.29 ± 1.93 | 62.57 ± 2.60 | 95.06 ± 0.67 |
| Energy | 90.43 ± 0.24 | 87.11 ± 0.39 | 92.33 ± 0.14 | 57.72 ± 0.79 | 59.50 ± 5.79 | 95.27 ± 0.74 |
| MaxLogit | 90.13 ± 0.08 | 87.01 ± 0.33 | 92.04 ± 0.03 | 57.94 ± 0.41 | 60.22 ± 4.23 | 95.38 ± 0.68 |
| Entropy | 89.74 ± 0.13 | 86.83 ± 0.36 | 91.44 ± 0.15 | 57.46 ± 0.73 | 59.00 ± 3.59 | 95.26 ± 0.60 |
| KNN-50 | 90.48 ± 0.44 | 88.48 ± 0.60 | 92.05 ± 0.34 | 51.28 ± 2.28 | 54.00 ± 1.20 | 95.53 ± 0.72 |
| Pooled Mahalanobis | 60.51 ± 2.72 | 53.31 ± 2.59 | 67.49 ± 1.65 | 97.51 ± 1.05 | 97.72 ± 0.78 | 95.26 ± 0.40 |

Fixed ID-validation operating targets (OOD acceptance, mean ± sample SD, %):

| Method | 80% | 90% | 95% | 99% |
| --- | ---: | ---: | ---: | ---: |
| MSP | 22.91 ± 0.87 | 44.63 ± 2.97 | 62.57 ± 2.60 | 89.35 ± 2.32 |
| Energy | 13.94 ± 1.40 | 37.90 ± 3.57 | 59.50 ± 5.79 | 89.51 ± 2.18 |
| MaxLogit | 15.69 ± 1.39 | 39.37 ± 2.49 | 60.22 ± 4.23 | 88.47 ± 3.27 |
| Entropy | 16.84 ± 0.92 | 39.03 ± 2.58 | 59.00 ± 3.59 | 87.73 ± 3.76 |
| KNN-50 | 16.78 ± 0.52 | 36.24 ± 0.26 | 54.00 ± 1.20 | 81.83 ± 1.45 |
| Pooled Mahalanobis | 78.32 ± 5.90 | 92.52 ± 3.04 | 97.72 ± 0.78 | 99.91 ± 0.08 |

## Evidence and scope

Per-image raw logits/features, predictions, official labels and rows, twelve score vectors and all 80/90/95/99 ID/OOD acceptance masks are retained locally for each seed. The image cache and feature scores were independently audited against all decoded IDX pixels and refitted training-ID feature banks. Score audits independently recomputed all formulas, 468 metric fields, 432 masks and fixed threshold counts across three seeds. [Completion manifest](completion.json) binds the report, figures, audits and regression evidence.

Fashion-MNIST is grayscale 28×28 product imagery converted to 32×32 RGB. Its source, color and resolution differ sharply from CIFAR-100, so these measurements cannot isolate semantic novelty, establish universal generalization or reproduce an OpenOOD leaderboard. This sequential extension is not added to the previously fixed four-population macro. All six methods and both temperature variants remain in frozen order, with no post-test winner selection. Sample SD reflects three trained classifier seeds, not image-level confidence intervals.

Figures: [AUROC](auroc.png) and [fixed95 OOD acceptance](fixed95-acceptance.png).
