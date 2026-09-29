# Complete local OOD baseline

Three fixed 100-epoch CIFAR-100 classifiers; ID test accuracy 75.77 ± 0.30%.

All six methods across four natural OOD populations are complete. Phase audits checked 291 statistic records and all underlying raw metrics. The Textures phase additionally independently reconstructed 12 executed ID/OOD input hashes. Previous phases and synthetic diagnostics are preserved.

| Natural OOD | ID count | OOD count | OOD prevalence |
|---|---:|---:|---:|
| cifar10 | 10000 | 10000 | 50.00% |
| mnist | 10000 | 10000 | 50.00% |
| svhn | 10000 | 26032 | 72.25% |
| textures | 10000 | 1880 | 15.82% |

## cifar10

| Method | AUROC % | FPR95-ID % |
|---|---:|---:|
| msp | 76.86 ± 0.10 | 81.09 ± 1.04 |
| energy | 77.75 ± 0.13 | 81.46 ± 1.60 |
| maxlogit | 77.95 ± 0.12 | 80.84 ± 1.44 |
| entropy | 77.80 ± 0.13 | 80.30 ± 1.73 |
| knn | 75.82 ± 0.58 | 81.75 ± 1.56 |
| mahalanobis | 54.02 ± 0.99 | 96.33 ± 0.33 |

## mnist

| Method | AUROC % | FPR95-ID % |
|---|---:|---:|
| msp | 75.84 ± 2.69 | 85.66 ± 1.21 |
| energy | 80.02 ± 1.83 | 85.25 ± 7.25 |
| maxlogit | 79.87 ± 1.47 | 85.20 ± 6.00 |
| entropy | 77.54 ± 2.08 | 86.15 ± 4.41 |
| knn | 79.80 ± 2.51 | 85.96 ± 3.00 |
| mahalanobis | 59.24 ± 10.21 | 97.29 ± 2.76 |

## svhn

| Method | AUROC % | FPR95-ID % |
|---|---:|---:|
| msp | 82.58 ± 2.68 | 72.48 ± 6.88 |
| energy | 87.38 ± 2.51 | 68.68 ± 9.32 |
| maxlogit | 87.12 ± 2.49 | 68.21 ± 8.42 |
| entropy | 84.96 ± 2.78 | 67.71 ± 8.31 |
| knn | 87.99 ± 2.08 | 59.13 ± 8.27 |
| mahalanobis | 63.33 ± 1.96 | 94.09 ± 1.45 |

## textures

| Method | AUROC % | FPR95-ID % |
|---|---:|---:|
| msp | 77.62 ± 0.51 | 81.12 ± 0.49 |
| energy | 78.75 ± 0.85 | 78.26 ± 0.75 |
| maxlogit | 78.87 ± 0.81 | 78.81 ± 0.80 |
| entropy | 78.91 ± 0.60 | 78.71 ± 1.13 |
| knn | 86.07 ± 0.51 | 62.13 ± 2.53 |
| mahalanobis | 76.15 ± 1.42 | 71.93 ± 0.78 |

## equal-weight natural macro

| Method | AUROC % | FPR95-ID % |
|---|---:|---:|
| msp | 78.23 ± 1.32 | 80.09 ± 1.65 |
| energy | 80.97 ± 0.79 | 78.41 ± 2.23 |
| maxlogit | 80.95 ± 0.83 | 78.27 ± 1.71 |
| entropy | 79.80 ± 1.23 | 78.22 ± 1.78 |
| knn | 82.42 ± 1.11 | 72.24 ± 2.34 |
| mahalanobis | 63.19 ± 2.20 | 89.91 ± 0.32 |

## Interpretation

Each ± is sample SD across training seeds, not a confidence interval. Macro first averages the four natural datasets equally within each seed, then summarizes those three paired values. Noise and constant images are excluded. No samples from different OOD populations are pooled.

AUPR retains its population-specific counts/prevalence in the original reports; dataset prevalence differs, so AP values across these populations have different reference levels. FPR95-ID uses the benchmark ID-test threshold; validation-only operating points remain separately saved in each raw result.

This is an independent local CIFAR protocol, not full OpenOOD reproduction or a leaderboard claim. All checkpoints use ID validation selection and all feature fits use the training ID split. Temperature, k, shrinkage and score directions were not changed from OOD tests.
