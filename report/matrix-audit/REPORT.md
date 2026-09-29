# Independent complete-seed audit

Training seeds: [0, 1, 2]; 145 statistic records verified.

Raw per-seed metrics, ID-only checkpoint selection, split and source hashes were audited separately. Means and sample SD were independently recomputed with Python statistics, without importing the production aggregation function.

## Paired differences from MSP

Each value is method minus MSP on the same classifier, followed by sample SD across training seeds. Positive AUROC differences and negative FPR95-ID differences favor the listed method. These are descriptive paired differences, not confidence intervals, significance tests, or a test-set method-selection rule.

### cifar10

| Method | AUROC difference pp | FPR95-ID difference pp |
|---|---:|---:|
| energy | +0.89 ± 0.08 | +0.37 ± 0.79 |
| maxlogit | +1.09 ± 0.08 | -0.25 ± 0.79 |
| entropy | +0.94 ± 0.06 | -0.80 ± 0.86 |
| knn | -1.03 ± 0.56 | +0.66 ± 0.90 |
| mahalanobis | -22.84 ± 0.98 | +15.23 ± 1.15 |

### mnist

| Method | AUROC difference pp | FPR95-ID difference pp |
|---|---:|---:|
| energy | +4.18 ± 3.59 | -0.41 ± 6.66 |
| maxlogit | +4.03 ± 2.96 | -0.45 ± 5.36 |
| entropy | +1.70 ± 0.69 | +0.49 ± 3.68 |
| knn | +3.96 ± 0.43 | +0.30 ± 4.03 |
| mahalanobis | -16.61 ± 8.58 | +11.63 ± 3.59 |

### noise

| Method | AUROC difference pp | FPR95-ID difference pp |
|---|---:|---:|
| energy | +3.82 ± 5.76 | -5.97 ± 5.97 |
| maxlogit | +3.58 ± 5.00 | -5.57 ± 5.55 |
| entropy | +1.21 ± 0.73 | -5.62 ± 5.39 |
| knn | +13.02 ± 21.19 | -27.68 ± 32.39 |
| mahalanobis | +1.82 ± 34.69 | +26.85 ± 109.88 |

Only all explicitly expected, completed seeds enter this audit. Single-seed diagnostic smoke does not establish training-seed variability. All artifacts remain local.
