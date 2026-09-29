# Independent complete-seed audit

Training seeds: [0, 1, 2]; 73 statistic records verified.

Raw per-seed metrics, ID-only checkpoint selection, split and source hashes were audited separately. Means and sample SD were independently recomputed with Python statistics, without importing the production aggregation function.

## Paired differences from MSP

Each value is method minus MSP on the same classifier, followed by sample SD across training seeds. Positive AUROC differences and negative FPR95-ID differences favor the listed method. These are descriptive paired differences, not confidence intervals, significance tests, or a test-set method-selection rule.

### svhn

| Method | AUROC difference pp | FPR95-ID difference pp |
|---|---:|---:|
| energy | +4.80 ± 0.82 | -3.81 ± 3.80 |
| maxlogit | +4.53 ± 0.68 | -4.27 ± 2.27 |
| entropy | +2.38 ± 0.18 | -4.77 ± 1.77 |
| knn | +5.40 ± 0.73 | -13.35 ± 1.74 |
| mahalanobis | -19.25 ± 4.45 | +21.61 ± 8.32 |

Only all explicitly expected, completed seeds enter this audit. Single-seed diagnostic smoke does not establish training-seed variability. All artifacts remain local.
