# Independent complete-seed audit

Training seeds: [0, 1, 2]; 73 statistic records verified.

Raw per-seed metrics, ID-only checkpoint selection, split and source hashes were audited separately. Means and sample SD were independently recomputed with Python statistics, without importing the production aggregation function.

## Paired differences from MSP

Each value is method minus MSP on the same classifier, followed by sample SD across training seeds. Positive AUROC differences and negative FPR95-ID differences favor the listed method. These are descriptive paired differences, not confidence intervals, significance tests, or a test-set method-selection rule.

### textures

| Method | AUROC difference pp | FPR95-ID difference pp |
|---|---:|---:|
| energy | +1.12 ± 0.50 | -2.85 ± 0.48 |
| maxlogit | +1.25 ± 0.43 | -2.30 ± 0.33 |
| entropy | +1.29 ± 0.13 | -2.41 ± 0.72 |
| knn | +8.45 ± 0.93 | -18.99 ± 2.47 |
| mahalanobis | -1.47 ± 1.78 | -9.18 ± 0.41 |

Only all explicitly expected, completed seeds enter this audit. Single-seed diagnostic smoke does not establish training-seed variability. All artifacts remain local.
