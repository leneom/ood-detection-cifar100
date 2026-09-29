# Class-holdout RMD components and paired fixed-threshold decisions

Post-result descriptive diagnostic over every fixed partition/model, all seven references and all four original ID-validation targets. Positive deltas mean RMD accepts more rows than the named reference. Both ID and OOD deltas are required because each method has its own validation threshold; these comparisons do not isolate ranking or establish causality.

| Reference | Target ID validation % | ID acceptance change, pp | OOD acceptance change, pp |
|---|---:|---:|---:|
| msp | 80 | -0.56 ± 0.49 | +0.24 ± 3.03 |
| msp | 90 | +0.46 ± 0.41 | +3.96 ± 2.22 |
| msp | 95 | -0.03 ± 0.63 | +3.08 ± 3.62 |
| msp | 99 | +0.02 ± 0.11 | +1.24 ± 0.92 |
| energy | 80 | -0.43 ± 1.06 | +2.19 ± 3.10 |
| energy | 90 | +0.17 ± 0.61 | +4.32 ± 3.92 |
| energy | 95 | -0.44 ± 0.78 | +2.32 ± 3.66 |
| energy | 99 | -0.20 ± 0.27 | +0.62 ± 1.06 |
| maxlogit | 80 | -0.44 ± 0.85 | +2.78 ± 2.71 |
| maxlogit | 90 | +0.29 ± 0.41 | +4.85 ± 3.80 |
| maxlogit | 95 | -0.32 ± 0.78 | +2.76 ± 3.62 |
| maxlogit | 99 | -0.05 ± 0.35 | +1.42 ± 1.62 |
| entropy | 80 | -0.49 ± 0.53 | +3.44 ± 2.95 |
| entropy | 90 | +0.35 ± 0.52 | +5.81 ± 4.09 |
| entropy | 95 | -0.25 ± 0.80 | +3.33 ± 4.40 |
| entropy | 99 | -0.08 ± 0.43 | +1.13 ± 1.51 |
| knn | 80 | -0.57 ± 0.80 | +0.78 ± 5.22 |
| knn | 90 | -0.07 ± 0.44 | +2.09 ± 4.44 |
| knn | 95 | -0.35 ± 0.68 | +1.24 ± 3.73 |
| knn | 99 | -0.21 ± 0.24 | +0.29 ± 0.88 |
| mahalanobis | 80 | -0.78 ± 0.78 | -30.23 ± 2.89 |
| mahalanobis | 90 | -0.25 ± 0.54 | -17.91 ± 2.97 |
| mahalanobis | 95 | -0.38 ± 0.05 | -10.43 ± 2.18 |
| mahalanobis | 99 | -0.18 ± 0.16 | -2.34 ± 0.77 |
| matched_cpu_md | 80 | -0.78 ± 0.78 | -30.23 ± 2.89 |
| matched_cpu_md | 90 | -0.25 ± 0.54 | -17.91 ± 2.97 |
| matched_cpu_md | 95 | -0.38 ± 0.05 | -10.43 ± 2.18 |
| matched_cpu_md | 99 | -0.18 ± 0.16 | -2.34 ± 0.77 |

Values are equal-weight means of three fixed partition means ± sample SD of partition means, not confidence intervals. Every one of the 504 model/reference/target/population transition rows, complete per-row uint8 transition codes, ID correctness splits, 54 component-distribution rows and per-partition values are saved in transitions.csv, transitions-split*.npz, components.csv and summary.json. RMD is exactly matched class distance minus background distance. Original scores, fit, thresholds and checkpoints are unchanged.
