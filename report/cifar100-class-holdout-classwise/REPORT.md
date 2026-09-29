# Fixed95 classwise acceptance across all CIFAR-100 fine classes

Post-hoc descriptive diagnostic on the completed nine-model fixed-partition study. Every original fine class has 100 official test images per partition. The threshold for each model and score was fitted only on its separate 1,200-row ID validation role. The table includes all six fixed methods, both temperature variants and all three partitions; no class or detector was selected by its OOD outcome.

ID mean is the average test acceptance across the 80 ID classes and three model seeds. OOD min / median / max are the distribution across all 20 held-out fine classes of each class's three-seed mean. These are descriptive class ranges, not uncertainty intervals. All 10,800 exact model/score/class counts, class names, ID correctness and accepted-error counts are in `classwise.csv`; all 72 full quantile groups are in `summary.json`. The original whole-population report remains the benchmark source.

| Partition | Variant | Method | ID mean accepted % | OOD class min % | OOD class median % | OOD class max % |
|---:|---|---|---:|---:|---:|---:|
| 2031 | T1 | msp | 95.43 | 72.67 | 87.83 | 95.33 |
| 2031 | T1 | energy | 95.87 | 70.67 | 87.17 | 98.33 |
| 2031 | T1 | maxlogit | 95.72 | 70.00 | 87.67 | 98.00 |
| 2031 | T1 | entropy | 95.61 | 69.00 | 88.33 | 96.67 |
| 2031 | T1 | knn | 96.11 | 72.33 | 91.33 | 98.00 |
| 2031 | T1 | mahalanobis | 95.67 | 90.67 | 97.83 | 100.00 |
| 2031 | calibrated | msp | 95.53 | 70.33 | 88.50 | 96.00 |
| 2031 | calibrated | energy | 95.75 | 71.67 | 86.00 | 97.33 |
| 2031 | calibrated | maxlogit | 95.72 | 70.00 | 87.67 | 98.00 |
| 2031 | calibrated | entropy | 95.74 | 69.33 | 88.00 | 98.67 |
| 2031 | calibrated | knn | 96.11 | 72.33 | 91.33 | 98.00 |
| 2031 | calibrated | mahalanobis | 95.67 | 90.67 | 97.83 | 100.00 |
| 2032 | T1 | msp | 95.07 | 75.00 | 82.83 | 96.33 |
| 2032 | T1 | energy | 95.60 | 71.00 | 86.67 | 96.67 |
| 2032 | T1 | maxlogit | 95.51 | 70.67 | 86.00 | 96.67 |
| 2032 | T1 | entropy | 95.48 | 73.33 | 86.17 | 96.33 |
| 2032 | T1 | knn | 95.09 | 72.67 | 86.00 | 97.00 |
| 2032 | T1 | mahalanobis | 94.85 | 86.33 | 96.50 | 100.00 |
| 2032 | calibrated | msp | 95.49 | 75.33 | 85.17 | 96.67 |
| 2032 | calibrated | energy | 95.52 | 70.00 | 87.33 | 96.67 |
| 2032 | calibrated | maxlogit | 95.51 | 70.67 | 86.00 | 96.67 |
| 2032 | calibrated | entropy | 95.52 | 71.67 | 85.83 | 97.33 |
| 2032 | calibrated | knn | 95.09 | 72.67 | 86.00 | 97.00 |
| 2032 | calibrated | mahalanobis | 94.85 | 86.33 | 96.50 | 100.00 |
| 2033 | T1 | msp | 94.24 | 65.00 | 77.33 | 99.00 |
| 2033 | T1 | energy | 94.48 | 61.67 | 81.67 | 98.33 |
| 2033 | T1 | maxlogit | 94.38 | 61.33 | 81.33 | 98.33 |
| 2033 | T1 | entropy | 94.30 | 60.33 | 77.67 | 98.67 |
| 2033 | T1 | knn | 94.49 | 61.33 | 80.67 | 98.67 |
| 2033 | T1 | mahalanobis | 95.25 | 92.67 | 96.33 | 100.00 |
| 2033 | calibrated | msp | 94.16 | 62.00 | 77.67 | 99.00 |
| 2033 | calibrated | energy | 94.56 | 63.67 | 81.67 | 98.33 |
| 2033 | calibrated | maxlogit | 94.38 | 61.33 | 81.33 | 98.33 |
| 2033 | calibrated | entropy | 94.40 | 61.67 | 82.00 | 98.00 |
| 2033 | calibrated | knn | 94.49 | 61.33 | 80.67 | 98.67 |
| 2033 | calibrated | mahalanobis | 95.25 | 92.67 | 96.33 | 100.00 |

All displayed counts and quantiles were independently reconstructed from saved scores, validation thresholds, original test fine labels and acceptance masks; see `audit.json`. Class partition 2031 and the aggregate OOD outcomes were already viewed before this diagnostic, so these ranges are exploratory and make no fresh-confirmation or causal claim.
