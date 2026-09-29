# Oxford-IIIT Pet full breed/species fixed-threshold diagnostic

Post-result descriptive analysis of all 3,669 official test images. The two official species codes are kept numeric. No breed or image is filtered, and no method or threshold is selected from these OOD outcomes.

For each frozen method below, values are three training-seed means ± sample SD (%) at the existing 95% ID-validation target. Breed min/median/max are quantiles of 37 unweighted breed-level three-seed mean rates; breed denominators differ, so these are not the overall Pet rate.

| Variant | Method | All Pet | Species 1 | Species 2 | Breed min / median / max |
| --- | --- | ---: | ---: | ---: | ---: |
| T1 | msp | 76.45 ± 0.61 | 75.15 ± 0.37 | 77.07 ± 0.92 | 64.00 / 76.33 / 87.00 |
| T1 | energy | 76.76 ± 2.01 | 75.18 ± 1.91 | 77.51 ± 2.07 | 64.67 / 76.33 / 87.67 |
| T1 | maxlogit | 76.39 ± 1.10 | 74.70 ± 1.13 | 77.19 ± 1.16 | 63.33 / 76.33 / 86.67 |
| T1 | entropy | 75.59 ± 0.93 | 73.74 ± 0.76 | 76.47 ± 1.01 | 62.67 / 76.33 / 87.33 |
| T1 | knn | 79.52 ± 1.82 | 79.85 ± 1.52 | 79.36 ± 1.99 | 61.67 / 79.67 / 92.00 |
| T1 | mahalanobis | 95.88 ± 0.49 | 97.13 ± 0.45 | 95.28 ± 0.57 | 88.33 / 96.63 / 99.67 |
| calibrated | msp | 75.07 ± 1.05 | 73.18 ± 1.17 | 75.97 ± 1.27 | 60.67 / 75.00 / 87.33 |
| calibrated | energy | 77.89 ± 2.00 | 77.01 ± 2.12 | 78.31 ± 1.96 | 67.00 / 78.35 / 88.00 |
| calibrated | maxlogit | 76.39 ± 1.10 | 74.70 ± 1.13 | 77.19 ± 1.16 | 63.33 / 76.33 / 86.67 |
| calibrated | entropy | 75.90 ± 0.89 | 73.85 ± 0.68 | 76.87 ± 1.04 | 61.67 / 76.33 / 87.33 |
| calibrated | knn | 79.52 ± 1.82 | 79.85 ± 1.52 | 79.36 ± 1.99 | 61.67 / 79.67 / 92.00 |
| calibrated | mahalanobis | 95.88 ± 0.49 | 97.13 ± 0.45 | 95.28 ± 0.57 | 88.33 / 96.63 / 99.67 |

The [complete group CSV](group_counts.csv) retains all 37 breeds, both species codes, both variants, six methods, four frozen targets and three seed counts. The [summary](summary.json) retains all 48 setting distributions; the [group arrays](group_values.npz) retain counts, denominators and complete row partitions. Different breed/source/resizing characteristics are confounded. The three trained seeds are not a population-level confidence interval, and these previously viewed OOD examples cannot validate a new detector.
