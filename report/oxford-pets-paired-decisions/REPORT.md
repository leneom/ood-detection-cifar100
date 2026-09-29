# Oxford-IIIT Pet paired fixed-threshold decisions

Post-result descriptive diagnostic of the complete 10,000 ID-test and 3,669 Pet-test images. Both methods in each pair use their own unchanged ID-validation threshold. A difference in acceptance is therefore not a causal attribution to ranking. No pair was selected to alter a detector.

Each value below is a three-training-seed mean ± sample SD in percentage points at the frozen 95% ID-validation target. Positive first-minus-second means the first method accepts more samples. All 720 tables, both variants and all four targets are retained in [tables.csv](tables.csv); complete row-level codes are in six NPZ files.

| T=1 first | Second | ID first−second | Pet first−second | Pet first only | Pet second only |
| --- | --- | ---: | ---: | ---: | ---: |
| msp | energy | 0.02 ± 0.33 | -0.31 ± 1.62 | 7.40 ± 0.96 | 7.71 ± 0.67 |
| msp | maxlogit | -0.17 ± 0.18 | 0.06 ± 0.88 | 4.95 ± 0.83 | 4.89 ± 0.36 |
| msp | entropy | -0.05 ± 0.21 | 0.86 ± 0.41 | 5.06 ± 0.47 | 4.20 ± 0.33 |
| msp | knn | -0.33 ± 0.49 | -3.07 ± 1.21 | 8.94 ± 0.40 | 12.01 ± 0.85 |
| msp | mahalanobis | -0.06 ± 0.65 | -19.42 ± 0.55 | 3.88 ± 0.40 | 23.30 ± 0.57 |
| energy | maxlogit | -0.20 ± 0.20 | 0.37 ± 0.95 | 2.93 ± 0.34 | 2.55 ± 0.70 |
| energy | entropy | -0.07 ± 0.16 | 1.17 ± 1.21 | 4.36 ± 0.62 | 3.19 ± 0.59 |
| energy | knn | -0.35 ± 0.28 | -2.76 ± 1.42 | 9.57 ± 0.21 | 12.33 ± 1.47 |
| energy | mahalanobis | -0.08 ± 0.97 | -19.12 ± 1.54 | 4.11 ± 0.49 | 23.22 ± 2.02 |
| maxlogit | entropy | 0.13 ± 0.04 | 0.80 ± 0.61 | 2.13 ± 0.16 | 1.33 ± 0.55 |
| maxlogit | knn | -0.15 ± 0.44 | -3.13 ± 1.46 | 8.74 ± 0.42 | 11.87 ± 1.38 |
| maxlogit | mahalanobis | 0.12 ± 0.82 | -19.49 ± 0.61 | 4.07 ± 0.44 | 23.56 ± 1.05 |
| entropy | knn | -0.28 ± 0.41 | -3.93 ± 1.00 | 8.03 ± 0.45 | 11.97 ± 0.90 |
| entropy | mahalanobis | -0.01 ± 0.86 | -20.29 ± 0.63 | 4.02 ± 0.43 | 24.31 ± 0.88 |
| knn | mahalanobis | 0.27 ± 1.03 | -16.35 ± 1.62 | 2.61 ± 0.28 | 18.96 ± 1.60 |

The Entropy/KNN-50 row was motivated by the already viewed aggregate results: KNN-50 had higher Pet AUROC yet higher Pet acceptance at fixed95. Its discordant per-image cells show which accepted decisions differ; this is not a new winner-selection comparison. ID-test acceptance is shown alongside Pet acceptance because the actual coverages can differ. Three trained seeds and one sequentially chosen OOD population do not provide a confidence interval or universal generalization claim.
