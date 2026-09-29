# Class-holdout RMD extension: all eight fixed T=1 columns

Exploratory post-hoc extension on the same nine frozen models and official test populations. RMD and matched CPU MD were fitted on 36,000 optimization-ID features only with the previously fixed 1% covariance shrinkage; all nine fits and scores passed independent audits. The six original T=1 methods are unchanged controls. Values are equal-weight means over three fixed partition means ± sample SD of those partition means, not confidence intervals.

| Method | AUROC % | AUPR-out % | AUPR-in % | FPR95-ID % | ID accepted at fixed95 % | OOD accepted at fixed95 % |
|---|---:|---:|---:|---:|---:|---:|
| msp | 75.46 ± 1.88 | 38.79 ± 3.03 | 92.55 ± 0.65 | 83.03 ± 2.87 | 94.91 ± 0.61 | 82.72 ± 4.77 |
| energy | 74.92 ± 1.50 | 39.11 ± 2.30 | 92.00 ± 0.52 | 82.71 ± 2.13 | 95.32 ± 0.73 | 83.48 ± 4.41 |
| maxlogit | 75.15 ± 1.56 | 39.43 ± 2.60 | 92.07 ± 0.53 | 82.26 ± 2.43 | 95.20 ± 0.72 | 83.05 ± 4.39 |
| entropy | 76.13 ± 1.86 | 39.92 ± 2.96 | 92.71 ± 0.64 | 82.21 ± 2.96 | 95.13 ± 0.72 | 82.48 ± 5.34 |
| knn | 74.49 ± 2.08 | 38.07 ± 3.32 | 91.93 ± 0.50 | 83.91 ± 2.90 | 95.23 ± 0.82 | 84.57 ± 4.87 |
| mahalanobis | 53.16 ± 2.35 | 20.49 ± 0.80 | 83.34 ± 1.37 | 95.97 ± 0.50 | 95.26 ± 0.41 | 96.24 ± 0.73 |
| rmd | 74.29 ± 1.65 | 36.30 ± 2.04 | 92.04 ± 0.54 | 85.99 ± 1.81 | 94.88 ± 0.37 | 85.81 ± 2.75 |
| matched_cpu_md | 53.16 ± 2.35 | 20.49 ± 0.80 | 83.34 ± 1.37 | 95.97 ± 0.50 | 95.26 ± 0.41 | 96.24 ± 0.73 |

The fixed95 operational threshold is the separate 1,200-row ID-validation order statistic for each model and method. FPR95-ID instead uses an ID **test** threshold and is a separate benchmark operating point. The full nine-model points and per-partition means/SDs are in per-model.csv and summary.json. RMD is not promoted as a selected winner; prior OOD exposure and shared test populations preclude a new confirmation claim.
