# Fixed three-partition CIFAR-100 class-holdout sensitivity

Descriptive only: class seed 2031 was already studied before the additional two partitions were declared. Each partition uses three independently trained 100-epoch models, all 8,000 ID and 2,000 held-out-class official test images. The displayed ± is the sample SD of three **partition means**, not a confidence interval; each partition's three training-seed values and within-partition SD are in the CSV and JSON.

Classifier ID-test accuracy (%): 76.59 ± 1.04. ID-fitted temperature: 1.3802 ± 0.0248 across partition means.

## T=1

| Method | AUROC % | AUPR-out % | AUPR-in % | FPR95-ID % | ID accepted at fixed95 % | OOD accepted at fixed95 % |
|---|---:|---:|---:|---:|---:|---:|
| msp | 75.46 ± 1.88 | 38.79 ± 3.03 | 92.55 ± 0.65 | 83.03 ± 2.87 | 94.91 ± 0.61 | 82.72 ± 4.77 |
| energy | 74.92 ± 1.50 | 39.11 ± 2.30 | 92.00 ± 0.52 | 82.71 ± 2.13 | 95.32 ± 0.73 | 83.48 ± 4.41 |
| maxlogit | 75.15 ± 1.56 | 39.43 ± 2.60 | 92.07 ± 0.53 | 82.26 ± 2.43 | 95.20 ± 0.72 | 83.05 ± 4.39 |
| entropy | 76.13 ± 1.86 | 39.92 ± 2.96 | 92.71 ± 0.64 | 82.21 ± 2.96 | 95.13 ± 0.72 | 82.48 ± 5.34 |
| knn | 74.49 ± 2.08 | 38.07 ± 3.32 | 91.93 ± 0.50 | 83.91 ± 2.90 | 95.23 ± 0.82 | 84.57 ± 4.87 |
| mahalanobis | 53.16 ± 2.35 | 20.49 ± 0.80 | 83.34 ± 1.37 | 95.97 ± 0.50 | 95.26 ± 0.41 | 96.24 ± 0.73 |

## ID-only fitted temperature

| Method | AUROC % | AUPR-out % | AUPR-in % | FPR95-ID % | ID accepted at fixed95 % | OOD accepted at fixed95 % |
|---|---:|---:|---:|---:|---:|---:|
| msp | 75.92 ± 1.87 | 39.57 ± 3.07 | 92.65 ± 0.64 | 82.83 ± 2.85 | 95.06 ± 0.78 | 82.77 ± 5.81 |
| energy | 74.65 ± 1.42 | 38.77 ± 2.11 | 91.92 ± 0.50 | 82.97 ± 1.67 | 95.28 ± 0.63 | 83.74 ± 3.99 |
| maxlogit | 75.15 ± 1.56 | 39.43 ± 2.60 | 92.07 ± 0.53 | 82.26 ± 2.43 | 95.20 ± 0.72 | 83.05 ± 4.39 |
| entropy | 76.20 ± 1.74 | 39.98 ± 2.68 | 92.71 ± 0.60 | 82.31 ± 2.38 | 95.22 ± 0.71 | 83.13 ± 4.61 |
| knn | 74.49 ± 2.08 | 38.07 ± 3.32 | 91.93 ± 0.50 | 83.91 ± 2.90 | 95.23 ± 0.82 | 84.57 ± 4.87 |
| mahalanobis | 53.16 ± 2.35 | 20.49 ± 0.80 | 83.34 ± 1.37 | 95.97 ± 0.50 | 95.26 ± 0.41 | 96.24 ± 0.73 |

All four ID-validation threshold targets (80/90/95/99), achieved ID-test coverage and per-partition points are in `operating.csv` and `summary.json`. The complete raw logits, features, scores, thresholds and masks remain in the nine source runs. No method, partition, checkpoint or operating point was selected by OOD test performance.

These three fixed partition retrainings do not isolate a causal class effect or estimate the distribution over all possible partitions.
