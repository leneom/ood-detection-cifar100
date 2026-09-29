# Frozen-threshold failure analysis

Classifier selected epoch: 10. ID test accuracy: 52.69%.

All thresholds are the original ID-validation-calibrated thresholds. No detector is refitted, no test-set threshold is chosen, and ID misclassifications remain ID.

## CIFAR10

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.04 | 8.14 | 54.54 | 89.58 |
| energy | 1.48 | 8.12 | 54.42 | 89.62 |
| maxlogit | 1.14 | 8.58 | 54.64 | 89.22 |
| entropy | 1.27 | 7.95 | 54.43 | 88.93 |
| knn | 3.28 | 7.31 | 53.75 | 92.68 |
| mahalanobis | 7.29 | 3.11 | 51.59 | 97.09 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## MNIST

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.04 | 8.14 | 54.54 | 98.26 |
| energy | 1.48 | 8.12 | 54.42 | 95.21 |
| maxlogit | 1.14 | 8.58 | 54.64 | 97.12 |
| entropy | 1.27 | 7.95 | 54.43 | 97.89 |
| knn | 3.28 | 7.31 | 53.75 | 99.34 |
| mahalanobis | 7.29 | 3.11 | 51.59 | 98.18 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## NOISE

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.04 | 8.14 | 54.54 | 100.00 |
| energy | 1.48 | 8.12 | 54.42 | 100.00 |
| maxlogit | 1.14 | 8.58 | 54.64 | 100.00 |
| entropy | 1.27 | 7.95 | 54.43 | 100.00 |
| knn | 3.28 | 7.31 | 53.75 | 99.48 |
| mahalanobis | 7.29 | 3.11 | 51.59 | 0.00 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## Interpretation limits

Rejection of classification mistakes can improve accuracy on the remaining accepted ID samples, but it does not demonstrate natural OOD detection. OOD acceptance must be assessed separately. This report explains a fixed model and threshold; it is not a method-selection experiment.

![ID rejection by correctness](id-rejection-by-correctness.png)
