# Frozen-threshold failure analysis

Classifier selected epoch: 89. ID test accuracy: 76.11%.

All thresholds are the original ID-validation-calibrated thresholds. No detector is refitted, no test-set threshold is chosen, and ID misclassifications remain ID.

## CIFAR10

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.43 | 13.48 | 78.40 | 82.18 |
| energy | 1.83 | 11.85 | 78.01 | 82.49 |
| maxlogit | 1.67 | 12.35 | 78.14 | 81.93 |
| entropy | 1.55 | 13.31 | 78.35 | 81.08 |
| knn | 1.80 | 13.23 | 78.29 | 81.82 |
| mahalanobis | 4.59 | 5.19 | 76.23 | 96.62 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## MNIST

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.43 | 13.48 | 78.40 | 86.44 |
| energy | 1.83 | 11.85 | 78.01 | 82.32 |
| maxlogit | 1.67 | 12.35 | 78.14 | 83.06 |
| entropy | 1.55 | 13.31 | 78.35 | 84.82 |
| knn | 1.80 | 13.23 | 78.29 | 90.62 |
| mahalanobis | 4.59 | 5.19 | 76.23 | 99.61 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## NOISE

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.43 | 13.48 | 78.40 | 100.00 |
| energy | 1.83 | 11.85 | 78.01 | 100.00 |
| maxlogit | 1.67 | 12.35 | 78.14 | 100.00 |
| entropy | 1.55 | 13.31 | 78.35 | 100.00 |
| knn | 1.80 | 13.23 | 78.29 | 39.82 |
| mahalanobis | 4.59 | 5.19 | 76.23 | 0.00 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## Interpretation limits

Rejection of classification mistakes can improve accuracy on the remaining accepted ID samples, but it does not demonstrate natural OOD detection. OOD acceptance must be assessed separately. This report explains a fixed model and threshold; it is not a method-selection experiment.

![ID rejection by correctness](id-rejection-by-correctness.png)

[Six-method local failure gallery with official test indices](../../runs/cifar100-seed0/failure-gallery/REPORT.md). These are fixed-rule extreme examples, not representative samples.
