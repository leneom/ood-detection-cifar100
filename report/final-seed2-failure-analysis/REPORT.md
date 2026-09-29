# Frozen-threshold failure analysis

Classifier selected epoch: 92. ID test accuracy: 75.64%.

All thresholds are the original ID-validation-calibrated thresholds. No detector is refitted, no test-set threshold is chosen, and ID misclassifications remain ID.

## CIFAR10

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.59 | 16.05 | 78.45 | 81.37 |
| energy | 1.90 | 13.63 | 77.91 | 82.03 |
| maxlogit | 1.72 | 14.70 | 78.15 | 81.06 |
| entropy | 1.75 | 14.78 | 78.17 | 80.60 |
| knn | 1.82 | 14.37 | 78.07 | 83.58 |
| mahalanobis | 4.63 | 4.80 | 75.67 | 96.95 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## MNIST

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.59 | 16.05 | 78.45 | 86.39 |
| energy | 1.90 | 13.63 | 77.91 | 83.33 |
| maxlogit | 1.72 | 14.70 | 78.15 | 84.04 |
| entropy | 1.75 | 14.78 | 78.17 | 85.62 |
| knn | 1.82 | 14.37 | 78.07 | 86.11 |
| mahalanobis | 4.63 | 4.80 | 75.67 | 98.45 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## NOISE

| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |
|---|---:|---:|---:|---:|
| msp | 1.59 | 16.05 | 78.45 | 11.91 |
| energy | 1.90 | 13.63 | 77.91 | 0.49 |
| maxlogit | 1.72 | 14.70 | 78.15 | 1.48 |
| entropy | 1.75 | 14.78 | 78.17 | 1.59 |
| knn | 1.82 | 14.37 | 78.07 | 1.24 |
| mahalanobis | 4.63 | 4.80 | 75.67 | 100.00 |

Top accepted-OOD classifier output IDs and representative sample indices are retained in `failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.

## Interpretation limits

Rejection of classification mistakes can improve accuracy on the remaining accepted ID samples, but it does not demonstrate natural OOD detection. OOD acceptance must be assessed separately. This report explains a fixed model and threshold; it is not a method-selection experiment.

![ID rejection by correctness](id-rejection-by-correctness.png)
