# CIFAR-100 held-out-class heterogeneity — descriptive diagnostic

All 20 frozen held-out fine classes contribute exactly 100 official-test OOD images each. Every class AUROC uses the same complete 8,000 ID-test scores. The six detectors, three checkpoints, temperature fits and ID-only thresholds are inherited unchanged from the completed class-holdout study. This is a post-hoc description of the same test set, not an extra confirmation or method selection.

The following are distributions across **20 class means over three training seeds**: min / Q1 / median / Q3 / max, in percent. Quantiles use linear interpolation. They are not confidence intervals or a measure of training-seed variation.

## T1

| Method | Class AUROC min / Q1 / median / Q3 / max | Fixed95 OOD acceptance min / Q1 / median / Q3 / max |
|---|---:|---:|
| MSP | 61.08 / 69.36 / 73.25 / 79.37 / 85.38 | 72.67 / 82.17 / 87.83 / 92.08 / 95.33 |
| Energy | 57.79 / 68.63 / 73.78 / 78.54 / 87.79 | 70.67 / 80.00 / 87.17 / 93.58 / 98.33 |
| MaxLogit | 58.25 / 68.86 / 73.64 / 78.58 / 87.84 | 70.00 / 79.67 / 87.67 / 92.58 / 98.00 |
| Entropy | 61.71 / 69.15 / 74.16 / 80.17 / 87.37 | 69.00 / 79.83 / 88.33 / 93.00 / 96.67 |
| KNN-50 | 56.17 / 65.18 / 74.68 / 80.25 / 89.16 | 72.33 / 84.67 / 91.33 / 93.42 / 98.00 |
| Pooled Mahalanobis | 37.45 / 45.21 / 52.57 / 58.73 / 66.76 | 90.67 / 96.67 / 97.83 / 98.42 / 100.00 |

## calibrated

| Method | Class AUROC min / Q1 / median / Q3 / max | Fixed95 OOD acceptance min / Q1 / median / Q3 / max |
|---|---:|---:|
| MSP | 61.46 / 69.15 / 74.09 / 79.97 / 86.85 | 70.33 / 80.92 / 88.50 / 92.50 / 96.00 |
| Energy | 57.49 / 68.39 / 73.89 / 78.39 / 87.50 | 71.67 / 79.83 / 86.00 / 94.00 / 97.33 |
| MaxLogit | 58.25 / 68.86 / 73.64 / 78.58 / 87.84 | 70.00 / 79.67 / 87.67 / 92.58 / 98.00 |
| Entropy | 62.02 / 69.20 / 74.07 / 80.14 / 88.26 | 69.33 / 80.17 / 88.00 / 93.25 / 98.67 |
| KNN-50 | 56.17 / 65.18 / 74.68 / 80.25 / 89.16 | 72.33 / 84.67 / 91.33 / 93.42 / 98.00 |
| Pooled Mahalanobis | 37.45 / 45.21 / 52.57 / 58.73 / 66.76 | 90.67 / 96.67 / 97.83 / 98.42 / 100.00 |

## Full class data and limits

[class_auroc.csv](class_auroc.csv) lists all 240 class/method/temperature rows and each seed value. [class_acceptance.csv](class_acceptance.csv) lists all 960 class/method/temperature/ID-target rows, exact accepted counts, training-seed mean/sample SD and actual full ID-test acceptance. [class_values.npz](class_values.npz) retains all 720 class AUROCs, 2,880 class acceptance counts, exact original test rows, fine/coarse IDs and names. The independent [audit](audit.json) checks every value against frozen source scores and masks.

Each class contributes only 100 OOD images; a one-seed class acceptance rate moves in 1-percentage-point steps. Classes, methods and seeds share images and the full ID reference, so the distribution is dependent and cannot be interpreted as a confidence interval. The per-class OOD:ID ratio is 100:8,000, so per-class AUPR is deliberately omitted. No class was selected or labeled by semantic distance, and no per-class result is used to tune a detector. The original aggregate report remains unchanged.
