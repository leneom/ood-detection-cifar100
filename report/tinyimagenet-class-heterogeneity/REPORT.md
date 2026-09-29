# Tiny ImageNet class-heterogeneity diagnostic

Post-test descriptive stratification of the already completed Tiny ImageNet evaluation. The 200 official synsets each contribute exactly 50 OOD images; each class AUROC compares those 50 against the same complete 10,000 CIFAR-100 ID-test scores. All classifiers, methods, temperature values and operating thresholds remain frozen. This is not an additional held-out confirmation or method-selection exercise.

The tables show the distribution across **200 classes of each class’s three-seed mean**. Values are percent; columns are min, 25th percentile, median, 75th percentile and max (linear percentile convention). These are neither confidence intervals nor train-seed variability.

## T1

| Method | Class AUROC min / Q1 / median / Q3 / max | Fixed95 OOD acceptance min / Q1 / median / Q3 / max |
| --- | ---: | ---: |
| MSP | 55.18 / 75.11 / 79.20 / 82.34 / 87.10 | 58.00 / 74.00 / 78.67 / 84.00 / 95.33 |
| Energy | 54.05 / 75.75 / 80.23 / 84.28 / 90.74 | 52.67 / 73.83 / 80.00 / 85.33 / 96.67 |
| MaxLogit | 54.59 / 75.92 / 80.60 / 84.41 / 90.34 | 54.67 / 73.33 / 79.33 / 84.67 / 96.00 |
| Entropy | 56.09 / 75.97 / 80.46 / 83.96 / 88.98 | 54.00 / 72.00 / 78.00 / 84.00 / 94.67 |
| KNN-50 | 50.33 / 77.00 / 81.89 / 86.16 / 91.28 | 56.00 / 72.50 / 79.67 / 85.33 / 96.67 |
| Pooled Mahalanobis | 37.34 / 55.14 / 58.54 / 62.75 / 76.99 | 76.00 / 93.33 / 96.00 / 97.33 / 100.00 |

## calibrated

| Method | Class AUROC min / Q1 / median / Q3 / max | Fixed95 OOD acceptance min / Q1 / median / Q3 / max |
| --- | ---: | ---: |
| MSP | 56.03 / 75.66 / 80.12 / 83.36 / 88.33 | 56.00 / 72.00 / 78.00 / 83.50 / 96.00 |
| Energy | 53.69 / 75.58 / 79.93 / 84.14 / 90.74 | 53.33 / 74.67 / 80.00 / 85.33 / 96.67 |
| MaxLogit | 54.59 / 75.92 / 80.60 / 84.41 / 90.34 | 54.67 / 73.33 / 79.33 / 84.67 / 96.00 |
| Entropy | 57.26 / 76.21 / 80.85 / 84.55 / 90.26 | 52.67 / 72.67 / 78.67 / 84.67 / 95.33 |
| KNN-50 | 50.33 / 77.00 / 81.89 / 86.16 / 91.28 | 56.00 / 72.50 / 79.67 / 85.33 / 96.67 |
| Pooled Mahalanobis | 37.34 / 55.14 / 58.54 / 62.75 / 76.99 | 76.00 / 93.33 / 96.00 / 97.33 / 100.00 |

## Interpretation and data

All class IDs and exact 50 source rows, 7,200 class AUROCs, and 28,800 class operating counts are in [class_values.npz](class_values.npz). Each of the 200 equal-size class AUROCs averages exactly to the corresponding original all-image AUROC; class accepted counts sum to each original all-image OOD count. Full independent checks are in [audit.json](audit.json).

A single-seed 50-image class acceptance rate moves in 2-percentage-point steps; the shown three-seed mean moves in two-thirds-of-a-point steps. Shared ID scores and shared images make these class and method values dependent. No synset was hand-labeled “near”, “far”, “hard” or “easy”; class extremes have no selection role. AUPR is omitted for class comparisons because the 50:10,000 OOD:ID prevalence differs sharply from the original 10,000:10,000 evaluation. This diagnostic cannot separate semantic overlap from image-source and 64→32 resolution effects.

Figures: [class AUROC distributions](class-auroc.png) and [fixed95 OOD acceptance distributions](class-fixed95-acceptance.png).
