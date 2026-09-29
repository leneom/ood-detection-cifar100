# OOD Detection: Full 100-epoch schedule, seed 2

ID: cifar100; selected checkpoint epoch 92; ID test accuracy **75.64%**.

All detector scores increase toward OOD. Checkpoint selection uses ID validation accuracy only.

This is an independently trained CIFAR study, not a full OpenOOD reproduction. A pilot or one training seed does not support a multiseed conclusion.

## CIFAR10

near-OOD natural images; official test split.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 76.81 [76.22, 77.48] | 72.52 | 81.61 | 94.89 | 81.37 |
| Energy | 77.78 [77.16, 78.45] | 73.15 | 81.48 | 95.24 | 82.03 |
| MaxLogit | 77.99 [77.37, 78.67] | 73.47 | 80.65 | 95.12 | 81.06 |
| Entropy | 77.69 [77.10, 78.34] | 73.48 | 80.43 | 95.08 | 80.60 |
| KNN | 75.16 [74.54, 75.86] | 71.31 | 83.30 | 95.12 | 83.58 |
| Mahalanobis | 52.90 [52.16, 53.67] | 50.12 | 96.63 | 95.33 | 96.95 |

Energy − MSP AUROC difference: 0.97 percentage points (paired sample-bootstrap 95% CI [0.68, 1.36]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![cifar10 ROC](cifar10-roc.png)

![cifar10 empirical score distributions](cifar10-score-cdf.png)

## MNIST

far-OOD handwritten digits; official test; bilinear 28->32; repeated RGB.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 73.06 [72.34, 73.78] | 67.21 | 86.53 | 94.89 | 86.39 |
| Energy | 79.55 [78.94, 80.18] | 73.81 | 82.68 | 95.24 | 83.33 |
| MaxLogit | 79.00 [78.40, 79.68] | 72.82 | 83.53 | 95.12 | 84.04 |
| Entropy | 75.26 [74.57, 75.93] | 69.77 | 85.48 | 95.08 | 85.62 |
| KNN | 77.42 [76.86, 78.14] | 70.37 | 85.85 | 95.12 | 86.11 |
| Mahalanobis | 55.72 [54.86, 56.49] | 50.38 | 98.18 | 95.33 | 98.45 |

Energy − MSP AUROC difference: 6.49 percentage points (paired sample-bootstrap 95% CI [6.18, 6.87]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![mnist ROC](mnist-roc.png)

![mnist empirical score distributions](mnist-score-cdf.png)

## NOISE

uniform pixel noise; synthetic diagnostic only.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 97.73 [97.52, 97.91] | 96.38 | 12.30 | 94.89 | 11.91 |
| Energy | 98.62 [98.41, 98.79] | 96.38 | 0.37 | 95.24 | 0.49 |
| MaxLogit | 98.67 [98.49, 98.83] | 97.16 | 1.20 | 95.12 | 1.48 |
| Entropy | 98.63 [98.45, 98.80] | 96.72 | 1.56 | 95.08 | 1.59 |
| KNN | 98.38 [98.19, 98.59] | 95.78 | 1.13 | 95.12 | 1.24 |
| Mahalanobis | 81.30 [80.60, 81.99] | 66.29 | 100.00 | 95.33 | 100.00 |

Energy − MSP AUROC difference: 0.89 percentage points (paired sample-bootstrap 95% CI [0.74, 1.06]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![noise ROC](noise-roc.png)

![noise empirical score distributions](noise-score-cdf.png)

## Interpretation boundary

FPR95-ID is the OOD acceptance rate at the empirical 95% ID-test operating point. The separately reported validation-calibrated operating point uses no test values to fit its threshold. High AUROC alone does not imply useful rejection at high ID acceptance.

Intervals use 300 stratified paired bootstrap draws of ID/OOD test samples, fixed seed 2026. These approximate percentile intervals describe sample uncertainty conditional on this trained classifier. They do not estimate uncertainty from retraining or replace training-seed mean/sample SD.

Uniform noise is a diagnostic, not a natural OOD benchmark. Scores below chance are retained with their predeclared orientation; they are not flipped after observing test performance.

Checkpoint SHA-256: `4b50a24c474aecb753ed27d3fdf98ac7c02a25543e4f5840bfee06f38ffc241e`.
