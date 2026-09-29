# OOD Detection: Full 100-epoch schedule, seed 0

ID: cifar100; selected checkpoint epoch 89; ID test accuracy **76.11%**.

All detector scores increase toward OOD. Checkpoint selection uses ID validation accuracy only.

This is an independently trained CIFAR study, not a full OpenOOD reproduction. A pilot or one training seed does not support a multiseed conclusion.

## CIFAR10

near-OOD natural images; official test split.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 76.97 [76.33, 77.70] | 73.38 | 79.90 | 95.69 | 82.18 |
| Energy | 77.86 [77.20, 78.58] | 73.88 | 79.86 | 95.78 | 82.49 |
| MaxLogit | 78.05 [77.40, 78.79] | 74.35 | 79.50 | 95.78 | 81.93 |
| Entropy | 77.95 [77.31, 78.67] | 74.40 | 78.50 | 95.64 | 81.08 |
| KNN | 76.06 [75.41, 76.72] | 73.08 | 80.18 | 95.47 | 81.82 |
| Mahalanobis | 54.35 [53.64, 55.13] | 51.15 | 96.38 | 95.27 | 96.62 |

Energy − MSP AUROC difference: 0.88 percentage points (paired sample-bootstrap 95% CI [0.55, 1.30]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![cifar10 ROC](cifar10-roc.png)

![cifar10 empirical score distributions](cifar10-score-cdf.png)

## MNIST

far-OOD handwritten digits; official test; bilinear 28->32; repeated RGB.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 76.04 [75.37, 76.77] | 70.59 | 84.28 | 95.69 | 86.44 |
| Energy | 82.04 [81.51, 82.64] | 76.34 | 79.63 | 95.78 | 82.32 |
| MaxLogit | 81.57 [81.02, 82.19] | 75.83 | 80.22 | 95.78 | 83.06 |
| Entropy | 78.03 [77.38, 78.72] | 72.74 | 82.11 | 95.64 | 84.82 |
| KNN | 79.55 [78.97, 80.16] | 70.74 | 89.01 | 95.47 | 90.62 |
| Mahalanobis | 51.25 [50.54, 52.04] | 46.96 | 99.49 | 95.27 | 99.61 |

Energy − MSP AUROC difference: 6.00 percentage points (paired sample-bootstrap 95% CI [5.66, 6.38]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![mnist ROC](mnist-roc.png)

![mnist empirical score distributions](mnist-score-cdf.png)

## NOISE

uniform pixel noise; synthetic diagnostic only.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 58.11 [57.20, 59.06] | 48.66 | 100.00 | 95.69 | 100.00 |
| Energy | 68.57 [67.69, 69.50] | 55.21 | 100.00 | 95.78 | 100.00 |
| MaxLogit | 67.45 [66.57, 68.37] | 54.39 | 100.00 | 95.78 | 100.00 |
| Entropy | 60.14 [59.25, 61.16] | 49.76 | 100.00 | 95.64 | 100.00 |
| KNN | 95.60 [95.30, 95.89] | 92.44 | 35.00 | 95.47 | 39.82 |
| Mahalanobis | 99.94 [99.90, 99.97] | 99.88 | 0.00 | 95.27 | 0.00 |

Energy − MSP AUROC difference: 10.46 percentage points (paired sample-bootstrap 95% CI [9.97, 11.02]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![noise ROC](noise-roc.png)

![noise empirical score distributions](noise-score-cdf.png)

## Interpretation boundary

FPR95-ID is the OOD acceptance rate at the empirical 95% ID-test operating point. The separately reported validation-calibrated operating point uses no test values to fit its threshold. High AUROC alone does not imply useful rejection at high ID acceptance.

Intervals use 300 stratified paired bootstrap draws of ID/OOD test samples, fixed seed 2026. These approximate percentile intervals describe sample uncertainty conditional on this trained classifier. They do not estimate uncertainty from retraining or replace training-seed mean/sample SD.

Uniform noise is a diagnostic, not a natural OOD benchmark. Scores below chance are retained with their predeclared orientation; they are not flipped after observing test performance.

Checkpoint SHA-256: `f6c164834b98b8c3cca2a301a2288efbe3dc700b2680e09aee4f9694669d5f7e`.
