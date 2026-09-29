# OOD Detection: 10-epoch pilot, seed 0

ID: cifar100; selected checkpoint epoch 10; ID test accuracy **52.69%**.

All detector scores increase toward OOD. Checkpoint selection uses ID validation accuracy only.

This is an independently trained CIFAR study, not a full OpenOOD reproduction. A pilot or one training seed does not support a multiseed conclusion.

## MNIST

far-OOD handwritten digits; official test; bilinear 28->32; repeated RGB.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 66.41 [65.66, 67.16] | 57.39 | 97.77 | 95.60 | 98.26 |
| Energy | 76.98 [76.32, 77.67] | 66.57 | 93.97 | 95.38 | 95.21 |
| MaxLogit | 75.24 [74.58, 75.94] | 64.37 | 96.50 | 95.34 | 97.12 |
| Entropy | 72.95 [72.28, 73.66] | 62.68 | 96.99 | 95.57 | 97.89 |
| KNN | 75.23 [74.54, 75.94] | 62.44 | 99.35 | 94.81 | 99.34 |
| Mahalanobis | 42.11 [41.48, 42.94] | 43.55 | 98.38 | 94.69 | 98.18 |

Energy − MSP AUROC difference: 10.58 percentage points (paired sample-bootstrap 95% CI [10.03, 11.06]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![mnist ROC](mnist-roc.png)

![mnist empirical score distributions](mnist-score-cdf.png)

## NOISE

uniform pixel noise; synthetic diagnostic only.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 10.89 [10.43, 11.39] | 32.64 | 100.00 | 95.60 | 100.00 |
| Energy | 7.06 [6.68, 7.52] | 31.91 | 100.00 | 95.38 | 100.00 |
| MaxLogit | 6.81 [6.44, 7.25] | 31.86 | 100.00 | 95.34 | 100.00 |
| Entropy | 10.75 [10.28, 11.25] | 32.62 | 100.00 | 95.57 | 100.00 |
| KNN | 89.63 [89.07, 90.11] | 76.60 | 99.58 | 94.81 | 99.48 |
| Mahalanobis | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 94.69 | 0.00 |

Energy − MSP AUROC difference: -3.83 percentage points (paired sample-bootstrap 95% CI [-4.30, -3.42]).

ID count: 10000; OOD count: 10000. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![noise ROC](noise-roc.png)

![noise empirical score distributions](noise-score-cdf.png)

## Interpretation boundary

FPR95-ID is the OOD acceptance rate at the empirical 95% ID-test operating point. The separately reported validation-calibrated operating point uses no test values to fit its threshold. High AUROC alone does not imply useful rejection at high ID acceptance.

Intervals use 300 stratified paired bootstrap draws of ID/OOD test samples, fixed seed 2026. These approximate percentile intervals describe sample uncertainty conditional on this trained classifier. They do not estimate uncertainty from retraining or replace training-seed mean/sample SD.

Uniform noise is a diagnostic, not a natural OOD benchmark. Scores below chance are retained with their predeclared orientation; they are not flipped after observing test performance.

Checkpoint SHA-256: `0ff378d75e99480d7c799c420af3754ea57d8cd001fe16414603df9775d412e5`.
