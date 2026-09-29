# OOD Detection: Full 3-epoch schedule, seed 7

ID: synthetic; selected checkpoint epoch 1; ID test accuracy **75.00%**.

All detector scores increase toward OOD. Checkpoint selection uses ID validation accuracy only.

This is an independently trained CIFAR study, not a full OpenOOD reproduction. A pilot or one training seed does not support a multiseed conclusion.

## NOISE

uniform pixel noise; synthetic diagnostic only.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 25.00 [15.62, 34.38] | 36.90 | 100.00 | 89.06 | 100.00 |
| Energy | 75.00 [64.80, 84.38] | 60.38 | 100.00 | 98.44 | 100.00 |
| MaxLogit | 25.00 [15.62, 34.38] | 36.90 | 100.00 | 95.31 | 100.00 |
| Entropy | 25.00 [15.62, 34.38] | 36.90 | 100.00 | 98.44 | 100.00 |
| KNN | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 93.75 | 0.00 |
| Mahalanobis | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 95.31 | 0.00 |

Energy − MSP AUROC difference: 50.00 percentage points (paired sample-bootstrap 95% CI [37.50, 62.50]).

ID count: 64; OOD count: 64. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![noise ROC](noise-roc.png)

![noise empirical score distributions](noise-score-cdf.png)

## CONSTANT

constant gray images; synthetic diagnostic only.

| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |
|---|---:|---:|---:|---:|---:|
| MSP | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 89.06 | 0.00 |
| Energy | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 98.44 | 0.00 |
| MaxLogit | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 95.31 | 0.00 |
| Entropy | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 98.44 | 0.00 |
| KNN | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 93.75 | 0.00 |
| Mahalanobis | 100.00 [100.00, 100.00] | 100.00 | 0.00 | 95.31 | 0.00 |

Energy − MSP AUROC difference: 0.00 percentage points (paired sample-bootstrap 95% CI [0.00, 0.00]).

ID count: 64; OOD count: 64. AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.

![constant ROC](constant-roc.png)

![constant empirical score distributions](constant-score-cdf.png)

## Interpretation boundary

FPR95-ID is the OOD acceptance rate at the empirical 95% ID-test operating point. The separately reported validation-calibrated operating point uses no test values to fit its threshold. High AUROC alone does not imply useful rejection at high ID acceptance.

Intervals use 300 stratified paired bootstrap draws of ID/OOD test samples, fixed seed 2026. These approximate percentile intervals describe sample uncertainty conditional on this trained classifier. They do not estimate uncertainty from retraining or replace training-seed mean/sample SD.

Uniform noise is a diagnostic, not a natural OOD benchmark. Scores below chance are retained with their predeclared orientation; they are not flipped after observing test performance.

Checkpoint SHA-256: `6a1a6214db36d44e70655e77ac5f08f9a0fceae00944be577a7a5e45f5038f5d`.
