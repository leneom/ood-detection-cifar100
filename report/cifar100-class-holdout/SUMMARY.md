# CIFAR-100 class-holdout detection — three-seed report

One fine class in each of 20 coarse groups was held out before training. Each fresh 80-class ResNet-18 used only 36,000 ID optimization images; the 1,600/1,200/1,200 ID-validation roles respectively selected the checkpoint, fitted temperature, and fixed acceptance thresholds. The official test set contributes all 8,000 ID and 2,000 held-out-class OOD images in original order. Each detector is fixed in advance. Values below are mean ± sample SD over training seeds 0, 1, 2; all six methods are shown without a test-based winner.

ID-test classifier accuracy: **76.96 ± 0.38%**. Selected epochs: [96, 76, 81].

## T=1

| Method | AUROC (%) | AUPR-out (%) | AUPR-in (%) | FPR95-ID (%) | ID accepted at 95% ID-validation target (%) | OOD accepted at same threshold (%) |
|---|---:|---:|---:|---:|---:|---:|
| msp | 74.11 ± 0.92 | 36.83 ± 1.36 | 92.07 ± 0.36 | 85.42 ± 1.06 | 95.43 ± 0.81 | 86.45 ± 1.39 |
| energy | 73.57 ± 0.73 | 37.38 ± 1.00 | 91.52 ± 0.26 | 84.00 ± 0.49 | 95.87 ± 0.14 | 86.47 ± 0.06 |
| maxlogit | 73.76 ± 0.76 | 37.54 ± 1.12 | 91.58 ± 0.27 | 83.68 ± 1.02 | 95.72 ± 0.56 | 86.05 ± 1.04 |
| entropy | 74.64 ± 0.87 | 37.85 ± 1.21 | 92.20 ± 0.34 | 84.30 ± 1.16 | 95.61 ± 0.90 | 86.13 ± 1.48 |
| knn | 73.09 ± 1.23 | 36.21 ± 1.61 | 91.54 ± 0.52 | 85.13 ± 1.12 | 96.11 ± 0.31 | 88.35 ± 0.93 |
| mahalanobis | 52.33 ± 0.82 | 20.06 ± 0.45 | 82.90 ± 0.61 | 96.42 ± 0.62 | 95.67 ± 0.08 | 96.93 ± 0.42 |

## ID-only fitted temperature

| Method | AUROC (%) | AUPR-out (%) | AUPR-in (%) | FPR95-ID (%) | ID accepted at 95% ID-validation target (%) | OOD accepted at same threshold (%) |
|---|---:|---:|---:|---:|---:|---:|
| msp | 74.48 ± 0.89 | 37.47 ± 1.34 | 92.15 ± 0.34 | 85.23 ± 1.03 | 95.53 ± 1.08 | 86.80 ± 1.84 |
| energy | 73.36 ± 0.69 | 37.15 ± 0.92 | 91.45 ± 0.24 | 83.92 ± 0.46 | 95.75 ± 0.35 | 86.35 ± 0.85 |
| maxlogit | 73.76 ± 0.76 | 37.54 ± 1.12 | 91.58 ± 0.27 | 83.68 ± 1.02 | 95.72 ± 0.56 | 86.05 ± 1.04 |
| entropy | 74.68 ± 0.77 | 37.97 ± 1.10 | 92.19 ± 0.30 | 83.82 ± 1.09 | 95.74 ± 0.54 | 86.32 ± 0.90 |
| knn | 73.09 ± 1.23 | 36.21 ± 1.61 | 91.54 ± 0.52 | 85.13 ± 1.12 | 96.11 ± 0.31 | 88.35 ± 0.93 |
| mahalanobis | 52.33 ± 0.82 | 20.06 ± 0.45 | 82.90 ± 0.61 | 96.42 ± 0.62 | 95.67 ± 0.08 | 96.93 ± 0.42 |

FPR95-ID above uses an ID **test** order statistic for the benchmark; the two acceptance columns use the separate frozen ID-validation threshold. AUPR uses the saved 8,000:2,000 population ratio. The calibrated variant changes MSP, Energy and Entropy; MaxLogit and both feature scores remain unchanged by temperature.

The images share the official CIFAR-100 source and 32×32 format, but this experiment also changes the trained classifier and ID label space relative to earlier 100-class runs. Earlier CIFAR-100 images were used in this project, so this is not untouched external confirmation. These observations do not identify a causal effect of source matching or justify choosing a method from this test set.

Machine-readable per-seed values: `summary.json`, `metrics.csv`, `operating.csv`; original logits, features, scores, masks, checkpoints and independent audits remain in `runs/cifar100-class-holdout/`.
