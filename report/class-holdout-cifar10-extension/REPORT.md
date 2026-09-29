# Frozen 80-class holdout models on full CIFAR-10 test

Exploratory post-hoc extension selected after the same-source held-out-class
results had been viewed. CIFAR-10 was used in earlier project work. All three
original 100-epoch selected checkpoints, their 36,000 optimization-ID feature
banks, ID-only temperatures and validation thresholds were reused unchanged.
All 10,000 official CIFAR-10 test rows were scored against the same 8,000
CIFAR-100 ID test rows. No CIFAR-10 image was used for fit or selection.

CIFAR-10 OOD prevalence is 55.56% (10,000/18,000), while the original
held-out-class OOD prevalence was 20% (2,000/10,000). Their AUPR values
must not be directly compared as though class proportions were identical.
Full AUROC, AUPR-out/in, FPR95-ID and four original ID-validation operating
targets for all methods/variants/seeds are in `metrics.csv` and `operating.csv`.
Raw 80-class logits, 512-dimensional features, per-image scores and masks
remain in the local run folders. The table is descriptive, not a selected winner.

| Variant | Method | CIFAR-10 AUROC mean ± seed SD | Held-out-class AUROC mean ± seed SD | CIFAR-10 OOD acceptance at original ID-val 95% | Held-out-class OOD acceptance at original ID-val 95% |
|---|---|---:|---:|---:|---:|
| T1 | msp | 76.24 ± 0.53% | 74.11 ± 0.92% | 84.21 ± 2.20% | 86.45 ± 1.39% |
| T1 | energy | 76.49 ± 0.58% | 73.57 ± 0.73% | 85.38 ± 0.59% | 86.47 ± 0.06% |
| T1 | maxlogit | 76.69 ± 0.58% | 73.76 ± 0.76% | 84.55 ± 1.63% | 86.05 ± 1.04% |
| T1 | entropy | 77.02 ± 0.53% | 74.64 ± 0.87% | 84.20 ± 2.03% | 86.13 ± 1.48% |
| T1 | knn | 75.48 ± 0.25% | 73.09 ± 1.23% | 85.10 ± 0.79% | 88.35 ± 0.93% |
| T1 | mahalanobis | 53.61 ± 1.14% | 52.33 ± 0.82% | 96.79 ± 0.24% | 96.93 ± 0.42% |
| calibrated | msp | 76.82 ± 0.57% | 74.48 ± 0.89% | 84.22 ± 2.67% | 86.80 ± 1.84% |
| calibrated | energy | 76.23 ± 0.58% | 73.36 ± 0.69% | 85.53 ± 1.27% | 86.35 ± 0.85% |
| calibrated | maxlogit | 76.69 ± 0.58% | 73.76 ± 0.76% | 84.55 ± 1.63% | 86.05 ± 1.04% |
| calibrated | entropy | 77.25 ± 0.54% | 74.68 ± 0.77% | 84.61 ± 1.68% | 86.32 ± 0.90% |
| calibrated | knn | 75.48 ± 0.25% | 73.09 ± 1.23% | 85.10 ± 0.79% | 88.35 ± 0.93% |
| calibrated | mahalanobis | 53.61 ± 1.14% | 52.33 ± 0.82% | 96.79 ± 0.24% | 96.93 ± 0.42% |

Both populations share the same ID test scores and ID-validation
thresholds. FPR95-ID uses the ID test order statistic instead and is
separate from deployable calibration. Different semantic label sets,
the earlier use of CIFAR-10, and this post-hoc population choice limit
any cross-population interpretation. No external confirmation or causal
source-matching conclusion is claimed.
