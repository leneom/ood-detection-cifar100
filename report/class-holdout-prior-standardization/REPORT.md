# Fixed-20%-prior precision-recall description

All three already completed 80-class models, both temperature variants and
all six unchanged scores are included. Every observed ID and OOD score is
retained: the 8,000 ID rows each receive weight 0.8/8,000, while the 2,000
held-out-class or 10,000 CIFAR-10 rows each receive 0.2/n_OOD. Grouped score
ties contribute together to noninterpolated average precision. AUPR-in
reverses the positive label and score sign while retaining these weights.

This 20% OOD prior matches the observed held-out-class test mixture; it is a
**hypothetical display prior**, not an estimated deployment prevalence.
The full CIFAR-10 test population originally had 55.56% OOD prevalence.
For T=1 Energy, its natural-prior CIFAR-10 AUPR-out was 76.26%
(seed SD 0.43 pp); at the fixed 20% prior
it is 39.94% (seed SD 0.59 pp).
Complete per-seed natural and standardized values are in `statistics.csv`.

| Variant | Method | Held-out AP-out @20% | CIFAR-10 AP-out @20% | Held-out AP-in @20% | CIFAR-10 AP-in @20% |
|---|---|---:|---:|---:|---:|
| T1 | msp | 36.83 ± 1.36% | 39.48 ± 0.26% | 92.07 ± 0.36% | 92.95 ± 0.24% |
| T1 | energy | 37.38 ± 1.00% | 39.94 ± 0.59% | 91.52 ± 0.26% | 92.87 ± 0.28% |
| T1 | maxlogit | 37.54 ± 1.12% | 40.34 ± 0.50% | 91.58 ± 0.27% | 92.93 ± 0.27% |
| T1 | entropy | 37.85 ± 1.21% | 40.72 ± 0.46% | 92.20 ± 0.34% | 93.14 ± 0.25% |
| T1 | knn | 36.21 ± 1.61% | 40.48 ± 1.19% | 91.54 ± 0.52% | 91.75 ± 0.06% |
| T1 | mahalanobis | 20.06 ± 0.45% | 20.48 ± 0.58% | 82.90 ± 0.61% | 83.45 ± 0.54% |
| calibrated | msp | 37.47 ± 1.34% | 40.33 ± 0.36% | 92.15 ± 0.34% | 93.09 ± 0.26% |
| calibrated | energy | 37.15 ± 0.92% | 39.52 ± 0.61% | 91.45 ± 0.24% | 92.79 ± 0.28% |
| calibrated | maxlogit | 37.54 ± 1.12% | 40.34 ± 0.50% | 91.58 ± 0.27% | 92.93 ± 0.27% |
| calibrated | entropy | 37.97 ± 1.10% | 40.77 ± 0.51% | 92.19 ± 0.30% | 93.20 ± 0.26% |
| calibrated | knn | 36.21 ± 1.61% | 40.48 ± 1.19% | 91.54 ± 0.52% | 91.75 ± 0.06% |
| calibrated | mahalanobis | 20.06 ± 0.45% | 20.48 ± 0.58% | 82.90 ± 0.61% | 83.45 ± 0.54% |

The held-out-class 20%-prior values reproduce its original natural
AUPR. Reweighting changes only the precision-recall mixture arithmetic:
it does not align semantic difficulty, data source, class splits or score
distributions. Both OOD tests had already been seen, and CIFAR-10 was
chosen after the held-out-class result. This is exploratory description,
not independent confirmation, a detector ranking, a confidence interval,
or a causal comparison of dataset shifts.
