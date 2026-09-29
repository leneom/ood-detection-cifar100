# CIFAR-100 same-source class holdout: paired image bootstrap

Post-hoc description of the completed three-seed 80-ID/20-OOD-class study.
The 1,000 paired draws resample 100 test images within each of the 100 observed
fine classes. The same draw is shared by all models and methods. Every draw
retains 8,000 ID and 2,000 OOD images. This interval describes image sampling
conditional on these classes, this class split and these three trained models;
it is not a confidence interval for unseen classes or new training seeds.
Training-seed sample SD and paired image bootstrap quantiles are separate.
All original thresholds and scores are frozen. No method winner or test-based
setting is selected. Complete numerical values are in `statistics.csv`;
all 1,000 raw draws and metric arrays remain in the immutable run blocks.

| Variant | Method | AUROC original mean ± seed SD | Paired image q2.5–q97.5 | OOD acceptance at original ID-val 95% target: mean ± seed SD | Paired image q2.5–q97.5 |
|---|---|---:|---:|---:|---:|
| T1 | msp | 74.11 ± 0.92% | 73.22–75.00% | 86.45 ± 1.39% | 85.45–87.37% |
| T1 | energy | 73.57 ± 0.73% | 72.58–74.53% | 86.47 ± 0.06% | 85.33–87.58% |
| T1 | maxlogit | 73.76 ± 0.76% | 72.76–74.72% | 86.05 ± 1.04% | 84.98–87.10% |
| T1 | entropy | 74.64 ± 0.87% | 73.75–75.55% | 86.13 ± 1.48% | 85.00–87.23% |
| T1 | knn | 73.09 ± 1.23% | 72.16–73.98% | 88.35 ± 0.93% | 87.32–89.33% |
| T1 | mahalanobis | 52.33 ± 0.82% | 51.28–53.37% | 96.93 ± 0.42% | 96.38–97.45% |
| calibrated | msp | 74.48 ± 0.89% | 73.59–75.37% | 86.80 ± 1.84% | 85.80–87.80% |
| calibrated | energy | 73.36 ± 0.69% | 72.35–74.31% | 86.35 ± 0.85% | 85.20–87.48% |
| calibrated | maxlogit | 73.76 ± 0.76% | 72.76–74.72% | 86.05 ± 1.04% | 84.98–87.10% |
| calibrated | entropy | 74.68 ± 0.77% | 73.76–75.58% | 86.32 ± 0.90% | 85.25–87.45% |
| calibrated | knn | 73.09 ± 1.23% | 72.16–73.98% | 88.35 ± 0.93% | 87.32–89.33% |
| calibrated | mahalanobis | 52.33 ± 0.82% | 51.28–53.37% | 96.93 ± 0.42% | 96.38–97.45% |

The original AUROC and OOD acceptance refer to the full official test
population; the bootstrap quantiles refer to the fixed three-model mean.
FPR95-ID is recomputed from each draw's ID test order statistic, while
fixed95 acceptance always uses the original disjoint ID-validation threshold.
The same images underlie the held-out-class heterogeneity analysis; these
descriptive analyses are not independent external confirmations.
