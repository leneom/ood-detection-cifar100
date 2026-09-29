# Interim observations: 2 of 3 predeclared seeds

Seeds 0 and 1 completed their unchanged 100-epoch schedules; seed 2 is still pending. This is a per-seed comparison, not the final three-seed benchmark. No method, orientation or hyperparameter is selected from these observations.

| Seed | Selected epoch (ID validation only) | ID test accuracy % |
|---|---:|---:|
| 0 | 89 | 76.11 |
| 1 | 96 | 75.56 |

## cifar10

| Method | Seed 0 AUROC % | Seed 1 AUROC % | Seed 0 FPR95-ID % | Seed 1 FPR95-ID % |
|---|---:|---:|---:|---:|
| msp | 76.97 | 76.79 | 79.90 | 81.77 |
| energy | 77.86 | 77.61 | 79.86 | 83.05 |
| maxlogit | 78.05 | 77.81 | 79.50 | 82.37 |
| entropy | 77.95 | 77.77 | 78.50 | 81.96 |
| knn | 76.06 | 76.25 | 80.18 | 81.78 |
| mahalanobis | 54.35 | 54.81 | 96.38 | 95.97 |

## mnist

| Method | Seed 0 AUROC % | Seed 1 AUROC % | Seed 0 FPR95-ID % | Seed 1 FPR95-ID % |
|---|---:|---:|---:|---:|
| msp | 76.04 | 78.43 | 84.28 | 86.16 |
| energy | 82.04 | 78.48 | 79.63 | 93.43 |
| maxlogit | 81.57 | 79.05 | 80.22 | 91.86 |
| entropy | 78.03 | 79.33 | 82.11 | 90.85 |
| knn | 79.55 | 82.43 | 89.01 | 83.02 |
| mahalanobis | 51.25 | 70.74 | 99.49 | 94.19 |

## noise

| Method | Seed 0 AUROC % | Seed 1 AUROC % | Seed 0 FPR95-ID % | Seed 1 FPR95-ID % |
|---|---:|---:|---:|---:|
| msp | 58.11 | 98.35 | 100.00 | 6.90 |
| energy | 68.57 | 98.47 | 100.00 | 0.91 |
| maxlogit | 67.45 | 98.80 | 100.00 | 1.28 |
| entropy | 60.14 | 99.03 | 100.00 | 0.77 |
| knn | 95.60 | 99.28 | 35.00 | 0.02 |
| mahalanobis | 99.94 | 78.41 | 0.00 | 99.74 |

## What remains uncertain

Similar ID accuracy does not establish similar OOD behavior. Energy minus MSP AUROC on MNIST is approximately +6.00 pp for seed 0 and +0.05 pp for seed 1. Corresponding FPR95-ID differences are -4.65 pp and +7.27 pp; a better AUROC can coexist with a worse operating point. Both seeds retain high natural-OOD acceptance. Wait for the third predeclared seed before reporting the final seed mean/SD.

Uniform-noise detector behavior changes substantially between these classifiers. Noise is a synthetic diagnostic and does not establish natural-image performance. The code/source hashes, detector fit population, and training configuration excluding seed match. The noise generator uses a fresh NumPy RNG with the fixed default seed 110. A posthoc pixel reconstruction hash is recorded in noise-protocol-probe.json; an input pixel hash was not captured during the original evaluations.

All six methods are shown, including poor results. Bootstrap sample intervals from each classifier are separate from training-seed variability. No score sign, temperature, k, shrinkage or validation-calibration rule was changed.
