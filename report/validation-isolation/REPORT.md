# Three-way ID-validation isolation replication

Exploratory paired three-training-seed result on the previously inspected CIFAR-10, MNIST, SVHN and Textures OOD tests. Noise is a separate synthetic diagnostic. No detector or checkpoint was selected from OOD performance, and these data do not provide independent confirmation.

The original 5,000 ID-validation examples selected checkpoints. Old T=1 operating thresholds used all 5,000; old calibrated temperature and threshold fits used disjoint 2,500/2,500 subsets of those same 5,000. In the new predeclared run, each class contributes 20 checkpoint rows, 15 temperature-fit rows and 15 operational-threshold rows (2,000/1,500/1,500 mutually disjoint total). All seeds completed the same 100-epoch training schedule. Temperature and four fixed operating thresholds were fit from their designated ID rows only. The old/new comparison also changes selection and fitting sample sizes; a difference cannot be attributed solely to validation reuse.

Old selected epochs: [89, 96, 92]; new: [92, 96, 98]. ID-test accuracy old 75.77 ± 0.30%, new 75.80 ± 0.27%, paired change +0.03 ± 0.08 percentage points. ID-fitted temperature old 1.3654, new 1.3177 (three seeds each).

Each cell below is a three-seed mean ± sample SD in percent. Δ is the paired new-minus-old mean ± sample SD in percentage points. FPR95-ID is a benchmark test-derived threshold; acceptance95 uses the fixed ID-validation threshold.

## T1 natural-OOD macro

| Method | AUROC old | AUROC new | Δ AUROC | FPR95-ID old | FPR95-ID new | Δ FPR95-ID | Acceptance95 old | Acceptance95 new | Δ acceptance95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 78.23 ± 1.32 | 78.11 ± 1.42 | -0.12 ± 0.10 | 80.09 ± 1.65 | 80.31 ± 2.10 | +0.23 ± 0.49 | 80.59 ± 2.18 | 81.00 ± 2.69 | +0.40 ± 0.72 |
| Energy | 80.97 ± 0.79 | 80.78 ± 0.87 | -0.20 ± 0.17 | 78.41 ± 2.23 | 78.80 ± 1.88 | +0.39 ± 0.46 | 78.62 ± 2.60 | 79.74 ± 3.04 | +1.12 ± 0.70 |
| MaxLogit | 80.95 ± 0.83 | 80.76 ± 0.94 | -0.19 ± 0.17 | 78.27 ± 1.71 | 78.58 ± 1.97 | +0.31 ± 0.37 | 78.68 ± 2.21 | 80.17 ± 2.40 | +1.49 ± 0.35 |
| Entropy | 79.80 ± 1.23 | 79.67 ± 1.35 | -0.13 ± 0.12 | 78.22 ± 1.78 | 78.85 ± 2.14 | +0.63 ± 0.59 | 78.89 ± 1.79 | 79.93 ± 2.99 | +1.04 ± 1.21 |
| KNN-50 | 82.42 ± 1.11 | 82.32 ± 1.18 | -0.10 ± 0.14 | 72.24 ± 2.34 | 72.32 ± 2.53 | +0.07 ± 0.60 | 73.03 ± 2.90 | 74.20 ± 4.84 | +1.17 ± 2.00 |
| Pooled Maha | 63.19 ± 2.20 | 63.24 ± 2.24 | +0.06 ± 0.20 | 89.91 ± 0.32 | 89.86 ± 0.24 | -0.05 ± 0.12 | 90.48 ± 0.26 | 90.35 ± 0.63 | -0.14 ± 0.62 |

Natural macro is the equal-weight mean of four named OOD populations within each seed, followed by a three-seed mean and sample SD.

## calibrated natural-OOD macro

| Method | AUROC old | AUROC new | Δ AUROC | FPR95-ID old | FPR95-ID new | Δ FPR95-ID | Acceptance95 old | Acceptance95 new | Δ acceptance95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MSP | 79.52 ± 1.20 | 79.24 ± 1.35 | -0.28 ± 0.15 | 79.11 ± 1.49 | 79.59 ± 2.20 | +0.48 ± 0.72 | 78.70 ± 2.84 | 79.87 ± 2.58 | +1.17 ± 0.27 |
| Energy | 80.85 ± 0.77 | 80.66 ± 0.83 | -0.18 ± 0.19 | 78.50 ± 2.57 | 79.07 ± 2.03 | +0.58 ± 0.60 | 79.08 ± 2.34 | 80.01 ± 2.89 | +0.93 ± 0.99 |
| MaxLogit | 80.95 ± 0.83 | 80.76 ± 0.94 | -0.19 ± 0.17 | 78.27 ± 1.71 | 78.58 ± 1.97 | +0.31 ± 0.37 | 78.44 ± 2.28 | 80.17 ± 2.40 | +1.73 ± 0.56 |
| Entropy | 80.90 ± 1.00 | 80.65 ± 1.16 | -0.25 ± 0.19 | 78.24 ± 1.94 | 78.61 ± 2.01 | +0.37 ± 0.63 | 78.18 ± 1.96 | 79.73 ± 2.95 | +1.55 ± 1.42 |
| KNN-50 | 82.42 ± 1.11 | 82.32 ± 1.18 | -0.10 ± 0.14 | 72.24 ± 2.34 | 72.32 ± 2.53 | +0.07 ± 0.60 | 73.19 ± 3.62 | 74.20 ± 4.84 | +1.01 ± 1.83 |
| Pooled Maha | 63.19 ± 2.20 | 63.24 ± 2.24 | +0.06 ± 0.20 | 89.91 ± 0.32 | 89.86 ± 0.24 | -0.05 ± 0.12 | 90.46 ± 0.33 | 90.35 ± 0.63 | -0.11 ± 0.34 |

Natural macro is the equal-weight mean of four named OOD populations within each seed, followed by a three-seed mean and sample SD.

## Interpretation and provenance

Raw per-image logits, features, six detector scores, correctness and all 80/90/95/99 acceptance masks are retained in each seed's local extraction and scoring folders. The machine-readable summary includes every method, variant and natural OOD population, plus separate noise. Independent feature, score and summary audits bind the frozen inputs and outputs.

AUROC, AUPR-out, AUPR-in, FPR95-ID, every operating rate and all three paired seed values are in [summary.json](summary.json). AUPR depends on the saved ID/OOD population counts there. The sample SD describes three training seeds, not a confidence interval over images. The four OOD test populations had already informed earlier method development; no best-method, causal or deployment claim follows from these comparisons.

Figures: [paired AUROC changes](paired-auroc.png) and [fixed95 natural macro OOD acceptance](fixed95-acceptance.png).
