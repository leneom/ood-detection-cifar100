# Study log

This log records the studies in the order they were carried out. Each entry gives the question, the design choices that matter and the main result, with links to the frozen protocol and the full report. Unless stated otherwise, values are the mean ± sample SD over three training seeds. They are not confidence intervals.

An overview is in the [README](../README.md). Shared definitions are in [PROTOCOL.md](PROTOCOL.md), and concept explanations in Chinese are in the [learning guide](LEARNING_GUIDE.zh-CN.md).

## Part I — Exploration on the development datasets

### 1. Pipeline and pilot

- Fixed stratified CIFAR-100 split (seed 2026): 45,000 training and 5,000 validation images. ResNet-18 for 32×32 inputs, with exact resume. The resume restores optimizer, scheduler, GradScaler and all RNG states, and checks configuration, split and source hashes.
- Six detectors were fixed before any full run: MSP, Energy, MaxLogit, Entropy, normalised KNN (k = 50) and final-layer pooled Mahalanobis (1% shrinkage). Feature statistics come from the 45,000 training images only.
- A 10-epoch pilot (52.7% ID accuracy) validated the pipeline. Its results are kept as a pilot and are not used for conclusions.
- Reports: [pilot](../report/pilot-seed0-all/REPORT.md), [pilot failure analysis](../report/pilot-failure-analysis/REPORT.md).

### 2. Three-seed baseline on four natural OOD datasets

- Three 100-epoch models, with the checkpoint chosen by ID validation accuracy (selected epochs 89/96/92). ID test accuracy 75.77 ± 0.30%.
- OOD: CIFAR-10 (10,000), MNIST (10,000), SVHN (26,032) and DTD Textures (1,880, official partition 1). Uniform noise is kept as a separate synthetic check.
- Four-dataset macro AUROC: KNN 82.42 ± 1.11, Energy 80.97 ± 0.79, MaxLogit 80.95 ± 0.83, Entropy 79.80 ± 1.23, MSP 78.23 ± 1.32, Mahalanobis 63.19 ± 2.20.
- At 95% ID acceptance (FPR95), 72–90% of OOD images are still accepted, depending on the method. A higher AUROC does not mean useful rejection at a high-recall operating point.
- Reports: [baseline](../report/baseline-complete/REPORT.md), [figures](../report/matrix-comparison/REPORT.md), [seed-0 failure analysis](../report/final-seed0-failure-analysis/REPORT.md), [output distributions](../report/output-diagnostics-seeds0-1/REPORT.md).

### 3. ID-only temperature calibration and operating points

- The temperature was fitted by NLL on ID validation only (T ≈ 1.36). Thresholds come from a disjoint ID-validation subset at 80/90/95/99% target acceptance.
- Calibration works as calibration: ECE falls from 9.02% to 3.48%, and NLL from 1.002 to 0.931. OOD rejection barely changes: at 95%, MSP macro OOD acceptance moves from 79.78% to 78.70%, and the error rate among retained ID images stays at about 21.5%.
- Protocols: [calibration](CALIBRATION_PHASE2.md), [operating points](SELECTIVE_RISK_PHASE2.md). Reports: [calibration](../report/calibration-phase2/REPORT.md), [operating points](../report/selective-risk-phase2/REPORT.md).

### 4. Feature-geometry scores: ViM, Residual, ReAct

- All statistics come from training features only, with D = 256 and a clipping quantile of 0.99 fixed in advance.
- Macro AUROC: ReAct 81.43 ± 0.67, ViM 81.26 ± 1.20, Energy 80.97 ± 0.79, Residual 53.80 ± 2.28. ViM shows that the macro hides opposite effects: +9.58 pp on Textures but −4.05 pp on CIFAR-10 and −3.79 pp on SVHN.
- Protocol: [GEOMETRY_PHASE3](GEOMETRY_PHASE3.md). Reports: [results](../report/geometry-phase3/REPORT.md), [decision diagnostics](../report/geometry-diagnostics/REPORT.md).

### 5. Relative Mahalanobis distance (RMD)

- Class and background covariances come from training features with fixed 1% shrinkage, and the signed relative distance is kept.
- Macro AUROC: RMD 81.61 ± 1.57 versus matched Mahalanobis 63.19 ± 2.20, a paired gain of +18.43 ± 1.72 pp. RMD does not exceed KNN (82.42). On CIFAR-10, most of RMD's mean ID–OOD separation comes from the background term; this is a decomposition of means, not a causal attribution.
- Protocol: [RMD_PHASE4](RMD_PHASE4.md). Reports: [results](../report/rmd-phase4/REPORT.md), [component diagnostics](../report/rmd-diagnostics/REPORT.md).

### 6. Activation shaping: ASH-P/B/S

- The pruning percentile was fixed at p = 65, keeping 179 of 512 coordinates, with no tuning.
- Macro AUROC: ASH-S 83.16 ± 0.72, ASH-P 81.42 ± 0.71, ASH-B 69.48 ± 2.21. ASH-S minus Energy is +2.18 ± 0.78 pp. On CIFAR-10 the change is about 0 pp; on MNIST, SVHN and Textures it is positive.
- This was the strongest exploratory result, and it became hypothesis H1 in study 12.
- Protocol: [ASH_PHASE5](ASH_PHASE5.md). Reports: [results](../report/ash-phase5/REPORT.md), [support/scale diagnostics](../report/ash-diagnostics/REPORT.md).

### 7. Uncertainty: paired image bootstrap and shared-sample dependence

- 1,000 paired bootstrap resamples of the test images, with the same weights shared by all methods and models. The ASH-S minus Energy macro interval is [+2.05, +2.31] pp.
- These intervals cover image sampling only, for fixed models and fixed datasets. Because seeds and datasets share the same ID test images, treating them as independent samples understates the image-level SD by a factor of 1.4–1.7.
- Protocols: [PAIRED_BOOTSTRAP](PAIRED_BOOTSTRAP.md), [BOOTSTRAP_DEPENDENCE](BOOTSTRAP_DEPENDENCE.md). Reports: [bootstrap](../report/paired-bootstrap/REPORT.md), [dependence](../report/bootstrap-dependence/REPORT.md).

### 8. Three-way validation isolation

- Concern: the 5,000 validation images had been used both to select checkpoints and to fit temperatures and thresholds.
- Three new models were trained with disjoint per-class validation roles: 2,000 for checkpoint selection, 1,500 for temperature fitting and 1,500 for threshold calibration. Selected epochs were 92/96/98.
- ID accuracy was 75.80 ± 0.27%, against 75.77 ± 0.30% before. Energy macro AUROC moved from 80.97 to 80.78, a paired change of −0.20 ± 0.17 pp. The reuse did not distort the earlier results. All later studies use these isolated models.
- Protocol: [VALIDATION_ISOLATION](VALIDATION_ISOLATION.md). Report: [isolation](../report/validation-isolation/REPORT.md).

### 9. Project assessment

- An evidence index of the eleven studies above, with their limitations: validation reuse (fixed in study 8), exploratory comparisons on datasets already seen, and no external benchmark.
- Report: [assessment](../report/project-assessment/REPORT.md).

## Part II — New data

### 10. New OOD datasets

| Dataset | Images | Status | Energy AUROC | KNN AUROC |
|---|---:|---|---:|---:|
| Tiny ImageNet-200 (validation) | 10,000 | protocol frozen before the first scoring | 78.95 ± 0.33 | 80.17 ± 0.28 |
| Fashion-MNIST (test) | 10,000 | added after Tiny ImageNet | 90.42 ± 0.16 | 90.48 ± 0.44 |
| Places365-Standard (validation) | 36,500 | added later | 75.93 ± 0.85 | 75.56 ± 0.84 |
| Oxford-IIIT Pets (test) | 3,669 | added later | 81.60 ± 0.78 | 83.56 ± 0.09 |

- Only the six original scores were computed on these datasets at this stage, so they stayed unscored for ASH, RMD and LogitNorm.
- Protocols: [Tiny ImageNet](TINYIMAGENET_CONFIRMATION.md), [Fashion-MNIST](FASHIONMNIST_EXTENSION.md), [Places365](PLACES365_EXTENSION.md), [Oxford Pets](OXFORD_PETS_EXTENSION.md).
- Reports: [Tiny ImageNet](../report/tinyimagenet-confirmation/REPORT.md), [Fashion-MNIST](../report/fashionmnist-extension/REPORT.md), [Places365](../report/places365-extension/REPORT.md), [Oxford Pets](../report/oxford-pets-extension/REPORT.md).
- Descriptive breakdowns: [Tiny ImageNet classes](../report/tinyimagenet-class-heterogeneity/REPORT.md), [Places365 scenes](../report/places365-scene-diagnostic/REPORT.md), [Pets paired decisions](../report/oxford-pets-paired-decisions/REPORT.md), [Pets breeds](../report/oxford-pets-breed-diagnostic/REPORT.md).

### 11. Same-source held-out classes

- Question: how well do the detectors work when OOD images come from the same source as ID, so that only the semantics differ?
- 80 ID and 20 held-out CIFAR-100 fine classes. The partition was frozen before training, and the 36,000 training images exclude the held-out classes entirely. Validation roles are 1,600/1,200/1,200.
- First partition (2031): ID accuracy 76.96 ± 0.38%, Energy AUROC 73.57 ± 0.73%.
- Three partitions (2031/2032/2033) × three seeds = nine models. ID accuracy 76.59 ± 1.04%. AUROC: Entropy 76.13, MSP 75.46, MaxLogit 75.15, Energy 74.92, KNN 74.49, RMD 74.29, Mahalanobis 53.16. At 95%, 82–86% of novel-class images are accepted. Here ± is the SD of the three partition means.
- Follow-ups on the same models:
  - On CIFAR-10, Energy AUROC is 76.49 ± 0.58% and 85.4% of images are accepted at 95%.
  - At a common 20% OOD prior, AUPR-out on CIFAR-10 drops from 76.3% to 39.9%. The original value reflects CIFAR-10's 56% share of the test set.
  - Per-class acceptance varies widely: for Energy on partition 2031, the accepted share of a held-out class ranges from 71% to 98%.
- Protocols: [CIFAR100_CLASS_HOLDOUT](CIFAR100_CLASS_HOLDOUT.md), [CLASS_HOLDOUT_REPLICATION](CLASS_HOLDOUT_REPLICATION.md). Reports: [first partition](../report/cifar100-class-holdout/SUMMARY.md), [three partitions](../report/cifar100-class-holdout-replication/SUMMARY.md), [RMD](../report/class-holdout-rmd-extension/REPORT.md), [CIFAR-10](../report/class-holdout-cifar10-extension/REPORT.md), [fixed prior](../report/class-holdout-prior-standardization/REPORT.md), [per class](../report/cifar100-class-holdout-classwise/REPORT.md).

## Part III — Pre-registered tests

### 12. ASH-S and RMD confirmation

- Hypotheses: H1, ASH-S > Energy; H2, RMD > matched Mahalanobis. They were tested on the four datasets from study 10, where no ASH or RMD score had ever been computed. The protocol and a 73-input declaration were committed in `59e914f` before scoring.
- **H1 not confirmed.** Tiny ImageNet −0.24 ± 0.04 pp (all seeds negative), Fashion-MNIST +1.23, Places365 +1.09, Oxford Pets +0.22 (mixed).
- **H2 confirmed.** +19.9 to +27.6 pp on all four datasets and all seeds. RMD is still not better than Energy or KNN overall.
- Protocol: [POSTHOC_CONFIRMATION](POSTHOC_CONFIRMATION.md). Report: [confirmation](../report/posthoc-confirmation/REPORT.md).

### 13. LogitNorm representation study

- Only the training loss changes, to LogitNorm with τ = 0.04. Everything else matches the three isolated cross-entropy models, which serve as seed-matched comparators. The protocol was frozen before training (`3b734e6`), and the evaluation code before any LogitNorm OOD output existed (`b8f8703`).
- **H3 (MSP improves) not confirmed.** Supported on 4 of 8 datasets and contradicted on 3.
- **H4 (KNN improves) not confirmed.** Supported on 2 of 8 datasets and contradicted on 5.
- ID accuracy fell by 2.00 ± 0.48 pp. The gains are concentrated on grayscale sources: at the fixed 95% threshold, the lowest OOD acceptance fell from 54% to 15% on Fashion-MNIST and from 87% to 36% on MNIST. On the four natural-photo datasets, no method goes below about 76%.
- Confound: the ID-only checkpoint rule picked epochs 57/55/54 for LogitNorm, against 92/96/98 for cross-entropy.
- Protocol: [LOGITNORM_REPRESENTATION](LOGITNORM_REPRESENTATION.md). Report: [LogitNorm](../report/logitnorm-representation/REPORT.md).

## Open question and next step

ASH-S and LogitNorm help where there is an obvious low-level source difference, such as grayscale digits and clothing, but not on object-centric photos or same-source held-out classes. The next study tests this directly with a 2×2 design on the nine same-source models: held-out versus ID classes, each with and without an added low-level cue (grayscale, or downsampling from 32 to 16 and back to 32). The predictions and decision rule will be committed before scoring, as in studies 12 and 13.
