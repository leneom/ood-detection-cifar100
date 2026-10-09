# Do Post-hoc OOD Detection Gains Survive New Data? A Controlled CIFAR-100 Study

This project studies out-of-distribution (OOD) detection with fixed CIFAR-100 classifiers. It compares 13 post-hoc scores and one training-loss change across eight OOD datasets and a same-source held-out-class protocol, using three training seeds throughout. The two method claims that came out of exploration were then retested on datasets they had never been scored on, with the hypotheses and decision rules committed to git before any scoring.

[中文说明](README.zh-CN.md) · [Study log](docs/PROGRESS.md) · [Core protocol](docs/PROTOCOL.md) · [Concept guide (Chinese)](docs/LEARNING_GUIDE.zh-CN.md)

## Key findings

**1. An exploratory gain did not survive new datasets.** On the four datasets used during development, ASH-S beat Energy by +2.18 pp macro AUROC, with a narrow image-bootstrap interval of [+2.05, +2.31] pp. On four new datasets the mean gain fell to about +0.6 pp, and on Tiny ImageNet it was negative for all three seeds (−0.24 ± 0.04 pp). The same pre-registered test confirmed that RMD beats the Mahalanobis distance on every dataset and seed (+20 to +28 pp). The uncertainty that mattered came from which datasets were chosen, and neither the image bootstrap nor seed variation measures that.

**2. The large gains came from datasets with obvious low-level differences.** ASH-S and LogitNorm helped most on grayscale sources. With LogitNorm, the lowest Fashion-MNIST OOD acceptance at a fixed 95% ID threshold fell from 54% to 15%. On object-centric photos (CIFAR-10, Tiny ImageNet, Oxford Pets), the AUROC changes were small and mixed in sign, from −3.5 to +1.6 pp. LogitNorm also lowered ID accuracy by 2.0 pp.

**3. When the source difference is removed, the methods are nearly indistinguishable.** On held-out CIFAR-100 classes (three class partitions × three seeds), the five usable scores span 74.5–76.1% AUROC, and 82–85% of novel-class images still pass the 95% ID threshold.

These results agree with SC-OOD (Yang et al., 2021), which argued that common OOD benchmarks reward sensitivity to low-level differences between data sources. This project does not propose a new method. Its contribution is controlled, pre-registered evidence for that view at CIFAR scale, including the negative results.

![Paired AUROC differences for the two pre-registered hypotheses](report/posthoc-confirmation/paired_auroc_differences.png)

*Each dot is one training seed. Filled dots are datasets added for the pre-registered test; hollow dots are datasets used during development.*

## Setup

- **ID data.** CIFAR-100 with a fixed stratified split (split seed 2026): 45,000 training and 5,000 validation images. The official 10,000 test images are used only for evaluation.
- **Classifier.** ResNet-18 adapted to 32×32 inputs, trained from scratch with cross-entropy. SGD with lr 0.1, momentum 0.9, weight decay 5e-4 and batch 128, for 100 epochs, with the lr multiplied by 0.1 after epochs 50, 75 and 90. Three seeds; ID test accuracy 75.8 ± 0.3%.
- **Validation roles.** The 5,000 validation images are split per class into disjoint sets for checkpoint selection (2,000), temperature fitting (1,500) and threshold calibration (1,500). No OOD image is used to select or fit anything.
- **OOD data.** CIFAR-10, MNIST, SVHN and Textures (DTD) were used during development. Tiny ImageNet-200, Fashion-MNIST, Places365 and Oxford-IIIT Pets were added later and never used to develop ASH-S, RMD or LogitNorm.
- **Same-source semantic OOD.** 80 CIFAR-100 classes are ID and 20 are held out of training. Three class partitions × three seeds give nine 80-class models.
- **Scores.** MSP, Energy, MaxLogit, Entropy, KNN (k = 50), Mahalanobis, ViM, Residual, ReAct, RMD and ASH-P/B/S. In every case a higher score means more OOD-like. Training-time change: LogitNorm (τ = 0.04).
- **Metrics.** AUROC, AUPR-in/out, FPR95, and OOD acceptance at thresholds calibrated on ID validation for 80/90/95/99% ID acceptance.

## How the evidence is controlled

- **ID-only choices.** Checkpoints, temperatures, thresholds and feature statistics come only from ID training or validation data.
- **Pre-registration.** For the confirmation studies, the protocol, the hypotheses and a hash-bound declaration of all inputs were committed before scoring: `59e914f` for ASH-S/RMD, `3b734e6` and `b8f8703` for LogitNorm. A hypothesis counts as *supported* if all three seed-paired differences are positive, *contradicted* if all are negative, and *mixed* otherwise.
- **Independent recomputation.** Each study has a separate audit script that recomputes the scores and metrics from the saved raw logits and features. Each study's `completion.json` records SHA-256 hashes of its inputs and outputs.
- **Nothing discarded.** Failed attempts and the fixes applied to them are kept next to the reports (`report/*/…attempt-*`).

## Study map

| # | Study | Question | Outcome | Report |
|---|---|---|---|---|
| 1 | Baseline | Six classic scores on four natural OOD sets | Four-dataset macro AUROC: Energy 81.0%, KNN 82.4%, MSP 78.2%, Mahalanobis 63.2% | [baseline](report/baseline-complete/REPORT.md) |
| 2 | Calibration and operating points | Does ID temperature scaling help OOD rejection? | ECE falls from 9.0% to 3.5%, but MSP OOD acceptance at 95% only moves from 79.8% to 78.7% | [calibration](report/calibration-phase2/REPORT.md), [operating points](report/selective-risk-phase2/REPORT.md) |
| 3 | ViM, Residual, ReAct | Does training-feature geometry help? | ViM gains +9.6 pp on Textures but loses 4.1 pp on CIFAR-10; no uniform gain | [geometry](report/geometry-phase3/REPORT.md) |
| 4 | RMD | Does subtracting a background Gaussian fix Mahalanobis? | +18.4 pp macro over Mahalanobis, but not above KNN (82.4%) | [RMD](report/rmd-phase4/REPORT.md) |
| 5 | ASH-P/B/S | Does activation shaping help? | ASH-S is 2.2 pp above Energy (exploratory) | [ASH](report/ash-phase5/REPORT.md) |
| 6 | Image bootstrap | How large is image-sampling uncertainty? | Intervals are narrow; shared ID images make naive SDs 1.4–1.7× too small | [bootstrap](report/paired-bootstrap/REPORT.md), [dependence](report/bootstrap-dependence/REPORT.md) |
| 7 | Validation isolation | Did reusing validation data bias results? | Retrained with disjoint roles; Energy macro AUROC moves from 81.0% to 80.8% | [isolation](report/validation-isolation/REPORT.md) |
| 8 | New OOD datasets | Do results hold on unseen data? | Tiny ImageNet (pre-declared), Fashion-MNIST, Places365 and Oxford Pets, all six classic scores | [Tiny ImageNet](report/tinyimagenet-confirmation/REPORT.md), [Fashion](report/fashionmnist-extension/REPORT.md), [Places365](report/places365-extension/REPORT.md), [Pets](report/oxford-pets-extension/REPORT.md) |
| 9 | Class holdout | How hard is purely semantic OOD? | About 75% AUROC; 82–85% of novel classes accepted at 95% | [one partition](report/cifar100-class-holdout/SUMMARY.md), [three partitions](report/cifar100-class-holdout-replication/SUMMARY.md) |
| 10 | **ASH-S / RMD confirmation** | Do the exploratory winners replicate? | ASH-S > Energy **not confirmed**; RMD > Mahalanobis **confirmed** | [confirmation](report/posthoc-confirmation/REPORT.md) |
| 11 | **LogitNorm** | Does changing the loss give a better representation? | MSP and KNN gains **not confirmed**; gains limited to grayscale sources | [LogitNorm](report/logitnorm-representation/REPORT.md) |

Descriptive follow-ups break the frozen results down without selecting anything: per-class results ([Tiny ImageNet](report/tinyimagenet-class-heterogeneity/REPORT.md), [held-out classes](report/cifar100-class-holdout-classes/REPORT.md)), per-scene and per-breed results ([Places365](report/places365-scene-diagnostic/REPORT.md), [Pets](report/oxford-pets-breed-diagnostic/REPORT.md)), and per-sample decision changes ([geometry](report/geometry-diagnostics/REPORT.md), [RMD](report/rmd-diagnostics/REPORT.md), [ASH](report/ash-diagnostics/REPORT.md)).

## Reproducing

The experiments used Python 3.10, PyTorch 2.13 with CUDA 12.6, torchvision 0.28 and NumPy 2.2 on one RTX 4080 Laptop GPU. A 100-epoch ResNet-18 takes about 17 minutes. The analysis scripts also need SciPy, scikit-learn and Matplotlib.

```bash
pip install -e . scipy scikit-learn matplotlib
python -m unittest discover -s tests -v

# CPU smoke test on synthetic data
python train.py --dataset synthetic --model tiny --epochs 3 --device cpu --output-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/best.pt --ood noise constant --device cpu --output-dir runs/smoke/evaluation

# One full CIFAR-100 model and its baseline evaluation
python train.py --download --amp --epochs 100 --seed 0 --output-dir runs/cifar100-seed0
python tools/prepare_svhn.py && python tools/prepare_textures.py   # verified SVHN / DTD caches
python evaluate.py --checkpoint runs/cifar100-seed0/best.pt --ood cifar10 mnist svhn textures --download --output-dir runs/cifar100-seed0/eval
```

Training can be paused with `--stop-after N` and continued exactly with `--resume`. The resume restores optimizer, scheduler, scaler and all RNG states. Each study's protocol in `docs/` lists its scripts in `tools/`. Datasets, checkpoints and raw per-image outputs (`data/`, `runs/`, about 13 GB) are not included in the repository.

The experiments were first run on a Windows workstation. Before release, machine-specific absolute paths in documents and records were replaced with repository-relative paths (or `python` for interpreter paths), and PowerShell commands with POSIX shell equivalents. No numbers changed. Hash references in the completion manifests were updated in dependency order. The old and new SHA-256 of every affected file are listed in [RELEASE_PATH_NORMALIZATION.json](docs/RELEASE_PATH_NORMALIZATION.json), and the original files remain in the git history.

## Repository layout

```text
src/ood/          data loading, models, training, scores, metrics, evaluation
train.py          ID-only classifier training (exact resume)
evaluate.py       score a frozen classifier on ID and OOD data
run_matrix.py     three-seed training/evaluation controller
tools/            per-study scripts:
                    prepare_* / extract_* / score_*   build inputs and scores
                    audit_*                           independent recomputation
                    build_* / render_* / plot_*       reports and figures
                    freeze_* / verify_* / complete_*  pre-registration and completion checks
tests/            unit tests (metrics, scores, protocol guards)
docs/             one protocol per study, study log, concept guide
report/           one folder per study: REPORT.md, figures, summary.json, audit, completion.json
```

## Limitations

- One ID dataset, one architecture (ResNet-18 at 32×32) and three seeds per condition. There are no ImageNet-scale or pre-trained models.
- The four development datasets were seen while methods were being added, so comparisons on them are exploratory. Only the pre-registered studies are confirmatory.
- This is not an OpenOOD reproduction. Splits, preprocessing and models differ, so the numbers are not leaderboard-comparable.
- LogitNorm used a single τ, and the ID-only checkpoint rule picked epochs 54–57, against 92–98 for cross-entropy. The original paper reports almost no CIFAR-100 accuracy cost; here it was 2.0 pp.
- The low-level-cue explanation in finding 2 is a hypothesis drawn from the pattern of results. It has not yet been tested by a controlled manipulation.

## Next step

The next experiment is a controlled 2×2 test on the same-source models: held-out classes versus ID classes, each with and without an added low-level cue (grayscale, or downsampling from 32 to 16 and back to 32). If the ASH-S and LogitNorm gains follow the cue rather than the novel classes, the hypothesis in finding 2 is supported.

## References

- Hendrycks & Gimpel, [MSP baseline](https://arxiv.org/abs/1610.02136), ICLR 2017
- Liu et al., [Energy-based OOD detection](https://arxiv.org/abs/2010.03759), NeurIPS 2020
- Lee et al., [Mahalanobis detector](https://arxiv.org/abs/1807.03888), NeurIPS 2018
- Sun et al., [ReAct](https://arxiv.org/abs/2111.12797), NeurIPS 2021
- Ren et al., [Relative Mahalanobis distance](https://arxiv.org/abs/2106.09022), 2021
- Sun et al., [Deep nearest neighbours (KNN)](https://proceedings.mlr.press/v162/sun22d.html), ICML 2022
- Wang et al., [ViM](https://arxiv.org/abs/2203.10807), CVPR 2022
- Wei et al., [LogitNorm](https://arxiv.org/abs/2205.09310), ICML 2022
- Djurisic et al., [ASH](https://arxiv.org/abs/2209.09858), ICLR 2023
- Yang et al., [Semantically coherent OOD detection (SC-OOD)](https://openaccess.thecvf.com/content/ICCV2021/html/Yang_Semantically_Coherent_Out-of-Distribution_Detection_ICCV_2021_paper.html), ICCV 2021
- Vaze et al., [Open-set recognition: a good closed-set classifier is all you need?](https://arxiv.org/abs/2110.06207), ICLR 2022
- Yang et al., [OpenOOD](https://arxiv.org/abs/2210.07242), NeurIPS 2022; Zhang et al., [OpenOOD v1.5](https://arxiv.org/abs/2306.09301), 2023

This project was developed with the help of AI coding assistants.
