# Predeclared phase 2: ID-only temperature calibration

Declared 2026-09-27 after the complete four-natural-dataset baseline, before any phase-2 temperature fitting. The original baseline, raw arrays, checkpoints and reports remain frozen.

## Question and endpoint

Measure how ID-only probability calibration changes confidence and post-hoc OOD scores on the same classifiers. Primary calibration endpoints are ID-test NLL, multiclass Brier score and ECE with 15 fixed equal-width confidence bins. OOD AUROC/AP/FPR95 and validation-calibrated acceptance are secondary descriptive endpoints; they never choose temperature, bounds, binning, methods or a checkpoint. No improvement in OOD detection is assumed.

Use all predeclared seeds 0, 1, 2, their original ID-validation-selected frozen checkpoints and the same four natural populations: CIFAR-10 test, MNIST test, full SVHN test and DTD official test partition 1. Preserve counts and prevalence. Synthetic noise remains a separate diagnostic and is excluded from the natural macro.

## Fixed calibration split and reuse boundary

Recover the 5,000 original ID-validation labels from MD5-verified CIFAR-100 training files and the saved validation indices, preserving the raw validation-logit row order. Within each of the 100 classes, deterministically shuffle its 50 validation row indices using NumPy default_rng seed 2027; assign the first 25 to temperature fitting and the remaining 25 to threshold calibration. Sort each combined row-index array and save both arrays, labels, counts and SHA-256. The exact same split is used for all three classifiers.

Temperature-fit and threshold-calibration subsets are disjoint, each with 2,500 ID images. Both belong to the original validation set previously used for checkpoint selection. They are not entirely fresh holdouts relative to model selection; report that reuse explicitly and do not claim conformal coverage. ID training and the 10,000 official ID test samples retain their phase-1 roles. OOD test images/labels do not enter fitting or threshold selection.

## Temperature objective and optimizer

Fit one positive scalar T per classifier by minimizing mean temperature-fit ID NLL:

`mean(logsumexp(z / T) - z_y / T)`.

Use saved logits, CPU float64 arithmetic and SciPy bounded scalar minimization in log(T), with fixed bounds T in [0.05, 20], xatol=1e-8, maxiter=200. Record optimizer success, objective, T, boundary proximity and the original T=1 objective. If optimization fails, produces nonfinite values, or increases fitting NLL relative to T=1 by more than 1e-12, retain T=1 and record the reason. These rules use ID fitting data only. No per-OOD temperature or test-selected sweep is allowed.

## Matched controls and scoring

Compare T=1 and the ID-fitted T on identical saved logits and the same threshold-calibration subset. Recompute MSP, Energy and Entropy with identical CPU float64 arithmetic for both controls, preserving all phase-1 score definitions and orientation. MaxLogit is an unchanged-logit control; KNN and Mahalanobis use their already saved scores and never refit features. Their T=1 and calibrated arrays/thresholds must be identical. Positive T must preserve argmax predictions and ID classification accuracy.

This phase's matched classical-score computation is float64. Original float32/GPU baseline scores remain untouched; record numerical differences of the phase-2 T=1 control from the original saved scores and metrics. Do not present those controls as byte-identical replicas of phase 1 or hide tie/precision effects.

Fit each operational 95% ID threshold only on the 2,500 threshold-calibration scores. Apply that fixed threshold to all ID/OOD test populations. Report FPR95-ID benchmark thresholds separately, with achieved ID TPR and ties as in phase 1. The original baseline's 5,000-validation-image operational thresholds remain separately documented; method comparisons in phase 2 use matched 2,500-image controls to avoid a threshold-population confound.

## Saved evidence and interpretation

Save immutable original-result/logit/score/checkpoint hashes, calibration indices and labels, optimizer records, T=1/calibrated scores, ID-test calibration metrics, raw OOD metrics/counts, fitted thresholds and achieved acceptance. Verify ID/OOD/source hashes against the completed baseline and the Textures executed-input audit. Archive analysis code, versions and precision.

Report paired per-seed changes and mean/sample SD across the three classifiers, with equal-weight natural macro computed within each seed. Three seed SD is descriptive, not a confidence interval or a significance claim. Any later sample bootstrap is conditional on the saved classifier and fitted T unless calibration fitting is explicitly resampled. No method or temperature is promoted using OOD test improvements.

Calibration can change confidence without changing classification accuracy. Ranking quality, high-ID-acceptance OOD rejection and ID probability calibration remain distinct measured outcomes. The complete phase-1 baseline remains the reference; this is an ID-only calibration study, not a replacement benchmark or an OpenOOD reproduction.
