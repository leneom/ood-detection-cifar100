# Predeclared phase 3: training-ID feature geometry

Declared after the complete frozen baseline, ID-only probability calibration and fixed operating-point analysis, before exporting training feature banks or fitting/evaluating any new geometry detector. This is an exploratory extension of the independent local CIFAR protocol; it is not the authors' ImageNet benchmark or an OpenOOD reproduction. Original phase-1/2 outputs remain frozen.

## Question and fixed scope

Measure whether information outside class-logit confidence changes natural OOD detection and operational rejection on the same three CIFAR-100 classifiers. Probability calibration, retained ID error risk and OOD rejection already have separate measured endpoints; no improvement of a geometry method is assumed. No new classifier training or checkpoint selection occurs.

Use all seeds 0, 1, 2; their frozen best-ID-validation checkpoints; all 45,000 original ID training images; and the same official ID test and four natural OOD populations. Uniform noise remains diagnostic and excluded from natural macros. Fit every new detector statistic using training ID only. Validation/test/OOD features never enter its subspace, scale or clipping fit.

## Reference and fixed detector definitions

Reference: [ViM paper](https://arxiv.org/abs/2203.10807) and [authors' benchmark.py at commit dabf9e5b242dbd31c15e09ff12af3d11f009f79c](https://github.com/haoqiwang/vim/blob/dabf9e5b242dbd31c15e09ff12af3d11f009f79c/benchmark.py). The exact inspected file is kept locally under `runs/geometry-phase3/` with SHA-256 `dd69155449d5a73a7ec31c748a7660ed287bf469c4688c6e5d18d1a26c8ee75c`. Third-party benchmark code is a read-only reference, not executed. Our metric/tie conventions and OOD-positive orientation remain the local protocol.

Let f be the original unnormalized final 512-dimensional pooled feature, W/b the frozen 100-class affine head, and z_affine = W f + b computed in CPU float64. No L2 normalization is used in this extension.

- Classifier origin: u = -pinv(W) b, with float64 NumPy pinv rcond=1e-15.
- Fit the training second moment C = mean((f-u)(f-u)^T), without subtracting the empirical training mean. Symmetric float64 eigh gives sorted eigenvectors. Fix principal dimension D=256 by the inspected code's dimension rule for 512-dimensional features; the remaining 256 vectors form residual subspace R. No dimension sweep occurs.
- Residual score: r(f) = norm((f-u) R), Euclidean norm, higher is OOD.
- Train-only scale: alpha = mean(max z_affine_train) / mean(r_train). Require a finite positive residual denominator and positive finite alpha; otherwise fail and retain evidence without changing the score sign or fit rule.
- ViM anomaly score: alpha*r(f) - logsumexp(z_affine), with T=1.
- ReAct: fix one global clipping level c to the 0.99 quantile of all coordinates of training f, NumPy float64 quantile with method='linear'. Upper-cap features at c with no added lower bound, and score -logsumexp(W min(f,c) + b), T=1. No percentile sweep occurs.

Save unmodified affine Energy, -logsumexp(z_affine), as a numerical control alongside the six existing phase-2 T=1 controls. It separates the feature-to-head float64 reconstruction from ReAct/ViM changes. It is not an OOD-selected method or a replacement for the original baseline. The three named additions are ViM, Residual and ReAct Energy; original baseline comparisons retain every method.

## Training-bank extraction and source identity

First confirm no training/evaluation/controller process is active; export one seed at a time using the verified Windows GPU environment. Use the frozen classifier in eval/no-grad mode, original deterministic evaluation preprocessing and original saved training-index order, with no train-time crop/flip or AMP. Do not modify original training/evaluation modules. Save all 45,000 training features/logits/labels/indices plus W/b, checkpoint hash, exact input-pixel hash, code snapshots and package versions. Data and checkpoints remain local and ignored.

Verify the exported bank indices exactly match the original training split, have 450 samples per class and exclude all 5,000 ID-validation indices. Bind raw training pixels to the already independently audited Textures-phase training fingerprint. A fixed first 64 ID-validation rows are only an extraction identity check against saved canonical features/logits, never detector fitting. Their forward values must agree with old saved GPU outputs at rtol=atol=1e-6. Do not relax this rule from OOD results.

For evaluation, reuse canonical original eval-final ID-test/ID-validation features and each original population's saved OOD features. Preserve every row's relationship to its saved logits and labels. Verify Wf+b reconstruction against original logits on every evaluation row with fixed atol=2e-5, rtol=2e-6, record absolute errors and any prediction changes, and fail on violations. Save full reconstructed affine logits and ReAct clipped logits as scoring intermediates. Original classifier predictions define ID correctness/accuracy; clipped intermediate prediction changes and accuracy are separate diagnostics, never substituted silently.

## Thresholds, endpoints and comparisons

Primary operational threshold: the same 2,500 phase-2 threshold-calibration ID-validation rows, accepting at least 95% empirically via ceil(0.95*n)-1 order statistic and score <= threshold. The temperature-fit rows do not choose these thresholds. Secondary full-5,000-validation threshold may be recorded separately for the original phase-1 comparison, never mixed with the primary matched controls. These validation images previously selected checkpoints; report reuse and do not claim fresh-holdout or conformal coverage.

For every new detector and every retained control, save raw ID/validation/OOD scores, benchmark FPR95-ID with achieved ID TPR/ties, AUROC/AP-out/AP-in, counts/prevalence, primary operational threshold and its achieved ID/OOD acceptance. Benchmark ID-test thresholds are descriptive metrics only. Report all four previously fixed ID-calibration targets 80/90/95/99% as secondary ID error/coverage and OOD acceptance using the same rules as `SELECTIVE_RISK_PHASE2.md`; no target is selected using OOD.

Preserve six phase-2 matched T=1 controls exactly. Save paired changes against affine Energy and original matched Energy/MSP separately, per seed and population; all controls and original float32/GPU precision differences remain explicit. Four-natural-dataset macros are computed within seed, then summarized by mean/sample SD across all three seeds. Noise never enters these macros. No test-selected dimension, quantile, scale, temperature, score orientation, target, classifier or promoted best method is allowed.

## Verification and saved evidence

Before fitting, save the protocol fingerprint, reference-file identity, source input/checkpoint/feature/label/split hashes and training-bank metadata. Archive fitted u, eigenspectrum/residual basis, second-moment matrix, alpha, clipping level and training-only sufficient statistics. Reject incompatible reuse and retain failed attempts locally.

Verify orthonormality, symmetric second moment, residual projection identities, affine reconstruction and clipping definitions. Independently cross-check subspace/scaling/clipping and raw scores using separate NumPy/SciPy/sklearn mathematics, then independently verify OOD metrics, every threshold/count, original-control parity and paired seed summaries. Scientific plots retain seed pairing; any sample SD is descriptive, not a confidence interval or significance claim. Save all raw scoring intermediates and update progress after executable work.
