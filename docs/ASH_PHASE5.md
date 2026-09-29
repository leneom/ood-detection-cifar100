# Fixed ASH-P/B/S extension on frozen pooled features

Declared after complete, audited RMD and its post-result diagnostics, before any new ASH feature shaping or score calculation. This is a local exploratory literature baseline; earlier OOD results are already known. It does not reproduce the authors' DenseNet/ImageNet training, full dataset suite or parameter search, and makes no confirmatory or deployment claim.

## Readonly reference and one fixed setting

Reference: [authors' repository](https://github.com/andrijazz/ash), commit `0528fd21a32fffde6a1ace08465f1d2879f6afd7`; pinned ash.py SHA-256 `7fe26f9ac1430198bcc7eb167ee08772674d622ac2ca60ca7d209f1eb46abfee`. The [paper](https://arxiv.org/abs/2209.09858v2), Algorithms 1-3 and placement paragraph, is pinned as local paper-v2.pdf SHA-256 `f132db44e0dae25108bad07a4ab4bc9f55cf3b16dd86b149ec8428705b2e93b8`. README/code/PDF bytes and URLs are recorded in runs/ash-phase5/reference.json. Downloaded code is inspected but never imported or executed.

Use pruning percentile p=65 for all three variants, matching the inspected function defaults in the pinned author code. This is one declared setting, not the paper's dataset-specific best setting; no 65/85/90/95 comparison or local sweep occurs. Include ASH-P, ASH-B and ASH-S regardless of result. Do not combine them with ReAct, change layer placement or recalibrate temperature.

## Frozen classifier, representation and numerical gates

Use seeds 0/1/2, the same ID-validation-selected original 100-epoch checkpoints, original saved 512D final pooled features, original classifier W/b, and existing official input/row provenance. Reshape conceptually to [batch,512,1,1], after the last average pool and before the frozen linear head. This matches the author's after-pooling placement, with our existing ResNet18 architecture rather than their benchmark models.

CPU float64 throughout new shaping and affine scoring. Require finite nonnegative original features (no silent ReLU, abs, clipping or L2 normalization), original dimension 512, finite W/b, and valid head/provenance checks. No classifier training, detector fitting, train-statistic threshold, validation optimization or OOD-dependent setting occurs.

With d=512 and p=65, k=d-int(np.round(d*p/100))=179, using NumPy nearest-even rounding as in the pinned code. Select exactly k coordinates per row by descending feature value, breaking equal values by ascending original coordinate index. This explicit deterministic tie rule is a local adaptation to torch.topk's unspecified equal-value order. Save the full selected indices/values and cutoff tie counts; do not switch rules after looking at OOD metrics. Do not substitute a quantile threshold whose tie behavior could retain a different number.

For row f, let S1=sum(f), let h be f with unselected coordinates zero, and S2=sum(h). Shape three variants:

- ASH-P: h, unchanged selected values.
- ASH-B: replace the k selected coordinates with S1/k, others zero. Preserve total activation S1 within fixed atol=1e-8/rtol=1e-10.
- ASH-S: h*exp(S1/S2), including the exponential; it is not h*(S1/S2). Require S1>0, S2>0 and finite ratio/scaled outputs. Do not invent an epsilon, truncate a scale or rescue a degenerate row using OOD outcomes. A failure retains the attempt and pauses incompatible scoring.

All query population sums/supports are on-the-fly definitions of the score, not fitted or selected detector settings. Freeze source and input hashes before any calculation. Validate kernels and all training-ID feature constraints independently before OOD scoring: exact support/cardinality, stable ties, sums, ASH-B conservation, ASH-S exponential, original head identity and output finiteness. Use manual sorting/math and independent CPU torch operations; CUDA_VISIBLE_DEVICES=-1 must actually hide CUDA. Independent float64 shaped-logit/score agreement requires atol=1e-8/rtol=1e-10, fixed here before outputs.

## Scores and unchanged correctness

For each variant, z_shaped=W*f_shaped+b, anomaly E=-logsumexp(z_shaped), T=1 and higher_is_ood. This sign follows our original anomaly-score convention; the reference get_energy_score uses positive logsumexp as ID confidence. Never flip sign after results. Save original selected support/values, S1/S2, ratio, exp(ratio), ASH-B fill, per-variant full scoring logits, and raw scores for every row. Sparse support plus retained input features defines the complete shaped feature vector; full dense shaped-feature copies are optional.

Class predictions from shaped logits can change, including differences between P/S in the presence of a bias. For ID validation/test, separately record each intermediate accuracy and original-to-shaped argmax-change counts. Preserve original labels, logits, predictions and correctness. Every primary ID risk/coverage endpoint uses the unchanged original classifier; no shaped prediction replaces it silently.

## Populations, controls and all comparisons

Report separately ID validation 5,000, ID test 10,000, CIFAR-10 10,000, MNIST 10,000, SVHN 26,032, Textures partition-1 test 1,880, and noise 10,000. Noise stays synthetic, outside every natural macro. Use all three original training banks only for ID-only preflight/provenance checks; they do not become evaluation/test populations or determine a new percentile.

Preserve all twelve phase-4 score columns/metrics exactly, including affine Energy and matched CPU MD numerical controls; add three named ASH Energy columns. Thus there are thirteen named methods, two numerical controls and fifteen columns total. All three variants are reported in P/B/S order, never selecting a winning variant.

For every column use existing AUROC, AP-out, AP-in, benchmark FPR95-ID (with achieved ID TPR/tie counts), explicit ID/OOD n and prevalence, unchanged primary 2,500 ID-validation threshold rows and separate secondary full-5,000 validation thresholds. Primary rows were previously used in checkpoint selection; no fresh-holdout or conformal claim. Reuse all four fixed ID-calibration targets [0.80,0.90,0.95,0.99] and save raw ID/validation/calibration/OOD scores, exact masks and original-correctness ID counts. Distinguish achieved test coverage from calibration targets and benchmark FPR95-ID from operational OOD acceptance.

Save ASH-P/B/S-minus-each-of-the-twelve-existing-reference-columns paired differences for every seed/dataset and equal-weight four-natural-dataset macro. Compute the macro within seed, then mean/sample SD across all three; SD is descriptive, not CI/significance. Retain all signs/poor results. No OOD sweep, checkpoint change, accuracy-based post-result variant selection or promoted best method.

## Completion gates

Require independent preflight mathematical/head audit before OOD shaping/scoring, then full-sample formula/support/intermediate/metric/count/control/hash audit, independent Python paired-summary audit and visually inspected scientific PNG/SVG. Save duplicate locks, source snapshots, failed attempts and status/logs locally; reject incompatible partial reuse. Finish only after all seeds/populations and these gates pass, with the full regression suite. The original project, prior sources/scores and all reference files remain unchanged; no publication/outreach.
