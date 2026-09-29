# Exploratory phase 4: fixed training-ID relative Mahalanobis

Declared after the completed baseline, probability/operating-grid studies, geometry extension and posthoc component/transition diagnostics, before any RMD fit or scoring. This adds a literature-defined baseline; prior OOD results are already known, so it is an exploratory extension rather than a confirmatory unseen-test study. No new parameter is selected from those results. Original checkpoints/features/scores and all prior protocols remain frozen.

## Reference and local adaptation

Reference: [Ren et al., A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection, arXiv:2106.09022v1](https://arxiv.org/abs/2106.09022v1), Section 2 and Algorithm 1. The paper defines a shared within-class Gaussian distance and subtracts a label-independent training background Gaussian distance. Its published confidence sign is reversed here to retain the existing higher-is-OOD convention. No improvement in this local experiment is assumed.

The versioned paper is retained read-only at `runs/rmd-phase4/paper-v1.pdf`, SHA-256 `f43936322f6c6e46a104cc0411376f60d886985f2ad6d08d3ac0aec5ddc4ef3e`. A complementary [author ViT example](https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection/blob/0cbe0026737b661a0de8f8d554e57d39bf1b13ae/ViT_for_strong_near_OOD_detection.ipynb) is pinned to commit `0cbe0026737b661a0de8f8d554e57d39bf1b13ae`, notebook SHA-256 `015085c36ae45709ccc3746a601c12776cd25be7e8aedfbbd2aff47afdd2dd62`. Neither third-party file is executed. This example uses a different pretrained ViT and sample-covariance conventions; our formulas follow the paper's population denominator and the existing local regularization below. This is not a reproduction of its models or benchmark numbers.

## Fixed population and covariance fit

Use all seeds 0/1/2, their original ID-validation-selected frozen classifiers, and the complete independently audited phase-3 45,000-row training banks. Each bank has 450 rows per original ID class, no validation overlap, 512-dimensional unnormalized pooled features and original class labels. No additional training, GPU feature extraction, L2 normalization, empirical-mean feature preprocessing or sample removal occurs. Gaussian means are fitted as part of the distance definition; feature preprocessing remains unchanged.

For training f_i and original labels y_i, N=45,000, K=100, d=512:

- Class means mu_k = mean(f_i for y_i=k).
- Shared within-class covariance C_w = sum((f_i-mu_yi)(f_i-mu_yi)^T)/N.
- Background mean mu_0 = mean(f_i), ignoring labels.
- Background covariance C_0 = sum((f_i-mu_0)(f_i-mu_0)^T)/N.
- Independently regularize both with the original fixed rho=0.01: C_reg = (1-rho)*C + rho*max(trace(C)/d,1e-12)*I. Each covariance uses its own training-only trace. This explicit local shrinkage adaptation is inherited from the original Mahalanobis baseline, not the paper's claim of hyperparameter-free fitting. No shrinkage, normalization, dimensionality or score-weight sweep occurs.
- Fit and score in CPU float64. Save raw and regularized covariances, means, class counts, inverse matrices, eigenvalues, condition numbers and training/index/checkpoint/source hashes before OOD scoring. Symmetrize each numerical inverse as (P+P.T)/2 after inversion; require positive-definite covariance and inverse identity agreement at atol=rtol=1e-8. Fail rather than changing rho on a singular/nonfinite fit.

Training ID alone determines these statistics. ID validation/test and every OOD population never enter covariance fitting, checkpoint selection or any new score setting. Independently verify class means, chunked covariance, population denominators, trace regularization, SPD and inverse/Cholesky equivalence before scoring.

## Scores, precision controls and raw evidence

For a fixed feature f, q_k = (f-mu_k)^T P_w (f-mu_k), q_0 = (f-mu_0)^T P_0 (f-mu_0). The new RMD anomaly score is min_k(q_k) - q_0. Background subtraction has weight exactly 1. Do not add Gaussian log-determinants, priors, square roots, absolute values, temperature, test-dependent scaling or learned coefficients. Signed RMD can be negative and must not be clipped or flipped after viewing results.

Compute q_k/q_0 in float64. Require all raw squared distances >= -1e-8; any larger negative or nonfinite value fails. Only tiny negative roundoff within that fixed bound may be replaced by zero before the minimum/subtraction, with count and maximum correction recorded. The RMD difference itself is never clipped. Independent Cholesky whitening/direct distances must agree with fixed atol=1e-8/rtol=1e-9; preserve failed attempts without relaxing a bound using OOD outcomes.

Archive every population's full 100-class distance matrix, nearest class index, background distance, matched within-class minimum and signed RMD. Preserve row order and original ID labels/predictions/correctness. Reuse existing canonical ID/validation and each saved OOD feature population, with the exact original checkpoint/input provenance already independently audited. No scoring intermediate replaces original classifier predictions for ID error/coverage endpoints.

Retain all ten phase-3 score columns byte-for-byte: six matched T=1 baselines, ViM, Residual, ReAct and affine Energy. Add one matched CPU-refitted Mahalanobis numerical control from these same banks/covariances to isolate background subtraction. The original Mahalanobis score remains a separate frozen baseline; do not silently replace it with this numerical/fit control. Thus this phase has ten named methods and two numeric controls, twelve total columns. Background q_0 is an intermediate, not an additional selected detector.

## Metrics, thresholds and all comparisons

Use unchanged official ID test 10,000, CIFAR-10 10,000, MNIST 10,000, SVHN 26,032 and DTD partition-1 test 1,880. Noise 10,000 remains separate synthetic diagnostics and is excluded from natural macros.

For all twelve score columns report AUROC/AP-out/AP-in, ID/OOD counts/prevalence, benchmark FPR95-ID with achieved TPR/ties, and primary operational thresholds from the same 2,500 ID-validation threshold rows. Those validation images previously selected checkpoints; no fresh-holdout/conformal guarantee is claimed. Preserve secondary full-5,000 validation thresholds separately. All four previously fixed calibration targets 80/90/95/99% retain original ID-correct/incorrect counts, ID-test coverage, retained error and separate OOD acceptance.

Save exact raw ID/validation/calibration/OOD scores and boolean decisions. Independently verify new formulas, every raw metric/threshold/count, original ten-control score/metric parity and unchanged correctness. Report every fixed column without promotion. Save RMD-minus-each-of-the-eleven-reference-columns differences per seed and dataset; compute the four-natural-dataset macro within seed, then mean/sample SD across all three. Noise never enters a natural macro. SD is descriptive, not CI or significance. Negative/poor results remain reported with the predefined direction.

Capture protocol/reference/source/input hashes before fitting, fitted-state audits before OOD evaluation, then raw/paired summary audits and visually checked scientific PNG/SVG. Reject incompatible reuse, avoid active-process conflicts and keep everything local. A fit alone or unaudited/partial evaluation must not be called a completed RMD benchmark.
