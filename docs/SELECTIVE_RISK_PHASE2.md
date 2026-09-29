# Predeclared follow-up: fixed ID-only acceptance targets

Declared after the completed, independently audited phase-2 temperature-calibration study and before computing the new operating-point grid. This is an analysis of frozen outputs, with no additional training, feature fitting or temperature optimization.

## Question

Does the observed improvement in ID probability calibration also change classification-error rejection and OOD acceptance at fixed ID-calibrated operating points? Classification accuracy, retained ID error risk, achieved ID coverage and OOD acceptance remain distinct endpoints. No target or method is chosen using OOD results.

## Fixed populations and settings

Use all three frozen classifiers, seeds 0, 1, 2. Reuse their audited phase-2 T=1 and ID-fitted-temperature scores for all six methods: MSP, Energy, MaxLogit, Entropy, KNN, Mahalanobis. Preserve canonical ID array choices and CPU float64 classical-score precision from phase 2; feature and unchanged-logit controls must remain identical across variants.

Fix target empirical ID acceptance to exactly the grid **0.80, 0.90, 0.95, 0.99** for every method and variant. Use only the previously saved 2,500 threshold-calibration ID-validation rows to choose each threshold: ascending order statistic at index `ceil(target * 2500) - 1`, accepting scores `<= threshold` and preserving ties. The other 2,500 ID temperature-fit rows never select these thresholds. Both validation subsets were previously used in checkpoint selection; they are reused validation, not fresh holdouts, and do not establish coverage guarantees.

Apply each threshold unchanged to 10,000 official CIFAR-100 ID test images and the four natural OOD populations: CIFAR-10 (10,000), MNIST (10,000), SVHN (26,032), DTD test partition 1 (1,880). Retain uniform noise only as a separate synthetic diagnostic excluded from all natural macros.

## Endpoints and evidence

For every seed/method/variant/target, save target, exact threshold, calibration population/count/achieved acceptance and threshold ties. On ID test, save total/accepted/rejected counts, correct/incorrect counts, achieved coverage, retained classification-error risk `incorrect_accepted / accepted`, accepted accuracy, correct-ID rejection fraction and incorrect-ID rejection fraction. If an accepted or class-conditioned denominator is zero, save null rather than inventing a value.

For each OOD population, save accepted/rejected counts, acceptance fraction and exact input-score hashes. OOD labels do not enter threshold selection or any fitting. Keep OOD acceptance distinct from retained ID error risk; do not define classification correctness for foreign OOD labels or combine populations under an assumed deployment prevalence.

Save ID-correctness flags derived only from the original ID test labels and frozen predictions, all source-result/score/split/checkpoint hashes, output arrays/counts, code snapshots and versions. Independently reconstruct ID/OOD decisions, ties and counts from raw saved scores, and verify matched unchanged-logit/feature controls. At the 0.95 target, operating thresholds and acceptance must match the already audited phase-2 records exactly.

Report all grid points, per-seed values, paired calibrated-minus-T=1 changes and mean/sample SD across three seeds. Compute any four-dataset natural macro within each seed before the seed summary. Preserve per-dataset populations; exclude noise. No OOD-selected target, method promotion, significance claim or confidence-interval claim is allowed. Scientific plots should retain seed pairing and distinguish intended calibration acceptance from achieved ID test coverage.
