# Frozen protocol v0.1 — 2026-09-27

## Question and scope

Compare post-hoc OOD scoring on the exact same CIFAR-100 classifier. This is a new static OOD detection study, not a continuation of the class-incremental training experiments. A possible later research question is how feature geometry and calibration explain differences between near- and far-OOD detection.

## Isolation

Training uses 45,000 CIFAR-100 images. Validation uses 5,000 distinct CIFAR-100 training-split images, chosen with a fixed stratified seed 2026. The 10,000 official CIFAR-100 test images never enter optimization or checkpoint selection. OOD images and labels never enter training, checkpoint selection, temperature fitting, or detector fitting in phase 1.

Random crop (32×32 after zero padding 4) and horizontal flip occur only during training. Every evaluation dataset uses the same CIFAR-100 channel normalization. ResNet-18 has a 3×3 stride-1 first convolution and no max pool; weights are randomly initialized. A full training budget is 100 epochs with SGD and 50/75/90 milestones. `--stop-after` pauses that fixed schedule, rather than defining a different short schedule. AMP is an explicitly recorded speed setting; deterministic algorithms remain enabled.

All seeds share the same real-data split; model/augmentation/order RNG varies with training seed. Intended comparison seeds are 0, 1, 2. A single unfinished seed is a pilot, not a multiseed result. Save an epoch-specific copy of a selected checkpoint before evaluating it while training continues, to ensure results and hash refer to identical weights.

## Scores and metrics

Let z be the vector of classifier logits. Higher anomaly scores consistently mean more OOD:

- MSP: 1 − max softmax(z/T)
- Energy: −T logsumexp(z/T)
- MaxLogit: −max z
- Entropy: −Σ p log p, p=softmax(z/T)

Temperature T=1 is fixed in the initial experiment. Post-hoc Energy does not mean energy-based fine-tuning or Outlier Exposure. A temperature sweep on test OOD must remain descriptive; selecting the best test temperature is prohibited.

AUROC treats OOD as positive, integrates the grouped-threshold ROC, and gives tied ID/OOD pairs half credit. AUPR-out is non-interpolated average precision with OOD positive; AUPR-in reverses labels and score signs. AUPR depends on the saved population ratio, so report counts and prevalence with it.

FPR95-ID takes the smallest empirical ID test score threshold that accepts at least 95% of ID. Its value is the fraction of OOD with score ≤ threshold. Ties are accepted together, and the achieved ID TPR is saved. This is a benchmark operating point, not a deployable threshold tuned without test data.

Separately, calibrate the identical order statistic on ID validation scores alone. Report actual ID-test acceptance and OOD acceptance under this fixed validation threshold. It need not achieve exactly 95% ID-test acceptance. This distinction is essential to honest reporting.

## Data and benchmark boundary

CIFAR-100 versus CIFAR-10 is an initial natural-image near-OOD pairing, consistent with the [OpenOOD v1.5 categorization](https://github.com/Jingkang50/OpenOOD/wiki/OpenOOD-v1.5-methods-%26-benchmarks-overview). That reference additionally uses Tiny ImageNet near-OOD and MNIST, SVHN, Textures, Places365 far-OOD. Our first dataset pair and independently trained classifier do not reproduce the full OpenOOD protocol or permit direct leaderboard comparison.

Uniform pixel noise and constant images are synthetic diagnostics only. High detection performance there does not establish natural-OOD performance. Semantic overlap between CIFAR datasets is a benchmark limitation; a future held-out-class protocol must record the semantic definition and class split rather than casually relabeling corruptions as OOD.

## Provenance and resume

Save config, package/GPU versions, actual train/validation indices and hashes, training source hashes, per-epoch history, full resume state, ID-only best checkpoint, raw logits, raw scores, checkpoint SHA-256, and metric JSON. Existing evaluations are not overwritten. Reject resume when training configuration, indices, or source hashes differ. Only deserialize locally generated training checkpoints.

## Predeclared next work

1. Verify synthetic end-to-end GPU run and CPU resume equality.
2. Train a real CIFAR-100 pilot. Report ID accuracy with all detection results.
3. Continue the unchanged 100-epoch schedule; evaluate a frozen final selected checkpoint.
4. Run seeds 1 and 2 on the same split. Report mean and sample SD; separate seed variability from sample bootstrap uncertainty.
5. Add SVHN and a feature-based KNN/Mahalanobis baseline fitted on training ID features only.
6. Build plots and error analysis from saved scores. Choose a focused improvement only after the baseline results, using separate validation information.
