# Frozen LogitNorm representation study — 2026-09-29

## Question

Every post-hoc comparison so far used the same cross-entropy (CE) ResNet-18. Within that representation, the usable score functions differ by at most a few AUROC points. At a fixed 95% ID-validation threshold, about 76–85% of natural OOD images are still accepted. The [ASH-S / RMD confirmation](POSTHOC_CONFIRMATION.md) found no score-function change that altered this picture.

This study changes the **training objective** instead, and nothing else. Three new classifiers use LogitNorm (Wei et al., ICML 2022). They are trained under exactly the three-way ID-isolated protocol of the existing CE models ([VALIDATION_ISOLATION](VALIDATION_ISOLATION.md)). The existing CE models are the seed-matched comparators.

## Fixed training change

The only change is the loss:

```
loss = CE( z / (τ · (‖z‖₂ + 1e-7)), y ),  τ = 0.04
```

The loss is computed in float32 on the logits `z` produced under the same AMP autocast as before. τ = 0.04 is the LogitNorm reference setting and is fixed here without tuning. At evaluation time the model's raw, unnormalised logits are scored exactly like the CE models' logits.

Everything else is identical to the CE isolated models and is enforced by the training guard:

- ResNet-18 with a 3×3 stride-1 stem and no max-pool;
- the same 45,000 training indices and split seed 2026;
- SGD at learning rate 0.1, momentum 0.9, weight decay 5e-4, batch size 128;
- 100 epochs with milestones 50/75/90;
- AMP and deterministic algorithms;
- training seeds 0, 1 and 2;
- checkpoint selection by accuracy on the same 2,000 ID checkpoint-role validation rows.

The code lives in `src/ood/logitnorm_train.py`, which reuses the frozen partition guard from `src/ood/isolated_train.py` unchanged. Output goes to `runs/logitnorm-representation/seed{s}`.

Known non-identical factor: the CE models were trained under Windows and these run under WSL2 Linux, with the same torch 2.13.0+cu126, numpy 2.2.6 and GPU. The environment is recorded. Seed-matched pairs share initialisation and data-order seeds, but not bitwise training trajectories.

## Fixed evaluation

For each LogitNorm model:

1. **Extract.** Collect logits and 512-d features for the 45,000 training-ID, 5,000 ID-validation and 10,000 ID-test images, plus eight OOD populations: CIFAR-10, MNIST, SVHN, Textures, Tiny ImageNet, Fashion-MNIST, Places365 and Oxford Pets. Every input must be byte-identical to the CE extractions: the ID and original-four pixel fingerprints recorded in `runs/validation-isolation/extract-seed*/manifest.json`, and the cache SHA-256 recorded for each later population.
2. **Score.** Use the six original T=1 methods with the original code and settings: MSP, Energy, MaxLogit, Entropy, KNN-50 (normalised features, fitted on the 45,000 training features) and pooled Mahalanobis (1% shrinkage).
3. **Set thresholds.** Operating targets 80/90/95/99% use the order-statistic threshold from the frozen 1,500-row ID threshold role.

CE comparator values are the existing saved CE scores for the same rows. There is no temperature variant, no new method and no OOD-based choice.

## Hypotheses and decision rule

For population P and seed s, Δ(s, P) = AUROC(LogitNorm model s) − AUROC(CE model s), for the same method and the same ID/OOD rows.

- **H3 (output level):** Δ > 0 for **MSP**. This is LogitNorm's headline claim.
- **H4 (feature level):** Δ > 0 for **KNN-50**. KNN uses only the penultimate features, so H4 asks whether the representation itself separates OOD better, rather than only the logit geometry.

All eight populations are primary. The LogitNorm-versus-CE contrast has never been observed on any of them, and none influenced the choice of LogitNorm or τ.

The sign rule is the same as in [POSTHOC_CONFIRMATION](POSTHOC_CONFIRMATION.md):

- **Per population:** supported if all three seed differences are positive; contradicted if all are negative; mixed otherwise.
- **Across the eight populations:** confirmed if supported on all eight; partially confirmed if supported on one to seven and contradicted on none; not confirmed otherwise.

## Secondary, descriptive only

- **ID cost:** ID-test accuracy difference, seed-matched.
- **Other methods:** AUROC, AUPR-out/in and FPR95-ID differences for the other four methods.
- **Operating points:** fixed-95% OOD acceptance and achieved ID-test acceptance for all six methods under both objectives.
- **Absolute check:** the lowest fixed-95% OOD acceptance among the six methods for each population, under each objective. This asks whether any method drops below the ~76% floor seen with CE.

## Pre-freeze engineering checks (ID training data only)

Before this document was frozen, two checks were run:

- **Synthetic set.** On the 4-class, 160-image synthetic smoke set, ResNet-18 with LogitNorm stayed at chance for 40 epochs (loss ≈ 12), while CE fit the set within 5 epochs. The loss matches the reference formula `CE(z/(‖z‖+1e-7)/τ)` exactly and is invariant to scaling z. The failure was therefore treated as a property of that toy problem, not a code error.
- **Real schedule.** The first 3 epochs of the real seed-0 schedule were run into a scratch directory, then discarded. ID checkpoint-role validation accuracy rose 8.1 → 15.5 → 19.8%, against 12.9 → 22.9 → 27.6% for the CE seed-0 history.

No OOD row, and no ID-test row, was used in either check.

## Failure handling

If a training run produces a non-finite loss, or fails for an engineering reason, the failure is recorded and the run is resumed or restarted unchanged. τ, the learning rate and every other setting stay fixed. A seed that cannot be completed is reported as missing; then a hypothesis cannot be "confirmed".

## Boundaries

The eight populations were previously used to evaluate other methods. The comparison has three seeds per objective and one architecture. τ is a single published setting, not a sweep. The class-holdout protocol is not included; it would need new 80-class LogitNorm models.
