## Interpretation

**Neither hypothesis is confirmed.** LogitNorm improved MSP AUROC on all three seeds for four of eight populations: MNIST, Tiny ImageNet, Fashion-MNIST and Places365. It worsened MSP on all three seeds for three: CIFAR-10 (−3.50 pp), Textures (−7.63 pp) and Oxford Pets (−2.45 pp). SVHN was mixed. For KNN-50, which reads only the penultimate features, LogitNorm helped on MNIST and Fashion-MNIST and was worse on all three seeds for five populations. Under this protocol LogitNorm did not produce a representation that separates OOD better in general. It also cost 2.00 ± 0.48 pp ID-test accuracy.

**Where it helped, it helped a lot, and only on grayscale sources.** At the fixed 95% ID-validation threshold:

| Population | Lowest OOD acceptance, CE | Lowest OOD acceptance, LogitNorm |
|---|---:|---:|
| Fashion-MNIST | 54.0% (KNN-50) | 14.8% (Entropy) |
| MNIST | 86.7% (MSP) | 36.0% (Entropy; ±23 pp across seeds for MSP) |
| CIFAR-10 | 81.7% | 85.3% |
| Tiny ImageNet | 77.9% | 79.1% |
| Places365 | 81.6% | 82.3% |
| Oxford Pets | 75.6% | 83.9% |

On the four natural-photo populations, no method under either objective gets below about 76% OOD acceptance. Pooled Mahalanobis degraded sharply under LogitNorm, by −5 to −50 pp AUROC, so the shared-covariance Gaussian model fits these features worse.

**Post-hoc observation (not part of any decision).** This is the second intervention with the same split. ASH-S ([confirmation report](../posthoc-confirmation/REPORT.md)) and now LogitNorm both help on populations with an obvious low-level source difference, such as grayscale digits and clothing. Neither helps on object-centric natural photos. The accumulated evidence suggests the attainable gains here come from low-level image statistics rather than semantic novelty. That is a hypothesis, not a result. The same-source CIFAR-100 class-holdout protocol removes the low-level cue, so it is the natural place to test it.

**Two confounds limit the attribution to the loss alone.**

- **Checkpoint epoch.** LogitNorm's ID checkpoint-validation accuracy peaked just after the first learning-rate drop and then declined; for seed 0 it went from 75.65% at epoch 57 to 74.85% at epoch 100. The predeclared ID-only rule therefore selected epochs 57/55/54, against 92/96/98 for CE. The comparison is between objectives *as selected by the frozen rule*, not at equal training length.
- **τ.** τ = 0.04 is a single reference setting, not tuned for CIFAR-100 or ResNet-18. A different τ could behave differently. Choosing τ from these results would be test-set tuning; a τ study would need its own ID-only criterion or a fresh OOD population.

**Limits.** Three seeds per objective, one architecture, 100 epochs. The CE and LogitNorm models were trained on different operating systems with the same library versions. All eight populations had been used to evaluate other methods, though never for this contrast.
