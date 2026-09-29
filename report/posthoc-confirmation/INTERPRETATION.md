## Interpretation

**H2 is confirmed, and it is the less informative of the two.** RMD beats matched MD by +19.9 to +27.6 pp AUROC on every primary population and every seed. At fixed 95% it also accepts 6–28 pp fewer OOD images, while ID-test acceptance barely moves (+0.24 ± 0.55 pp). The background subtraction reliably repairs pooled Mahalanobis on these classifiers. The same contrast was already large on the original four populations and on the class-holdout models, however, so this outcome was expected. The more useful comparison is against the simple baselines. There, RMD − Energy is +0.00 ± 0.43 pp on Tiny ImageNet, −2.35 ± 1.20 on Fashion-MNIST, +3.41 ± 1.00 on Places365 and −3.05 ± 0.32 on Oxford Pets. **"RMD fixes MD" does not mean "RMD is a better detector than Energy or KNN".**

**H1 is not confirmed.** ASH-S − Energy is:

| Population | Difference (pp) | Sign pattern |
|---|---:|---|
| Tiny ImageNet | −0.24 ± 0.04 | contradicted (all three seeds negative) |
| Fashion-MNIST | +1.23 ± 0.39 | supported |
| Places365 | +1.09 ± 0.12 | supported |
| Oxford Pets | +0.22 ± 0.21 | mixed |

The exploratory macro gain of +2.18 pp does not transfer uniformly. Where it does appear, it is about half that size. At fixed 95%, ASH-S lowers OOD acceptance clearly only on Fashion-MNIST (−8.94 ± 3.20 pp). On Tiny ImageNet and Pets the difference is within seed variation. The model replication on the development populations reproduces the original pattern with the new classifiers: CIFAR-10 −0.00, MNIST +3.45, SVHN +2.11, Textures +3.29 pp.

**Post-hoc observation (not part of any decision).** ASH-S gains appear where the OOD source differs from CIFAR in low-level statistics: grayscale Fashion-MNIST and MNIST, SVHN digits, Textures, Places365 scenes. The gain shrinks to at most +0.22 pp on the three object-centric natural-photo populations: Tiny ImageNet (−0.24), CIFAR-10 (−0.00) and Oxford Pets (+0.22). That pattern was noticed only after scoring. It is a candidate hypothesis, not a finding. A direct test would predeclare ASH-S ≤ Energy on the same-source class-holdout models, where image statistics are held fixed.

**Big picture.** No method in this test brings fixed-95% OOD acceptance below about 76% on Tiny ImageNet, Places365 or Oxford Pets. Among Energy, KNN-50, ASH-P, ASH-S and RMD, the AUROC spread on a population is 1.5 pp on Tiny ImageNet and at most 5.0 pp (Oxford Pets). Only pooled/matched MD and ASH-B fall far below that band. This agrees with the project-level reading that the representation, not the score function, limits detection here.

**Limits.** All four primary populations had been scored with other methods. Three training seeds make the sign rule a coarse evidence standard. Every population is a single dataset with its own resize and source artefacts. The class-holdout models were not tested.
