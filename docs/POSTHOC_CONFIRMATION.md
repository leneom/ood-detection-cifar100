# Frozen ASH-S / RMD contrast confirmation — 2026-09-29

## Why this test exists

ASH-S and RMD are the two post-hoc methods that looked better than their references during exploratory work on the **original** three classifiers and the **original** four natural OOD populations:

- [ASH phase 5](ASH_PHASE5.md): ASH-S minus T=1 Energy, same-seed natural-macro AUROC +2.18 ± 0.78 pp (CIFAR-10 −0.01, MNIST +3.39, SVHN +2.04, Textures +3.31 pp).
- [RMD phase 4](RMD_PHASE4.md): RMD minus matched CPU Mahalanobis, natural-macro AUROC +18.43 ± 1.72 pp. RMD minus matched MD was also seen to be large on the nine [class-holdout](CLASS_HOLDOUT_RMD_EXTENSION.md) models (74.29 vs 53.16%).

Both methods were adopted after viewing those OOD results. Every later population — Tiny ImageNet, Fashion-MNIST, Places365 and Oxford-IIIT Pet — was scored only with the six original methods, using the three [ID-isolated](VALIDATION_ISOLATION.md) classifiers. A repository search before this document was written found no ASH or RMD score computed with the isolated classifiers or on any of those four populations.

This study therefore tests two predeclared contrasts on **method × population pairs that have never been observed**. It is not an untouched-dataset confirmation: all four populations were previously scored with other methods.

## Fixed inputs

The three isolated classifiers (seeds 0/1/2, selected epochs 92/96/98) and their saved outputs are reused unchanged. The exact files and SHA-256 values are bound in `runs/posthoc-confirmation/declaration.json`, written together with this document and before any scoring.

- Classifier head: `fc.weight`/`fc.bias` from `runs/validation-isolation/seed{s}/final-best.pt`.
- Training ID: the 45,000 saved optimisation-ID features and labels, `runs/validation-isolation/extract-seed{s}/id_train_raw.npz`.
- ID validation: the 1,500-row `threshold_rows` role from `runs/validation-isolation/partition.npz`, applied to `id_validation_raw.npz`.
- ID test: all 10,000 official CIFAR-100 test images, `id_test_raw.npz`.
- OOD: the saved `raw.npz` features and logits of each population below.
- References: the saved T=1 Energy, pooled Mahalanobis and KNN-50 scores for the same rows.

An ID-test-only reconstruction check was run before writing this document, to set tolerances: `features @ W.T + b` reproduced the saved logits with maximum absolute difference 4.8e-6 and identical argmax, and matched CPU MD reproduced the saved pooled-Mahalanobis ID-test scores to relative difference 2.5e-14. No OOD row was used.

## Fixed methods

No parameter is tuned. Scores are higher-is-OOD.

- **ASH-S** (primary): `tools/ash_core.py` unchanged. Percentile 65, so 179 of 512 coordinates are retained (stable ties by lower index). Shaped feature `h·exp(S1/S2)`, passed through the original head, then `−logsumexp` in CPU float64 (T=1).
- **ASH-P, ASH-B**: by-products of the same function; reported only in the complete tables as secondary.
- **RMD** (primary): `tools/rmd_core.py` unchanged. Class and background Gaussians fitted only on the 45,000 training-ID features and labels, population covariance denominator N, fixed 1% trace shrinkage. `R = M − B`.
- **Matched CPU MD**: the `M` term of the same fit. It must reproduce the saved pooled Mahalanobis score within relative 1e-9 on every ID and OOD row, or the run stops.
- **References**: saved T=1 Energy (also recomputed from logits within 2e-5), pooled Mahalanobis, KNN-50.

## Populations

The ID reference is always the 10,000-image CIFAR-100 test set.

| Role | Population | OOD rows |
|---|---|---:|
| Primary | Tiny ImageNet-200 official validation | 10,000 |
| Primary | Fashion-MNIST official test | 10,000 |
| Primary | Places365-Standard official validation | 36,500 |
| Primary | Oxford-IIIT Pet official test | 3,669 |
| Secondary | CIFAR-10 test, MNIST test, SVHN test, Textures | 10,000 / 10,000 / 26,032 / 1,880 |

The secondary populations were used to develop both methods. Scoring them with the isolated classifiers is a **model replication** of the exploratory result, labelled as such, and it cannot support or refute the hypotheses. Synthetic noise and the 80-class class-holdout models are out of scope.

## Hypotheses and decision rule

For each classifier seed s and population P, the paired difference is computed on the same model and the same rows:

- **H1:** Δ₁(s, P) = AUROC(ASH-S) − AUROC(T=1 Energy) > 0
- **H2:** Δ₂(s, P) = AUROC(RMD) − AUROC(matched CPU MD) > 0

AUROC uses the project metric (`src/ood/metrics.py`, OOD positive, half credit for ties).

A hypothesis is **supported on P** only if Δ(s, P) > 0 for **all three** seeds. It is **contradicted on P** only if Δ(s, P) < 0 for all three seeds. Otherwise it is **mixed on P**.

Across the four primary populations, a hypothesis is:

- **confirmed** if supported on all four;
- **partially confirmed** if supported on one to three and contradicted on none;
- **not confirmed** otherwise.

Mean ± sample SD over seeds is reported for every Δ, but it does not enter the decision. No p-values are computed: with three training seeds, sign consistency is the predeclared evidence standard. The exploratory effect sizes above are the prior expectation; a positive but much smaller effect is reported as smaller, never re-described as confirmed magnitude.

## Secondary, descriptive only

These outputs are declared now and carry no decision:

- **Operating points.** For every computed method, targets 80/90/95/99% use the order-statistic threshold from the 1,500 ID threshold-role rows (`id_threshold`). Report the achieved ID-test acceptance and the OOD acceptance. Main contrasts are ASH-S − Energy and RMD − MD at fixed 95%. Methods keep their own thresholds, so these are not equal-coverage comparisons.
- **Other AUROC contrasts.** ASH-S − KNN-50, RMD − KNN-50 and RMD − Energy.
- **Full metrics.** AUROC, AUPR-out, AUPR-in and FPR95-ID for ASH-P/B/S, RMD, matched MD and the three references, on all eight populations.
- **ID side effects.** ASH intermediate head accuracy and the count of changed ID-test argmax predictions. Retained-error reporting always uses the original classifier.

## Boundaries

- No training, checkpoint, threshold, temperature, percentile, shrinkage or score direction is changed or fitted on OOD data.
- If a run fails, the failure and its cause are recorded and the declared computation is re-executed unchanged after an engineering fix. Settings are never altered in response to results.
- Prior test exposure of these populations, the 32×32 resize of higher-resolution sources and Fashion-MNIST's grayscale source remain limitations. A confirmed contrast here means "the direction replicated on four previously unobserved method-population pairs with new models". It does not mean the method is generally superior.

## Procedure

1. Write this document, `tools/score_posthoc_confirmation.py` and the declaration (hashes of document, sources and inputs) before any ASH or RMD computation on the isolated classifiers.
2. Score each seed on CPU with CUDA hidden. Save raw scores, masks and a manifest; guards stop the run on any hash or tolerance failure.
3. An independent CPU auditor reimplements ASH-S (lexsort top-k, SciPy logsumexp), RMD (explicit covariances and Cholesky solves) and AUROC (scikit-learn). It recomputes every primary Δ and every decision.
4. Build the report, run the regression suite and write a completion manifest.
