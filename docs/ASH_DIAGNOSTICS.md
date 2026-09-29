# ASH post-result support, scale and decision diagnostics

This is a post-result descriptive phase following the complete frozen p=65 ASH-P/B/S experiment. Earlier detection results are known. Bind the completed phase artifacts and this scope before computing new diagnostic moments or transitions. Do not tune a percentile, combine variants, select a winner, refit a detector, recompute a threshold or change original correctness. All work is local.

## Fixed populations and saved inputs

Use seeds 0/1/2, all seven existing scoring populations: validation ID (5,000), test ID (10,000), CIFAR-10 (10,000), MNIST (10,000), SVHN (26,032), Textures partition-1 test (1,880), and noise (10,000). Noise never enters a natural macro. The training banks remain preflight/provenance inputs, not an additional test population. Use frozen ASH intermediates, raw scores, original labels/head, unchanged phase-4 references and already saved masks. No new feature extraction or score/threshold calculation.

## Full support/scale descriptions

Save original sum S1, retained sum S2, retained fraction q=S2/S1, dropped fraction 1-q, ratio r=S1/S2, exp(r), ASH-B fill S1/179, cutoff value and cutoff tie counts. Verify q*r=1 and S1=S2/q with fixed atol=1e-8/rtol=1e-10. Preserve every row, row indices, finite values and signed quantities; do not epsilon-rescue or clip.

Save the already frozen P/B/S Energy and their signed differences from the frozen affine Energy numerical control. Audit the saved logits identity z_S=exp(r)*z_P+(1-exp(r))*b, retaining the original classifier bias; compare with an independently reconstructed head without replacing saved scores. Report population mean, population SD (ddof=0), minimum/maximum and fixed linear quantiles [0.01,0.05,0.25,0.5,0.75,0.95,0.99]. Save covariance/Pearson between retained fraction and each signed Energy change; zero variance makes Pearson undefined, never silently zero. Report every natural OOD-minus-test-ID mean difference per seed and three-seed mean/sample SD. These identities/correlations describe saved data and do not explain causality or imply AUROC from means.

For plotting, retain up to 1,000 uniformly sampled rows without replacement for each seed/population, RNG base 7859 plus seed; save sorted sampled indices. Plot all seven populations and all three variants in fixed order, with all-point-independent bounds. No chosen percentile or selected population subset.

## Separate intermediate classification transitions

For both test ID and full validation ID, preserve original and all three shaped predictions/labels. Save exhaustive four-state correctness codes: 0 both original/shaped correct, 1 shaped only correct, 2 original only correct, 3 neither correct. Also save original-to-shaped argmax changes and P-versus-S disagreement codes. Per-class correctness cells use all 100 original classes with explicit original correct/incorrect and total denominators. Validation was previously used for checkpoint/temperature/threshold work; these are descriptive reused-validation diagnostics. OOD has no classifier-correctness label. None of these shaped predictions replaces the original classifier correctness for rejection risk.

## Fixed acceptance transitions

Compare each of ASH-P/B/S against every one of the twelve old reference columns in protocol order: MSP, Energy, MaxLogit, Entropy, KNN, Mahalanobis, ViM, Residual, ReAct Energy, affine Energy numeric control, RMD, matched CPU MD numeric control. Reuse all four declared ID-calibration targets [0.80,0.90,0.95,0.99] and saved masks exactly. There are 3 variants * 12 references * 4 targets * 3 seeds = 432 settings, with all six test populations per setting. No winning reference/variant or favorable target selection.

Save exhaustive uint8 state codes 0 both accept, 1 ASH only accepts, 2 reference only accepts, 3 neither accepts. Preserve counts and fractions with each population's full denominator. Net OOD acceptance change equals (ASH-only minus reference-only)/n; positive is more OOD accepted. For test ID, additionally save original-correct/incorrect cells with their explicit subgroup denominators and 100-class cells. Save original ID coverage and retained-error-risk paired changes separately: equal calibration targets need not mean equal test coverage. Equal-weight four-natural-population macro is computed within seed before cross-seed sample SD; noise is always separate.

## Completion

Freeze source/inputs before execution and preserve any failed attempts. Require all seeds/populations/settings, independent full-vector formula/manual moments/quantiles/Counter audit, independent Python statistics and displayed-table audit, visually inspected scientific PNG/SVG, and the full regression suite. Original training/checkpoints/previous sources/raw artifacts remain unchanged. This is diagnostic evidence, not a significance, confirmatory, deployment or causal claim.
