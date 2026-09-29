# Saved-bootstrap dependence diagnostics

Post-result descriptive analysis to run only after the complete audited paired image-bootstrap report. Bind that report, the ten raw blocks and all original/source hashes in a predeclaration before covariance calculation. No new sampling, fitting, threshold, method choice or hypothesis test. Existing results are known; this phase describes the dependence already preserved by the declared resampling scheme.

Use all 1,000 original replicate indices, all fifteen columns including two numerical controls, all three fixed classifier seeds, all five OOD populations (noise separate), all six per-dataset ranking endpoints, four natural ranking macros and the four operational targets. The 66 endpoint definitions must exactly match the complete bootstrap report. Keep all 105 later-minus-earlier pairs; do not select references, metrics or cells by covariance sign.

For each endpoint retain the full 45-by-45 sample covariance and Pearson correlation of the ordered seed-major, method-minor replicate variables, using ddof=1 and common paired image draws. Retain means, marginal variances, ordering and finite counts. Correlation is undefined for zero variance; preserve None in JSON and NaN in raw arrays with an explicit finite mask. Never replace undefined correlations with zero or drop a draw. Any nonfinite replicate invalidates the affected covariance/interval description rather than silently producing a complete-case result.

For each method and every fixed pair at each endpoint, take the three original per-seed replicate metrics or paired differences. Save their 3-by-3 covariance and verify

Var[(X0+X1+X2)/3] = [sum_s Var(Xs) + 2 sum_(s<t) Cov(Xs,Xt)]/9.

Report the observed fixed-three variance, diagonal contribution, cross-seed covariance contribution and signed identity residual. For mathematical context save sum_s Var(Xs)/9 as the counterfactual variance obtained by assuming independence. Label it an algebraic comparison only, not an alternative confidence interval or recommended sampling scheme. Save an SD ratio only when its denominator is strictly positive; otherwise preserve undefined. None of these are training-seed uncertainty estimates because trained models are fixed.

Similarly, for each seed/method, the four natural populations' AUROC/AP-out/AP-in/FPR95 metrics and their OOD acceptance at all four operating targets form a four-component vector. Save full covariance/correlation and verify the natural macro variance identity with denominator 16. Populations have distinct OOD draws but share the same ID draw; macro variance need not equal a diagonal-only approximation. Noise must not enter this vector.

Independent audit reconstructs ordered endpoint vectors directly from raw blocks rather than importing the producer's endpoint helper, recomputes centering/covariance/Pearson/manual decompositions with Python arithmetic/statistics, and validates every field, pair, denominator, undefined mask, scope and source/input hash. Audit every report field and plot coordinate; inspect scientific PNG/SVG. Source/raw/report hashes, full bootstrap completion and appropriate CPU regression gate precede diagnostic completion.

Report dependence as a property of this fixed empirical resampling experiment. Correlation does not establish shared causal failure mechanisms, training-seed generalization, population coverage, multiplicity correction or deployable detector quality. No result-dependent filtering or new method/threshold selection. After this bounded phase, consolidate the project's reproducibility index, evidence/limitations and research-readiness assessment before proposing further training work.
