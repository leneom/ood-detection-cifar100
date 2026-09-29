# Frozen RMD component and decision diagnostics

This is a fixed post-result descriptive scope. Execute only after all three phase-4 fits, raw evaluation, paired summaries, visual QA and completion gates have passed. RMD outcomes and earlier method outcomes may already be known; this analysis is neither confirmation nor a tuning stage. Bind all completed input/audit/source hashes before calculating these new diagnostic summaries.

## Populations and unchanged inputs

Use seeds 0/1/2 and existing phase-4 intermediates, raw scores and fixed-grid decisions only. Report validation ID 5,000; test ID 10,000; CIFAR-10 10,000; MNIST 10,000; SVHN 26,032; Textures 1,880; noise 10,000, in that order. Noise remains synthetic and excluded from natural macros. Do not compute a new training population, fit a detector, recalibrate any threshold, change the coefficient or reverse the signed score. Original ID classifier correctness and labels are unchanged. The previously reused validation is not a fresh holdout.

## Signed component descriptions

Let M be the saved matched CPU minimum class distance, B the saved background distance, and R=M-B the saved signed RMD. Preserve full M/B/R vectors and original row indices. Independently verify R=M-B at atol=1e-8/rtol=1e-10; keep negative R without clipping. For each population describe n, mean, population SD (ddof=0), min/max, and fixed linear quantiles [0.01,0.05,0.25,0.50,0.75,0.95,0.99].

Report population covariance/Pearson correlation of M and B, with denominator n and null correlation for zero variance. Independently verify Var(R)=Var(M)+Var(B)-2Cov(M,B), atol=1e-8/rtol=1e-10. For each OOD-minus-test-ID mean change report delta M, delta B and delta R=delta M-delta B; also report validation-minus-test-ID. These are signed descriptions, not AUROC, evidence of causal representation changes or grounds to alter the fitted score.

## All fixed decision comparisons

Compare RMD with all eleven other frozen columns, in protocol order: MSP, Energy, MaxLogit, Entropy, KNN, original Mahalanobis, ViM, Residual, ReAct Energy, affine Energy numeric control, matched CPU Mahalanobis numeric control. Use every saved ID-calibration target [0.80,0.90,0.95,0.99], totaling 44 settings per seed, 132 overall. No comparison or operating target is selected for a favorable result.

For test ID and all five OOD populations save exhaustive per-row uint8 transition codes: 0 both accept, 1 RMD only accepts, 2 reference only accepts, 3 neither accepts. Save exact four-cell counts/fractions and signed acceptance change (cell1-cell2)/n. Split ID cells by original correct/incorrect predictions and by all 100 original classes, keeping explicit denominator counts; OOD has no classifier-correctness endpoint. Use identical saved decisions, including their different method-specific thresholds. Report achieved ID-coverage differences; this is not an exactly equal test-coverage comparison or an isolation of ranking alone.

## Evidence and illustrations

All statistics use all saved rows. A joint M/B scatter may use at most 1,000 rows per population, sampled without replacement with default_rng(6829+100*seed+population_ordinal), sorted row indices. Save indices, label subsampling explicitly and derive axis bounds from the plotted samples, without selecting examples by favorable detection. A transition figure reports all four natural datasets and eleven references at the already fixed 95% target, with every seed and descriptive sample SD; all four targets stay in the raw/summary output. CDFs, if included, use all rows.

Independently reconstruct moments with Python math/statistics, quantiles by sorting/manual linear interpolation, transition cells by Python counts, and per-class ID cells from original labels. Verify same-seed differences before mean/sample SD over the three seeds (ddof=1); SD is descriptive, not CI/significance. Save input/source hashes, raw arrays/indices, audits, tables and PNG/SVG; visually inspect figures before marking complete. No interpretation authorizes OOD test selection or modification of any previous protocol, source, checkpoint, score or threshold.
