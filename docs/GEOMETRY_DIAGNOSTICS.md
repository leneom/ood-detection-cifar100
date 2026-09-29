# Post-result geometry diagnostics: fixed descriptive scope

Declared after the audited three-seed geometry evaluation, before computing the following population/component and transition summaries. This is a posthoc explanation of already measured differences, not a new benchmark, hypothesis confirmation or parameter-selection stage. The observed Textures versus CIFAR-10/SVHN differences motivate these diagnostics; no causal conclusion is assumed.

## Frozen inputs and scope

Use all seeds 0/1/2 and the exact completed phase-3 banks, fits, original logits/features, scores and fixed-grid masks. Bind the completed raw/summary/fit audits and their source artifacts before analysis. No model training, new data, detector fit, score orientation change, threshold recalibration or hyperparameter sweep occurs. Preserve all prior outputs, including poor Residual results.

Populations are separately reported: training ID 45,000, validation ID 5,000, test ID 10,000, CIFAR-10 10,000, MNIST 10,000, SVHN 26,032, Textures 1,880 and noise 10,000. Noise is diagnostic only; no natural macro includes it. Original classifier predictions define ID correctness. Previously reused checkpoint-selection validation does not become a fresh holdout.

## Components and exact identities

Let r be the fixed residual norm, v=alpha*r the train-scaled residual, E=-logsumexp(z_affine) and S=v+E the fixed ViM score. For each population save all-sample vectors v/E/S, residual and unnormalized feature L2 norm; use the saved training bank and fitted state for training ID, and existing phase-3 intermediates/scores for the other populations.

- Independently verify S=v+E and the saved phase-3 ViM scores on every existing scoring row; fitting remains unchanged.
- Summarize n, arithmetic mean, population SD (ddof=0), minimum, maximum and fixed quantiles [0.01,0.05,0.25,0.50,0.75,0.95,0.99] using linear interpolation for r/v/E/S/feature norm.
- Report population covariance and Pearson correlation between v and E. Use population denominator n; zero variance yields null correlation, never a replacement value. Verify Var(S)=Var(v)+Var(E)+2*Cov(v,E) with fixed float64 tolerances atol=1e-8/rtol=1e-10.
- Decompose each OOD-minus-ID-test mean ViM difference as delta(mean v)+delta(mean E), with both signed contributions retained. Also preserve validation-minus-training and ID-test-minus-training mean component differences. These population moments describe outputs; they do not prove why a representation changed or why a dataset is detected.

## Fixed-threshold transitions

For all three additions [ViM, Residual, ReAct Energy], all three already fixed references [affine Energy, original matched Energy, MSP], and all four ID-calibration targets [0.80,0.90,0.95,0.99], reuse the exact saved masks. Never select the most favorable comparison or target.

For each seed/target/comparison/population count four exhaustive states: both accepted, addition accepted/reference rejected, addition rejected/reference accepted, neither accepted. Save raw transition codes in original row order with a fixed documented mapping. For test ID split each state by original correct versus incorrect classification, preserving denominator counts and class labels; for each OOD population report states without a classifier-correctness label. Do not treat classifier errors and OOD membership as equivalent.

Primary thresholds are already selected by the same 2,500 ID-validation rows; changed thresholds between methods are part of the fixed operational comparison. Transitions do not isolate a pure ranking effect and cannot be presented as a paired comparison at exactly equal achieved ID-test coverage.

## Figures, statistics and checks

Use all samples for statistics. Optional joint component scatter uses at most 1,000 rows per population, sampled without replacement via default_rng(4313+100*seed+population_ordinal), sorted indices; population order is the eight listed above. Record sampled indices and mark plots as subsampled descriptive illustrations, not metric populations. Any CDF uses all rows. Do not choose axis ranges or examples by optimizing detector performance.

Preserve all population/seed summaries and every signed difference. Summarize mean and sample SD across three seeds with same-seed pairing; seed SD is descriptive, not CI. Training-population SD (ddof=0) is separate from between-seed SD (ddof=1). Independently verify identities/moments with separate mathematics, quantiles with manual sorting/interpolation, raw transition codes with Python contingency counts, and seed statistics with Python statistics. Save raw arrays/indices, input/source hashes, audits and scientific PNG/SVG; inspect rendered figures before completion.

No diagnosis in this phase authorizes modifying the frozen protocol or choosing a setting using OOD tests. A future method experiment requires its own predeclared ID-only development/selection population before execution.
