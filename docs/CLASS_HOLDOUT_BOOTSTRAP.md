# Frozen class-holdout paired image bootstrap v0.1 — 2026-09-28

## Scope

Quantify **conditional image-sampling variation** of the completed three-seed CIFAR-100 80-ID/20-held-out-class result. This is post-hoc uncertainty description on already viewed official test images. It does not add a new test population, retrain a model, fit a detector, modify a threshold, select a method, or provide nominal population coverage. Keep the original aggregate and 20-class diagnostic immutable.

The analysis is declared before drawing any bootstrap indices or reading bootstrap outcomes. Exact inputs: the verified 160-file class-holdout completion, the verified 36-file class diagnostic, their frozen original-row/class mapping, all three seed-specific ID/OOD score vectors, three score manifests, and saved fixed-threshold masks. All methods, model seeds and thresholds remain unchanged.

## Fixed draw schedule

- Exactly 1,000 replicates, indexed 0…999. Replicate `r` uses `numpy.random.default_rng(2035 + r)`. Visit the 80 original ID fine classes in ascending ID order, then the 20 original held-out OOD fine classes in ascending ID order. For each class, draw 100 **within-class row positions with replacement** using the generator; save aligned per-image integer multiplicities. Every replicate therefore has exactly 8,000 ID and 2,000 OOD images, preserving the observed 80/20 class mix. This resamples images conditional on the fixed classes and class split; it does not estimate uncertainty over possible held-out-class choices.
- Use the **same ID/OOD multiplicity vectors** for all three fixed checkpoints, both temperature variants and all six fixed detectors. Save every replicate's weights and raw metric arrays. No labels, logits, features, fitted temperatures or scores are recalculated; only weighted summaries of frozen raw scores/masks are computed.
- For every replicate/model/variant/method, calculate AUROC with tied ID/OOD scores receiving half credit, AUPR-out, AUPR-in, and FPR95-ID using the empirical ID-test 95% order statistic **within that replicate**. Keep that benchmark FPR95 separate from the frozen ID-validation operating threshold. Also calculate actual ID-test and OOD-test acceptance at the original fixed 95% ID-validation threshold from saved masks. Preserve the original 8,000:2,000 prevalence for AUPR.
- For each of the 12 method/variant cells and each metric, report the original three-seed mean and training-seed sample SD **separately** from the bootstrap distribution of the fixed three-model mean. Display the latter's median and linear-interpolated 2.5th/97.5th percentiles as descriptive resampling intervals, not as formal confidence intervals or significance tests. Publish all 12 cells; no pairwise winner, cherry-picked replicate, or new method setting.

## Independent checks and limits

Source-bind the draw declaration to this protocol and both completed phases before sampling. Independently regenerate all 1,000 class-stratified integer-weight vectors, verify exact class/overall counts and shared use across methods/models, independently recalculate every frozen-mask weighted acceptance and at least the first ten expanded-image ranking replicates with scikit-learn, and audit every reported seed statistic/quantile and output hash. Preserve partial blocks on failure; only a complete 1,000-replicate run may be reported as complete.

These draws reuse the same observed images and class list. The three training models share the test images, so draw dependence must be retained. Training-seed SD measures a different source of variation and is not a confidence interval. Neither interval supplies new external validation or corrects earlier test-guided research decisions.
