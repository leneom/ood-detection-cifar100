# Exploratory fixed class-holdout RMD extension

This follows already viewed nine-model class-holdout and other OOD results. It is a separately labeled post-hoc method extension, not an untouched confirmation and not an alteration of the completed six-method report.

Use exactly class partition seeds 2031/2032/2033 and each model seed 0/1/2, with their immutable ID-validation-selected checkpoints and already independently audited extraction archives. Fit only on each model's 36,000 optimization-ID pooled 512-dimensional, unnormalized features and contiguous 80-class ID targets (450 rows per class); no ID validation/test or OOD feature enters fitting. Reuse the existing `tools/rmd_core.py` fixed CPU float64 formula and rho=0.01 population-covariance shrinkage from `docs/RMD_PHASE4.md`, without modification:

- shared within-class class means/covariance and label-independent background mean/covariance;
- equal weight background subtraction: RMD = min class Mahalanobis squared distance - background squared distance;
- higher score means more OOD; retain signed negative values;
- save matched CPU class Mahalanobis distance as a numerical/fit control, distinct from the original frozen Mahalanobis baseline.

No sweep or refitting based on OOD outcomes. Before any OOD scoring, save every fitted mean/covariance/precision/eigenvalue and fit diagnostic with training-ID/source/checkpoint hashes; independently verify the training-bank membership, class balance, means/covariances, fixed shrinkage, inverse identity, positive definiteness and fixed-probe distances. Refuse a partial fit rather than changing rho.

After all nine fit audits pass, evaluate all already saved official ID test and held-out-class OOD feature rows plus the separate ID temperature/threshold validation roles. The primary operating threshold is the original target95 order statistic of 1,200 ID threshold-validation rows, with ties accepted. Retain all four original targets 80/90/95/99 and AUROC/AP-out/AP-in/FPR95-ID as descriptive outputs, and preserve complete raw scores, masks, logits/feature input hashes and per-model metrics. Compare RMD, matched CPU MD and all six unchanged T=1 baseline columns for all nine models and three partitions; no OOD-based method choice or deployment claim. Audit every population/score/metric independently. Do not overwrite or edit the original six-method report or raw extraction.
