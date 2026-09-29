"""Tie-aware metrics. Every input score is larger for OOD.

FPR95 uses ID as the positive population: accept at least 95% of ID and
measure the fraction of OOD accepted. AUPR-out uses OOD as positive.
No test OOD labels are used to fit scores or select checkpoints.
"""
from __future__ import annotations

import numpy as np


def vector(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite 1-D vector")
    return values


def binary_curve(labels, scores):
    scores = vector(scores)
    labels = np.asarray(labels)
    if labels.shape != scores.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary and match scores")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("both populations are required")
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels)[ends].astype(float)
    fp = ends + 1 - tp
    recall = np.r_[0., tp / positives]
    fpr = np.r_[0., fp / negatives]
    precision = np.r_[1., tp / (tp + fp)]
    return fpr, recall, precision


def average_precision(labels, scores):
    _, recall, precision = binary_curve(labels, scores)
    return float(np.sum(np.diff(recall) * precision[1:]))


def id_threshold(id_scores, target_tpr=0.95):
    scores = vector(id_scores)
    if not 0 < target_tpr <= 1:
        raise ValueError("target_tpr must lie in (0, 1]")
    # Observed order statistic, no interpolation; ties are accepted together.
    index = int(np.ceil(target_tpr * len(scores))) - 1
    return float(np.partition(scores, index)[index])


def detection_metrics(id_scores, ood_scores, calibration_scores=None):
    id_scores, ood_scores = vector(id_scores), vector(ood_scores)
    scores = np.r_[id_scores, ood_scores]
    labels = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    fpr, tpr, _ = binary_curve(labels, scores)
    # np.trapz also supports the NumPy 1.x environment.
    auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    threshold = id_threshold(id_scores)
    result = {
        "auroc": auroc,
        "aupr_out": average_precision(labels, scores),
        "aupr_in": average_precision(1 - labels, -scores),
        "fpr95_id": float(np.mean(ood_scores <= threshold)),
        "id_tpr_at_fpr95": float(np.mean(id_scores <= threshold)),
        "benchmark_threshold": threshold,
        "n_id": len(id_scores), "n_ood": len(ood_scores),
        "ood_prevalence": float(labels.mean()),
    }
    if calibration_scores is not None:
        threshold = id_threshold(calibration_scores)
        result.update({
            "validation_threshold": threshold,
            "id_test_acceptance_at_validation_threshold": float(np.mean(id_scores <= threshold)),
            "ood_acceptance_at_validation_threshold": float(np.mean(ood_scores <= threshold)),
            "calibration_id_acceptance": float(np.mean(vector(calibration_scores) <= threshold)),
        })
    return result
