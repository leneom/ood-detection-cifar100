"""Fixed ID-only thresholds and separate ID error / OOD acceptance counts."""
import numpy as np


def vector(scores):
    result = np.asarray(scores)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError('scores must be a nonempty finite vector')
    return result


def threshold_for_target(calibration_scores, target):
    scores = vector(calibration_scores)
    if not np.isfinite(target) or not 0 < target <= 1:
        raise ValueError('target must be in (0, 1]')
    index = int(np.ceil(target * len(scores))) - 1
    threshold = float(np.partition(scores, index)[index])
    accepted = scores <= threshold
    return threshold, {'n': len(scores), 'accepted': int(accepted.sum()),
                       'rejected': int((~accepted).sum()), 'achieved_acceptance': float(accepted.mean()),
                       'equal_threshold_count': int((scores == threshold).sum()),
                       'order_statistic_index': index}, accepted


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def id_decisions(scores, correct, threshold):
    scores = vector(scores)
    correct = np.asarray(correct)
    if correct.shape != scores.shape or correct.dtype != np.bool_:
        raise ValueError('correctness must be a matching boolean vector')
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    accepted = scores <= threshold
    rejected = ~accepted
    ct, it = int(correct.sum()), int((~correct).sum())
    ca, ia = int((correct & accepted).sum()), int((~correct & accepted).sum())
    cr, ir = int((correct & rejected).sum()), int((~correct & rejected).sum())
    count = int(accepted.sum())
    return {'n': len(scores), 'accepted': count, 'rejected': len(scores) - count,
            'correct_total': ct, 'incorrect_total': it,
            'correct_accepted': ca, 'incorrect_accepted': ia, 'correct_rejected': cr, 'incorrect_rejected': ir,
            'coverage': fraction(count, len(scores)), 'retained_error_risk': fraction(ia, count),
            'accepted_accuracy': fraction(ca, count), 'unconditional_accuracy': fraction(ct, len(scores)),
            'correct_rejection_fraction': fraction(cr, ct), 'incorrect_rejection_fraction': fraction(ir, it),
            'equal_threshold_count': int((scores == threshold).sum())}, accepted


def ood_decisions(scores, threshold):
    scores = vector(scores)
    if not np.isfinite(threshold):
        raise ValueError('threshold must be finite')
    accepted = scores <= threshold
    return {'n': len(scores), 'accepted': int(accepted.sum()), 'rejected': int((~accepted).sum()),
            'acceptance': float(accepted.mean()), 'equal_threshold_count': int((scores == threshold).sum())}, accepted
