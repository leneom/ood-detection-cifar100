"""CPU-only phase-2 mathematics; no training or feature-detector dependencies."""
import numpy as np


def stratified_calibration_rows(labels, seed=2027):
    labels = np.asarray(labels)
    if labels.shape != (5000,) or not np.array_equal(np.unique(labels), np.arange(100)):
        raise ValueError('expected the original 5000-row, 100-class validation set')
    rng = np.random.default_rng(seed)
    fit, threshold = [], []
    for label in range(100):
        rows = np.flatnonzero(labels == label)
        if len(rows) != 50:
            raise ValueError('validation must contain exactly 50 rows per class')
        rng.shuffle(rows)
        fit.extend(rows[:25])
        threshold.extend(rows[25:])
    return np.sort(np.asarray(fit, dtype=np.int64)), np.sort(np.asarray(threshold, dtype=np.int64))


def log_probabilities(logits, temperature=1.):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError('expected finite nonempty logits matrix')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    z = z / temperature
    maximum = z.max(axis=1, keepdims=True)
    normalizer = maximum + np.log(np.exp(z - maximum).sum(axis=1, keepdims=True))
    return z - normalizer, normalizer[:, 0]


def validated_labels(labels, logits):
    y = np.asarray(labels)
    if y.shape != (len(logits),) or y.dtype.kind not in 'iu':
        raise ValueError('labels must be a matching integer vector')
    if np.any(y < 0) or np.any(y >= logits.shape[1]):
        raise ValueError('label outside logits class range')
    return y


def nll(logits, labels, temperature=1.):
    log_p, _ = log_probabilities(logits, temperature)
    labels = validated_labels(labels, log_p)
    return float(-log_p[np.arange(len(labels)), labels].mean())


def fit_temperature(fit_logits, fit_labels):
    """The signature accepts fitting rows only, never test or OOD arrays."""
    from scipy.optimize import minimize_scalar
    original = nll(fit_logits, fit_labels)
    result = minimize_scalar(lambda log_t: nll(fit_logits, fit_labels, np.exp(log_t)),
                             bounds=(np.log(.05), np.log(20.)), method='bounded',
                             options={'xatol': 1e-8, 'maxiter': 200})
    candidate = float(np.exp(result.x))
    objective = float(result.fun)
    reason = None
    if not result.success:
        reason = 'optimizer_failure'
    elif not np.isfinite(candidate) or not np.isfinite(objective):
        reason = 'nonfinite_optimizer_result'
    elif objective > original + 1e-12:
        reason = 'fit_NLL_increased'
    temperature = 1. if reason else candidate
    return {'temperature': temperature, 'candidate_temperature': candidate,
            'success': bool(result.success), 'message': str(result.message),
            'iterations': int(result.nit), 'evaluations': int(result.nfev),
            'candidate_fit_nll': objective, 'fit_nll_T1': original,
            'selected_fit_nll': nll(fit_logits, fit_labels, temperature),
            'fallback_reason': reason, 'bounds': [.05, 20.],
            'distance_to_log_bound': float(min(np.log(temperature / .05), np.log(20. / temperature))),
            'near_bound_log_distance_lt_1e-4': bool(min(np.log(temperature / .05), np.log(20. / temperature)) < 1e-4),
            'method': 'SciPy bounded scalar minimize in log(T)',
            'xatol': 1e-8, 'maxiter': 200, 'population': '2500 ID temperature-fit rows only',
            'precision': 'CPU float64'}


def logit_scores(logits, temperature=1.):
    log_p, normalizer = log_probabilities(logits, temperature)
    p = np.exp(log_p)
    return {'msp': 1. - p.max(axis=1), 'energy': -temperature * normalizer,
            'entropy': -(p * log_p).sum(axis=1),
            'maxlogit': -np.asarray(logits, dtype=np.float64).max(axis=1)}


def calibration_metrics(logits, labels, temperature=1.):
    log_p, _ = log_probabilities(logits, temperature)
    y = validated_labels(labels, log_p)
    p = np.exp(log_p)
    correct = log_p.argmax(axis=1) == y
    confidence = p.max(axis=1)
    # [lower, upper), with confidence=1 included in the last bin.
    bins = np.minimum((confidence * 15).astype(np.int64), 14)
    ece, records = 0., []
    for index in range(15):
        mask = bins == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_confidence)
        records.append({'lower': index / 15, 'upper': (index + 1) / 15,
                        'count': count, 'accuracy': accuracy, 'confidence': mean_confidence})
    brier = (np.square(p).sum(axis=1) - 2 * p[np.arange(len(y)), y] + 1).mean()
    return {'nll': float(-log_p[np.arange(len(y)), y].mean()), 'brier': float(brier),
            'ece15': float(ece), 'accuracy': float(correct.mean()), 'n': len(y),
            'mean_confidence': float(confidence.mean()), 'bins': records,
            'brier_definition': 'mean sum over classes of (p - one_hot_y)^2',
            'ECE_definition': '15 equal-width confidence bins; lower inclusive, upper exclusive; last includes 1'}
