"""Fixed training-ID-only ViM/Residual/ReAct mathematics in CPU float64."""
import numpy as np


def validate(features, weight, bias):
    f, w, b = [np.asarray(a, dtype=np.float64) for a in (features, weight, bias)]
    if f.ndim != 2 or not len(f) or not np.isfinite(f).all():
        raise ValueError('features must be finite nonempty [N,D]')
    if w.ndim != 2 or b.shape != (len(w),) or w.shape[1] != f.shape[1]:
        raise ValueError('affine head dimensions mismatch')
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError('affine head must be finite')
    return f, w, b


def negative_logsumexp(logits):
    maximum = logits.max(axis=1)
    return -maximum - np.log(np.exp(logits - maximum[:, None]).sum(axis=1))


def fit_geometry(training_features, weight, bias, principal_dim=256, quantile=.99):
    f, w, b = validate(training_features, weight, bias)
    if not isinstance(principal_dim, int) or not 0 < principal_dim < f.shape[1]:
        raise ValueError('principal dimension must be strictly within feature dimension')
    if quantile != .99:
        raise ValueError('ReAct quantile is fixed at 0.99')
    origin = -np.linalg.pinv(w, rcond=1e-15) @ b
    centered = f - origin
    second_moment = centered.T @ centered / len(f)
    eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    principal = np.ascontiguousarray(eigenvectors[:, :principal_dim])
    residual = np.ascontiguousarray(eigenvectors[:, principal_dim:])
    train_residual = np.linalg.norm(centered @ residual, axis=1)
    train_max = (f @ w.T + b).max(axis=1)
    mean_residual, mean_max = float(train_residual.mean()), float(train_max.mean())
    if not np.isfinite(mean_residual) or mean_residual <= 0:
        raise ValueError('nonpositive/nonfinite training residual denominator')
    alpha = mean_max / mean_residual
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('nonpositive/nonfinite alpha; score orientation cannot be changed')
    clipping = float(np.quantile(f, .99, method='linear'))
    metadata = {'training_count': len(f), 'feature_dimensions': f.shape[1], 'principal_dimensions': principal_dim,
                'residual_dimensions': residual.shape[1], 'alpha': alpha, 'clipping_level': clipping,
                'react_global_quantile': .99, 'quantile_method': 'linear', 'pinv_rcond': 1e-15,
                'train_mean_residual': mean_residual, 'train_mean_max_affine_logit': mean_max,
                'fit_population': 'original ID training split only', 'normalization': 'unnormalized final pooled features',
                'precision': 'CPU float64', 'temperature': 1., 'sample_mean_subtracted': False,
                'eigenspectrum_cut_gap': float(eigenvalues[principal_dim - 1] - eigenvalues[principal_dim])}
    arrays = {'origin': origin, 'second_moment': second_moment, 'eigenvalues_descending': eigenvalues,
              'principal_basis': principal, 'residual_basis': residual,
              'training_residual': train_residual, 'training_max_affine_logit': train_max}
    return metadata, arrays


def geometry_scores(features, weight, bias, metadata, state):
    f, w, b = validate(features, weight, bias)
    centered = f - state['origin']
    residual = np.linalg.norm(centered @ state['residual_basis'], axis=1)
    affine = f @ w.T + b
    capped = np.minimum(f, metadata['clipping_level'])
    react = capped @ w.T + b
    energy = negative_logsumexp(affine)
    virtual = metadata['alpha'] * residual
    scores = {'affine_energy': energy, 'residual': residual, 'vim': virtual + energy,
              'react_energy': negative_logsumexp(react)}
    intermediates = {'affine_logits': affine, 'react_logits': react,
                     'residual_norm': residual, 'virtual_logit': virtual,
                     'clipped_coordinate_count': (f > metadata['clipping_level']).sum(axis=1)}
    if not all(np.isfinite(v).all() for v in list(scores.values()) + list(intermediates.values())):
        raise ValueError('nonfinite geometry output')
    return scores, intermediates
