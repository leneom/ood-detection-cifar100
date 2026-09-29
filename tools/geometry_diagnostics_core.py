"""Descriptive components and exhaustive transitions; no score or threshold fitting."""
import numpy as np

QUANTILES = [.01, .05, .25, .5, .75, .95, .99]
COMPONENTS = ['residual', 'virtual_logit', 'affine_energy', 'vim', 'feature_norm']
STATE_NAMES = ['both_accepted', 'addition_only_accepted', 'reference_only_accepted', 'neither_accepted']


def finite_vector(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('nonempty finite vector required')
    return a


def describe(arrays):
    vectors = {k: finite_vector(arrays[k]) for k in COMPONENTS}
    n = len(vectors['vim'])
    if any(len(v) != n for v in vectors.values()):
        raise ValueError('component rows must match')
    v, e, s = [vectors[k] for k in ['virtual_logit', 'affine_energy', 'vim']]
    if not np.allclose(s, v + e, atol=2e-9, rtol=0):
        raise ValueError('ViM component identity failed')
    result = {}
    for key, a in vectors.items():
        result[key] = {'n': n, 'mean': float(a.mean()), 'population_sd': float(a.std(ddof=0)),
                       'min': float(a.min()), 'max': float(a.max()),
                       'quantiles': dict(zip(map(str, QUANTILES), np.quantile(a, QUANTILES, method='linear').tolist()))}
    vc, ec = v-v.mean(), e-e.mean()
    variance_v, variance_e, variance_s = float(np.mean(vc**2)), float(np.mean(ec**2)), float(s.var(ddof=0))
    covariance = float(np.mean(vc*ec))
    combined = variance_v + variance_e + 2*covariance
    if not np.isclose(variance_s, combined, atol=1e-8, rtol=1e-10):
        raise ValueError('ViM population variance decomposition failed')
    result['joint'] = {'n': n, 'virtual_variance': variance_v, 'energy_variance': variance_e,
                       'vim_variance': variance_s, 'covariance_virtual_energy': covariance,
                       'pearson_virtual_energy': covariance/np.sqrt(variance_v*variance_e) if variance_v > 0 and variance_e > 0 else None,
                       'variance_component_sum': combined, 'variance_identity_error': abs(variance_s-combined),
                       'score_identity_max_error': float(np.max(np.abs(s-v-e)))}
    return result


def mean_difference(population, reference):
    values = {k: population[k]['mean']-reference[k]['mean'] for k in ['virtual_logit','affine_energy','vim']}
    values['sum_virtual_energy'] = values['virtual_logit'] + values['affine_energy']
    if not np.isclose(values['vim'], values['sum_virtual_energy'], atol=1e-8, rtol=1e-10):
        raise ValueError('mean component difference identity failed')
    values['identity_error'] = abs(values['vim']-values['sum_virtual_energy'])
    return values


def transitions(addition, reference, correct=None):
    a, r = np.asarray(addition), np.asarray(reference)
    if a.ndim != 1 or not len(a) or a.shape != r.shape or a.dtype != np.bool_ or r.dtype != np.bool_:
        raise ValueError('aligned boolean decision masks required')
    # code 0 both, 1 addition only, 2 reference only, 3 neither.
    codes = 2*(~a).astype(np.uint8) + (~r).astype(np.uint8)
    counts = np.bincount(codes, minlength=4)
    result = {'n': len(a), 'state_counts': dict(zip(STATE_NAMES, map(int, counts))),
              'state_fractions': dict(zip(STATE_NAMES, (counts/len(a)).tolist())),
              'addition_acceptance': float(a.mean()), 'reference_acceptance': float(r.mean()),
              'acceptance_change': float((int(counts[1])-int(counts[2]))/len(a))}
    if correct is not None:
        c = np.asarray(correct)
        if c.shape != a.shape or c.dtype != np.bool_:
            raise ValueError('original correctness must be an aligned boolean mask')
        for name, subset in [('correct', c), ('incorrect', ~c)]:
            total = int(subset.sum())
            numbers = np.bincount(codes[subset], minlength=4)
            result[name] = {'n': total, 'state_counts': dict(zip(STATE_NAMES, map(int, numbers))),
                            'state_fractions': dict(zip(STATE_NAMES, (numbers/total).tolist())) if total else {k: None for k in STATE_NAMES}}
    return result, codes
