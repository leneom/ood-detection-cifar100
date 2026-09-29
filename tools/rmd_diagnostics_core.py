"""Signed frozen RMD descriptions and exhaustive fixed-mask transitions."""
import numpy as np
QUANTILES=[.01,.05,.25,.5,.75,.95,.99]
COMPONENTS=['mahalanobis_cpu','background_distance','rmd']
STATE_NAMES=['both_accept','rmd_only_accept','reference_only_accept','neither_accept']


def finite_vector(a):
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('nonempty finite vector required')
    return a


def describe(arrays):
    vectors={k:finite_vector(arrays[k]) for k in COMPONENTS}
    m,b,r=[vectors[k] for k in COMPONENTS]
    n=len(r)
    if any(len(v)!=n for v in vectors.values()):
        raise ValueError('component rows must match')
    if not np.allclose(r,m-b,atol=1e-8,rtol=1e-10):
        raise ValueError('signed RMD identity failed')
    result={}
    for k,a in vectors.items():
        result[k]={'n':n,'mean':float(a.mean()),'population_sd':float(a.std(ddof=0)),
                   'min':float(a.min()),'max':float(a.max()),
                   'quantiles':dict(zip(map(str,QUANTILES),np.quantile(a,QUANTILES,method='linear').tolist()))}
    mc,bc=m-m.mean(),b-b.mean()
    mv,bv,rv=float(np.mean(mc**2)),float(np.mean(bc**2)),float(r.var(ddof=0))
    covariance=float(np.mean(mc*bc))
    combined=mv+bv-2*covariance
    if not np.isclose(rv,combined,atol=1e-8,rtol=1e-10):
        raise ValueError('subtracted population variance identity failed')
    result['joint']={'n':n,'MD_variance':mv,'background_variance':bv,'rmd_variance':rv,
                     'covariance_MD_background':covariance,
                     'pearson_MD_background':covariance/np.sqrt(mv*bv) if mv>0 and bv>0 else None,
                     'variance_component_difference':combined,'variance_identity_error':abs(rv-combined),
                     'score_identity_max_error':float(np.max(np.abs(r-(m-b))))}
    return result


def mean_difference(population,reference):
    values={k:population[k]['mean']-reference[k]['mean'] for k in COMPONENTS}
    values['MD_minus_background']=values['mahalanobis_cpu']-values['background_distance']
    if not np.isclose(values['rmd'],values['MD_minus_background'],atol=1e-8,rtol=1e-10):
        raise ValueError('signed mean difference failed')
    values['identity_error']=abs(values['rmd']-values['MD_minus_background'])
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
