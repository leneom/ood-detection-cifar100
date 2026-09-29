"""Fixed training-ID population-covariance RMD in CPU float64."""
import numpy as np

SHRINKAGE = .01
NEGATIVE_DISTANCE_TOLERANCE = 1e-8


def features(value):
    f = np.asarray(value,dtype=np.float64)
    if f.ndim!=2 or not len(f) or not f.shape[1] or not np.isfinite(f).all():
        raise ValueError('finite nonempty [N,D] features required')
    return f


def fit_rmd(training_features,labels,shrinkage=SHRINKAGE):
    f=features(training_features)
    y=np.asarray(labels)
    if y.shape!=(len(f),) or y.dtype.kind not in 'iu' or (y<0).any():
        raise ValueError('aligned nonnegative integer training labels required')
    classes=np.unique(y)
    if not np.array_equal(classes,np.arange(len(classes))):
        raise ValueError('training classes must be contiguous from zero')
    if shrinkage!=SHRINKAGE:
        raise ValueError('shrinkage fixed at 0.01')
    counts=np.array([(y==k).sum() for k in classes],dtype=np.int64)
    means=np.stack([f[y==k].mean(axis=0) for k in classes])
    background_mean=f.mean(axis=0)
    within=f-means[y]
    background=f-background_mean
    raw=dict(within=within.T@within/len(f),background=background.T@background/len(f))
    state={'classes':classes,'class_counts':counts,'class_means':means,'background_mean':background_mean}
    metadata={'training_count':len(f),'class_count':len(classes),'feature_dimensions':f.shape[1],
              'shrinkage':shrinkage,'covariance_denominator':'N','precision':'CPU float64',
              'feature_normalization':'none','fit_population':'original ID training split only',
              'background_subtraction_weight':1.,'score_direction':'higher_is_ood',
              'negative_distance_tolerance':NEGATIVE_DISTANCE_TOLERANCE}
    for name,cov in raw.items():
        scale=max(float(np.trace(cov)/f.shape[1]),1e-12)
        regularized=(1-shrinkage)*cov+shrinkage*scale*np.eye(f.shape[1])
        eigenvalues=np.linalg.eigvalsh(regularized)
        if not np.isfinite(regularized).all() or not np.isfinite(eigenvalues).all() or eigenvalues[0]<=0:
            raise ValueError('nonfinite or nonpositive covariance; fixed shrinkage cannot change')
        precision=np.linalg.inv(regularized)
        precision=(precision+precision.T)/2
        if not np.allclose(precision@regularized,np.eye(f.shape[1]),atol=1e-8,rtol=1e-8):
            raise ValueError('inverse identity failed')
        state.update({name+'_covariance_raw':cov,name+'_covariance':regularized,name+'_precision':precision,
                      name+'_eigenvalues':eigenvalues,name+'_raw_eigenvalues':np.linalg.eigvalsh(cov)})
        metadata.update({name+'_trace_scale':scale,name+'_condition_number':float(eigenvalues[-1]/eigenvalues[0])})
    return metadata,state


def nonnegative_distances(raw):
    if not np.isfinite(raw).all() or raw.min() < -NEGATIVE_DISTANCE_TOLERANCE:
        raise ValueError('nonfinite or too-negative squared distance')
    negatives=raw<0
    return np.maximum(raw,0),int(negatives.sum()),float(-raw[negatives].min()) if negatives.any() else 0.


def rmd_scores(query_features,state):
    f=features(query_features)
    means,origin=state['class_means'],state['background_mean']
    if means.ndim!=2 or means.shape[1]!=f.shape[1] or origin.shape!=(f.shape[1],):
        raise ValueError('fitted/query dimensions differ')
    # Common-origin algebra improves stability without changing any Gaussian distance.
    centered=f-origin
    offsets=means-origin
    p=state['within_precision']
    projected=centered@p
    raw_class=(projected*centered).sum(axis=1)[:,None]-2*projected@offsets.T+((offsets@p)*offsets).sum(axis=1)[None,:]
    raw_bg=((centered@state['background_precision'])*centered).sum(axis=1)
    class_distance,nc,cc=nonnegative_distances(raw_class)
    background_distance,nb,cb=nonnegative_distances(raw_bg)
    nearest=class_distance.argmin(axis=1)
    md=class_distance[np.arange(len(f)),nearest]
    signed=md-background_distance
    if not np.isfinite(signed).all():
        raise ValueError('nonfinite RMD score')
    scores={'mahalanobis_cpu':md,'rmd':signed}
    middle={'class_distances':class_distance,'background_distance':background_distance,
            'nearest_class':state['classes'][nearest],'mahalanobis_cpu':md,'rmd':signed}
    diagnostics={'negative_class_distance_count':nc,'max_class_roundoff_correction':cc,
                 'negative_background_distance_count':nb,'max_background_roundoff_correction':cb,
                 'raw_min_class_distance':float(raw_class.min()),'raw_min_background_distance':float(raw_bg.min())}
    return scores,middle,diagnostics
