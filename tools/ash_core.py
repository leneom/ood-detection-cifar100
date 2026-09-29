"""Fixed ASH-P/B/S, stable selected coordinates, CPU float64 anomaly Energy."""
import numpy as np
PERCENTILE=65
METHODS=['ash_p_energy','ash_b_energy','ash_s_energy']


def shape_ash(features,percentile=PERCENTILE):
    f=np.asarray(features,dtype=np.float64)
    if percentile!=PERCENTILE or f.ndim!=2 or not f.shape[0] or not f.shape[1] or not np.isfinite(f).all() or (f<0).any():
        raise ValueError('fixed percentile and finite nonnegative 2D features required')
    k=f.shape[1]-int(np.round(f.shape[1]*percentile/100.))
    if not 0<k<f.shape[1]: raise ValueError('invalid retained count')
    indices=np.argsort(-f,axis=1,kind='stable')[:,:k]
    values=np.take_along_axis(f,indices,axis=1)
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        s1=f.sum(axis=1); s2=values.sum(axis=1)
        if not np.isfinite(s1).all() or not np.isfinite(s2).all() or (s1<=0).any() or (s2<=0).any():
            raise ValueError('positive finite original/retained sums required')
        ratio=s1/s2; scale=np.exp(ratio); fill=s1/k
        h=np.zeros_like(f); np.put_along_axis(h,indices,values,axis=1)
        binary=np.zeros_like(f); np.put_along_axis(binary,indices,np.broadcast_to(fill[:,None],values.shape),axis=1)
        scaled=h*scale[:,None]
    if not np.isfinite(scaled).all() or not np.allclose(binary.sum(axis=1),s1,atol=1e-8,rtol=1e-10):
        raise ValueError('finite sharpening / ASH-B sum conservation failed')
    cutoff=values[:,-1]
    equal=(f==cutoff[:,None]).sum(axis=1)
    greater=(f>cutoff[:,None]).sum(axis=1)
    middle={'selected_indices':indices,'selected_values':values,'original_sum':s1,'retained_sum':s2,
            'ratio':ratio,'exp_ratio':scale,'binary_fill':fill,'cutoff':cutoff,'cutoff_equal_count':equal,
            'cutoff_greater_count':greater,'cutoff_selected_equal_count':k-greater}
    return {'ash_p_energy':h,'ash_b_energy':binary,'ash_s_energy':scaled},middle


def ash_scores(features,weight,bias):
    shapes,middle=shape_ash(features)
    w,b=np.asarray(weight,dtype=np.float64),np.asarray(bias,dtype=np.float64)
    if w.ndim!=2 or w.shape[1]!=shapes[METHODS[0]].shape[1] or b.shape!=(w.shape[0],) or not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError('finite matching original affine head required')
    scores={}
    for method,f in shapes.items():
        logits=f@w.T+b
        if not np.isfinite(logits).all(): raise ValueError('nonfinite shaped logits')
        middle[method+'_logits']=logits
        scores[method]=-np.logaddexp.reduce(logits,axis=1)
        if not np.isfinite(scores[method]).all(): raise ValueError('nonfinite anomaly Energy')
    return scores,middle
