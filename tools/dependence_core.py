"""Stable full-draw moments; no pairwise deletion and no fabricated zero correlations."""
import numpy as np

def moments(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<2 or x.shape[1]<1:raise ValueError('expected >=2 draws and >=1 variables')
    finite=np.isfinite(x);counts=finite.sum(0);valid=counts==len(x)
    means=np.full(x.shape[1],np.nan);cov=np.full((x.shape[1],x.shape[1]),np.nan)
    if valid.any():
        columns=np.flatnonzero(valid);delta=x[:,valid]-x[0,valid]
        shifted_mean=delta.mean(0);centered=delta-shifted_mean
        means[valid]=x[0,valid]+shifted_mean
        cov[np.ix_(columns,columns)]=(centered.T@centered)/(len(x)-1)
    variance=np.diag(cov).copy();positive=valid&(variance>0)
    defined=positive[:,None]&positive[None,:];denom=np.sqrt(variance[:,None])*np.sqrt(variance[None,:])
    corr=np.divide(cov,denom,out=np.full_like(cov,np.nan),where=defined)
    return {'means':means,'covariance':cov,'correlation':corr,'correlation_defined':defined,'finite_mask':finite,'finite_counts':counts,'variable_defined':valid}

def decomposition(values):
    x=np.asarray(values,dtype=np.float64);m=moments(x);n=x.shape[1]
    observed=moments(x.mean(1)[:,None])['covariance'][0,0]
    if m['variable_defined'].all():
        diagonal=float(np.trace(m['covariance'])/(n*n))
        cross=float(2*np.triu(m['covariance'],1).sum()/(n*n))
        reconstructed=diagonal+cross
        result={'defined':True,'draws':len(x),'components':n,'observed_mean_variance':float(observed),
                'diagonal_contribution':diagonal,'cross_covariance_contribution':cross,'reconstructed_variance':reconstructed,
                'signed_identity_residual':float(observed-reconstructed),'diagonal_only_counterfactual_variance':diagonal,
                'sd_ratio_to_diagonal_only':float(np.sqrt(observed/diagonal)) if diagonal>0 else None}
    else:
        result={'defined':False,'draws':len(x),'components':n,**{k:None for k in ['observed_mean_variance','diagonal_contribution','cross_covariance_contribution','reconstructed_variance','signed_identity_residual','diagonal_only_counterfactual_variance','sd_ratio_to_diagonal_only']}}
    return m,result

def json_finite(array):
    x=np.asarray(array)
    if x.ndim==0:return float(x) if np.isfinite(x) else None
    return [json_finite(v) for v in x]
