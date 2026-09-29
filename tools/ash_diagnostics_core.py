"""Frozen ASH support/scale descriptions and original-correctness transitions."""
import numpy as np
QUANTILES=[.01,.05,.25,.5,.75,.95,.99]
NEW=['ash_p_energy','ash_b_energy','ash_s_energy']
SCALARS=['original_sum','retained_sum','retained_fraction','dropped_fraction','ratio','exp_ratio','binary_fill','cutoff','cutoff_equal_count','cutoff_greater_count','cutoff_selected_equal_count']
COMPONENTS=SCALARS+['affine_energy']+NEW+[m+'_minus_affine' for m in NEW]
STATE_NAMES=['both_accept','ash_only_accept','reference_only_accept','neither_accept']
CLASS_STATES=['both_correct','shaped_only_correct','original_only_correct','neither_correct']

def finite(a):
 a=np.asarray(a)
 if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('aligned finite nonempty vectors required')
 return a

def assemble(middle,scores,part,bias):
 a={k:finite(middle[k]).copy() for k in SCALARS if k not in ['retained_fraction','dropped_fraction']}
 s1,s2=a['original_sum'],a['retained_sum']
 if not (s1>0).all() or not (s2>0).all():raise ValueError('strictly positive sums required')
 a['retained_fraction']=s2/s1;a['dropped_fraction']=1-a['retained_fraction']
 if not np.allclose(a['retained_fraction']*a['ratio'],1,atol=1e-8,rtol=1e-10) or not np.allclose(s1,s2/a['retained_fraction'],atol=1e-8,rtol=1e-10):raise ValueError('retained fraction identity failed')
 for m in ['affine_energy']+NEW:a[m]=finite(scores[m+'_'+part]).copy()
 if any(len(v)!=len(s1) for v in a.values()):raise ValueError('unaligned saved vectors')
 for m in NEW:a[m+'_minus_affine']=a[m]-a['affine_energy']
 zp,zs=middle['ash_p_energy_logits'],middle['ash_s_energy_logits']
 b=np.asarray(bias);c=a['exp_ratio'][:,None]
 expected=c*zp+(1-c)*b
 if zp.ndim!=2 or zs.shape!=zp.shape or b.shape!=(zp.shape[1],) or not np.isfinite(expected).all() or not np.allclose(zs,expected,atol=1e-8,rtol=1e-10):raise ValueError('bias-aware S/P logits identity failed')
 return {k:a[k] for k in COMPONENTS},float(np.abs(zs-expected).max())

def describe(arrays,logit_error):
 vectors={k:finite(arrays[k]).astype(float) for k in COMPONENTS};n=len(vectors['ratio'])
 if any(len(a)!=n for a in vectors.values()):raise ValueError('component length mismatch')
 result={k:{'n':n,'mean':float(v.mean()),'population_sd':float(v.std(ddof=0)),'min':float(v.min()),'max':float(v.max()),'quantiles':dict(zip(map(str,QUANTILES),np.quantile(v,QUANTILES,method='linear').tolist()))} for k,v in vectors.items()}
 q=vectors['retained_fraction'];qc=q-q.mean();qv=float(np.mean(qc**2));joint={}
 for m in NEW:
  d=vectors[m+'_minus_affine'];dc=d-d.mean();dv=float(np.mean(dc**2));cov=float(np.mean(qc*dc))
  joint[m]={'n':n,'fraction_variance':qv,'delta_energy_variance':dv,'covariance_fraction_delta':cov,'pearson_fraction_delta':cov/np.sqrt(qv*dv) if qv>0 and dv>0 else None}
 result['joint']=joint;result['identities']={'fraction_ratio_max_error':float(np.abs(q*vectors['ratio']-1).max()),'sum_reconstruction_max_error':float(np.abs(vectors['original_sum']-vectors['retained_sum']/q).max()),'bias_aware_logits_max_error':logit_error}
 return result

def mean_difference(population,reference):return {k:population[k]['mean']-reference[k]['mean'] for k in COMPONENTS}

def transitions(addition,reference,correct=None):
 a,r=np.asarray(addition),np.asarray(reference)
 if a.ndim!=1 or not len(a) or a.shape!=r.shape or a.dtype!=np.bool_ or r.dtype!=np.bool_:raise ValueError('aligned boolean masks required')
 codes=2*(~a).astype(np.uint8)+(~r).astype(np.uint8);counts=np.bincount(codes,minlength=4)
 result={'n':len(a),'state_counts':dict(zip(STATE_NAMES,map(int,counts))),'state_fractions':dict(zip(STATE_NAMES,(counts/len(a)).tolist())),'addition_acceptance':float(a.mean()),'reference_acceptance':float(r.mean()),'acceptance_change':float((int(counts[1])-int(counts[2]))/len(a))}
 if correct is not None:
  c=np.asarray(correct)
  if c.shape!=a.shape or c.dtype!=np.bool_:raise ValueError('original boolean correctness required')
  for name,subset in [('correct',c),('incorrect',~c)]:
   total=int(subset.sum());nums=np.bincount(codes[subset],minlength=4)
   result[name]={'n':total,'state_counts':dict(zip(STATE_NAMES,map(int,nums))),'state_fractions':dict(zip(STATE_NAMES,(nums/total).tolist())) if total else {k:None for k in STATE_NAMES}}
  ar=float((a&~c).sum()/a.sum()) if a.any() else None;rr=float((r&~c).sum()/r.sum()) if r.any() else None
  result['addition_retained_error_risk']=ar;result['reference_retained_error_risk']=rr;result['retained_error_risk_change']=ar-rr if ar is not None and rr is not None else None
 return result,codes

def classification(predictions,original,labels):
 p,o,y=np.asarray(predictions),np.asarray(original),np.asarray(labels)
 if p.ndim!=1 or not len(p) or p.shape!=o.shape or p.shape!=y.shape:raise ValueError('aligned nonempty class predictions required')
 c=p==y;oc=o==y;codes=2*(~c).astype(np.uint8)+(~oc).astype(np.uint8);counts=np.bincount(codes,minlength=4)
 return {'n':len(y),'original_accuracy':float(oc.mean()),'shaped_accuracy':float(c.mean()),'accuracy_change':float((int(counts[1])-int(counts[2]))/len(y)),'argmax_changes':int((p!=o).sum()),'state_counts':dict(zip(CLASS_STATES,map(int,counts))),'state_fractions':dict(zip(CLASS_STATES,(counts/len(y)).tolist()))},codes
