"""Integer-weight repeated-image metrics; all equal scores are grouped."""
import math
import numpy as np
RANKING=['auroc','aupr_out','aupr_in','fpr95_id','id_tpr_at_fpr95','benchmark_threshold']

def weights(a,n):
 a=np.asarray(a)
 if a.shape!=(n,) or not np.issubdtype(a.dtype,np.integer) or (a<0).any() or int(a.sum())<=0:raise ValueError('aligned nonnegative integer weights with positive total required')
 return a.astype(np.int64)

class RankingPlan:
 def __init__(self,inside,outside):
  self.inside=np.asarray(inside,dtype=float);self.outside=np.asarray(outside,dtype=float)
  if any(a.ndim!=1 or not len(a) or not np.isfinite(a).all() for a in [self.inside,self.outside]):raise ValueError('finite nonempty score vectors required')
  z=np.r_[self.inside,self.outside];self.order=np.argsort(z,kind='stable');sorted_z=z[self.order]
  self.starts=np.r_[0,np.flatnonzero(np.diff(sorted_z))+1];self.id_flags=self.order<len(self.inside)
  self.id_order=np.argsort(self.inside,kind='stable')
 def evaluate(self,id_weights,ood_weights):
  wi,wo=weights(id_weights,len(self.inside)),weights(ood_weights,len(self.outside));ni,no=int(wi.sum()),int(wo.sum())
  w=np.r_[wi,wo][self.order];i=np.add.reduceat(np.where(self.id_flags,w,0),self.starts);o=np.add.reduceat(np.where(self.id_flags,0,w),self.starts)
  ci,co=np.cumsum(i),np.cumsum(o)
  auc=float(np.sum(o*(ci-.5*i))/(ni*no))
  desc_i,desc_o=np.cumsum(i[::-1]),np.cumsum(o[::-1]);dt=desc_i+desc_o;at=ci+co
  out_precision=np.divide(desc_o,dt,out=np.zeros(len(dt),dtype=float),where=dt>0)
  in_precision=np.divide(ci,at,out=np.zeros(len(at),dtype=float),where=at>0)
  apout=float(np.sum(o[::-1]*out_precision)/no);apin=float(np.sum(i*in_precision)/ni)
  index=int(np.searchsorted(np.cumsum(wi[self.id_order]),math.ceil(.95*ni),side='left'))
  threshold=float(self.inside[self.id_order[index]])
  fpr=float(wo[self.outside<=threshold].sum()/no);tpr=float(wi[self.inside<=threshold].sum()/ni)
  return np.asarray([auc,apout,apin,fpr,tpr,threshold],dtype=float)

def operational_counts(id_weights,ood_weights_by_population,masks,correct,datasets):
 wi=weights(id_weights,len(correct));c=np.asarray(correct)
 if c.dtype!=np.bool_ or c.shape!=wi.shape:raise ValueError('aligned original boolean correctness required')
 # masks are [method,target,row], all thresholds remain frozen.
 accepted=np.asarray(masks['id_accepted'])
 if accepted.dtype!=np.bool_ or accepted.ndim!=3 or accepted.shape[-1]!=len(wi):raise ValueError('frozen aligned masks required')
 id_n=np.sum(accepted*wi,axis=-1,dtype=np.int64);bad=np.sum(accepted*(wi*(~c)),axis=-1,dtype=np.int64)
 oo=np.stack([np.sum(masks[d+'_accepted']*weights(ood_weights_by_population[d],masks[d+'_accepted'].shape[-1]),axis=-1,dtype=np.int64) for d in datasets],axis=-1)
 coverage=id_n/wi.sum();risk=np.divide(bad,id_n,out=np.full(id_n.shape,np.nan),where=id_n>0)
 acceptance=oo/np.asarray([int(ood_weights_by_population[d].sum()) for d in datasets])
 return id_n,bad,oo,coverage,risk,acceptance
