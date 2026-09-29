"""Freeze inputs before all-one parity and independent expanded synthetic checks."""
from pathlib import Path
import shutil
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_io import ROOT,read,write,digest,require,now
from bootstrap_io import RUN,METHODS,DATASETS,COUNTS,bundle
from bootstrap_core import RankingPlan,operational_counts,RANKING

def main():
 require(not (RUN/'preflight.json').exists() and not (RUN/'preflight-inputs.json').exists(),'existing preflight retained')
 pre=read(RUN/'predeclaration.json');inputs=dict(pre['original_artifacts']);inputs['runs/paired-bootstrap/predeclaration.json']=digest(RUN/'predeclaration.json')
 for n,h in inputs.items():require(digest(ROOT/n)==h,'predeclared input changed')
 files=[Path(__file__),ROOT/'tools/bootstrap_io.py',ROOT/'tools/bootstrap_core.py',ROOT/'tools/geometry_io.py',ROOT/'tests/test_bootstrap.py']
 sources={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files};snapshot=RUN/'preflight-source';snapshot.mkdir(exist_ok=False)
 for p in files:shutil.copy2(p,snapshot/p.name)
 manifest={'state':'frozen_before_full_benchmark_weighted_point_checks','prepared_at':now(),'original_artifacts':inputs,'source_hashes':sources}
 write(RUN/'preflight-inputs.json',manifest)
 scores,masks,results,alignment=bundle();write(RUN/'row-alignment.json',alignment)
 rng=np.random.default_rng(9241);synthetic=0
 for iteration in range(50):
  zi,zo=rng.integers(-3,4,7).astype(float),rng.integers(-3,4,8).astype(float)
  wi,wo=rng.integers(0,5,7),rng.integers(0,5,8)
  if not wi.sum() or not wo.sum():continue
  a=RankingPlan(zi,zo).evaluate(wi,wo);inside,outside=np.repeat(zi,wi),np.repeat(zo,wo)
  labels=np.r_[np.zeros(len(inside)),np.ones(len(outside))];values=np.r_[inside,outside]
  threshold=sorted(inside.tolist())[int(np.ceil(.95*len(inside)))-1]
  actual=np.array([roc_auc_score(labels,values),average_precision_score(labels,values),average_precision_score(1-labels,-values),np.mean(outside<=threshold),np.mean(inside<=threshold),threshold])
  require(np.allclose(a,actual,atol=1e-12,rtol=0),'expanded synthetic metrics mismatch');synthetic+=1
 checked=counts_checked=0
 for seed in range(3):
  wi=np.ones(10000,dtype=np.uint32);wo={d:np.ones(COUNTS[d],dtype=np.uint32) for d in DATASETS}
  for d in DATASETS:
   for m in METHODS:
    point=RankingPlan(scores[seed][d][m+'_id'],scores[seed][d][m+'_ood']).evaluate(wi,wo[d])
    raw=results[seed]['datasets'][d]['methods'][m]['primary_2500']
    require(np.allclose(point,[raw[k] for k in RANKING],atol=1e-12,rtol=0),'all-one original ranking parity mismatch');checked+=len(RANKING)
  a,b,o,cov,risk,accept=operational_counts(wi,wo,masks[seed],masks[seed]['id_correct'],DATASETS)
  points={(p['method'],p['target']):p for p in results[seed]['operating_points']}
  for mi,m in enumerate(METHODS):
   for ti,target in enumerate([.8,.9,.95,.99]):
    p=points[(m,target)];require(a[mi,ti]==p['id_test']['accepted'] and b[mi,ti]==p['id_test']['incorrect_accepted'],'all-one ID counts differ')
    require(abs(cov[mi,ti]-p['id_test']['coverage'])<1e-12 and abs(risk[mi,ti]-p['id_test']['retained_error_risk'])<1e-12,'all-one ID endpoints differ');counts_checked+=4
    for di,d in enumerate(DATASETS):require(o[mi,ti,di]==p['ood'][d]['accepted'] and abs(accept[mi,ti,di]-p['ood'][d]['acceptance'])<1e-12,'all-one OOD endpoints differ');counts_checked+=2
 for n,h in inputs.items():require(digest(ROOT/n)==h,'input changed during preflight')
 for n,h in sources.items():require(digest(ROOT/n)==h,'source changed during preflight')
 audit={'state':'passed','audited_at':now(),'synthetic_expanded_sklearn_cases':synthetic,'all_one_ranking_fields':checked,'all_one_operational_fields':counts_checked,'inputs_sha256':digest(RUN/'preflight-inputs.json'),'row_alignment_sha256':digest(RUN/'row-alignment.json'),'source_sha256':digest(__file__),'boundary':'no indexed full-benchmark bootstrap draws produced; independent synthetic expanded checks and complete original all-one parity'}
 write(RUN/'preflight.json',audit);print(audit,flush=True)
if __name__=='__main__':main()
