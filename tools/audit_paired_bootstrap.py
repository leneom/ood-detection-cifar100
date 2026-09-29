"""Regenerate every draw/count and independently expand first ten ranking resamples."""
import argparse,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from geometry_io import ROOT,read,write,digest,load,require,now
from bootstrap_io import RUN,METHODS,DATASETS,COUNTS,TARGETS,bundle

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
 status=read(RUN/'status.json');attempt=status['output_name'];manifest_path=RUN/f'{attempt}-inputs.json';manifest=read(manifest_path)
 require(status['state'] in ['raw_complete_pending_independent_audit','paused_at_fixed_block_boundary','complete'],'controller still running or failed')
 if not args.allow_partial:require(status['completed_replicates']==1000,'full bootstrap incomplete')
 for n,h in manifest['original_artifacts'].items():require(digest(ROOT/n)==h,'original bootstrap input changed')
 for n,h in manifest['source_hashes'].items():require(digest(ROOT/n)==digest(RUN/f'{attempt}-source'/Path(n).name)==h,'executed bootstrap source changed')
 scores,masks,results,alignment=bundle();out=RUN/attempt
 markers=sorted(out.glob('block-*.json'));require(len(markers)==len(status['completed_blocks']) and [int(p.stem.split('-')[-1]) for p in markers]==status['completed_blocks'],'completed blocks disagree')
 checks={'regenerated_weight_vectors':0,'weighted_operational_count_cells':0,'weighted_operational_rate_cells':0,'independent_expanded_ranking_fields':0,'ranking_range_rows':0,'macro_fields':0};hashes={}
 def agree(a,b,label):require(a.shape==b.shape and np.allclose(a,b,rtol=0,atol=1e-12,equal_nan=True),label)
 for marker in markers:
  meta=read(marker);start=meta['replicate_start'];wp=marker.with_name(marker.stem+'-weights.npz');mp=marker.with_name(marker.stem+'-metrics.npz')
  require(meta['inputs_sha256']==digest(manifest_path) and digest(wp)==meta['weights_sha256'] and digest(mp)==meta['metrics_sha256'],'committed block changed')
  w,a=load(wp),load(mp);rep=np.arange(start,start+100,dtype=np.int64);require(np.array_equal(w['replicates'],rep) and np.array_equal(a['replicates'],rep),'replicate indices changed')
  for local,b in enumerate(rep):
   rng=np.random.default_rng(8939+int(b))
   for d,n in COUNTS.items():
    expected=np.bincount(rng.integers(0,n,size=n),minlength=n).astype(np.uint32)
    require(w[d].shape==(100,n) and w[d].dtype==np.uint32 and np.array_equal(w[d][local],expected) and int(expected.sum())==n,'indexed bootstrap weights differ');checks['regenerated_weight_vectors']+=1
  require(a['ranking'].shape==(100,3,5,15,6) and a['ranking'].dtype==np.float64 and np.isfinite(a['ranking']).all(),'ranking archive shape/dtype invalid')
  require((a['ranking'][...,:5]>=-1e-12).all() and (a['ranking'][...,:5]<=1+1e-12).all(),'ranking probability range invalid');checks['ranking_range_rows']+=100*3*5*15
  for seed in range(3):
   # Integer matrix products independently replace producer broadcast/reduction.
   mask=masks[seed]['id_accepted'];correct=masks[seed]['id_correct'];wi=w['id'].astype(np.uint32)
   ea=(wi@mask.reshape(60,10000).astype(np.uint32).T).reshape(100,15,4).astype(np.int64)
   eb=(wi@(mask&(~correct)).reshape(60,10000).astype(np.uint32).T).reshape(100,15,4).astype(np.int64)
   eo=np.stack([(w[d]@masks[seed][d+'_accepted'].reshape(60,COUNTS[d]).astype(np.uint32).T).reshape(100,15,4) for d in DATASETS],axis=-1).astype(np.int64)
   for key,actual in [('id_accepted',ea),('id_incorrect_accepted',eb),('ood_accepted',eo)]:require(a[key].dtype==np.int64 and np.array_equal(a[key][:,seed],actual),'independent weighted count differs '+key);checks['weighted_operational_count_cells']+=actual.size
   risk=np.divide(eb,ea,out=np.full(ea.shape,np.nan),where=ea>0)
   for key,actual in [('id_coverage',ea/10000),('retained_error_risk',risk),('ood_acceptance',eo/np.array([COUNTS[d] for d in DATASETS]))]:agree(a[key][:,seed],actual,'weighted rate disagrees '+key);checks['weighted_operational_rate_cells']+=actual.size
  natural=sum(a['ranking'][:,:,d,:,:4] for d in range(4))/4;op_natural=sum(a['ood_acceptance'][...,d] for d in range(4))/4
  agree(a['natural_ranking_macro'],natural,'within-seed ranking macro differs');agree(a['natural_ood_acceptance_macro'],op_natural,'within-seed acceptance macro differs');checks['macro_fields']+=natural.size+op_natural.size
  for local,b in enumerate(rep):
   if b>=10:continue
   for seed in range(3):
    for di,d in enumerate(DATASETS):
     for mi,m in enumerate(METHODS):
      inside=np.repeat(scores[seed][d][m+'_id'],w['id'][local]);outside=np.repeat(scores[seed][d][m+'_ood'],w[d][local])
      label=np.r_[np.zeros(len(inside)),np.ones(len(outside))];score=np.r_[inside,outside]
      threshold=sorted(inside.tolist())[math.ceil(.95*len(inside))-1]
      expected=np.asarray([roc_auc_score(label,score),average_precision_score(label,score),average_precision_score(1-label,-score),np.mean(outside<=threshold),np.mean(inside<=threshold),threshold])
      agree(a['ranking'][local,seed,di,mi],expected,'expanded independent ranking differs');checks['independent_expanded_ranking_fields']+=6
  for p in [marker,wp,mp]:hashes[str(p.relative_to(ROOT)).replace('\\','/')]=digest(p)
  print('independent block passed:',start//100,flush=True)
 complete=len(markers)==10
 report={'state':'passed' if complete else 'partial_passed','audited_at':now(),'completed_replicates':len(markers)*100,'expected_replicates':1000,'checks':checks,'output_name':attempt,'inputs_sha256':digest(manifest_path),'original_input_hash_checks':len(manifest['original_artifacts']),'output_artifacts':hashes,'independent_source_sha256':digest(__file__),'boundary':'conditional paired image bootstrap; first ten expanded sklearn ranking draws, all weights and operational counts; partial audit never marks benchmark complete'}
 for n,h in manifest['original_artifacts'].items():require(digest(ROOT/n)==h,'input changed during audit')
 write(RUN/('raw-audit.json' if complete else 'block-check-audit.json'),report);print({k:v for k,v in report.items() if k!='output_artifacts'},flush=True)
if __name__=='__main__':main()
