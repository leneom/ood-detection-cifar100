"""One CPU controller; immutable 100-draw blocks of the fixed 1,000-image schedule."""
from pathlib import Path
import os,argparse,re,shutil,time
import numpy as np
from geometry_io import ROOT,read,write,digest,save_arrays,require,now
from bootstrap_io import RUN,METHODS,DATASETS,COUNTS,TARGETS,bundle,draw_counts
from bootstrap_core import RankingPlan,operational_counts,RANKING

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='analysis');parser.add_argument('--stop-after-blocks',type=int,default=10);args=parser.parse_args()
 require(re.fullmatch(r'analysis(?:-attempt-\d{3})?',args.output_name) and 1<=args.stop_after_blocks<=10,'invalid fixed-schedule pause/name')
 import msvcrt
 with (RUN/'controller.lock').open('a+b') as lock:
  lock.seek(0)
  if not lock.read(1):lock.write(b'0');lock.flush()
  lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  try:
   require(read(RUN/'preflight.json')['state']=='passed','kernel/point preflight incomplete')
   pf=read(RUN/'preflight-inputs.json')
   for name,h in pf['source_hashes'].items():require(digest(ROOT/name)==digest(RUN/'preflight-source'/Path(name).name)==h,'preflight-executed source changed')
   require(read(RUN/'preflight.json')['inputs_sha256']==digest(RUN/'preflight-inputs.json'),'preflight inputs changed')
   inputs=dict(pf['original_artifacts'])
   for name in ['preflight-inputs.json','preflight.json','row-alignment.json']:inputs['runs/paired-bootstrap/'+name]=digest(RUN/name)
   for name,h in inputs.items():require(digest(ROOT/name)==h,'immutable input changed '+name)
   files=[Path(__file__),ROOT/'tools/bootstrap_io.py',ROOT/'tools/bootstrap_core.py',ROOT/'tools/geometry_io.py']
   sources={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files};manifest_path=RUN/f'{args.output_name}-inputs.json'
   if manifest_path.exists():
    manifest=read(manifest_path);require(manifest['original_artifacts']==inputs and manifest['source_hashes']==sources,'incompatible bootstrap resume')
   else:
    snap=RUN/f'{args.output_name}-source';snap.mkdir(exist_ok=False)
    for p in files:shutil.copy2(p,snap/p.name)
    manifest={'state':'frozen_before_indexed_full_bootstrap_draws','prepared_at':now(),'original_artifacts':inputs,'source_hashes':sources,'replicates':1000,'block_size':100,'methods':METHODS,'datasets':DATASETS,'targets':TARGETS,'numpy':np.__version__,'ranking_fields':RANKING,'output_name':args.output_name}
    write(manifest_path,manifest)
   scores,masks,results,alignment=bundle();require(alignment['per_seed']==read(RUN/'row-alignment.json')['per_seed'],'row alignment changed')
   plans=[{d:[RankingPlan(scores[seed][d][m+'_id'],scores[seed][d][m+'_ood']) for m in METHODS] for d in DATASETS} for seed in range(3)]
   output=RUN/args.output_name;output.mkdir(exist_ok=True);completed=[]
   for block in range(10):
    marker=output/f'block-{block:03d}.json';wp=output/f'block-{block:03d}-weights.npz';mp=output/f'block-{block:03d}-metrics.npz'
    if marker.exists():
     previous=read(marker);require(previous['inputs_sha256']==digest(manifest_path) and previous['replicate_start']==block*100 and previous['replicate_end_exclusive']==(block+1)*100,'block identity changed')
     require(digest(wp)==previous['weights_sha256'] and digest(mp)==previous['metrics_sha256'],'committed bootstrap block changed');completed.append(block);continue
    if block>=args.stop_after_blocks:break
    require(not wp.exists() and not mp.exists(),'partial block retained; use a fresh output-name')
    start=time.monotonic();write(RUN/'status.json',{'state':'running','pid':os.getpid(),'output_name':args.output_name,'stage':'CPU fixed image bootstrap','block':block,'completed_blocks':completed,'updated_at':now()})
    draws={d:np.empty((100,n),dtype=np.uint32) for d,n in COUNTS.items()};replicates=np.arange(block*100,(block+1)*100,dtype=np.int64)
    ranking=np.empty((100,3,5,15,6),dtype=float);id_a=np.empty((100,3,15,4),dtype=np.int64);id_bad=np.empty_like(id_a)
    oo_a=np.empty((100,3,15,4,5),dtype=np.int64);coverage=np.empty_like(id_a,dtype=float);risk=np.empty_like(coverage);accept=np.empty_like(oo_a,dtype=float)
    for local,rep in enumerate(replicates):
     count=draw_counts(int(rep))
     for d,n in COUNTS.items():require(count[d].dtype==np.uint32 and int(count[d].sum())==n,'draw count invariant failed');draws[d][local]=count[d]
     wi=count['id'];wo={d:count[d] for d in DATASETS}
     for seed in range(3):
      id_a[local,seed],id_bad[local,seed],oo_a[local,seed],coverage[local,seed],risk[local,seed],accept[local,seed]=operational_counts(wi,wo,masks[seed],masks[seed]['id_correct'],DATASETS)
      for di,d in enumerate(DATASETS):
       for mi,plan in enumerate(plans[seed][d]):ranking[local,seed,di,mi]=plan.evaluate(wi,wo[d])
    save_arrays(wp,replicates=replicates,**draws)
    save_arrays(mp,replicates=replicates,ranking=ranking,id_accepted=id_a,id_incorrect_accepted=id_bad,ood_accepted=oo_a,id_coverage=coverage,retained_error_risk=risk,ood_acceptance=accept,natural_ood_acceptance_macro=accept[...,:4].mean(-1),natural_ranking_macro=ranking[:,:,:4,:,:4].mean(2))
    write(marker,{'state':'committed','computed_at':now(),'inputs_sha256':digest(manifest_path),'replicate_start':block*100,'replicate_end_exclusive':(block+1)*100,'weights_sha256':digest(wp),'metrics_sha256':digest(mp),'elapsed_seconds':time.monotonic()-start,'boundary':'all fifteen columns and three models use identical image counts; fixed operational masks'})
    completed.append(block);print({'block':block,'completed_replicates':len(completed)*100,'elapsed_seconds':round(time.monotonic()-start,2)},flush=True)
   for name,h in inputs.items():require(digest(ROOT/name)==h,'input changed during bootstrap')
   for name,h in sources.items():require(digest(ROOT/name)==h,'source changed during bootstrap')
   state='raw_complete_pending_independent_audit' if len(completed)==10 else 'paused_at_fixed_block_boundary'
   write(RUN/'status.json',{'state':state,'output_name':args.output_name,'completed_blocks':completed,'completed_replicates':len(completed)*100,'expected_replicates':1000,'original_artifacts_unchanged':True,'updated_at':now()})
  except Exception as error:
   write(RUN/'status.json',{'state':'failed','output_name':args.output_name,'pid':os.getpid(),'error':repr(error),'updated_at':now()});raise
  finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
if __name__=='__main__':main()
