"""Frozen full-benchmark bootstrap inputs and row-alignment evidence."""
from pathlib import Path
import ast
import numpy as np
from geometry_io import ROOT,read,load,digest,require,now
RUN=ROOT/'runs/paired-bootstrap'
SOURCE=ROOT/'runs/ash-phase5'
METHODS=['msp','energy','maxlogit','entropy','knn','mahalanobis','vim','residual','react_energy','affine_energy','rmd','mahalanobis_cpu','ash_p_energy','ash_b_energy','ash_s_energy']
DATASETS=['cifar10','mnist','svhn','textures','noise']
NATURAL=DATASETS[:4]
COUNTS={'id':10000,'cifar10':10000,'mnist':10000,'svhn':26032,'textures':1880,'noise':10000}
TARGETS=[.8,.9,.95,.99]

def draw_counts(rep):
 rng=np.random.default_rng(8939+rep)
 return {d:np.bincount(rng.integers(0,n,size=n),minlength=n).astype(np.uint32) for d,n in COUNTS.items()}

def bundle():
 pre=read(RUN/'predeclaration.json')
 require(pre['seeds']==[0,1,2] and pre['methods']==METHODS and pre['datasets']==DATASETS and pre['targets']==TARGETS and pre['bootstrap_replicates']==1000 and pre['block_size']==100 and pre['rng_base']==8939 and pre['pair_count']==105,'bootstrap declaration changed')
 for name,h in pre['original_artifacts'].items():require(digest(ROOT/name)==h,'frozen bootstrap input changed '+name)
 require(read(SOURCE/'status.json')['state']=='complete','ASH score source incomplete')
 score_banks=[];mask_banks=[];results=[];provenance=[];labels0=None;pixel0=None;data_root=None;snapshot_hashes={}
 for seed in range(3):
  out=SOURCE/f'seed{seed}/evaluation';r=read(out/'results.json');require(r['methods']==METHODS and r['grid_targets']==TARGETS,'score settings changed')
  scores={d:load(out/f'{d}_scores.npz') for d in DATASETS};masks=load(out/'decisions.npz')
  original=ROOT/f'runs/cifar100-seed{seed}';labels=load(original/'eval-final/id_logits.npz')['labels']
  if seed==0:labels0=labels
  require(np.array_equal(labels,labels0),'ID row labels differ across classifiers')
  for d in DATASETS:
   for m in METHODS:
    require(scores[d][m+'_id'].shape==(10000,) and scores[d][m+'_ood'].shape==(COUNTS[d],),'score row count changed')
    require(np.array_equal(scores[d][m+'_id'],scores['cifar10'][m+'_id']),'canonical ID score changed across datasets')
  evidence={'seed':seed,'captured_pixels':{},'historical_source_alignment':{}}
  for folder in ['eval-final','eval-extension-svhn','eval-extension-textures']:
   historical=read(original/f'{folder}/results.json');config=historical['training_config']
   if data_root is None:data_root=config['data_root']
   require(config['data_root']==data_root and config['dataset']=='cifar100' and config['epochs']==100,'shared data root/protocol changed')
   require(historical['checkpoint_sha256']==r['checkpoint_sha256'],'historical checkpoint changed')
   hashes=read(original/f'{folder}/evaluation_source_hashes.json')
   for filename,h in hashes.items():require(digest(original/f'{folder}/evaluation_source/{filename}')==h,'executed historical source changed')
   loader_hashes={k:hashes[k] for k in ['evaluate.py','datasets.py','data.py','cache.py']}
   if seed==0:snapshot_hashes[folder]=loader_hashes
   require(loader_hashes==snapshot_hashes[folder],'population loader/source differs across seed')
   evidence['historical_source_alignment'][folder]=loader_hashes
   if folder=='eval-extension-textures':
    pixel={k:historical['input_pixels'][k] for k in ['id_test']};pixel['textures']=historical['input_pixels']['ood']['textures']
    if seed==0:pixel0=pixel
    require(pixel==pixel0,'captured ID/Textures pixels differ across seed');evidence['captured_pixels']=pixel
   if folder=='eval-final':
    tree=ast.parse((original/f'{folder}/evaluation_source/datasets.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_ood')
    require(ast.literal_eval(fn.args.defaults[-1])==110,'synthetic noise seed changed')
    tree=ast.parse((original/f'{folder}/evaluation_source/evaluate.py').read_text(encoding='utf-8'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_ood']
    require(len(calls)==1 and len(calls[0].args)==4 and not calls[0].keywords,'historical OOD call overrides common defaults')
  score_banks.append(scores);mask_banks.append(masks);results.append(r);provenance.append(evidence)
 alignment={'state':'passed','checked_at':now(),'per_seed':provenance,'common_data_root':data_root,'noise_rng_seed':110,'boundary':'ID/Textures captured executed pixel hashes; earlier natural/noise populations use matched archived executed loaders, fixed source archives/caches/order and noise RNG; no retroactive executed pixel hash fabricated'}
 return score_banks,mask_banks,results,alignment
