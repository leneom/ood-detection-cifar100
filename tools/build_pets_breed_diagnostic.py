"""Build the frozen post-result Pet breed/species acceptance census."""
from __future__ import annotations
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/oxford-pets-breed-diagnostic'
OUT=ROOT/'report/oxford-pets-breed-diagnostic'
METHODS=('msp','energy','maxlogit','entropy','knn','mahalanobis')
VARIANTS=('T1','calibrated')
TARGETS=(80,90,95,99)
QKEYS=('min','q25','median','q75','max')

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def fmt(values):return f'{100*statistics.mean(values):.2f} ± {100*statistics.stdev(values):.2f}'
def part(values,start,stop):
 order=np.argsort(values,kind='stable').astype(np.int32)
 sizes=np.bincount(values,minlength=stop+1)[start:stop+1].astype(np.int32)
 offsets=np.r_[0,np.cumsum(sizes)].astype(np.int32)
 require(int(offsets[-1])==3669 and np.array_equal(np.sort(order),np.arange(3669)),
         'complete mutually exclusive partition')
 return order,sizes,offsets

def main():
 require(not OUT.exists(),'write-once breed diagnostic output exists')
 d=read(BASE/'declaration.json')
 require(d['state']=='frozen_before_group_acceptance_calculation' and
         d['seeds']==[0,1,2] and tuple(d['variants'])==VARIANTS and
         tuple(d['methods'])==METHODS and tuple(d['targets'])==TARGETS and
         d['breed_labels']==list(range(1,38)) and d['species_codes']==[1,2] and
         d['ood_count']==3669 and len(d['input_sha256'])==14,'frozen census')
 for name,expected in d['input_sha256'].items():require(sha(ROOT/name)==expected,'source changed '+name)
 with np.load(ROOT/'data/pets-test.npz',allow_pickle=False) as cache:
  labels=cache['labels'].astype(np.int16).copy()
  species=cache['species'].astype(np.int8).copy()
  names=cache['filenames'].copy()
 require(labels.shape==species.shape==names.shape==(3669,) and
         set(map(int,labels))==set(range(1,38)) and set(map(int,species))=={1,2},
         'official full group labels')
 prefix=[]
 for name in names:
  left,sep,right=str(name).rpartition('_')
  require(sep=='_' and right.lower().endswith('.jpg') and
          right[:-4].isdigit() and bool(left),'official breed filename format')
  prefix.append(left)
 breed_names=[]
 for label in range(1,38):
  indices=np.flatnonzero(labels==label)
  names_for_label={prefix[i] for i in indices}
  require(len(names_for_label)==1 and len(set(map(int,species[indices])))==1,
          'one name/species per class')
  breed_names.append(next(iter(names_for_label)))
 require(len(set(breed_names))==37,'distinct official breed names')
 breed_order,breed_sizes,breed_offsets=part(labels,1,37)
 species_order,species_sizes,species_offsets=part(species,1,2)
 breed_counts=np.empty((3,2,6,4,37),dtype=np.int16)
 species_counts=np.empty((3,2,6,4,2),dtype=np.int16)
 for seed in range(3):
  path=ROOT/f'runs/oxford-pets-extension/score-seed{seed}'
  manifest=read(path/'manifest.json')
  require(manifest['seed']==seed and manifest['n_ood']==3669,'frozen seed manifest')
  with np.load(path/'acceptance_masks.npz',allow_pickle=False) as masks:
   for vi,variant in enumerate(VARIANTS):
    for mi,method in enumerate(METHODS):
     for ti,target in enumerate(TARGETS):
      key=f'{variant}_{method}_target{target}_OOD'
      accepted=np.asarray(masks[key],dtype=bool)
      require(accepted.shape==(3669,),'complete Pet decision vector')
      breed_counts[seed,vi,mi,ti]=np.bincount(labels,weights=accepted.astype(np.int16),minlength=38)[1:38]
      species_counts[seed,vi,mi,ti]=np.bincount(species,weights=accepted.astype(np.int16),minlength=3)[1:3]
      total=manifest['results'][variant][method]['operating_grid'][f'target{target}']['ood_accepted']
      require(int(breed_counts[seed,vi,mi,ti].sum())==
              int(species_counts[seed,vi,mi,ti].sum())==int(total),'aggregate conservation')
 OUT.mkdir(parents=True)
 np.savez_compressed(OUT/'group_values.npz',breed_counts=breed_counts,
  species_counts=species_counts,breed_sizes=breed_sizes,species_sizes=species_sizes,
  breed_order=breed_order,breed_offsets=breed_offsets,
  species_order=species_order,species_offsets=species_offsets,
  breed_names=np.array(breed_names),breed_labels=np.arange(1,38,dtype=np.int16),
  species_codes=np.array([1,2],dtype=np.int8),variants=np.array(VARIANTS),
  methods=np.array(METHODS),targets=np.array(TARGETS))
 with (OUT/'group_counts.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f)
  writer.writerow(['variant','method','target','group_type','group_code','group_name',
                   'denominator_per_seed','seed0_accepted','seed1_accepted',
                   'seed2_accepted','total_accepted','mean_acceptance_rate'])
  for vi,variant in enumerate(VARIANTS):
   for mi,method in enumerate(METHODS):
    for ti,target in enumerate(TARGETS):
     for kind,counts,sizes,group_names in (
        ('breed',breed_counts,breed_sizes,breed_names),
        ('species',species_counts,species_sizes,['species_1','species_2'])):
      for gi,name in enumerate(group_names):
       points=[int(x) for x in counts[:,vi,mi,ti,gi]]
       n=int(sizes[gi])
       writer.writerow([variant,method,target,kind,gi+1,name,n,*points,
                        sum(points),sum(points)/(3*n)])
 results={}
 for vi,variant in enumerate(VARIANTS):
  results[variant]={}
  for mi,method in enumerate(METHODS):
   results[variant][method]={}
   for ti,target in enumerate(TARGETS):
    breed_rates=(breed_counts[:,vi,mi,ti,:]/breed_sizes).mean(axis=0)
    q=np.quantile(breed_rates,[0,.25,.5,.75,1],method='linear')
    overall=[int(breed_counts[seed,vi,mi,ti].sum())/3669 for seed in range(3)]
    species_rates={str(code):[int(species_counts[seed,vi,mi,ti,code-1])/int(species_sizes[code-1])
                                for seed in range(3)] for code in (1,2)}
    results[variant][method][f'target{target}']={
      'breed_quantiles':dict(zip(QKEYS,map(float,q))),
      'breed_count':37,'breed_size_min':int(breed_sizes.min()),
      'breed_size_max':int(breed_sizes.max()),
      'overall':{'per_seed':overall,'mean':statistics.mean(overall),
                 'sample_sd':statistics.stdev(overall)},
      'species':{code:{'n':int(species_sizes[int(code)-1]),'per_seed':rates,
                       'mean':statistics.mean(rates),'sample_sd':statistics.stdev(rates)}
                 for code,rates in species_rates.items()}}
 summary={'state':'built_pending_independent_audit',
          'declaration_sha256':sha(BASE/'declaration.json'),
          'cache_sha256':sha(ROOT/'data/pets-test.npz'),
          'group_count_cells':5616,'csv_rows':1872,'results':results,
          'boundary':'Post-result complete breed/species descriptive rates; no method selection.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 lines=['# Oxford-IIIT Pet full breed/species fixed-threshold diagnostic','',
  'Post-result descriptive analysis of all 3,669 official test images. The two official species codes are kept numeric. '
  'No breed or image is filtered, and no method or threshold is selected from these OOD outcomes.','',
  'For each frozen method below, values are three training-seed means ± sample SD (%) at the existing 95% '
  'ID-validation target. Breed min/median/max are quantiles of 37 unweighted breed-level three-seed mean rates; '
  'breed denominators differ, so these are not the overall Pet rate.','',
  '| Variant | Method | All Pet | Species 1 | Species 2 | Breed min / median / max |',
  '| --- | --- | ---: | ---: | ---: | ---: |']
 for variant in VARIANTS:
  for method in METHODS:
   point=results[variant][method]['target95']
   q=point['breed_quantiles']
   vals=' / '.join(f'{100*q[k]:.2f}' for k in ('min','median','max'))
   lines.append(f"| {variant} | {method} | {fmt(point['overall']['per_seed'])} | "
                f"{fmt(point['species']['1']['per_seed'])} | "
                f"{fmt(point['species']['2']['per_seed'])} | {vals} |")
 lines += ['',
  'The [complete group CSV](group_counts.csv) retains all 37 breeds, both species codes, both variants, six methods, '
  'four frozen targets and three seed counts. The [summary](summary.json) retains all 48 setting distributions; '
  'the [group arrays](group_values.npz) retain counts, denominators and complete row partitions. '
  'Different breed/source/resizing characteristics are confounded. The three trained seeds are not '
  'a population-level confidence interval, and these previously viewed OOD examples cannot validate a new detector.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps({'state':summary['state'],'group_count_cells':5616,'csv_rows':1872,
                   'declaration_sha256':summary['declaration_sha256']}))

if __name__=='__main__':main()