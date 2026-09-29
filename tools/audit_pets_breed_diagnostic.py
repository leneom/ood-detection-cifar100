"""Independent all-row audit of frozen Pet breed/species acceptance."""
from __future__ import annotations
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/oxford-pets-breed-diagnostic'
OUT=ROOT/'report/oxford-pets-breed-diagnostic'
METHODS=('msp','energy','maxlogit','entropy','knn','mahalanobis')
VARIANTS=('T1','calibrated');TARGETS=(80,90,95,99)

def check(ok,msg):
 if not ok:raise AssertionError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while True:
   block=stream.read(1<<20)
   if not block:break
   h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def close(a,b,msg):check(math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),msg)
def format_seed(values):return f'{100*statistics.mean(values):.2f} ± {100*statistics.stdev(values):.2f}'

def main():
 from tools.verify_pets_completion import validate as verify_pet
 from tools.verify_pets_paired_decisions import validate as verify_pair
 for completion,validator in ((ROOT/'report/oxford-pets-extension/completion.json',verify_pet),
                              (ROOT/'report/oxford-pets-paired-decisions/completion.json',verify_pair)):
  record=read(completion)
  check(record['state']=='complete' and record['inputs_sha256']==validator(),
        'earlier hash-bound completion '+str(completion))
 d=read(BASE/'declaration.json')
 check(d['state']=='frozen_before_group_acceptance_calculation' and
       d['seeds']==[0,1,2] and d['variants']==list(VARIANTS) and
       d['methods']==list(METHODS) and d['targets']==list(TARGETS) and
       d['breed_labels']==list(range(1,38)) and d['species_codes']==[1,2] and
       d['ood_count']==3669 and len(d['input_sha256'])==14,'complete frozen grid')
 for rel,expected in d['input_sha256'].items():check(sha(ROOT/rel)==expected,'frozen source '+rel)
 with np.load(ROOT/'data/pets-test.npz',allow_pickle=False) as z:
  labels=z['labels'].copy(); species=z['species'].copy(); filenames=z['filenames'].copy()
 check(labels.shape==species.shape==filenames.shape==(3669,) and
       set(map(int,labels))==set(range(1,38)) and set(map(int,species))=={1,2},
       'cache census')
 breed_rows=[np.flatnonzero(labels==i) for i in range(1,38)]
 species_rows=[np.flatnonzero(species==i) for i in (1,2)]
 names=[]
 for indices in breed_rows:
  prefixes={str(filenames[i]).rsplit('_',1)[0] for i in indices}
  check(len(prefixes)==1 and len(set(map(int,species[indices])))==1,
        'one official filename prefix/species per breed label')
  names.append(next(iter(prefixes)))
 check(len(set(names))==37,'distinct breed names')
 with np.load(OUT/'group_values.npz',allow_pickle=False) as z:
  cube={name:z[name].copy() for name in z.files}
 check(set(cube)=={'breed_counts','species_counts','breed_sizes','species_sizes',
                  'breed_order','breed_offsets','species_order','species_offsets',
                  'breed_names','breed_labels','species_codes','variants','methods','targets'},
       'complete saved arrays')
 check(cube['breed_counts'].shape==(3,2,6,4,37) and
       cube['species_counts'].shape==(3,2,6,4,2) and
       np.array_equal(cube['breed_sizes'],[len(x) for x in breed_rows]) and
       np.array_equal(cube['species_sizes'],[len(x) for x in species_rows]) and
       np.array_equal(cube['breed_names'],names) and
       np.array_equal(cube['breed_labels'],range(1,38)) and
       np.array_equal(cube['species_codes'],[1,2]) and
       np.array_equal(cube['variants'],VARIANTS) and
       np.array_equal(cube['methods'],METHODS) and
       np.array_equal(cube['targets'],TARGETS), 'array census')
 for kind,rows in (('breed',breed_rows),('species',species_rows)):
  order=cube[f'{kind}_order'];offsets=cube[f'{kind}_offsets']
  check(order.shape==(3669,) and offsets.shape==(len(rows)+1,) and
        int(offsets[0])==0 and int(offsets[-1])==3669 and
        np.array_equal(np.sort(order),np.arange(3669)),'complete '+kind+' partition')
  for i,indices in enumerate(rows):
   check(np.array_equal(order[offsets[i]:offsets[i+1]],indices),
         f'{kind} saved row membership {i}')
 check(cube['breed_counts'].dtype.kind in 'iu' and
       cube['species_counts'].dtype.kind in 'iu','integer counts')
 with (OUT/'group_counts.csv').open(newline='',encoding='utf-8') as f:
  csv_rows=list(csv.DictReader(f))
 check(len(csv_rows)==1872,'complete CSV census')
 lookup={}
 for row in csv_rows:
  key=(row['variant'],row['method'],int(row['target']),row['group_type'],int(row['group_code']))
  check(key not in lookup,'duplicate group CSV row')
  lookup[key]=row
 summary=read(OUT/'summary.json')
 check(summary['state']=='built_pending_independent_audit' and
       summary['declaration_sha256']==sha(BASE/'declaration.json') and
       summary['cache_sha256']==sha(ROOT/'data/pets-test.npz') and
       summary['group_count_cells']==5616 and summary['csv_rows']==1872,
       'summary input binding')
 report=(OUT/'REPORT.md').read_text(encoding='utf-8').splitlines()
 checked=0; mask_count=0; display=0
 for seed in range(3):
  path=ROOT/f'runs/oxford-pets-extension/score-seed{seed}'
  manifest=read(path/'manifest.json')
  check(manifest['seed']==seed and manifest['n_ood']==3669,'seed manifest')
  with (np.load(path/'scores.npz',allow_pickle=False) as raw,
        np.load(path/'acceptance_masks.npz',allow_pickle=False) as old_masks):
   for vi,variant in enumerate(VARIANTS):
    for mi,method in enumerate(METHODS):
     for ti,target in enumerate(TARGETS):
      point=manifest['results'][variant][method]['operating_grid'][f'target{target}']
      key=f'{variant}_{method}'
      accept=np.less_equal(raw[key],point['threshold'])
      check(accept.shape==(3669,) and
            np.array_equal(accept,old_masks[f'{key}_target{target}_OOD']) and
            int(np.count_nonzero(accept))==point['ood_accepted'],
            'raw score / old threshold / mask identity')
      mask_count+=1
      for kind,groups in (('breed',breed_rows),('species',species_rows)):
       counts=np.array([int(np.count_nonzero(accept[indices])) for indices in groups])
       check(np.array_equal(counts,cube[f'{kind}_counts'][seed,vi,mi,ti]) and
             int(counts.sum())==point['ood_accepted'],
             f'{seed}/{variant}/{method}/{target}/{kind} group count')
       checked+=len(groups)
 check(checked==5616 and mask_count==144,'full count/mask grid')
 for vi,variant in enumerate(VARIANTS):
  for mi,method in enumerate(METHODS):
   for ti,target in enumerate(TARGETS):
    point=summary['results'][variant][method][f'target{target}']
    breed=cube['breed_counts'][:,vi,mi,ti,:]
    species_cube=cube['species_counts'][:,vi,mi,ti,:]
    rates=(breed/cube['breed_sizes']).mean(axis=0)
    quant=np.quantile(rates,[0,.25,.5,.75,1],method='linear')
    check(point['breed_count']==37 and
          point['breed_size_min']==int(cube['breed_sizes'].min()) and
          point['breed_size_max']==int(cube['breed_sizes'].max()),'breed size metadata')
    for name,value in zip(('min','q25','median','q75','max'),quant):
     close(point['breed_quantiles'][name],value,'all breed quantiles')
    all_rates=[int(breed[s].sum())/3669 for s in range(3)]
    for key,values in [('overall',all_rates)]+[(f'species:{code}',
       [int(species_cube[s,code-1])/len(species_rows[code-1]) for s in range(3)])
       for code in (1,2)]:
     record=point['overall'] if key=='overall' else point['species'][key.split(':')[1]]
     if key!='overall':check(record['n']==len(species_rows[int(key[-1])-1]),'species size')
     for a,b in zip(record['per_seed'],values):close(a,b,'seed rate')
     close(record['mean'],statistics.mean(values),'mean rate')
     close(record['sample_sd'],statistics.stdev(values),'sample SD')
    for kind,counts,sizes,group_names in (
        ('breed',breed,cube['breed_sizes'],names),
        ('species',species_cube,cube['species_sizes'],['species_1','species_2'])):
     for gi,name in enumerate(group_names):
      row=lookup.pop((variant,method,target,kind,gi+1))
      points=[int(x) for x in counts[:,gi]]
      check(row['group_name']==name and int(row['denominator_per_seed'])==int(sizes[gi]) and
            [int(row[f'seed{s}_accepted']) for s in range(3)]==points and
            int(row['total_accepted'])==sum(points),'complete CSV row')
      close(row['mean_acceptance_rate'],sum(points)/(3*int(sizes[gi])),'CSV mean')
    if target==95:
     q=point['breed_quantiles']
     vals=' / '.join(f'{100*q[k]:.2f}' for k in ('min','median','max'))
     line=(f"| {variant} | {method} | {format_seed(all_rates)} | "
           f"{format_seed(point['species']['1']['per_seed'])} | "
           f"{format_seed(point['species']['2']['per_seed'])} | {vals} |")
     check(line in report,'fixed95 displayed all-method row')
     display+=1
 check(not lookup and display==12 and len(summary['results'])==2,
       'full CSV/report/summary grid')
 result={'state':'independently_audited','declaration_sha256':sha(BASE/'declaration.json'),
         'group_counts_checked':checked,'source_masks_checked':mask_count,
         'csv_rows_checked':1872,'summary_cells_checked':48,'display_rows_checked':display,
         'breed_groups':37,'species_groups':2}
 path=OUT/'audit.json'
 if path.exists():check(read(path)==result,'read-only retained audit identity')
 else:path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result))

if __name__=='__main__':main()