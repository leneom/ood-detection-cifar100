"""Independent all-row recomputation of Places365 scene acceptance diagnostic."""
from __future__ import annotations
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/places365-extension'
OUT=ROOT/'report/places365-scene-diagnostic'
VARIANTS=('T1','calibrated')
METHODS=('msp','energy','maxlogit','entropy','knn','mahalanobis')
TARGETS=(80,90,95,99)
DISPLAY=('MSP','Energy','MaxLogit','Entropy','KNN-50','Pooled Mahalanobis')
QKEYS=('min','q25','median','q75','max')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(ok,why):
 if not ok:raise AssertionError(why)
def quantile_linear(values,p):
 ordered=sorted(map(float,values));pos=(len(ordered)-1)*p;lo=int(pos);hi=min(lo+1,len(ordered)-1)
 return ordered[lo]*(hi-pos)+ordered[hi]*(pos-lo) if hi!=lo else ordered[lo]

def main():
 check(not (OUT/'audit.json').exists(),'audit already exists')
 d=read(RUN/'scene-diagnostic-declaration.json')
 summary=read(OUT/'summary.json')
 saved=arrays(OUT/'scene_values.npz')
 cache=arrays(ROOT/'data/places365-validation.npz')
 labels=cache['labels']
 check(d['protocol_sha256']==sha(ROOT/'docs/PLACES365_SCENE_DIAGNOSTIC.md') and
       summary['declaration_sha256']==sha(RUN/'scene-diagnostic-declaration.json') and
       summary['producer_source_sha256']==sha(ROOT/'tools/build_places365_scene_diagnostic.py') and
       summary['scene_values_sha256']==sha(OUT/'scene_values.npz') and
       summary['scene_csv_sha256']==sha(OUT/'scene_counts.csv'),'frozen output binding')
 for name,expected in d['input_sha256'].items():check(sha(ROOT/name)==expected,'frozen input '+name)
 counts=saved['counts']
 check(counts.shape==(3,2,6,4,365) and saved['row_indices'].shape==(365,100) and
       tuple(saved['variants'])==VARIANTS and tuple(saved['methods'])==METHODS and
       tuple(saved['targets'])==TARGETS and np.array_equal(saved['labels'],np.arange(365)),
       'fixed cube and axes')
 names={}
 for line in (ROOT/'data/places365-categories_places365.txt').read_text(encoding='utf-8').splitlines():
  name,number=line.rsplit(' ',1);names[int(number)]=name
 check(len(names)==365 and all(saved['categories'][i]==names[i] for i in range(365)),'official class names')
 for scene in range(365):
  rows=np.flatnonzero(labels==scene)
  check(len(rows)==100 and np.array_equal(saved['row_indices'][scene],rows),'full exact scene rows')
 expected=np.empty_like(counts)
 for seed in range(3):
  scores=arrays(RUN/f'score-seed{seed}/scores.npz')
  masks=arrays(RUN/f'score-seed{seed}/acceptance_masks.npz')
  manifest=read(RUN/f'score-seed{seed}/manifest.json')
  for vi,variant in enumerate(VARIANTS):
   for mi,method in enumerate(METHODS):
    vector=scores[f'{variant}_{method}']
    check(vector.shape==(36500,) and np.isfinite(vector).all(),'full raw score')
    for ti,target in enumerate(TARGETS):
     t=f'target{target}'
     threshold=manifest['results'][variant][method]['operating_grid'][t]['threshold']
     decisions=np.less_equal(vector,threshold).astype(np.uint8)
     check(np.array_equal(decisions,masks[f'{variant}_{method}_{t}_OOD']),'raw score mask reconstruction')
     expected[seed,vi,mi,ti]=np.bincount(labels,weights=decisions,minlength=365).astype(np.int16)
     check(np.array_equal(expected[seed,vi,mi,ti],counts[seed,vi,mi,ti]),'per-scene counts')
     check(int(expected[seed,vi,mi,ti].sum())==manifest['results'][variant][method]['operating_grid'][t]['ood_accepted'],
           'manifest aggregate identity')
 with (OUT/'scene_counts.csv').open(newline='',encoding='utf-8') as stream:
  csvrows=list(csv.DictReader(stream))
 check(len(csvrows)==17520,'complete scene CSV')
 n=0
 for vi,variant in enumerate(VARIANTS):
  for mi,method in enumerate(METHODS):
   for ti,target in enumerate(TARGETS):
    r=summary['results'][variant][method][f'target{target}']
    scene_rates=[sum(map(int,expected[:,vi,mi,ti,scene]))/300 for scene in range(365)]
    check(r['scene_count']==365 and r['examples_per_scene_per_seed']==100,
          'summary scene denominators')
    for key,p in zip(QKEYS,(0,.25,.5,.75,1)):
     check(abs(r['quantiles'][key]-quantile_linear(scene_rates,p))<1e-12,'independent quantile')
    total=int(expected[:,vi,mi,ti,:].sum())
    check(r['seed_accepted']==[int(x) for x in expected[:,vi,mi,ti,:].sum(axis=1)] and
          abs(r['overall_mean_acceptance']-total/(3*36500))<1e-12 and
          abs(sum(scene_rates)/365-total/(3*36500))<1e-12,
          'balanced scene aggregate identity')
    for scene in range(365):
     row=csvrows[n];n+=1
     points=[int(x) for x in expected[:,vi,mi,ti,scene]]
     check((row['variant'],row['method'],int(row['target']),int(row['scene_label']),row['scene_name'])==
           (variant,method,target,scene,names[scene]) and
           int(row['denominator_per_seed'])==100 and
           [int(row[f'seed{i}_accepted']) for i in range(3)]==points and
           int(row['total_accepted'])==sum(points) and
           abs(float(row['mean_acceptance_rate'])-sum(points)/300)<1e-12,'complete CSV row')
 report=(OUT/'REPORT.md').read_text(encoding='utf-8')
 for variant in VARIANTS:
  check(f'## {variant}\n' in report,'display section')
  section=report.split(f'## {variant}\n',1)[1].split('## ',1)[0]
  for method,name in zip(METHODS,DISPLAY):
   r=summary['results'][variant][method]['target95']
   values=[r['quantiles'][q] for q in QKEYS]+[r['overall_mean_acceptance']]
   row='| '+name+' | '+' | '.join(f'{100*x:.2f}' for x in values)+' |'
   check(section.count(row)==1,'displayed numeric row')
 png=OUT/'fixed95-scene-distribution.png';svg=OUT/'fixed95-scene-distribution.svg'
 with Image.open(png) as im:im.verify()
 with Image.open(png) as im:check(im.width>=1800 and im.height>=900,'legible plotted canvas')
 check('<svg' in svg.read_text(encoding='utf-8')[:500] and
       '[fixed95 distribution figure](fixed95-scene-distribution.png)' in report,
       'report figure linkage')
 record={'state':'passed','checked':{'scene_counts':int(counts.size),'csv_rows':len(csvrows),
          'summary_cells':2*6*4,'display_rows':12,'figures':2},
         'declaration_sha256':sha(RUN/'scene-diagnostic-declaration.json'),
         'summary_sha256':sha(OUT/'summary.json'),'cube_sha256':sha(OUT/'scene_values.npz'),
         'csv_sha256':sha(OUT/'scene_counts.csv'),'report_sha256':sha(OUT/'REPORT.md'),
         'figure_sha256':{p.name:sha(p) for p in (png,svg)},
         'source_sha256':sha(__file__),'audited_at':datetime.now(timezone.utc).isoformat(),
         'boundary':'Independent raw-score threshold reconstruction of post-result descriptive scene diagnostic.'}
 (OUT/'audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'state':'passed','checked':record['checked']}))

if __name__=='__main__':main()