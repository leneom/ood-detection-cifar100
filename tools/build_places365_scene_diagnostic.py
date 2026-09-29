"""Produce the declared post-result Places365 all-scene acceptance diagnostic."""
from __future__ import annotations
import csv, hashlib, json
from datetime import datetime,timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
def require(ok,why):
 if not ok:raise AssertionError(why)
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
 require(not OUT.exists(),'diagnostic output already exists')
 d=read(RUN/'scene-diagnostic-declaration.json')
 require(d['state']=='declared_after_aggregate_before_scene_stratification' and
         d['protocol_sha256']==sha(ROOT/'docs/PLACES365_SCENE_DIAGNOSTIC.md') and
         d['seeds']==[0,1,2] and tuple(d['variants'])==VARIANTS and
         tuple(d['methods'])==METHODS and tuple(d['targets'])==TARGETS,
         'frozen diagnostic declaration')
 for name,expected in d['input_sha256'].items():require(sha(ROOT/name)==expected,'frozen input '+name)
 cache=arrays(ROOT/'data/places365-validation.npz')
 labels=cache['labels']
 require(labels.shape==(36500,) and set(labels.tolist())==set(range(365)),'official labels')
 row_index=np.stack([np.flatnonzero(labels==i) for i in range(365)]).astype(np.int32)
 require(row_index.shape==(365,100) and np.array_equal(np.sort(row_index.ravel()),np.arange(36500)),
         'full balanced row partition')
 cats={}
 for line in (ROOT/'data/places365-categories_places365.txt').read_text(encoding='utf-8').splitlines():
  name,key=line.rsplit(' ',1); k=int(key)
  require(k not in cats,'unique official category');cats[k]=name
 require(set(cats)==set(range(365)),'complete official category list')
 counts=np.empty((3,2,6,4,365),dtype=np.int16)
 for seed in range(3):
  masks=arrays(RUN/f'score-seed{seed}/acceptance_masks.npz')
  manifest=read(RUN/f'score-seed{seed}/manifest.json')
  require(manifest['seed']==seed and manifest['n_ood']==36500,'seed manifest')
  for vi,variant in enumerate(VARIANTS):
   for mi,method in enumerate(METHODS):
    for ti,target in enumerate(TARGETS):
     vector=masks[f'{variant}_{method}_target{target}_OOD']
     require(vector.shape==(36500,) and np.isin(vector,[0,1]).all(),'binary full mask')
     counts[seed,vi,mi,ti]=vector[row_index].sum(axis=1)
     expected=manifest['results'][variant][method]['operating_grid'][f'target{target}']['ood_accepted']
     require(int(counts[seed,vi,mi,ti].sum())==expected,'aggregate identity')
 OUT.mkdir(parents=True)
 np.savez_compressed(OUT/'scene_values.npz',counts=counts,row_indices=row_index,
     labels=np.arange(365,dtype=np.int16),categories=np.array([cats[i] for i in range(365)]),
     variants=np.array(VARIANTS),methods=np.array(METHODS),targets=np.array(TARGETS))
 with (OUT/'scene_counts.csv').open('w',newline='',encoding='utf-8') as stream:
  writer=csv.writer(stream)
  writer.writerow(['variant','method','target','scene_label','scene_name','denominator_per_seed',
                   'seed0_accepted','seed1_accepted','seed2_accepted','total_accepted',
                   'mean_acceptance_rate'])
  for vi,variant in enumerate(VARIANTS):
   for mi,method in enumerate(METHODS):
    for ti,target in enumerate(TARGETS):
     for scene in range(365):
      points=[int(x) for x in counts[:,vi,mi,ti,scene]]
      writer.writerow([variant,method,target,scene,cats[scene],100,*points,sum(points),sum(points)/300])
 results={}
 for vi,variant in enumerate(VARIANTS):
  results[variant]={}
  for mi,method in enumerate(METHODS):
   results[variant][method]={}
   for ti,target in enumerate(TARGETS):
    scene_rate=counts[:,vi,mi,ti,:].mean(axis=0)/100
    quant=np.quantile(scene_rate,[0,.25,.5,.75,1],method='linear')
    results[variant][method][f'target{target}']={
        'quantiles':dict(zip(QKEYS,map(float,quant))),
        'seed_accepted':[int(v) for v in counts[:,vi,mi,ti,:].sum(axis=1)],
        'overall_mean_acceptance':float(counts[:,vi,mi,ti,:].sum()/(3*36500)),
        'scene_count':365,'examples_per_scene_per_seed':100}
 summary={'state':'scene_diagnostic_built_pending_independent_audit',
          'declaration_sha256':sha(RUN/'scene-diagnostic-declaration.json'),
          'protocol_sha256':sha(ROOT/'docs/PLACES365_SCENE_DIAGNOSTIC.md'),
          'producer_source_sha256':sha(__file__),
          'scene_values_sha256':sha(OUT/'scene_values.npz'),
          'scene_csv_sha256':sha(OUT/'scene_counts.csv'),
          'results':results,'created_at':datetime.now(timezone.utc).isoformat(),
          'boundary':'Post-result balanced-scene descriptive distribution; no method selection.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 lines=['# Places365 scene acceptance heterogeneity','',
        'This post-result diagnostic partitions all 36,500 official Places365 validation images into 365 scenes of 100 images. The three existing CIFAR-100 models, six methods, both temperature variants and all four ID-validation thresholds are unchanged. Scene labels are used only to describe the saved decisions. No detector, model or threshold was selected with these scene results.','',
        'Each value below is a quantile of the 365 **scene-level three-seed mean OOD acceptance rates** at the fixed 95% ID-validation target, in percent. A lower acceptance rate means more images rejected as OOD. Full per-scene/per-seed counts for all four targets are in [scene_counts.csv](scene_counts.csv); the complete grid and exact seed aggregates are in [summary.json](summary.json).','']
 for variant in VARIANTS:
  lines += [f'## {variant}','','| Method | Min | Q25 | Median | Q75 | Max | Overall OOD accepted |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
  for method,name in zip(METHODS,DISPLAY):
   r=results[variant][method]['target95']
   vals=[r['quantiles'][q] for q in QKEYS]+[r['overall_mean_acceptance']]
   lines.append('| '+name+' | '+' | '.join(f'{100*v:.2f}' for v in vals)+' |')
  lines.append('')
 lines += ['## Interpretation','',
           'All 365 scenes and all 100 images per scene are retained. Because scenes have equal size, the average of the 365 scene acceptance rates equals the saved overall OOD acceptance rate for each seed; this identity is independently checked. Scene ranges describe this image collection after a 256→32 resize. They are neither confidence intervals nor evidence that scene semantics caused the differences. Comparisons remain post-result and do not update the completed [aggregate Places365 report](../places365-extension/REPORT.md).','',
           'The [fixed95 distribution figure](fixed95-scene-distribution.png) shows every method and both temperature variants; full numerical values and independent audit remain local.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 fig,axes=plt.subplots(1,2,figsize=(12,5.7),sharey=True,constrained_layout=True)
 for vi,(ax,variant,color) in enumerate(zip(axes,VARIANTS,('#247ba0','#dc5f3b'))):
  sample=[counts[:,vi,mi,2,:].mean(axis=0) for mi in range(6)]
  bp=ax.boxplot(sample,vert=False,patch_artist=True,showfliers=False,
       tick_labels=DISPLAY,whis=(0,100))
  for box in bp['boxes']:box.set(facecolor=color,edgecolor=color,alpha=.65)
  for item in bp['medians']:item.set(color='black',linewidth=1.2)
  ax.set_xlim(0,100);ax.grid(axis='x',alpha=.25)
  ax.set_title(variant);ax.set_xlabel('OOD accepted at fixed95 (%)')
 axes[0].invert_yaxis()
 fig.suptitle('Places365 scene-level mean acceptance across 3 seeds (365 scenes)')
 fig.savefig(OUT/'fixed95-scene-distribution.png',dpi=180)
 fig.savefig(OUT/'fixed95-scene-distribution.svg')
 plt.close(fig)
 print(json.dumps({'state':summary['state'],'scene_cells':int(counts.size),
                   'csv_rows':2*6*4*365}))

if __name__=='__main__':main()