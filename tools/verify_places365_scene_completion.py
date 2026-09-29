"""Write once or read-only verify the Places365 all-scene diagnostic."""
from __future__ import annotations
import argparse,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/places365-extension'
OUT=ROOT/'report/places365-scene-diagnostic'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def check(ok,why):
 if not ok:raise AssertionError(why)
def inputs(d):
 names=['docs/PLACES365_SCENE_DIAGNOSTIC.md',
   'runs/places365-extension/scene-diagnostic-declaration.json',
   'tools/build_places365_scene_diagnostic.py',
   'tools/audit_places365_scene_diagnostic.py',
   'tools/verify_places365_scene_completion.py',
   'report/places365-scene-diagnostic/scene_values.npz',
   'report/places365-scene-diagnostic/scene_counts.csv',
   'report/places365-scene-diagnostic/summary.json',
   'report/places365-scene-diagnostic/REPORT.md',
   'report/places365-scene-diagnostic/fixed95-scene-distribution.png',
   'report/places365-scene-diagnostic/fixed95-scene-distribution.svg',
   'report/places365-scene-diagnostic/audit.json',
   'report/places365-scene-diagnostic/figure-inspection.json',
   'report/places365-scene-diagnostic/regression.log']
 return list(dict.fromkeys(names+list(d['input_sha256'])))
def validate():
 d=read(RUN/'scene-diagnostic-declaration.json')
 s=read(OUT/'summary.json')
 a=read(OUT/'audit.json')
 v=read(OUT/'figure-inspection.json')
 check(d['state']=='declared_after_aggregate_before_scene_stratification' and
       d['protocol_sha256']==sha(ROOT/'docs/PLACES365_SCENE_DIAGNOSTIC.md') and
       d['seeds']==[0,1,2], 'frozen post-result declaration')
 for name,expected in d['input_sha256'].items():check(sha(ROOT/name)==expected,'frozen input '+name)
 check(s['state']=='scene_diagnostic_built_pending_independent_audit' and
       s['declaration_sha256']==sha(RUN/'scene-diagnostic-declaration.json') and
       s['producer_source_sha256']==sha(ROOT/'tools/build_places365_scene_diagnostic.py') and
       s['scene_values_sha256']==sha(OUT/'scene_values.npz') and
       s['scene_csv_sha256']==sha(OUT/'scene_counts.csv'), 'summary binding')
 check(a['state']==v['state']=='passed' and
       a['source_sha256']==sha(ROOT/'tools/audit_places365_scene_diagnostic.py') and
       a['summary_sha256']==sha(OUT/'summary.json') and
       a['cube_sha256']==sha(OUT/'scene_values.npz') and
       a['csv_sha256']==sha(OUT/'scene_counts.csv') and
       a['report_sha256']==sha(OUT/'REPORT.md') and
       a['checked']=={'scene_counts':52560,'csv_rows':17520,'summary_cells':48,
                      'display_rows':12,'figures':2}, 'independent exhaustive audit')
 for name,digest in a['figure_sha256'].items():check(digest==sha(OUT/name),'audited plot '+name)
 for name,item in v['images'].items():check(item['sha256']==sha(OUT/name),'visually inspected plot '+name)
 with np.load(OUT/'scene_values.npz',allow_pickle=False) as z:
  check(z['counts'].shape==(3,2,6,4,365) and z['row_indices'].shape==(365,100),
        'full class cube')
 regression=(OUT/'regression.log').read_text(encoding='utf-8')
 check('Ran 91 tests' in regression and re.search(r'(?m)^OK\s*$',regression) and
       'FAILED' not in regression,'complete CPU regression')
 return {name:sha(ROOT/name) for name in inputs(d)}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
 target=OUT/'completion.json'
 if args.write:
  check(not target.exists(),'completion already exists')
  hashes=validate()
  record={'state':'complete','completed_at':datetime.now(timezone.utc).isoformat(),
          'scenes':365,'images_per_scene':100,'seeds':[0,1,2],
          'independent_scene_counts':52560,'csv_rows':17520,
          'regression':{'total':91,'exit_code':0},'inputs_sha256':hashes,
          'boundary':'Post-result scene heterogeneity only; no OOD-based method selection or causal claim.'}
  target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 record=read(target)
 check(record['state']=='complete' and record['scenes']==365 and
       record['inputs_sha256']==validate(),'hash-bound completion')
 print(json.dumps({'state':'verified_complete','files':len(record['inputs_sha256']),
                   'scenes':365,'counts':52560}))
if __name__=='__main__':main()