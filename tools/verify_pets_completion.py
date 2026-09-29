"""Write-once or read-only verify the Oxford-IIIT Pet test extension."""
from __future__ import annotations
import argparse,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/oxford-pets-extension'
OUT=ROOT/'report/oxford-pets-extension'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(ok,why):
 if not ok:raise AssertionError(why)
def inputs(d):
 names=['docs/OXFORD_PETS_EXTENSION.md',
    'runs/oxford-pets-extension/declaration.json',
    'runs/oxford-pets-extension/download-status.json',
    'runs/oxford-pets-extension/download.stdout.log',
    'runs/oxford-pets-extension/download.stderr.log',
    'runs/oxford-pets-extension/cache-status.json',
    'runs/oxford-pets-extension/cache-manifest.json',
    'runs/oxford-pets-extension/cache-audit.json',
    'runs/oxford-pets-extension/executed_cache_source.py',
    'runs/oxford-pets-extension/extraction-launch.json',
    'runs/oxford-pets-extension/extraction-status.json',
    'runs/oxford-pets-extension/gpu-guard-launch.json',
    'runs/oxford-pets-extension/controller.stdout.log',
    'runs/oxford-pets-extension/controller.stderr.log',
    'data/pets-images.tar.gz','data/pets-annotations.tar.gz','data/pets-test.npz',
    'report/oxford-pets-extension/summary.json',
    'report/oxford-pets-extension/REPORT.md',
    'report/oxford-pets-extension/auroc.png',
    'report/oxford-pets-extension/auroc.svg',
    'report/oxford-pets-extension/fixed95-acceptance.png',
    'report/oxford-pets-extension/fixed95-acceptance.svg',
    'report/oxford-pets-extension/display-audit.json',
    'report/oxford-pets-extension/figure-inspection.json',
    'report/oxford-pets-extension/regression.log',
    'tools/freeze_pets.py','tools/download_pets.py',
    'tools/prepare_pets.py','tools/audit_pets_cache.py',
    'tools/extract_pets.py','tools/run_pets_extraction.py',
    'tools/audit_pets_features.py','tools/score_pets.py',
    'tools/audit_pets_scores.py','tools/build_pets_report.py',
    'tools/audit_pets_report.py','tools/verify_pets_completion.py']
 names.extend(d['input_sha256'])
 for seed in range(3):
  names += [f'runs/oxford-pets-extension/extract-seed{seed}/status.json',
    f'runs/oxford-pets-extension/extract-seed{seed}/manifest.json',
    f'runs/oxford-pets-extension/extract-seed{seed}/raw.npz',
    f'runs/oxford-pets-extension/extract-seed{seed}/T1_scores.npz',
    f'runs/oxford-pets-extension/extract-seed{seed}/feature_scores.npz',
    f'runs/oxford-pets-extension/extract-seed{seed}.stdout.log',
    f'runs/oxford-pets-extension/extract-seed{seed}.stderr.log',
    f'runs/oxford-pets-extension/feature-audit-seed{seed}/audit.json',
    f'runs/oxford-pets-extension/score-seed{seed}/status.json',
    f'runs/oxford-pets-extension/score-seed{seed}/manifest.json',
    f'runs/oxford-pets-extension/score-seed{seed}/scores.npz',
    f'runs/oxford-pets-extension/score-seed{seed}/acceptance_masks.npz',
    f'runs/oxford-pets-extension/score-audit-seed{seed}/audit.json']
 return list(dict.fromkeys(names))
def validate():
 d=read(RUN/'declaration.json')
 download=read(RUN/'download-status.json')
 cache=read(RUN/'cache-audit.json')
 cm=read(RUN/'cache-manifest.json')
 controller=read(RUN/'extraction-status.json')
 summary=read(OUT/'summary.json')
 display=read(OUT/'display-audit.json')
 visual=read(OUT/'figure-inspection.json')
 require(d['state']=='protocol_frozen_before_pet_archive_access' and
         d['protocol_sha256']==sha(ROOT/'docs/OXFORD_PETS_EXTENSION.md') and
         d['source_sha256']==sha(ROOT/'tools/freeze_pets.py'),
         'frozen protocol and freeze source')
 for name,value in d['input_sha256'].items():require(sha(ROOT/name)==value,'frozen input '+name)
 require(download['state']=='verified_archives_pending_pixel_cache' and
         download['declaration_sha256']==sha(RUN/'declaration.json') and
         download['source_sha256']==sha(ROOT/'tools/download_pets.py'),
         'official download state')
 for role in ('images','annotations'):
  item=download['files'][role]
  require(item['state']=='verified_archive' and
          item['sha256']==sha(ROOT/item['relative_path']) and
          item['md5']==d['expected_md5'][role], 'official archive '+role)
 require(cm['state']=='verified_source_and_pixels_pending_independent_audit' and
         cm['count']==3669 and cache['state']=='passed' and
         cache['images_checked']==cache['resized_pixel_arrays_checked']==3669 and
         cm['cache_sha256']==cache['cache_sha256']==sha(ROOT/'data/pets-test.npz') and
         cache['manifest_sha256']==sha(RUN/'cache-manifest.json') and
         cm['source_sha256']==sha(ROOT/'tools/prepare_pets.py') and
         cache['source_sha256']==sha(ROOT/'tools/audit_pets_cache.py'),
         'all-image independent pixel audit')
 require(controller['state']=='three_seeds_extracted_pending_audit_and_scoring' and
         controller['completed_seeds']==[0,1,2], 'single sequential GPU controller')
 launch=read(RUN/'extraction-launch.json')
 require(launch['extract_source_sha256']==sha(ROOT/'tools/extract_pets.py') and
         launch['controller_source_sha256']==sha(ROOT/'tools/run_pets_extraction.py'),
         'GPU source launch binding')
 require(summary['state']=='three_seed_pets_scored_and_audited' and
         summary['id_test_count']==10000 and summary['ood_count']==3669 and
         summary['ood_prevalence']==3669/13669 and
         display['state']==visual['state']=='passed' and
         display['summary_sha256']==sha(OUT/'summary.json') and
         display['report_sha256']==sha(OUT/'REPORT.md'),
         'three-seed complete report')
 for name,item in visual['images'].items():
  require(item['sha256']==sha(OUT/name) and item['size'][0]>=1500 and
          item['size'][1]>=800,'visually inspected '+name)
 log=(OUT/'regression.log').read_text(encoding='utf-8')
 require('Ran 91 tests' in log and re.search(r'(?m)^OK\s*$',log) and
         'FAILED' not in log,'full CPU regression')
 for seed in range(3):
  ep=RUN/f'extract-seed{seed}';sp=RUN/f'score-seed{seed}'
  e=read(ep/'manifest.json');s=read(sp/'manifest.json')
  fa=read(RUN/f'feature-audit-seed{seed}/audit.json')
  sa=read(RUN/f'score-audit-seed{seed}/audit.json')
  sr=summary['seed_records'][seed]
  require(e['seed']==s['seed']==fa['seed']==sa['seed']==seed and
          e['count']==s['n_ood']==3669 and s['n_id_test']==10000 and
          s['dataset']=='oxford_iiit_pet_official_test' and
          e['checkpoint_sha256']==s['checkpoint_sha256']==sr['checkpoint_sha256'] and
          fa['state']==sa['state']=='passed' and
          fa['extraction_manifest_sha256']==sha(ep/'manifest.json') and
          sa['scoring_manifest_sha256']==sha(sp/'manifest.json') and
          s['prior_id_score_manifest_sha256']==
          sha(ROOT/f'runs/validation-isolation/score-seed{seed}/manifest.json'),
          'seed extraction/score lineage '+str(seed))
  for folder,manifest in ((ep,e),(sp,s)):
   for name,value in manifest['outputs'].items():
    require(sha(folder/name)==value,'stage output '+name)
   for name,value in manifest['source_hashes'].items():
    require(sha(ROOT/name)==value and sha(folder/'source_snapshot'/name)==value,
            'source snapshot '+name)
 return {name:sha(ROOT/name) for name in inputs(d)}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
 target=OUT/'completion.json'
 if args.write:
  require(not target.exists(),'completion already exists')
  hashes=validate()
  record={'state':'complete','completed_at':datetime.now(timezone.utc).isoformat(),
          'population':'Oxford-IIIT Pet official test, all 3669 images',
          'id_test_count':10000,'ood_count':3669,'seeds':[0,1,2],
          'primary_cells':12,'independent_score_metric_fields':468,
          'independent_score_masks':432,'independent_summary_seed_values':828,
          'regression':{'total':91,'exit_code':0},
          'inputs_sha256':hashes,
          'boundary':'Sequentially chosen single pet OOD test population; fixed ID-only fits and thresholds; no OOD-based selection or combined benchmark claim.'}
  target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 record=read(target)
 require(record['state']=='complete' and record['seeds']==[0,1,2] and
         record['inputs_sha256']==validate(),'hash-bound read-only completion')
 print(json.dumps({'state':'verified_complete','files':len(record['inputs_sha256']),
                   'seeds':record['seeds']}))
if __name__=='__main__':main()