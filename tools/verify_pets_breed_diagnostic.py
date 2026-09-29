"""Write-once, then read-only hash-bound Pet breed diagnostic completion."""
from __future__ import annotations
import argparse,hashlib,json,re,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'runs/oxford-pets-breed-diagnostic'
OUT=ROOT/'report/oxford-pets-breed-diagnostic'

def require(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def validate():
 from tools.verify_pets_completion import validate as pet_validate
 from tools.verify_pets_paired_decisions import validate as paired_validate
 for name,validator in (('oxford-pets-extension',pet_validate),
                        ('oxford-pets-paired-decisions',paired_validate)):
  record=read(ROOT/f'report/{name}/completion.json')
  require(record['state']=='complete' and record['inputs_sha256']==validator(),
          'prior completed evidence '+name)
 declaration=read(BASE/'declaration.json')
 require(declaration['state']=='frozen_before_group_acceptance_calculation' and
         len(declaration['input_sha256'])==14,'frozen source declaration')
 for name,expected in declaration['input_sha256'].items():
  require(sha(ROOT/name)==expected,'source changed '+name)
 audit=read(OUT/'audit.json')
 summary=read(OUT/'summary.json')
 require(audit['state']=='independently_audited' and
         audit['declaration_sha256']==sha(BASE/'declaration.json') and
         audit['group_counts_checked']==5616 and
         audit['source_masks_checked']==144 and
         audit['csv_rows_checked']==1872 and
         audit['summary_cells_checked']==48 and
         audit['display_rows_checked']==12 and
         summary['declaration_sha256']==sha(BASE/'declaration.json') and
         summary['group_count_cells']==5616 and summary['csv_rows']==1872,
         'complete independent 37-breed/2-species audit')
 log=(OUT/'regression.log').read_text(encoding='utf-8',errors='replace')
 require(re.search(r'Ran 91 tests in [0-9.]+s',log) and log.rstrip().endswith('OK'),
         '91-test regression')
 names=list(declaration['input_sha256'])
 names += ['runs/oxford-pets-breed-diagnostic/declaration.json',
           'tools/build_pets_breed_diagnostic.py',
           'tools/audit_pets_breed_diagnostic.py',
           'tools/verify_pets_breed_diagnostic.py']
 names += [f'report/oxford-pets-breed-diagnostic/{name}' for name in
           ('group_values.npz','group_counts.csv','summary.json','REPORT.md',
            'audit.json','regression.log')]
 require(len(names)==len(set(names))==24,'complete 24-file census')
 return {name:sha(ROOT/name) for name in names}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true')
 args=parser.parse_args();target=OUT/'completion.json'
 if args.write:
  require(not target.exists(),'completion already exists')
  hashes=validate()
  record={'state':'complete','completed_at':datetime.now(timezone.utc).isoformat(),
          'breed_groups':37,'species_groups':2,'group_count_cells':5616,
          'csv_rows':1872,'regression':{'tests':91,'exit_code':0},
          'boundary':'Post-result complete subgroup description only; no OOD-based method selection.',
          'inputs_sha256':hashes}
  target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 record=read(target)
 require(record['state']=='complete' and record['breed_groups']==37 and
         record['species_groups']==2 and record['group_count_cells']==5616 and
         record['inputs_sha256']==validate(),'hash-bound read-only completion')
 print(json.dumps({'state':'verified_complete','files':len(record['inputs_sha256']),
                   'group_count_cells':record['group_count_cells']}))
if __name__=='__main__':main()