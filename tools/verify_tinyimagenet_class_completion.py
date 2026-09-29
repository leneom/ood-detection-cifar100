"""Create or read-only verify the hash-bound class-diagnostic completion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report/tinyimagenet-class-heterogeneity'
RUN=ROOT/'runs/tinyimagenet-confirmation'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def require(ok,why):
    if not ok: raise AssertionError(why)

def paths():
    files=[
      'docs/TINYIMAGENET_CLASS_HETEROGENEITY.md',
      'runs/tinyimagenet-confirmation/class-heterogeneity-declaration.json',
      'report/tinyimagenet-confirmation/completion.json',
      'tools/build_tinyimagenet_class_heterogeneity.py',
      'tools/audit_tinyimagenet_class_heterogeneity.py',
      'tools/render_tinyimagenet_class_figures.py',
      'tools/verify_tinyimagenet_class_completion.py',
      'report/tinyimagenet-class-heterogeneity/class_values.npz',
      'report/tinyimagenet-class-heterogeneity/summary.json',
      'report/tinyimagenet-class-heterogeneity/REPORT.md',
      'report/tinyimagenet-class-heterogeneity/class-auroc.png',
      'report/tinyimagenet-class-heterogeneity/class-auroc.svg',
      'report/tinyimagenet-class-heterogeneity/class-fixed95-acceptance.png',
      'report/tinyimagenet-class-heterogeneity/class-fixed95-acceptance.svg',
      'report/tinyimagenet-class-heterogeneity/audit.json',
      'report/tinyimagenet-class-heterogeneity/figure-inspection.json',
      'report/tinyimagenet-class-heterogeneity/regression.log',
      'report/tinyimagenet-class-heterogeneity/render-attempt-1/audit.json',
      'report/tinyimagenet-class-heterogeneity/render-attempt-1/audit-before-report-wording.json',
      'report/tinyimagenet-class-heterogeneity/render-attempt-1/reconstruction.json',
      'report/tinyimagenet-class-heterogeneity/render-attempt-1/class-auroc.png',
      'report/tinyimagenet-class-heterogeneity/render-attempt-1/class-auroc.svg',
      'report/tinyimagenet-class-heterogeneity/render-attempt-1/class-fixed95-acceptance.png',
      'report/tinyimagenet-class-heterogeneity/render-attempt-1/class-fixed95-acceptance.svg',
    ]
    return files


def validate():
    d=read(RUN/'class-heterogeneity-declaration.json')
    s=read(OUT/'summary.json')
    a=read(OUT/'audit.json')
    v=read(OUT/'figure-inspection.json')
    require(d['state']=='declared_before_per_synset_scoring' and
            d['protocol_sha256']==sha(ROOT/'docs/TINYIMAGENET_CLASS_HETEROGENEITY.md') and
            d['aggregate_completion_sha256']==sha(ROOT/'report/tinyimagenet-confirmation/completion.json'),
            'frozen diagnostic declaration')
    require(s['class_count']==200 and s['images_per_class']==50 and
            s['declaration_sha256']==sha(RUN/'class-heterogeneity-declaration.json') and
            s['class_values_sha256']==sha(OUT/'class_values.npz') and
            a['state']==v['state']=='passed' and
            a['summary_sha256']==sha(OUT/'summary.json') and
            a['report_sha256']==sha(OUT/'REPORT.md') and
            a['class_values_sha256']==sha(OUT/'class_values.npz') and
            a['audit_source_sha256']==sha(ROOT/'tools/audit_tinyimagenet_class_heterogeneity.py'),
            'independent diagnostic summary/audit')
    require(a['checked']['class_auroc']==7200 and
            a['checked']['operating_counts']==28800 and
            a['checked']['aggregate_identities']==180 and
            a['checked']['quantiles']==120, 'complete independent math')
    for name,digest in a['figure_sha256'].items():
        require(digest==sha(OUT/name),'audited final figure')
    for name,item in v['images'].items():
        require(item['sha256']==sha(OUT/name),'visually inspected final PNG')
    for name,digest in s['source_hashes'].items():
        require(digest==sha(ROOT/name),'underlying frozen score source')
    with np.load(OUT/'class_values.npz',allow_pickle=False) as z:
        require(z['auroc'].shape==(3,2,6,200) and
                z['accepted'].shape==(3,2,6,4,200) and
                z['row_indices'].shape==(200,50), 'complete saved classes')
    regression=(OUT/'regression.log').read_text(encoding='utf-8')
    require('Ran 91 tests' in regression and 'OK (skipped=2)' in regression and
            'FAILED' not in regression,'full regression')
    return {name:sha(ROOT/name) for name in paths()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    target=OUT/'completion.json'
    if args.write:
        require(not target.exists(),'completion already exists')
        hashes=validate()
        record={'state':'complete','completed_at':datetime.now(timezone.utc).isoformat(),
                'classes':200,'images_per_class':50,'seeds':[0,1,2],
                'independent_class_auroc':7200,'independent_operating_counts':28800,
                'regression':{'total':91,'passed':89,'skipped':2,'exit_code':0},
                'inputs_sha256':hashes,
                'boundary':'Post-test class heterogeneity diagnostic only; no new held-out confirmation or method selection.'}
        target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    record=read(target)
    require(record['state']=='complete' and record['classes']==200,'completion header')
    actual=validate()
    require(actual==record['inputs_sha256'],'hash-bound completion inputs changed')
    print(json.dumps({'state':'verified_complete','files':len(actual),'classes':200}))

if __name__=='__main__': main()
