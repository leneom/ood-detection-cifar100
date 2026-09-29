"""Independent searchsorted audit of every Tiny ImageNet class diagnostic."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/tinyimagenet-confirmation'
OUT = ROOT / 'report/tinyimagenet-class-heterogeneity'
VARIANTS = ('T1','calibrated')
METHODS = ('msp','energy','maxlogit','entropy','knn','mahalanobis')
TARGETS = (80,90,95,99)
NAMES = ('MSP','Energy','MaxLogit','Entropy','KNN-50','Pooled Mahalanobis')
QKEYS = ('min','q25','median','q75','max')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def arrays(p):
    with np.load(p,allow_pickle=False) as x:
        return {k:x[k] for k in x.files}

def require(ok,why):
    if not ok: raise AssertionError(why)

def quantile_linear(values, q):
    ordered=sorted(map(float,values))
    pos=(len(ordered)-1)*q
    lower=int(pos)
    upper=min(lower+1,len(ordered)-1)
    frac=pos-lower
    return ordered[lower]*(1-frac)+ordered[upper]*frac


def main():
    require(not (OUT/'audit.json').exists(),'prior audit exists')
    declaration=read(RUN/'class-heterogeneity-declaration.json')
    summary=read(OUT/'summary.json')
    cache=arrays(ROOT/'data/tinyimagenet-validation.npz')
    saved=arrays(OUT/'class_values.npz')
    require(declaration['protocol_sha256']==sha(ROOT/'docs/TINYIMAGENET_CLASS_HETEROGENEITY.md') and
            declaration['aggregate_completion_sha256']==sha(ROOT/'report/tinyimagenet-confirmation/completion.json') and
            summary['declaration_sha256']==sha(RUN/'class-heterogeneity-declaration.json') and
            summary['producer_source_sha256']==sha(ROOT/'tools/build_tinyimagenet_class_heterogeneity.py') and
            summary['class_values_sha256']==sha(OUT/'class_values.npz'), 'frozen output binding')
    require(saved['auroc'].shape==(3,2,6,200) and saved['accepted'].shape==(3,2,6,4,200) and
            saved['row_indices'].shape==(200,50) and
            tuple(saved['targets'])==TARGETS and tuple(saved['variants'])==VARIANTS and
            tuple(saved['methods'])==METHODS,'fixed arrays')
    synsets=sorted(set(map(str,cache['synsets'])))
    require(len(synsets)==200 and np.array_equal(saved['synsets'],np.array(synsets)),
            'synset census/order')
    for ci,synset in enumerate(synsets):
        expected=np.where(cache['synsets']==synset)[0]
        require(len(expected)==50 and np.array_equal(saved['row_indices'][ci],expected),
                'every exact row index')
    checks={'class_auroc':0,'operating_counts':0,'aggregate_identities':0,
            'quantiles':0,'report_rows':0,'figures':0}
    for seed in range(3):
        rawpath=RUN/f'extract-seed{seed}/raw.npz'
        scorepath=RUN/f'score-seed{seed}/scores.npz'
        maskpath=RUN/f'score-seed{seed}/acceptance_masks.npz'
        idpath=ROOT/f'runs/validation-isolation/score-seed{seed}/id_test_scores.npz'
        raw,score,mask,inside=(arrays(p) for p in (rawpath,scorepath,maskpath,idpath))
        require(np.array_equal(raw['synsets'],cache['synsets']) and
                np.array_equal(raw['filenames'],cache['filenames']), 'extracted row alignment')
        for p in (rawpath,scorepath,maskpath,idpath):
            name=p.relative_to(ROOT).as_posix()
            require(summary['source_hashes'][name]==sha(p),'source hash '+name)
        manifest=read(RUN/f'score-seed{seed}/manifest.json')
        for vi,variant in enumerate(VARIANTS):
            for mi,method in enumerate(METHODS):
                key=f'{variant}_{method}'
                x=np.sort(inside[key])
                y=score[key]
                values=y[saved['row_indices']]
                lo=np.searchsorted(x,values,side='left')
                hi=np.searchsorted(x,values,side='right')
                expected=(lo+(hi-lo)/2).mean(axis=1)/len(x)
                require(np.allclose(saved['auroc'][seed,vi,mi],expected,rtol=0,atol=2e-14),
                        key+' independent pairwise AUROC')
                checks['class_auroc']+=200
                original=manifest['results'][variant][method]
                require(abs(float(expected.mean())-original['metrics']['auroc'])<2e-12,
                        key+' aggregate AUROC identity')
                checks['aggregate_identities']+=1
                for ti,target in enumerate(TARGETS):
                    threshold=original['operating_grid'][f'target{target}']['threshold']
                    expected_counts=np.count_nonzero(values<=threshold,axis=1)
                    require(np.array_equal(saved['accepted'][seed,vi,mi,ti],expected_counts),
                            key+str(target)+' independent class counts')
                    maskkey=f'{key}_target{target}_OOD'
                    require(np.array_equal(mask[maskkey],np.asarray(y<=threshold,dtype=np.uint8)) and
                            int(expected_counts.sum())==original['operating_grid'][f'target{target}']['ood_accepted'],
                            key+str(target)+' original mask/global count')
                    checks['operating_counts']+=200
                    checks['aggregate_identities']+=1
    report=(OUT/'REPORT.md').read_text(encoding='utf-8')
    for vi,variant in enumerate(VARIANTS):
        section=report.split(f'## {variant}\n',1)[1].split('## ',1)[0]
        for mi,method in enumerate(METHODS):
            d=summary['distributions'][variant][method]
            raw_values=(saved['auroc'][:,vi,mi].mean(axis=0),
                        saved['accepted'][:,vi,mi,2].mean(axis=0)/50)
            for title,values in zip(('class_mean_auroc','class_mean_fixed95_ood_acceptance'),raw_values):
                for key,q in zip(QKEYS,(0,.25,.5,.75,1)):
                    expected=quantile_linear(values,q)
                    require(abs(d[title][key]-expected)<2e-15,
                            variant+method+title+key+' independent percentile')
                    checks['quantiles']+=1
            fmt=lambda data:' / '.join(f'{100*data[k]:.2f}' for k in QKEYS)
            row=f"| {NAMES[mi]} | {fmt(d['class_mean_auroc'])} | {fmt(d['class_mean_fixed95_ood_acceptance'])} |"
            require(section.count(row)==1,variant+method+' displayed row')
            checks['report_rows']+=1
    for stem in ('class-auroc','class-fixed95-acceptance'):
        with Image.open(OUT/f'{stem}.png') as im:
            im.verify()
        with Image.open(OUT/f'{stem}.png') as im:
            require(im.width>=1500 and im.height>=800,'figure canvas')
        require('<svg' in (OUT/f'{stem}.svg').read_text(encoding='utf-8')[:500],
                'SVG vector figure')
        checks['figures']+=2
    record={'state':'passed','checked':checks,
            'class_values_sha256':sha(OUT/'class_values.npz'),
            'summary_sha256':sha(OUT/'summary.json'),
            'report_sha256':sha(OUT/'REPORT.md'),
            'figure_sha256':{p.name:sha(p) for p in sorted(OUT.glob('*.png'))+sorted(OUT.glob('*.svg'))},
            'audit_source_sha256':sha(__file__),
            'audited_at':datetime.now(timezone.utc).isoformat(),
            'boundary':'All 200 classes audited; descriptive post-test only; no tuning or new held-out confirmation.'}
    (OUT/'audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'passed','checked':checks}),flush=True)


if __name__=='__main__': main()
