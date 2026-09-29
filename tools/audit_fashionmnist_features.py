"""Independent 45k-ID-fit feature-score probes for Fashion-MNIST extraction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fashionmnist-extension'
ISOLATED=ROOT/'runs/validation-isolation'
sys.path.insert(0,str(ROOT/'tools'))
from audit_isolated_features import fit_training_id,probe_rows,score_probes


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def require(ok,why):
    if not ok: raise AssertionError(why)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    seed=parser.parse_args().seed
    folder=RUN/f'extract-seed{seed}'
    manifest,status=read(folder/'manifest.json'),read(folder/'status.json')
    cache=read(RUN/'cache-manifest.json')
    cache_audit=read(RUN/'cache-audit.json')
    require(status['state']=='extracted_pending_independent_audit_and_scoring' and
            status['manifest_sha256']==sha(folder/'manifest.json') and
            manifest['seed']==seed and manifest['cache_sha256']==cache['cache_sha256'] and
            manifest['cache_audit_sha256']==sha(RUN/'cache-audit.json') and
            cache_audit['state']=='passed','complete audited frozen extraction required')
    for name,expected in manifest['source_hashes'].items():
        require(sha(folder/'source_snapshot'/name)==expected==sha(ROOT/name),
                'extraction source changed: '+name)
    for name,expected in manifest['outputs'].items():
        require(sha(folder/name)==expected,'extraction output changed: '+name)
    raw,feature=arrays(folder/'raw.npz'),arrays(folder/'feature_scores.npz')
    cached=arrays(ROOT/'data/fashionmnist-test-32x32.npz')
    require(np.array_equal(raw['source_rows'],np.arange(10000)) and
            np.array_equal(raw['labels'],cached['labels']) and
            raw['features'].shape==(10000,512) and
            raw['logits'].shape==(10000,100) and
            np.array_equal(raw['predictions'],raw['logits'].argmax(axis=1)),
            'Fashion raw/label/score alignment')
    old=ISOLATED/f'extract-seed{seed}'
    old_manifest=read(old/'manifest.json')
    require(manifest['training_id_bank_sha256']==sha(old/'id_train_raw.npz')==
            old_manifest['outputs']['id_train_raw.npz'],
            '45k training-ID feature bank changed')
    training=arrays(old/'id_train_raw.npz')
    original=arrays(ROOT/f'runs/cifar100-seed{seed}/split_indices.npz')
    require(np.array_equal(training['original_indices'],original['train']),
            'training feature rows not 45k ID split')
    bank,means,precision,scale=fit_training_id(training['features'],training['labels'])
    rows=probe_rows(10000)
    knn,maha=score_probes(raw['features'][rows],bank,means,precision)
    require(feature['knn'].shape==(10000,) and
            feature['mahalanobis'].shape==(10000,) and
            np.allclose(knn,feature['knn'][rows],atol=1e-4,rtol=1e-5) and
            np.allclose(maha,feature['mahalanobis'][rows],atol=1e-3,rtol=1e-5),
            'independent KNN/Mahalanobis fixed probes')
    destination=RUN/f'feature-audit-seed{seed}'
    require(not destination.exists(),'feature audit already exists')
    destination.mkdir()
    shutil.copyfile(__file__,destination/'executed_audit_source.py')
    record={'state':'passed','seed':seed,'fit_population':'45k ID training features only',
            'probe_rows':rows.tolist(),'probe_count':len(rows),
            'knn_max_abs_error':float(np.max(np.abs(knn-feature['knn'][rows]))),
            'mahalanobis_max_abs_error':float(np.max(np.abs(maha-feature['mahalanobis'][rows]))),
            'pooled_covariance_mean_diagonal':scale,
            'extraction_manifest_sha256':sha(folder/'manifest.json'),
            'cache_manifest_sha256':sha(RUN/'cache-manifest.json'),
            'fit_math_source_sha256':sha(ROOT/'tools/audit_isolated_features.py'),
            'source_sha256':sha(__file__),
            'audited_at':datetime.now(timezone.utc).isoformat(),
            'boundary':'Independent ID-training refit and Fashion query probes; no Fashion-based fit.'}
    (destination/'audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'passed','seed':seed,'probes':len(rows)}),flush=True)

if __name__=='__main__': main()
