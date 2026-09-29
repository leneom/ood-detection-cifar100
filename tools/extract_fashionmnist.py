"""Frozen Fashion-MNIST inference from one selected isolated ID classifier."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fashionmnist-extension'
ISOLATED=ROOT/'runs/validation-isolation'
CACHE=ROOT/'data/fashionmnist-test-32x32.npz'
sys.path.insert(0,str(ROOT))
from src.ood.features import extract,KNN,Mahalanobis
from src.ood.models import build_model
from src.ood.runtime import file_hash,seed_all,environment
from src.ood.scores import METHODS as LOGIT_METHODS,score_logits

SOURCES=('tools/extract_fashionmnist.py','src/ood/features.py','src/ood/models.py',
         'src/ood/runtime.py','src/ood/data.py','src/ood/scores.py',
         'src/ood/datasets.py','tools/prepare_fashionmnist.py',
         'docs/FASHIONMNIST_EXTENSION.md')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p,record):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(p)

def save(p,**arrays):
    temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('wb') as f: np.savez_compressed(f,**arrays)
    temp.replace(p)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--batch-size',type=int,default=256)
    args=parser.parse_args()
    if not torch.cuda.is_available() or args.batch_size<=0:
        raise RuntimeError('verified CUDA and positive batch size required')
    declaration=read(RUN/'declaration.json')
    cache_manifest=read(RUN/'cache-manifest.json')
    cache_status=read(RUN/'cache-status.json')
    cache_audit=read(RUN/'cache-audit.json')
    if not (declaration['state']=='frozen_before_any_fashion_data_access' and
            declaration['protocol_sha256']==file_hash(ROOT/'docs/FASHIONMNIST_EXTENSION.md') and
            cache_status['manifest_sha256']==file_hash(RUN/'cache-manifest.json') and
            cache_audit['state']=='passed' and
            cache_audit['manifest_sha256']==file_hash(RUN/'cache-manifest.json') and
            cache_manifest['cache_sha256']==cache_audit['cache_sha256']==file_hash(CACHE) and
            cache_manifest['source_sha256']==file_hash(ROOT/'tools/prepare_fashionmnist.py') and
            cache_manifest['mnist_transform_source_sha256']==file_hash(ROOT/'src/ood/datasets.py') and
            declaration['previous_aggregate_completion_sha256']==
            file_hash(ROOT/'report/tinyimagenet-confirmation/completion.json') and
            declaration['previous_diagnostic_completion_sha256']==
            file_hash(ROOT/'report/tinyimagenet-class-heterogeneity/completion.json')):
        raise ValueError('frozen source/pixel/previous-completion verification failed')
    previous=read(ROOT/'report/validation-isolation/completion.json')
    if previous['state']!='complete' or previous['three_full_training_seeds']!=[0,1,2]:
        raise ValueError('three isolated classifiers incomplete')
    old=ISOLATED/f'extract-seed{args.seed}'
    old_manifest=read(old/'manifest.json')
    old_status=read(old/'status.json')
    selected=ISOLATED/f'seed{args.seed}/final-best.pt'
    if not (old_status['manifest_sha256']==file_hash(old/'manifest.json') and
            old_manifest['checkpoint_sha256']==file_hash(selected) and
            old_manifest['checkpoint_epoch']==(92,96,98)[args.seed] and
            old_manifest['outputs']['id_train_raw.npz']==file_hash(old/'id_train_raw.npz')):
        raise ValueError('frozen checkpoint or training-ID bank differs')
    score_manifest=read(ISOLATED/f'score-seed{args.seed}/manifest.json')
    score_audit=read(ISOLATED/f'audit-seed{args.seed}/audit.json')
    if not (score_audit['state']=='passed' and
            score_audit['scoring_manifest_sha256']==
            file_hash(ISOLATED/f'score-seed{args.seed}/manifest.json') and
            score_manifest['checkpoint_sha256']==old_manifest['checkpoint_sha256']):
        raise ValueError('frozen ID-only scoring/calibration differs')
    destination=RUN/f'extract-seed{args.seed}'
    if destination.exists(): raise FileExistsError('Fashion extraction already exists')
    source_hashes={name:file_hash(ROOT/name) for name in SOURCES}
    with np.load(CACHE,allow_pickle=False) as z:
        images=z['images']
        labels=z['labels']
        source_rows=z['source_rows']
    if not (images.shape==(10000,32,32,3) and images.dtype==np.uint8 and
            labels.shape==(10000,) and np.array_equal(source_rows,np.arange(10000))):
        raise ValueError('frozen NHWC Fashion cache differs')
    with np.load(old/'id_train_raw.npz',allow_pickle=False) as z:
        training_features=torch.from_numpy(z['features'].copy())
        training_labels=torch.from_numpy(z['labels'].copy())
    if training_features.shape!=(45000,512) or training_labels.shape!=(45000,):
        raise ValueError('45k training-ID feature bank differs')
    config=read(ISOLATED/f'seed{args.seed}/config.json')
    seed_all(0)
    torch.set_grad_enabled(False)
    weights=torch.load(selected,map_location='cpu',weights_only=True)
    if weights['epoch']!=old_manifest['checkpoint_epoch'] or weights['config']!=config:
        raise ValueError('selected checkpoint payload changed')
    model=build_model(config['model'],config['classes']).to('cuda')
    model=model.to(memory_format=torch.channels_last)
    model.load_state_dict(weights['model'])
    model.eval()
    destination.mkdir()
    snapshots=destination/'source_snapshot'
    for name in SOURCES:
        target=snapshots/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    write(destination/'status.json',{'state':'extracting','seed':args.seed,
                                     'started_at':datetime.now(timezone.utc).isoformat()})
    logits,features=extract(model,torch.from_numpy(images),args.batch_size,'cuda')
    knn=KNN(k=50,device='cuda').fit(training_features)
    maha=Mahalanobis(shrinkage=.01,device='cuda').fit(training_features,training_labels)
    t1={method:score_logits(logits,method,1.).numpy() for method in LOGIT_METHODS}
    feature_scores={'knn':knn.score(features).numpy(),
                    'mahalanobis':maha.score(features).numpy()}
    save(destination/'raw.npz',logits=logits.numpy(),features=features.numpy(),
         predictions=logits.argmax(dim=1).numpy(),source_rows=source_rows,
         labels=labels)
    save(destination/'T1_scores.npz',**t1)
    save(destination/'feature_scores.npz',**feature_scores)
    outputs={name:file_hash(destination/name) for name in
             ('raw.npz','T1_scores.npz','feature_scores.npz')}
    if file_hash(CACHE)!=cache_manifest['cache_sha256'] or any(
            file_hash(ROOT/name)!=value for name,value in source_hashes.items()):
        raise RuntimeError('frozen source/cache changed during extraction')
    manifest={'state':'extracted_pending_independent_audit_and_scoring',
              'seed':args.seed,'count':10000,
              'checkpoint_sha256':old_manifest['checkpoint_sha256'],
              'checkpoint_epoch':old_manifest['checkpoint_epoch'],
              'training_id_bank_sha256':old_manifest['outputs']['id_train_raw.npz'],
              'cache_sha256':cache_manifest['cache_sha256'],
              'cache_audit_sha256':file_hash(RUN/'cache-audit.json'),
              'isolation_completion_sha256':file_hash(ROOT/'report/validation-isolation/completion.json'),
              'prior_tiny_completion_sha256':file_hash(ROOT/'report/tinyimagenet-confirmation/completion.json'),
              'id_score_manifest_sha256':file_hash(ISOLATED/f'score-seed{args.seed}/manifest.json'),
              'source_hashes':source_hashes,'outputs':outputs,
              'environment':environment(),
              'created_at':datetime.now(timezone.utc).isoformat(),
              'boundary':'Frozen classifier/ID-only feature fit; no Fashion-MNIST-dependent fitting.'}
    write(destination/'manifest.json',manifest)
    write(destination/'status.json',{'state':manifest['state'],'seed':args.seed,
                                     'manifest_sha256':file_hash(destination/'manifest.json')})
    print(json.dumps({'state':manifest['state'],'seed':args.seed,'count':10000}),flush=True)

if __name__=='__main__': main()
