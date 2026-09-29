"""Frozen saved-draw scope, before any dependence computation."""
import numpy as np
from geometry_io import ROOT,read,load,digest,require
RUN=ROOT/'runs/bootstrap-dependence'
SOURCE=ROOT/'runs/paired-bootstrap'

def guard():
    pre=read(RUN/'predeclaration.json')
    require(pre['replicates']==1000 and pre['endpoints']==66 and pre['pair_count']==105 and pre['ddof']==1 and pre['seeds']==[0,1,2] and pre['new_sampling_or_fitting'] is False,'dependence declaration changed')
    for n,h in pre['original_artifacts'].items():require(digest(ROOT/n)==h,'declared input changed '+n)
    require(digest(ROOT/pre['protocol'])==pre['protocol_sha256'] and read(SOURCE/'status.json')['state']=='complete','bootstrap/protocol incomplete')
    return pre

def saved_vectors():
    blocks=[load(SOURCE/'analysis'/f'block-{i:03d}-metrics.npz') for i in range(10)]
    arrays={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    require(np.array_equal(arrays['replicates'],np.arange(1000)),'replicate order changed')
    manifest=read(SOURCE/'analysis-inputs.json');datasets=manifest['datasets'];fields=manifest['ranking_fields'];targets=manifest['targets'];endpoints=[]
    for di,d in enumerate(datasets):
        for fi,f in enumerate(fields):endpoints.append((f'ranking/{d}/{f}',arrays['ranking'][:,:,di,:,fi]))
    for fi,f in enumerate(fields[:4]):endpoints.append((f'ranking/natural_macro/{f}',arrays['natural_ranking_macro'][:,:,:,fi]))
    for ti,t in enumerate(targets):
        for key in ['id_coverage','retained_error_risk']:endpoints.append((f'operating/{t:g}/{key}',arrays[key][:,:,:,ti]))
        for di,d in enumerate(datasets):endpoints.append((f'operating/{t:g}/{d}_acceptance',arrays['ood_acceptance'][:,:,:,ti,di]))
        endpoints.append((f'operating/{t:g}/natural_ood_acceptance_macro',arrays['natural_ood_acceptance_macro'][:,:,:,ti]))
    original=read(ROOT/'report/paired-bootstrap/summary.json')
    require(len(endpoints)==66 and {k for k,v in endpoints}==set(original['endpoints']),'endpoint scope changed')
    natural=[]
    for fi,f in enumerate(fields[:4]):
        for seed in range(3):
            for mi,m in enumerate(manifest['methods']):natural.append((f'ranking/{f}/seed{seed}/{m}',arrays['ranking'][:,seed,:4,mi,fi]))
    for ti,t in enumerate(targets):
        for seed in range(3):
            for mi,m in enumerate(manifest['methods']):natural.append((f'operating/{t:g}/seed{seed}/{m}',arrays['ood_acceptance'][:,seed,mi,ti,:4]))
    require(len(natural)==360 and all(v.shape==(1000,4) for _,v in natural),'natural vector scope changed')
    return endpoints,natural,original
