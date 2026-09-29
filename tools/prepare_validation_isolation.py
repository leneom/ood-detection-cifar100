"""Two-stage source-bound declaration then deterministic three-way ID-validation draw."""
import argparse
from pathlib import Path
import numpy as np
from geometry_io import ROOT,read,write,load,save_arrays,digest,require,now

RUN=ROOT/'runs/validation-isolation'
METHODS=['msp','energy','maxlogit','entropy','knn','mahalanobis']
DATASETS=['cifar10','mnist','svhn','textures','noise']
TARGETS=[.8,.9,.95,.99]

def source_paths():
    names=['docs/VALIDATION_ISOLATION.md','report/project-assessment/completion-audit.json','report/project-assessment/index.json',
           'runs/project-assessment/status.json','report/baseline-complete/overview.json','report/paired-bootstrap/completion-audit.json',
           'report/bootstrap-dependence/completion-audit.json','runs/calibration-phase2/calibration_split.npz',
           'src/ood/train.py','src/ood/data.py','src/ood/models.py','src/ood/runtime.py']
    for seed in range(3):
        for file in ['split_indices.npz','split.json','config.json','source_snapshot/train.py','final-best.pt']:
            names.append(f'runs/cifar100-seed{seed}/{file}')
    return names

def validate_old():
    old=load(ROOT/'runs/calibration-phase2/calibration_split.npz');labels=old['validation_labels'];canonical=old['original_validation_indices']
    require(labels.shape==(5000,) and canonical.shape==(5000,) and np.array_equal(np.unique(labels,return_counts=True)[1],np.full(100,50)),
            'old validation does not have 50 examples/class')
    baseline=[]
    for seed in range(3):
        split=load(ROOT/f'runs/cifar100-seed{seed}/split_indices.npz');config=read(ROOT/f'runs/cifar100-seed{seed}/config.json')
        require(np.array_equal(split['val'],canonical) and len(split['train'])==45000 and len(split['val'])==5000 and not np.intersect1d(split['train'],split['val']).size,'old train/validation split mismatch')
        require(config['dataset']=='cifar100' and config['seed']==seed and config['split_seed']==2026 and config['val_per_class']==50 and config['epochs']==100 and config['model']=='resnet18' and config['batch_size']==128 and config['amp'] is True and config['no_deterministic'] is False,'old training budget mismatch')
        if seed:require(np.array_equal(split['train'],baseline[0]) and np.array_equal(split['val'],baseline[1]),'old cross-seed split mismatch')
        else:baseline=[split['train'],split['val']]
    return labels,canonical

def declaration():
    require(not (RUN/'predeclaration.json').exists(),'partition declaration already exists')
    for n in source_paths():require((ROOT/n).is_file(),'missing fixed evidence '+n)
    validate_old()
    inputs={n:digest(ROOT/n) for n in source_paths()}
    RUN.mkdir(parents=True,exist_ok=True)
    declaration={'state':'declared_before_three_way_partition_draw','declared_at':now(),'protocol':'docs/VALIDATION_ISOLATION.md','protocol_sha256':inputs['docs/VALIDATION_ISOLATION.md'],
                 'source_sha256':digest(__file__),'original_artifacts':inputs,'seeds':[0,1,2],'original_split_seed':2026,'partition_rng':2029,
                 'per_class':[20,15,15],'roles':['checkpoint','temperature','threshold'],'methods':METHODS,'datasets':DATASETS,'targets':TARGETS,
                 'training_epochs':100,'training_count':45000,'old_validation_count':5000,'new_gpu_training_started':False,
                 'boundary':'ID validation role isolation only; old OOD populations already explored, not a fresh confirmatory test'}
    write(RUN/'predeclaration.json',declaration)
    write(RUN/'status.json',{'state':'declared_not_partitioned','declared_at':now(),'bound_input_count':len(inputs),'trainer_running':False})
    print({'state':'declared_not_partitioned','input_count':len(inputs),'rng':2029},flush=True)

def partition():
    pre=read(RUN/'predeclaration.json');status=read(RUN/'status.json')
    require(pre['state']=='declared_before_three_way_partition_draw' and status['state']=='declared_not_partitioned' and pre['source_sha256']==digest(__file__),'partition predeclaration/source changed')
    require(not (RUN/'partition.npz').exists() and not (RUN/'partition.json').exists(),'partition already exists')
    for n,h in pre['original_artifacts'].items():require(digest(ROOT/n)==h,'frozen predecessor changed '+n)
    labels,indices=validate_old();rng=np.random.default_rng(2029);rows=[[],[],[]]
    for c in range(100):
        belonging=np.flatnonzero(labels==c);require(len(belonging)==50,'class validation count changed')
        shuffled=rng.permutation(belonging)
        rows[0].extend(shuffled[:20]);rows[1].extend(shuffled[20:35]);rows[2].extend(shuffled[35:])
    arrays=[np.sort(np.asarray(v,dtype=np.int64)) for v in rows]
    require([len(a) for a in arrays]==[2000,1500,1500] and np.array_equal(np.sort(np.concatenate(arrays)),np.arange(5000)),'partition does not cover exactly disjoint validation')
    for a,n in zip(arrays,[20,15,15]):require(np.array_equal(np.bincount(labels[a],minlength=100),np.full(100,n)),'per-class role count changed')
    save_arrays(RUN/'partition.npz',checkpoint_rows=arrays[0],temperature_rows=arrays[1],threshold_rows=arrays[2],
                checkpoint_original_indices=indices[arrays[0]],temperature_original_indices=indices[arrays[1]],
                threshold_original_indices=indices[arrays[2]],original_validation_indices=indices,validation_labels=labels)
    from hashlib import sha256
    metadata={'state':'partition_prepared_pending_independent_audit','prepared_at':now(),'predeclaration_sha256':digest(RUN/'predeclaration.json'),
              'source_sha256':digest(__file__),'partition_sha256':digest(RUN/'partition.npz'),'rng':2029,'numpy':np.__version__,
              'role_counts':{'checkpoint':2000,'temperature':1500,'threshold':1500},
              'per_class':{'checkpoint':20,'temperature':15,'threshold':15},
              'original_validation_indices_sha256':sha256(indices.astype('<i8').tobytes()).hexdigest(),
              'boundary':'No model training/temperature fitting/threshold fitting or OOD evaluation has started'}
    write(RUN/'partition.json',metadata)
    write(RUN/'status.json',{'state':'partition_prepared_pending_independent_audit','prepared_at':now(),'partition_sha256':digest(RUN/'partition.npz'),'bound_input_count':len(pre['original_artifacts']),'trainer_running':False})
    print({'state':'partition_prepared_pending_independent_audit','roles':metadata['role_counts'],'bound_input_count':len(pre['original_artifacts'])},flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['declare','partition']);stage=p.parse_args().stage
    declaration() if stage=='declare' else partition()

if __name__=='__main__':main()
