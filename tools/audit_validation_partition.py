"""Independently reconstruct the frozen 20/15/15 per-class ID roles."""
import hashlib
import numpy as np
from pathlib import Path
from geometry_io import ROOT,read,write,load,digest,require,now

RUN=ROOT/'runs/validation-isolation'

def main():
    pre=read(RUN/'predeclaration.json');meta=read(RUN/'partition.json');status=read(RUN/'status.json')
    require(pre['state']=='declared_before_three_way_partition_draw' and status['state']=='partition_prepared_pending_independent_audit' and meta['state']=='partition_prepared_pending_independent_audit','partition lifecycle changed')
    require(pre['source_sha256']==meta['source_sha256']==digest(ROOT/'tools/prepare_validation_isolation.py') and meta['predeclaration_sha256']==digest(RUN/'predeclaration.json'),'partition source/declaration changed')
    require(meta['partition_sha256']==status['partition_sha256']==digest(RUN/'partition.npz') and meta['rng']==pre['partition_rng']==2029 and pre['per_class']==[20,15,15],'partition artifact/settings changed')
    for n,h in pre['original_artifacts'].items():require(digest(ROOT/n)==h,'original frozen source changed '+n)
    old=load(ROOT/'runs/calibration-phase2/calibration_split.npz');labels=old['validation_labels'];indices=old['original_validation_indices'];actual=load(RUN/'partition.npz')
    require(set(actual)=={'checkpoint_rows','temperature_rows','threshold_rows','checkpoint_original_indices','temperature_original_indices','threshold_original_indices','original_validation_indices','validation_labels'},'partition schema changed')
    require(np.array_equal(actual['original_validation_indices'],indices) and np.array_equal(actual['validation_labels'],labels),'original validation rows or labels changed')
    rng=np.random.default_rng(2029);expected=[[],[],[]]
    for c in range(100):
        rows=np.where(labels==c)[0];require(len(rows)==50,'original class count changed')
        shuffled=rng.permutation(rows)
        for destination,part in zip(expected,[shuffled[:20],shuffled[20:35],shuffled[35:]]):destination.extend(int(v) for v in part)
    combined=[];checks={'regenerated_original_positions':0,'per_class_role_cells':0,'cross_seed_original_split_checks':0,'original_index_alignments':0}
    for role,need,rows in zip(['checkpoint','temperature','threshold'],[20,15,15],expected):
        generated=np.sort(np.asarray(rows,dtype=np.int64));found=actual[role+'_rows'];original=actual[role+'_original_indices']
        require(found.dtype==np.int64 and found.shape==(need*100,) and np.array_equal(found,generated) and np.all(found[1:]>found[:-1]),'regenerated role differs '+role)
        require(np.array_equal(original,indices[found]),'role original index differs '+role)
        require(np.array_equal(np.bincount(labels[found],minlength=100),np.full(100,need)),'class role count differs '+role)
        combined.extend(int(v) for v in found);checks['regenerated_original_positions']+=len(found);checks['per_class_role_cells']+=100;checks['original_index_alignments']+=len(original)
    require(sorted(combined)==list(range(5000)) and len(set(combined))==5000,'validation roles overlap or omit')
    require(hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==meta['original_validation_indices_sha256'],'validation index hash differs')
    for seed in range(3):
        split=load(ROOT/f'runs/cifar100-seed{seed}/split_indices.npz')
        require(np.array_equal(split['val'],indices) and len(split['train'])==45000 and not np.intersect1d(split['train'],indices).size,'old split differs for seed '+str(seed))
        for role in ['checkpoint','temperature','threshold']:
            require(not np.intersect1d(split['train'],actual[role+'_original_indices']).size,'new role overlaps old training '+role);checks['cross_seed_original_split_checks']+=1
    result={'state':'passed','audited_at':now(),'predeclaration_sha256':digest(RUN/'predeclaration.json'),'partition_sha256':digest(RUN/'partition.npz'),
            'partition_metadata_sha256':digest(RUN/'partition.json'),'independent_source_sha256':digest(__file__),'bound_input_count':len(pre['original_artifacts']),
            'checks':checks,'boundary':'Three disjoint 20/15/15 per-class ID validation roles; no GPU trainer, temperature fit, threshold fit or OOD test run yet'}
    for n,h in pre['original_artifacts'].items():require(digest(ROOT/n)==h,'source changed during independent partition audit '+n)
    write(RUN/'partition-audit.json',result)
    write(RUN/'status.json',{'state':'partition_ready_training_not_started','partition_sha256':result['partition_sha256'],'audit_sha256':digest(RUN/'partition-audit.json'),
                             'bound_input_count':len(pre['original_artifacts']),'trainer_running':False,'updated_at':now()})
    print(result,flush=True)

if __name__=='__main__':main()
