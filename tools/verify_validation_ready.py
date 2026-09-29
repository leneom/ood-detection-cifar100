"""Readonly gate before implementing/starting the isolated GPU training schedule."""
import numpy as np
from geometry_io import ROOT,read,load,digest,require

RUN=ROOT/'runs/validation-isolation'

def main():
    status=read(RUN/'status.json');audit=read(RUN/'partition-audit.json');pre=read(RUN/'predeclaration.json');meta=read(RUN/'partition.json')
    require(status['state']=='partition_ready_training_not_started' and status['trainer_running'] is False,'validation-isolation trainer already running or partition not ready')
    require(audit['state']=='passed' and audit['predeclaration_sha256']==digest(RUN/'predeclaration.json') and audit['partition_sha256']==status['partition_sha256']==meta['partition_sha256']==digest(RUN/'partition.npz'),'partition audit binding changed')
    require(audit['partition_metadata_sha256']==digest(RUN/'partition.json') and pre['source_sha256']==meta['source_sha256']==digest(ROOT/'tools/prepare_validation_isolation.py') and audit['independent_source_sha256']==digest(ROOT/'tools/audit_validation_partition.py'),'executed preparation/audit source changed')
    for n,h in pre['original_artifacts'].items():require(digest(ROOT/n)==h,'frozen predecessor changed '+n)
    a=load(RUN/'partition.npz');old=load(ROOT/'runs/calibration-phase2/calibration_split.npz')
    require(np.array_equal(a['original_validation_indices'],old['original_validation_indices']) and np.array_equal(a['validation_labels'],old['validation_labels']),'original validation source changed')
    for role,n in [('checkpoint',2000),('temperature',1500),('threshold',1500)]:
        require(len(a[role+'_rows'])==n and np.array_equal(a[role+'_original_indices'],a['original_validation_indices'][a[role+'_rows']]),'role count/index changed '+role)
    require(np.array_equal(np.sort(np.concatenate([a[k+'_rows'] for k in ['checkpoint','temperature','threshold']])),np.arange(5000)),'roles overlap or omit')
    print({'state':'verified_readonly','roles':{'checkpoint':2000,'temperature':1500,'threshold':1500},'bound_inputs':len(pre['original_artifacts']),
           'boundary':'ID-only disjoint partition verified. This checks saved evidence; it does not start a GPU process or establish fresh OOD confirmation.'},flush=True)

if __name__=='__main__':main()
