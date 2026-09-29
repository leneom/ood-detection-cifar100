"""Training-ID-only ASH numerical preflight; no OOD shaping or scoring."""
import os
from pathlib import Path
import shutil
import hashlib
import numpy as np
from geometry_io import ROOT,read,write,load,save_arrays,digest,require,now
from ash_core import ash_scores,METHODS
RUN=ROOT/'runs/ash-phase5'


def main():
    import msvcrt
    with (RUN/'preflight.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            declaration=read(RUN/'predeclaration.json')
            require(declaration['expected_seeds']==[0,1,2] and declaration['pruning_percentile']==65 and declaration['retained_coordinates']==179 and declaration['named_additions']==METHODS,'fixed settings changed')
            inputs=dict(declaration['original_artifacts']);inputs['runs/ash-phase5/predeclaration.json']=digest(RUN/'predeclaration.json')
            for name,h in inputs.items():require(digest(ROOT/name)==h,'frozen input changed: '+name)
            files=[Path(__file__),ROOT/'tools/ash_core.py',ROOT/'tools/geometry_io.py']
            sources={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files}
            manifest_path=RUN/'preflight-inputs.json'
            if manifest_path.exists():
                manifest=read(manifest_path)
                require(manifest['original_artifacts']==inputs and manifest['source_hashes']==sources,'incompatible preflight reuse')
            else:
                snapshot=RUN/'preflight-source';snapshot.mkdir(exist_ok=True)
                for p in files:shutil.copy2(p,snapshot/p.name)
                write(manifest_path,{'state':'frozen_before_training_ID_preflight','prepared_at':now(),'original_artifacts':inputs,'source_hashes':sources,'percentile':65,'k':179,'methods':METHODS,'numpy':np.__version__,'fit_population':'none; training ID used for numerical checks only'})
            head=read(ROOT/'runs/geometry-phase3/head-audit.json'); bank_audit=read(ROOT/'runs/geometry-phase3/bank-audit.json')
            require(head['state']==bank_audit['state']=='passed','prior head/bank audit absent')
            for seed in [0,1,2]:
                write(RUN/'status.json',{'state':'running','stage':'training-ID-only ASH preflight','pid':os.getpid(),'completed_preflight_seeds':list(range(seed)),'updated_at':now()})
                out=RUN/f'seed{seed}';out.mkdir(exist_ok=True)
                if (out/'preflight.json').exists():
                    meta=read(out/'preflight.json')
                    require(meta['inputs_sha256']==digest(manifest_path) and meta['raw_sha256']==digest(out/'preflight.npz'),'completed preflight changed')
                    continue
                require(not (out/'preflight.npz').exists(),'partial preflight retained')
                source=ROOT/f'runs/geometry-phase3/seed{seed}/training-bank.npz'
                bank=load(source);f=bank['features'];w=bank['weight'];b=bank['bias']
                original=load(ROOT/f'runs/cifar100-seed{seed}/split_indices.npz')
                require(f.shape==(45000,512) and f.dtype==np.float32 and w.shape==(100,512) and b.shape==(100,),'bank/head shape changed')
                require(np.array_equal(bank['indices'],original['train']) and not np.intersect1d(bank['indices'],original['val']).size and np.array_equal(np.bincount(bank['labels'],minlength=100),np.full(100,450)),'ID split/balance changed')
                identity=head['per_seed'][seed]
                require(digest(source)==identity['bank_sha256']==bank_audit['per_seed'][seed]['bank_sha256'],'native audited bank changed')
                require(hashlib.sha256(w.tobytes()).hexdigest()==identity['weight_sha256'] and hashlib.sha256(b.tobytes()).hexdigest()==identity['bias_sha256'],'original checkpoint head changed')
                scores,middle=ash_scores(f,w,b)
                affine=f.astype(float)@w.astype(float).T+b.astype(float)
                middle.update({m:s for m,s in scores.items()});middle.update({'affine_logits':affine,'labels':bank['labels'],'indices':bank['indices']})
                save_arrays(out/'preflight.npz',**middle)
                meta={'state':'complete_pending_independent_audit','seed':seed,'checked_at':now(),'training_rows':45000,'feature_dimensions':512,'class_count':100,'percentile':65,'k':179,'score_direction':'higher_is_ood','inputs_sha256':digest(manifest_path),'raw_sha256':digest(out/'preflight.npz'),'training_bank_sha256':digest(source),'checkpoint_sha256':identity['checkpoint_sha256'],'head_audit_sha256':digest(ROOT/'runs/geometry-phase3/head-audit.json'),'bank_audit_sha256':digest(ROOT/'runs/geometry-phase3/bank-audit.json'),'original_GPU_float32_max_affine_difference':float(np.abs(affine-bank['logits']).max()),'boundary':'training ID preflight, no fitting, no OOD shaping'}
                write(out/'preflight.json',meta)
                print('seed',seed,'training-ID ASH preflight saved',flush=True)
                del bank,middle,scores,affine
            for name,h in inputs.items():require(digest(ROOT/name)==h,'original input changed during preflight')
            for name,h in sources.items():require(digest(ROOT/name)==h,'preflight source changed')
            write(RUN/'preflight-status.json',{'state':'complete','seeds':[0,1,2],'completed_at':now()})
            write(RUN/'status.json',{'state':'preflight_complete_pending_independent_audit','completed_preflight_seeds':[0,1,2],'ASH_OOD_scoring_started':False,'updated_at':now()})
        except Exception as error:
            write(RUN/'status.json',{'state':'failed','stage':'ASH preflight','error':repr(error),'updated_at':now()});raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':main()
