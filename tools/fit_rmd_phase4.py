"""Train-only fixed RMD fitting with archived inputs and source snapshots."""
import os
from pathlib import Path
import shutil
import numpy as np
from geometry_io import ROOT,read,write,digest,load,save_arrays,require,now
from rmd_core import fit_rmd

RUN = ROOT/'runs/rmd-phase4'


def main():
    import msvcrt
    with (RUN/'fit.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            prereg=read(RUN/'predeclaration.json')
            require(prereg['expected_seeds']==[0,1,2] and prereg['training_count']==45000 and
                    prereg['class_count']==100 and prereg['feature_dimensions']==512 and prereg['shrinkage']==.01,
                    'predeclared settings changed')
            inputs=dict(prereg['original_artifacts'])
            inputs['runs/rmd-phase4/predeclaration.json']=digest(RUN/'predeclaration.json')
            for name,h in inputs.items():require(digest(ROOT/name)==h,'frozen original changed: '+name)
            files=[Path(__file__),ROOT/'tools/rmd_core.py',ROOT/'tools/geometry_io.py']
            sources={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files}
            manifest_path=RUN/'fit-inputs.json'
            if manifest_path.exists():
                manifest=read(manifest_path)
                require(manifest['original_artifacts']==inputs and manifest['source_hashes']==sources,'incompatible fit reuse')
            else:
                snapshot=RUN/'fit-source';snapshot.mkdir(exist_ok=True)
                for p in files:shutil.copy2(p,snapshot/p.name)
                manifest={'state':'inputs_frozen_before_RMD_fitting','prepared_at':now(),'original_artifacts':inputs,
                          'source_hashes':sources,'numpy':np.__version__,'fit_population':'original training ID only',
                          'class_count':100,'feature_dimensions':512,'shrinkage':.01}
                write(manifest_path,manifest)
            for seed in [0,1,2]:
                write(RUN/'status.json',{'state':'running','stage':'training-ID-only CPU RMD fit','pid':os.getpid(),
                                       'completed_fit_seeds':list(range(seed)),'updated_at':now()})
                output=RUN/f'seed{seed}';output.mkdir(exist_ok=True)
                if (output/'fit.json').exists():
                    previous=read(output/'fit.json')
                    require(previous['fit_inputs_sha256']==digest(manifest_path) and previous['fit_state_sha256']==digest(output/'fit-state.npz'),'completed fit changed')
                    continue
                require(not (output/'fit-state.npz').exists(),'partial state retained; do not overwrite')
                source=ROOT/f'runs/geometry-phase3/seed{seed}'
                with np.load(source/'training-bank.npz',allow_pickle=False) as bank:
                    f,y,indices=bank['features'],bank['labels'],bank['indices']
                original=load(ROOT/f'runs/cifar100-seed{seed}/split_indices.npz')
                require(f.shape==(45000,512) and f.dtype==np.float32 and y.shape==(45000,) and
                        np.array_equal(np.bincount(y,minlength=100),np.full(100,450)),'training bank dimensions/balance changed')
                require(np.array_equal(indices,original['train']) and not np.intersect1d(indices,original['val']).size,'training/validation overlap or order changed')
                metadata,state=fit_rmd(f,y)
                save_arrays(output/'fit-state.npz',**state)
                metadata.update({'state':'complete','seed':seed,'fitted_at':now(),'fit_inputs_sha256':digest(manifest_path),
                                 'training_bank_sha256':digest(source/'training-bank.npz'),
                                 'checkpoint_sha256':read(source/'fit.json')['checkpoint_sha256'],
                                 'fit_state_sha256':digest(output/'fit-state.npz')})
                write(output/'fit.json',metadata)
                print(__import__('json').dumps({'seed':seed,'state':'RMD_fit_saved','training_rows':45000,
                                              'within_condition':metadata['within_condition_number'],
                                              'background_condition':metadata['background_condition_number']}),flush=True)
            for name,h in inputs.items():require(digest(ROOT/name)==h,'original input changed during fit')
            for name,h in sources.items():require(digest(ROOT/name)==h,'source changed during fit')
            write(RUN/'fit-status.json',{'state':'complete','completed_seeds':[0,1,2],'original_artifacts_unchanged':True,'completed_at':now()})
            write(RUN/'status.json',{'state':'fits_complete_pending_independent_audit','completed_fit_seeds':[0,1,2],
                                    'RMD_OOD_scoring_started':False,'completed_at':now()})
        except Exception as error:
            write(RUN/'status.json',{'state':'failed','stage':'RMD fitting','error':repr(error),'updated_at':now()})
            raise
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':main()
