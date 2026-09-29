"""Fit every fixed geometry statistic on independently audited training banks, before OOD scoring."""
import os
from pathlib import Path
import shutil
import numpy as np
from geometry_io import ROOT, RUN, digest, read, write, load, save_arrays, require, now, frozen_inputs
from geometry_core import fit_geometry


def main():
    import msvcrt
    with (RUN / 'fit.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            bank_audit = read(RUN / 'bank-audit.json')
            require(bank_audit['state'] == 'passed', 'training banks not independently audited')
            require(read(RUN / 'bank-export-status.json')['state'] == 'complete', 'bank producer still active/incomplete')
            originals = frozen_inputs()
            paths = [RUN / 'bank-audit.json', RUN / 'head-audit.json', RUN / 'bank-export-inputs.json', RUN / 'bank-export-status.json']
            for seed in [0, 1, 2]:
                directory = RUN / f'seed{seed}'
                record = bank_audit['per_seed'][seed]
                require(digest(directory / 'training-bank.npz') == record['bank_sha256'], 'audited training bank changed')
                require(digest(directory / 'bank.json') == record['bank_metadata_sha256'], 'bank metadata changed')
                paths.extend([directory / 'training-bank.npz', directory / 'bank.json', directory / 'extraction-identity.npz'])
            originals.update({str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths})
            source_files = [Path(__file__), ROOT / 'tools/geometry_core.py', ROOT / 'tools/geometry_io.py']
            sources = {p.name: digest(p) for p in source_files}
            manifest_path = RUN / 'fit-inputs.json'
            if manifest_path.exists():
                manifest = read(manifest_path)
                require(manifest['original_artifacts'] == originals and manifest['source_hashes'] == sources, 'fit source/input mismatch')
            else:
                require(not any(RUN.glob('seed*/fit.json')), 'fit exists before frozen fit inputs')
                snapshot = RUN / 'fit-source'
                snapshot.mkdir(exist_ok=True)
                for path in source_files:
                    shutil.copy2(path, snapshot / path.name)
                manifest = {'state': 'inputs_frozen_before_geometry_fitting', 'prepared_at': now(),
                            'original_artifacts': originals, 'source_hashes': sources,
                            'principal_dimensions': 256, 'react_global_quantile': .99,
                            'fit_population': '45000 original ID training features only', 'OOD_used_in_fitting': False,
                            'numpy': np.__version__, 'BLAS_threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']}}
                write(manifest_path, manifest)
            write(RUN / 'fit-status.json', {'state': 'running', 'pid': os.getpid(), 'started_at': now()})
            for seed in [0, 1, 2]:
                directory = RUN / f'seed{seed}'
                path = directory / 'fit.json'
                if path.exists():
                    fitted = read(path)
                    require(fitted['fit_inputs_sha256'] == digest(manifest_path) and fitted['fit_state_sha256'] == digest(directory / 'fit-state.npz'),
                            'completed fit changed')
                    continue
                bank = load(directory / 'training-bank.npz')
                metadata, state = fit_geometry(bank['features'], bank['weight'], bank['bias'], principal_dim=256, quantile=.99)
                require(metadata['training_count'] == 45000 and metadata['feature_dimensions'] == 512, 'fit population mismatch')
                save_arrays(directory / 'fit-state.npz', **state)
                metadata.update({'state': 'complete', 'seed': seed, 'fitted_at': now(),
                                 'fit_inputs_sha256': digest(manifest_path), 'training_bank_sha256': digest(directory / 'training-bank.npz'),
                                 'fit_state_sha256': digest(directory / 'fit-state.npz'),
                                 'checkpoint_sha256': read(directory / 'bank.json')['checkpoint_sha256']})
                write(path, metadata)
                print(__import__('json').dumps({'seed': seed, 'alpha': metadata['alpha'], 'clipping_level': metadata['clipping_level'],
                                              'D': 256, 'fit_count': 45000}), flush=True)
                del bank, state
                write(RUN / 'fit-status.json', {'state': 'running', 'pid': os.getpid(), 'completed_seeds': list(range(seed + 1)), 'updated_at': now()})
            for name, expected in originals.items():
                require(digest(ROOT / name) == expected, 'original fit input changed')
            write(RUN / 'fit-status.json', {'state': 'complete', 'pid': os.getpid(), 'completed_seeds': [0, 1, 2],
                                           'original_artifacts_unchanged': True, 'completed_at': now()})
            write(RUN / 'status.json', {'state': 'ready_for_evaluation', 'completed_training_banks': [0, 1, 2],
                                       'completed_geometry_fits': [0, 1, 2], 'OOD_scoring_started': False, 'updated_at': now()})
        except Exception as error:
            write(RUN / 'fit-status.json', {'state': 'failed', 'pid': os.getpid(), 'error': repr(error), 'updated_at': now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
