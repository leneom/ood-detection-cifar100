"""Evaluate additional OOD only after the original matrix has finished."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from run_matrix import child, process_alive, save_status
from ood.cache import load_svhn_cache
from ood.textures_cache import load_textures_cache


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hash_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def matrix_ready(folder, seeds):
    """Return false while active; fail closed on paused/failed/incomplete matrices."""
    folder = Path(folder)
    status = read(folder / 'status.json')
    if status.get('state') in ('failed', 'paused') or (folder / 'STOP').exists():
        raise InterruptedError('original matrix failed or paused; extension must wait for recovery')
    lock = folder / 'controller.lock'
    if lock.exists() and process_alive(int(lock.read_text().strip())):
        return False
    if status.get('state') != 'complete':
        raise InterruptedError('original matrix has stopped without completing')
    if set(status.get('completed_seeds', [])) != set(seeds):
        raise ValueError('original matrix completed seed set differs')
    return True


def other_gpu_python():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    # Windows reports browser/rendering processes here too; Python identifies our
    # training/evaluation compute jobs without blocking forever on desktop apps.
    return [int(row[0]) for row in csv.reader(output.splitlines())
            if len(row) == 2 and 'python' in row[1].lower() and int(row[0]) != os.getpid()]


def verify_extension(base_path, extension_path, checkpoint, datasets):
    base, result = read(base_path / 'results.json'), read(extension_path / 'results.json')
    for key in ('training_config', 'checkpoint_epoch', 'checkpoint_sha256', 'temperature', 'feature_fit'):
        if result[key] != base[key]:
            raise ValueError(f'extension differs from baseline: {key}')
    for key in ('torch', 'torchvision', 'numpy', 'cuda_runtime', 'gpu'):
        if result.get('environment', {}).get(key) != base.get('environment', {}).get(key):
            raise ValueError(f'extension evaluation environment differs: {key}')
    if base.get('evaluation_source_hashes'):
        for name in ('data.py', 'models.py', 'features.py', 'scores.py', 'metrics.py'):
            if result['evaluation_source_hashes'][name] != base['evaluation_source_hashes'][name]:
                raise ValueError(f'extension detector source differs: {name}')
    if result['checkpoint_sha256'] != hash_file(checkpoint) or set(result['datasets']) != set(datasets):
        raise ValueError('extension checkpoint or dataset list differs')
    baseline_name = next(iter(base['datasets']))
    expected_methods = set(base['datasets'][baseline_name]['methods'])
    if any(set(entry['methods']) != expected_methods for entry in result['datasets'].values()):
        raise ValueError('extension detector methods differ')
    comparison = {}
    for filename in ('id_logits.npz', 'validation_logits.npz', 'features.npz'):
        with np.load(base_path / filename, allow_pickle=False) as original, np.load(extension_path / filename, allow_pickle=False) as current:
            for key in original.files:
                if not np.allclose(original[key], current[key], atol=1e-6, rtol=1e-6):
                    raise ValueError('extension changed ID logits or extracted features')
    with np.load(base_path / f'{baseline_name}_scores.npz', allow_pickle=False) as original:
        for dataset in datasets:
            with np.load(extension_path / f'{dataset}_scores.npz', allow_pickle=False) as scores:
                for method in expected_methods:
                    for part in ('id', 'val'):
                        reference, current = original[f'{method}_{part}'], scores[f'{method}_{part}']
                        # Inverse covariance amplifies sub-micro-unit float32 feature
                        # differences. This is a numerical comparison tolerance, not
                        # an experimental shrinkage/threshold/score adjustment.
                        relative_tolerance = 1e-4 if method == 'mahalanobis' else 1e-6
                        if reference.shape != current.shape or not np.allclose(reference, current, atol=1e-6, rtol=relative_tolerance):
                            raise ValueError('extension changed ID scores or calibration population')
                        comparison[f'{dataset}/{method}/{part}'] = {
                            'maximum_absolute_difference': float(np.abs(reference - current).max()),
                            'relative_tolerance': relative_tolerance, 'absolute_tolerance': 1e-6}
    (extension_path / 'baseline-comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--dataset', choices=['cifar100', 'synthetic'], default='cifar100')
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--matrix-control-dir', default='runs/matrix')
    parser.add_argument('--control-dir', default='runs/svhn-extension')
    parser.add_argument('--ood', nargs='+', choices=['svhn', 'textures', 'noise', 'constant'], default=['svhn'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cuda')
    parser.add_argument('--analysis-python', required=True)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('duplicate seeds')
    os.chdir(ROOT)
    control = Path(args.control_dir)
    control.mkdir(parents=True, exist_ok=True)
    lock = control / 'controller.lock'
    if lock.exists():
        if process_alive(int(lock.read_text().strip())):
            raise RuntimeError('extension controller is already active')
        lock.unlink()
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        save_status(control, state='waiting_for_matrix', controller_pid=os.getpid(), seeds=args.seeds)
        while not matrix_ready(args.matrix_control_dir, args.seeds):
            if (control / 'STOP').exists():
                raise InterruptedError('extension STOP requested')
            time.sleep(10)
        completed, evaluations = [], []
        for seed in args.seeds:
            if (control / 'STOP').exists():
                raise InterruptedError('extension STOP requested')
            if args.device != 'cpu':
                while active := other_gpu_python():
                    save_status(control, state='waiting_for_gpu_python', active_pids=active, controller_pid=os.getpid())
                    if (control / 'STOP').exists():
                        raise InterruptedError('extension STOP requested')
                    time.sleep(10)
            run = Path(args.runs_root) / f'{args.dataset}-seed{seed}'
            base_path, checkpoint = run / 'eval-final', run / 'final-best.pt'
            base, status = read(base_path / 'results.json'), read(run / 'status.json')
            if status.get('state') != 'complete' or status.get('epoch') != base['training_config']['epochs']:
                raise ValueError('extension requires full completed training')
            if hash_file(checkpoint) != base['checkpoint_sha256']:
                raise ValueError('baseline frozen checkpoint hash differs')
            provenance = None
            if 'svhn' in args.ood:
                _, _, provenance = load_svhn_cache(base['training_config']['data_root'])
            if 'textures' in args.ood:
                if len(args.ood) != 1:
                    raise ValueError('Textures extension uses a separate predeclared population')
                _, _, provenance = load_textures_cache(base['training_config']['data_root'])
            evaluation = run / ('eval-extension-' + '-'.join(args.ood))
            if not (evaluation / 'results.json').exists():
                if evaluation.exists() and any(evaluation.iterdir()):
                    evaluation.rename(evaluation.with_name(evaluation.name + '-failed-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
                methods = list(base['datasets'][next(iter(base['datasets']))]['methods'])
                fit = base.get('feature_fit')
                command = [sys.executable, '-u', 'evaluate.py', '--checkpoint', str(checkpoint), '--ood', *args.ood,
                           '--output-dir', str(evaluation), '--device', args.device, '--temperature', str(base['temperature']),
                           '--methods', *[m for m in methods if m not in ('knn', 'mahalanobis')]]
                if fit:
                    command += ['--feature-methods', *[m for m in methods if m in ('knn', 'mahalanobis')],
                                '--knn-k', str(fit['knn_k']), '--shrinkage', str(fit['mahalanobis_shrinkage'])]
                child(command, run, 'extension-evaluate', control)
            extension_result = verify_extension(base_path, evaluation, checkpoint, args.ood)
            if 'textures' in args.ood:
                pixels = extension_result.get('input_pixels', {}).get('ood', {}).get('textures', {})
                if pixels.get('sha256') != provenance['images_sha256'] or pixels.get('shape') != provenance['shape']:
                    raise ValueError('executed Textures input differs from the audited cached pixels')
            if provenance:
                (evaluation / 'data_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
            report = control / f'report-seed{seed}'
            child([args.analysis_python, 'tools/audit_evaluation.py', '--evaluation', str(evaluation),
                   '--output', str(report / 'independent-audit.json')], run, 'extension-audit', control)
            child([args.analysis_python, 'tools/build_report.py', '--evaluation', str(evaluation), '--output-dir', str(report),
                   '--label', (f'Synthetic-classifier diagnostic only, seed {seed}' if args.dataset == 'synthetic'
                               else f'Completed baseline extension, seed {seed}')], run, 'extension-report', control)
            child([args.analysis_python, 'tools/analyze_failures.py', '--evaluation', str(evaluation),
                   '--output-dir', str(report / 'failure-analysis')], run, 'extension-failure-analysis', control)
            completed.append(seed)
            evaluations.append(str(evaluation))
            save_status(control, state='seed_complete', controller_pid=os.getpid(), completed_seeds=completed)
        child([args.analysis_python, 'tools/aggregate_results.py', '--evaluations', *evaluations,
               '--output-dir', str(control / 'summary')], control, 'extension-summary', control)
        save_status(control, state='complete', controller_pid=os.getpid(), completed_seeds=completed)
    except InterruptedError as error:
        save_status(control, state='paused', reason=str(error), controller_pid=os.getpid())
    except Exception as error:
        save_status(control, state='failed', error=repr(error), controller_pid=os.getpid())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
