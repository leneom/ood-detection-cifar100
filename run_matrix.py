"""Sequential, resumable local experiment controller; one training child at a time."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def process_alive(pid):
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.GetExitCodeProcess.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error in (0, 87):
                return False
            raise ctypes.WinError(error)
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def save_status(folder, **status):
    status['updated_at'] = datetime.now(timezone.utc).isoformat()
    temp = folder / 'status.json.tmp'
    temp.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    temp.replace(folder / 'status.json')


def child(command, folder, name, control):
    if (control / 'STOP').exists():
        raise InterruptedError('STOP requested before starting next stage')
    with (folder / f'{name}.stdout.log').open('a', encoding='utf-8') as stdout, (folder / f'{name}.stderr.log').open('a', encoding='utf-8') as stderr:
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, **kwargs)
        save_status(control, state='running', stage=name, child_pid=process.pid, command=command)
        try:
            while True:
                try:
                    code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    if (control / 'STOP').exists():
                        process.terminate()
                        process.wait(timeout=20)
                        raise InterruptedError('STOP requested; last completed epoch remains resumable')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--dataset', choices=['cifar100', 'synthetic'], default='cifar100')
    p.add_argument('--model', choices=['resnet18', 'tiny'], default='resnet18')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--ood', nargs='+', default=['cifar10', 'mnist', 'noise'])
    p.add_argument('--feature-methods', nargs='*', default=['knn', 'mahalanobis'])
    p.add_argument('--knn-k', type=int, default=50)
    p.add_argument('--runs-root', default='runs')
    p.add_argument('--control-dir', default='runs/matrix')
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--plot-python', help='optional Python with NumPy and Matplotlib')
    return p


def main():
    args = parser().parse_args()
    control = Path(args.control_dir)
    control.mkdir(parents=True, exist_ok=True)
    lock = control / 'controller.lock'
    if lock.exists():
        previous_pid = int(lock.read_text().strip())
        if process_alive(previous_pid):
            raise RuntimeError(f'controller {previous_pid} is already active')
        lock.unlink()  # Only this controller's stale lock, not any run data.
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        save_status(control, state='starting', controller_pid=os.getpid(), seeds=args.seeds)
        if args.wait_pid:
            save_status(control, state='waiting_for_existing_training', wait_pid=args.wait_pid, controller_pid=os.getpid())
            while process_alive(args.wait_pid):
                if (control / 'STOP').exists():
                    raise InterruptedError('STOP requested while awaiting existing training')
                time.sleep(10)
        results = []
        for seed in args.seeds:
            run = Path(args.runs_root) / f'{args.dataset}-seed{seed}'
            run.mkdir(parents=True, exist_ok=True)
            status_path = run / 'status.json'
            status = json.loads(status_path.read_text(encoding='utf-8-sig')) if status_path.exists() else {}
            config_path = run / 'config.json'
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding='utf-8-sig'))
                for key, expected in [('dataset', args.dataset), ('model', args.model), ('seed', seed),
                                      ('epochs', args.epochs), ('batch_size', args.batch_size), ('amp', args.amp)]:
                    if config[key] != expected:
                        raise ValueError(f'existing run has incompatible {key}')
            source_snapshot = run / 'source_snapshot'
            if not source_snapshot.exists():
                shutil.copytree(ROOT / 'src' / 'ood', source_snapshot, ignore=shutil.ignore_patterns('__pycache__'))
            if status.get('state') != 'complete' or status.get('epoch') != args.epochs:
                command = [sys.executable, '-u', 'train.py', '--dataset', args.dataset, '--model', args.model,
                           '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--seed', str(seed),
                           '--output-dir', str(run), '--download']
                if args.amp:
                    command.append('--amp')
                if (run / 'last.pt').exists():
                    command.append('--resume')
                child(command, run, 'matrix-train', control)
            # Freeze selected ID-validation weights before all test evaluation.
            snapshot = run / 'final-best.pt'
            if not snapshot.exists():
                shutil.copyfile(run / 'best.pt', snapshot)
            evaluation = run / 'eval-final'
            if not (evaluation / 'results.json').exists():
                # A previous failed evaluation is preserved in a sibling directory.
                if evaluation.exists() and any(evaluation.iterdir()):
                    previous = evaluation.with_name('eval-final-failed-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
                    evaluation.rename(previous)
                command = [sys.executable, '-u', 'evaluate.py', '--checkpoint', str(snapshot), '--ood', *args.ood,
                           '--output-dir', str(evaluation), '--download', '--knn-k', str(args.knn_k)]
                if args.feature_methods:
                    command += ['--feature-methods', *args.feature_methods]
                child(command, run, 'matrix-evaluate', control)
            result = json.loads((evaluation / 'results.json').read_text(encoding='utf-8'))
            with snapshot.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
            if digest is None:
                hasher = hashlib.sha256()
                with snapshot.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        hasher.update(block)
                digest = hasher.hexdigest()
            if result['checkpoint_sha256'] != digest or set(result['datasets']) != set(args.ood):
                raise ValueError('existing evaluation checkpoint or datasets differ')
            expected_methods = {'msp', 'energy', 'maxlogit', 'entropy', *args.feature_methods}
            if any(set(entry['methods']) != expected_methods for entry in result['datasets'].values()):
                raise ValueError('existing evaluation methods differ')
            if 'knn' in args.feature_methods and result['feature_fit']['knn_k'] != args.knn_k:
                raise ValueError('existing evaluation k differs')
            results.append(result)
            if args.plot_python:
                child([args.plot_python, 'tools/build_report.py', '--evaluation', str(evaluation),
                       '--output-dir', f'report/final-seed{seed}', '--label', f'Full {args.epochs}-epoch schedule, seed {seed}'],
                      run, 'matrix-report', control)
            save_status(control, state='seed_complete', seed=seed, controller_pid=os.getpid(), completed_seeds=[r['training_config']['seed'] for r in results])
        # Keep per-seed values; statistics can be reconstructed independently.
        temp = control / 'results.json.tmp'
        temp.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(control / 'results.json')
        evaluation_paths = [str(Path(args.runs_root) / f'{args.dataset}-seed{seed}' / 'eval-final') for seed in args.seeds]
        child([sys.executable, 'tools/aggregate_results.py', '--evaluations', *evaluation_paths,
               '--output-dir', str(control / 'summary')], control, 'matrix-summary', control)
        save_status(control, state='complete', controller_pid=os.getpid(), completed_seeds=args.seeds)

    except InterruptedError as error:
        save_status(control, state='paused', reason=str(error))
    except Exception as error:
        save_status(control, state='failed', error=repr(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
