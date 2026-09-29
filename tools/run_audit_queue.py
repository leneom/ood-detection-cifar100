"""CPU-only independent audits after the main matrix and its extension finish.

Does not modify active controllers or src/ood. A failed/paused producer stops
the queue. Existing audit reports are retained in separate run directories.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_matrix import child, process_alive, save_status


def producer_ready(folder, seeds):
    folder = Path(folder)
    status = json.loads((folder / 'status.json').read_text(encoding='utf-8-sig'))
    if status.get('state') in ('failed', 'paused') or (folder / 'STOP').exists():
        raise InterruptedError('producer failed or paused; recover it before auditing')
    lock = folder / 'controller.lock'
    if lock.exists() and process_alive(int(lock.read_text().strip())):
        return False
    if status.get('state') != 'complete':
        raise InterruptedError('producer stopped without completing')
    completed = status.get('completed_seeds', [])
    if len(completed) != len(set(completed)) or sorted(completed) != sorted(seeds):
        raise ValueError('producer must finish all expected seeds exactly once')
    return True


def reserve_report(folder):
    folder = Path(folder)
    if folder.exists():
        folder = folder.with_name(folder.name + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def local_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('audit inputs/outputs must stay inside ood_detection')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--dataset', choices=['cifar100', 'synthetic'], default='cifar100')
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--matrix-control-dir', default='runs/matrix')
    parser.add_argument('--extension-control-dir', default='runs/svhn-extension')
    parser.add_argument('--extension-ood', nargs='+', default=['svhn'])
    parser.add_argument('--control-dir', default='runs/audit-queue')
    parser.add_argument('--report-root', default='report')
    parser.add_argument('--analysis-python', required=True)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or len(args.extension_ood) != len(set(args.extension_ood)):
        raise ValueError('duplicate seeds or extension datasets')
    if any(not name.isalnum() for name in args.extension_ood):
        raise ValueError('extension dataset names must be simple identifiers')
    os.chdir(ROOT)
    control = local_path(args.control_dir)
    runs, reports = local_path(args.runs_root), local_path(args.report_root)
    matrix, extension = local_path(args.matrix_control_dir), local_path(args.extension_control_dir)
    control.mkdir(parents=True, exist_ok=True)
    lock = control / 'controller.lock'
    if lock.exists():
        if process_alive(int(lock.read_text().strip())):
            raise RuntimeError('audit queue is already active')
        lock.unlink()
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    completed = []
    # Mask CUDA and constrain CPU math threads in this queue and its own children.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    stages = [('matrix', matrix, 'eval-final', 'matrix-audit'),
              ('extension', extension, 'eval-extension-' + '-'.join(args.extension_ood),
               '-'.join(args.extension_ood) + '-matrix-audit')]
    try:
        for name, producer, evaluation_name, report_name in stages:
            save_status(control, state='waiting_for_' + name, controller_pid=os.getpid(),
                        seeds=args.seeds, completed_audits=completed)
            while not producer_ready(producer, args.seeds):
                if (control / 'STOP').exists():
                    raise InterruptedError('audit queue STOP requested')
                time.sleep(10)
            if (control / 'STOP').exists():
                raise InterruptedError('audit queue STOP requested')
            evaluations = [runs / f'{args.dataset}-seed{seed}' / evaluation_name for seed in args.seeds]
            summary = producer / 'summary' / 'summary.json'
            if not summary.is_file() or any(not (path / 'results.json').is_file() for path in evaluations):
                raise ValueError('completed producer has missing evaluation or summary files')
            report = reserve_report(reports / report_name)
            command = [args.analysis_python, '-u', 'tools/audit_matrix.py', '--evaluations',
                       *map(str, evaluations), '--summary', str(summary), '--expected-seeds',
                       *map(str, args.seeds), '--output-dir', str(report)]
            child(command, control, name + '-independent-audit', control)
            result = json.loads((report / 'matrix-audit.json').read_text(encoding='utf-8'))
            if result.get('status') != 'passed' or sorted(result['seeds']) != sorted(args.seeds):
                raise ValueError('independent audit did not pass for all expected seeds')
            completed.append({'stage': name, 'report': str(report),
                              'summary_sha256': result['summary_sha256'],
                              'statistics_checked': result['statistics_checked']})
            save_status(control, state='audit_complete', controller_pid=os.getpid(),
                        seeds=args.seeds, completed_audits=completed)
            print(f'{name} independent audit passed; {report}', flush=True)
        save_status(control, state='complete', controller_pid=os.getpid(),
                    seeds=args.seeds, completed_audits=completed)
    except InterruptedError as error:
        save_status(control, state='paused', controller_pid=os.getpid(),
                    reason=str(error), completed_audits=completed)
    except Exception as error:
        save_status(control, state='failed', controller_pid=os.getpid(),
                    error=repr(error), completed_audits=completed)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
