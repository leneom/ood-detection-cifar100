"""CPU-only execution of the predeclared fixed ID-calibrated operating-point grid."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import numpy as np
from selective_risk_core import threshold_for_target, id_decisions, ood_decisions

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/selective-risk-phase2'
CALIBRATION = ROOT / 'runs/calibration-phase2'
REPORT = ROOT / 'report/calibration-phase2'
METHODS = ['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis']
VARIANTS = ['T1', 'calibrated']
TARGETS = [.8, .9, .95, .99]
NATURAL = ['cifar10', 'mnist', 'svhn', 'textures']
DATASETS = NATURAL + ['noise']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    pre = read(RUN / 'predeclaration.json')
    require(pre['expected_seeds'] == [0, 1, 2] and pre['ID_acceptance_targets'] == TARGETS, 'predeclared settings differ')
    for path, expected in [(ROOT / pre['protocol'], pre['protocol_sha256']),
                           (REPORT / 'summary.json', pre['calibration_summary_sha256']),
                           (REPORT / 'audit.json', pre['calibration_raw_audit_sha256']),
                           (REPORT / 'summary-audit.json', pre['calibration_summary_audit_sha256'])]:
        require(digest(path) == expected, 'predeclared protocol/source changed')
    require(read(CALIBRATION / 'status.json')['state'] == 'complete', 'calibration producer incomplete')
    audit = read(REPORT / 'audit.json')
    require(audit['state'] == 'passed' and read(REPORT / 'summary-audit.json')['state'] == 'passed', 'calibration unaudited')
    require(digest(CALIBRATION / 'inputs.json') == audit['inputs_sha256'], 'calibration inputs changed')
    previous = read(CALIBRATION / 'inputs.json')
    paths = {ROOT / name for name in previous['original_artifacts']}
    for name, expected in previous['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'original baseline artifact changed: ' + name)
    paths.update([RUN / 'predeclaration.json', ROOT / pre['protocol'], CALIBRATION / 'inputs.json',
                  CALIBRATION / 'status.json', CALIBRATION / 'calibration_split.npz', REPORT / 'summary.json',
                  REPORT / 'audit.json', REPORT / 'summary-audit.json', REPORT / 'completion-audit.json'])
    for seed in [0, 1, 2]:
        directory = CALIBRATION / f'seed{seed}'
        result = read(directory / 'results.json')
        require(digest(directory / 'results.json') == audit['results_sha256'][f'seed{seed}'], 'audited result changed')
        require(result['state'] == 'complete' and result['seed'] == seed, 'calibration seed incomplete')
        require(digest(directory / 'temperature_fit.json') == result['temperature_fit_json_sha256'], 'temperature record changed')
        require(result['checkpoint_sha256'] == read(ROOT / f'runs/cifar100-seed{seed}/eval-final/results.json')['checkpoint_sha256'], 'frozen checkpoint changed')
        paths.update([directory / 'results.json', directory / 'temperature_fit.json'])
        for name, expected in result['output_array_sha256'].items():
            require(digest(directory / name) == expected, 'calibrated raw score changed')
            paths.add(directory / name)
    provenance = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in sorted(paths)}
    sources = {p.name: digest(p) for p in [Path(__file__), ROOT / 'tools/selective_risk_core.py']}
    path = RUN / 'inputs.json'
    if path.exists():
        inputs = read(path)
        require(inputs['original_artifacts'] == provenance and inputs['source_hashes'] == sources, 'refuse changed source/input resume')
        return inputs
    require(not any(RUN.glob('seed*/results.json')), 'operating grid exists before frozen inputs')
    snapshot = RUN / 'analysis_source'
    snapshot.mkdir(exist_ok=True)
    for name in sources:
        shutil.copy2(ROOT / 'tools' / name, snapshot / name)
    inputs = {'state': 'inputs_frozen_before_grid_computation', 'prepared_at': now(),
              'original_artifacts': provenance, 'source_hashes': sources,
              'seeds': [0, 1, 2], 'methods': METHODS, 'variants': VARIANTS, 'targets': TARGETS,
              'natural_datasets': NATURAL, 'synthetic_diagnostics': ['noise'],
              'threshold_population': 'same 2500 ID-validation threshold rows, disjoint from temperature fitting',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'CPU'},
              'boundary': 'saved scores only; no training/temperature/feature fitting; validation reused for checkpoint selection'}
    write(path, inputs)
    return inputs


def run_seed(seed):
    destination = RUN / f'seed{seed}'
    result_path = destination / 'results.json'
    if result_path.exists():
        result = read(result_path)
        require(result['inputs_sha256'] == digest(RUN / 'inputs.json'), 'completed grid seed inputs changed')
        require(digest(destination / 'decisions.npz') == result['decisions_sha256'], 'completed decisions changed')
        return result
    destination.mkdir(exist_ok=True)
    previous = read(CALIBRATION / f'seed{seed}/results.json')
    raw = {dataset: load(CALIBRATION / f'seed{seed}/{dataset}_scores.npz') for dataset in DATASETS}
    original = load(ROOT / f'runs/cifar100-seed{seed}/eval-final/id_logits.npz')
    predicted = original['logits'].argmax(axis=1)
    correct = predicted == original['labels']
    split = load(CALIBRATION / 'calibration_split.npz')
    threshold_rows = split['threshold_rows']
    require(len(correct) == 10000 and len(threshold_rows) == 2500, 'ID populations changed')
    canonical = raw['cifar10']
    id_masks = np.empty((2, 6, 4, len(correct)), dtype=np.bool_)
    calibration_masks = np.empty((2, 6, 4, len(threshold_rows)), dtype=np.bool_)
    ood_masks = {d: np.empty((2, 6, 4, previous['datasets'][d]['variants']['T1']['msp']['n_ood']), dtype=np.bool_) for d in DATASETS}
    records = []
    for vi, variant in enumerate(VARIANTS):
        for mi, method in enumerate(METHODS):
            inside = canonical[f'{variant}_{method}_id']
            full_validation = canonical[f'{variant}_{method}_val']
            calibration = canonical[f'{variant}_{method}_threshold']
            require(np.array_equal(calibration, full_validation[threshold_rows]), 'wrong threshold-calibration rows')
            for dataset in DATASETS:
                for part in ['id', 'val', 'threshold']:
                    require(np.array_equal(raw[dataset][f'{variant}_{method}_{part}'], canonical[f'{variant}_{method}_{part}']),
                            'datasets do not share identical canonical ID/calibration scores')
            for ti, target in enumerate(TARGETS):
                threshold, calibration_result, calibration_mask = threshold_for_target(calibration, target)
                id_result, id_mask = id_decisions(inside, correct, threshold)
                id_masks[vi, mi, ti] = id_mask
                calibration_masks[vi, mi, ti] = calibration_mask
                ood = {}
                for dataset in DATASETS:
                    score = raw[dataset][f'{variant}_{method}_ood']
                    ood_result, ood_mask = ood_decisions(score, threshold)
                    ood_masks[dataset][vi, mi, ti] = ood_mask
                    ood_result.update({'diagnostic_only': dataset == 'noise',
                                       'kind': previous['datasets'][dataset]['kind'],
                                       'source_score_file_sha256': digest(CALIBRATION / f'seed{seed}/{dataset}_scores.npz'),
                                       'score_array_sha256': hashlib.sha256(score.tobytes()).hexdigest(),
                                       'score_dtype': str(score.dtype),
                                       'id_ood_population_prevalence': len(score) / (len(inside) + len(score))})
                    ood[dataset] = ood_result
                    if target == .95:
                        saved = previous['datasets'][dataset]['variants'][variant][method]
                        require(threshold == saved['validation_threshold'], '95% threshold differs from audited phase 2')
                        require(id_result['coverage'] == saved['id_test_acceptance_at_validation_threshold'], '95% ID coverage differs')
                        require(ood_result['acceptance'] == saved['ood_acceptance_at_validation_threshold'], '95% OOD acceptance differs')
                        require(calibration_result['achieved_acceptance'] == saved['calibration_id_acceptance'], '95% calibration acceptance differs')
                records.append({'method': method, 'variant': variant, 'target': target, 'threshold': threshold,
                                'calibration': calibration_result, 'id_test': id_result, 'ood': ood,
                                'equal_weight_natural_ood_acceptance': float(np.mean([ood[d]['acceptance'] for d in NATURAL]))})
    for mi, method in enumerate(METHODS):
        if method in ('maxlogit', 'knn', 'mahalanobis'):
            require(np.array_equal(id_masks[0, mi], id_masks[1, mi]), 'unchanged-logit/feature ID decisions changed')
            require(np.array_equal(calibration_masks[0, mi], calibration_masks[1, mi]), 'control thresholds changed')
            for masks in ood_masks.values():
                require(np.array_equal(masks[0, mi], masks[1, mi]), 'feature OOD decisions changed')
            controls = [r for r in records if r['method'] == method]
            left = [{k: v for k, v in r.items() if k != 'variant'} for r in controls if r['variant'] == 'T1']
            right = [{k: v for k, v in r.items() if k != 'variant'} for r in controls if r['variant'] == 'calibrated']
            # Source file hash/dtype are identical for the two score arrays in the same archive.
            for a, b in zip(left, right):
                require(a == b, 'control thresholds/counts changed')
    np.savez_compressed(destination / 'decisions.npz', id_accepted=id_masks, calibration_accepted=calibration_masks,
                        id_correct=correct, id_labels=original['labels'], id_predictions=predicted,
                        threshold_rows=threshold_rows, **{f'{d}_accepted': masks for d, masks in ood_masks.items()})
    result = {'state': 'complete', 'seed': seed, 'computed_at': now(),
              'inputs_sha256': digest(RUN / 'inputs.json'), 'checkpoint_sha256': previous['checkpoint_sha256'],
              'calibrated_temperature': previous['temperature_fit']['temperature'],
              'decision_axes': ['variant', 'method', 'target', 'sample'], 'variants': VARIANTS, 'methods': METHODS, 'targets': TARGETS,
              'n_operating_points': len(records), 'operating_points': records,
              'decisions_sha256': digest(destination / 'decisions.npz'),
              'id_correct_flags_sha256': hashlib.sha256(correct.tobytes()).hexdigest(),
              'all_95pct_points_match_calibration_phase2': True,
              'boundary': 'ID-only thresholds; test ID error risk and OOD acceptance separate; no test-selected settings'}
    write(result_path, result)
    print(json.dumps({'seed': seed, 'operating_points': len(records), '95pct_phase2_parity': True}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import msvcrt
    with (RUN / 'controller.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            inputs = prepare()
            if args.prepare_only:
                print(json.dumps({'state': inputs['state'], 'original_artifacts': len(inputs['original_artifacts'])}), flush=True)
                return
            write(RUN / 'status.json', {'state': 'running', 'pid': os.getpid(), 'started_at': now()})
            for seed in [0, 1, 2]:
                run_seed(seed)
                write(RUN / 'status.json', {'state': 'running', 'pid': os.getpid(), 'completed_seeds': list(range(seed + 1)), 'updated_at': now()})
            for name, expected in inputs['original_artifacts'].items():
                require(digest(ROOT / name) == expected, 'original artifact changed during grid computation')
            write(RUN / 'status.json', {'state': 'complete', 'pid': os.getpid(), 'completed_seeds': [0, 1, 2],
                                      'operating_points': 144, 'original_artifacts_unchanged': True, 'completed_at': now()})
        except Exception as error:
            write(RUN / 'status.json', {'state': 'failed', 'pid': os.getpid(), 'error': repr(error), 'updated_at': now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
