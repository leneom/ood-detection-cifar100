"""Execute predeclared ID-only phase 2 on frozen local artifacts, CPU only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import sys
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ood.metrics import detection_metrics
from calibration_core import stratified_calibration_rows, fit_temperature, logit_scores, calibration_metrics

METHODS = ('msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis')
NATURAL = ('cifar10', 'mnist', 'svhn', 'textures')
DATASETS = NATURAL + ('noise',)
DIRECTORIES = {'cifar10': 'eval-final', 'mnist': 'eval-final', 'noise': 'eval-final',
               'svhn': 'eval-extension-svhn', 'textures': 'eval-extension-textures'}
OUTPUT = ROOT / 'runs/calibration-phase2'


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def phase_inputs():
    """Bind original arrays, producer snapshots, selections and independent audits."""
    prereg = read(OUTPUT / 'predeclaration.json')
    overview_path = ROOT / 'report/baseline-complete/overview.json'
    require(digest(ROOT / prereg['protocol']) == prereg['protocol_sha256'], 'phase-2 protocol changed')
    require(digest(overview_path) == prereg['baseline_overview_sha256'], 'completed baseline changed')
    overview = read(overview_path)
    complete_path = ROOT / 'report/baseline-complete/completion-audit.json'
    complete = read(complete_path)
    require(complete['state'] == 'passed' and complete['overview_sha256'] == digest(overview_path), 'baseline incomplete')
    require(overview['seeds'] == [0, 1, 2] and overview['natural_datasets'] == list(NATURAL), 'population changed')
    pixel_path = ROOT / 'report/textures-matrix-audit/input-pixel-audit.json'
    require(digest(pixel_path) == overview['executed_input_audit_sha256'], 'executed input audit changed')
    pixel_audit = read(pixel_path)
    require(pixel_audit['state'] == 'passed' and pixel_audit['population_pixel_hash_checks'] == 12, 'input audit incomplete')
    paths = {ROOT / 'docs/PROTOCOL.md', ROOT / prereg['protocol'], OUTPUT / 'predeclaration.json',
             overview_path, complete_path, pixel_path}
    for phase in overview['phases']:
        audit_name = 'matrix' if phase['phase'] == 'matrix' else phase['phase'].replace('-extension', '-matrix')
        audit_path = ROOT / f'report/{audit_name}-audit/matrix-audit.json'
        summary_path = ROOT / f"runs/{phase['phase']}/summary/summary.json"
        require(digest(audit_path) == phase['audit_sha256'], 'baseline audit hash mismatch')
        require(digest(summary_path) == phase['summary_sha256'], 'baseline summary hash mismatch')
        require(read(audit_path)['status'] == 'passed', 'baseline phase failed audit')
        paths.update([audit_path, summary_path])
        for reference in phase['results']:
            result_path = Path(reference['path'])
            require(digest(result_path) == reference['results_sha256'], 'baseline result hash mismatch')
            result = read(result_path)
            directory = result_path.parent
            run = directory.parent
            seed = reference['seed']
            require(result['training_config']['seed'] == seed, 'seed mismatch')
            require(result['checkpoint_selection'] == 'ID validation accuracy only', 'checkpoint selected using OOD')
            require(result['temperature'] == 1. and result['score_direction'] == 'higher_is_ood', 'original scoring changed')
            require(digest(result['checkpoint']) == result['checkpoint_sha256'], 'checkpoint changed')
            baseline = read(run / 'eval-final/results.json')
            for key in ['checkpoint_sha256', 'checkpoint_epoch', 'training_config', 'feature_fit']:
                require(result[key] == baseline[key], 'cross-phase classifier/fit changed')
            history = read(run / 'history.json')
            status = read(run / 'status.json')
            require(status['state'] == 'complete' and status['epoch'] == 100 and len(history) == 100,
                    'training not complete')
            require(max(history, key=lambda row: row['val_accuracy'])['epoch'] == result['checkpoint_epoch'],
                    'not first best ID validation checkpoint')
            for name, expected in result['evaluation_source_hashes'].items():
                path = directory / 'evaluation_source' / name
                require(digest(path) == expected, 'evaluation source changed')
                paths.add(path)
            for name, expected in read(run / 'source_hashes.json').items():
                path = run / 'source_snapshot' / name
                require(digest(path) == expected and result['evaluation_source_hashes'][name] == expected,
                        'training snapshot identity changed')
                paths.add(path)
            paths.update(directory.glob('*.npz'))
            paths.update([result_path, Path(result['checkpoint']), run / 'split_indices.npz', run / 'split.json',
                          run / 'history.json', run / 'status.json', run / 'source_hashes.json'])
            if phase['phase'] == 'textures-extension':
                audited = next(row for row in pixel_audit['per_seed'] if row['seed'] == seed)
                require(audited['results_sha256'] == digest(result_path), 'pixel audit/result mismatch')
                for population in ['id_train', 'id_validation', 'id_test']:
                    require(result['input_pixels'][population] == audited['populations'][population], 'ID pixels changed')
                require(result['input_pixels']['ood']['textures'] == audited['populations']['textures'], 'OOD pixels changed')
    # Check original indices and recover official labels before any fitting.
    recovered = {}
    for name, expected in [('train', '16019d7e3df5f24257cddd939b257f8d'),
                           ('test', 'f0ef6b0ae62326f3e7ffdfab6717acfc')]:
        path = ROOT / 'data/cifar-100-python' / name
        require(digest(path, 'md5') == expected, 'official CIFAR label source failed MD5')
        with path.open('rb') as stream:
            recovered[name] = pickle.load(stream, encoding='latin1')
        paths.add(path)
    val_indices = arrays(ROOT / 'runs/cifar100-seed0/split_indices.npz')['val']
    require(val_indices.shape == (5000,), 'original validation size mismatch')
    for seed in [0, 1, 2]:
        run = ROOT / f'runs/cifar100-seed{seed}'
        split = arrays(run / 'split_indices.npz')
        require(np.array_equal(split['val'], val_indices), 'validation order differs between seeds')
        require(len(np.unique(np.r_[split['train'], split['val']])) == 50000, 'original split overlapping/incomplete')
        record = read(run / 'split.json')
        for key in ('train', 'val'):
            require(hashlib.sha256(split[key].astype(np.int64).tobytes()).hexdigest() == record[f'{key}_indices_sha256'],
                    'original split hash mismatch')
        main = run / 'eval-final'
        canonical_id = arrays(main / 'id_logits.npz')
        canonical_val = arrays(main / 'validation_logits.npz')['logits']
        require(canonical_id['logits'].shape == (10000, 100) and canonical_val.shape == (5000, 100), 'ID logit shape mismatch')
        require(np.array_equal(canonical_id['labels'], recovered['test']['fine_labels']), 'official ID test label mismatch')
        canonical_scores = arrays(main / 'cifar10_scores.npz')
        for directory in [main, run / 'eval-extension-svhn', run / 'eval-extension-textures']:
            other = arrays(directory / 'id_logits.npz')
            require(np.array_equal(other['labels'], canonical_id['labels']), 'extension ID labels differ')
            require(np.allclose(other['logits'], canonical_id['logits'], rtol=1e-6, atol=1e-6), 'extension ID logits differ')
            require(np.allclose(arrays(directory / 'validation_logits.npz')['logits'], canonical_val, rtol=1e-6, atol=1e-6),
                    'extension validation logits differ')
            for dataset in read(directory / 'results.json')['datasets']:
                saved = arrays(directory / f'{dataset}_scores.npz')
                for method in METHODS:
                    for part in ('id', 'val'):
                        key = f'{method}_{part}'
                        tolerance = 1e-4 if method == 'mahalanobis' else 1e-6
                        require(np.allclose(saved[key], canonical_scores[key], rtol=tolerance, atol=1e-6),
                                'original cross-phase ID scores differ beyond frozen extension tolerance')
    labels = np.asarray(recovered['train']['fine_labels'], dtype=np.int64)[val_indices]
    images = recovered['train']['data'].reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)[val_indices].copy()
    require(hashlib.sha256(images.tobytes()).hexdigest() == pixel_audit['per_seed'][0]['populations']['id_validation']['sha256'],
            'recovered validation order/input hash mismatch')
    return {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in sorted(paths)}, labels, val_indices


def prepare():
    provenance, labels, val_indices = phase_inputs()
    code_paths = [Path(__file__), ROOT / 'tools/calibration_core.py', ROOT / 'src/ood/metrics.py']
    code_hashes = {path.name: digest(path) for path in code_paths}
    manifest_path = OUTPUT / 'inputs.json'
    split_path = OUTPUT / 'calibration_split.npz'
    fit_rows, threshold_rows = stratified_calibration_rows(labels)
    if manifest_path.exists():
        saved = read(manifest_path)
        require(saved['original_artifacts'] == provenance and saved['analysis_source_hashes'] == code_hashes,
                'refuse resume with changed artifacts/analysis source')
        require(digest(split_path) == saved['split_sha256'], 'saved calibration split changed')
        split = arrays(split_path)
        for name, expected in [('fit_rows', fit_rows), ('threshold_rows', threshold_rows),
                               ('validation_labels', labels), ('original_validation_indices', val_indices)]:
            require(np.array_equal(split[name], expected), 'saved split/labels differ from predeclared partition')
        return saved, split
    require(not any(OUTPUT.glob('seed*/results.json')), 'fitting artifacts exist before input manifest')
    require(not split_path.exists(), 'orphaned calibration split; inspect rather than replace')
    np.savez_compressed(split_path, fit_rows=fit_rows, threshold_rows=threshold_rows,
                        validation_labels=labels, original_validation_indices=val_indices)
    snapshot = OUTPUT / 'analysis_source'
    snapshot.mkdir(exist_ok=True)
    for path in code_paths:
        shutil.copy2(path, snapshot / path.name)
    manifest = {'state': 'inputs_frozen_before_temperature_fitting', 'prepared_at': now(),
                'original_artifacts': provenance, 'analysis_source_hashes': code_hashes,
                'split_sha256': digest(split_path), 'split_seed': 2027,
                'fit_count': 2500, 'threshold_count': 2500, 'each_subset_counts_per_class': [25] * 100,
                'validation_labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
                'original_validation_indices_sha256': hashlib.sha256(val_indices.tobytes()).hexdigest(),
                'threshold_population': '2500 disjoint ID-validation rows',
                'selection_reuse_boundary': 'both subsets previously used for checkpoint selection; not fresh holdouts',
                'canonical_ID_arrays': 'eval-final id/validation logits and cifar10 feature ID/validation scores for every OOD',
                'cross_phase_ID_comparison': 'logits/classical/KNN rtol=atol=1e-6; Mahalanobis rtol=1e-4, atol=1e-6',
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                                'device': 'CPU', 'classical_score_precision': 'float64', 'feature_scores': 'original dtype unchanged'}}
    write(manifest_path, manifest)
    return manifest, arrays(split_path)


def run_seed(seed, split):
    destination = OUTPUT / f'seed{seed}'
    existing = destination / 'results.json'
    if existing.exists():
        result = read(existing)
        require(result['inputs_sha256'] == digest(OUTPUT / 'inputs.json'), 'completed seed input changed')
        for name, expected in result['output_array_sha256'].items():
            require(digest(destination / name) == expected, 'completed calibration output changed')
        return result
    destination.mkdir(exist_ok=True)
    main = ROOT / f'runs/cifar100-seed{seed}/eval-final'
    id_data = arrays(main / 'id_logits.npz')
    val_logits = arrays(main / 'validation_logits.npz')['logits']
    y, fit_rows, threshold_rows = split['validation_labels'], split['fit_rows'], split['threshold_rows']
    # This is the only optimizer call; it receives exactly the ID fitting rows.
    fit = fit_temperature(val_logits[fit_rows], y[fit_rows])
    fit.update({'started_after_inputs_preparation': True, 'fitted_at': now(),
                'fit_rows_sha256': hashlib.sha256(fit_rows.tobytes()).hexdigest()})
    write(destination / 'temperature_fit.json', fit)
    temperature = fit['temperature']
    variants = {'T1': 1., 'calibrated': temperature}
    probabilities = {variant: calibration_metrics(id_data['logits'], id_data['labels'], value)
                     for variant, value in variants.items()}
    require(probabilities['T1']['accuracy'] == probabilities['calibrated']['accuracy'], 'positive T changed accuracy')
    baseline_result = read(main / 'results.json')
    require(abs(probabilities['T1']['accuracy'] - baseline_result['id_test_accuracy']) < 1e-7, 'accuracy differs from baseline')
    calibration_populations = {part: {variant: calibration_metrics(val_logits[rows], y[rows], value)
                                    for variant, value in variants.items()}
                              for part, rows in [('fit', fit_rows), ('threshold', threshold_rows)]}
    canonical_feature = arrays(main / 'cifar10_scores.npz')
    shared = {variant: {'id': logit_scores(id_data['logits'], value), 'val': logit_scores(val_logits, value)}
              for variant, value in variants.items()}
    for variant in variants:
        for method in ('knn', 'mahalanobis'):
            for part in ('id', 'val'):
                shared[variant][part][method] = canonical_feature[f'{method}_{part}']
    datasets, precision, output_hashes = {}, {}, {}
    for dataset in DATASETS:
        directory = ROOT / f'runs/cifar100-seed{seed}' / DIRECTORIES[dataset]
        original = read(directory / 'results.json')['datasets'][dataset]
        old_scores = arrays(directory / f'{dataset}_scores.npz')
        outside_logits = arrays(directory / f'{dataset}_logits.npz')['logits']
        outside = {variant: logit_scores(outside_logits, value) for variant, value in variants.items()}
        for variant in variants:
            for method in ('knn', 'mahalanobis'):
                outside[variant][method] = old_scores[f'{method}_ood']
        payload, dataset_results, differences = {}, {}, {}
        for variant in variants:
            dataset_results[variant] = {}
            for method in METHODS:
                inside = shared[variant]['id'][method]
                val = shared[variant]['val'][method]
                ood = outside[variant][method]
                calibrated = val[threshold_rows]
                metric = detection_metrics(inside, ood, calibrated)
                benchmark, threshold = metric['benchmark_threshold'], metric['validation_threshold']
                metric.update({'benchmark_ID_equal_threshold_count': int((inside == benchmark).sum()),
                               'threshold_calibration_equal_threshold_count': int((calibrated == threshold).sum()),
                               'threshold_calibration_n': len(calibrated),
                               'threshold_population': '2500 disjoint threshold-calibration ID-validation rows'})
                dataset_results[variant][method] = metric
                for part, score in [('id', inside), ('ood', ood), ('val', val), ('threshold', calibrated)]:
                    payload[f'{variant}_{method}_{part}'] = score
        for method in METHODS:
            differences[method] = {'max_abs_score_difference': {}, 'metric_delta_T1_minus_original': {},
                                   'original_5000_validation_threshold_metrics': original['methods'][method]}
            for part in ('id', 'ood', 'val'):
                differences[method]['max_abs_score_difference'][part] = float(np.max(np.abs(
                    payload[f'T1_{method}_{part}'].astype(np.float64) - old_scores[f'{method}_{part}'])))
            # Original threshold metrics use 5000 rows; compare numerical scores/metrics separately on that population.
            full_val_control = detection_metrics(payload[f'T1_{method}_id'], payload[f'T1_{method}_ood'],
                                                payload[f'T1_{method}_val'])
            differences[method]['T1_float64_or_canonical_ID_full_5000_validation_metrics'] = full_val_control
            for key, value in original['methods'][method].items():
                if key in full_val_control:
                    differences[method]['metric_delta_T1_minus_original'][key] = full_val_control[key] - value
        for method in ('maxlogit', 'knn', 'mahalanobis'):
            for part in ('id', 'ood', 'val', 'threshold'):
                require(np.array_equal(payload[f'T1_{method}_{part}'], payload[f'calibrated_{method}_{part}']),
                        'unchanged-logit/feature control changed')
            require(dataset_results['T1'][method] == dataset_results['calibrated'][method], 'control thresholds changed')
        filename = f'{dataset}_scores.npz'
        np.savez_compressed(destination / filename, **payload)
        output_hashes[filename] = digest(destination / filename)
        datasets[dataset] = {'kind': original['kind'], 'diagnostic_only': dataset == 'noise',
                             'variants': dataset_results}
        precision[dataset] = differences
    result = {'state': 'complete', 'seed': seed, 'completed_at': now(), 'inputs_sha256': digest(OUTPUT / 'inputs.json'),
              'checkpoint_sha256': baseline_result['checkpoint_sha256'], 'checkpoint_epoch': baseline_result['checkpoint_epoch'],
              'temperature_fit': fit, 'id_test': probabilities, 'ID_validation_subsets': calibration_populations,
              'datasets': datasets, 'precision_and_original_thresholds': precision,
              'output_array_sha256': output_hashes, 'temperature_fit_json_sha256': digest(destination / 'temperature_fit.json'),
              'unchanged_controls': ['maxlogit', 'knn', 'mahalanobis'],
              'boundary': 'ID-only fit; reused selection validation; OOD secondary; float64 matched classical scores; no feature refitting'}
    write(existing, result)
    print(json.dumps({'seed': seed, 'temperature': temperature, 'id_test_T1': {k: probabilities['T1'][k] for k in ['nll', 'brier', 'ece15']},
                      'id_test_calibrated': {k: probabilities['calibrated'][k] for k in ['nll', 'brier', 'ece15']}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    # OS lock released on process exit, including failures; never duplicate this CPU controller.
    import msvcrt
    with (OUTPUT / 'controller.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            manifest, split = prepare()
            if args.prepare_only:
                print(json.dumps({'state': manifest['state'], 'split_sha256': manifest['split_sha256'],
                                  'original_artifact_count': len(manifest['original_artifacts'])}), flush=True)
                return
            write(OUTPUT / 'status.json', {'state': 'running', 'pid': os.getpid(), 'started_at': now(), 'seeds': [0, 1, 2]})
            for seed in [0, 1, 2]:
                run_seed(seed, split)
                write(OUTPUT / 'status.json', {'state': 'running', 'pid': os.getpid(), 'completed_seeds': list(range(seed + 1)), 'updated_at': now()})
            # Reverify every original artifact at completion, including protocol/checkpoints/source/raw arrays.
            for name, expected in manifest['original_artifacts'].items():
                require(digest(ROOT / name) == expected, 'original artifact changed during phase 2')
            write(OUTPUT / 'status.json', {'state': 'complete', 'pid': os.getpid(), 'completed_seeds': [0, 1, 2],
                                          'original_artifacts_unchanged': True, 'completed_at': now()})
        except Exception as error:
            write(OUTPUT / 'status.json', {'state': 'failed', 'pid': os.getpid(), 'error': repr(error), 'updated_at': now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
