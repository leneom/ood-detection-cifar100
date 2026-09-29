"""CPU scoring of frozen saved features with audited training-only geometry fits."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import numpy as np
from geometry_io import (ROOT, RUN, CALIBRATION, METHODS, ADDITIONS, DATASETS, NATURAL, FOLDERS,
                         digest, read, write, load, save_arrays, require, now)
from geometry_core import geometry_scores
from selective_risk_core import threshold_for_target, id_decisions, ood_decisions
sys.path.insert(0, str(ROOT / 'src'))
from ood.metrics import detection_metrics

TARGETS = [.8, .9, .95, .99]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='evaluation')
    args = parser.parse_args()
    require(re.fullmatch(r'evaluation(?:-attempt-\d{3})?', args.output_name), 'invalid local attempt name')
    import msvcrt
    with (RUN / 'evaluation.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            require(read(RUN / 'fit-status.json')['state'] == 'complete', 'fits incomplete')
            audit = read(RUN / 'fit-audit.json')
            require(audit['state'] == 'passed', 'fits not independently audited')
            fit_inputs = read(RUN / 'fit-inputs.json')
            inputs = dict(fit_inputs['original_artifacts'])
            paths = [RUN / 'fit-inputs.json', RUN / 'fit-status.json', RUN / 'fit-audit.json']
            for seed in [0, 1, 2]:
                directory = RUN / f'seed{seed}'
                require(digest(directory / 'fit.json') == audit['per_seed'][seed]['fit_json_sha256'], 'audited fit changed')
                require(digest(directory / 'fit-state.npz') == audit['per_seed'][seed]['fit_state_sha256'], 'fitted state changed')
                paths.extend([directory / 'fit.json', directory / 'fit-state.npz'])
            inputs.update({str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths})
            for name, expected in inputs.items():
                require(digest(ROOT / name) == expected, 'original scoring input changed')
            files = [Path(__file__), ROOT / 'tools/geometry_io.py', ROOT / 'tools/geometry_core.py',
                     ROOT / 'tools/selective_risk_core.py', ROOT / 'src/ood/metrics.py']
            sources = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files}
            manifest_path = RUN / f'{args.output_name}-inputs.json'
            if manifest_path.exists():
                manifest = read(manifest_path)
                require(manifest['original_artifacts'] == inputs and manifest['source_hashes'] == sources, 'evaluation source/input mismatch')
            else:
                snapshot = RUN / f'{args.output_name}-source'
                snapshot.mkdir(exist_ok=True)
                for path in files:
                    shutil.copy2(path, snapshot / path.name)
                manifest = {'state': 'inputs_frozen_before_geometry_OOD_scoring', 'prepared_at': now(),
                            'original_artifacts': inputs, 'source_hashes': sources, 'output_name': args.output_name,
                            'methods': METHODS + ADDITIONS, 'new_methods': ADDITIONS[:3], 'numeric_control': 'affine_energy',
                            'primary_threshold_calibration_count': 2500, 'targets': TARGETS, 'numpy': np.__version__,
                            'boundary': 'saved features only; all geometry fits training ID only; OOD never selects settings'}
                write(manifest_path, manifest)
            write(RUN / 'status.json', {'state': 'running', 'stage': 'CPU geometry evaluation', 'pid': os.getpid(), 'started_at': now()})
            split = load(CALIBRATION / 'calibration_split.npz')
            threshold_rows = split['threshold_rows']
            for seed in [0, 1, 2]:
                directory = RUN / f'seed{seed}'
                output = directory / args.output_name
                if (output / 'results.json').exists():
                    result = read(output / 'results.json')
                    require(result['inputs_sha256'] == digest(manifest_path), 'completed geometry evaluation inputs changed')
                    for name, expected in result['output_file_sha256'].items():
                        require(digest(output / name) == expected, 'completed geometry output changed')
                    continue
                require(not output.exists() or not any(output.iterdir()), 'partial evaluation retained; use a fresh --output-name attempt')
                output.mkdir(exist_ok=True)
                metadata, state = read(directory / 'fit.json'), load(directory / 'fit-state.npz')
                with np.load(directory / 'training-bank.npz', allow_pickle=False) as bank:
                    weight, bias = bank['weight'], bank['bias']
                main = ROOT / f'runs/cifar100-seed{seed}/eval-final'
                id_data, val_data = load(main / 'id_logits.npz'), load(main / 'validation_logits.npz')
                features = load(main / 'features.npz')
                population_scores, diagnostics = {}, {}
                def score_population(name, feature, original_logits, labels=None):
                    score, middle = geometry_scores(feature, weight, bias, metadata, state)
                    save_arrays(output / f'{name}_intermediates.npz', **middle)
                    require(np.allclose(middle['affine_logits'], original_logits, atol=2e-5, rtol=2e-6),
                            name + ': affine reconstruction exceeds frozen tolerance')
                    diagnostics[name] = {'n': len(feature),
                                         'max_affine_logit_error': float(np.max(np.abs(middle['affine_logits'] - original_logits))),
                                         'affine_argmax_changes': int((middle['affine_logits'].argmax(1) != original_logits.argmax(1)).sum()),
                                         'react_argmax_changes': int((middle['react_logits'].argmax(1) != original_logits.argmax(1)).sum()),
                                         'clipped_coordinate_fraction': float(middle['clipped_coordinate_count'].sum() / feature.size)}
                    if labels is not None:
                        diagnostics[name].update({'original_accuracy': float((original_logits.argmax(1) == labels).mean()),
                                                   'react_intermediate_accuracy': float((middle['react_logits'].argmax(1) == labels).mean())})
                    return score
                population_scores['id'] = score_population('id', features['id'], id_data['logits'], id_data['labels'])
                population_scores['val'] = score_population('validation', features['validation'], val_data['logits'], split['validation_labels'])
                for dataset in DATASETS:
                    source = ROOT / f'runs/cifar100-seed{seed}' / FOLDERS[dataset]
                    population_scores[dataset] = score_population(dataset, load(source / f'{dataset}_features.npz')['features'],
                                                                  load(source / f'{dataset}_logits.npz')['logits'])
                correct = id_data['logits'].argmax(1) == id_data['labels']
                masks = {'id_correct': correct, 'threshold_rows': threshold_rows,
                         'id_accepted': np.empty((10, 4, len(correct)), dtype=bool),
                         'calibration_accepted': np.empty((10, 4, 2500), dtype=bool)}
                for dataset in DATASETS:
                    masks[f'{dataset}_accepted'] = np.empty((10, 4, len(population_scores[dataset]['vim'])), dtype=bool)
                datasets, grid = {}, []
                raw_by_dataset = {}
                for dataset in DATASETS:
                    baseline = load(CALIBRATION / f'seed{seed}/{dataset}_scores.npz')
                    source_result = read(CALIBRATION / f'seed{seed}/results.json')['datasets'][dataset]
                    saved = {}
                    metrics = {}
                    for method in METHODS + ADDITIONS:
                        inside = baseline[f'T1_{method}_id'] if method in METHODS else population_scores['id'][method]
                        outside = baseline[f'T1_{method}_ood'] if method in METHODS else population_scores[dataset][method]
                        validation = baseline[f'T1_{method}_val'] if method in METHODS else population_scores['val'][method]
                        selected = validation[threshold_rows]
                        primary = detection_metrics(inside, outside, selected)
                        primary.update({'benchmark_ID_equal_threshold_count': int((inside == primary['benchmark_threshold']).sum()),
                                        'threshold_calibration_equal_threshold_count': int((selected == primary['validation_threshold']).sum()),
                                        'threshold_calibration_n': 2500})
                        if method in METHODS:
                            old = source_result['variants']['T1'][method]
                            for key, value in primary.items():
                                require(value == old[key], 'phase-2 matched T1 control metric changed: ' + key)
                        full = detection_metrics(inside, outside, validation)
                        metrics[method] = {'primary_2500': primary,
                                           'secondary_5000_threshold': {k: full[k] for k in ['validation_threshold', 'calibration_id_acceptance',
                                               'id_test_acceptance_at_validation_threshold', 'ood_acceptance_at_validation_threshold']}}
                        for part, array in [('id', inside), ('ood', outside), ('val', validation), ('threshold', selected)]:
                            saved[f'{method}_{part}'] = array
                    save_arrays(output / f'{dataset}_scores.npz', **saved)
                    raw_by_dataset[dataset] = saved
                    datasets[dataset] = {'kind': source_result['kind'], 'diagnostic_only': dataset == 'noise', 'methods': metrics}
                canonical = raw_by_dataset['cifar10']
                for mi, method in enumerate(METHODS + ADDITIONS):
                    for ti, target in enumerate(TARGETS):
                        threshold, cal, cal_mask = threshold_for_target(canonical[f'{method}_threshold'], target)
                        inside, id_mask = id_decisions(canonical[f'{method}_id'], correct, threshold)
                        masks['id_accepted'][mi, ti] = id_mask
                        masks['calibration_accepted'][mi, ti] = cal_mask
                        outside = {}
                        for dataset in DATASETS:
                            outside[dataset], mask = ood_decisions(raw_by_dataset[dataset][f'{method}_ood'], threshold)
                            masks[f'{dataset}_accepted'][mi, ti] = mask
                        grid.append({'method': method, 'target': target, 'threshold': threshold, 'calibration': cal, 'id_test': inside,
                                     'ood': outside, 'equal_weight_natural_ood_acceptance': float(np.mean([outside[d]['acceptance'] for d in NATURAL]))})
                save_arrays(output / 'decisions.npz', **masks)
                result = {'state': 'complete', 'seed': seed, 'evaluated_at': now(), 'inputs_sha256': digest(manifest_path),
                          'fit_json_sha256': digest(directory / 'fit.json'), 'checkpoint_sha256': metadata['checkpoint_sha256'],
                          'methods': METHODS + ADDITIONS, 'named_additions': ADDITIONS[:3], 'numeric_control': 'affine_energy',
                          'score_direction': 'higher_is_ood', 'temperature': 1., 'primary_calibration_count': 2500,
                          'datasets': datasets, 'operating_points': grid, 'grid_targets': TARGETS,
                          'intermediate_diagnostics': diagnostics,
                          'original_ID_classifier_accuracy': float(correct.mean()),
                          'all_six_T1_controls_preserved': True,
                          'output_file_sha256': {p.name: digest(p) for p in output.glob('*.npz')},
                          'boundary': 'exploratory training-ID-only geometry extension; no OOD-selected settings; ReAct logits scoring intermediates only'}
                write(output / 'results.json', result)
                print(__import__('json').dumps({'seed': seed, 'state': 'scores_saved', 'methods': len(result['methods']),
                                              'natural_datasets': 4, 'synthetic_diagnostics': 1, 'controls_preserved': True}), flush=True)
                write(RUN / 'status.json', {'state': 'running', 'stage': 'CPU geometry evaluation', 'pid': os.getpid(),
                                           'completed_evaluation_seeds': list(range(seed + 1)), 'updated_at': now()})
            for name, expected in inputs.items():
                require(digest(ROOT / name) == expected, 'original scoring input changed during evaluation')
            for name, expected in sources.items():
                require(digest(ROOT / name) == expected, 'scoring source changed during evaluation')
            write(RUN / 'status.json', {'state': 'evaluation_complete_pending_independent_audit', 'pid': os.getpid(),
                                       'completed_evaluation_seeds': [0, 1, 2], 'output_name': args.output_name,
                                       'original_artifacts_unchanged': True, 'completed_at': now()})
        except Exception as error:
            write(RUN / 'status.json', {'state': 'failed', 'stage': 'CPU geometry evaluation', 'pid': os.getpid(), 'error': repr(error), 'updated_at': now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
