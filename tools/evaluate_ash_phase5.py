"""CPU ASH scoring after independent training-ID-only numerical preflight."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import numpy as np
from geometry_io import (ROOT, RUN, CALIBRATION, METHODS, ADDITIONS, DATASETS, NATURAL, FOLDERS,
                         digest, read, write, load, save_arrays, require, now)
RUN = ROOT / "runs/ash-phase5"
GEOMETRY = ROOT / "runs/rmd-phase4"
CONTROLS = METHODS + ADDITIONS + ["rmd","mahalanobis_cpu"]
NEW = ["ash_p_energy","ash_b_energy","ash_s_energy"]
ALL = CONTROLS + NEW
from ash_core import ash_scores
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
            require(read(RUN / 'preflight-status.json')['state'] == 'complete', 'preflight incomplete')
            audit = read(RUN / 'preflight-audit.json')
            require(audit['state'] == 'passed', 'preflight not independently audited')
            fit_inputs = read(RUN / 'preflight-inputs.json')
            inputs = dict(fit_inputs['original_artifacts'])
            paths = [RUN / 'preflight-inputs.json', RUN / 'preflight-status.json', RUN / 'preflight-audit.json']
            for seed in [0, 1, 2]:
                directory = RUN / f'seed{seed}'
                require(digest(directory / 'preflight.json') == audit['per_seed'][seed]['preflight_json_sha256'], 'audited fit changed')
                require(digest(directory / 'preflight.npz') == audit['per_seed'][seed]['raw_sha256'], 'fitted state changed')
                paths.extend([directory / 'preflight.json', directory / 'preflight.npz'])
            inputs.update({str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in paths})
            for name, expected in inputs.items():
                require(digest(ROOT / name) == expected, 'original scoring input changed')
            files = [Path(__file__), ROOT / 'tools/geometry_io.py', ROOT / 'tools/ash_core.py',
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
                manifest = {'state': 'inputs_frozen_before_ASH_OOD_scoring', 'prepared_at': now(),
                            'original_artifacts': inputs, 'source_hashes': sources, 'output_name': args.output_name,
                            'methods': ALL, 'new_methods': NEW, 'numeric_controls': ['affine_energy', 'mahalanobis_cpu'],
                            'primary_threshold_calibration_count': 2500, 'targets': TARGETS, 'numpy': np.__version__,
                            'boundary': 'saved features only; training-ID preflight only; no fitted statistics; OOD never selects settings'}
                write(manifest_path, manifest)
            write(RUN / 'status.json', {'state': 'running', 'stage': 'CPU ASH evaluation', 'pid': os.getpid(), 'started_at': now()})
            split = load(CALIBRATION / 'calibration_split.npz')
            threshold_rows = split['threshold_rows']
            for seed in [0, 1, 2]:
                directory = RUN / f'seed{seed}'
                output = directory / args.output_name
                if (output / 'results.json').exists():
                    result = read(output / 'results.json')
                    require(result['inputs_sha256'] == digest(manifest_path), 'completed ASH evaluation inputs changed')
                    for name, expected in result['output_file_sha256'].items():
                        require(digest(output / name) == expected, 'completed ASH output changed')
                    continue
                require(not output.exists() or not any(output.iterdir()), 'partial evaluation retained; use a fresh --output-name attempt')
                output.mkdir(exist_ok=True)
                metadata = read(directory/'preflight.json')
                bank=load(ROOT/f'runs/geometry-phase3/seed{seed}/training-bank.npz')
                weight,bias=bank['weight'],bank['bias']
                main = ROOT / f'runs/cifar100-seed{seed}/eval-final'
                id_data, val_data = load(main / 'id_logits.npz'), load(main / 'validation_logits.npz')
                features = load(main / 'features.npz')
                population_scores, diagnostics = {}, {}
                def score_population(name,feature,original_logits,labels=None):
                    require(feature.shape[1]==512,'fixed pooled dimensions changed')
                    score,middle=ash_scores(feature,weight,bias)
                    affine=feature.astype(float)@weight.astype(float).T+bias.astype(float)
                    middle['original_affine_logits']=affine
                    diag={'n':len(feature),'selected_coordinates':179,'cutoff_tie_rows':int((middle['cutoff_equal_count']>middle['cutoff_selected_equal_count']).sum()),'zero_cutoff_rows':int((middle['cutoff']==0).sum()),'minimum_original_sum':float(middle['original_sum'].min()),'minimum_retained_sum':float(middle['retained_sum'].min()),'minimum_ratio':float(middle['ratio'].min()),'maximum_ratio':float(middle['ratio'].max()),'maximum_exp_ratio':float(middle['exp_ratio'].max()),'max_original_float32_affine_difference':float(np.abs(affine-original_logits).max())}
                    if labels is not None:
                        middle['original_labels']=labels;middle['original_predictions']=original_logits.argmax(1)
                        diag['original_accuracy']=float((original_logits.argmax(1)==labels).mean())
                        for method in NEW:
                            prediction=middle[method+'_logits'].argmax(1)
                            diag[method+'_accuracy']=float((prediction==labels).mean())
                            diag[method+'_argmax_changes']=int((prediction!=original_logits.argmax(1)).sum())
                        diag['P_S_argmax_disagreements']=int((middle['ash_p_energy_logits'].argmax(1)!=middle['ash_s_energy_logits'].argmax(1)).sum())
                    save_arrays(output/f'{name}_intermediates.npz',**middle)
                    diagnostics[name]=diag
                    return score
                population_scores['id'] = score_population('id', features['id'], id_data['logits'], id_data['labels'])
                population_scores['val'] = score_population('validation', features['validation'], val_data['logits'], split['validation_labels'])
                for dataset in DATASETS:
                    source = ROOT / f'runs/cifar100-seed{seed}' / FOLDERS[dataset]
                    population_scores[dataset] = score_population(dataset, load(source / f'{dataset}_features.npz')['features'],
                                                                  load(source / f'{dataset}_logits.npz')['logits'])
                correct = id_data['logits'].argmax(1) == id_data['labels']
                masks = {'id_correct': correct, 'threshold_rows': threshold_rows,
                         'id_accepted': np.empty((len(ALL), 4, len(correct)), dtype=bool),
                         'calibration_accepted': np.empty((len(ALL), 4, 2500), dtype=bool)}
                for dataset in DATASETS:
                    masks[f'{dataset}_accepted'] = np.empty((len(ALL), 4, len(population_scores[dataset]['ash_p_energy'])), dtype=bool)
                datasets, grid = {}, []
                raw_by_dataset = {}
                for dataset in DATASETS:
                    baseline = load(GEOMETRY / f'seed{seed}/evaluation/{dataset}_scores.npz')
                    source_result = read(GEOMETRY / f'seed{seed}/evaluation/results.json')['datasets'][dataset]
                    saved = {}
                    metrics = {}
                    for method in ALL:
                        inside = baseline[f'{method}_id'] if method in CONTROLS else population_scores['id'][method]
                        outside = baseline[f'{method}_ood'] if method in CONTROLS else population_scores[dataset][method]
                        validation = baseline[f'{method}_val'] if method in CONTROLS else population_scores['val'][method]
                        selected = validation[threshold_rows]
                        primary = detection_metrics(inside, outside, selected)
                        primary.update({'benchmark_ID_equal_threshold_count': int((inside == primary['benchmark_threshold']).sum()),
                                        'threshold_calibration_equal_threshold_count': int((selected == primary['validation_threshold']).sum()),
                                        'threshold_calibration_n': 2500})
                        if method in CONTROLS:
                            old = source_result['methods'][method]['primary_2500']
                            for key, value in primary.items():
                                require(value == old[key], 'phase-4 frozen control metric changed: ' + key)
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
                for mi, method in enumerate(ALL):
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
                          'preflight_json_sha256': digest(directory / 'preflight.json'), 'checkpoint_sha256': metadata['checkpoint_sha256'],
                          'methods': ALL, 'named_additions': NEW, 'numeric_controls': ['affine_energy','mahalanobis_cpu'],
                          'score_direction': 'higher_is_ood', 'temperature': 1., 'primary_calibration_count': 2500,
                          'datasets': datasets, 'operating_points': grid, 'grid_targets': TARGETS,
                          'intermediate_diagnostics': diagnostics,
                          'original_ID_classifier_accuracy': float(correct.mean()),
                          'all_twelve_phase4_controls_preserved': True, 'pruning_percentile':65, 'retained_coordinates':179,
                          'output_file_sha256': {p.name: digest(p) for p in output.glob('*.npz')},
                          'boundary': 'fixed exploratory ASH extension; no fitted statistics; original classifier correctness'}
                write(output / 'results.json', result)
                print(__import__('json').dumps({'seed': seed, 'state': 'scores_saved', 'methods': len(result['methods']),
                                              'natural_datasets': 4, 'synthetic_diagnostics': 1, 'controls_preserved': 12}), flush=True)
                write(RUN / 'status.json', {'state': 'running', 'stage': 'CPU ASH evaluation', 'pid': os.getpid(),
                                           'completed_evaluation_seeds': list(range(seed + 1)), 'updated_at': now()})
            for name, expected in inputs.items():
                require(digest(ROOT / name) == expected, 'original scoring input changed during evaluation')
            for name, expected in sources.items():
                require(digest(ROOT / name) == expected, 'scoring source changed during evaluation')
            write(RUN / 'status.json', {'state': 'evaluation_complete_pending_independent_audit', 'pid': os.getpid(),
                                       'completed_evaluation_seeds': [0, 1, 2], 'output_name': args.output_name,
                                       'original_artifacts_unchanged': True, 'completed_at': now()})
        except Exception as error:
            write(RUN / 'status.json', {'state': 'failed', 'stage': 'CPU ASH evaluation', 'pid': os.getpid(), 'error': repr(error), 'updated_at': now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
