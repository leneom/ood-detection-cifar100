"""Consolidate audited frozen-classifier phases without modifying raw results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

METHODS = ('msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis')
NATURAL = ('cifar10', 'mnist', 'svhn', 'textures')
STABLE_SOURCES = ('data.py', 'models.py', 'runtime.py', 'train.py', 'features.py', 'scores.py', 'metrics.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stat(values):
    values = list(map(float, values))
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'per_seed': values}


def fmt(value):
    return f"{100 * value['mean']:.2f} ± {100 * value['sample_sd']:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=Path('runs'))
    parser.add_argument('--report-root', type=Path, default=Path('report'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    phases = [('matrix', 'eval-final', 'matrix-audit'),
              ('svhn-extension', 'eval-extension-svhn', 'svhn-matrix-audit'),
              ('textures-extension', 'eval-extension-textures', 'textures-matrix-audit')]
    combined, provenance, audit_count = {}, [], 0
    for control_name, evaluation_name, audit_name in phases:
        summary_path = args.runs_root / control_name / 'summary' / 'summary.json'
        audit_path = args.report_root / audit_name / 'matrix-audit.json'
        audit = read(audit_path)
        if audit['status'] != 'passed' or audit['seeds'] != [0, 1, 2] or audit['summary_sha256'] != digest(summary_path):
            raise ValueError('each exact phase summary requires a passed complete-seed audit')
        audit_count += audit['statistics_checked']
        phase_summary = read(summary_path)
        if phase_summary['seeds'] != [0, 1, 2]:
            raise ValueError('phase summary expected-seed identity differs')
        refs, phase_results = [], []
        for seed in (0, 1, 2):
            path = args.runs_root / f'cifar100-seed{seed}' / evaluation_name / 'results.json'
            result = read(path)
            phase_results.append(result)
            if result['training_config']['seed'] != seed:
                raise ValueError('phase seed identity differs')
            if seed not in combined:
                combined[seed] = {'reference': result, 'datasets': {}}
            base = combined[seed]['reference']
            for key in ('checkpoint_sha256', 'checkpoint_epoch', 'training_config', 'feature_fit',
                        'temperature', 'score_direction', 'id_test_accuracy'):
                if result[key] != base[key]:
                    raise ValueError(f'phase differs for the same frozen classifier: {key}')
            for source in STABLE_SOURCES:
                if result['evaluation_source_hashes'][source] != base['evaluation_source_hashes'][source]:
                    raise ValueError('phase changes training/detector implementation')
            for name, entry in result['datasets'].items():
                if name in combined[seed]['datasets'] or set(entry['methods']) != set(METHODS):
                    raise ValueError('duplicate dataset or changed six-method population')
                combined[seed]['datasets'][name] = entry
            refs.append({'seed': seed, 'path': str(path.resolve()), 'results_sha256': digest(path)})
        for name, entry in phase_results[0]['datasets'].items():
            for method in METHODS:
                for metric in ('auroc', 'fpr95_id'):
                    current = stat(result['datasets'][name]['methods'][method][metric] for result in phase_results)
                    saved = phase_summary['datasets'][name][method][metric]
                    if not np.allclose([current['mean'], current['sample_sd'], *current['per_seed']],
                                       [saved['mean'], saved['sample_sd'], *saved['per_seed']], atol=1e-12, rtol=0):
                        raise ValueError('current raw result no longer matches audited phase statistics')
        provenance.append({'phase': control_name, 'summary_sha256': digest(summary_path),
                           'audit_sha256': digest(audit_path), 'results': refs})
    pixel_path = args.report_root / 'textures-matrix-audit' / 'input-pixel-audit.json'
    pixel_audit = read(pixel_path)
    if pixel_audit['state'] != 'passed' or pixel_audit['seeds'] != [0, 1, 2] or pixel_audit['population_pixel_hash_checks'] != 12:
        raise ValueError('new-phase executed inputs require the complete pixel audit')
    for item in pixel_audit['per_seed']:
        if digest(Path(item['evaluation']) / 'results.json') != item['results_sha256']:
            raise ValueError('executed-input audit no longer binds current Textures results')
    populations = {}
    per_dataset = {}
    for name in NATURAL:
        entries = [combined[seed]['datasets'][name] for seed in (0, 1, 2)]
        counts = {(row['n_id'], row['n_ood'], row['ood_prevalence']) for entry in entries for row in entry['methods'].values()}
        if len(counts) != 1:
            raise ValueError('dataset population differs between methods/seeds')
        n_id, n_ood, prevalence = counts.pop()
        populations[name] = {'n_id': n_id, 'n_ood': n_ood, 'ood_prevalence': prevalence}
        per_dataset[name] = {method: {metric: stat(entry['methods'][method][metric] for entry in entries)
                                     for metric in ('auroc', 'fpr95_id')} for method in METHODS}
    macro, macro_checks = {}, 0
    for method in METHODS:
        macro[method] = {}
        for metric in ('auroc', 'fpr95_id'):
            matrix = np.asarray([[combined[seed]['datasets'][name]['methods'][method][metric]
                                  for name in NATURAL] for seed in (0, 1, 2)])
            values = [statistics.mean(row) for row in matrix.tolist()]
            saved = stat(values)
            if not np.allclose([saved['mean'], saved['sample_sd']],
                               [matrix.mean(axis=1).mean(), matrix.mean(axis=1).std(ddof=1)], atol=1e-12, rtol=0):
                raise ValueError('independent equal-weight macro statistic check failed')
            macro[method][metric] = saved
            macro_checks += 1
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('overview output must be empty; retain earlier reports')
    output.mkdir(parents=True, exist_ok=True)
    record = {'seeds': [0, 1, 2], 'natural_datasets': list(NATURAL), 'populations': populations,
              'id_test_accuracy': stat(combined[seed]['reference']['id_test_accuracy'] for seed in (0, 1, 2)),
              'per_dataset': per_dataset, 'equal_weight_natural_macro': macro, 'phases': provenance,
              'phase_audit_statistics_checked': audit_count, 'macro_statistic_checks': macro_checks,
              'executed_input_audit_sha256': digest(pixel_path), 'tool_sha256': digest(__file__),
              'boundary': 'independent local protocol; descriptive sample SD, not confidence intervals or method selection'}
    (output / 'overview.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    lines = ['# Complete local OOD baseline', '',
             f"Three fixed 100-epoch CIFAR-100 classifiers; ID test accuracy {fmt(record['id_test_accuracy'])}%.", '',
             f'All six methods across four natural OOD populations are complete. Phase audits checked {audit_count} '
             'statistic records and all underlying raw metrics. The Textures phase additionally independently '
             'reconstructed 12 executed ID/OOD input hashes. Previous phases and synthetic diagnostics are preserved.', '',
             '| Natural OOD | ID count | OOD count | OOD prevalence |', '|---|---:|---:|---:|']
    for name, population in populations.items():
        lines.append(f"| {name} | {population['n_id']} | {population['n_ood']} | {100 * population['ood_prevalence']:.2f}% |")
    for name, methods in [*per_dataset.items(), ('equal-weight natural macro', macro)]:
        lines += ['', f'## {name}', '', '| Method | AUROC % | FPR95-ID % |', '|---|---:|---:|']
        for method in METHODS:
            lines.append(f"| {method} | {fmt(methods[method]['auroc'])} | {fmt(methods[method]['fpr95_id'])} |")
    lines += ['', '## Interpretation', '',
              'Each ± is sample SD across training seeds, not a confidence interval. Macro first averages the four '
              'natural datasets equally within each seed, then summarizes those three paired values. Noise and constant '
              'images are excluded. No samples from different OOD populations are pooled.', '',
              'AUPR retains its population-specific counts/prevalence in the original reports; dataset prevalence differs, '
              'so AP values across these populations have different reference levels. FPR95-ID uses the benchmark '
              'ID-test threshold; validation-only operating points remain separately saved in each raw result.', '',
              'This is an independent local CIFAR protocol, not full OpenOOD reproduction or a leaderboard claim. '
              'All checkpoints use ID validation selection and all feature fits use the training ID split. '
              'Temperature, k, shrinkage and score directions were not changed from OOD tests.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
