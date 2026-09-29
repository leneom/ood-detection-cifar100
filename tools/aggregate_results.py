"""Aggregate completed, compatible training seeds; retain sample SD separately."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def collect(evaluations):
    results, seeds, reference = [], set(), None
    for directory in map(Path, evaluations):
        result = json.loads((directory / 'results.json').read_text(encoding='utf-8-sig'))
        config = result['training_config']
        status = json.loads((directory.parent / 'status.json').read_text(encoding='utf-8-sig'))
        if status.get('state') != 'complete' or status.get('epoch') != config['epochs']:
            raise ValueError('only completed full-schedule runs can enter seed statistics')
        seed = config['seed']
        if seed in seeds:
            raise ValueError('duplicate training seed')
        seeds.add(seed)
        comparison = {key: value for key, value in config.items() if key != 'seed'}
        schema = {dataset: {method: (values['n_id'], values['n_ood']) for method, values in entry['methods'].items()}
                  for dataset, entry in result['datasets'].items()}
        protocol = (comparison, schema, result['temperature'], result.get('feature_fit'))
        if reference is None:
            reference = protocol
        elif protocol != reference:
            raise ValueError('training configuration, evaluation populations, or detector protocol differs')
        results.append(result)
    if not results:
        raise ValueError('at least one completed evaluation is required')
    return sorted(results, key=lambda result: result['training_config']['seed'])


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('statistics require finite nonempty observations')
    return {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else None,
            'n_seeds': len(values), 'per_seed': values.tolist()}


def aggregate(results):
    summary = {'seeds': [result['training_config']['seed'] for result in results],
               'id_test_accuracy': statistics([result['id_test_accuracy'] for result in results]),
               'datasets': {}, 'natural_macro': {},
               'checkpoint_provenance': [{'seed': result['training_config']['seed'],
                                         'selected_epoch': result['checkpoint_epoch'],
                                         'sha256': result['checkpoint_sha256']} for result in results]}
    metrics = ['auroc', 'aupr_out', 'aupr_in', 'fpr95_id',
               'id_test_acceptance_at_validation_threshold', 'ood_acceptance_at_validation_threshold']
    for dataset, entry in results[0]['datasets'].items():
        summary['datasets'][dataset] = {
            method: {metric: statistics([result['datasets'][dataset]['methods'][method][metric] for result in results])
                     for metric in metrics} for method in entry['methods']}
    natural = [name for name in results[0]['datasets'] if name not in ('noise', 'constant')]
    if natural:
        common = set.intersection(*(set(summary['datasets'][name]) for name in natural))
        for method in sorted(common):
            summary['natural_macro'][method] = {
                metric: statistics([np.mean([result['datasets'][name]['methods'][method][metric] for name in natural])
                                    for result in results]) for metric in metrics}
        summary['natural_macro_datasets'] = natural
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluations', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    results = collect(args.evaluations)
    summary = aggregate(results)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    def fmt(stat):
        mean, sd = stat['mean'] * 100, stat['sample_sd']
        return f'{mean:.2f} ± {sd * 100:.2f}' if sd is not None else f'{mean:.2f} (one seed)'
    text = ['# Completed OOD baseline seed comparison', '',
            f"Training seeds: {summary['seeds']}. ID test accuracy: {fmt(summary['id_test_accuracy'])}%.", '',
            'Values are mean ± sample standard deviation across independently trained seeds. '
            'These are not test-sample bootstrap intervals.', '']
    csv_rows = []
    groups = list(summary['datasets'].items())
    if summary['natural_macro']:
        groups.append(('natural_macro', summary['natural_macro']))
    for dataset, methods in groups:
        text += [f'## {dataset}', '', '| Method | AUROC % | AUPR-out % | FPR95-ID % |', '|---|---:|---:|---:|']
        for method, values in methods.items():
            text.append(f"| {method} | {fmt(values['auroc'])} | {fmt(values['aupr_out'])} | {fmt(values['fpr95_id'])} |")
            for metric, stat in values.items():
                csv_rows.append({'dataset': dataset, 'method': method, 'metric': metric, **{key: value for key, value in stat.items() if key != 'per_seed'}})
        text.append('')
    text += ['Natural macro uses equal weights across natural OOD datasets; synthetic diagnostics are excluded. '
             'This independent protocol is not an OpenOOD leaderboard reproduction.', '']
    (output / 'SUMMARY.md').write_text('\n'.join(text), encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output / 'SUMMARY.md')


if __name__ == '__main__':
    main()
