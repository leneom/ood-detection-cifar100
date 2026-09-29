"""Audit complete-seed summaries independently with Python's statistics module."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys


METRICS = ('auroc', 'aupr_out', 'aupr_in', 'fpr95_id',
           'id_test_acceptance_at_validation_threshold', 'ood_acceptance_at_validation_threshold')


def stats(values):
    values = list(map(float, values))
    if not values or not all(map(math.isfinite, values)):
        raise ValueError('finite, nonempty observations required')
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
            'n_seeds': len(values), 'per_seed': values}


def verify_stat(saved, values):
    expected = stats(values)
    if saved['n_seeds'] != expected['n_seeds'] or len(saved['per_seed']) != len(values):
        raise ValueError('seed population differs')
    errors = [abs(a - b) for a, b in zip(saved['per_seed'], expected['per_seed'])]
    errors.append(abs(saved['mean'] - expected['mean']))
    if expected['sample_sd'] is None:
        if saved['sample_sd'] is not None:
            raise ValueError('single-seed SD must be unavailable')
    elif saved['sample_sd'] is None:
        raise ValueError('multiseed SD missing')
    else:
        errors.append(abs(saved['sample_sd'] - expected['sample_sd']))
    if not all(math.isfinite(error) and error < 1e-12 for error in errors):
        raise ValueError('saved means, sample SD, or per-seed values differ')
    return max(errors)


def compare(summary, results, expected_seeds):
    results = sorted(results, key=lambda r: r['training_config']['seed'])
    seeds = [r['training_config']['seed'] for r in results]
    if len(seeds) != len(set(seeds)) or seeds != sorted(expected_seeds) or summary['seeds'] != seeds:
        raise ValueError('all predeclared seeds are required exactly once')
    maximum_error = verify_stat(summary['id_test_accuracy'], [r['id_test_accuracy'] for r in results])
    checks = 1
    datasets = results[0]['datasets']
    if set(summary['datasets']) != set(datasets):
        raise ValueError('summary dataset set differs')
    natural = [name for name in datasets if name not in ('noise', 'constant')]
    if natural and summary.get('natural_macro_datasets') != natural:
        raise ValueError('natural macro includes diagnostics or omits natural OOD')
    paired = {}
    for dataset, entry in datasets.items():
        if set(summary['datasets'][dataset]) != set(entry['methods']):
            raise ValueError('summary method set differs')
        paired[dataset] = {}
        for method in entry['methods']:
            for metric in METRICS:
                values = [r['datasets'][dataset]['methods'][method][metric] for r in results]
                maximum_error = max(maximum_error, verify_stat(summary['datasets'][dataset][method][metric], values))
                checks += 1
            if method != 'msp' and 'msp' in entry['methods']:
                paired[dataset][method] = {
                    metric: stats([r['datasets'][dataset]['methods'][method][metric] -
                                   r['datasets'][dataset]['methods']['msp'][metric] for r in results])
                    for metric in ('auroc', 'fpr95_id')}
    if natural:
        common = set.intersection(*(set(datasets[name]['methods']) for name in natural))
        if set(summary['natural_macro']) != common:
            raise ValueError('natural macro method set differs')
        for method in common:
            for metric in METRICS:
                values = [statistics.mean(r['datasets'][name]['methods'][method][metric] for name in natural)
                          for r in results]
                maximum_error = max(maximum_error, verify_stat(summary['natural_macro'][method][metric], values))
                checks += 1
    provenance = [{'seed': r['training_config']['seed'], 'selected_epoch': r['checkpoint_epoch'],
                   'sha256': r['checkpoint_sha256']} for r in results]
    if summary['checkpoint_provenance'] != provenance:
        raise ValueError('summary checkpoint provenance differs')
    return {'status': 'passed', 'seeds': seeds, 'statistics_checked': checks,
            'maximum_statistics_error': maximum_error, 'paired_method_minus_msp': paired}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluations', nargs='+', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--expected-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if len(args.expected_seeds) != len(set(args.expected_seeds)):
        raise ValueError('duplicate expected seeds')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_evaluation import audit
    results, audits, protocol = [], [], None
    for path in map(Path, args.evaluations):
        audits.append(audit(path))  # Includes raw scores, checkpoint, sources, split and full schedule.
        result = json.loads((path / 'results.json').read_text(encoding='utf-8'))
        config = {key: value for key, value in result['training_config'].items() if key != 'seed'}
        schema = {name: {method: [row['n_id'], row['n_ood']] for method, row in entry['methods'].items()}
                  for name, entry in result['datasets'].items()}
        current = (config, schema, result['temperature'], result['feature_fit'],
                   result['score_direction'], result['evaluation_source_hashes'], result['environment'])
        if protocol is None:
            protocol = current
        elif protocol != current:
            raise ValueError('seed configuration, detector protocol, populations, source or environment differs')
        results.append(result)
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    output = Path(args.output_dir)
    report = compare(summary, results, args.expected_seeds)
    report.update({'per_seed_raw_audits': audits, 'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                   'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'python': platform.python_version(), 'audited_at': datetime.now(timezone.utc).isoformat()})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'matrix-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Independent complete-seed audit', '',
             f"Training seeds: {report['seeds']}; {report['statistics_checked']} statistic records verified.", '',
             'Raw per-seed metrics, ID-only checkpoint selection, split and source hashes were audited separately. '
             'Means and sample SD were independently recomputed with Python statistics, without importing '
             'the production aggregation function.', '',
             '## Paired differences from MSP', '',
             'Each value is method minus MSP on the same classifier, followed by sample SD across training seeds. '
             'Positive AUROC differences and negative FPR95-ID differences favor the listed method. '
             'These are descriptive paired differences, not confidence intervals, significance tests, '
             'or a test-set method-selection rule.', '']
    for dataset, methods in report['paired_method_minus_msp'].items():
        lines += [f'### {dataset}', '', '| Method | AUROC difference pp | FPR95-ID difference pp |', '|---|---:|---:|']
        for method, values in methods.items():
            def fmt(row):
                sd = row['sample_sd']
                return f"{100 * row['mean']:+.2f} ± {100 * sd:.2f}" if sd is not None else f"{100 * row['mean']:+.2f} (one seed)"
            lines.append(f"| {method} | {fmt(values['auroc'])} | {fmt(values['fpr95_id'])} |")
        lines.append('')
    lines += ['Only all explicitly expected, completed seeds enter this audit. Single-seed diagnostic smoke '
              'does not establish training-seed variability. All artifacts remain local.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'seeds': report['seeds'],
                      'statistics_checked': report['statistics_checked'], 'output': str(output)}))


if __name__ == '__main__':
    main()
