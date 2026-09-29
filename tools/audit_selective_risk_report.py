"""Independent paired seed statistics, natural macro and 720-row count export audit."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/selective-risk-phase2'
OUT = ROOT / 'report/selective-risk-phase2'
METHODS = ['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis']
TARGETS = [.8, .9, .95, .99]
NATURAL = ['cifar10', 'mnist', 'svhn', 'textures']
DATASETS = NATURAL + ['noise']
ID_METRICS = ['coverage', 'retained_error_risk', 'accepted_accuracy', 'unconditional_accuracy',
              'correct_rejection_fraction', 'incorrect_rejection_fraction']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    summary = read(OUT / 'summary.json')
    raw = read(OUT / 'audit.json')
    require(raw['state'] == 'passed' and digest(OUT / 'audit.json') == summary['raw_audit_sha256'], 'raw audit changed')
    require(summary['methods'] == METHODS and summary['targets'] == TARGETS and summary['seeds'] == [0, 1, 2], 'grid settings changed')
    require(summary['natural_datasets'] == NATURAL and summary['synthetic_diagnostics'] == ['noise'], 'macro populations changed')
    lookups = []
    for seed in [0, 1, 2]:
        path = RUN / f'seed{seed}/results.json'
        require(digest(path) == raw['results_sha256'][f'seed{seed}'] == summary['results_sha256'][f'seed{seed}'], 'source result changed')
        lookups.append({(p['method'], p['variant'], p['target']): p for p in read(path)['operating_points']})
    checks, maximum = 0, 0.
    def check_stat(saved, values):
        nonlocal checks, maximum
        require(saved['n_seeds'] == 3 and len(saved['per_seed']) == 3, 'seed list missing')
        for actual, expected in zip(saved['per_seed'], values):
            require((actual is None and expected is None) or (actual is not None and expected is not None and abs(actual - expected) < 1e-12),
                    'incorrect paired/per-seed values')
        valid = [v for v in values if v is not None]
        require(saved['n_defined'] == len(valid), 'undefined population omitted silently')
        expected = {'mean': statistics.fmean(valid) if valid else None,
                    'sample_sd': statistics.stdev(valid) if len(valid) > 1 else None}
        for name, value in expected.items():
            if value is None:
                require(saved[name] is None, 'undefined statistic invented')
            else:
                error = abs(saved[name] - value)
                maximum = max(maximum, error)
                require(error < 1e-12, 'incorrect mean/sample SD')
            checks += 1
    def check_pair(saved, before, after):
        check_stat(saved['T1'], before)
        check_stat(saved['calibrated'], after)
        check_stat(saved['paired_delta_calibrated_minus_T1'], [b - a if a is not None and b is not None else None for a, b in zip(before, after)])
    for method in METHODS:
        for target in TARGETS:
            key = f'{target:.2f}'
            for metric in ID_METRICS:
                check_pair(summary['id_test'][method][key][metric],
                           [r[(method, 'T1', target)]['id_test'][metric] for r in lookups],
                           [r[(method, 'calibrated', target)]['id_test'][metric] for r in lookups])
            for dataset in DATASETS:
                check_pair(summary['ood_acceptance'][dataset][method][key],
                           [r[(method, 'T1', target)]['ood'][dataset]['acceptance'] for r in lookups],
                           [r[(method, 'calibrated', target)]['ood'][dataset]['acceptance'] for r in lookups])
            before = [statistics.fmean(r[(method, 'T1', target)]['ood'][d]['acceptance'] for d in NATURAL) for r in lookups]
            after = [statistics.fmean(r[(method, 'calibrated', target)]['ood'][d]['acceptance'] for d in NATURAL) for r in lookups]
            check_pair(summary['equal_weight_natural_ood_acceptance'][method][key], before, after)
    with (OUT / 'per-seed-counts.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 720, 'count export incomplete')
    combinations = set()
    csv_checks = 0
    for row in rows:
        seed, method, variant = int(row['seed']), row['method'], row['variant']
        target, dataset = float(row['target_ID_calibration_acceptance']), row['OOD_dataset']
        identity = seed, method, variant, target, dataset
        require(identity not in combinations, 'duplicate count export row')
        combinations.add(identity)
        point = lookups[seed][(method, variant, target)]
        expected = {'threshold': point['threshold'], 'calibration_n': point['calibration']['n'],
                    'calibration_accepted': point['calibration']['accepted'],
                    'calibration_actual_acceptance': point['calibration']['achieved_acceptance']}
        expected.update({'ID_' + k: v for k, v in point['id_test'].items()})
        ood = point['ood'][dataset]
        expected.update({'OOD_diagnostic_only': ood['diagnostic_only'], 'OOD_n': ood['n'], 'OOD_accepted': ood['accepted'],
                         'OOD_rejected': ood['rejected'], 'OOD_acceptance': ood['acceptance'],
                         'OOD_equal_threshold_count': ood['equal_threshold_count'],
                         'ID_OOD_population_prevalence': ood['id_ood_population_prevalence'],
                         'OOD_score_array_sha256': ood['score_array_sha256']})
        for key, value in expected.items():
            require(row[key] == (str(value) if value is not None else ''), 'count export field changed: ' + key)
            csv_checks += 1
    require(len(combinations) == 3 * 6 * 2 * 4 * 5, 'count export population missing')
    require(summary['report_source_sha256'] == digest(ROOT / 'tools/build_selective_risk_report.py'), 'report source changed')
    artifacts = {p.name: digest(p) for p in OUT.iterdir() if p.suffix in ['.md', '.png', '.svg', '.csv']}
    audit = {'state': 'passed', 'statistics_checked': checks, 'maximum_statistics_error': maximum,
             'CSV_rows_checked': len(rows), 'CSV_fields_checked': csv_checks,
             'summary_sha256': digest(OUT / 'summary.json'), 'raw_audit_sha256': digest(OUT / 'audit.json'),
             'report_artifact_sha256': artifacts, 'audit_source_sha256': digest(__file__),
             'audited_at': datetime.now(timezone.utc).isoformat(),
             'boundary': 'Python statistics verifies paired seed values/sample SD; equal-weight natural macro within seed; all targets and populations retained'}
    (OUT / 'summary-audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
