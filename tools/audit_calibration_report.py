"""Rebuild paired seed/macro summaries with Python statistics, separately from NumPy producer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/calibration-phase2'
RUN = ROOT / 'runs/calibration-phase2'
METHODS = ['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis']
NATURAL = ['cifar10', 'mnist', 'svhn', 'textures']
ENDPOINTS = ['auroc', 'aupr_out', 'aupr_in', 'fpr95_id', 'id_tpr_at_fpr95',
             'id_test_acceptance_at_validation_threshold', 'ood_acceptance_at_validation_threshold']


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
    raw_audit = read(OUT / 'audit.json')
    require(raw_audit['state'] == 'passed' and digest(OUT / 'audit.json') == summary['raw_audit_sha256'], 'raw audit changed')
    require(summary['seeds'] == [0, 1, 2] and summary['natural_datasets'] == NATURAL and summary['synthetic_diagnostics'] == ['noise'], 'population changed')
    results = []
    for seed in [0, 1, 2]:
        path = RUN / f'seed{seed}/results.json'
        require(digest(path) == raw_audit['results_sha256'][f'seed{seed}'] == summary['result_sha256'][f'seed{seed}'], 'seed source changed')
        results.append(read(path))
    checks, maximum_error = 0, 0.
    def check_stat(saved, values):
        nonlocal checks, maximum_error
        require(saved['n_seeds'] == 3 and saved['per_seed'] == values, 'per-seed pairing/list mismatch')
        for key, value in [('mean', statistics.fmean(values)), ('sample_sd', statistics.stdev(values))]:
            error = abs(saved[key] - value)
            maximum_error = max(maximum_error, error)
            require(error < 1e-12, key + ' mismatch (sample SD required)')
            checks += 1
    def check_pair(saved, before, after):
        check_stat(saved['T1'], before)
        check_stat(saved['calibrated'], after)
        check_stat(saved['paired_delta_calibrated_minus_T1'], [b - a for a, b in zip(before, after)])
    check_stat(summary['temperature'], [r['temperature_fit']['temperature'] for r in results])
    for metric in ['nll', 'brier', 'ece15', 'accuracy', 'mean_confidence']:
        check_pair(summary['primary_ID_test'][metric], [r['id_test']['T1'][metric] for r in results],
                   [r['id_test']['calibrated'][metric] for r in results])
    for dataset in NATURAL + ['noise']:
        for method in METHODS:
            for metric in ENDPOINTS:
                check_pair(summary['per_dataset'][dataset][method][metric],
                           [r['datasets'][dataset]['variants']['T1'][method][metric] for r in results],
                           [r['datasets'][dataset]['variants']['calibrated'][method][metric] for r in results])
        record = summary['counts'][dataset]
        for r in results:
            for variant in ['T1', 'calibrated']:
                for method in METHODS:
                    saved = r['datasets'][dataset]['variants'][variant][method]
                    for key in ['n_id', 'n_ood', 'ood_prevalence']:
                        require(record[key] == saved[key], 'population/prevalence mismatch')
    for method in METHODS:
        for metric in ENDPOINTS:
            before = [statistics.fmean(r['datasets'][d]['variants']['T1'][method][metric] for d in NATURAL) for r in results]
            after = [statistics.fmean(r['datasets'][d]['variants']['calibrated'][method][metric] for d in NATURAL) for r in results]
            # Producer used NumPy mean; allow rounding in per-seed macro values while requiring correct inputs/pairing.
            saved = summary['equal_weight_natural_macro'][method][metric]
            for name, expected in [('T1', before), ('calibrated', after), ('paired_delta_calibrated_minus_T1', [b - a for a, b in zip(before, after)])]:
                require(all(abs(x - y) < 1e-12 for x, y in zip(saved[name]['per_seed'], expected)), 'macro includes wrong populations')
                check_stat(saved[name], saved[name]['per_seed'])
                require(abs(saved[name]['mean'] - statistics.fmean(expected)) < 1e-12, 'incorrect natural macro')
    for method in METHODS:
        p = summary['original_baseline_precision_controls'][method]
        values = [r['precision_and_original_thresholds'][d][method] for r in results for d in NATURAL + ['noise']]
        require(p['max_absolute_score_difference'] == max(x['max_abs_score_difference'][part] for x in values for part in ['id', 'val', 'ood']), 'precision summary hides score delta')
        for metric, key in [('auroc', 'max_absolute_AUROC_difference'), ('fpr95_id', 'max_absolute_FPR95_difference')]:
            require(p[key] == max(abs(x['metric_delta_T1_minus_original'][metric]) for x in values), 'precision summary hides metric delta')
    require(summary['report_source_sha256'] == digest(ROOT / 'tools/build_calibration_report.py'), 'report source changed')
    artifacts = {p.name: digest(p) for p in OUT.iterdir() if p.suffix in ['.md', '.png', '.svg']}
    output = {'state': 'passed', 'statistics_checked': checks, 'maximum_statistic_error': maximum_error,
              'summary_sha256': digest(OUT / 'summary.json'), 'raw_audit_sha256': digest(OUT / 'audit.json'),
              'report_artifact_sha256': artifacts, 'audit_source_sha256': digest(__file__),
              'audited_at': datetime.now(timezone.utc).isoformat(),
              'boundary': 'independent mean/sample SD and paired deltas; four fixed natural datasets within each seed; synthetic noise excluded'}
    (OUT / 'summary-audit.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
