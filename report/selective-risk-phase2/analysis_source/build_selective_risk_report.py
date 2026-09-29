"""All fixed operating-point summaries and paired scientific plots, no selection."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/selective-risk-phase2'
OUT = ROOT / 'report/selective-risk-phase2'
METHODS = ['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis']
TARGETS = [.8, .9, .95, .99]
VARIANTS = ['T1', 'calibrated']
NATURAL = ['cifar10', 'mnist', 'svhn', 'textures']
DATASETS = NATURAL + ['noise']
ID_METRICS = ['coverage', 'retained_error_risk', 'accepted_accuracy', 'unconditional_accuracy',
              'correct_rejection_fraction', 'incorrect_rejection_fraction']
DISPLAY = dict(zip(METHODS, ['MSP', 'Energy', 'MaxLogit', 'Entropy', 'KNN', 'Mahalanobis']))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(values):
    valid = [v for v in values if v is not None]
    return {'per_seed': values, 'n_seeds': 3, 'n_defined': len(valid),
            'mean': float(np.mean(valid)) if valid else None,
            'sample_sd': float(np.std(valid, ddof=1)) if len(valid) > 1 else None}


def paired(before, after):
    delta = [b - a if a is not None and b is not None else None for a, b in zip(before, after)]
    return {'T1': stats(before), 'calibrated': stats(after), 'paired_delta_calibrated_minus_T1': stats(delta)}


def fmt(record):
    if record['mean'] is None:
        return 'undefined'
    sd = f"{record['sample_sd'] * 100:.2f}" if record['sample_sd'] is not None else 'undefined'
    return f"{record['mean'] * 100:.2f} ± {sd}"


def main():
    audit = read(OUT / 'audit.json')
    if audit['state'] != 'passed' or audit['inputs_sha256'] != digest(RUN / 'inputs.json'):
        raise ValueError('missing/stale raw decision audit')
    results = []
    lookups = []
    for seed in [0, 1, 2]:
        path = RUN / f'seed{seed}/results.json'
        if digest(path) != audit['results_sha256'][f'seed{seed}']:
            raise ValueError('audited result changed')
        result = read(path)
        results.append(result)
        lookups.append({(p['method'], p['variant'], p['target']): p for p in result['operating_points']})
    id_test, outside, macro = {}, {d: {} for d in DATASETS}, {}
    for method in METHODS:
        id_test[method], macro[method] = {}, {}
        for dataset in DATASETS:
            outside[dataset][method] = {}
        for target in TARGETS:
            key = f'{target:.2f}'
            id_test[method][key] = {}
            for metric in ID_METRICS:
                id_test[method][key][metric] = paired(
                    [r[(method, 'T1', target)]['id_test'][metric] for r in lookups],
                    [r[(method, 'calibrated', target)]['id_test'][metric] for r in lookups])
            for dataset in DATASETS:
                outside[dataset][method][key] = paired(
                    [r[(method, 'T1', target)]['ood'][dataset]['acceptance'] for r in lookups],
                    [r[(method, 'calibrated', target)]['ood'][dataset]['acceptance'] for r in lookups])
            macro[method][key] = paired([r[(method, 'T1', target)]['equal_weight_natural_ood_acceptance'] for r in lookups],
                                       [r[(method, 'calibrated', target)]['equal_weight_natural_ood_acceptance'] for r in lookups])
    summary = {'state': 'complete_audited_grid', 'seeds': [0, 1, 2], 'methods': METHODS, 'variants': VARIANTS,
               'targets': TARGETS, 'natural_datasets': NATURAL, 'synthetic_diagnostics': ['noise'],
               'id_test': id_test, 'ood_acceptance': outside, 'equal_weight_natural_ood_acceptance': macro,
               'operating_points': 144, 'threshold_calibration_population': 2500,
               'results_sha256': audit['results_sha256'], 'inputs_sha256': audit['inputs_sha256'],
               'raw_audit_sha256': digest(OUT / 'audit.json'), 'report_source_sha256': digest(__file__),
               'created_at': datetime.now(timezone.utc).isoformat(),
               'boundary': 'all four fixed targets; actual test coverage distinct from calibration target; reused selection validation; sample SD not CI'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    # All 144 settings × five OOD populations, retaining exact counts and separate ID denominators.
    rows = []
    for r in results:
        for p in r['operating_points']:
            for dataset in DATASETS:
                o = p['ood'][dataset]
                row = {'seed': r['seed'], 'method': p['method'], 'variant': p['variant'], 'target_ID_calibration_acceptance': p['target'],
                       'threshold': p['threshold'], 'calibration_n': p['calibration']['n'],
                       'calibration_accepted': p['calibration']['accepted'],
                       'calibration_actual_acceptance': p['calibration']['achieved_acceptance']}
                row.update({'ID_' + key: value for key, value in p['id_test'].items()})
                row.update({'OOD_dataset': dataset, 'OOD_diagnostic_only': o['diagnostic_only'], 'OOD_n': o['n'],
                            'OOD_accepted': o['accepted'], 'OOD_rejected': o['rejected'], 'OOD_acceptance': o['acceptance'],
                            'OOD_equal_threshold_count': o['equal_threshold_count'],
                            'ID_OOD_population_prevalence': o['id_ood_population_prevalence'],
                            'OOD_score_array_sha256': o['score_array_sha256']})
                rows.append(row)
    with (OUT / 'per-seed-counts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Fixed ID-only operating points: complete three-seed analysis', '',
             'All six scores, both matched T=1 and ID-fitted-temperature variants, all three frozen classifiers and all four predeclared ID-calibration targets (80/90/95/99%) are reported. Thresholds use only the same 2,500 ID-validation threshold-calibration rows; there is no new fitting. Both phase-2 validation subsets previously selected checkpoints, so empirical calibration coverage is not a deployment guarantee.', '',
             'ID retained error risk = incorrect accepted ID / all accepted ID. OOD acceptance = accepted OOD / all OOD for its own population. These are distinct denominators; OOD classification correctness or an assumed deployment mixture is not defined here. Lower retained risk and OOD acceptance are desirable, alongside the separately measured ID coverage.', '',
             '## All operating points', '',
             'Each entry is mean ± sample SD across training seeds 0/1/2, in percent. The target is empirical ID-calibration acceptance; achieved ID-test coverage is measured separately. Natural OOD macro averages CIFAR-10, MNIST, SVHN and Textures within each seed, with noise excluded.', '',
             '| Method | Target (%) | ID coverage T=1 | ID coverage fitted T | ID retained error T=1 | ID retained error fitted T | Natural OOD accepted T=1 | Natural OOD accepted fitted T |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for target in TARGETS:
            key = f'{target:.2f}'
            i, e, o = id_test[method][key]['coverage'], id_test[method][key]['retained_error_risk'], macro[method][key]
            lines.append(f"| {DISPLAY[method]} | {target * 100:g} | {fmt(i['T1'])} | {fmt(i['calibrated'])} | {fmt(e['T1'])} | {fmt(e['calibrated'])} | {fmt(o['T1'])} | {fmt(o['calibrated'])} |")
    lines += ['', '![ID retained error versus achieved test coverage](id-risk-coverage.png)', '',
              '![Natural OOD acceptance versus achieved test coverage](natural-ood-coverage.png)', '',
              '## Paired calibrated-minus-T=1 changes', '', 'Changes are percentage points, paired within each seed before mean/sample SD. No method or operating target is selected from these OOD observations.', '',
              '| Method | Target (%) | ID coverage delta | ID retained error delta | Correct-ID rejection delta | Incorrect-ID rejection delta | Natural OOD acceptance delta |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for target in TARGETS:
            key = f'{target:.2f}'
            entries = [id_test[method][key][m]['paired_delta_calibrated_minus_T1'] for m in
                       ['coverage', 'retained_error_risk', 'correct_rejection_fraction', 'incorrect_rejection_fraction']]
            entries.append(macro[method][key]['paired_delta_calibrated_minus_T1'])
            lines.append('| ' + DISPLAY[method] + f' | {target * 100:g} | ' + ' | '.join(fmt(v) for v in entries) + ' |')
    lines += ['', '## Per-population OOD acceptance', '',
              'All populations retain their original sizes: 10,000 ID test; CIFAR-10 10,000, MNIST 10,000, SVHN 26,032 and Textures 1,880. AP/population mixture is not used to choose thresholds. Each table entry below is OOD acceptance, percent; target refers only to the fixed ID-calibration subset.', '',
              '| Dataset | Method | Target (%) | T=1 | ID-fitted T | Paired delta (pp) |', '|---|---|---:|---:|---:|---:|']
    for dataset in NATURAL:
        for method in METHODS:
            for target in TARGETS:
                p = outside[dataset][method][f'{target:.2f}']
                lines.append(f"| {dataset} | {DISPLAY[method]} | {target * 100:g} | {fmt(p['T1'])} | {fmt(p['calibrated'])} | {fmt(p['paired_delta_calibrated_minus_T1'])} |")
    lines += ['', '## Saved evidence and audit', '',
              'Per-seed results save exact thresholds, calibration ties/counts, correct/incorrect ID contingency counts, OOD counts/score hashes and every sample-level boolean acceptance mask. [per-seed-counts.csv](per-seed-counts.csv) retains all 720 seed/setting/OOD population rows; repeated ID columns are the same setting, not independent observations. summary.json preserves every seed, paired delta and all natural/synthetic summaries. A zero denominator would remain null with the number of defined seeds reported explicitly; no such undefined endpoint occurred here.', '',
              f"Independent audit: {audit['original_artifact_hash_checks']} original hashes, {audit['independent_count_and_metric_checks']} count/metric checks, {audit['per_sample_decision_masks_checked']} per-sample acceptance masks, {audit['95pct_phase2_parity_checks']} 95%-point parity checks, {audit['unchanged_control_operating_points_checked']} unchanged-control settings and {audit['nested_acceptance_checks']} nesting checks. Python sorting and Counter contingency counts independently reconstruct the production NumPy thresholds/counts.", '',
              'Every 95% point exactly matches the completed phase-2 operational threshold and acceptance. MaxLogit/KNN/Mahalanobis thresholds and decisions are identical across temperature variants. The original baseline and probability-calibration artifacts remain frozen. Natural OOD acceptance is operational validation-threshold acceptance, not benchmark FPR95-ID; achieved ID test coverage need not equal its calibration target.', '',
              'Three-seed sample SD is descriptive, not a confidence interval or significance test. Probability calibration, ID error rejection and natural OOD rejection are separately measured outcomes. Uniform noise remains a separate synthetic diagnostic and never enters a natural macro.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    colors = ['#0072B2', '#D55E00', '#009E73']
    legend = [Line2D([], [], color=c, label=f'seed {s}') for s, c in enumerate(colors)]
    legend += [Line2D([], [], color='gray', linestyle='--', marker='o', label='T=1'),
               Line2D([], [], color='gray', linestyle='-', marker='^', label='ID-fitted T')]
    for filename, endpoint, ylabel, title in [
        ('id-risk-coverage', 'risk', 'Retained ID classification error (%)', 'ID-only operating points: retained error versus achieved ID test coverage'),
        ('natural-ood-coverage', 'ood', 'Equal-weight natural OOD acceptance (%)', 'ID-only operating points: natural OOD acceptance versus achieved ID test coverage')]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
        for method, axis in zip(METHODS, axes.flat):
            for seed, color in enumerate(colors):
                for variant, style, marker in [('T1', '--', 'o'), ('calibrated', '-', '^')]:
                    points = [lookups[seed][(method, variant, target)] for target in TARGETS]
                    xs = [p['id_test']['coverage'] * 100 for p in points]
                    ys = [p['id_test']['retained_error_risk'] * 100 if endpoint == 'risk'
                          else p['equal_weight_natural_ood_acceptance'] * 100 for p in points]
                    axis.plot(xs, ys, color=color, linestyle=style, marker=marker, linewidth=1.4, markersize=4)
            axis.set_title(DISPLAY[method])
            axis.grid(alpha=.2)
            axis.set_xlabel('Achieved ID test coverage (%)')
            axis.set_ylabel(ylabel)
        fig.suptitle(title, fontsize=13)
        fig.legend(handles=legend, loc='lower center', ncol=5, bbox_to_anchor=(.5, .04), frameon=False)
        fig.text(.5, .015, 'Four fixed calibration targets: 80/90/95/99%. Colored curves retain each seed; dashed/solid temperature variants.', ha='center', fontsize=9)
        fig.tight_layout(rect=[0, .095, 1, .95])
        for extension in ['png', 'svg']:
            fig.savefig(OUT / f'{filename}.{extension}', dpi=180)
        plt.close(fig)
    print(json.dumps({'state': summary['state'], 'per_seed_CSV_rows': len(rows), '95pct_MSP': id_test['msp']['0.95'],
                      'MSP_natural_OOD_grid': macro['msp']}), flush=True)


if __name__ == '__main__':
    main()
