"""Report all fixed endpoints from audited saved phase-2 outputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/calibration-phase2'
OUT = ROOT / 'report/calibration-phase2'
METHODS = ['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis']
NATURAL = ['cifar10', 'mnist', 'svhn', 'textures']
DATASETS = NATURAL + ['noise']
ENDPOINTS = ['auroc', 'aupr_out', 'aupr_in', 'fpr95_id', 'id_tpr_at_fpr95',
             'id_test_acceptance_at_validation_threshold', 'ood_acceptance_at_validation_threshold']
DISPLAY = {'msp': 'MSP', 'energy': 'Energy', 'maxlogit': 'MaxLogit', 'entropy': 'Entropy',
           'knn': 'KNN', 'mahalanobis': 'Mahalanobis'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {'mean': float(x.mean()), 'sample_sd': float(x.std(ddof=1)), 'per_seed': x.tolist(), 'n_seeds': 3}


def paired(before, after):
    a, b = np.asarray(before), np.asarray(after)
    return {'T1': stats(a), 'calibrated': stats(b), 'paired_delta_calibrated_minus_T1': stats(b - a)}


def fmt(record, scale=100):
    return f"{record['mean'] * scale:.2f} ± {record['sample_sd'] * scale:.2f}"


def paired_plot(axis, record, label, scale, colors):
    for seed, color in enumerate(colors):
        axis.plot([0, 1], [record['T1']['per_seed'][seed] * scale, record['calibrated']['per_seed'][seed] * scale],
                  marker='o', color=color, linewidth=1.5, label=f'seed {seed}')
    axis.errorbar([-.08, 1.08], [record['T1']['mean'] * scale, record['calibrated']['mean'] * scale],
                  yerr=[record['T1']['sample_sd'] * scale, record['calibrated']['sample_sd'] * scale],
                  color='black', fmt='s', capsize=4, label='mean ± sample SD', markersize=5)
    axis.set_xticks([0, 1], ['T = 1', 'ID-fitted T'])
    axis.set_xlim(-.35, 1.35)
    axis.set_ylabel(label)
    axis.grid(axis='y', alpha=.25)


def main():
    audit = read(OUT / 'audit.json')
    if audit['state'] != 'passed' or audit['inputs_sha256'] != digest(RUN / 'inputs.json'):
        raise ValueError('missing or stale independent calibration audit')
    results = [read(RUN / f'seed{s}/results.json') for s in [0, 1, 2]]
    for seed in [0, 1, 2]:
        if digest(RUN / f'seed{seed}/results.json') != audit['results_sha256'][f'seed{seed}']:
            raise ValueError('audited seed result changed')
    primary = {metric: paired([r['id_test']['T1'][metric] for r in results],
                              [r['id_test']['calibrated'][metric] for r in results])
               for metric in ['nll', 'brier', 'ece15', 'accuracy', 'mean_confidence']}
    per_dataset = {}
    for dataset in DATASETS:
        per_dataset[dataset] = {}
        for method in METHODS:
            per_dataset[dataset][method] = {}
            for metric in ENDPOINTS:
                per_dataset[dataset][method][metric] = paired(
                    [r['datasets'][dataset]['variants']['T1'][method][metric] for r in results],
                    [r['datasets'][dataset]['variants']['calibrated'][method][metric] for r in results])
    macro = {}
    for method in METHODS:
        macro[method] = {}
        for metric in ENDPOINTS:
            before = [np.mean([r['datasets'][d]['variants']['T1'][method][metric] for d in NATURAL]) for r in results]
            after = [np.mean([r['datasets'][d]['variants']['calibrated'][method][metric] for d in NATURAL]) for r in results]
            macro[method][metric] = paired(before, after)
    precision_summary = {}
    for method in METHODS:
        precision_summary[method] = {'max_absolute_score_difference': max(
            r['precision_and_original_thresholds'][d][method]['max_abs_score_difference'][part]
            for r in results for d in DATASETS for part in ['id', 'val', 'ood']),
            'max_absolute_AUROC_difference': max(abs(r['precision_and_original_thresholds'][d][method]['metric_delta_T1_minus_original']['auroc'])
                                               for r in results for d in DATASETS),
            'max_absolute_FPR95_difference': max(abs(r['precision_and_original_thresholds'][d][method]['metric_delta_T1_minus_original']['fpr95_id'])
                                                for r in results for d in DATASETS)}
    summary = {'state': 'complete_audited_phase2', 'seeds': [0, 1, 2], 'natural_datasets': NATURAL,
               'synthetic_diagnostics': ['noise'], 'temperature': stats([r['temperature_fit']['temperature'] for r in results]),
               'primary_ID_test': primary, 'per_dataset': per_dataset, 'equal_weight_natural_macro': macro,
               'original_baseline_precision_controls': precision_summary,
               'counts': {d: {'n_id': results[0]['datasets'][d]['variants']['T1']['msp']['n_id'],
                             'n_ood': results[0]['datasets'][d]['variants']['T1']['msp']['n_ood'],
                             'ood_prevalence': results[0]['datasets'][d]['variants']['T1']['msp']['ood_prevalence']} for d in DATASETS},
               'inputs_sha256': audit['inputs_sha256'], 'result_sha256': audit['results_sha256'],
               'raw_audit_sha256': digest(OUT / 'audit.json'), 'report_source_sha256': digest(__file__),
               'created_at': datetime.now(timezone.utc).isoformat(),
               'boundary': 'matched 2500-row ID threshold calibration; descriptive paired seed SD; no confidence interval/significance/method selection; noise excluded from macro'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# ID-only temperature calibration: complete three-seed phase 2', '',
             'Original classifiers, baseline metrics and raw artifacts remain frozen. One temperature per seed was fitted on 2,500 ID validation rows. The disjoint 2,500 remaining validation rows select all operational thresholds. Both subsets previously served checkpoint selection: they are not entirely fresh holdouts, and no conformal coverage is claimed.', '',
             'All classical T=1 and calibrated scores use matched CPU float64 arithmetic. MaxLogit and saved KNN/Mahalanobis scores and thresholds are unchanged controls; features are never refitted. OOD endpoints are descriptive and never select a temperature, method or checkpoint.', '',
             '## Fixed temperatures', '', '| Seed | Selected epoch | T | Fit NLL T=1 | Fit NLL calibrated | Optimizer |',
             '|---|---:|---:|---:|---:|---|']
    for r in results:
        f = r['temperature_fit']
        lines.append(f"| {r['seed']} | {r['checkpoint_epoch']} | {f['temperature']:.6f} | {f['fit_nll_T1']:.6f} | {f['selected_fit_nll']:.6f} | {'success' if f['success'] else 'fallback'} |")
    lines += ['', '## Primary ID-test endpoints', '', 'Mean ± sample SD across seeds 0, 1, 2. Delta is paired calibrated minus T=1, computed within each seed. Brier is the unnormalized sum over all classes; ECE uses 15 fixed equal-width confidence bins.', '',
              '| Endpoint | T=1 | ID-fitted T | Paired delta |', '|---|---:|---:|---:|']
    for metric, label, scale in [('nll', 'NLL', 1), ('brier', 'Brier', 1), ('ece15', 'ECE (%) / delta pp', 100),
                                 ('accuracy', 'Accuracy (%) / delta pp', 100)]:
        record = primary[metric]
        def primary_fmt(x):
            return f"{x['mean'] * scale:.4f} ± {x['sample_sd'] * scale:.4f}"
        lines.append(f"| {label} | {primary_fmt(record['T1'])} | {primary_fmt(record['calibrated'])} | {primary_fmt(record['paired_delta_calibrated_minus_T1'])} |")
    lines += ['', '![ID-test paired calibration](id-calibration.png)', '',
              '## Secondary OOD endpoints', '',
              'AUROC and benchmark FPR95-ID are percentages; paired changes are percentage points. FPR95-ID uses an ID-test benchmark threshold and remains separate from the operational ID-validation threshold. Lower FPR is better. Four natural populations retain their original sizes/prevalences; AP is available in summary.json and per-seed results.', '',
              '| Dataset | Method | AUROC T=1 | AUROC fitted T | Delta AUROC | FPR95 T=1 | FPR95 fitted T | Delta FPR95 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in NATURAL:
        for method in METHODS:
            a, f = per_dataset[dataset][method]['auroc'], per_dataset[dataset][method]['fpr95_id']
            lines.append(f"| {dataset} | {DISPLAY[method]} | {fmt(a['T1'])} | {fmt(a['calibrated'])} | {fmt(a['paired_delta_calibrated_minus_T1'])} | {fmt(f['T1'])} | {fmt(f['calibrated'])} | {fmt(f['paired_delta_calibrated_minus_T1'])} |")
    lines += ['', '## Equal-weight natural macro', '', 'Average the four natural datasets within each seed, then summarize the three seeds. Noise is excluded.', '',
              '| Method | AUROC T=1 | AUROC fitted T | Paired AUROC delta | FPR95 T=1 | FPR95 fitted T | Paired FPR95 delta |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        a, f = macro[method]['auroc'], macro[method]['fpr95_id']
        lines.append(f"| {DISPLAY[method]} | {fmt(a['T1'])} | {fmt(a['calibrated'])} | {fmt(a['paired_delta_calibrated_minus_T1'])} | {fmt(f['T1'])} | {fmt(f['calibrated'])} | {fmt(f['paired_delta_calibrated_minus_T1'])} |")
    lines += ['', '![Natural macro paired calibration effects](ood-calibration.png)', '',
              '## Operational 2,500-row ID-validation thresholds', '',
              'Thresholds select at least 95% empirical acceptance on their fixed ID-only calibration subset. Achieved ID-test acceptance is measured, not guaranteed; each operational threshold is applied unchanged to OOD. Original 5,000-row baseline thresholds remain in per-seed precision_and_original_thresholds records.', '',
              '| Dataset | Method | ID accepted T=1 (%) | ID accepted fitted T (%) | OOD accepted T=1 (%) | OOD accepted fitted T (%) |',
              '|---|---|---:|---:|---:|---:|']
    for dataset in NATURAL:
        for method in METHODS:
            i = per_dataset[dataset][method]['id_test_acceptance_at_validation_threshold']
            o = per_dataset[dataset][method]['ood_acceptance_at_validation_threshold']
            lines.append(f"| {dataset} | {DISPLAY[method]} | {fmt(i['T1'])} | {fmt(i['calibrated'])} | {fmt(o['T1'])} | {fmt(o['calibrated'])} |")
    lines += ['', '## Precision and independent verification', '',
              'The original GPU float32 baseline is not overwritten. Phase-2 T=1 uses float64 for classical methods and canonical original eval-final ID feature scores for all populations. Extension ID logits were verified within rtol=atol=1e-6; saved Mahalanobis ID scores within the already frozen extension tolerance rtol=1e-4, atol=1e-6. These differences are logged explicitly, alongside the original full-validation threshold metrics.', '',
              '| Method | Max absolute score difference | Max absolute AUROC difference | Max absolute FPR95 difference |',
              '|---|---:|---:|---:|']
    for method in METHODS:
        p = precision_summary[method]
        lines.append(f"| {DISPLAY[method]} | {p['max_absolute_score_difference']:.6g} | {p['max_absolute_AUROC_difference']:.6g} | {p['max_absolute_FPR95_difference']:.6g} |")
    lines += ['', f"Independent audit: {audit['original_artifact_hash_checks']} original artifact hashes, {audit['independent_AUROC_AP_checks']} sklearn AUROC/AP checks, {audit['raw_score_formula_control_checks']} raw-score formula/control checks, {audit['threshold_count_prevalence_checks']} threshold/count/prevalence checks, {audit['probability_endpoint_checks']} probability endpoint checks and {audit['precision_and_original_threshold_checks']} precision/original-threshold checks. A separate convex inverse-temperature root calculation verifies all three fitting optima.", '',
              'Three-seed sample SD describes variation across these classifiers; it is not a confidence interval or significance test. Probability calibration, score ranking and high-ID-acceptance OOD rejection are separate endpoints. Synthetic noise outputs are retained only as diagnostics in summary.json. No method is promoted using OOD test results.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    colors = ['#0072B2', '#D55E00', '#009E73']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for axis, metric, label, scale in zip(axes, ['nll', 'brier', 'ece15'], ['NLL (lower better)', 'Multiclass Brier (lower better)', 'ECE, 15 bins (%) (lower better)'], [1, 1, 100]):
        paired_plot(axis, primary[metric], label, scale, colors)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, .05), frameon=False)
    fig.suptitle('ID-only temperature fitting: unchanged classifiers, 10,000 ID test images')
    fig.text(.5, .015, 'Lines connect the same seed; black squares show mean ± sample SD (not a confidence interval).', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .13, 1, .93])
    for suffix in ['png', 'svg']:
        fig.savefig(OUT / f'id-calibration.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, metric, label in zip(axes, ['auroc', 'fpr95_id'], ['AUROC (%) (higher better)', 'FPR95-ID (%) (lower better)']):
        for index, method in enumerate(METHODS):
            record = macro[method][metric]
            for seed, color in enumerate(colors):
                offset = (seed - 1) * .055
                xs = [index - .18 + offset, index + .18 + offset]
                ys = [record['T1']['per_seed'][seed] * 100, record['calibrated']['per_seed'][seed] * 100]
                axis.plot(xs, ys, color=color, linewidth=1.2)
                axis.scatter(xs[0], ys[0], color=color, marker='o', s=24)
                axis.scatter(xs[1], ys[1], color=color, marker='^', s=28)
            for variant, xpos in [('T1', index - .24), ('calibrated', index + .24)]:
                axis.errorbar(xpos, record[variant]['mean'] * 100, yerr=record[variant]['sample_sd'] * 100,
                              fmt='s', color='black', capsize=3, markersize=4, linewidth=1)
        axis.set_xticks(range(6), [DISPLAY[m] for m in METHODS], rotation=20, ha='right')
        axis.set_ylabel(label)
        axis.grid(axis='y', alpha=.25)
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color=c, marker='o', label=f'seed {i}') for i, c in enumerate(colors)]
    legend += [Line2D([], [], color='gray', marker='o', linestyle='', label='T=1 (left)'),
               Line2D([], [], color='gray', marker='^', linestyle='', label='ID-fitted T (right)'),
               Line2D([], [], color='black', marker='s', linestyle='', label='mean ± sample SD')]
    fig.legend(handles=legend, loc='lower center', ncol=3, bbox_to_anchor=(.5, .035), frameon=False)
    fig.suptitle('Secondary OOD endpoints: equal-weight macro of four fixed natural populations')
    fig.text(.5, .012, 'Paired seeds; OOD never selects T. SD is descriptive. MaxLogit/KNN/Mahalanobis remain unchanged controls.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .20, 1, .93])
    for suffix in ['png', 'svg']:
        fig.savefig(OUT / f'ood-calibration.{suffix}', dpi=180)
    plt.close(fig)
    print(json.dumps({'state': summary['state'], 'temperature': summary['temperature'], 'primary_ID_test': primary,
                      'natural_macro_changed_methods': {m: {k: macro[m][k]['paired_delta_calibrated_minus_T1'] for k in ['auroc', 'fpr95_id']}
                                                        for m in ['msp', 'energy', 'entropy']}}), flush=True)


if __name__ == '__main__':
    main()
