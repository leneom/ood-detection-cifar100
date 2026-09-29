"""All-method paired seed summaries and scientific plots after the raw audit."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_io import ROOT, RUN, METHODS, ADDITIONS, DATASETS, NATURAL, digest, read, write, require, now

RUN = ROOT / 'runs/ash-phase5'
OUT = ROOT / 'report/ash-phase5'
REFERENCES = METHODS + ADDITIONS + ['rmd','mahalanobis_cpu']
NEW = ['ash_p_energy','ash_b_energy','ash_s_energy']
ALL = REFERENCES+NEW
TARGETS = [.8, .9, .95, .99]
ENDPOINTS = ['auroc', 'aupr_out', 'aupr_in', 'fpr95_id', 'id_test_acceptance_at_validation_threshold', 'ood_acceptance_at_validation_threshold']
LABELS = dict(msp='MSP', energy='Energy', maxlogit='MaxLogit', entropy='Entropy', knn='KNN', mahalanobis='Mahalanobis',
              vim='ViM', residual='Residual', react_energy='ReAct Energy', affine_energy='Affine Energy*', rmd='RMD', mahalanobis_cpu='CPU MD*',ash_p_energy='ASH-P',ash_b_energy='ASH-B',ash_s_energy='ASH-S')


def stats(values):
    a = np.asarray(values, dtype=float)
    require(a.shape == (3,) and np.isfinite(a).all(), 'summary needs exactly three finite seeds')
    return {'per_seed': a.tolist(), 'mean': float(a.mean()), 'sample_sd': float(a.std(ddof=1))}


def main():
    audit = read(OUT / 'audit.json')
    require(audit['state'] == 'passed', 'raw evaluation audit missing')
    for name, expected in audit['output_artifacts'].items():
        require(digest(ROOT / name) == expected, 'audited raw artifact changed')
    attempt = audit['output_name']
    results = [read(RUN / f'seed{s}/{attempt}/results.json') for s in [0, 1, 2]]
    fits = [read(RUN / f'seed{s}/preflight.json') for s in [0, 1, 2]]
    per_dataset = {}
    for dataset in DATASETS:
        per_dataset[dataset] = {}
        for method in ALL:
            per_dataset[dataset][method] = {}
            for block in ['primary_2500', 'secondary_5000_threshold']:
                metrics = results[0]['datasets'][dataset]['methods'][method][block]
                per_dataset[dataset][method][block] = {
                    metric: stats([r['datasets'][dataset]['methods'][method][block][metric] for r in results]) for metric in metrics}
    macro = {m: {metric: stats([np.mean([r['datasets'][d]['methods'][m]['primary_2500'][metric] for d in NATURAL])
                              for r in results]) for metric in ENDPOINTS} for m in ALL}
    paired = {}
    for method in NEW:
        paired[method] = {}
        for reference in REFERENCES:
            paired[method][reference] = {}
            for dataset in DATASETS + ['natural_macro']:
                paired[method][reference][dataset] = {}
                for metric in ENDPOINTS:
                    if dataset == 'natural_macro':
                        a, b = macro[method][metric]['per_seed'], macro[reference][metric]['per_seed']
                    else:
                        a = per_dataset[dataset][method]['primary_2500'][metric]['per_seed']
                        b = per_dataset[dataset][reference]['primary_2500'][metric]['per_seed']
                    paired[method][reference][dataset][metric] = stats(np.asarray(a) - np.asarray(b))
    records = [{(x['method'], x['target']): x for x in r['operating_points']} for r in results]
    grid = {}
    for method in ALL:
        grid[method] = {}
        for target in TARGETS:
            points = [p[(method, target)] for p in records]
            endpoints = {f'id_{key}': stats([p['id_test'][key] for p in points]) for key in
                         ['coverage', 'retained_error_risk', 'correct_rejection_fraction', 'incorrect_rejection_fraction']}
            endpoints['calibration_achieved_acceptance'] = stats([p['calibration']['achieved_acceptance'] for p in points])
            endpoints['natural_macro_ood_acceptance'] = stats([p['equal_weight_natural_ood_acceptance'] for p in points])
            endpoints.update({f'{d}_ood_acceptance': stats([p['ood'][d]['acceptance'] for p in points]) for d in DATASETS})
            grid[method][str(target)] = endpoints
    fit_summary = {k: stats([fit[k] for fit in fits]) for k in ['training_rows','feature_dimensions','class_count','percentile','k']}
    pop_diag={population:{key:stats([r['intermediate_diagnostics'][population][key] for r in results]) for key in results[0]['intermediate_diagnostics'][population]} for population in results[0]['intermediate_diagnostics']}
    id_diag=pop_diag['id']
    summary = {'state': 'computed_pending_independent_summary_audit', 'created_at': now(), 'expected_seeds': [0, 1, 2],
               'methods': ALL, 'named_additions': NEW, 'numeric_controls': ['affine_energy','mahalanobis_cpu'],
               'natural_datasets': NATURAL, 'synthetic_diagnostics': ['noise'], 'threshold_calibration_n': 2500,
               'per_dataset': per_dataset, 'natural_macro': macro, 'paired_changes': paired, 'operating_grid': grid,
               'preflight_statistics': fit_summary, 'population_intermediate_diagnostics':pop_diag, 'ID_intermediate_diagnostics': id_diag, 'raw_audit_sha256': digest(OUT / 'audit.json'),
               'result_sha256': [digest(RUN / f'seed{s}/{attempt}/results.json') for s in [0, 1, 2]],
               'source_sha256': digest(__file__), 'boundary': 'fixed exploratory extension; seed SD is descriptive, not CI; all methods reported'}
    write(OUT / 'summary.json', summary)
    def pct(stat):
        return f"{100*stat['mean']:.2f} ± {100*stat['sample_sd']:.2f}"
    lines = ['# ASH-P/B/S: complete three-seed frozen evaluation', '',
             'All three frozen CIFAR-100 classifiers are evaluated on the original four natural OOD datasets. ASH-P/B/S use one fixed p=65 (k=179 out of 512 final pooled coordinates), following the inspected author function defaults. Stable descending values and ascending original coordinate ties are an explicit local adaptation. P retains values, B fills selected coordinates with original_sum/k, and S uses the exponential exp(original_sum/retained_sum). No model/detector statistic, percentile or temperature is fitted; independent training-ID-only preflight precedes OOD scoring. All twelve previous columns are unchanged; starred columns are numerical controls.', '',
             'Values below are mean ± sample SD across seeds 0/1/2, in percent or percentage points. SD describes these three training seeds; it is not a confidence interval. Every method is reported in protocol order, without test-selected settings or a promoted best method.', '',
             'Primary operating thresholds use the same 2,500 ID-validation calibration rows at 95% empirical acceptance. These images previously selected checkpoints, so this is reused validation, not a fresh holdout or a coverage guarantee. FPR95-ID uses a separate ID-test benchmark threshold. Four-dataset macros weight datasets equally within each seed; noise is excluded.', '',
             '| Method | Natural macro AUROC | Benchmark FPR95-ID ↓ | Primary OOD acceptance ↓ | ID-test coverage | Retained ID error ↓ |',
             '|---|---:|---:|---:|---:|---:|']
    for method in ALL:
        g = grid[method]['0.95']
        lines.append(f"| {LABELS[method]} | {pct(macro[method]['auroc'])} | {pct(macro[method]['fpr95_id'])} | {pct(macro[method]['ood_acceptance_at_validation_threshold'])} | {pct(g['id_coverage'])} | {pct(g['id_retained_error_risk'])} |")
    lines += ['', '## Separate natural populations', '']
    for dataset in NATURAL:
        n = results[0]['datasets'][dataset]['methods']['msp']['primary_2500']['n_ood']
        lines += [f'### {dataset}: ID 10,000; OOD {n:,}', '',
                  '| Method | AUROC ↑ | AUPR-out ↑ | AUPR-in ↑ | FPR95-ID ↓ | Primary OOD acceptance ↓ |',
                  '|---|---:|---:|---:|---:|---:|']
        for method in ALL:
            record = per_dataset[dataset][method]['primary_2500']
            lines.append('| ' + LABELS[method] + ' | ' + ' | '.join(pct(record[k]) for k in ['auroc','aupr_out','aupr_in','fpr95_id','ood_acceptance_at_validation_threshold']) + ' |')
        lines += ['', 'AUPR depends on this explicit ID/OOD population ratio; comparisons across datasets do not share prevalence.', '']
    lines += ['## Paired natural macro changes', '',
              'Each difference is calculated within the same seed before mean/sample SD. Positive AUROC is higher; negative acceptance/FPR is lower. Full per-dataset and per-seed differences against all twelve fixed references are in summary.json.', '',
              '| Method minus reference | AUROC change (pp) | FPR95-ID change (pp) | Primary OOD acceptance change (pp) |',
              '|---|---:|---:|---:|']
    for method in NEW:
        for reference in REFERENCES:
            p = paired[method][reference]['natural_macro']
            lines.append(f"| {LABELS[method]} − {LABELS[reference]} | {pct(p['auroc'])} | {pct(p['fpr95_id'])} | {pct(p['ood_acceptance_at_validation_threshold'])} |")
    lines += ['', '## Classification and fitting boundaries', '',
              f"Original classifier ID-test accuracy: {pct(id_diag['original_accuracy'])}%. All ID risk/coverage endpoints use these unchanged original predictions. Shaped intermediate predictions can change and are reported separately below.", '',
              'Training-ID numerical preflight checked every one of the 45,000 rows per seed with independent CPU torch support/shaping/linear logits and fixed manual sort/math rows. No covariance, percentile, temperature or scale coefficient was learned from these checks. ASH-B preserves activation sums; ASH-S uses the actual exponential and rejects degenerate/nonfinite sums. The declared float64 audit tolerances are atol=1e-8, rtol=1e-10.', '',
              '| Population | Original accuracy % | Variant | Shaped accuracy % | Argmax changes per seed |',
              '|---|---:|---|---:|---|']
    for population in ['id','validation']:
        diag=pop_diag[population]
        for method in NEW:
            lines.append(f"| {population} | {pct(diag['original_accuracy'])} | {LABELS[method]} | {pct(diag[method+'_accuracy'])} | {diag[method+'_argmax_changes']['per_seed']} |")
    lines += ['', 'P-versus-S argmax differences (original bias retained) for ID test/validation: '+str(pop_diag['id']['P_S_argmax_disagreements']['per_seed'])+' / '+str(pop_diag['validation']['P_S_argmax_disagreements']['per_seed'])+'.', '',
              'Synthetic noise is retained separately and excluded from every natural macro. This is a fixed exploratory adaptation on our ResNet18 after earlier OOD results were observed, not a reproduction of the authors’ DenseNet/ImageNet benchmark or their best parameter search. All three variants are reported without a selected winner. No OOD test selected a checkpoint, percentile, score direction, temperature or target.', '',
              '## Scientific figures', '', '![Natural AUROC with all seeds](natural-auroc.png)', '',
              '![Fixed ID calibration grid with separate seed panels](operating-grid.png)', '',
              '## Reproduction and evidence', '',
              'Run with the verified base CPU Python; set OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS to 4 for bounded local computation. For torch CPU auditing on Windows, CUDA_VISIBLE_DEVICES=-1 was verified to hide CUDA; an empty value was ineffective in the observed shell. Historical frozen reports remain unchanged.', '',
              '```powershell', '& C:/Users/zxzx/anaconda3/python.exe tools/audit_ash_evaluation.py',
              '& C:/Users/zxzx/anaconda3/python.exe tools/build_ash_report.py',
              '& C:/Users/zxzx/anaconda3/python.exe tools/audit_ash_report.py', '```', '',
              'Raw: original banks are runs/geometry-phase3/seedN/training-bank.npz; training preflight and new evaluation are runs/ash-phase5/seedN/preflight.npz, evaluation/*_scores.npz, *_intermediates.npz and decisions.npz. Full shaped features are exactly reconstructible from saved selected indices/values, sums/scales and frozen inputs. Provenance and independent preflight audit: runs/ash-phase5/*.json. This report audit.json binds all raw artifacts; summary-audit.json independently checks every statistical record. Completion is recorded separately after both audits and visual inspection.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    colors = ['#006d77', '#e29526', '#9b3a62']
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharex=True, constrained_layout=True)
    x = np.arange(len(ALL))
    for axis, dataset in zip(axes.flat, NATURAL):
        means = [per_dataset[dataset][m]['primary_2500']['auroc']['mean']*100 for m in ALL]
        sd = [per_dataset[dataset][m]['primary_2500']['auroc']['sample_sd']*100 for m in ALL]
        axis.errorbar(x, means, yerr=sd, fmt='none', ecolor='#555555', capsize=4, lw=1.2, label='mean ± sample SD')
        for seed, color in enumerate(colors):
            axis.scatter(x+(seed-1)*.13, [per_dataset[dataset][m]['primary_2500']['auroc']['per_seed'][seed]*100 for m in ALL],
                         s=26, color=color, label=f'seed {seed}', zorder=3)
        axis.set(title=dataset, ylabel='AUROC (%)', ylim=(0, 101))
        axis.set_xticks(x, [LABELS[m] for m in ALL], rotation=48, ha='right')
        axis.axhline(50, ls=':', color='gray', lw=1)
        axis.grid(axis='y', alpha=.2)
    axes[0,0].legend(fontsize=9, ncols=2, loc='lower left')
    fig.suptitle('Natural OOD: all frozen methods and training seeds\nAsterisks identify numerical controls; no OOD-selected settings', fontsize=15)
    for suffix in ['png','svg']:
        fig.savefig(OUT / f'natural-auroc.{suffix}', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    palette = plt.get_cmap('tab20').colors
    for seed in [0, 1, 2]:
        for mi, method in enumerate(ALL):
            points = [records[seed][(method,t)] for t in TARGETS]
            xx = [p['id_test']['coverage']*100 for p in points]
            axes[0,seed].plot(xx, [p['id_test']['retained_error_risk']*100 for p in points], 'o-',
                             color=palette[mi], lw=1.3, ms=3.5, label=LABELS[method])
            axes[1,seed].plot(xx, [p['equal_weight_natural_ood_acceptance']*100 for p in points], 'o-',
                             color=palette[mi], lw=1.3, ms=3.5)
        axes[0,seed].set(title=f'seed {seed}', ylabel='Retained ID error (%)', xlabel='Achieved ID-test coverage (%)')
        axes[1,seed].set(ylabel='Natural macro OOD accepted (%)', xlabel='Achieved ID-test coverage (%)', ylim=(0,101))
        for ax in axes[:,seed]:
            ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc='outside lower center', ncols=5, fontsize=9)
    fig.suptitle('Fixed ASH-P/B/S, separate per-seed risk/coverage and OOD acceptance\nPoints use ID-calibration targets 80/90/95/99%; correctness is from the original classifier', fontsize=14)
    for suffix in ['png','svg']:
        fig.savefig(OUT / f'operating-grid.{suffix}', dpi=160)
    plt.close(fig)
    print(__import__('json').dumps({'state': summary['state'], 'natural_macro': {m: {k: macro[m][k] for k in ['auroc','fpr95_id','ood_acceptance_at_validation_threshold']} for m in ALL}}), flush=True)


if __name__ == '__main__':
    main()
