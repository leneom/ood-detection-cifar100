"""Build a scientific report from saved scores, without importing PyTorch.

Run using a plotting environment with NumPy and Matplotlib. Bootstrap intervals
reflect test-sample uncertainty for a fixed classifier, not seed variability.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ood.metrics import binary_curve, detection_metrics, id_threshold

NAMES = {'msp': 'MSP', 'energy': 'Energy', 'maxlogit': 'MaxLogit', 'entropy': 'Entropy',
         'knn': 'KNN', 'mahalanobis': 'Mahalanobis'}
COLORS = ['#2764ad', '#db672c', '#2d9560', '#a766be', '#a18a24', '#ba445d']


def auroc(inside, outside):
    labels = np.r_[np.zeros(len(inside)), np.ones(len(outside))]
    fpr, tpr, _ = binary_curve(labels, np.r_[inside, outside])
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def bootstrap(scores, methods, repetitions=300, seed=2026):
    if repetitions < 20:
        raise ValueError('use at least 20 bootstrap repetitions')
    rng = np.random.default_rng(seed)
    n_id, n_ood = len(scores[f'{methods[0]}_id']), len(scores[f'{methods[0]}_ood'])
    areas = np.empty((repetitions, len(methods)))
    fprs = np.empty_like(areas)
    for repeat in range(repetitions):
        id_indices = rng.integers(0, n_id, n_id)
        ood_indices = rng.integers(0, n_ood, n_ood)
        for j, method in enumerate(methods):
            inside, outside = scores[f'{method}_id'][id_indices], scores[f'{method}_ood'][ood_indices]
            areas[repeat, j] = auroc(inside, outside)
            fprs[repeat, j] = np.mean(outside <= id_threshold(inside))
    intervals = {method: {'auroc_ci95': np.quantile(areas[:, j], [.025, .975]).tolist(),
                         'fpr95_ci95': np.quantile(fprs[:, j], [.025, .975]).tolist()}
                 for j, method in enumerate(methods)}
    if 'energy' in methods and 'msp' in methods:
        differences = areas[:, methods.index('energy')] - areas[:, methods.index('msp')]
        intervals['energy_minus_msp'] = {'auroc_difference_ci95': np.quantile(differences, [.025, .975]).tolist()}
    return intervals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bootstrap', type=int, default=300)
    p.add_argument('--label', default='Pilot; fixed classifier')
    args = p.parse_args()
    evaluation, output = Path(args.evaluation), Path(args.output_dir)
    result = json.loads((evaluation / 'results.json').read_text(encoding='utf-8-sig'))
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False})
    markdown = [f'# OOD Detection: {args.label}', '',
                f"ID: {result['id_dataset']}; selected checkpoint epoch {result['checkpoint_epoch']}; "
                f"ID test accuracy **{100 * result['id_test_accuracy']:.2f}%**.", '',
                'All detector scores increase toward OOD. Checkpoint selection uses ID validation accuracy only.', '',
                'This is an independently trained CIFAR study, not a full OpenOOD reproduction. '
                'A pilot or one training seed does not support a multiseed conclusion.', '']
    all_intervals, csv_rows = {}, []
    for dataset, entry in result['datasets'].items():
        methods = list(entry['methods'])
        with np.load(evaluation / f'{dataset}_scores.npz') as archive:
            scores = {key: archive[key] for key in archive.files}
        # Recompute every reported number from the raw saved scores.
        for method in methods:
            recomputed = detection_metrics(scores[f'{method}_id'], scores[f'{method}_ood'], scores[f'{method}_val'])
            for key in ['auroc', 'aupr_out', 'aupr_in', 'fpr95_id', 'validation_threshold']:
                if not np.isclose(recomputed[key], entry['methods'][method][key], atol=1e-10, rtol=1e-10):
                    raise ValueError(f'raw-score verification failed: {dataset}/{method}/{key}')
        intervals = bootstrap(scores, methods, args.bootstrap)
        all_intervals[dataset] = intervals
        markdown += [f'## {dataset.upper()}', '', entry['kind'] + '.', '',
                     '| Method | AUROC % (95% sample CI) | AUPR-out % | FPR95-ID % | ID test accept % (val threshold) | OOD accept % (val threshold) |',
                     '|---|---:|---:|---:|---:|---:|']
        for method in methods:
            values = entry['methods'][method]
            low, high = np.asarray(intervals[method]['auroc_ci95']) * 100
            markdown.append(f"| {NAMES[method]} | {values['auroc']*100:.2f} [{low:.2f}, {high:.2f}] | "
                            f"{values['aupr_out']*100:.2f} | {values['fpr95_id']*100:.2f} | "
                            f"{values['id_test_acceptance_at_validation_threshold']*100:.2f} | "
                            f"{values['ood_acceptance_at_validation_threshold']*100:.2f} |")
            csv_rows.append({'dataset': dataset, 'method': method, **values})
        if 'energy_minus_msp' in intervals:
            delta = (entry['methods']['energy']['auroc'] - entry['methods']['msp']['auroc']) * 100
            low, high = np.asarray(intervals['energy_minus_msp']['auroc_difference_ci95']) * 100
            markdown += ['', f'Energy − MSP AUROC difference: {delta:.2f} percentage points '
                          f'(paired sample-bootstrap 95% CI [{low:.2f}, {high:.2f}]).']
        markdown += ['', f"ID count: {entry['methods'][methods[0]]['n_id']}; "
                     f"OOD count: {entry['methods'][methods[0]]['n_ood']}. "
                     'AUROC/AUPR higher is better; FPR95-ID and OOD acceptance lower is better.', '']
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        for j, method in enumerate(methods):
            inside, outside = scores[f'{method}_id'], scores[f'{method}_ood']
            labels = np.r_[np.zeros(len(inside)), np.ones(len(outside))]
            fpr, tpr, _ = binary_curve(labels, np.r_[inside, outside])
            ax.plot(fpr, tpr, color=COLORS[j % len(COLORS)], lw=1.5,
                    label=f"{NAMES[method]} ({entry['methods'][method]['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], '--', color='#999999', lw=1)
        ax.set(xlabel='ID false rejection rate (OOD-positive ROC)', ylabel='OOD detection rate',
               title=f'CIFAR-100 vs {dataset.upper()} | {args.label}', xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f'{dataset}-roc.png')
        fig.savefig(output / f'{dataset}-roc.svg')
        plt.close(fig)
        rows = (len(methods) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(12, 3.4 * rows), squeeze=False)
        for ax, method in zip(axes.flat, methods):
            for population, color in [('id', '#2764ad'), ('ood', '#db672c')]:
                values = np.sort(scores[f'{method}_{population}'])
                ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, lw=1.4, label=population.upper())
            ax.axvline(entry['methods'][method]['validation_threshold'], color='#666666', ls='--', lw=1, label='Val 95% threshold')
            ax.set(title=NAMES[method], xlabel='Anomaly score (higher = OOD)', ylabel='Empirical CDF', ylim=(0, 1))
            ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(methods):]:
            ax.set_visible(False)
        fig.suptitle(f'CIFAR-100 vs {dataset.upper()} | Score distributions', y=1.01)
        fig.tight_layout()
        fig.savefig(output / f'{dataset}-score-cdf.png', bbox_inches='tight')
        plt.close(fig)
        markdown += [f'![{dataset} ROC]({dataset}-roc.png)', '', f'![{dataset} empirical score distributions]({dataset}-score-cdf.png)', '']
    markdown += ['## Interpretation boundary', '',
                 'FPR95-ID is the OOD acceptance rate at the empirical 95% ID-test operating point. '
                 'The separately reported validation-calibrated operating point uses no test values to fit its threshold. '
                 'High AUROC alone does not imply useful rejection at high ID acceptance.', '',
                 f'Intervals use {args.bootstrap} stratified paired bootstrap draws of ID/OOD test samples, fixed seed 2026. '
                 'These approximate percentile intervals describe sample uncertainty conditional on this trained classifier. '
                 'They do not estimate uncertainty from retraining or replace training-seed mean/sample SD.', '',
                 'Uniform noise is a diagnostic, not a natural OOD benchmark. Scores below chance are retained with '
                 'their predeclared orientation; they are not flipped after observing test performance.', '',
                 f"Checkpoint SHA-256: `{result['checkpoint_sha256']}`.", '']
    (output / 'REPORT.md').write_text('\n'.join(markdown), encoding='utf-8')
    (output / 'bootstrap.json').write_text(json.dumps({'repetitions': args.bootstrap, 'seed': 2026, 'datasets': all_intervals}, indent=2), encoding='utf-8')
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
