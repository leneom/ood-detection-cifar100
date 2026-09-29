"""Plot audited per-seed values and mean/sample SD; never fit or rank methods."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'msp': 'MSP', 'energy': 'Energy', 'maxlogit': 'MaxLogit',
          'entropy': 'Entropy', 'knn': 'KNN', 'mahalanobis': 'Mahalanobis'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    if audit.get('status') != 'passed' or audit.get('summary_sha256') != digest(args.summary):
        raise ValueError('a passed independent audit must bind the exact plotted summary')
    if summary['seeds'] != [0, 1, 2] or audit['seeds'] != summary['seeds']:
        raise ValueError('all three predeclared seeds are required')
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('figure output must be empty; existing artifacts are retained')
    output.mkdir(parents=True, exist_ok=True)
    datasets = list(summary['datasets'])
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.5, 3.05 * len(datasets)), squeeze=False)
    colors = ['#2764ad', '#d77720', '#30854c']
    marks = ['o', 's', '^']
    plotted = []
    for row, dataset in enumerate(datasets):
        methods = list(summary['datasets'][dataset])
        x = np.arange(len(methods), dtype=float)
        for column, metric in enumerate(('auroc', 'fpr95_id')):
            ax = axes[row, column]
            records = [summary['datasets'][dataset][method][metric] for method in methods]
            values = np.asarray([record['per_seed'] for record in records], dtype=float) * 100
            means = np.asarray([record['mean'] for record in records]) * 100
            sd = np.asarray([record['sample_sd'] for record in records]) * 100
            if values.shape != (len(methods), 3) or not np.isfinite(values).all() or not np.isfinite(sd).all():
                raise ValueError('invalid plotted seed population')
            for index, seed in enumerate(summary['seeds']):
                ax.scatter(x + (index - 1) * .14 - .07, values[:, index], s=32,
                           color=colors[index], marker=marks[index], label=f'Seed {seed}', zorder=3)
            ax.errorbar(x + .30, means, yerr=sd, color='#242424', fmt='D', ms=4,
                        capsize=3, lw=1.3, label='Mean ± sample SD', zorder=4)
            ax.set_xticks(x + .06, [LABELS.get(method, method) for method in methods], rotation=18)
            low, high = float((means - sd).min()), float((means + sd).max())
            ax.set_ylim(min(-2., low - 3), max(102., high + 3))
            ax.set_xlim(-.55, len(methods) - .4)
            ax.set_ylabel('AUROC (%)' if metric == 'auroc' else 'FPR95-ID / OOD acceptance (%)')
            kind = 'synthetic diagnostic' if dataset in ('noise', 'constant') else 'natural OOD'
            direction = 'higher is better' if metric == 'auroc' else 'lower is better'
            ax.set_title(f'{dataset.upper()} — {kind}; {direction}')
            ax.grid(axis='y', alpha=.2, zorder=0)
            if metric == 'auroc':
                ax.axhline(50, color='#999999', ls=':', lw=.9)
            plotted.append({'dataset': dataset, 'metric': metric, 'methods': methods,
                            'per_seed_percent': values.tolist(), 'mean_percent': means.tolist(),
                            'sample_sd_pp': sd.tolist()})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, .985))
    fig.suptitle('Frozen CIFAR-100 classifiers: all three training seeds', y=1.01, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / 'seed-comparison.png', bbox_inches='tight')
    fig.savefig(output / 'seed-comparison.svg', bbox_inches='tight')
    plt.close(fig)
    provenance = {'summary_sha256': digest(args.summary), 'independent_audit_sha256': digest(args.audit),
                  'tool_sha256': digest(__file__), 'seeds': summary['seeds'], 'plotted_values': plotted,
                  'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                  'boundary': 'three training-seed observations and sample SD; no confidence interval or method selection'}
    (output / 'figure-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    text = ['# Audited three-seed comparison', '',
            'Colored points are the three independently trained classifiers, with identical data split and scoring protocol. '
            'Black diamonds and whiskers show mean and sample SD across training seeds; these are not confidence intervals '
            'and differ from the single-classifier sample bootstrap intervals in each seed report. '
            'Symmetric SD whiskers can extend outside the rate range; the plotted seed rates stay within 0–100%.', '',
            'Synthetic diagnostics are labeled separately and excluded from the natural macro. '
            'All methods keep their original temperature, score direction and training-only feature fitting.', '',
            '![Training seed comparison](seed-comparison.png)', '',
            f"Input summary SHA-256: `{provenance['summary_sha256']}`.", '',
            'The independent audit passed for the exact summary plotted here. '
            'Figures are descriptive and do not establish statistical significance or select a method from OOD tests.', '']
    (output / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
