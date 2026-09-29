"""Describe frozen logits/features on CPU; fit no detector or parameter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import logsumexp


COLORS = {'id_test': '#222222', 'id_validation': '#777777', 'cifar10': '#2878b5',
          'mnist': '#d76b2a', 'noise': '#3a9656', 'svhn': '#8860ab', 'constant': '#bd487b'}
QUANTILES = [.01, .05, .5, .95, .99]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def describe(logits, features):
    logits, features = logits.astype(np.float64), features.astype(np.float64)
    if logits.ndim != 2 or features.ndim != 2 or len(logits) != len(features):
        raise ValueError('logit/feature population mismatch')
    if not len(logits) or not np.isfinite(logits).all() or not np.isfinite(features).all():
        raise ValueError('nonfinite or empty arrays')
    maximum, offset = logits.max(1), logits.mean(1)
    normalizer = logsumexp(logits, axis=1)
    centered_normalizer = logsumexp(logits - offset[:, None], axis=1)
    confidence = np.exp(maximum - normalizer)
    energy = -normalizer
    # Exact T=1 identities, checked independently on every saved population.
    max_identity_error = float(np.abs(energy - (-maximum + np.log(confidence))).max())
    offset_identity_error = float(np.abs(energy - (-offset - centered_normalizer)).max())
    if max_identity_error > 1e-10 or offset_identity_error > 1e-10:
        raise ValueError('energy decomposition identity failed')
    arrays = {'confidence_t1': confidence, 'feature_l2_norm': np.linalg.norm(features, axis=1),
              'energy_t1': energy, 'max_logit': maximum, 'mean_logit_offset': offset,
              'centered_logsumexp': centered_normalizer}
    records = {name: {'mean': float(values.mean()),
                      'quantiles': dict(zip(map(str, QUANTILES), np.quantile(values, QUANTILES).tolist()))}
               for name, values in arrays.items()}
    records.update(count=len(logits), feature_dimensions=features.shape[1],
                   confidence_at_least_099=float((confidence >= .99).mean()),
                   energy_maxlogit_identity_max_error=max_identity_error,
                   energy_offset_identity_max_error=offset_identity_error)
    return arrays, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluations', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases, reports, seeds = [], [], []
    for evaluation in map(Path, args.evaluations):
        result = json.loads((evaluation / 'results.json').read_text(encoding='utf-8'))
        status = json.loads((evaluation.parent / 'status.json').read_text(encoding='utf-8'))
        if status['state'] != 'complete' or status['epoch'] != result['training_config']['epochs']:
            raise ValueError('describe completed frozen classifiers only')
        if result['temperature'] != 1.:
            raise ValueError('this fixed diagnostic describes T=1 only')
        if digest(result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('checkpoint hash mismatch')
        seed = result['training_config']['seed']
        if seed in seeds:
            raise ValueError('duplicate training seed')
        seeds.append(seed)
        with np.load(evaluation / 'features.npz', allow_pickle=False) as archive:
            id_features, val_features = archive['id'], archive['validation']
        with np.load(evaluation / 'id_logits.npz', allow_pickle=False) as archive:
            id_logits, labels = archive['logits'], archive['labels']
        if not np.isclose((id_logits.argmax(1) == labels).mean(), result['id_test_accuracy'], atol=1e-7):
            raise ValueError('ID accuracy mismatch')
        with np.load(evaluation / 'validation_logits.npz', allow_pickle=False) as archive:
            val_logits = archive['logits']
        populations = {'id_test': (id_logits, id_features), 'id_validation': (val_logits, val_features)}
        for name in result['datasets']:
            with np.load(evaluation / f'{name}_logits.npz', allow_pickle=False) as archive:
                logits = archive['logits']
            with np.load(evaluation / f'{name}_features.npz', allow_pickle=False) as archive:
                features = archive['features']
            populations[name] = logits, features
        arrays, population_records = {}, {}
        for name, (logits, features) in populations.items():
            arrays[name], population_records[name] = describe(logits, features)
            if name in result['datasets']:
                with np.load(evaluation / f'{name}_scores.npz', allow_pickle=False) as scores:
                    if not np.allclose(arrays[name]['energy_t1'], scores['energy_ood'], atol=2e-5, rtol=2e-6):
                        raise ValueError('derived Energy differs from saved scoring')
                    if not np.allclose(1 - arrays[name]['confidence_t1'], scores['msp_ood'], atol=2e-6):
                        raise ValueError('derived confidence differs from saved MSP')
        cases.append((seed, arrays))
        reports.append({'seed': seed, 'id_test_accuracy': result['id_test_accuracy'],
                        'selected_epoch': result['checkpoint_epoch'], 'checkpoint_sha256': result['checkpoint_sha256'],
                        'populations': population_records,
                        'artifact_hashes': {path.name: digest(path) for path in evaluation.glob('*.npz')}})
    figure, axes = plt.subplots(3, len(cases), figsize=(6.1 * len(cases), 10.5), dpi=150, squeeze=False)
    plotted = [('confidence_t1', 'Max softmax probability (T=1)'),
               ('feature_l2_norm', 'Final-layer feature L2 norm'), ('energy_t1', 'Energy score (T=1)')]
    handles, legend_labels = [], []
    for column, (seed, arrays) in enumerate(cases):
        for row, (metric, xlabel) in enumerate(plotted):
            ax = axes[row, column]
            for name, values in arrays.items():
                ordered = np.sort(values[metric], kind='stable')
                line, = ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                                label=name, color=COLORS.get(name), linewidth=1.5,
                                linestyle='--' if name == 'id_validation' else '-')
                if row == column == 0:
                    handles.append(line)
                    legend_labels.append(name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Empirical cumulative fraction')
            ax.set_title(f'Seed {seed}')
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if metric == 'confidence_t1':
                ax.set_xlim(0, 1)
    # Use common axes across seeds; no data-dependent rescaling per population.
    for row in range(3):
        limits = [axes[row, column].get_xlim() for column in range(len(cases))]
        for column in range(len(cases)):
            axes[row, column].set_xlim(min(x[0] for x in limits), max(x[1] for x in limits))
    figure.legend(handles, legend_labels, loc='upper center', ncol=min(5, len(handles)), frameon=False,
                  bbox_to_anchor=(.5, .985))
    figure.suptitle('Frozen classifiers: confidence, feature magnitude and Energy', fontsize=14, y=1.01)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output / 'distributions.png', bbox_inches='tight')
    figure.savefig(output / 'distributions.svg', bbox_inches='tight')
    plt.close(figure)
    record = {'seeds_described': seeds, 'expected_training_seeds': [0, 1, 2],
              'purpose': 'descriptive frozen-output diagnostics; no fitting or selection',
              'temperature': 1., 'quantile_levels': QUANTILES, 'classifiers': reports,
              'interpretation_limits': 'Feature magnitudes across independently trained networks have different coordinate scales. '
                                       'These descriptors do not explain a causal mechanism or define a fitted OOD detector. '
                                       'Synthetic populations are diagnostic only. Training-seed summary requires all predeclared seeds.',
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
              'tool_sha256': digest(__file__), 'created_at': datetime.now(timezone.utc).isoformat()}
    (output / 'diagnostics.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Frozen-output distribution diagnostics', '',
             f'Described training seeds: {seeds}; predeclared seeds: [0, 1, 2]. No across-seed mean/SD is inferred here.', '',
             'This CPU analysis reads existing logits/features and changes no classifier, score, threshold, '
             'temperature or detector parameter. ID validation and test populations remain distinct. '
             'No ID-test or OOD features are used to fit a detector.', '',
             'At T=1, Energy = -max(logit) + log(max softmax probability). It also equals '
             '-mean(logit) - logsumexp(logit - mean(logit)). Both identities are checked on every saved population. '
             'These decompositions describe logit magnitude and relative separation; they do not prove causation.', '']
    for classifier in reports:
        lines += [f"## Seed {classifier['seed']}", '',
                  f"ID test accuracy {100 * classifier['id_test_accuracy']:.2f}%; selected epoch {classifier['selected_epoch']}.", '',
                  '| Population | Count | Median confidence | Confidence >=99% | Median feature norm | Median Energy | Median logit mean |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name, row in classifier['populations'].items():
            median = lambda key: row[key]['quantiles']['0.5']
            lines.append(f"| {name} | {row['count']} | {median('confidence_t1'):.4f} | "
                         f"{100 * row['confidence_at_least_099']:.2f}% | {median('feature_l2_norm'):.3f} | "
                         f"{median('energy_t1'):.3f} | {median('mean_logit_offset'):.3f} |")
        lines.append('')
    lines += ['![Frozen output distributions](distributions.png)', '', record['interpretation_limits'], '',
              'All quantiles, source-array SHA-256 values, environment and identity residuals are retained in diagnostics.json. '
              'OOD results remain descriptive and cannot select a hyperparameter. All data and artifacts remain local.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(output), 'seeds': seeds, 'populations_checked': sum(len(r['populations']) for r in reports)}))


if __name__ == '__main__':
    main()
