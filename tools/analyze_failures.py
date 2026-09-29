"""Describe frozen-threshold failure patterns from saved artifacts, on CPU.

This script does not change scores, thresholds, weights, or hyperparameters.
ID errors remain ID samples; their rejection is reported separately to explain
the score distributions without redefining the benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rate(mask):
    return float(mask.mean()) if len(mask) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    evaluation, output = Path(args.evaluation), Path(args.output_dir)
    result = json.loads((evaluation / 'results.json').read_text(encoding='utf-8'))
    with np.load(evaluation / 'id_logits.npz', allow_pickle=False) as archive:
        logits, labels = archive['logits'], archive['labels']
    if logits.ndim != 2 or labels.shape != (len(logits),) or not np.isfinite(logits).all():
        raise ValueError('invalid ID logits/labels')
    predictions = logits.argmax(axis=1)
    correct = predictions == labels
    if not np.isclose(correct.mean(), result['id_test_accuracy'], atol=1e-7):
        raise ValueError('saved logits do not match reported ID accuracy')
    output.mkdir(parents=True, exist_ok=True)
    analysis = {'checkpoint_sha256': result['checkpoint_sha256'], 'selected_epoch': result['checkpoint_epoch'],
                'id_accuracy': float(correct.mean()), 'threshold_source': 'saved ID-validation-only thresholds',
                'datasets': {}}
    lines = ['# Frozen-threshold failure analysis', '',
             f"Classifier selected epoch: {result['checkpoint_epoch']}. ID test accuracy: {100 * correct.mean():.2f}%.", '',
             'All thresholds are the original ID-validation-calibrated thresholds. No detector is refitted, '
             'no test-set threshold is chosen, and ID misclassifications remain ID.', '']
    class_rows, id_summary = [], None
    for dataset, entry in result['datasets'].items():
        with np.load(evaluation / f'{dataset}_logits.npz', allow_pickle=False) as archive:
            ood_predictions = archive['logits'].argmax(axis=1)
        methods = entry['methods']
        data = {}
        lines += [f'## {dataset.upper()}', '',
                  '| Method | Correct ID rejected % | Incorrect ID rejected % | Accuracy of accepted ID % | OOD accepted % |',
                  '|---|---:|---:|---:|---:|']
        with np.load(evaluation / f'{dataset}_scores.npz', allow_pickle=False) as scores:
            for method, metrics in methods.items():
                inside, outside = scores[f'{method}_id'], scores[f'{method}_ood']
                if inside.shape != labels.shape or outside.shape != ood_predictions.shape:
                    raise ValueError('scores do not match logits population')
                threshold = metrics['validation_threshold']
                accepted, ood_accepted = inside <= threshold, outside <= threshold
                if not np.isclose(accepted.mean(), metrics['id_test_acceptance_at_validation_threshold']):
                    raise ValueError('ID threshold behavior differs from saved metrics')
                if not np.isclose(ood_accepted.mean(), metrics['ood_acceptance_at_validation_threshold']):
                    raise ValueError('OOD threshold behavior differs from saved metrics')
                rejected_correct = rate(~accepted[correct])
                rejected_incorrect = rate(~accepted[~correct])
                accepted_accuracy = rate(correct[accepted])
                counts = np.bincount(ood_predictions[ood_accepted], minlength=logits.shape[1])
                frequent = np.argsort(-counts, kind='stable')[:5]
                top_classes = [{'classifier_output_class_id': int(class_id), 'accepted_ood_count': int(counts[class_id]),
                                'fraction_of_accepted_ood': float(counts[class_id] / max(1, counts.sum()))}
                               for class_id in frequent if counts[class_id]]
                row = {'correct_id_rejection': rejected_correct, 'incorrect_id_rejection': rejected_incorrect,
                       'accepted_id_accuracy': accepted_accuracy, 'id_acceptance': float(accepted.mean()),
                       'ood_acceptance': float(ood_accepted.mean()), 'accepted_ood_count': int(ood_accepted.sum()),
                       'accepted_ood_top_classifier_outputs': top_classes,
                       'lowest_anomaly_ood_indices': np.argsort(outside, kind='stable')[:20].tolist(),
                       'highest_anomaly_id_indices': np.argsort(-inside, kind='stable')[:20].tolist()}
                data[method] = row
                def fmt(value):
                    return f'{100 * value:.2f}' if value is not None else 'N/A'
                lines.append(f'| {method} | {fmt(rejected_correct)} | {fmt(rejected_incorrect)} | '
                             f'{fmt(accepted_accuracy)} | {fmt(row["ood_acceptance"])} |')
                for label in np.unique(labels):
                    mask = labels == label
                    class_rows.append({'dataset': dataset, 'method': method, 'id_class': int(label),
                                       'count': int(mask.sum()), 'classifier_accuracy': float(correct[mask].mean()),
                                       'id_rejection_rate': float((~accepted[mask]).mean())})
        analysis['datasets'][dataset] = data
        if id_summary is None:
            id_summary = data
        lines += ['', 'Top accepted-OOD classifier output IDs and representative sample indices are retained in '
                  '`failure-analysis.json`. Output-class concentration is descriptive and does not define OOD semantics.', '']
    methods = list(id_summary)
    locations = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
    ax.bar(locations - .18, [100 * (id_summary[m]['correct_id_rejection'] or 0) for m in methods], .36,
           label='Correctly classified ID', color='#2764ad')
    ax.bar(locations + .18, [100 * (id_summary[m]['incorrect_id_rejection'] or 0) for m in methods], .36,
           label='Misclassified ID', color='#db672c')
    highest_rate = max(100 * (id_summary[m][key] or 0) for m in methods
                       for key in ('correct_id_rejection', 'incorrect_id_rejection'))
    ax.set_ylim(0, max(1., highest_rate * 1.25))
    ax.set_xticks(locations, methods)
    ax.set_ylabel('ID rejection rate at validation threshold (%)')
    ax.set_title('Fixed classifier: which ID samples are rejected?')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'id-rejection-by-correctness.png')
    plt.close(fig)
    lines += ['## Interpretation limits', '',
              'Rejection of classification mistakes can improve accuracy on the remaining accepted ID samples, '
              'but it does not demonstrate natural OOD detection. OOD acceptance must be assessed separately. '
              'This report explains a fixed model and threshold; it is not a method-selection experiment.', '',
              '![ID rejection by correctness](id-rejection-by-correctness.png)', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'failure-analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding='utf-8')
    with (output / 'id-class-rejection.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
