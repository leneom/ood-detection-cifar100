"""CPU-only qualitative inspection of frozen thresholds and official test pixels.

Ranking rules are fixed and explicit. Test labels annotate images only; they do
not select OOD samples, fit a detector, or change any experimental parameter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax


SOURCES = {
    'cifar100': ('cifar-100-python', 'test', 'f0ef6b0ae62326f3e7ffdfab6717acfc',
                 'meta', '7973b15100ade9c7d40fb424638fde48', 'fine_labels', 'fine_label_names'),
    'cifar10': ('cifar-10-batches-py', 'test_batch', '40351d587109b95175f43aff81a1287e',
                'batches.meta', '5ff9c542aee3614f3951f8cda6e48888', 'labels', 'label_names'),
}


def load_test(root, name):
    folder, test_file, test_md5, meta_file, meta_md5, labels_key, names_key = SOURCES[name]
    payloads, provenance = [], {}
    for filename, expected in ((test_file, test_md5), (meta_file, meta_md5)):
        path = Path(root) / folder / filename
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != expected:
            raise ValueError(f'{name}/{filename} official test checksum mismatch')
        payloads.append(pickle.loads(raw, encoding='latin1'))
        provenance[filename] = {'md5': expected, 'sha256': hashlib.sha256(raw).hexdigest()}
    test, metadata = payloads
    images = test['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels, names = np.asarray(test[labels_key]), metadata[names_key]
    if images.shape != (10000, 32, 32, 3) or images.dtype != np.uint8 or labels.shape != (10000,):
        raise ValueError('unexpected official CIFAR test shape')
    return images, labels, names, provenance


def rank(scores, mask, count, descending):
    indices = np.flatnonzero(mask)
    order = np.argsort(-scores[indices] if descending else scores[indices], kind='stable')
    return indices[order[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--columns', type=int, default=6)
    parser.add_argument('--methods', nargs='+', default=['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis'])
    args = parser.parse_args()
    if not 1 <= args.columns <= 8:
        raise ValueError('columns must lie between 1 and 8')
    evaluation, output = Path(args.evaluation), Path(args.output_dir)
    result = json.loads((evaluation / 'results.json').read_text(encoding='utf-8'))
    if result['id_dataset'] != 'cifar100' or 'cifar10' not in result['datasets']:
        raise ValueError('gallery requires the CIFAR-100/CIFAR-10 pairing')
    checkpoint = Path(result['checkpoint'])
    with checkpoint.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != result['checkpoint_sha256']:
            raise ValueError('frozen checkpoint hash mismatch')
    data_root = result['training_config']['data_root']
    id_images, id_labels, id_names, id_source = load_test(data_root, 'cifar100')
    ood_images, ood_labels, ood_names, ood_source = load_test(data_root, 'cifar10')
    with np.load(evaluation / 'id_logits.npz', allow_pickle=False) as archive:
        id_logits, saved_labels = archive['logits'], archive['labels']
    with np.load(evaluation / 'cifar10_logits.npz', allow_pickle=False) as archive:
        ood_logits = archive['logits']
    if not np.array_equal(saved_labels, id_labels):
        raise ValueError('official ID test order differs from saved labels')
    if id_logits.shape != (10000, 100) or ood_logits.shape != (10000, 100):
        raise ValueError('logit population mismatch')
    id_predictions, ood_predictions = id_logits.argmax(1), ood_logits.argmax(1)
    correct = id_predictions == id_labels
    id_confidence = softmax(id_logits.astype(np.float64), axis=1).max(1)
    ood_confidence = softmax(ood_logits.astype(np.float64), axis=1).max(1)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'checkpoint_sha256': result['checkpoint_sha256'], 'checkpoint_epoch': result['checkpoint_epoch'],
                'evaluation': str(evaluation.resolve()), 'threshold_source': 'saved ID-validation-only thresholds',
                'selection': 'descending anomaly among each rejected-ID subgroup; ascending anomaly among accepted OOD; stable test-index tie order',
                'label_usage': 'annotation only; OOD ranking does not use true or predicted labels',
                'display_transform': 'original official uint8 RGB pixels; nearest-neighbor display; no evaluation transform changed',
                'sources': {'cifar100': id_source, 'cifar10': ood_source}, 'methods': {},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
                'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'created_at': datetime.now(timezone.utc).isoformat()}
    report = ['# Frozen-threshold failure examples', '',
              f"Completed classifier seed {result['training_config']['seed']}, selected epoch {result['checkpoint_epoch']}; "
              f"ID test accuracy {100 * result['id_test_accuracy']:.2f}%.", '',
              'Each row contains a deterministic set of extreme examples, not a random or representative sample. '
              'All decisions use the saved ID-validation-only threshold. First two rows rank rejected ID by '
              'decreasing anomaly; the last row ranks accepted CIFAR-10 OOD by increasing anomaly, with ties '
              'resolved by original test index. True OOD labels only annotate images.', '',
              'Confidence is max softmax at T=1; it is a classifier score, not evidence that the input is ID. '
              'Fine CIFAR-100 output labels and CIFAR-10 true labels belong to different label spaces. '
              'The gallery does not define a new semantic split or choose a hyperparameter.', '']
    with np.load(evaluation / 'cifar10_scores.npz', allow_pickle=False) as scores:
        for method in args.methods:
            metrics = result['datasets']['cifar10']['methods'][method]
            threshold = metrics['validation_threshold']
            inside, outside = scores[f'{method}_id'], scores[f'{method}_ood']
            rejected, accepted_ood = inside > threshold, outside <= threshold
            if not np.isclose(accepted_ood.mean(), metrics['ood_acceptance_at_validation_threshold']):
                raise ValueError('OOD acceptance disagrees with saved metrics')
            groups = [('Correct ID rejected', rejected & correct, True),
                      ('Misclassified ID rejected', rejected & ~correct, True),
                      ('CIFAR-10 OOD accepted', accepted_ood, False)]
            rows = []
            fig, axes = plt.subplots(3, args.columns, figsize=(args.columns * 2.5, 9.1), dpi=150, squeeze=False)
            for row_number, (title, mask, descending) in enumerate(groups):
                is_ood = row_number == 2
                selected = rank(outside if is_ood else inside, mask, args.columns, descending)
                records = []
                for column, ax in enumerate(axes[row_number]):
                    ax.set_axis_off()
                    if column >= len(selected):
                        ax.text(.5, .5, 'No qualifying sample', ha='center', va='center', transform=ax.transAxes, fontsize=9)
                        continue
                    index = int(selected[column])
                    images, labels, names, predictions, confidence, anomaly = (
                        (ood_images, ood_labels, ood_names, ood_predictions, ood_confidence, outside) if is_ood
                        else (id_images, id_labels, id_names, id_predictions, id_confidence, inside))
                    true_name, predicted_name = names[int(labels[index])], id_names[int(predictions[index])]
                    ax.imshow(images[index], interpolation='nearest')
                    ax.set_title(f'#{index}: true {true_name}\npred {predicted_name} | conf {confidence[index]:.4f}', fontsize=9, pad=5)
                    ax.text(.5, -.07, f'score {anomaly[index]:.5g}', ha='center', va='top', transform=ax.transAxes, fontsize=9)
                    records.append({'official_test_index': index, 'true_class': true_name, 'predicted_id_class': predicted_name,
                                    'max_softmax_t1': float(confidence[index]), 'anomaly_score': float(anomaly[index]),
                                    'validation_threshold': threshold, 'accepted': bool(anomaly[index] <= threshold)})
                axes[row_number, 0].text(-.18, .5, f'{title}\nn={int(mask.sum())}', ha='right', va='center',
                                         transform=axes[row_number, 0].transAxes, fontsize=10)
                rows.append({'group': title, 'population_count': int(mask.sum()), 'examples': records})
            fig.suptitle(f'CIFAR-100 vs CIFAR-10 | {method.upper()} | fixed validation threshold {threshold:.5g}', fontsize=13, y=.985)
            fig.subplots_adjust(left=.17, right=.985, top=.90, bottom=.045, wspace=.22, hspace=.5)
            fig.savefig(output / f'{method}-examples.png')
            fig.savefig(output / f'{method}-examples.svg')
            plt.close(fig)
            manifest['methods'][method] = {'validation_threshold': threshold, 'rows': rows}
            report += [f'## {method.upper()}', '',
                       f"CIFAR-10 OOD acceptance at the validation threshold: {100 * accepted_ood.mean():.2f}%.", '',
                       f'![{method} examples]({method}-examples.png)', '']
    report += ['These examples illustrate failure modes only. Aggregate rates remain in the evaluation report; '
               'method rankings or semantic conclusions cannot be established from an extreme-example gallery.', '',
               'Original dataset pixels and rendered galleries remain local under `runs/`.', '']
    (output / 'gallery.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({'output': str(output), 'methods': list(manifest['methods']),
                      'examples': sum(len(row['examples']) for item in manifest['methods'].values() for row in item['rows'])}))


if __name__ == '__main__':
    main()
