"""Independent CPU audit of saved evaluations; never changes experiment artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import logsumexp
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def logit_scores(logits, temperature):
    z = logits.astype(np.float64) / temperature
    log_p = z - logsumexp(z, axis=1, keepdims=True)
    probabilities = np.exp(log_p)
    return {'msp': 1 - probabilities.max(axis=1),
            'energy': -temperature * logsumexp(z, axis=1),
            'maxlogit': -logits.max(axis=1),
            'entropy': -(probabilities * log_p).sum(axis=1)}


def audit(evaluation):
    evaluation = Path(evaluation)
    result = json.loads((evaluation / 'results.json').read_text(encoding='utf-8'))
    run, config = evaluation.parent, result['training_config']
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    history = json.loads((run / 'history.json').read_text(encoding='utf-8'))
    require(status['state'] == 'complete' and status['epoch'] == config['epochs'], 'unfinished training')
    require(len(history) == config['epochs'], 'incomplete epoch history')
    require([row['epoch'] for row in history] == list(range(1, config['epochs'] + 1)), 'epoch order mismatch')
    selected = max(history, key=lambda row: row['val_accuracy'])
    require(selected['epoch'] == result['checkpoint_epoch'], 'checkpoint is not first ID-validation maximum')
    require(digest(result['checkpoint']) == result['checkpoint_sha256'], 'checkpoint hash mismatch')
    require(result['checkpoint_selection'] == 'ID validation accuracy only', 'selection protocol mismatch')
    require(result['score_direction'] == 'higher_is_ood', 'score orientation mismatch')
    for name, expected in result['evaluation_source_hashes'].items():
        require(digest(evaluation / 'evaluation_source' / name) == expected, 'evaluation source snapshot mismatch')
    source = json.loads((run / 'source_hashes.json').read_text(encoding='utf-8'))
    for name, expected in source.items():
        require(digest(run / 'source_snapshot' / name) == expected, 'training source snapshot mismatch')
        require(result['evaluation_source_hashes'][name] == expected, 'evaluation used changed training source')
    split = json.loads((run / 'split.json').read_text(encoding='utf-8'))
    with np.load(run / 'split_indices.npz', allow_pickle=False) as indices:
        train, validation = indices['train'], indices['val']
    require(not np.intersect1d(train, validation).size, 'train/validation overlap')
    require(len(np.unique(np.concatenate([train, validation]))) == len(train) + len(validation), 'duplicate indices')
    for name, indices in [('train', train), ('val', validation)]:
        require(hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest() == split[f'{name}_indices_sha256'], 'split hash mismatch')
        require(len(indices) == split[f'{name}_count'], 'split count mismatch')
    if result.get('feature_fit'):
        fit = result['feature_fit']
        require(fit['population'] == 'ID training split only' and fit['count'] == len(train), 'feature fit population mismatch')
        require(fit['indices_sha256'] == split['train_indices_sha256'], 'feature fit split mismatch')
    with np.load(evaluation / 'id_logits.npz', allow_pickle=False) as archive:
        id_logits, labels = archive['logits'], archive['labels']
    with np.load(evaluation / 'validation_logits.npz', allow_pickle=False) as archive:
        val_logits = archive['logits']
    require(np.isclose((id_logits.argmax(1) == labels).mean(), result['id_test_accuracy'], atol=1e-7), 'ID accuracy mismatch')
    expected_scores = {'id': logit_scores(id_logits, result['temperature']),
                       'val': logit_scores(val_logits, result['temperature'])}
    checks, max_error = 0, 0.
    for dataset, entry in result['datasets'].items():
        with np.load(evaluation / f'{dataset}_logits.npz', allow_pickle=False) as archive:
            expected_scores['ood'] = logit_scores(archive['logits'], result['temperature'])
        with np.load(evaluation / f'{dataset}_scores.npz', allow_pickle=False) as scores:
            for method, saved in entry['methods'].items():
                inside, outside, calibrated = (scores[f'{method}_{part}'] for part in ('id', 'ood', 'val'))
                require((len(inside), len(outside), len(calibrated)) == (saved['n_id'], saved['n_ood'], len(validation)), 'score population mismatch')
                require(all(a.ndim == 1 and np.isfinite(a).all() for a in (inside, outside, calibrated)), 'invalid raw scores')
                truth = np.concatenate([np.zeros(len(inside)), np.ones(len(outside))])
                values = np.concatenate([inside, outside])
                independent = {'auroc': roc_auc_score(truth, values),
                               'aupr_out': average_precision_score(truth, values),
                               'aupr_in': average_precision_score(1 - truth, -values)}
                for key, actual in independent.items():
                    error = abs(actual - saved[key])
                    max_error = max(max_error, error)
                    require(error < 1e-12, f'{dataset}/{method}/{key} independent metric mismatch')
                    checks += 1
                for part, array in [('id', inside), ('val', calibrated), ('ood', outside)]:
                    if method in expected_scores[part]:
                        require(np.allclose(array, expected_scores[part][method], atol=2e-5, rtol=2e-6), 'raw score/logit mismatch')
                benchmark = np.sort(inside)[int(np.ceil(.95 * len(inside))) - 1]
                threshold = np.sort(calibrated)[int(np.ceil(.95 * len(calibrated))) - 1]
                threshold_values = {'benchmark_threshold': float(benchmark), 'validation_threshold': float(threshold),
                                    'fpr95_id': float((outside <= benchmark).mean()),
                                    'id_tpr_at_fpr95': float((inside <= benchmark).mean()),
                                    'calibration_id_acceptance': float((calibrated <= threshold).mean()),
                                    'id_test_acceptance_at_validation_threshold': float((inside <= threshold).mean()),
                                    'ood_acceptance_at_validation_threshold': float((outside <= threshold).mean()),
                                    'ood_prevalence': len(outside) / (len(inside) + len(outside))}
                for key, value in threshold_values.items():
                    require(abs(value - saved[key]) < 1e-12, f'{dataset}/{method}/{key} mismatch')
    for archive_path in evaluation.glob('*.npz'):
        with np.load(archive_path, allow_pickle=False) as archive:
            require(all(np.isfinite(archive[key]).all() for key in archive.files), f'nonfinite array in {archive_path.name}')
    return {'status': 'passed', 'evaluation': str(evaluation.resolve()), 'seed': config['seed'],
            'selected_epoch': result['checkpoint_epoch'], 'best_id_validation_accuracy': selected['val_accuracy'],
            'id_test_accuracy': result['id_test_accuracy'], 'independent_sklearn_checks': checks,
            'maximum_metric_error': max_error, 'checkpoint_sha256': result['checkpoint_sha256'],
            'checks': ['full schedule and ID-only selection', 'checkpoint/source/split hashes',
                       'feature fitting population', 'raw logits and scores', 'independent AUROC/AP',
                       'empirical thresholds and acceptance', 'all saved arrays finite'],
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
            'audited_at': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.evaluation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
