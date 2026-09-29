"""Independent phase-2 audit: SciPy probability math, convex beta root, sklearn metrics.

Does not import the production calibration core or the project's metrics module.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import statistics
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import logsumexp, softmax
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/calibration-phase2'
REPORT = ROOT / 'report/calibration-phase2'
METHODS = ('msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis')
NATURAL = ('cifar10', 'mnist', 'svhn', 'textures')
DATASETS = NATURAL + ('noise',)
FOLDERS = {'cifar10': 'eval-final', 'mnist': 'eval-final', 'noise': 'eval-final',
           'svhn': 'eval-extension-svhn', 'textures': 'eval-extension-textures'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=1e-12):
    require(np.isfinite(actual) and abs(actual - expected) <= tolerance, f'{label}: {actual} != {expected}')


def scores(logits, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    log_p = scaled - logsumexp(scaled, axis=1, keepdims=True)
    p = np.exp(log_p)
    return {'msp': 1 - p.max(axis=1), 'energy': -temperature * logsumexp(scaled, axis=1),
            'entropy': -(p * log_p).sum(axis=1), 'maxlogit': -np.asarray(logits, dtype=np.float64).max(axis=1)}


def probability_audit(logits, labels, temperature, saved):
    z = logits.astype(np.float64) / temperature
    lp = z - logsumexp(z, axis=1, keepdims=True)
    p = np.exp(lp)
    one_hot = np.eye(z.shape[1], dtype=np.float64)[labels]
    correct = logits.argmax(axis=1) == labels
    confidence = p.max(axis=1)
    ece = 0.
    for index in range(15):
        lo, hi = index / 15, (index + 1) / 15
        mask = (confidence >= lo) & ((confidence < hi) if index < 14 else (confidence <= 1.))
        record = saved['bins'][index]
        require(int(mask.sum()) == record['count'], 'ECE bin counts mismatch')
        close(record['lower'], lo, 'ECE lower boundary')
        close(record['upper'], hi, 'ECE upper boundary')
        if mask.any():
            accuracy, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            close(record['accuracy'], accuracy, 'ECE bin accuracy')
            close(record['confidence'], conf, 'ECE bin confidence')
            ece += mask.mean() * abs(accuracy - conf)
        else:
            require(record['accuracy'] is None and record['confidence'] is None, 'empty bin convention')
    values = {'nll': float(-lp[np.arange(len(labels)), labels].mean()),
              'brier': float(np.square(p - one_hot).sum(axis=1).mean()), 'ece15': float(ece),
              'accuracy': float(correct.mean()), 'mean_confidence': float(confidence.mean()), 'n': len(labels)}
    for name, expected in values.items():
        close(saved[name], expected, 'ID probability ' + name)
    return 6


def independent_metrics(inside, outside, calibration):
    y = np.r_[np.zeros(len(inside)), np.ones(len(outside))]
    values = np.r_[inside, outside]
    benchmark = np.sort(inside)[int(np.ceil(.95 * len(inside))) - 1]
    threshold = np.sort(calibration)[int(np.ceil(.95 * len(calibration))) - 1]
    return {'auroc': float(roc_auc_score(y, values)),
            'aupr_out': float(average_precision_score(y, values)),
            'aupr_in': float(average_precision_score(1 - y, -values)),
            'benchmark_threshold': float(benchmark), 'validation_threshold': float(threshold),
            'fpr95_id': float(np.mean(outside <= benchmark)),
            'id_tpr_at_fpr95': float(np.mean(inside <= benchmark)),
            'calibration_id_acceptance': float(np.mean(calibration <= threshold)),
            'id_test_acceptance_at_validation_threshold': float(np.mean(inside <= threshold)),
            'ood_acceptance_at_validation_threshold': float(np.mean(outside <= threshold)),
            'ood_prevalence': len(outside) / (len(inside) + len(outside)),
            'n_id': len(inside), 'n_ood': len(outside),
            'benchmark_ID_equal_threshold_count': int(np.sum(inside == benchmark)),
            'threshold_calibration_equal_threshold_count': int(np.sum(calibration == threshold)),
            'threshold_calibration_n': len(calibration)}


def main():
    status = read(RUN / 'status.json')
    require(status['state'] == 'complete' and status['completed_seeds'] == [0, 1, 2], 'calibration incomplete')
    manifest = read(RUN / 'inputs.json')
    for name, expected in manifest['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'original input changed: ' + name)
    for name, expected in manifest['analysis_source_hashes'].items():
        path = ROOT / ('src/ood/metrics.py' if name == 'metrics.py' else 'tools/' + name)
        require(digest(path) == expected and digest(RUN / 'analysis_source' / name) == expected, 'analysis snapshot changed')
    prereg = read(RUN / 'predeclaration.json')
    require(prereg['declared_at'] < manifest['prepared_at'], 'inputs not frozen after predeclaration')
    require(digest(RUN / 'calibration_split.npz') == manifest['split_sha256'], 'split file changed')
    split = load(RUN / 'calibration_split.npz')
    with (ROOT / 'data/cifar-100-python/train').open('rb') as stream:
        official = pickle.load(stream, encoding='latin1')
    y = np.asarray(official['fine_labels'], dtype=np.int64)[split['original_validation_indices']]
    require(np.array_equal(y, split['validation_labels']), 'recovered validation labels mismatch')
    # Reconstruct the fixed RNG recipe without calling production split code.
    generator = np.random.default_rng(2027)
    fit, threshold = [], []
    for label in range(100):
        rows = np.where(y == label)[0]
        require(len(rows) == 50, 'unbalanced original validation')
        generator.shuffle(rows)
        fit += rows[:25].tolist()
        threshold += rows[25:].tolist()
    fit, threshold = np.array(sorted(fit)), np.array(sorted(threshold))
    require(np.array_equal(fit, split['fit_rows']) and np.array_equal(threshold, split['threshold_rows']), 'partition recipe mismatch')
    require(not np.intersect1d(fit, threshold).size and np.array_equal(np.sort(np.r_[fit, threshold]), np.arange(5000)), 'overlap/incomplete partition')
    independent_checks = probability_checks = score_checks = threshold_checks = precision_checks = 0
    root_records, result_hashes = [], {}
    for seed in [0, 1, 2]:
        path = RUN / f'seed{seed}'
        result = read(path / 'results.json')
        require(result['state'] == 'complete' and result['inputs_sha256'] == digest(RUN / 'inputs.json'), 'seed input mismatch')
        require(result['temperature_fit_json_sha256'] == digest(path / 'temperature_fit.json'), 'optimizer record changed')
        require(result['temperature_fit'] == read(path / 'temperature_fit.json'), 'optimizer records disagree')
        for name, expected in result['output_array_sha256'].items():
            require(digest(path / name) == expected, 'phase-2 raw arrays changed')
        original = ROOT / f'runs/cifar100-seed{seed}/eval-final'
        id_data = load(original / 'id_logits.npz')
        val_logits = load(original / 'validation_logits.npz')['logits']
        source = read(original / 'results.json')
        require(result['checkpoint_sha256'] == source['checkpoint_sha256'] and result['checkpoint_epoch'] == source['checkpoint_epoch'], 'frozen checkpoint changed')
        optimizer = result['temperature_fit']
        require(manifest['prepared_at'] < optimizer['fitted_at'], 'fit ran before split/inputs frozen')
        require(optimizer['fit_rows_sha256'] == hashlib.sha256(fit.astype(np.int64).tobytes()).hexdigest(), 'fit row hash mismatch')
        t = optimizer['temperature']
        require(.05 <= t <= 20, 'temperature outside predeclared bounds')
        fitting = val_logits[fit].astype(np.float64)
        targets = fitting[np.arange(len(fit)), y[fit]]
        def beta_derivative(beta):
            return float((np.sum(softmax(beta * fitting, axis=1) * fitting, axis=1) - targets).mean())
        low, high = 1 / 20, 1 / .05
        if beta_derivative(low) >= 0:
            beta = low
        elif beta_derivative(high) <= 0:
            beta = high
        else:
            beta = brentq(beta_derivative, low, high, xtol=1e-14, rtol=1e-14)
        independent_t = 1 / beta
        independent_nll = float((logsumexp(fitting / independent_t, axis=1) - targets / independent_t).mean())
        close(optimizer['fit_nll_T1'], float((logsumexp(fitting, axis=1) - targets).mean()), 'T1 fitting NLL')
        if optimizer['fallback_reason'] is None:
            require(optimizer['success'], 'successful fit marked failed')
            close(t, independent_t, 'temperature vs independent convex beta root', tolerance=2e-6)
            close(optimizer['selected_fit_nll'], independent_nll, 'ID-only optimum NLL', tolerance=2e-12)
        else:
            require(t == 1., 'fallback did not retain T1')
        require(optimizer['selected_fit_nll'] <= optimizer['fit_nll_T1'] + 1e-12, 'fitting NLL increased')
        root_records.append({'seed': seed, 'fitted_T': t, 'independent_beta_root_T': independent_t,
                             'T_absolute_error': abs(t - independent_t), 'beta_derivative_at_fitted_T': beta_derivative(1 / t),
                             'independent_fit_NLL': independent_nll})
        canonical_features = load(original / 'cifar10_scores.npz')
        for variant, temp in [('T1', 1.), ('calibrated', t)]:
            probability_checks += probability_audit(id_data['logits'], id_data['labels'], temp, result['id_test'][variant])
            for subset, rows in [('fit', fit), ('threshold', threshold)]:
                probability_checks += probability_audit(val_logits[rows], y[rows], temp, result['ID_validation_subsets'][subset][variant])
            require(np.array_equal((id_data['logits'].astype(np.float64) / temp).argmax(1), id_data['logits'].argmax(1)), 'argmax changed')
        for dataset in DATASETS:
            raw = load(path / f'{dataset}_scores.npz')
            directory = ROOT / f'runs/cifar100-seed{seed}' / FOLDERS[dataset]
            old_scores = load(directory / f'{dataset}_scores.npz')
            ood_logits = load(directory / f'{dataset}_logits.npz')['logits']
            old_result = read(directory / 'results.json')['datasets'][dataset]
            require(result['datasets'][dataset]['diagnostic_only'] == (dataset == 'noise'), 'synthetic population mislabeled')
            for variant, temp in [('T1', 1.), ('calibrated', t)]:
                expected = {'id': scores(id_data['logits'], temp), 'val': scores(val_logits, temp), 'ood': scores(ood_logits, temp)}
                for method in METHODS:
                    for part in ['id', 'val', 'ood']:
                        actual = raw[f'{variant}_{method}_{part}']
                        if method in expected[part]:
                            require(np.allclose(actual, expected[part][method], rtol=1e-12, atol=2e-12), 'classical scores differ from independent SciPy')
                        else:
                            reference = old_scores[f'{method}_ood'] if part == 'ood' else canonical_features[f'{method}_{part}']
                            require(np.array_equal(actual, reference), 'feature control differs from saved frozen scores')
                        require(np.isfinite(actual).all() and actual.ndim == 1, 'invalid score array')
                        score_checks += 1
                    require(np.array_equal(raw[f'{variant}_{method}_threshold'], raw[f'{variant}_{method}_val'][threshold]), 'threshold uses wrong rows')
                    measured = independent_metrics(raw[f'{variant}_{method}_id'], raw[f'{variant}_{method}_ood'], raw[f'{variant}_{method}_threshold'])
                    saved = result['datasets'][dataset]['variants'][variant][method]
                    for key, value in measured.items():
                        close(saved[key], value, f'{dataset}/{variant}/{method}/{key}')
                        if key in ('auroc', 'aupr_out', 'aupr_in'):
                            independent_checks += 1
                        else:
                            threshold_checks += 1
            for method in ('maxlogit', 'knn', 'mahalanobis'):
                for part in ('id', 'val', 'ood', 'threshold'):
                    require(np.array_equal(raw[f'T1_{method}_{part}'], raw[f'calibrated_{method}_{part}']), 'negative control changed')
                require(result['datasets'][dataset]['variants']['T1'][method] == result['datasets'][dataset]['variants']['calibrated'][method], 'control metrics changed')
            for method in METHODS:
                precision = result['precision_and_original_thresholds'][dataset][method]
                require(precision['original_5000_validation_threshold_metrics'] == old_result['methods'][method], 'original full-validation threshold hidden/changed')
                for part in ('id', 'val', 'ood'):
                    expected = float(np.max(np.abs(raw[f'T1_{method}_{part}'].astype(np.float64) - old_scores[f'{method}_{part}'])))
                    close(precision['max_abs_score_difference'][part], expected, 'precision delta')
                    precision_checks += 1
                full = independent_metrics(raw[f'T1_{method}_id'], raw[f'T1_{method}_ood'], raw[f'T1_{method}_val'])
                for key, old in old_result['methods'][method].items():
                    close(precision['T1_float64_or_canonical_ID_full_5000_validation_metrics'][key], full[key], 'full-validation control metric')
                    close(precision['metric_delta_T1_minus_original'][key], full[key] - old, 'precision metric delta')
                    precision_checks += 2
        result_hashes[f'seed{seed}'] = digest(path / 'results.json')
    REPORT.mkdir(parents=True, exist_ok=True)
    output = {'state': 'passed', 'audited_at': datetime.now(timezone.utc).isoformat(),
              'original_artifact_hash_checks': len(manifest['original_artifacts']),
              'independent_AUROC_AP_checks': independent_checks, 'threshold_count_prevalence_checks': threshold_checks,
              'probability_endpoint_checks': probability_checks, 'raw_score_formula_control_checks': score_checks,
              'precision_and_original_threshold_checks': precision_checks,
              'independent_temperature_checks': root_records, 'results_sha256': result_hashes,
              'inputs_sha256': digest(RUN / 'inputs.json'), 'split_sha256': digest(RUN / 'calibration_split.npz'),
              'audit_source_sha256': digest(__file__),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__},
              'boundary': 'ID-only convex beta root vs log(T) minimizer; all four natural OOD and separate noise diagnostic; no test-selected parameters'}
    (REPORT / 'audit.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
