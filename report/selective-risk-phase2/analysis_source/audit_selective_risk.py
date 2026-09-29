"""Independently reconstruct sorted thresholds and decisions using Python counts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/selective-risk-phase2'
CALIBRATION = ROOT / 'runs/calibration-phase2'
OUT = ROOT / 'report/selective-risk-phase2'
METHODS = ['msp', 'energy', 'maxlogit', 'entropy', 'knn', 'mahalanobis']
VARIANTS = ['T1', 'calibrated']
TARGETS = [.8, .9, .95, .99]
NATURAL = ['cifar10', 'mnist', 'svhn', 'textures']
DATASETS = NATURAL + ['noise']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(n, d):
    return n / d if d else None


def main():
    status = read(RUN / 'status.json')
    require(status['state'] == 'complete' and status['completed_seeds'] == [0, 1, 2] and status['operating_points'] == 144,
            'producer incomplete')
    manifest = read(RUN / 'inputs.json')
    for name, expected in manifest['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'original input changed: ' + name)
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / 'tools' / name) == digest(RUN / 'analysis_source' / name) == expected, 'producer source changed')
    prereg = read(RUN / 'predeclaration.json')
    require(prereg['declared_at'] < manifest['prepared_at'], 'analysis predates inputs')
    require(prereg['ID_acceptance_targets'] == TARGETS and prereg['expected_seeds'] == [0, 1, 2], 'settings changed')
    split = load(CALIBRATION / 'calibration_split.npz')
    require(len(split['threshold_rows']) == 2500 and not np.intersect1d(split['fit_rows'], split['threshold_rows']).size,
            'threshold/temperature fit overlap')
    checks = masks_checked = parity95_checked = controls_checked = nested_checks = 0
    result_hashes = {}
    def check(saved, actual, label):
        nonlocal checks
        for key, value in actual.items():
            if value is None:
                require(saved[key] is None, label + '/' + key + ': undefined denominator replaced')
            else:
                require(abs(saved[key] - value) < 1e-12, label + '/' + key + ': count/metric mismatch')
            checks += 1
    def check_mask(saved, expected, label):
        nonlocal masks_checked
        require(saved.dtype == np.bool_ and np.array_equal(saved, expected), label + ': raw acceptance mask mismatch')
        masks_checked += 1
    for seed in [0, 1, 2]:
        directory = RUN / f'seed{seed}'
        result = read(directory / 'results.json')
        require(result['state'] == 'complete' and result['seed'] == seed and result['n_operating_points'] == 48, 'seed incomplete')
        require(result['computed_at'] > manifest['prepared_at'] and result['inputs_sha256'] == digest(RUN / 'inputs.json'), 'seed inputs mismatch')
        require(result['methods'] == METHODS and result['variants'] == VARIANTS and result['targets'] == TARGETS, 'decision axes changed')
        require(digest(directory / 'decisions.npz') == result['decisions_sha256'], 'saved decision archive changed')
        decisions = load(directory / 'decisions.npz')
        source = load(ROOT / f'runs/cifar100-seed{seed}/eval-final/id_logits.npz')
        prediction = source['logits'].argmax(axis=1)
        truth = (prediction == source['labels']).tolist()
        require(np.array_equal(decisions['id_labels'], source['labels']) and np.array_equal(decisions['id_predictions'], prediction),
                'frozen labels/predictions changed')
        require(np.array_equal(decisions['id_correct'], truth), 'correctness flags changed')
        require(hashlib.sha256(decisions['id_correct'].tobytes()).hexdigest() == result['id_correct_flags_sha256'], 'correctness hash changed')
        require(np.array_equal(decisions['threshold_rows'], split['threshold_rows']), 'threshold rows changed')
        previous = read(CALIBRATION / f'seed{seed}/results.json')
        require(result['checkpoint_sha256'] == previous['checkpoint_sha256'], 'checkpoint changed')
        require(result['calibrated_temperature'] == previous['temperature_fit']['temperature'], 'temperature changed')
        raw = {d: load(CALIBRATION / f'seed{seed}/{d}_scores.npz') for d in DATASETS}
        records = {(r['variant'], r['method'], r['target']): r for r in result['operating_points']}
        require(len(records) == 48, 'duplicated or missing grid point')
        for vi, variant in enumerate(VARIANTS):
            for mi, method in enumerate(METHODS):
                inside = raw['cifar10'][f'{variant}_{method}_id'].tolist()
                calibration_array = raw['cifar10'][f'{variant}_{method}_threshold']
                calibration = calibration_array.tolist()
                require(np.array_equal(calibration_array, raw['cifar10'][f'{variant}_{method}_val'][split['threshold_rows']]), 'wrong calibration rows')
                for ti, target in enumerate(TARGETS):
                    saved = records[(variant, method, target)]
                    index = int(np.ceil(target * len(calibration))) - 1
                    threshold = sorted(calibration)[index]
                    require(saved['threshold'] == threshold, 'order statistic differs from independent sorting')
                    ca_mask = [x <= threshold for x in calibration]
                    ca_count = sum(ca_mask)
                    check(saved['calibration'], {'n': len(calibration), 'accepted': ca_count, 'rejected': len(calibration) - ca_count,
                          'achieved_acceptance': ca_count / len(calibration), 'equal_threshold_count': sum(x == threshold for x in calibration),
                          'order_statistic_index': index}, 'calibration')
                    require(ca_count / len(calibration) >= target, 'target ID calibration acceptance not achieved')
                    check_mask(decisions['calibration_accepted'][vi, mi, ti], ca_mask, 'ID calibration')
                    accepted = [x <= threshold for x in inside]
                    counts = Counter(zip(truth, accepted))
                    ca, ia, cr, ir = counts[(True, True)], counts[(False, True)], counts[(True, False)], counts[(False, False)]
                    accepted_n, correct_n, incorrect_n = ca + ia, ca + cr, ia + ir
                    independent_id = {'n': len(inside), 'accepted': accepted_n, 'rejected': cr + ir,
                                      'correct_total': correct_n, 'incorrect_total': incorrect_n,
                                      'correct_accepted': ca, 'incorrect_accepted': ia, 'correct_rejected': cr, 'incorrect_rejected': ir,
                                      'coverage': accepted_n / len(inside), 'retained_error_risk': ratio(ia, accepted_n),
                                      'accepted_accuracy': ratio(ca, accepted_n), 'unconditional_accuracy': correct_n / len(inside),
                                      'correct_rejection_fraction': ratio(cr, correct_n), 'incorrect_rejection_fraction': ratio(ir, incorrect_n),
                                      'equal_threshold_count': sum(x == threshold for x in inside)}
                    check(saved['id_test'], independent_id, 'ID test')
                    check_mask(decisions['id_accepted'][vi, mi, ti], accepted, 'ID test')
                    outside_acceptance = []
                    for dataset in DATASETS:
                        outside = raw[dataset][f'{variant}_{method}_ood']
                        mask = [x <= threshold for x in outside.tolist()]
                        count = sum(mask)
                        check(saved['ood'][dataset], {'n': len(mask), 'accepted': count, 'rejected': len(mask) - count,
                              'acceptance': count / len(mask), 'equal_threshold_count': sum(x == threshold for x in outside.tolist())}, dataset)
                        check_mask(decisions[f'{dataset}_accepted'][vi, mi, ti], mask, dataset)
                        entry = saved['ood'][dataset]
                        require(entry['diagnostic_only'] == (dataset == 'noise'), 'synthetic population mislabeled')
                        require(entry['source_score_file_sha256'] == digest(CALIBRATION / f'seed{seed}/{dataset}_scores.npz'), 'score file hash changed')
                        require(entry['score_array_sha256'] == hashlib.sha256(outside.tobytes()).hexdigest() and entry['score_dtype'] == str(outside.dtype), 'score bytes/dtype changed')
                        require(entry['id_ood_population_prevalence'] == len(mask) / (len(inside) + len(mask)), 'population prevalence changed')
                        for part in ['id', 'val', 'threshold']:
                            require(np.array_equal(raw[dataset][f'{variant}_{method}_{part}'], raw['cifar10'][f'{variant}_{method}_{part}']),
                                    'canonical ID scores differ between populations')
                        if dataset in NATURAL:
                            outside_acceptance.append(count / len(mask))
                        if target == .95:
                            old = previous['datasets'][dataset]['variants'][variant][method]
                            require(threshold == old['validation_threshold'], '95% threshold lost parity')
                            require(independent_id['coverage'] == old['id_test_acceptance_at_validation_threshold'], '95% ID coverage lost parity')
                            require(count / len(mask) == old['ood_acceptance_at_validation_threshold'], '95% OOD acceptance lost parity')
                            parity95_checked += 3
                    require(abs(saved['equal_weight_natural_ood_acceptance'] - statistics.fmean(outside_acceptance)) < 1e-12,
                            'natural macro wrong or contains noise')
                for masks in [decisions['id_accepted'], decisions['calibration_accepted']] + [decisions[f'{d}_accepted'] for d in DATASETS]:
                    for ti in range(1, 4):
                        require(not np.any(masks[vi, mi, ti - 1] & ~masks[vi, mi, ti]), 'non-nested grid decisions')
                        nested_checks += 1
        for method in ['maxlogit', 'knn', 'mahalanobis']:
            for target in TARGETS:
                left = {k: v for k, v in records[('T1', method, target)].items() if k != 'variant'}
                right = {k: v for k, v in records[('calibrated', method, target)].items() if k != 'variant'}
                require(left == right, 'negative control counts/thresholds changed')
                controls_checked += 1
        result_hashes[f'seed{seed}'] = digest(directory / 'results.json')
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {'state': 'passed', 'audited_at': datetime.now(timezone.utc).isoformat(),
             'seeds': [0, 1, 2], 'operating_points': 144, 'original_artifact_hash_checks': len(manifest['original_artifacts']),
             'independent_count_and_metric_checks': checks, 'per_sample_decision_masks_checked': masks_checked,
             '95pct_phase2_parity_checks': parity95_checked, 'unchanged_control_operating_points_checked': controls_checked,
             'nested_acceptance_checks': nested_checks, 'results_sha256': result_hashes,
             'inputs_sha256': digest(RUN / 'inputs.json'), 'audit_source_sha256': digest(__file__),
             'boundary': 'Python sorted order statistic and Counter ID errors; all four predeclared targets; OOD labels never used; noise excluded from natural macro'}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
