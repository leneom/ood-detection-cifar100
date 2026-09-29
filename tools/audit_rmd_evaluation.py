"""Independent RMD Cholesky distances, sklearn metrics and Python contingency audit."""
from collections import Counter
import argparse
import math
import statistics
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from sklearn.metrics import roc_auc_score, average_precision_score
from geometry_io import ROOT, RUN, CALIBRATION, METHODS, ADDITIONS, DATASETS, NATURAL, FOLDERS, digest, read, write, load, require, now

RUN = ROOT / 'runs/rmd-phase4'
GEOMETRY = ROOT / 'runs/geometry-phase3'
TARGETS = [.8, .9, .95, .99]
CONTROLS = METHODS + ADDITIONS
NEW = ['rmd','mahalanobis_cpu']
ALL = CONTROLS + NEW
COUNTS = dict(cifar10=10000, mnist=10000, svhn=26032, textures=1880, noise=10000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='evaluation')
    args = parser.parse_args()
    status = read(RUN / 'status.json')
    require(status['state'] in ['evaluation_complete_pending_independent_audit', 'complete'] and
            status['completed_evaluation_seeds'] == [0, 1, 2] and status['output_name'] == args.output_name, 'producer incomplete')
    manifest_path = RUN / f'{args.output_name}-inputs.json'
    manifest = read(manifest_path)
    for name, expected in manifest['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'scoring input changed: ' + name)
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / name) == digest(RUN / f'{args.output_name}-source' / Path(name).name) == expected, 'executed source mismatch')
    require(read(RUN / 'fit-audit.json')['state'] == 'passed', 'fit audit missing')
    require(read(RUN / 'predeclaration.json')['declared_at'] < read(RUN / 'fit-inputs.json')['prepared_at'] <
            manifest['prepared_at'], 'fit/scoring chronology invalid')
    split = load(CALIBRATION / 'calibration_split.npz')
    rows = split['threshold_rows']
    require(len(rows) == 2500 and not np.intersect1d(rows, split['fit_rows']).size, 'threshold/temperature rows overlap')
    checks = dict(score_arrays=0, intermediate_arrays=0, raw_metrics=0, ranking_metrics=0,
                  contingency_fields=0, masks=0, control_arrays=0, control_metric_fields=0, nested_masks=0)
    hashes, diagnostics = {}, []

    def close(saved, actual, label, tolerance=1e-12):
        require(saved is None if actual is None else saved is not None and abs(saved - actual) <= tolerance, label)

    def array(saved, actual, label, category, tolerance=1e-8):
        require(saved.shape == actual.shape and np.isfinite(saved).all() and
                np.allclose(saved, actual, rtol=1e-9, atol=tolerance), label)
        checks[category] += 1

    def fields(saved, actual, label, category):
        for key, value in actual.items():
            close(saved[key], value, label + '/' + key)
            checks[category] += 1

    for seed in [0, 1, 2]:
        directory = RUN / f'seed{seed}'
        output = directory / args.output_name
        result = read(output / 'results.json')
        require(result['state'] == 'complete' and result['seed'] == seed and result['methods'] == ALL and
                result['grid_targets'] == TARGETS and result['primary_calibration_count'] == 2500 and
                result['score_direction'] == 'higher_is_ood' and result['temperature'] == 1., 'scoring configuration changed')
        require(result['inputs_sha256'] == digest(manifest_path) and result['evaluated_at'] > manifest['prepared_at'], 'seed provenance changed')
        require(result['named_additions']==['rmd'] and result['numeric_controls']==['affine_energy','mahalanobis_cpu'] and result['all_ten_phase3_controls_preserved'],'RMD/control identities changed')
        for name, expected in result['output_file_sha256'].items():
            require(digest(output / name) == expected, 'saved score/intermediate archive changed')
            hashes[str((output / name).relative_to(ROOT)).replace('\\', '/')] = expected
        hashes[str((output / 'results.json').relative_to(ROOT)).replace('\\', '/')] = digest(output / 'results.json')
        state, fit = load(directory / 'fit-state.npz'), read(directory / 'fit.json')
        require(result['fit_json_sha256'] == digest(directory / 'fit.json') and result['checkpoint_sha256'] == fit['checkpoint_sha256'], 'fit/checkpoint mismatch')
        within_lower = cholesky(state['within_covariance'],lower=True)
        background_lower = cholesky(state['background_covariance'],lower=True)
        white_means = solve_triangular(within_lower,(state['class_means']-state['background_mean']).T,lower=True).T
        main = ROOT / f'runs/cifar100-seed{seed}/eval-final'
        id_original, val_original = load(main / 'id_logits.npz'), load(main / 'validation_logits.npz')
        features = load(main / 'features.npz')
        correct = id_original['logits'].argmax(1) == id_original['labels']
        close(result['original_ID_classifier_accuracy'], float(correct.mean()), 'classifier accuracy changed')
        independent = {}
        for population in ['id', 'validation'] + DATASETS:
            if population in ['id', 'validation']:
                f = features[population].astype(float)
                original = id_original['logits'] if population == 'id' else val_original['logits']
                truth = id_original['labels'] if population == 'id' else split['validation_labels']
            else:
                source = ROOT / f'runs/cifar100-seed{seed}' / FOLDERS[population]
                f = load(source / f'{population}_features.npz')['features'].astype(float)
                original = load(source / f'{population}_logits.npz')['logits']
                truth = None
                require(len(f) == COUNTS[population], 'wrong OOD population')
            centered = f-state['background_mean']
            white = solve_triangular(within_lower,centered.T,lower=True).T
            distances = np.empty((len(f),100),dtype=float)
            for k,mean in enumerate(white_means):
                difference = white-mean
                distances[:,k] = np.einsum('ni,ni->n',difference,difference)
            bg_white = solve_triangular(background_lower,centered.T,lower=True).T
            background = np.einsum('ni,ni->n',bg_white,bg_white)
            md = distances.min(axis=1)
            signed = md-background
            middle = load(output / f'{population}_intermediates.npz')
            require(middle['class_distances'].shape==(len(f),100) and middle['class_distances'].dtype==np.float64,'class-distance population/precision changed')
            for key,actual in dict(class_distances=distances,background_distance=background,mahalanobis_cpu=md,rmd=signed).items():
                require(np.allclose(middle[key],actual,atol=1e-8,rtol=1e-9),'independent Cholesky distance failed: '+key)
                checks['intermediate_arrays'] += 1
            require(np.array_equal(middle['nearest_class'],middle['class_distances'].argmin(axis=1)),'nearest class index not original-order argmin')
            independent[population]=dict(mahalanobis_cpu=md,rmd=signed)
            offsets=state['class_means']-state['background_mean']
            projected=centered@state['within_precision']
            raw_class=(projected*centered).sum(axis=1)[:,None]-2*projected@offsets.T+((offsets@state['within_precision'])*offsets).sum(axis=1)[None,:]
            raw_bg=((centered@state['background_precision'])*centered).sum(axis=1)
            require(raw_class.min()>=-1e-8 and raw_bg.min()>=-1e-8,'roundoff gate violated')
            diag={'n':len(f),'negative_class_distance_count':int((raw_class<0).sum()),
                  'max_class_roundoff_correction':float(-raw_class[raw_class<0].min()) if (raw_class<0).any() else 0.,
                  'negative_background_distance_count':int((raw_bg<0).sum()),
                  'max_background_roundoff_correction':float(-raw_bg[raw_bg<0].min()) if (raw_bg<0).any() else 0.,
                  'raw_min_class_distance':float(raw_class.min()),'raw_min_background_distance':float(raw_bg.min()),
                  'rmd_min':float(middle['rmd'].min()),'rmd_max':float(middle['rmd'].max()),
                  'MD_control_min':float(middle['mahalanobis_cpu'].min())}
            if truth is not None:
                require(np.array_equal(middle['original_labels'],truth) and np.array_equal(middle['original_predictions'],original.argmax(1)),'original ID predictions/labels changed')
                diag['original_accuracy']=float((original.argmax(1)==truth).mean())
            fields(result['intermediate_diagnostics'][population],diag,'roundoff/population diagnostics','raw_metrics')
        raw = {d: load(output / f'{d}_scores.npz') for d in DATASETS}
        old_result = read(GEOMETRY / f'seed{seed}/evaluation/results.json')
        for dataset in DATASETS:
            old = load(GEOMETRY / f'seed{seed}/evaluation/{dataset}_scores.npz')
            require(result['datasets'][dataset]['diagnostic_only'] == (dataset == 'noise'), 'synthetic boundary changed')
            for method in ALL:
                saved = raw[dataset]
                if method in CONTROLS:
                    for part in ['id', 'ood', 'val', 'threshold']:
                        require(np.array_equal(saved[f'{method}_{part}'], old[f'{method}_{part}']), 'T1 control changed')
                        checks['control_arrays'] += 1
                else:
                    for part, pop in [('id', 'id'), ('ood', dataset), ('val', 'validation')]:
                        require(saved[f'{method}_{part}'].dtype == np.float64, 'new scores not float64')
                        array(saved[f'{method}_{part}'], independent[pop][method], 'independent new score', 'score_arrays')
                require(np.array_equal(saved[f'{method}_threshold'], saved[f'{method}_val'][rows]), 'wrong primary threshold rows')
                for part in ['id', 'val', 'threshold']:
                    require(np.array_equal(saved[f'{method}_{part}'], raw['cifar10'][f'{method}_{part}']), 'ID population varies across OOD datasets')
                inside, outside = saved[f'{method}_id'], saved[f'{method}_ood']
                cal, full = saved[f'{method}_threshold'], saved[f'{method}_val']
                require(len(inside) == 10000 and len(outside) == COUNTS[dataset] and len(cal) == 2500 and len(full) == 5000, 'wrong score count')
                labels = np.r_[np.zeros(len(inside), dtype=int), np.ones(len(outside), dtype=int)]
                values = np.r_[inside, outside]
                benchmark = sorted(inside.tolist())[math.ceil(.95 * len(inside)) - 1]
                threshold = sorted(cal.tolist())[math.ceil(.95 * len(cal)) - 1]
                actual = dict(auroc=roc_auc_score(labels, values), aupr_out=average_precision_score(labels, values),
                              aupr_in=average_precision_score(1 - labels, -values), fpr95_id=float((outside <= benchmark).mean()),
                              id_tpr_at_fpr95=float((inside <= benchmark).mean()), benchmark_threshold=benchmark,
                              n_id=len(inside), n_ood=len(outside), ood_prevalence=len(outside) / len(values),
                              validation_threshold=threshold, id_test_acceptance_at_validation_threshold=float((inside <= threshold).mean()),
                              ood_acceptance_at_validation_threshold=float((outside <= threshold).mean()), calibration_id_acceptance=float((cal <= threshold).mean()),
                              benchmark_ID_equal_threshold_count=int((inside == benchmark).sum()),
                              threshold_calibration_equal_threshold_count=int((cal == threshold).sum()), threshold_calibration_n=2500)
                primary = result['datasets'][dataset]['methods'][method]['primary_2500']
                fields(primary, actual, 'independent ranking/threshold metric', 'raw_metrics')
                checks['ranking_metrics'] += 3
                if method in CONTROLS:
                    fields(primary, {k: old_result['datasets'][dataset]['methods'][method]['primary_2500'][k] for k in primary},
                           'matched control metrics', 'control_metric_fields')
                threshold = sorted(full.tolist())[math.ceil(.95 * len(full)) - 1]
                fields(result['datasets'][dataset]['methods'][method]['secondary_5000_threshold'],
                       dict(validation_threshold=threshold, calibration_id_acceptance=float((full <= threshold).mean()),
                            id_test_acceptance_at_validation_threshold=float((inside <= threshold).mean()),
                            ood_acceptance_at_validation_threshold=float((outside <= threshold).mean())), 'full validation secondary', 'raw_metrics')
        masks = load(output / 'decisions.npz')
        require(np.array_equal(masks['id_correct'], correct) and np.array_equal(masks['threshold_rows'], rows), 'decision correctness/split changed')
        records = {(r['method'], r['target']): r for r in result['operating_points']}
        require(len(records) == 48 and len(result['operating_points']) == 48, 'missing/duplicate grid point')
        for mi, method in enumerate(ALL):
            previous_masks = None
            for ti, target in enumerate(TARGETS):
                point = records[(method, target)]
                cal = raw['cifar10'][f'{method}_threshold'].tolist()
                inside = raw['cifar10'][f'{method}_id'].tolist()
                index = math.ceil(target * len(cal)) - 1
                threshold = sorted(cal)[index]
                require(point['threshold'] == threshold, 'grid threshold differs from sorting')
                cal_mask, id_mask = [s <= threshold for s in cal], [s <= threshold for s in inside]
                fields(point['calibration'], dict(n=len(cal), accepted=sum(cal_mask), rejected=len(cal)-sum(cal_mask),
                       achieved_acceptance=sum(cal_mask)/len(cal), equal_threshold_count=sum(s == threshold for s in cal),
                       order_statistic_index=index), 'calibration counts', 'contingency_fields')
                counts = Counter(zip(correct.tolist(), id_mask))
                ca, ia, cr, ir = [counts[key] for key in [(True, True), (False, True), (True, False), (False, False)]]
                an, cn, inn = ca+ia, ca+cr, ia+ir
                ratio = lambda n, d: n/d if d else None
                fields(point['id_test'], dict(n=len(inside), accepted=an, rejected=cr+ir, correct_total=cn, incorrect_total=inn,
                       correct_accepted=ca, incorrect_accepted=ia, correct_rejected=cr, incorrect_rejected=ir,
                       coverage=an/len(inside), retained_error_risk=ratio(ia,an), accepted_accuracy=ratio(ca,an),
                       unconditional_accuracy=cn/len(inside), correct_rejection_fraction=ratio(cr,cn),
                       incorrect_rejection_fraction=ratio(ir,inn), equal_threshold_count=sum(s == threshold for s in inside)),
                       'original classifier contingency', 'contingency_fields')
                expected_masks = dict(id_accepted=id_mask, calibration_accepted=cal_mask)
                for dataset in DATASETS:
                    outside = raw[dataset][f'{method}_ood'].tolist()
                    mask = [s <= threshold for s in outside]
                    expected_masks[f'{dataset}_accepted'] = mask
                    fields(point['ood'][dataset], dict(n=len(outside), accepted=sum(mask), rejected=len(outside)-sum(mask),
                           acceptance=sum(mask)/len(outside), equal_threshold_count=sum(s == threshold for s in outside)),
                           'OOD counts', 'contingency_fields')
                close(point['equal_weight_natural_ood_acceptance'], statistics.fmean(point['ood'][d]['acceptance'] for d in NATURAL), 'natural macro')
                for key, actual in expected_masks.items():
                    require(masks[key].dtype == np.bool_ and np.array_equal(masks[key][mi,ti], actual), 'decision mask changed')
                    checks['masks'] += 1
                    if previous_masks is not None:
                        require(not np.any(np.asarray(previous_masks[key]) & ~np.asarray(actual)), 'non-nested acceptance')
                        checks['nested_masks'] += 1
                previous_masks = expected_masks
        diagnostics.append({'seed': seed, 'ID_original_accuracy': result['original_ID_classifier_accuracy'],
                            'ID_intermediates': result['intermediate_diagnostics']['id']})
        print('seed', seed, 'independent RMD evaluation passed', flush=True)
    out = ROOT / 'report/rmd-phase4'
    audit = {'state': 'passed', 'audited_at': now(), 'checks': checks,
             'original_input_hash_checks': len(manifest['original_artifacts']), 'output_name': args.output_name,
             'output_artifacts': hashes, 'evaluation_inputs_sha256': digest(manifest_path),
             'fit_audit_sha256': digest(RUN / 'fit-audit.json'), 'independent_source_sha256': digest(__file__),
             'per_seed_diagnostics': diagnostics,
             'boundary': 'all named detectors and numeric controls; original classifier correctness; noise excluded from natural macros'}
    write(out / 'audit.json', audit)
    print(__import__('json').dumps({k:v for k,v in audit.items() if k != 'output_artifacts'}), flush=True)


if __name__ == '__main__':
    main()
