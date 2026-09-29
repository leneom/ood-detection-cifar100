"""Independent Python-statistics reconstruction of every report summary record."""
import statistics
from pathlib import Path
from geometry_io import ROOT, RUN, METHODS, ADDITIONS, DATASETS, NATURAL, read, write, digest, require, now

RUN = ROOT / 'runs/ash-phase5'
OUT = ROOT / 'report/ash-phase5'
REFERENCES = METHODS+ADDITIONS+['rmd','mahalanobis_cpu']
NEW = ['ash_p_energy','ash_b_energy','ash_s_energy']
ALL = REFERENCES+NEW
TARGETS = [.8, .9, .95, .99]
ENDPOINTS = ['auroc', 'aupr_out', 'aupr_in', 'fpr95_id', 'id_test_acceptance_at_validation_threshold', 'ood_acceptance_at_validation_threshold']


def main():
    raw_audit, summary = read(OUT / 'audit.json'), read(OUT / 'summary.json')
    require(raw_audit['state'] == 'passed' and summary['raw_audit_sha256'] == digest(OUT / 'audit.json'), 'raw audit changed')
    require(summary['expected_seeds'] == [0,1,2] and summary['methods'] == ALL and summary['natural_datasets'] == NATURAL and
            summary['synthetic_diagnostics'] == ['noise'] and summary['threshold_calibration_n'] == 2500, 'summary population/settings changed')
    require(summary['source_sha256'] == digest(ROOT / 'tools/build_ash_report.py'), 'report producer changed')
    for name, expected in raw_audit['output_artifacts'].items():
        require(digest(ROOT / name) == expected, 'audited raw result changed')
    attempt = raw_audit['output_name']
    results = [read(RUN / f'seed{s}/{attempt}/results.json') for s in [0,1,2]]
    require(summary['result_sha256'] == [digest(RUN / f'seed{s}/{attempt}/results.json') for s in [0,1,2]], 'summary seed source changed')
    fits = [read(RUN / f'seed{s}/preflight.json') for s in [0,1,2]]
    checks = 0
    table_checks = 0
    def stat(saved, values, name):
        nonlocal checks
        values = list(values)
        require(len(values) == 3 and len(saved['per_seed']) == 3 and
                all(abs(a-b) < 1e-12 for a,b in zip(values, saved['per_seed'])), name + ': seed values differ')
        require(abs(statistics.fmean(values)-saved['mean']) < 1e-12 and
                abs(statistics.stdev(values)-saved['sample_sd']) < 1e-12, name + ': mean/sample SD differ')
        checks += 1
    def raw(dataset, method, metric):
        return [r['datasets'][dataset]['methods'][method]['primary_2500'][metric] for r in results]
    def macro(method, metric):
        return [statistics.fmean(r['datasets'][d]['methods'][method]['primary_2500'][metric] for d in NATURAL) for r in results]
    require(set(summary['per_dataset']) == set(DATASETS), 'missing/extra dataset')
    for dataset in DATASETS:
        require(set(summary['per_dataset'][dataset]) == set(ALL), 'missing method')
        for method in ALL:
            for block in ['primary_2500','secondary_5000_threshold']:
                metrics = results[0]['datasets'][dataset]['methods'][method][block]
                require(set(summary['per_dataset'][dataset][method][block]) == set(metrics), 'missing/extra metric')
                for metric in metrics:
                    stat(summary['per_dataset'][dataset][method][block][metric],
                         [r['datasets'][dataset]['methods'][method][block][metric] for r in results], 'dataset statistic')
    require(set(summary['natural_macro']) == set(ALL) and set(summary['paired_changes']) == set(NEW), 'macro/paired method set changed')
    for method in ALL:
        require(set(summary['natural_macro'][method]) == set(ENDPOINTS), 'macro endpoints changed')
        for metric in ENDPOINTS:
            stat(summary['natural_macro'][method][metric], macro(method, metric), 'natural macro')
    for method in NEW:
        require(set(summary['paired_changes'][method]) == set(REFERENCES), 'paired references changed')
        for reference in REFERENCES:
            require(set(summary['paired_changes'][method][reference]) == set(DATASETS + ['natural_macro']), 'paired datasets changed')
            for dataset in DATASETS + ['natural_macro']:
                for metric in ENDPOINTS:
                    a,b = (macro(method,metric),macro(reference,metric)) if dataset == 'natural_macro' else (raw(dataset,method,metric),raw(dataset,reference,metric))
                    stat(summary['paired_changes'][method][reference][dataset][metric], [x-y for x,y in zip(a,b)], 'paired statistic')
    records = [{(p['method'],p['target']): p for p in r['operating_points']} for r in results]
    require(set(summary['operating_grid']) == set(ALL), 'grid methods changed')
    for method in ALL:
        require(set(summary['operating_grid'][method]) == {str(t) for t in TARGETS}, 'grid targets changed')
        for target in TARGETS:
            points = [p[(method,target)] for p in records]
            expected = {f'id_{k}': [p['id_test'][k] for p in points] for k in
                        ['coverage','retained_error_risk','correct_rejection_fraction','incorrect_rejection_fraction']}
            expected['calibration_achieved_acceptance'] = [p['calibration']['achieved_acceptance'] for p in points]
            expected['natural_macro_ood_acceptance'] = [statistics.fmean(p['ood'][d]['acceptance'] for d in NATURAL) for p in points]
            expected.update({f'{d}_ood_acceptance': [p['ood'][d]['acceptance'] for p in points] for d in DATASETS})
            require(set(summary['operating_grid'][method][str(target)]) == set(expected), 'grid endpoint set changed')
            for key,values in expected.items():
                stat(summary['operating_grid'][method][str(target)][key], values, 'grid statistic')
    for key,saved in summary['preflight_statistics'].items():
        stat(saved, [f[key] for f in fits], 'fit statistic')
    for key,saved in summary['ID_intermediate_diagnostics'].items():
        stat(saved, [r['intermediate_diagnostics']['id'][key] for r in results], 'ID intermediate statistic')
    # Audit displayed rounded tables against raw-derived Python statistics.
    report = (OUT / 'REPORT.md').read_text(encoding='utf-8')
    labels = dict(msp='MSP',energy='Energy',maxlogit='MaxLogit',entropy='Entropy',knn='KNN',mahalanobis='Mahalanobis',
                  vim='ViM',residual='Residual',react_energy='ReAct Energy',affine_energy='Affine Energy*',rmd='RMD',mahalanobis_cpu='CPU MD*',ash_p_energy='ASH-P',ash_b_energy='ASH-B',ash_s_energy='ASH-S')
    def pct(values):
        return f'{100*statistics.fmean(values):.2f} ± {100*statistics.stdev(values):.2f}'
    for method in ALL:
        points = [r[(method,.95)] for r in records]
        expected = '| ' + labels[method] + ' | ' + ' | '.join([pct(macro(method,k)) for k in
                   ['auroc','fpr95_id','ood_acceptance_at_validation_threshold']] +
                   [pct([p['id_test'][k] for p in points]) for k in ['coverage','retained_error_risk']]) + ' |'
        require(expected in report, 'displayed macro table mismatch')
        table_checks += 5
        for dataset in NATURAL:
            expected = '| ' + labels[method] + ' | ' + ' | '.join(pct(raw(dataset,method,k)) for k in
                       ['auroc','aupr_out','aupr_in','fpr95_id','ood_acceptance_at_validation_threshold']) + ' |'
            require(expected in report, 'displayed dataset table mismatch')
            table_checks += 5
    for method in NEW:
        for reference in REFERENCES:
            values=[]
            for metric in ['auroc','fpr95_id','ood_acceptance_at_validation_threshold']:
                values.append(pct([a-b for a,b in zip(macro(method,metric),macro(reference,metric))]))
            expected='| '+labels[method]+' − '+labels[reference]+' | '+' | '.join(values)+' |'
            require(expected in report,'displayed paired table mismatch')
            table_checks+=3
    require(summary['named_additions']==NEW and summary['numeric_controls']==['affine_energy','mahalanobis_cpu'],'method identities changed')
    for population,saved in summary['population_intermediate_diagnostics'].items():
        for key,item in saved.items():stat(item,[r['intermediate_diagnostics'][population][key] for r in results],'population intermediate')
    for population in ['id','validation']:
        diag=[r['intermediate_diagnostics'][population] for r in results]
        for method in NEW:
            expected='| '+population+' | '+pct([d['original_accuracy'] for d in diag])+' | '+labels[method]+' | '+pct([d[method+'_accuracy'] for d in diag])+' | '+str([d[method+'_argmax_changes'] for d in diag])+' |'
            require(expected in report,'displayed shaped classification table mismatch')
            table_checks+=5
    outputs = [OUT / name for name in ['summary.json','REPORT.md','natural-auroc.png','natural-auroc.svg','operating-grid.png','operating-grid.svg']]
    require(all(p.exists() and p.stat().st_size > 1000 for p in outputs), 'report artifact missing')
    audit = {'state':'passed','audited_at':now(),'statistic_records_checked':checks,'displayed_table_fields_checked':table_checks,
             'raw_audit_sha256':digest(OUT / 'audit.json'),'artifacts':{p.name:digest(p) for p in outputs},
             'independent_source_sha256':digest(__file__), 'boundary':'Python statistics independently checks within-seed macros, same-seed differences and sample SD; noise excluded'}
    write(OUT / 'summary-audit.json',audit)
    print(__import__('json').dumps(audit),flush=True)


if __name__ == '__main__':
    main()
