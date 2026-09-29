"""Mark this phase complete only after fit, raw, summary and visual gates."""
from pathlib import Path
from geometry_io import ROOT, read, write, digest, require, now
RUN = ROOT / 'runs/ash-phase5'


def main():
    out = ROOT / 'report/ash-phase5'
    raw, summary, visual = [read(out / name) for name in ['audit.json','summary-audit.json','visual-qa.json']]
    require(raw['state'] == summary['state'] == visual['state'] == 'passed', 'completion gate missing')
    require(summary['raw_audit_sha256'] == digest(out / 'audit.json'), 'summary audit input changed')
    require(raw['preflight_audit_sha256'] == digest(RUN / 'preflight-audit.json'), 'fit audit changed')
    for name, expected in summary['artifacts'].items():
        require(digest(out / name) == expected, 'audited summary/report changed')
    for name, expected in visual['figures'].items():
        require(digest(out / name) == expected, 'visually checked figure changed')
    for name, expected in raw['output_artifacts'].items():
        require(digest(ROOT / name) == expected, 'audited raw output changed')
    manifest_path = RUN / f"{raw['output_name']}-inputs.json"
    require(raw['evaluation_inputs_sha256'] == digest(manifest_path), 'scoring manifest changed')
    manifest = read(manifest_path)
    for name, expected in manifest['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'frozen input changed')
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / name) == digest(RUN / f"{raw['output_name']}-source" / Path(name).name) == expected, 'scoring source changed')
    require(raw['independent_source_sha256'] == digest(ROOT / 'tools/audit_ash_evaluation.py') and
            summary['independent_source_sha256'] == digest(ROOT / 'tools/audit_ash_report.py'), 'independent source changed')
    log = (RUN / 'regression.log').read_text(encoding='utf-8')
    require('Ran 75 tests' in log and 'OK (skipped=2)' in log and 'FAILED' not in log,'regression evidence missing')
    completion = {'state':'complete','completed_at':now(),'seeds':[0,1,2],
                  'named_methods': ['msp','energy','maxlogit','entropy','knn','mahalanobis','vim','residual','react_energy','rmd','ash_p_energy','ash_b_energy','ash_s_energy'],
                  'numeric_controls':['affine_energy','mahalanobis_cpu'],'natural_ood':['cifar10','mnist','svhn','textures'],
                  'noise':'diagnostic only, excluded from natural macro', 'raw_checks':raw['checks'],
                  'summary_records_checked':summary['statistic_records_checked'],
                  'frozen_input_hash_checks':len(manifest['original_artifacts']),
                  'audits':{name:digest(out / name) for name in ['audit.json','summary-audit.json','visual-qa.json']},
                  'training_preflight_audit_sha256':digest(RUN / 'preflight-audit.json'),
                  'source_sha256':digest(__file__), 'regression_log_sha256':digest(RUN / 'regression.log'),
                  'regression_tests':{'run':75,'passed':73,'scipy_optimizer_skips_previously_verified':2}, 'boundary':'local exploratory extension; all parameters fixed before new OOD scoring'}
    write(out / 'completion-audit.json',completion)
    write(RUN / 'status.json', {'state':'complete', 'completed_evaluation_seeds':[0,1,2], 'output_name':raw['output_name'],
                               'original_artifacts_unchanged':True,'completion_audit_sha256':digest(out / 'completion-audit.json'),
                               'completed_at':completion['completed_at']})
    print(__import__('json').dumps(completion),flush=True)


if __name__ == '__main__':
    main()
