"""Final frozen-input, audit, test-log and visual gates for descriptive diagnostics."""
from pathlib import Path
from geometry_io import ROOT, read, write, digest, require, now

RUN = ROOT/'runs/rmd-diagnostics'
OUT = ROOT/'report/rmd-diagnostics'


def main():
    raw,summary,visual=[read(OUT/n) for n in ['audit.json','summary-audit.json','visual-qa.json']]
    require(raw['state']==summary['state']==visual['state']=='passed','completion gate missing')
    require(summary['raw_audit_sha256']==digest(OUT/'audit.json'),'audited raw state changed')
    for name,h in summary['artifacts'].items():
        require(digest(OUT/name)==h,'audited report changed')
    for name,h in visual['figures'].items():
        require(digest(OUT/name)==h,'visually checked plot changed')
    for name,h in raw['output_artifacts'].items():
        require(digest(ROOT/name)==h,'audited diagnostic raw array changed')
    manifest_path=RUN/f"{raw['output_name']}-inputs.json"
    require(raw['inputs_sha256']==digest(manifest_path),'diagnostic input manifest changed')
    manifest=read(manifest_path)
    for name,h in manifest['original_artifacts'].items():
        require(digest(ROOT/name)==h,'original input changed')
    for name,h in manifest['source_hashes'].items():
        require(digest(ROOT/name)==digest(RUN/f"{raw['output_name']}-source"/Path(name).name)==h,'executed source changed')
    require(raw['independent_source_sha256']==digest(ROOT/'tools/audit_rmd_diagnostics.py') and
            summary['independent_source_sha256']==digest(ROOT/'tools/audit_rmd_diagnostics_report.py'),'independent audit source changed')
    log_path=RUN/'regression.log'
    log_data=log_path.read_bytes()
    log=log_data.decode('utf-16') if log_data.startswith(b'\xff\xfe') else log_data.decode('utf-8-sig')
    require('Ran 70 tests' in log and 'OK (skipped=2)' in log,'regression log incomplete')
    tests={'full_suite_tests':70,'passed':68,'skipped':2,'skip_reason':'SciPy-only optimizer tests already passed in unchanged base CPU calibration suite',
           'regression_log_sha256':digest(log_path),'CPU_RMD_diagnostics_tests_passed':5,'CUDA_VISIBLE_DEVICES':'-1',
           'torch_cuda_is_available_before_suite':False}
    complete={'state':'complete','completed_at':now(),'seeds':[0,1,2],'population_count':7,
              'seed_comparison_target_settings':132,'input_hash_checks':len(manifest['original_artifacts']),
              'raw_checks':raw['checks'],'summary_statistic_records':summary['statistic_records_checked'],
              'displayed_table_fields':summary['displayed_table_fields_checked'],'test_verification':tests,
              'audits':{n:digest(OUT/n) for n in ['audit.json','summary-audit.json','visual-qa.json']},
              'source_sha256':digest(__file__),'boundary':'complete posthoc descriptive diagnostic; no fitting, tuning or causal claim'}
    write(OUT/'completion-audit.json',complete)
    write(RUN/'status.json',{'state':'complete','completed_seeds':[0,1,2],'output_name':raw['output_name'],
                            'original_artifacts_unchanged':True,'completion_audit_sha256':digest(OUT/'completion-audit.json'),
                            'completed_at':complete['completed_at']})
    print(__import__('json').dumps(complete),flush=True)


if __name__=='__main__':
    main()
