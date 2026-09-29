"""Complete only after all fixed rows, scalar audit, visual QA and regression."""
from pathlib import Path
from geometry_io import ROOT,read,write,digest,require,now

RUN=ROOT/'runs/bootstrap-dependence';REPORT=ROOT/'report/bootstrap-dependence'

def main():
    status=read(RUN/'status.json');require(status['state']=='raw_complete_pending_independent_audit' and status['completed_endpoints']==66 and status['method_or_pair_decompositions']==7920 and status['natural_decompositions']==360,'producer incomplete')
    attempt=status['output_name'];out=RUN/attempt;pre=read(RUN/'predeclaration.json');manifest=read(RUN/f'{attempt}-inputs.json');raw=read(out/'raw-manifest.json');audit=read(RUN/'raw-audit.json')
    require(audit['state']=='passed' and audit['analysis_inputs_sha256']==digest(RUN/f'{attempt}-inputs.json') and audit['raw_manifest_sha256']==digest(out/'raw-manifest.json') and audit['independent_source_sha256']==digest(ROOT/'tools/audit_bootstrap_dependence.py'),'independent full raw audit unbound')
    required={'endpoint_values':2970000,'endpoint_covariance_cells':133650,'endpoint_correlation_cells':133650,'endpoint_mean_count_cells':2970,
              'model_or_pair_decompositions':7920,'model_or_pair_covariance_cells':71280,'natural_decompositions':360,'natural_covariance_cells':5760,'json_fields':217980}
    require(audit['checks']==required and audit['bound_input_count']==786,'independent full raw coverage incomplete')
    require(manifest['original_artifacts']==pre['original_artifacts'] and raw['source_hashes']==manifest['source_hashes'] and raw['analysis_inputs_sha256']==digest(RUN/f'{attempt}-inputs.json'),'frozen scope changed')
    for n,h in {**manifest['original_artifacts'],**raw['outputs']}.items():require(digest(ROOT/n)==h,'frozen artifact changed '+n)
    for n,h in manifest['source_hashes'].items():require(digest(ROOT/n)==digest(RUN/f'{attempt}-source'/Path(n).name)==h,'executed producer source changed')
    report_inputs=read(REPORT/'inputs.json');build=read(REPORT/'build.json');summary_audit=read(REPORT/'summary-audit.json');visual=read(REPORT/'visual-qa.json')
    require(report_inputs['analysis_inputs_sha256']==digest(RUN/f'{attempt}-inputs.json') and report_inputs['raw_manifest_sha256']==digest(out/'raw-manifest.json') and report_inputs['raw_audit_sha256']==digest(RUN/'raw-audit.json') and report_inputs['raw_outputs']==raw['outputs'],'report source binding changed')
    require(build['state']=='built_pending_independent_display_audit' and build['inputs_sha256']==summary_audit['inputs_sha256']==digest(REPORT/'inputs.json') and summary_audit['build_sha256']==digest(REPORT/'build.json') and summary_audit['state']=='passed','report audit unbound')
    require(summary_audit['independent_source_sha256']==digest(ROOT/'tools/audit_dependence_report.py'),'independent report-auditor source changed')
    require(summary_audit['checks']=={'method_table_fields':360,'pair_table_fields':420,'natural_table_fields':45,'correlation_plot_fields':2025,'ratio_plot_fields':105},'display/plot scope incomplete')
    for n,h in report_inputs['source_hashes'].items():require(digest(ROOT/n)==digest(REPORT/'executed-source'/Path(n).name)==h,'executed report source changed')
    for n,h in build['outputs'].items():require(digest(REPORT/n)==h,'report/figure changed '+n)
    require(visual['state']=='passed' and set(visual['inspected_pngs'])=={'natural-auroc-correlation.png','three-model-ratios.png','four-natural-ratios.png'},'figure visual QA missing')
    for n,h in visual['figure_hashes'].items():require(digest(REPORT/n)==h,'inspected figure changed')
    env=read(RUN/'test-environment.json');log=(RUN/'regression.log').read_text(encoding='utf-8')
    require(env['CUDA_VISIBLE_DEVICES']=='-1' and env['torch_cuda_is_available_before_suite'] is False and 'Ran 91 tests' in log and 'OK (skipped=2)' in log,'full CPU-masked regression incomplete')
    artifacts={}
    for p in [RUN/'predeclaration.json',RUN/f'{attempt}-inputs.json',out/'raw-manifest.json',RUN/'raw-audit.json',RUN/'regression.log',RUN/'test-environment.json',REPORT/'inputs.json',REPORT/'build.json',REPORT/'summary-audit.json',REPORT/'visual-qa.json',*[REPORT/n for n in build['outputs']]]:
        artifacts[str(p.relative_to(ROOT)).replace('\\','/')]=digest(p)
    result={'state':'passed','completed_at':now(),'inputs':786,'replicates':1000,'endpoint_covariance_matrices':66,'variables_per_endpoint':45,
            'method_or_pair_decompositions':7920,'natural_decompositions':360,'artifact_hashes':artifacts,'completion_source_sha256':digest(__file__),
            'boundary':'Descriptive covariance of fixed paired image resamples; zero-variance correlations undefined; no training-seed uncertainty, causal claim, selection or external publication'}
    write(REPORT/'completion-audit.json',result)
    status.update(state='complete',completed_at=now(),report='report/bootstrap-dependence/REPORT.md',completion_audit_sha256=digest(REPORT/'completion-audit.json'))
    write(RUN/'status.json',status);print({k:v for k,v in result.items() if k!='artifact_hashes'},flush=True)

if __name__=='__main__':main()
