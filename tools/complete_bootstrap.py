"""Completion requires full raw/statistical audit, inspected figures and CPU regression."""
from pathlib import Path
from geometry_io import ROOT, read, write, digest, require, now

RUN=ROOT/'runs/paired-bootstrap';REPORT=ROOT/'report/paired-bootstrap'

def main():
    status=read(RUN/'status.json');raw=read(RUN/'raw-audit.json');summary=read(REPORT/'summary-audit.json');build=read(REPORT/'build.json');visual=read(REPORT/'visual-qa.json');manifest=read(RUN/'analysis-inputs.json');binding=read(REPORT/'inputs.json')
    require(status['state']=='raw_complete_pending_independent_audit' and status['completed_replicates']==1000 and status['completed_blocks']==list(range(10)),'full fixed raw schedule incomplete')
    require(raw['state']=='passed' and raw['completed_replicates']==1000 and raw['checks']['regenerated_weight_vectors']==6000 and raw['checks']['independent_expanded_ranking_fields']==13500,'independent raw audit incomplete')
    require(summary['state']=='passed' and summary['statistic_records']==31680 and summary['endpoints']==66 and summary['method_pairs']==105,'full summary audit incomplete')
    require(summary['summary_sha256']==digest(REPORT/'summary.json') and summary['build_sha256']==digest(REPORT/'build.json'),'summary audit unbound')
    require(raw['independent_source_sha256']==digest(ROOT/'tools/audit_paired_bootstrap.py') and summary['independent_source_sha256']==digest(ROOT/'tools/audit_bootstrap_report.py'),'independent executed audit source changed')
    require(raw['inputs_sha256']==binding['input_manifest_sha256']==digest(RUN/'analysis-inputs.json') and binding['raw_audit_sha256']==digest(RUN/'raw-audit.json') and build['inputs_sha256']==digest(REPORT/'inputs.json'),'input/audit binding changed')
    for n,h in {**manifest['original_artifacts'],**raw['output_artifacts']}.items():require(digest(ROOT/n)==h,'immutable input/raw output changed '+n)
    for n,h in manifest['source_hashes'].items():require(digest(ROOT/n)==digest(RUN/'analysis-source'/Path(n).name)==h,'raw producer source changed')
    for n,h in binding['source_hashes'].items():require(digest(ROOT/n)==digest(REPORT/'executed-source'/Path(n).name)==h,'summary producer source changed')
    for name,h in build['artifacts'].items():require(digest(REPORT/name)==h,'summary/figure artifact changed')
    require(visual['state']=='passed' and set(visual['inspected_pngs'])=={'natural-auroc.png','fixed95-operation.png'},'scientific figures not inspected')
    for n,h in visual['figure_hashes'].items():require(digest(REPORT/n)==h,'inspected figure changed')
    env=read(RUN/'test-environment.json');log=(RUN/'regression.log').read_text(encoding='utf-8')
    require(env['CUDA_VISIBLE_DEVICES']=='-1' and env['torch_cuda_is_available_before_suite'] is False and 'Ran 85 tests' in log and 'OK (skipped=2)' in log,'full masked CPU regression incomplete')
    artifacts={}
    for p in [RUN/'analysis-inputs.json',RUN/'raw-audit.json',RUN/'preflight.json',RUN/'preflight-inputs.json',RUN/'row-alignment.json',RUN/'regression.log',RUN/'test-environment.json',REPORT/'build.json',REPORT/'inputs.json',REPORT/'summary-audit.json',REPORT/'visual-qa.json',*[REPORT/n for n in build['artifacts']]]:artifacts[str(p.relative_to(ROOT)).replace('\\','/')]=digest(p)
    audit={'state':'passed','completed_at':now(),'replicates':1000,'seeds':[0,1,2],'methods':15,'pairs':105,'endpoints':66,'statistic_records':31680,'original_input_hash_checks':len(manifest['original_artifacts']),'artifacts':artifacts,'completion_source_sha256':digest(__file__),'boundary':'Conditional paired image percentile intervals only; seed SD remains separate, noise excluded from natural macro, no tuning/significance/coverage guarantee or external publication'}
    write(REPORT/'completion-audit.json',audit)
    status.update(state='complete',completed_at=now(),report='report/paired-bootstrap/REPORT.md',completion_audit_sha256=digest(REPORT/'completion-audit.json'))
    write(RUN/'status.json',status);print({k:v for k,v in audit.items() if k!='artifacts'},flush=True)

if __name__=='__main__':main()
