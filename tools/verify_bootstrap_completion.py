"""Readonly verification of completed/frozen bootstrap evidence; never rewrites audits."""
from pathlib import Path
from geometry_io import ROOT,read,digest,require

def main():
    run=ROOT/'runs/paired-bootstrap';report=ROOT/'report/paired-bootstrap'
    status=read(run/'status.json');complete=read(report/'completion-audit.json');raw=read(run/'raw-audit.json');stat=read(report/'summary-audit.json');summary=read(report/'summary.json')
    require(status['state']=='complete' and status['completed_replicates']==1000 and status['completed_blocks']==list(range(10)),'fixed bootstrap schedule incomplete')
    require(status['completion_audit_sha256']==digest(report/'completion-audit.json') and complete['state']=='passed','completion status binding invalid')
    require(complete['completion_source_sha256']==digest(ROOT/'tools/complete_bootstrap.py'),'executed completion source changed')
    for n,h in complete['artifacts'].items():require(digest(ROOT/n)==h,'completed evidence changed '+n)
    manifest=read(run/'analysis-inputs.json');binding=read(report/'inputs.json');build=read(report/'build.json')
    require(raw['state']=='passed' and raw['completed_replicates']==1000 and raw['inputs_sha256']==binding['input_manifest_sha256']==digest(run/'analysis-inputs.json'),'raw verification binding changed')
    require(binding['raw_audit_sha256']==digest(run/'raw-audit.json') and build['inputs_sha256']==digest(report/'inputs.json'),'report binding changed')
    for n,h in {**manifest['original_artifacts'],**raw['output_artifacts']}.items():require(digest(ROOT/n)==h,'frozen original/raw artifact changed '+n)
    for n,h in manifest['source_hashes'].items():require(digest(ROOT/n)==digest(run/'analysis-source'/Path(n).name)==h,'executed raw source changed')
    for n,h in binding['source_hashes'].items():require(digest(ROOT/n)==digest(report/'executed-source'/Path(n).name)==h,'executed report source changed')
    require(raw['independent_source_sha256']==digest(ROOT/'tools/audit_paired_bootstrap.py') and stat['independent_source_sha256']==digest(ROOT/'tools/audit_bootstrap_report.py'),'executed independent audit source changed')
    require(stat['state']=='passed' and stat['summary_sha256']==digest(report/'summary.json') and stat['build_sha256']==digest(report/'build.json'),'statistical audit binding changed')
    require(summary['statistic_records']==stat['statistic_records']==31680 and len(summary['endpoints'])==stat['endpoints']==66 and len(summary['methods'])==15 and len(summary['pairs'])==105,'fixed summary coverage changed')
    for e in summary['endpoints'].values():
        require(len(e['methods'])==15 and len(e['pairs'])==105,'method/pair omitted')
        for item in [*e['methods'].values(),*e['pairs'].values()]:require(set(item)=={'seed0','seed1','seed2','fixed_three'} and all(x['replicates']==1000 for x in item.values()),'seed/draw summary omitted')
    visual=read(report/'visual-qa.json');require(visual['state']=='passed','visual verification missing')
    for n,h in visual['figure_hashes'].items():require(digest(report/n)==h,'inspected figure changed')
    next_path=ROOT/'runs/bootstrap-dependence/predeclaration.json';next_count=0
    if next_path.exists():
        next_decl=read(next_path)
        for n,h in next_decl['original_artifacts'].items():require(digest(ROOT/n)==h,'next frozen scope input changed '+n)
        next_count=len(next_decl['original_artifacts'])
    print({'state':'verified_readonly','replicates':1000,'methods':15,'pairs':105,'endpoints':66,'statistics':31680,'original_inputs':len(manifest['original_artifacts']),'next_declared_inputs':next_count,'boundary':'Verified stored complete audit/source/raw/report/visual/test hashes and scope. Does not recompute statistics or modify frozen audit records.'},flush=True)

if __name__=='__main__':main()
