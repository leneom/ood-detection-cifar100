"""Bind the independently audited local research-readiness map."""
from pathlib import Path
from geometry_io import ROOT,read,write,digest,require,now

REPORT=ROOT/'report/project-assessment';RUN=ROOT/'runs/project-assessment'

def main():
    inputs=read(REPORT/'inputs.json');build=read(REPORT/'build.json');audit=read(REPORT/'audit.json');index=read(REPORT/'index.json')
    require(build['state']=='built_pending_independent_audit' and audit['state']=='passed' and index['phase_count']==11 and len(index['phases'])==11,'phase map/audit incomplete')
    require(build['inputs_sha256']==audit['inputs_sha256']==digest(REPORT/'inputs.json') and build['index_sha256']==audit['index_sha256']==digest(REPORT/'index.json') and build['report_sha256']==audit['report_sha256']==digest(REPORT/'REPORT.md'),'assessment audit binding changed')
    require(inputs['source_sha256']==build['source_sha256']==digest(ROOT/'tools/build_project_assessment.py')==digest(REPORT/'executed-source/build_project_assessment.py'),'assessment source changed')
    require(audit['independent_source_sha256']==digest(ROOT/'tools/audit_project_assessment.py'),'independent audit source changed')
    require(audit['checks']=={'phase_states':22,'phase_hashes':33,'report_links':22,'metric_values':72,'displayed_values':15,'input_hashes':40},'full phase/value/display audit incomplete')
    for n,h in inputs['input_sha256'].items():require(digest(ROOT/n)==h,'completed source changed '+n)
    artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in [REPORT/'inputs.json',REPORT/'index.json',REPORT/'REPORT.md',REPORT/'build.json',REPORT/'audit.json',ROOT/'docs/PROTOCOL.md']}
    result={'state':'passed','completed_at':now(),'phase_count':11,'bound_phase_inputs':40,'artifact_hashes':artifacts,'completion_source_sha256':digest(__file__),
            'boundary':'Research status/evidence map of existing local work; no new OOD test, classifier/detector selection, novelty claim or publication'}
    write(REPORT/'completion-audit.json',result)
    write(RUN/'status.json',{'state':'complete','completed_at':now(),'report':'report/project-assessment/REPORT.md','completion_audit_sha256':digest(REPORT/'completion-audit.json')})
    print({k:v for k,v in result.items() if k!='artifact_hashes'},flush=True)

if __name__=='__main__':main()
