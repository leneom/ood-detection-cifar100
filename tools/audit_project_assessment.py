"""Independent local phase-map and displayed-result audit."""
from pathlib import Path
import math
from geometry_io import ROOT,read,write,digest,require,now

REPORT=ROOT/'report/project-assessment'
EXPECTED=['baseline-complete','calibration-phase2','selective-risk-phase2','geometry-phase3','geometry-diagnostics',
          'rmd-phase4','rmd-diagnostics','ash-phase5','ash-diagnostics','paired-bootstrap','bootstrap-dependence']

def agree(a,b,name):require(math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),name+' differs')

def main():
    source=ROOT/'tools/build_project_assessment.py';inp=read(REPORT/'inputs.json');build=read(REPORT/'build.json');index=read(REPORT/'index.json');body=(REPORT/'REPORT.md').read_text(encoding='utf-8')
    require(inp['state']=='bound_before_assessment_serialization' and inp['phase_count']==11 and build['state']=='built_pending_independent_audit' and index['state']=='report_pending_independent_audit','report lifecycle changed')
    require(build['inputs_sha256']==digest(REPORT/'inputs.json') and build['index_sha256']==digest(REPORT/'index.json') and build['report_sha256']==digest(REPORT/'REPORT.md'),'output binding changed')
    require(inp['source_sha256']==build['source_sha256']==digest(source)==digest(REPORT/'executed-source'/source.name),'executed source changed')
    expected_paths=set()
    checks={'phase_states':0,'phase_hashes':0,'report_links':0,'metric_values':0,'displayed_values':0,'input_hashes':0}
    require([r['phase'] for r in index['phases']]==EXPECTED and index['phase_count']==len(EXPECTED)==11,'phase order/count changed')
    for i,(phase,row) in enumerate(zip(EXPECTED,index['phases'])):
        report=Path('report')/phase/'REPORT.md';audit=Path('report')/phase/'completion-audit.json';status=Path('runs')/('matrix' if phase=='baseline-complete' else phase)/'status.json'
        paths={'report':report,'completion_audit':audit,'run_status':status}
        require(set(row['paths'])==set(paths) and set(row['sha256'])==set(paths),'phase evidence schema changed')
        for k,p in paths.items():
            name=p.as_posix();expected_paths.add(name)
            require(row['paths'][k]==name and inp['input_sha256'][name]==row['sha256'][k]==digest(ROOT/p),'phase artifact hash changed');checks['phase_hashes']+=1
        a=read(ROOT/audit);s=read(ROOT/status)
        require(row['audit_state']==a['state'] and row['run_state']==s['state']=='complete' and a['state'] in ['passed','complete','complete_and_verified'],'phase unfinished');checks['phase_states']+=2
        require(f'../{phase}/REPORT.md' in body and f'../{phase}/completion-audit.json' in body,'phase report/audit link missing');checks['report_links']+=2
    other=['docs/PROTOCOL.md','docs/PAIRED_BOOTSTRAP.md','docs/BOOTSTRAP_DEPENDENCE.md','report/baseline-complete/overview.json',
           'runs/matrix/summary/summary.json','report/paired-bootstrap/summary.json','report/bootstrap-dependence/plot-data.json']
    require(set(inp['input_sha256'])==expected_paths|set(other) and len(inp['input_sha256'])==40,'input coverage changed')
    for name,h in inp['input_sha256'].items():require(digest(ROOT/name)==h,'assessed input changed '+name);checks['input_hashes']+=1
    base=read(ROOT/'runs/matrix/summary/summary.json');overview=read(ROOT/'report/baseline-complete/overview.json');summary=read(ROOT/'report/paired-bootstrap/summary.json');plot=read(ROOT/'report/bootstrap-dependence/plot-data.json')
    require(base['seeds']==overview['seeds']==[0,1,2] and summary['replicates']==1000 and len(summary['methods'])==15 and len(summary['endpoints'])==66 and len(summary['pairs'])==105,'underlying benchmark scope changed')
    metrics=index['metrics'];accuracy=base['id_test_accuracy']
    for k,v in [('id_test_accuracy_mean',accuracy['mean']),('id_test_accuracy_seed_sd',accuracy['sample_sd'])]:agree(metrics[k],v,k);checks['metric_values']+=1
    require(metrics['id_test_accuracy_by_seed']==accuracy['per_seed'],'seed accuracy list changed');checks['metric_values']+=3
    results=summary['endpoints']['ranking/natural_macro/auroc']
    for method,key in [('energy','energy_macro_auroc'),('knn','knn_macro_auroc'),('rmd','rmd_macro_auroc'),('mahalanobis_cpu','matched_cpu_md_macro_auroc'),('ash_s_energy','ash_s_macro_auroc')]:
        require(metrics[key]==results['methods'][method]['fixed_three'],'method summary point/interval changed');checks['metric_values']+=7
    required=[('ash_s_minus_energy_macro_auroc','ranking/natural_macro/auroc'),('ash_s_minus_energy_cifar10_auroc','ranking/cifar10/auroc'),
              ('ash_s_minus_energy_natural_acceptance_95','operating/0.95/natural_ood_acceptance_macro'),('ash_s_minus_energy_id_coverage_95','operating/0.95/id_coverage')]
    for key,endpoint in required:require(metrics[key]==summary['endpoints'][endpoint]['pairs']['ash_s_energy__minus__energy']['fixed_three'],key+' pair summary changed');checks['metric_values']+=7
    fixed=plot['three_model_ratios']['ranking/natural_macro/auroc'];natural=plot['four_natural_ratios']
    require(set(fixed)==set(summary['methods']) and set(natural)==set(summary['methods']) and all(len(row)==3 for row in natural.values()),'dependence display scope changed')
    require(metrics['three_model_sd_ratio_range_natural_macro_auroc']==[min(fixed.values()),max(fixed.values())] and metrics['four_natural_sd_ratio_range_natural_macro_auroc']==[min(v for row in natural.values() for v in row),max(v for row in natural.values() for v in row)],'dependence range changed');checks['metric_values']+=4
    def p(v):return f'{100*v:.2f}%'
    def pp(v):return f'{100*v:+.2f} pp'
    def ci(x):return f"[{100*x['percentile_lower']:+.2f}, {100*x['percentile_upper']:+.2f}] pp"
    displays=[f"{p(metrics['id_test_accuracy_mean'])} ± {100*metrics['id_test_accuracy_seed_sd']:.2f} pp",
              ', '.join(p(v) for v in metrics['id_test_accuracy_by_seed']),
              *[p(metrics[k]['original_point']) for k in ['energy_macro_auroc','knn_macro_auroc','rmd_macro_auroc','matched_cpu_md_macro_auroc','ash_s_macro_auroc']],
              *[pp(metrics[k]['original_point']) for k,_ in required],
              ci(metrics['ash_s_minus_energy_macro_auroc']),ci(metrics['ash_s_minus_energy_cifar10_auroc']),
              f"{metrics['three_model_sd_ratio_range_natural_macro_auroc'][0]:.2f}–{metrics['three_model_sd_ratio_range_natural_macro_auroc'][1]:.2f}",
              f"{metrics['four_natural_sd_ratio_range_natural_macro_auroc'][0]:.2f}–{metrics['four_natural_sd_ratio_range_natural_macro_auroc'][1]:.2f}"]
    for item in displays:require(item in body,'displayed evidence differs '+item);checks['displayed_values']+=1
    for required_phrase in ['checkpoint、temperature-fit、threshold-calibration三份','属于探索性','不能直接当完整OpenOOD基准','不**估计重新训练模型的方差','没有推送、发布或对外发送数据']:
        require(required_phrase in body,'material evidence boundary omitted')
    result={'state':'passed','audited_at':now(),'inputs_sha256':digest(REPORT/'inputs.json'),'index_sha256':digest(REPORT/'index.json'),'report_sha256':digest(REPORT/'REPORT.md'),
            'independent_source_sha256':digest(__file__),'checks':checks,'boundary':'Independently checked all 11 phase statuses, completion/report hashes/links, derived values, displayed figures, input/source hashes and material limitations'}
    write(REPORT/'audit.json',result);print(result,flush=True)

if __name__=='__main__':main()
