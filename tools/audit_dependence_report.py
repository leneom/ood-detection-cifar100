"""Independent table, figure-coordinate, source and bound-input audit."""
import math
from pathlib import Path
import numpy as np
from geometry_io import ROOT,read,write,load,digest,require,now

RUN=ROOT/'runs/bootstrap-dependence';REPORT=ROOT/'report/bootstrap-dependence';SOURCE=ROOT/'runs/paired-bootstrap'
PRIMARY=['ranking/natural_macro/auroc','operating/0.95/natural_ood_acceptance_macro','operating/0.95/id_coverage','operating/0.95/retained_error_risk']

def format_value(v):return 'undefined' if v is None else format(v,'.4f')
def format_variance(v):return 'undefined' if v is None else format(v,'.3e')

def main():
    status=read(RUN/'status.json');require(status['state']=='raw_complete_pending_independent_audit','raw dependence incomplete')
    attempt=status['output_name'];out=RUN/attempt;manifest=read(RUN/f'{attempt}-inputs.json');raw=read(out/'raw-manifest.json');audit=read(RUN/'raw-audit.json')
    binding=read(REPORT/'inputs.json');build=read(REPORT/'build.json');visual_inputs={**manifest['original_artifacts'],**raw['outputs']}
    require(audit['state']=='passed' and audit['analysis_inputs_sha256']==binding['analysis_inputs_sha256']==digest(RUN/f'{attempt}-inputs.json') and audit['raw_manifest_sha256']==binding['raw_manifest_sha256']==digest(out/'raw-manifest.json'),'raw audit/report binding changed')
    require(binding['raw_audit_sha256']==digest(RUN/'raw-audit.json') and binding['raw_outputs']==raw['outputs'] and build['inputs_sha256']==digest(REPORT/'inputs.json'),'display source binding changed')
    for n,h in visual_inputs.items():require(digest(ROOT/n)==h,'display input/raw output changed '+n)
    for n,h in binding['source_hashes'].items():require(digest(ROOT/n)==digest(REPORT/'executed-source'/Path(n).name)==h,'executed report source changed')
    for n,h in build['outputs'].items():require(digest(REPORT/n)==h,'display artifact changed')
    original=read(ROOT/'report/paired-bootstrap/summary.json');index=read(out/'endpoint-index.json');methods=index['methods'];pairs=index['pairs'];names=[e['endpoint'] for e in index['endpoints']]
    require(build['displayed_primary_endpoints']==PRIMARY and build['displayed_methods']==15 and build['displayed_pairs']==105 and build['displayed_natural_seed_method_rows']==45,'display scope changed')
    require(methods==original['methods'] and pairs==original['pairs'] and len(names)==66 and all(k in names for k in PRIMARY),'protocol order/scope changed')
    records={k:read(out/f'endpoint-{names.index(k):03d}.json') for k in PRIMARY};natural={r['key']:r for r in read(out/'natural-components.json')['records']}
    plot=read(REPORT/'plot-data.json');text=(REPORT/'REPORT.md').read_text(encoding='utf-8')
    check={'method_table_fields':0,'pair_table_fields':0,'natural_table_fields':0,'correlation_plot_fields':0,'ratio_plot_fields':0}
    for k in PRIMARY:
        for m in methods:
            r=records[k]['methods'][m];o=original['endpoints'][k]['methods'][m]['fixed_three']
            line=f"| {m} | {100*o['original_point']:.4f} | [{100*o['percentile_lower']:.4f}, {100*o['percentile_upper']:.4f}] | {format_variance(r['diagonal_contribution'])} | {format_variance(r['cross_covariance_contribution'])} | {format_value(r['sd_ratio_to_diagonal_only'])} |"
            require(line in text,'displayed method row changed '+k+'/'+m);check['method_table_fields']+=6
            v=plot['three_model_ratios'][k][m]
            require(v==r['sd_ratio_to_diagonal_only'],'three-model plot coordinate changed');check['ratio_plot_fields']+=1
    require(set(plot['three_model_ratios'])==set(PRIMARY) and all(set(v)==set(methods) for v in plot['three_model_ratios'].values()),'method figure coordinate scope changed')
    for p in pairs:
        line='| '+p['later']+' minus '+p['earlier']+' | '+' | '.join(format_value(records[k]['pairs'][p['key']]['sd_ratio_to_diagonal_only']) for k in PRIMARY)+' |'
        require(line in text,'displayed signed pair row changed');check['pair_table_fields']+=4
    for m in methods:
        expected=[natural[f'ranking/auroc/seed{s}/{m}']['sd_ratio_to_diagonal_only'] for s in range(3)]
        line='| '+m+' | '+' | '.join(format_value(v) for v in expected)+' |'
        require(line in text,'displayed four-natural row changed');check['natural_table_fields']+=3
        require(plot['four_natural_ratios'][m]==expected,'four-natural figure coordinates changed');check['ratio_plot_fields']+=3
    require(set(plot['four_natural_ratios'])==set(methods),'natural figure coordinate scope changed')
    key=PRIMARY[0];source=load(out/f'endpoint-{names.index(key):03d}.npz');p=plot['natural_macro_auroc_correlation'];labels=[f's{s}/{m}' for s in range(3) for m in methods]
    require(p['endpoint']==key and p['variable_order']==labels and p['defined_mask']==source['correlation_defined'].tolist(),'correlation figure order/undefined mask changed')
    require(len(p['correlation'])==45 and all(len(row)==45 for row in p['correlation']),'correlation figure shape changed')
    for i in range(45):
        for j in range(45):
            expected=float(source['correlation'][i,j]) if source['correlation_defined'][i,j] else None
            got=p['correlation'][i][j]
            require(got==expected,'correlation heatmap coordinate changed');check['correlation_plot_fields']+=1
    for n,h in manifest['original_artifacts'].items():require(digest(ROOT/n)==h,'original input changed during display audit')
    result={'state':'passed','audited_at':now(),'inputs_sha256':digest(REPORT/'inputs.json'),'build_sha256':digest(REPORT/'build.json'),
            'independent_source_sha256':digest(__file__),'checks':check,'boundary':'Every displayed method/pair/natural row and all plotted coordinates rechecked against frozen full raw endpoints and original bootstrap points; visual inspection separate'}
    write(REPORT/'summary-audit.json',result);print(result,flush=True)

if __name__=='__main__':main()
