"""Complete fixed-scope saved-bootstrap dependence report and scientific plots."""
import shutil
from pathlib import Path
import numpy as np
from geometry_io import ROOT,read,write,load,digest,require,now

RUN=ROOT/'runs/bootstrap-dependence';SOURCE=ROOT/'runs/paired-bootstrap';REPORT=ROOT/'report/bootstrap-dependence'
PRIMARY=['ranking/natural_macro/auroc','operating/0.95/natural_ood_acceptance_macro','operating/0.95/id_coverage','operating/0.95/retained_error_risk']

def fmt(value):return 'undefined' if value is None else f'{value:.4f}'

def main():
    status=read(RUN/'status.json');require(status['state']=='raw_complete_pending_independent_audit' and status['completed_endpoints']==66,'full dependence data incomplete')
    raw_audit=read(RUN/'raw-audit.json');require(raw_audit['state']=='passed','independent scalar audit incomplete')
    attempt=status['output_name'];source=RUN/attempt;raw_manifest=read(source/'raw-manifest.json');manifest=read(RUN/f'{attempt}-inputs.json')
    require(raw_audit['analysis_inputs_sha256']==raw_manifest['analysis_inputs_sha256']==digest(RUN/f'{attempt}-inputs.json') and raw_audit['raw_manifest_sha256']==digest(source/'raw-manifest.json'),'raw provenance changed')
    for name,h in {**manifest['original_artifacts'],**raw_manifest['outputs']}.items():require(digest(ROOT/name)==h,'frozen input/raw changed '+name)
    require(not REPORT.exists(),'report exists: retain it under an attempt name before any retry')
    REPORT.mkdir(parents=True);files=[Path(__file__),ROOT/'tools/geometry_io.py'];snap=REPORT/'executed-source';snap.mkdir(exist_ok=False)
    for p in files:shutil.copy2(p,snap/p.name)
    bind={'prepared_at':now(),'analysis_inputs_sha256':digest(RUN/f'{attempt}-inputs.json'),'raw_manifest_sha256':digest(source/'raw-manifest.json'),
          'raw_audit_sha256':digest(RUN/'raw-audit.json'),'raw_outputs':raw_manifest['outputs'],
          'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files}}
    write(REPORT/'inputs.json',bind)
    index=read(source/'endpoint-index.json');methods=index['methods'];pairs=index['pairs'];names=[item['endpoint'] for item in index['endpoints']]
    require(len(names)==66 and len(methods)==15 and len(pairs)==105 and all(k in names for k in PRIMARY),'fixed table scope changed')
    original=read(ROOT/'report/paired-bootstrap/summary.json')
    selected={key:read(source/f'endpoint-{names.index(key):03d}.json') for key in PRIMARY}
    natural=read(source/'natural-components.json');natural_records={x['key']:x for x in natural['records']}
    def entry(method,key):return selected[key]['methods'][method]
    def table_methods(key):
        rows=['| Method | Original point (%) | Conditional image 95% interval (%) | Diagonal contribution (variance) | Cross-model contribution (variance) | Observed/diagonal SD ratio |',
              '|---|---:|---:|---:|---:|---:|']
        for m in methods:
            x=entry(m,key);old=original['endpoints'][key]['methods'][m]['fixed_three']
            point=100*old['original_point'];lo=100*old['percentile_lower'];hi=100*old['percentile_upper']
            rows.append(f"| {m} | {point:.4f} | [{lo:.4f}, {hi:.4f}] | {fmt(x['diagonal_contribution'])} | {fmt(x['cross_covariance_contribution'])} | {fmt(x['sd_ratio_to_diagonal_only'])} |")
        return '\n'.join(rows)
    lines=['# Saved-bootstrap dependence diagnostics','',
           'All covariance and correlation entries use the **same 1,000 saved paired image draws**, with three frozen classifiers and fifteen fixed columns. Population/image sampling variability is conditional on those classifiers and thresholds; this is not training-seed variability or a new inferential test.',
           '', 'For a fixed method, the three-model mean variance equals (sum of three seed variances + two times the three cross-seed covariances)/9. The diagonal-only term is an explicitly labeled independence counterfactual, not an alternative confidence interval. Reported ratios compare observed SD with its diagonal-only algebraic counterpart; zero denominator is undefined.',
           '', 'The full 45×45 sample covariance, Pearson correlation and finite/undefined masks for all 66 endpoints are in immutable endpoint-000 through endpoint-065 artifacts. All 105 later-minus-earlier pairs have their own three-model decompositions. The four-natural covariance/decomposition for each seed/method and four ranking/four operating macros is in natural-components.*; noise is excluded. No setting was filtered by sign or strength.']
    titles=['Natural four-population AUROC','Fixed95 natural OOD acceptance','Fixed95 actual ID-test coverage','Fixed95 retained original-ID error risk']
    for key,title in zip(PRIMARY,titles):lines+=['',f'## {title}','',table_methods(key)]
    lines+=['','## All 105 later-minus-earlier pairs','',
            'Each cell is the observed/diagonal-only SD ratio of the paired method difference, using one common image draw per replicate. The signed pair definition matches the original bootstrap report. All component variances and covariance signs remain in raw JSON for every endpoint.',
            '', '| Later minus earlier | Natural macro AUROC | Fixed95 natural OOD acceptance | Fixed95 ID coverage | Fixed95 original retained error |','|---|---:|---:|---:|---:|']
    for p in pairs:lines.append('| '+p['later']+' minus '+p['earlier']+' | '+' | '.join(fmt(selected[k]['pairs'][p['key']]['sd_ratio_to_diagonal_only']) for k in PRIMARY)+' |')
    lines+=['','## Four-natural population macro dependence','',
            'The next table retains every method and three fixed classifiers for the AUROC macro. The other three ranking and four operating natural macros are in natural-components.json/npz. The four OOD populations have distinct OOD draws and share ID draws, so their macro variability includes cross-population terms.','',
            '| Method | Seed 0 observed/diagonal SD | Seed 1 | Seed 2 |','|---|---:|---:|---:|']
    for m in methods:lines.append('| '+m+' | '+' | '.join(fmt(natural_records[f'ranking/auroc/seed{s}/{m}']['sd_ratio_to_diagonal_only']) for s in range(3))+' |')
    lines+=['','## Interpretation and audit boundary','',
            'Pearson correlation is undefined when either full-draw marginal variance is zero; its mask and NaN in raw arrays/None in JSON are retained. Every finite count is saved; nonfinite draws invalidate the affected covariance instead of producing a pairwise complete-case matrix. Covariance is descriptive of this fixed empirical bootstrap and does not reveal causal failure mechanisms, guarantee population coverage, resolve validation reuse or justify method/threshold selection.',
            '', 'Raw endpoint data, independent scalar audit, source/input snapshots, regression and figure-coordinate audits are retained locally. The complete paired-bootstrap report supplies the original points and conditional intervals; this report adds no new score, training or threshold.',
            '', '![Full seed-method correlation matrix](natural-auroc-correlation.png)','',
            '![Cross-model SD ratios](three-model-ratios.png)','',
            '![Cross-population macro SD ratios](four-natural-ratios.png)','']
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plot={}
    key=PRIMARY[0];archive=load(source/f'endpoint-{names.index(key):03d}.npz');c=archive['correlation'];mask=archive['correlation_defined']
    labels=[f's{s}/{m}' for s in range(3) for m in methods]
    plot['natural_macro_auroc_correlation']={'endpoint':key,'variable_order':labels,'correlation':[[float(v) if mask[i,j] else None for j,v in enumerate(row)] for i,row in enumerate(c)],'defined_mask':mask.tolist()}
    fig,ax=plt.subplots(figsize=(15,13));cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#d0d0d0')
    im=ax.imshow(np.ma.masked_invalid(c),cmap=cmap,norm=Normalize(-1,1),interpolation='nearest')
    ax.set_xticks(range(45),labels,rotation=90,fontsize=5);ax.set_yticks(range(45),labels,fontsize=5)
    for bound in [14.5,29.5]:ax.axvline(bound,color='black',linewidth=.8);ax.axhline(bound,color='black',linewidth=.8)
    ax.set_title('Natural macro AUROC: 45 seed-method variables; same image draws\nGray = undefined correlation (zero variance or nonfinite); no cells dropped')
    fig.colorbar(im,ax=ax,label='Pearson correlation, fixed image draws',fraction=.045);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(REPORT/f'natural-auroc-correlation.{ext}',dpi=170)
    plt.close(fig)
    plot['three_model_ratios']={key:{m:entry(m,key)['sd_ratio_to_diagonal_only'] for m in methods} for key in PRIMARY}
    y=np.arange(15);fig,axes=plt.subplots(1,4,figsize=(19,8),sharey=True)
    for ax,key,title in zip(axes,PRIMARY,titles):
        x=[entry(m,key)['sd_ratio_to_diagonal_only'] for m in methods];valid=np.isfinite([v if v is not None else np.nan for v in x])
        ax.scatter(np.asarray(x,dtype=float)[valid],y[valid],s=26,color='#1965a0')
        ax.axvline(1,color='#b64b39',linewidth=1,linestyle='--');ax.set_yticks(y,methods);ax.set_ylim(14.7,-.7)
        ax.set_xlim(0,1.83);ax.set_title(title,fontsize=9);ax.set_xlabel('Observed / diagonal-only SD');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Three fixed models: paired-image covariance contribution; dashed = diagonal-only equality',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96])
    for ext in ['png','svg']:fig.savefig(REPORT/f'three-model-ratios.{ext}',dpi=170)
    plt.close(fig)
    plot['four_natural_ratios']={m:[natural_records[f'ranking/auroc/seed{s}/{m}']['sd_ratio_to_diagonal_only'] for s in range(3)] for m in methods}
    fig,ax=plt.subplots(figsize=(8,8))
    for s,color in enumerate(['#cf6b25','#6f5da8','#45854c']):
        x=[plot['four_natural_ratios'][m][s] for m in methods];valid=np.isfinite([v if v is not None else np.nan for v in x])
        ax.scatter(np.asarray(x,dtype=float)[valid],y[valid]+(s-1)*.16,color=color,s=28,label=f'fixed classifier seed{s}')
    ax.axvline(1,color='#b64b39',linewidth=1,linestyle='--');ax.set_yticks(y,methods);ax.set_ylim(14.7,-.7);ax.set_xlim(0,2.05)
    ax.set_xlabel('Observed / diagonal-only natural-macro AUROC SD');ax.set_title('Four natural OOD populations share the ID draw; noise excluded')
    ax.grid(axis='x',alpha=.2);ax.legend(loc='lower center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1])
    for ext in ['png','svg']:fig.savefig(REPORT/f'four-natural-ratios.{ext}',dpi=170)
    plt.close(fig);write(REPORT/'plot-data.json',plot)
    for n,h in bind['source_hashes'].items():require(digest(ROOT/n)==h,'report source changed during build')
    artifacts={p.name:digest(p) for p in [REPORT/'REPORT.md',REPORT/'plot-data.json',*sorted(REPORT.glob('*.png')),*sorted(REPORT.glob('*.svg'))]}
    write(REPORT/'build.json',{'state':'built_pending_independent_display_audit','built_at':now(),'inputs_sha256':digest(REPORT/'inputs.json'),'outputs':artifacts,
                               'displayed_primary_endpoints':PRIMARY,'displayed_methods':15,'displayed_pairs':105,'displayed_natural_seed_method_rows':45,
                               'boundary':'Figures/tables are a fixed subset for reading; all 66 endpoints, 45-variable matrices, 105 pairs and 360 macro decompositions remain in raw outputs'})
    print({'state':'built_pending_independent_display_audit','primary_endpoints':4,'method_rows':60,'pair_rows':105,'natural_rows':15},flush=True)

if __name__=='__main__':main()
