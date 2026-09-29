"""Full fixed-scope conditional image-bootstrap summaries and scientific figures."""
from pathlib import Path
import shutil
import numpy as np
from geometry_io import ROOT, read, write, load, digest, require, now
from bootstrap_io import RUN, SOURCE, METHODS, DATASETS, TARGETS, COUNTS

REPORT = ROOT / 'report/paired-bootstrap'
FIELDS = ['auroc','aupr_out','aupr_in','fpr95_id','id_tpr_at_fpr95','benchmark_threshold']

def endpoints(arrays, originals):
    # Every value is [replicate, seed, method]; points are [seed, method].
    for di,d in enumerate(DATASETS):
        for fi,f in enumerate(FIELDS):
            point=np.array([[r['datasets'][d]['methods'][m]['primary_2500'][f] for m in METHODS] for r in originals])
            yield f'ranking/{d}/{f}', arrays['ranking'][:,:,di,:,fi], point
    for fi,f in enumerate(FIELDS[:4]):
        point=np.array([[np.mean([r['datasets'][d]['methods'][m]['primary_2500'][f] for d in DATASETS[:4]]) for m in METHODS] for r in originals])
        yield f'ranking/natural_macro/{f}', arrays['natural_ranking_macro'][:,:,:,fi], point
    maps=[{(x['method'],x['target']):x for x in r['operating_points']} for r in originals]
    for ti,t in enumerate(TARGETS):
        for key,field in [('id_coverage','coverage'),('retained_error_risk','retained_error_risk')]:
            point=np.array([[r[m,t]['id_test'][field] for m in METHODS] for r in maps],dtype=float)
            yield f'operating/{t:g}/{key}', arrays[key][:,:,:,ti], point
        for di,d in enumerate(DATASETS):
            point=np.array([[r[m,t]['ood'][d]['acceptance'] for m in METHODS] for r in maps])
            yield f'operating/{t:g}/{d}_acceptance', arrays['ood_acceptance'][:,:,:,ti,di], point
        point=np.array([[r[m,t]['equal_weight_natural_ood_acceptance'] for m in METHODS] for r in maps])
        yield f'operating/{t:g}/natural_ood_acceptance_macro', arrays['natural_ood_acceptance_macro'][:,:,:,ti], point

def statistic(values, points, fixed):
    require(values.shape==(1000,), 'wrong full-schedule statistic length')
    defined=np.isfinite(values); count=int(defined.sum())
    original=float(np.mean(points)) if fixed else float(points)
    item={'original_point':original if np.isfinite(original) else None,'defined_replicates':count,'replicates':1000,
          'bootstrap_mean':None,'bootstrap_sample_sd':None,'percentile_lower':None,'percentile_upper':None,
          'original_three_seed_sample_sd':None}
    if fixed and np.isfinite(points).all():item['original_three_seed_sample_sd']=float(np.std(points,ddof=1))
    if count==1000:
        item.update(bootstrap_mean=float(np.mean(values)),bootstrap_sample_sd=float(np.std(values,ddof=1)),
                    percentile_lower=float(np.quantile(values,.025,method='linear')),percentile_upper=float(np.quantile(values,.975,method='linear')))
    return item

def cell(item):
    if item['percentile_lower'] is None:return 'unavailable'
    return f"{100*item['original_point']:.4f} [{100*item['percentile_lower']:.4f}, {100*item['percentile_upper']:.4f}]"

def main():
    raw=read(RUN/'raw-audit.json');require(raw['state']=='passed' and raw['completed_replicates']==1000,'full independent raw audit incomplete')
    status=read(RUN/'status.json');require(status['state'] in ['raw_complete_pending_independent_audit','complete'],'controller not finished')
    manifest=read(RUN/'analysis-inputs.json');require(raw['inputs_sha256']==digest(RUN/'analysis-inputs.json'),'raw audit binding changed')
    for n,h in {**manifest['original_artifacts'],**raw['output_artifacts']}.items():require(digest(ROOT/n)==h,'frozen input/output changed '+n)
    require(not (REPORT/'summary.json').exists(),'report already exists; preserve failed attempt before retry')
    REPORT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'tools/bootstrap_io.py',ROOT/'tools/geometry_io.py']
    snap=REPORT/'executed-source';snap.mkdir(exist_ok=False)
    for p in files:shutil.copy2(p,snap/p.name)
    binding={'prepared_at':now(),'raw_audit_sha256':digest(RUN/'raw-audit.json'),'input_manifest_sha256':digest(RUN/'analysis-inputs.json'),
             'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files},'raw_artifacts':raw['output_artifacts']}
    write(REPORT/'inputs.json',binding)
    blocks=[load(RUN/'analysis'/f'block-{i:03d}-metrics.npz') for i in range(10)]
    arrays={k:np.concatenate([b[k] for b in blocks],axis=0) for k in blocks[0]}
    require(np.array_equal(arrays['replicates'],np.arange(1000)),'full replicate ordering changed')
    originals=[read(SOURCE/f'seed{s}/evaluation/results.json') for s in range(3)]
    pairs=[{'earlier':METHODS[i],'later':METHODS[j],'key':METHODS[j]+'__minus__'+METHODS[i]} for i in range(15) for j in range(i+1,15)]
    summary={'state':'summarized_pending_independent_audit','created_at':now(),'replicates':1000,'methods':METHODS,'pairs':pairs,'targets':TARGETS,
             'counts':COUNTS,'quantiles':[.025,.975],'quantile_method':'linear','sample_sd_ddof':1,'endpoints':{},
             'boundary':'Conditional paired image sampling at fixed classifiers/thresholds; three training seeds are not resampled. Original seed SD is separate. Descriptive intervals, no significance/winner/population coverage guarantee. Noise outside natural macro; reused validation and posthoc scope remain limitations.'}
    records=0
    for key,values,points in endpoints(arrays,originals):
        entry={'methods':{},'pairs':{}}
        for mi,m in enumerate(METHODS):
            entry['methods'][m]={f'seed{s}':statistic(values[:,s,mi],points[s,mi],False) for s in range(3)}
            entry['methods'][m]['fixed_three']=statistic(values[:,:,mi].mean(1),points[:,mi],True);records+=4
        for p in pairs:
            i,j=METHODS.index(p['earlier']),METHODS.index(p['later']);v=values[:,:,j]-values[:,:,i];pt=points[:,j]-points[:,i]
            entry['pairs'][p['key']]={f'seed{s}':statistic(v[:,s],pt[s],False) for s in range(3)}
            entry['pairs'][p['key']]['fixed_three']=statistic(v.mean(1),pt,True);records+=4
        summary['endpoints'][key]=entry
    summary['statistic_records']=records;require(records==31680 and len(summary['endpoints'])==66 and len(pairs)==105,'fixed summary scope incomplete')
    write(REPORT/'summary.json',summary)
    def table(key):
        rows=['| Method | Original point / conditional image interval (%) | Original 3-seed SD (pp) |','|---|---:|---:|']
        for m in METHODS:
            x=summary['endpoints'][key]['methods'][m]['fixed_three'];sd=x['original_three_seed_sample_sd']
            rows.append(f'| {m} | {cell(x)} | {100*sd:.4f} |')
        return '\n'.join(rows)
    lines=['# Conditional paired image-bootstrap report','',summary['boundary'],'',
           'All 1,000 draws use identical ID/OOD image counts across methods and the three fixed models. Percentile intervals: [2.5%,97.5%], linear interpolation. The mean of three model metrics is computed inside each paired draw. Intervals are on the fixed-three mean; the seed SD column describes the original three model points, not the image interval.',
           '', '## Natural four-population macro AUROC','',table('ranking/natural_macro/auroc')]
    for d in DATASETS:
        lines+=['',f'## {d}: AUROC (noise diagnostic only)' if d=='noise' else f'## {d}: AUROC','',f"ID n=10000, OOD n={COUNTS[d]}, OOD prevalence={COUNTS[d]/(10000+COUNTS[d]):.6f}. AP depends on these proportions.",'',table(f'ranking/{d}/auroc')]
    for t in TARGETS:
        for field in ['id_coverage','retained_error_risk','natural_ood_acceptance_macro']:
            lines+=['',f'## Frozen calibration target {t:g}: {field}','',table(f'operating/{t:g}/{field}')]
    lines+=['','## All 105 protocol pairs: later minus earlier','',
            'No subset selected by results. AUROC differences and fixed95 operating differences are percentage points. ID coverage and retained original-classifier risk changes accompany OOD acceptance differences. Full JSON retains every seed, all six ranking fields, all five datasets, all four targets and every pair.','',
            '| Pair (later minus earlier) | Natural AUROC delta (pp) | Fixed95 natural OOD acceptance delta (pp) | Fixed95 ID coverage delta (pp) | Fixed95 retained error delta (pp) |',
            '|---|---:|---:|---:|---:|']
    pairkeys=['ranking/natural_macro/auroc','operating/0.95/natural_ood_acceptance_macro','operating/0.95/id_coverage','operating/0.95/retained_error_risk']
    for p in pairs:lines.append('| '+p['later']+' minus '+p['earlier']+' | '+' | '.join(cell(summary['endpoints'][k]['pairs'][p['key']]['fixed_three']) for k in pairkeys)+' |')
    lines+=['','## Artifacts and interpretation','',
            'Raw draw counts and metric/count tensors are in runs/paired-bootstrap/analysis/block-000 through block-009. Preflight, row-alignment, raw-audit and regression evidence remain local. summary.json includes all 31,680 statistic records; None retains undefined intervals without dropping draws. No training, detector fitting, threshold recalibration or OOD-test tuning occurred.',
            '', '![All natural AUROC intervals](natural-auroc.png)','', '![Fixed95 original-classifier operation](fixed95-operation.png)','']
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=[m.replace('_energy',' E').replace('mahalanobis_cpu','CPU MD').replace('mahalanobis','MD').replace('affine_energy','affine E') for m in METHODS]
    plot_data={}
    def panel(ax,key,title):
        x=summary['endpoints'][key]['methods'];y=np.arange(15)
        lo=[100*x[m]['fixed_three']['percentile_lower'] for m in METHODS];hi=[100*x[m]['fixed_three']['percentile_upper'] for m in METHODS];pt=[100*x[m]['fixed_three']['original_point'] for m in METHODS]
        ax.hlines(y,lo,hi,color='#1565a3',linewidth=2);ax.plot(pt,y,'o',color='#1565a3',markersize=4,label='fixed-three point / conditional 95% image interval')
        seeds=[[100*x[m][f'seed{s}']['original_point'] for m in METHODS] for s in range(3)]
        for s in range(3):ax.plot(seeds[s],y+(s-1)*.12,'.',color=['#cf6b25','#6f5da8','#45854c'][s],markersize=4,label=f'original seed{s} point')
        ax.set_yticks(y,labels);ax.set_ylim(14.7,-.7);ax.set_title(title);ax.set_xlabel('Percent');ax.grid(axis='x',alpha=.2)
        plot_data[key]={'original_percent':pt,'lower_percent':lo,'upper_percent':hi,'original_seed_percent':seeds,'method_order':METHODS}
    fig,axes=plt.subplots(1,4,figsize=(18,7),sharey=True)
    for ax,d in zip(axes,DATASETS[:4]):panel(ax,f'ranking/{d}/auroc',d)
    fig.legend(*axes[0].get_legend_handles_labels(),fontsize=8,loc='lower center',ncol=4);fig.suptitle('All methods in protocol order: natural AUROC; image intervals conditional on fixed models',fontsize=12);fig.tight_layout(rect=[0,.07,1,.95])
    for ext in ['png','svg']:fig.savefig(REPORT/f'natural-auroc.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,7),sharey=True)
    for ax,field,title in zip(axes,['natural_ood_acceptance_macro','id_coverage','retained_error_risk'],['Natural OOD accepted (lower is better)','Actual ID-test coverage','Retained original-ID error']):panel(ax,f'operating/0.95/{field}',title)
    fig.legend(*axes[0].get_legend_handles_labels(),fontsize=8,loc='lower center',ncol=4);fig.suptitle('Frozen 95% ID-calibration target; actual coverage varies by method',fontsize=12);fig.tight_layout(rect=[0,.07,1,.95])
    for ext in ['png','svg']:fig.savefig(REPORT/f'fixed95-operation.{ext}',dpi=180)
    plt.close(fig);write(REPORT/'plot-data.json',plot_data)
    for n,h in binding['source_hashes'].items():require(digest(ROOT/n)==h,'report producer source changed')
    write(REPORT/'build.json',{'state':'built_pending_independent_audit','built_at':now(),'statistic_records':records,'inputs_sha256':digest(REPORT/'inputs.json'),'artifacts':{p.name:digest(p) for p in [REPORT/'summary.json',REPORT/'REPORT.md',REPORT/'plot-data.json',*sorted(REPORT.glob('*.png')),*sorted(REPORT.glob('*.svg'))]}})
    print({'state':'built_pending_independent_audit','records':records,'endpoints':66,'pairs':105},flush=True)

if __name__=='__main__':main()
