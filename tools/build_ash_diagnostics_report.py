"""Complete three-seed diagnostic summaries and fixed illustrative plots."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_io import ROOT, digest, read, write, load, require, now

RUN = ROOT / 'runs/ash-diagnostics'
OUT = ROOT / 'report/ash-diagnostics'
POPULATIONS = ['validation_id','test_id','cifar10','mnist','svhn','textures','noise']
NEW = ['ash_p_energy','ash_b_energy','ash_s_energy']
REFERENCES = ['msp','energy','maxlogit','entropy','knn','mahalanobis','vim','residual','react_energy','affine_energy','rmd','mahalanobis_cpu']
TARGETS = [.8,.9,.95,.99]
NATURAL = ['cifar10','mnist','svhn','textures']
STATES = ['both_accept','ash_only_accept','reference_only_accept','neither_accept']
LABELS = dict(ash_p_energy='ASH-P',ash_b_energy='ASH-B',ash_s_energy='ASH-S',rmd='RMD',msp='MSP',energy='Energy',maxlogit='MaxLogit',entropy='Entropy',knn='KNN',mahalanobis='Mahalanobis',vim='ViM',residual='Residual',react_energy='ReAct',affine_energy='Affine Energy*',mahalanobis_cpu='CPU MD*')


def combine(nodes):
    if isinstance(nodes[0],dict):
        require(all(set(n)==set(nodes[0]) for n in nodes),'inconsistent seed keys')
        return {k:combine([n[k] for n in nodes]) for k in nodes[0]}
    require(len(nodes)==3 and all(x is None or np.isfinite(x) for x in nodes),'three finite/undefined seeds needed')
    complete = all(x is not None for x in nodes)
    return {'per_seed':nodes,'mean':float(np.mean(nodes)) if complete else None,
            'sample_sd':float(np.std(nodes,ddof=1)) if complete else None,
            'defined_seed_count':sum(x is not None for x in nodes)}


def main():
    audit=read(OUT/'audit.json')
    require(audit['state']=='passed','raw diagnostics not audited')
    for name,h in audit['output_artifacts'].items():
        require(digest(ROOT/name)==h,'audited diagnostics changed')
    attempt=audit['output_name']
    results=[read(RUN/f'seed{s}/{attempt}/results.json') for s in [0,1,2]]
    populations=combine([r['population_statistics'] for r in results])
    differences=combine([r['mean_component_differences'] for r in results])
    classifications=combine([r['classification_statistics'] for r in results])
    records=[{(x['addition'],x['reference'],x['target']):x for x in r['transitions']} for r in results]
    transitions={a:{ref:{str(t):combine([r[(a,ref,t)]['populations'] for r in records]) for t in TARGETS} for ref in REFERENCES} for a in NEW}
    natural_macro={a:{ref:{str(t):{k:combine([float(np.mean([r[(a,ref,t)]['populations'][d][k] for d in NATURAL])) for r in records]) for k in ['addition_acceptance','reference_acceptance','acceptance_change']} for t in TARGETS} for ref in REFERENCES} for a in NEW}
    summary={'state':'computed_pending_independent_summary_audit','created_at':now(),'seeds':[0,1,2],
             'population_order':POPULATIONS,'new_methods':NEW,'references':REFERENCES,'targets':TARGETS,
             'population_statistics':populations,'mean_component_differences':differences,'transitions':transitions,'classification_statistics':classifications,'natural_macro_transitions':natural_macro,
             'raw_audit_sha256':digest(OUT/'audit.json'),'result_sha256':[digest(RUN/f'seed{s}/{attempt}/results.json') for s in [0,1,2]],
             'source_sha256':digest(__file__), 'boundary':'posthoc descriptions; within-seed signed differences; all populations/targets; no tuning or causal claim'}
    write(OUT/'summary.json',summary)
    def value(stat,scale=1):
        return 'undefined' if stat['mean'] is None else f"{scale*stat['mean']:.3f} ± {scale*stat['sample_sd']:.3f}"
    lines=['# Frozen ASH support, scale, classification and decision diagnostics','',
           'Post-result descriptive analysis of every fixed p=65 ASH variant, three frozen seeds and seven separate scoring populations. All twelve old reference columns and four existing targets remain intact. No feature extraction, detector fit, percentile/temperature change or threshold calculation occurred.','',
           'Values are mean ± sample SD across seeds 0/1/2. Population SD uses ddof=0; all quantiles/covariances and full rows remain in raw/summary archives. Seed SD is descriptive, not CI. Means/correlations and acceptance transitions cannot establish a causal mechanism for AUROC.','',
           '## Activation support and scale','',
           'q=S2/S1 is the retained fraction of activation mass, distinct from 179/512 retained coordinates. r=S1/S2 and c=exp(r) are fixed sample definitions, not fitted parameters. Every row satisfies q*r=1. B fills retained coordinates with S1/179; S uses the actual exponential.','',
           '| Population | n per seed | Mean retained fraction q | Mean ratio r | Mean exp(r) | Mean binary fill |',
           '|---|---:|---:|---:|---:|---:|']
    for pop in POPULATIONS:
        p=populations[pop]
        lines.append(f"| {pop} | {int(p['ratio']['n']['mean']):,} | "+' | '.join(value(p[k]['mean']) for k in ['retained_fraction','ratio','exp_ratio','binary_fill'])+' |')
    lines+=['','Validation/test ID are separate. Noise remains synthetic and is excluded from every natural macro.','',
            '## Signed Energy changes and correlation','',
            'All Energy vectors are copied from frozen phase 5. Each delta is variant minus the original affine Energy numeric control, keeping both signs. Pearson is computed between retained fraction q and that signed delta; a zero-variance population is explicitly undefined. These scales differ by variant, and negative delta is not a ranking or acceptance guarantee.','',
            '| Population | Variant | Mean delta Energy | Pearson(q,delta Energy) |',
            '|---|---|---:|---:|']
    for pop in POPULATIONS:
        for m in NEW:
            lines.append('| '+pop+' | '+LABELS[m]+' | '+value(populations[pop][m+'_minus_affine']['mean'])+' | '+value(populations[pop]['joint'][m]['pearson_fraction_delta'])+' |')
    lines+=['','All saved logits satisfy z_S=c*z_P+(1-c)*b with the original bias. Thus positive scaling of the pruned feature does not guarantee unchanged P/S argmax in this biased head. The independent auditor also reconstructs every P/B/S head and audits its frozen Energy without replacing any score.','',
            '## Full population mean differences','',
            'Each mean difference is formed within the same seed. All eighteen component differences are in summary.json; the table keeps retained fraction and all three signed Energy-delta changes. No population or variant is selected for a favorable result.','',
            '| Population minus test ID | Delta retained fraction | Delta P Energy change | Delta B Energy change | Delta S Energy change |',
            '|---|---:|---:|---:|---:|']
    for comparison,vals in differences.items():
        lines.append('| '+comparison+' | '+' | '.join(value(vals[k]) for k in ['retained_fraction']+[m+'_minus_affine' for m in NEW])+' |')
    lines+=['','## Separate intermediate classification transitions','',
            'These compare each shaped prediction with the original classifier. Gains/losses refer to correctness, with each population denominator. Argmax changes additionally include changed predictions that remain incorrect. OOD has no correctness label. Validation is reused validation. Every rejection-risk endpoint still uses the original classifier correctness.','',
            '| Population | Variant | Original accuracy % | Shaped accuracy % | Shaped-only correct % | Original-only correct % | Accuracy change (pp) | Argmax changes per seed |',
            '|---|---|---:|---:|---:|---:|---:|---|']
    for pop in ['test_id','validation_id']:
        for m in NEW:
            p=classifications[pop][m]
            lines.append('| '+pop+' | '+LABELS[m]+' | '+' | '.join([value(p['original_accuracy'],100),value(p['shaped_accuracy'],100),value(p['state_fractions']['shaped_only_correct'],100),value(p['state_fractions']['original_only_correct'],100),value(p['accuracy_change'],100)])+' | '+str(p['argmax_changes']['per_seed'])+' |')
    lines+=['','P/S disagreements per seed, test ID / validation ID: '+str(classifications['test_id']['P_S_argmax']['disagreement_count']['per_seed'])+' / '+str(classifications['validation_id']['P_S_argmax']['disagreement_count']['per_seed'])+'.','']
    lines+=['','## Exhaustive fixed-threshold decision transitions','',
            'The primary 95% ID-calibration target is shown below because it was already the fixed primary operating point. Every 80/90/95/99% target is saved in summary.json. All 432 seed/comparison/target settings and per-class ID counts remain available as raw transition codes.','',
            'Both methods use their own frozen threshold selected from the same 2,500 reused ID-validation calibration rows. Thus these transitions include threshold differences; they are not a pure ranking effect or a comparison at exactly equal achieved ID-test coverage. Correct/incorrect ID use original classifier predictions, with separate subgroup denominators. OOD populations have no classifier-correctness flag.','',
            '| ASH minus reference | Population | Both accept % | ASH only % | Reference only % | Neither % | Acceptance change (pp) |',
            '|---|---|---:|---:|---:|---:|---:|']
    for addition in NEW:
        for reference in REFERENCES:
            pop=transitions[addition][reference]['0.95']
            for name in ['test_id','correct ID','incorrect ID']+NATURAL+['noise']:
                p=pop['test_id']['correct'] if name=='correct ID' else pop['test_id']['incorrect'] if name=='incorrect ID' else pop[name]
                # A subgroup's signed acceptance change is reconstructed from its exhaustive states.
                if name in ['correct ID','incorrect ID']:
                    raw=[]
                    group='correct' if name=='correct ID' else 'incorrect'
                    for r in records:
                        q=r[(addition,reference,.95)]['populations']['test_id'][group]['state_fractions']
                        raw.append(q['ash_only_accept']-q['reference_only_accept'])
                    change=combine(raw)
                else:
                    change=p['acceptance_change']
                lines.append('| '+LABELS[addition]+' − '+LABELS[reference]+' | '+name+' | '+
                             ' | '.join(value(p['state_fractions'][k],100) for k in STATES)+' | '+value(change,100)+' |')
    lines+=['','## Paired own-threshold operating endpoints','',
            'Every natural macro is the equal-weight mean over four natural populations within seed. These are separate from achieved ID coverage and original-classifier retained error. All 80/90/95/99% settings remain in summary.json; the existing primary 95% point is displayed.','',
            '| ASH minus reference | Natural OOD acceptance change (pp) | ID coverage change (pp) | Retained ID error change (pp) |',
            '|---|---:|---:|---:|']
    for a in NEW:
        for ref in REFERENCES:
            p=transitions[a][ref]['0.95']['test_id']
            lines.append('| '+LABELS[a]+' − '+LABELS[ref]+' | '+value(natural_macro[a][ref]['0.95']['acceptance_change'],100)+' | '+value(p['acceptance_change'],100)+' | '+value(p['retained_error_risk_change'],100)+' |')
    lines+=['','## Scientific figures','',
            'Scatter: fixed sorted random sample up to 1,000 rows per population, shared by the three variants. Full-population minima/maxima determine bounds independently of the sampled points. Statistics use all rows.','',
            '![All populations, variants and seeds](support-scale.png)','',
            '![All natural populations and twelve references](natural-transitions.png)','',
            '## Reproduction and evidence','',
            '```powershell','& C:/Users/zxzx/anaconda3/python.exe tools/run_ash_diagnostics.py','& C:/Users/zxzx/anaconda3/python.exe tools/audit_ash_diagnostics.py','& C:/Users/zxzx/anaconda3/python.exe tools/build_ash_diagnostics_report.py','& C:/Users/zxzx/anaconda3/python.exe tools/audit_ash_diagnostics_report.py','```','',
            'Raw: runs/ash-diagnostics/seedN/analysis/*_components.npz, *_classification.npz, transitions.npz and results.json. Full classification and acceptance state codes, per-class cells and explicit original-correct/incorrect denominators are saved. Immutable score/mask/head inputs come from completed phase 5. Audits bind source/input/raw/summary hashes; completion is separate after visual QA and regression.','',
            'No causal, significance, deployment or fresh-validation coverage claim follows from this posthoc descriptive phase.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    palette=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(3,3,figsize=(20,15),constrained_layout=True)
    plot_manifest={'support_scatter':{},'natural_transitions':{'target':.95,'population_order':NATURAL,'references':REFERENCES,'additions':NEW}}
    for seed in [0,1,2]:
        data={pop:load(RUN/f'seed{seed}/{attempt}/{pop}_components.npz') for pop in POPULATIONS}
        plot_manifest['support_scatter'][str(seed)]={}
        for mi,m in enumerate(NEW):
            ax=axes[mi,seed]
            xx=np.concatenate([a['retained_fraction'] for a in data.values()]);yy=np.concatenate([a[m+'_minus_affine'] for a in data.values()])
            xp=max(float(xx.max()-xx.min())*.03,1e-9);yp=max(float(yy.max()-yy.min())*.03,1e-9)
            xlim=[float(xx.min()-xp),float(xx.max()+xp)];ylim=[float(yy.min()-yp),float(yy.max()+yp)]
            for i,pop in enumerate(POPULATIONS):
                a=data[pop];idx=a['sample_rows'];ax.scatter(a['retained_fraction'][idx],a[m+'_minus_affine'][idx],s=3,alpha=.2,color=palette[i],label=pop,rasterized=True)
            ax.set(xlim=xlim,ylim=ylim,title=f'{LABELS[m]}, seed {seed}',xlabel='Retained activation mass S2/S1',ylabel='Variant minus affine Energy');ax.grid(alpha=.15)
            plot_manifest['support_scatter'][str(seed)][m]={'xlim':xlim,'ylim':ylim,'sample_n_per_population':1000,'bounds_from_full_populations':True}
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncols=4,markerscale=4,fontsize=11)
    fig.suptitle('Frozen support and Energy changes: all 3 variants, seeds and 7 populations\nSubsampled scatter; full-population bounds/statistics; correlation is descriptive',fontsize=15)
    for ext in ['png','svg']:fig.savefig(OUT/f'support-scale.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(22,17),constrained_layout=True)
    x=np.arange(len(REFERENCES));colors=['#006d77','#e29526','#9b3a62']
    for di,dataset in enumerate(NATURAL):
        for ai,a in enumerate(NEW):
            ax=axes[di,ai];points=[transitions[a][ref]['0.95'][dataset]['acceptance_change'] for ref in REFERENCES]
            ax.errorbar(x,[p['mean']*100 for p in points],yerr=[p['sample_sd']*100 for p in points],fmt='none',ecolor='#555555',capsize=3,label='mean ± sample SD')
            for seed,color in enumerate(colors):ax.scatter(x+(seed-1)*.12,[p['per_seed'][seed]*100 for p in points],s=20,color=color,label=f'seed {seed}',zorder=3)
            ax.axhline(0,ls=':',color='gray');ax.set(title=f'{dataset}: {LABELS[a]}',ylabel='Variant minus reference OOD acceptance (pp)')
            ax.set_xticks(x,[LABELS[r] for r in REFERENCES],rotation=48,ha='right',fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncols=4,fontsize=11)
    fig.suptitle('All fixed comparisons at the existing 95% ID-calibration target\nNegative = fewer OOD accepted; own frozen thresholds; * numerical controls',fontsize=16)
    for ext in ['png','svg']:fig.savefig(OUT/f'natural-transitions.{ext}',dpi=160)
    plt.close(fig)
    write(OUT/'plot-manifest.json',plot_manifest)
    print('Complete ASH diagnostic summary/figures saved, pending independent summary audit.',flush=True)

if __name__=='__main__':main()
