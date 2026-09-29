"""Complete three-seed diagnostic summaries and fixed illustrative plots."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_io import ROOT, digest, read, write, load, require, now

RUN = ROOT / 'runs/geometry-diagnostics'
OUT = ROOT / 'report/geometry-diagnostics'
POPULATIONS = ['training_id','validation_id','test_id','cifar10','mnist','svhn','textures','noise']
NEW = ['vim','residual','react_energy']
REFERENCES = ['affine_energy','energy','msp']
TARGETS = [.8,.9,.95,.99]
NATURAL = ['cifar10','mnist','svhn','textures']
STATES = ['both_accepted','addition_only_accepted','reference_only_accepted','neither_accepted']
LABELS = dict(vim='ViM',residual='Residual',react_energy='ReAct',affine_energy='Affine Energy',energy='Energy',msp='MSP')


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
    records=[{(x['addition'],x['reference'],x['target']):x for x in r['transitions']} for r in results]
    transitions={a:{ref:{str(t):combine([r[(a,ref,t)]['populations'] for r in records]) for t in TARGETS} for ref in REFERENCES} for a in NEW}
    summary={'state':'computed_pending_independent_summary_audit','created_at':now(),'seeds':[0,1,2],
             'population_order':POPULATIONS,'new_methods':NEW,'references':REFERENCES,'targets':TARGETS,
             'population_statistics':populations,'mean_component_differences':differences,'transitions':transitions,
             'raw_audit_sha256':digest(OUT/'audit.json'),'result_sha256':[digest(RUN/f'seed{s}/{attempt}/results.json') for s in [0,1,2]],
             'source_sha256':digest(__file__), 'boundary':'posthoc descriptions; within-seed signed differences; all populations/targets; no tuning or causal claim'}
    write(OUT/'summary.json',summary)
    def value(stat,scale=1):
        return 'undefined' if stat['mean'] is None else f"{scale*stat['mean']:.3f} ± {scale*stat['sample_sd']:.3f}"
    lines=['# Frozen geometry diagnostics: components and decision transitions','',
           'This posthoc analysis describes the already audited three-seed geometry results. No model, detector, hyperparameter, threshold or dataset was fitted/changed. The fixed scope includes every population and all three additions × three references × four existing targets. Prior raw artifacts/source hashes remain intact.','',
           'Values are mean ± sample SD across training seeds 0/1/2. Within-population SD (ddof=0), covariance and every fixed quantile are separately preserved in summary.json. Seed SD is descriptive, not CI; population means/correlations cannot establish a cause of AUROC changes.','',
           '## ViM score components','',
           '`S = v + E`, `v = alpha * residual`, `E = -logsumexp(affine_logits)`. Higher S is more OOD. The variance identity `Var(S)=Var(v)+Var(E)+2 Cov(v,E)` was independently verified on all samples.','',
           '| Population | n per seed | Mean v | Mean E | Mean S | Pearson(v,E) |',
           '|---|---:|---:|---:|---:|---:|']
    for pop in POPULATIONS:
        p=populations[pop]
        lines.append(f"| {pop} | {int(p['vim']['n']['mean']):,} | {value(p['virtual_logit']['mean'])} | {value(p['affine_energy']['mean'])} | {value(p['vim']['mean'])} | {value(p['joint']['pearson_virtual_energy'])} |")
    lines+=['','Noise is a synthetic diagnostic and is never included in a natural OOD aggregate. Training/validation/test ID remain distinct populations.','',
            '## Signed population mean differences','',
            'These are paired differences within the same seed. Both component signs are preserved; delta S equals delta v plus delta E. This identity describes mean score shifts and does not replace rank-based detection metrics.','',
            '| Population minus reference | Delta v | Delta E | Delta S |',
            '|---|---:|---:|---:|']
    for comparison,values in differences.items():
        lines.append('| '+comparison+' | '+' | '.join(value(values[k]) for k in ['virtual_logit','affine_energy','vim'])+' |')
    lines+=['','## Exhaustive fixed-threshold decision transitions','',
            'The primary 95% ID-calibration target is shown below because it was already the fixed primary operating point. Every 80/90/95/99% target is saved in summary.json. All 108 seed/comparison/target settings and per-class ID counts remain available as raw transition codes.','',
            'Both methods use their own frozen threshold selected from the same 2,500 reused ID-validation calibration rows. Thus these transitions include threshold differences; they are not a pure ranking effect or a comparison at exactly equal achieved ID-test coverage. Correct/incorrect ID use original classifier predictions, with separate subgroup denominators. OOD populations have no classifier-correctness flag.','',
            '| Addition minus reference | Population | Both accept % | Addition only % | Reference only % | Neither % | Acceptance change (pp) |',
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
                        raw.append(q['addition_only_accepted']-q['reference_only_accepted'])
                    change=combine(raw)
                else:
                    change=p['acceptance_change']
                lines.append('| '+LABELS[addition]+' − '+LABELS[reference]+' | '+name+' | '+
                             ' | '.join(value(p['state_fractions'][k],100) for k in STATES)+' | '+value(change,100)+' |')
    lines+=['','## Scientific figures','',
            'The joint scatter uses the predeclared sorted random sample of at most 1,000 rows per population. All statistics and decision counts use full populations. The dashed ViM boundary is the existing 95%-calibration threshold, not a newly fitted line.','',
            '![Joint components: all seeds and populations](joint-components.png)','',
            'The next figure shows acceptance changes for all natural datasets against every fixed reference, with per-seed points and mean/sample SD. Noise and ID transition tables remain separate; no result is promoted as a selected method.','',
            '![Natural decision changes at the fixed primary target](natural-transitions.png)','',
            '## Reproduction and evidence','',
            '```powershell','& C:/Users/zxzx/anaconda3/python.exe tools/run_geometry_diagnostics.py',
            '& C:/Users/zxzx/anaconda3/python.exe tools/audit_geometry_diagnostics.py',
            '& C:/Users/zxzx/anaconda3/python.exe tools/build_geometry_diagnostics_report.py',
            '& C:/Users/zxzx/anaconda3/python.exe tools/audit_geometry_diagnostics_report.py','```','',
            'Raw vectors/sampled indices and original ID labels: runs/geometry-diagnostics/seedN/analysis/*_components.npz. Transition codes and [addition,reference,target,ID_class,state,correctness] counts: transitions.npz. Original masks are reused byte-for-byte. Completed audits bind all source/input/raw artifact hashes; completion is recorded separately after summary audit and visual QA.','',
            'No causal explanation, significance claim, OOD tuning or fresh-validation coverage guarantee follows from this posthoc diagnostic. A future method study needs a separate declared ID-only development protocol.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    palette=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,3,figsize=(17,6),constrained_layout=True)
    plot_manifest={'joint_scatter':{},'natural_transitions':{'target':.95,'population_order':NATURAL,'references':REFERENCES,'additions':NEW}}
    for seed,ax in enumerate(axes):
        for i,pop in enumerate(POPULATIONS):
            a=load(RUN/f'seed{seed}/{attempt}/{pop}_components.npz');idx=a['sample_rows']
            ax.scatter(a['affine_energy'][idx],a['virtual_logit'][idx],s=4,alpha=.25,color=palette[i],label=pop,rasterized=True)
        xlim,ylim=ax.get_xlim(),ax.get_ylim()
        threshold=read(ROOT/f'runs/geometry-phase3/seed{seed}/evaluation/results.json')['datasets']['cifar10']['methods']['vim']['primary_2500']['validation_threshold']
        ax.plot(xlim,[threshold-x for x in xlim],'k--',lw=1.2,label='frozen 95% ID threshold')
        ax.set(xlim=xlim,ylim=ylim,title=f'seed {seed}',xlabel='Affine Energy E',ylabel='Train-scaled residual v')
        ax.grid(alpha=.15)
        plot_manifest['joint_scatter'][str(seed)]={'xlim':list(xlim),'ylim':list(ylim),'existing_threshold':threshold,'sample_n_per_population':1000}
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=5,markerscale=3,fontsize=10)
    fig.suptitle('ViM = v + E: fixed subsampled descriptive scatter\n8 separate populations × 3 frozen seeds; statistics use all rows',fontsize=14)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'joint-components.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(14,13),sharex=True,constrained_layout=True)
    x=np.arange(3);colors=['#006d77','#e29526','#9b3a62']
    for di,dataset in enumerate(NATURAL):
        for ri,reference in enumerate(REFERENCES):
            ax=axes[di,ri]
            records_to_plot=[transitions[a][reference]['0.95'][dataset]['acceptance_change'] for a in NEW]
            ax.errorbar(x,[p['mean']*100 for p in records_to_plot],yerr=[p['sample_sd']*100 for p in records_to_plot],
                        fmt='none',ecolor='#555555',capsize=4,label='mean ± sample SD')
            for seed,color in enumerate(colors):
                ax.scatter(x+(seed-1)*.10,[p['per_seed'][seed]*100 for p in records_to_plot],s=25,color=color,label=f'seed {seed}',zorder=3)
            ax.axhline(0,ls=':',color='gray')
            ax.set(title=f'{dataset} vs {LABELS[reference]}',ylabel='OOD acceptance change (pp)')
            ax.set_xticks(x,[LABELS[a] for a in NEW]);ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncols=4,fontsize=10)
    fig.suptitle('Existing 95% ID-calibration target: every natural dataset and fixed reference\nNegative values mean fewer OOD accepted; own frozen thresholds are used',fontsize=14)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'natural-transitions.{ext}',dpi=160)
    plt.close(fig)
    write(OUT/'plot-manifest.json',plot_manifest)
    print(__import__('json').dumps({'state':summary['state'],'mean_component_differences':differences}),flush=True)


if __name__=='__main__':
    main()
