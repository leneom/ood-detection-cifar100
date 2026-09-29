"""Independent Python-statistics checks of every seed record and displayed table."""
import statistics
from pathlib import Path
from geometry_io import ROOT, read, write, digest, load, require, now

RUN = ROOT/'runs/ash-diagnostics'
OUT = ROOT/'report/ash-diagnostics'
POPULATIONS = ['validation_id','test_id','cifar10','mnist','svhn','textures','noise']
NEW=['ash_p_energy','ash_b_energy','ash_s_energy']
REFERENCES=['msp','energy','maxlogit','entropy','knn','mahalanobis','vim','residual','react_energy','affine_energy','rmd','mahalanobis_cpu']
TARGETS=[.8,.9,.95,.99]
NATURAL = ['cifar10','mnist','svhn','textures']
STATES = ['both_accept','ash_only_accept','reference_only_accept','neither_accept']
LABELS = dict(ash_p_energy='ASH-P',ash_b_energy='ASH-B',ash_s_energy='ASH-S',rmd='RMD',msp='MSP',energy='Energy',maxlogit='MaxLogit',entropy='Entropy',knn='KNN',mahalanobis='Mahalanobis',vim='ViM',residual='Residual',react_energy='ReAct',affine_energy='Affine Energy*',mahalanobis_cpu='CPU MD*')


def main():
    raw,summary=read(OUT/'audit.json'),read(OUT/'summary.json')
    require(raw['state']=='passed' and summary['raw_audit_sha256']==digest(OUT/'audit.json'),'raw audit mismatch')
    require(summary['source_sha256']==digest(ROOT/'tools/build_ash_diagnostics_report.py'),'report producer changed')
    require(summary['seeds']==[0,1,2] and summary['population_order']==POPULATIONS and summary['new_methods']==NEW and
            summary['references']==REFERENCES and summary['targets']==TARGETS,'scope changed')
    for name,h in raw['output_artifacts'].items():
        require(digest(ROOT/name)==h,'raw artifact changed')
    attempt=raw['output_name']
    results=[read(RUN/f'seed{s}/{attempt}/results.json') for s in [0,1,2]]
    require(summary['result_sha256']==[digest(RUN/f'seed{s}/{attempt}/results.json') for s in [0,1,2]],'summary source hashes changed')
    checks=table_checks=0
    def check_tree(saved,nodes,label):
        nonlocal checks
        if isinstance(nodes[0],dict):
            require(set(saved)==set(nodes[0]) and all(set(n)==set(nodes[0]) for n in nodes),'statistic scope omitted: '+label)
            for key in nodes[0]:
                check_tree(saved[key],[n[key] for n in nodes],label+'/'+key)
            return
        require(set(saved)=={'per_seed','mean','sample_sd','defined_seed_count'} and len(nodes)==3,'invalid seed statistic schema')
        require(saved['per_seed']==nodes and saved['defined_seed_count']==sum(n is not None for n in nodes),'paired seed values changed')
        if any(n is None for n in nodes):
            require(saved['mean'] is None and saved['sample_sd'] is None,'undefined seeds silently dropped')
        else:
            require(abs(saved['mean']-statistics.fmean(nodes))<=1e-8+1e-10*abs(statistics.fmean(nodes)) and abs(saved['sample_sd']-statistics.stdev(nodes))<=1e-8+1e-10*abs(statistics.stdev(nodes)),'mean/sample SD mismatch')
        checks+=1
    check_tree(summary['population_statistics'],[r['population_statistics'] for r in results],'population')
    check_tree(summary['classification_statistics'],[r['classification_statistics'] for r in results],'classification')
    check_tree(summary['mean_component_differences'],[r['mean_component_differences'] for r in results],'signed differences')
    records=[{(p['addition'],p['reference'],p['target']):p for p in r['transitions']} for r in results]
    require(set(summary['transitions'])==set(NEW),'transition methods missing')
    for a in NEW:
        require(set(summary['transitions'][a])==set(REFERENCES),'reference omitted')
        for ref in REFERENCES:
            require(set(summary['transitions'][a][ref])=={str(t) for t in TARGETS},'target omitted')
            for t in TARGETS:
                check_tree(summary['transitions'][a][ref][str(t)],[r[(a,ref,t)]['populations'] for r in records],'transition')
    require(set(summary['natural_macro_transitions'])==set(NEW),'natural macro method scope changed')
    for a in NEW:
        require(set(summary['natural_macro_transitions'][a])==set(REFERENCES),'natural macro references changed')
        for ref in REFERENCES:
            require(set(summary['natural_macro_transitions'][a][ref])=={str(t) for t in TARGETS},'natural macro targets changed')
            for t in TARGETS:
                nodes=[{k:statistics.fmean(r[(a,ref,t)]['populations'][d][k] for d in NATURAL) for k in ['addition_acceptance','reference_acceptance','acceptance_change']} for r in records]
                check_tree(summary['natural_macro_transitions'][a][ref][str(t)],nodes,'within-seed natural macro')
    def formatted(values,scale=1):
        return 'undefined' if any(v is None for v in values) else f'{scale*statistics.fmean(values):.3f} ± {scale*statistics.stdev(values):.3f}'
    report=(OUT/'REPORT.md').read_text(encoding='utf-8')
    require('All 432 seed/comparison/target settings' in report,'displayed setting count changed')
    for pop in POPULATIONS:
        vals=[r['population_statistics'][pop] for r in results]
        expected=f"| {pop} | {vals[0]['ratio']['n']:,} | "+' | '.join(formatted([v[k]['mean'] for v in vals]) for k in ['retained_fraction','ratio','exp_ratio','binary_fill'])+' |'
        require(expected in report,'support table mismatch');table_checks+=4
        for m in NEW:
            expected='| '+pop+' | '+LABELS[m]+' | '+formatted([v[m+'_minus_affine']['mean'] for v in vals])+' | '+formatted([v['joint'][m]['pearson_fraction_delta'] for v in vals])+' |'
            require(expected in report,'Energy/correlation table mismatch');table_checks+=2
    for comparison in results[0]['mean_component_differences']:
        expected='| '+comparison+' | '+' | '.join(formatted([r['mean_component_differences'][comparison][k] for r in results]) for k in ['retained_fraction']+[m+'_minus_affine' for m in NEW])+' |'
        require(expected in report,'signed difference table mismatch');table_checks+=4
    for pop in ['test_id','validation_id']:
        for m in NEW:
            pts=[r['classification_statistics'][pop][m] for r in results]
            parts=[formatted([p[k] for p in pts],100) for k in ['original_accuracy','shaped_accuracy']]+[formatted([p['state_fractions'][k] for p in pts],100) for k in ['shaped_only_correct','original_only_correct']]+[formatted([p['accuracy_change'] for p in pts],100)]
            expected='| '+pop+' | '+LABELS[m]+' | '+' | '.join(parts)+' | '+str([p['argmax_changes'] for p in pts])+' |'
            require(expected in report,'classification display mismatch');table_checks+=6
    for addition in NEW:
        for reference in REFERENCES:
            for name in ['test_id','correct ID','incorrect ID']+NATURAL+['noise']:
                values=[r[(addition,reference,.95)]['populations'] for r in records]
                subgroup=name in ['correct ID','incorrect ID']
                group='correct' if name=='correct ID' else 'incorrect'
                points=[r['test_id'][group] if subgroup else r[name] for r in values]
                changes=[p['state_fractions']['ash_only_accept']-p['state_fractions']['reference_only_accept'] if subgroup else p['acceptance_change'] for p in points]
                expected='| '+LABELS[addition]+' − '+LABELS[reference]+' | '+name+' | '+' | '.join(
                         [formatted([p['state_fractions'][s] for p in points],100) for s in STATES]+[formatted(changes,100)])+' |'
                require(expected in report,'transition table mismatch')
                table_checks+=5
    for a in NEW:
        for ref in REFERENCES:
            nodes=[r[(a,ref,.95)]['populations'] for r in records]
            nat=[statistics.fmean(r[d]['acceptance_change'] for d in NATURAL) for r in nodes]
            expected='| '+LABELS[a]+' − '+LABELS[ref]+' | '+formatted(nat,100)+' | '+formatted([r['test_id']['acceptance_change'] for r in nodes],100)+' | '+formatted([r['test_id']['retained_error_risk_change'] for r in nodes],100)+' |'
            require(expected in report,'paired own-threshold endpoint display mismatch');table_checks+=3
    plot=read(OUT/'plot-manifest.json')
    require(plot['natural_transitions']==dict(target=.95,population_order=NATURAL,references=REFERENCES,additions=NEW),'natural figure changed fixed target/scope')
    sample_points=full_points=0
    for seed in [0,1,2]:
        for m in NEW:
            plot_seed=plot['support_scatter'][str(seed)][m]
            require(plot_seed['bounds_from_full_populations'],'scatter bounds selected from subsample')
            xxmin,xxmax,yymin,yymax=float('inf'),float('-inf'),float('inf'),float('-inf')
            for pop in POPULATIONS:
                a=load(RUN/f'seed{seed}/{attempt}/{pop}_components.npz');rows=a['sample_rows']
                require(plot_seed['sample_n_per_population']==len(rows),'scatter count mismatch')
                xx,yy=a['retained_fraction'],a[m+'_minus_affine']
                require(xx.min()>=plot_seed['xlim'][0] and xx.max()<=plot_seed['xlim'][1] and yy.min()>=plot_seed['ylim'][0] and yy.max()<=plot_seed['ylim'][1],'full population points cropped')
                xxmin=min(xxmin,float(xx.min()));xxmax=max(xxmax,float(xx.max()));yymin=min(yymin,float(yy.min()));yymax=max(yymax,float(yy.max()))
                sample_points+=len(rows);full_points+=len(xx)
            xp=max((xxmax-xxmin)*.03,1e-9);yp=max((yymax-yymin)*.03,1e-9)
            require(plot_seed['xlim']==[xxmin-xp,xxmax+xp] and plot_seed['ylim']==[yymin-yp,yymax+yp],'full-population-bound formula changed')
    outputs=[OUT/n for n in ['summary.json','REPORT.md','plot-manifest.json','support-scale.png','support-scale.svg','natural-transitions.png','natural-transitions.svg']]
    require(all(p.exists() and p.stat().st_size>1000 for p in outputs),'report artifact incomplete')
    audit={'state':'passed','audited_at':now(),'statistic_records_checked':checks,'displayed_table_fields_checked':table_checks,
           'scatter_points_and_bounds_checked':sample_points,'full_rows_and_bounds_checked':full_points,'raw_audit_sha256':digest(OUT/'audit.json'),
           'artifacts':{p.name:digest(p) for p in outputs},'independent_source_sha256':digest(__file__),
           'boundary':'Python statistics; no missing seeds or undefined-value imputation; every fixed comparison and target retained'}
    write(OUT/'summary-audit.json',audit)
    print(__import__('json').dumps(audit),flush=True)


if __name__=='__main__':
    main()
