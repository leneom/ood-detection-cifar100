"""Independent Python-statistics checks of every seed record and displayed table."""
import statistics
from pathlib import Path
from geometry_io import ROOT, read, write, digest, load, require, now

RUN = ROOT/'runs/geometry-diagnostics'
OUT = ROOT/'report/geometry-diagnostics'
POPULATIONS = ['training_id','validation_id','test_id','cifar10','mnist','svhn','textures','noise']
NEW, REFERENCES, TARGETS = ['vim','residual','react_energy'], ['affine_energy','energy','msp'], [.8,.9,.95,.99]
NATURAL = ['cifar10','mnist','svhn','textures']
STATES = ['both_accepted','addition_only_accepted','reference_only_accepted','neither_accepted']
LABELS = dict(vim='ViM',residual='Residual',react_energy='ReAct',affine_energy='Affine Energy',energy='Energy',msp='MSP')


def main():
    raw,summary=read(OUT/'audit.json'),read(OUT/'summary.json')
    require(raw['state']=='passed' and summary['raw_audit_sha256']==digest(OUT/'audit.json'),'raw audit mismatch')
    require(summary['source_sha256']==digest(ROOT/'tools/build_geometry_diagnostics_report.py'),'report producer changed')
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
            require(abs(saved['mean']-statistics.fmean(nodes))<1e-11 and abs(saved['sample_sd']-statistics.stdev(nodes))<1e-11,'mean/sample SD mismatch')
        checks+=1
    check_tree(summary['population_statistics'],[r['population_statistics'] for r in results],'population')
    check_tree(summary['mean_component_differences'],[r['mean_component_differences'] for r in results],'signed differences')
    records=[{(p['addition'],p['reference'],p['target']):p for p in r['transitions']} for r in results]
    require(set(summary['transitions'])==set(NEW),'transition methods missing')
    for a in NEW:
        require(set(summary['transitions'][a])==set(REFERENCES),'reference omitted')
        for ref in REFERENCES:
            require(set(summary['transitions'][a][ref])=={str(t) for t in TARGETS},'target omitted')
            for t in TARGETS:
                check_tree(summary['transitions'][a][ref][str(t)],[r[(a,ref,t)]['populations'] for r in records],'transition')
    def formatted(values,scale=1):
        return 'undefined' if any(v is None for v in values) else f'{scale*statistics.fmean(values):.3f} ± {scale*statistics.stdev(values):.3f}'
    report=(OUT/'REPORT.md').read_text(encoding='utf-8')
    for pop in POPULATIONS:
        values=[r['population_statistics'][pop] for r in results]
        expected=f"| {pop} | {values[0]['vim']['n']:,} | "+' | '.join(
            [formatted([r[key]['mean'] for r in values]) for key in ['virtual_logit','affine_energy','vim']]+
            [formatted([r['joint']['pearson_virtual_energy'] for r in values])])+' |'
        require(expected in report,'population table mismatch')
        table_checks+=4
    for comparison in results[0]['mean_component_differences']:
        expected='| '+comparison+' | '+' | '.join(formatted([r['mean_component_differences'][comparison][k] for r in results])
                                                 for k in ['virtual_logit','affine_energy','vim'])+' |'
        require(expected in report,'signed component table mismatch')
        table_checks+=3
    for addition in NEW:
        for reference in REFERENCES:
            for name in ['test_id','correct ID','incorrect ID']+NATURAL+['noise']:
                values=[r[(addition,reference,.95)]['populations'] for r in records]
                subgroup=name in ['correct ID','incorrect ID']
                group='correct' if name=='correct ID' else 'incorrect'
                points=[r['test_id'][group] if subgroup else r[name] for r in values]
                changes=[p['state_fractions']['addition_only_accepted']-p['state_fractions']['reference_only_accepted'] if subgroup else p['acceptance_change'] for p in points]
                expected='| '+LABELS[addition]+' − '+LABELS[reference]+' | '+name+' | '+' | '.join(
                         [formatted([p['state_fractions'][s] for p in points],100) for s in STATES]+[formatted(changes,100)])+' |'
                require(expected in report,'transition table mismatch')
                table_checks+=5
    plot=read(OUT/'plot-manifest.json')
    require(plot['natural_transitions']==dict(target=.95,population_order=NATURAL,references=REFERENCES,additions=NEW),'natural figure changed fixed target/scope')
    sample_points=0
    for seed in [0,1,2]:
        plot_seed=plot['joint_scatter'][str(seed)]
        frozen=read(ROOT/f'runs/geometry-phase3/seed{seed}/evaluation/results.json')
        require(plot_seed['existing_threshold']==frozen['datasets']['cifar10']['methods']['vim']['primary_2500']['validation_threshold'],'plotted threshold refitted')
        for pop in POPULATIONS:
            a=load(RUN/f'seed{seed}/{attempt}/{pop}_components.npz');rows=a['sample_rows']
            require(plot_seed['sample_n_per_population']==len(rows),'scatter count mismatch')
            xx,yy=a['affine_energy'][rows],a['virtual_logit'][rows]
            require(xx.min()>=plot_seed['xlim'][0] and xx.max()<=plot_seed['xlim'][1] and
                    yy.min()>=plot_seed['ylim'][0] and yy.max()<=plot_seed['ylim'][1],'scatter points cropped')
            sample_points+=len(rows)
    outputs=[OUT/n for n in ['summary.json','REPORT.md','plot-manifest.json','joint-components.png','joint-components.svg','natural-transitions.png','natural-transitions.svg']]
    require(all(p.exists() and p.stat().st_size>1000 for p in outputs),'report artifact incomplete')
    audit={'state':'passed','audited_at':now(),'statistic_records_checked':checks,'displayed_table_fields_checked':table_checks,
           'scatter_points_and_bounds_checked':sample_points,'raw_audit_sha256':digest(OUT/'audit.json'),
           'artifacts':{p.name:digest(p) for p in outputs},'independent_source_sha256':digest(__file__),
           'boundary':'Python statistics; no missing seeds or undefined-value imputation; every fixed comparison and target retained'}
    write(OUT/'summary-audit.json',audit)
    print(__import__('json').dumps(audit),flush=True)


if __name__=='__main__':
    main()
