"""Independent Python statistics/manual-quantile audit of the entire fixed report."""
from pathlib import Path
import math, statistics
import numpy as np
from geometry_io import ROOT, read, write, load, digest, require, now

RUN=ROOT/'runs/paired-bootstrap'; REPORT=ROOT/'report/paired-bootstrap'

def agree(actual,expected,label):
    if expected is None:require(actual is None,label+' undefined dropped');return
    require(actual is not None and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),label+' differs')

def manual_quantile(ordered,q):
    pos=(len(ordered)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)

def main():
    summary=read(REPORT/'summary.json');build=read(REPORT/'build.json');binding=read(REPORT/'inputs.json');raw=read(RUN/'raw-audit.json');manifest=read(RUN/'analysis-inputs.json')
    require(raw['state']=='passed' and raw['completed_replicates']==1000,'raw schedule/audit incomplete')
    require(build['inputs_sha256']==digest(REPORT/'inputs.json') and binding['raw_audit_sha256']==digest(RUN/'raw-audit.json') and binding['input_manifest_sha256']==digest(RUN/'analysis-inputs.json'),'report binding changed')
    for name,h in build['artifacts'].items():require(digest(REPORT/name)==h,'report artifact changed')
    for n,h in {**manifest['original_artifacts'],**raw['output_artifacts']}.items():require(digest(ROOT/n)==h,'frozen source/raw input changed')
    for n,h in binding['source_hashes'].items():require(digest(ROOT/n)==digest(REPORT/'executed-source'/Path(n).name)==h,'executed report source changed')
    methods=manifest['methods'];datasets=manifest['datasets'];targets=manifest['targets'];fields=manifest['ranking_fields']
    require(summary['methods']==methods and summary['targets']==targets and summary['replicates']==1000 and summary['quantiles']==[.025,.975] and summary['quantile_method']=='linear' and summary['sample_sd_ddof']==1,'statistical declaration changed')
    pairs=[{'earlier':methods[i],'later':methods[j],'key':methods[j]+'__minus__'+methods[i]} for i in range(15) for j in range(i+1,15)]
    require(summary['pairs']==pairs and len(pairs)==105,'pair coverage/sign changed')
    blocks=[load(RUN/'analysis'/f'block-{i:03d}-metrics.npz') for i in range(10)]
    require([int(v) for b in blocks for v in b['replicates']]==list(range(1000)),'draw index order changed')
    originals=[read(ROOT/f'runs/ash-phase5/seed{s}/evaluation/results.json') for s in range(3)]
    op=[{(x['method'],x['target']):x for x in r['operating_points']} for r in originals]
    endpoints=[]
    for di,d in enumerate(datasets):
        for fi,f in enumerate(fields):
            v=np.concatenate([b['ranking'][:,:,di,:,fi] for b in blocks]);pt=[[r['datasets'][d]['methods'][m]['primary_2500'][f] for m in methods] for r in originals]
            endpoints.append((f'ranking/{d}/{f}',v,pt))
    for fi,f in enumerate(fields[:4]):
        # Reconstruct equal natural weights directly from ranking, independent of saved macro.
        v=np.concatenate([sum(b['ranking'][:,:,di,:,fi] for di in range(4))/4 for b in blocks]);pt=[[statistics.fmean(r['datasets'][d]['methods'][m]['primary_2500'][f] for d in datasets[:4]) for m in methods] for r in originals]
        endpoints.append((f'ranking/natural_macro/{f}',v,pt))
    for ti,t in enumerate(targets):
        for key,field in [('id_coverage','coverage'),('retained_error_risk','retained_error_risk')]:
            endpoints.append((f'operating/{t:g}/{key}',np.concatenate([b[key][:,:,:,ti] for b in blocks]),[[r[m,t]['id_test'][field] for m in methods] for r in op]))
        for di,d in enumerate(datasets):
            endpoints.append((f'operating/{t:g}/{d}_acceptance',np.concatenate([b['ood_acceptance'][:,:,:,ti,di] for b in blocks]),[[r[m,t]['ood'][d]['acceptance'] for m in methods] for r in op]))
        v=np.concatenate([sum(b['ood_acceptance'][:,:,:,ti,di] for di in range(4))/4 for b in blocks]);pt=[[statistics.fmean(r[m,t]['ood'][d]['acceptance'] for d in datasets[:4]) for m in methods] for r in op]
        endpoints.append((f'operating/{t:g}/natural_ood_acceptance_macro',v,pt))
    require(set(summary['endpoints'])=={k for k,_,_ in endpoints} and len(endpoints)==66,'endpoint coverage changed')
    checks=0;numeric=0
    def check_stat(item,values,points,fixed,label):
        nonlocal checks,numeric
        require(set(item)=={'original_point','defined_replicates','replicates','bootstrap_mean','bootstrap_sample_sd','percentile_lower','percentile_upper','original_three_seed_sample_sd'},'stat field coverage changed')
        vals=[float(v) for v in values];finite=sum(math.isfinite(v) for v in vals)
        point=statistics.fmean(points) if fixed else float(points)
        expected={'original_point':point if math.isfinite(point) else None,'defined_replicates':finite,'replicates':1000,'bootstrap_mean':None,'bootstrap_sample_sd':None,'percentile_lower':None,'percentile_upper':None,'original_three_seed_sample_sd':None}
        if fixed and all(math.isfinite(v) for v in points):expected['original_three_seed_sample_sd']=statistics.stdev(points)
        if finite==1000:
            ordered=sorted(vals);expected.update(bootstrap_mean=statistics.fmean(vals),bootstrap_sample_sd=statistics.stdev(vals),percentile_lower=manual_quantile(ordered,.025),percentile_upper=manual_quantile(ordered,.975))
        for k,e in expected.items():agree(item[k],e,label+'/'+k);numeric+=1
        checks+=1
    for key,v,pt in endpoints:
        entry=summary['endpoints'][key];require(set(entry['methods'])==set(methods) and set(entry['pairs'])=={p['key'] for p in pairs},'method/pair missing')
        for kind,items in [('methods',[(m,i,None) for i,m in enumerate(methods)]),('pairs',[(p['key'],methods.index(p['later']),methods.index(p['earlier'])) for p in pairs])]:
            for name,j,i in items:
                item=entry[kind][name];require(set(item)=={'seed0','seed1','seed2','fixed_three'},'seed scope changed')
                points=[float(pt[s][j] if i is None else pt[s][j]-pt[s][i]) for s in range(3)]
                values=v[:,:,j] if i is None else v[:,:,j]-v[:,:,i]
                for s in range(3):check_stat(item[f'seed{s}'],values[:,s],points[s],False,key+'/'+name+f'/seed{s}')
                fixed=[statistics.fmean(float(x) for x in row) for row in values]
                check_stat(item['fixed_three'],fixed,points,True,key+'/'+name+'/fixed_three')
        print('summary endpoint passed:',key,flush=True)
    require(checks==31680==summary['statistic_records'],'summary record count differs')
    # Exact text values are rebuilt independently; no producer display helper import.
    def display(x):return 'unavailable' if x['percentile_lower'] is None else f"{100*x['original_point']:.4f} [{100*x['percentile_lower']:.4f}, {100*x['percentile_upper']:.4f}]"
    text=(REPORT/'REPORT.md').read_text(encoding='utf-8');display_checks=0
    tablekeys=['ranking/natural_macro/auroc']+[f'ranking/{d}/auroc' for d in datasets]+[f'operating/{t:g}/{f}' for t in targets for f in ['id_coverage','retained_error_risk','natural_ood_acceptance_macro']]
    for key in tablekeys:
        for m in methods:
            x=summary['endpoints'][key]['methods'][m]['fixed_three'];line=f"| {m} | {display(x)} | {100*x['original_three_seed_sample_sd']:.4f} |";require(line in text,'displayed method row differs');display_checks+=5
    pairkeys=['ranking/natural_macro/auroc','operating/0.95/natural_ood_acceptance_macro','operating/0.95/id_coverage','operating/0.95/retained_error_risk']
    for p in pairs:
        line='| '+p['later']+' minus '+p['earlier']+' | '+' | '.join(display(summary['endpoints'][k]['pairs'][p['key']]['fixed_three']) for k in pairkeys)+' |'
        require(line in text,'displayed pair row differs');display_checks+=16
    plots=read(REPORT/'plot-data.json');expected_keys=[f'ranking/{d}/auroc' for d in datasets[:4]]+[f'operating/0.95/{f}' for f in ['natural_ood_acceptance_macro','id_coverage','retained_error_risk']]
    require(set(plots)==set(expected_keys),'figure endpoint scope differs');plot_checks=0
    for key in expected_keys:
        p=plots[key];require(p['method_order']==methods,'figure reordered methods')
        for mi,m in enumerate(methods):
            st=summary['endpoints'][key]['methods'][m]
            for name,field in [('original_percent','original_point'),('lower_percent','percentile_lower'),('upper_percent','percentile_upper')]:agree(p[name][mi],100*st['fixed_three'][field],'figure interval');plot_checks+=1
            for s in range(3):agree(p['original_seed_percent'][s][mi],100*st[f'seed{s}']['original_point'],'figure seed point');plot_checks+=1
    for n,h in manifest['original_artifacts'].items():require(digest(ROOT/n)==h,'input changed during summary audit')
    audit={'state':'passed','audited_at':now(),'statistic_records':checks,'statistic_fields':numeric,'display_fields':display_checks,'plot_fields':plot_checks,'method_pairs':105,'endpoints':66,'summary_sha256':digest(REPORT/'summary.json'),'build_sha256':digest(REPORT/'build.json'),'independent_source_sha256':digest(__file__),'boundary':'Independent Python statistics/sample-SD/manual linear quantiles, every method/pair/seed/endpoint; plot coordinates audited, visual inspection remains required'}
    write(REPORT/'summary-audit.json',audit);print(audit,flush=True)

if __name__=='__main__':main()
