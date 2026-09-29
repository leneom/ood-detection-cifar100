"""Independent scalar covariance and variance-identity audit of every saved cell."""
import math,statistics
from pathlib import Path
import numpy as np
from geometry_io import ROOT,read,write,load,digest,require,now

RUN=ROOT/'runs/bootstrap-dependence';SOURCE=ROOT/'runs/paired-bootstrap'

def agree(got,expected,label,rtol=1e-9,atol=1e-12):
    if expected is None:require(got is None or (isinstance(got,float) and math.isnan(got)),label+' undefined value changed');return
    require(got is not None and math.isclose(float(got),float(expected),rel_tol=rtol,abs_tol=atol),label+' differs')

def scalar_moments(x):
    data=np.asarray(x,dtype=float)
    require(data.ndim==2 and data.shape[0]==1000,'bad scalar-audit draw shape')
    n,p=data.shape;cols=data.T.tolist();counts=[sum(math.isfinite(y) for y in col) for col in cols];valid=[c==n for c in counts]
    means=[None]*p;center=[None]*p;cov=[[None]*p for _ in range(p)];corr=[[None]*p for _ in range(p)];defined=[[False]*p for _ in range(p)]
    for i in range(p):
        if not valid[i]:continue
        shifted=[float(y-cols[i][0]) for y in cols[i]];base=statistics.fmean(shifted)
        means[i]=cols[i][0]+base;center[i]=[y-base for y in shifted]
    for i in range(p):
        if not valid[i]:continue
        for j in range(i,p):
            if not valid[j]:continue
            v=math.fsum(a*b for a,b in zip(center[i],center[j]))/(n-1);cov[i][j]=v;cov[j][i]=v
    for i in range(p):
        if cov[i][i] is None or cov[i][i]<=0:continue
        for j in range(p):
            if cov[j][j] is None or cov[j][j]<=0:continue
            corr[i][j]=cov[i][j]/(math.sqrt(cov[i][i])*math.sqrt(cov[j][j]));defined[i][j]=True
    return counts,means,cov,corr,defined

def scalar_decomposition(x):
    data=np.asarray(x,dtype=float);n=data.shape[1]
    counts,means,cov,corr,defined=scalar_moments(data)
    if any(v!=1000 for v in counts):
        d={'defined':False,'draws':1000,'components':n,**{k:None for k in ['observed_mean_variance','diagonal_contribution','cross_covariance_contribution','reconstructed_variance','signed_identity_residual','diagonal_only_counterfactual_variance','sd_ratio_to_diagonal_only']}}
    else:
        row_means=[statistics.fmean(row) for row in data.tolist()]
        _,_,mean_cov,_,_=scalar_moments(np.asarray(row_means)[:,None]);observed=mean_cov[0][0]
        diagonal=math.fsum(cov[i][i] for i in range(n))/(n*n)
        cross=2*math.fsum(cov[i][j] for i in range(n) for j in range(i+1,n))/(n*n)
        d={'defined':True,'draws':1000,'components':n,'observed_mean_variance':observed,
           'diagonal_contribution':diagonal,'cross_covariance_contribution':cross,'reconstructed_variance':diagonal+cross,
           'signed_identity_residual':observed-diagonal-cross,'diagonal_only_counterfactual_variance':diagonal,
           'sd_ratio_to_diagonal_only':math.sqrt(observed/diagonal) if diagonal>0 else None}
    return counts,means,cov,corr,defined,d

def check_matrix(arr,expected,label,rtol=1e-9,atol=1e-12):
    a=np.asarray(arr);e=np.asarray(expected,dtype=object)
    require(a.shape==e.shape,label+' shape changed')
    for ix in np.ndindex(a.shape):agree(a[ix].item() if hasattr(a[ix],'item') else a[ix],e[ix],label+str(ix),rtol,atol)
    return a.size

def main():
    status=read(RUN/'status.json');require(status['state']=='raw_complete_pending_independent_audit' and status['completed_endpoints']==66,'dependence producer incomplete')
    attempt=status['output_name'];out=RUN/attempt;pre=read(RUN/'predeclaration.json');manifest=read(RUN/f'{attempt}-inputs.json');raw=read(out/'raw-manifest.json')
    require(manifest['predeclaration_sha256']==digest(RUN/'predeclaration.json') and raw['analysis_inputs_sha256']==digest(RUN/f'{attempt}-inputs.json') and status['raw_manifest_sha256']==digest(out/'raw-manifest.json'),'frozen binding changed')
    require(manifest['original_artifacts']==pre['original_artifacts'] and raw['source_hashes']==manifest['source_hashes'],'input/source scope changed')
    for name,h in manifest['original_artifacts'].items():require(digest(ROOT/name)==h,'declared source changed '+name)
    for name,h in manifest['source_hashes'].items():require(digest(ROOT/name)==digest(RUN/f'{attempt}-source'/Path(name).name)==h,'executed source changed '+name)
    for name,h in raw['outputs'].items():require(digest(ROOT/name)==h,'raw dependence output changed '+name)
    original=read(ROOT/'report/paired-bootstrap/summary.json');base=read(SOURCE/'analysis-inputs.json');methods=base['methods'];fields=base['ranking_fields'];datasets=base['datasets'];targets=base['targets'];pairs=original['pairs']
    require(pre['methods']==methods and len(pairs)==105 and original['methods']==methods,'frozen method/pair order changed')
    blocks=[load(SOURCE/'analysis'/f'block-{i:03d}-metrics.npz') for i in range(10)];require([int(v) for b in blocks for v in b['replicates']]==list(range(1000)),'replicate index changed')
    # No import of dependence_io, dependence_core or producer endpoint reconstruction.
    endpoints=[]
    for di,d in enumerate(datasets):
        for fi,f in enumerate(fields):endpoints.append((f'ranking/{d}/{f}',np.concatenate([b['ranking'][:,:,di,:,fi] for b in blocks])))
    for fi,f in enumerate(fields[:4]):endpoints.append((f'ranking/natural_macro/{f}',np.concatenate([b['natural_ranking_macro'][:,:,:,fi] for b in blocks])))
    for ti,t in enumerate(targets):
        for key in ['id_coverage','retained_error_risk']:endpoints.append((f'operating/{t:g}/{key}',np.concatenate([b[key][:,:,:,ti] for b in blocks])))
        for di,d in enumerate(datasets):endpoints.append((f'operating/{t:g}/{d}_acceptance',np.concatenate([b['ood_acceptance'][:,:,:,ti,di] for b in blocks])))
        endpoints.append((f'operating/{t:g}/natural_ood_acceptance_macro',np.concatenate([b['natural_ood_acceptance_macro'][:,:,:,ti] for b in blocks])))
    index=read(out/'endpoint-index.json');require([p['endpoint'] for p in index['endpoints']]==[k for k,_ in endpoints] and index['methods']==methods and index['pairs']==pairs and len(endpoints)==66,'endpoint index/scope changed')
    checks={'endpoint_values':0,'endpoint_covariance_cells':0,'endpoint_correlation_cells':0,'endpoint_mean_count_cells':0,
            'model_or_pair_decompositions':0,'model_or_pair_covariance_cells':0,'natural_decompositions':0,'natural_covariance_cells':0,'json_fields':0}
    for ei,(key,raw_values) in enumerate(endpoints):
        actual=load(out/f'endpoint-{ei:03d}.npz');record=read(out/f'endpoint-{ei:03d}.json');flat=raw_values.reshape(1000,45)
        require(record['endpoint']==key and record['variables']==[f'seed{s}/{m}' for s in range(3) for m in methods],'variable order changed')
        require(actual['values'].shape==flat.shape and np.array_equal(actual['values'],flat,equal_nan=True),'raw endpoint vector differs');checks['endpoint_values']+=flat.size
        finite=np.isfinite(flat);require(np.array_equal(actual['finite_mask'],finite),'finite mask changed')
        counts,means,cov,corr,defined=scalar_moments(flat)
        require(record['finite_counts']==counts and actual['finite_counts'].tolist()==counts,'endpoint finite counts changed')
        require(np.array_equal(actual['correlation_defined'],defined),'undefined endpoint correlations changed')
        checks['endpoint_mean_count_cells']+=check_matrix(actual['means'],means,'endpoint mean')
        checks['endpoint_covariance_cells']+=check_matrix(actual['covariance'],cov,'endpoint covariance')
        checks['endpoint_correlation_cells']+=check_matrix(actual['correlation'],corr,'endpoint correlation')
        for i in range(45):agree(record['means'][i],means[i],'JSON endpoint mean');agree(record['marginal_variances'][i],cov[i][i],'JSON marginal variance');checks['json_fields']+=2
        require(record['correlation_undefined_count']==sum(not v for row in defined for v in row),'undefined correlation count differs')
        require(set(record['methods'])==set(methods) and set(record['pairs'])=={p['key'] for p in pairs},'method/pair coverage changed')
        for pi in range(120):
            if pi<15:kind='methods';name=methods[pi];component=raw_values[:,:,pi]
            else:
                kind='pairs';p=pairs[pi-15];name=p['key'];component=raw_values[:,:,methods.index(p['later'])]-raw_values[:,:,methods.index(p['earlier'])]
            cc,mm,cv,cr,df,dec=scalar_decomposition(component);entry=record[kind][name]
            require(entry['finite_counts']==cc and actual['model_or_pair_finite_counts'][pi].tolist()==cc,'three-model finite count changed')
            require(np.array_equal(actual['model_or_pair_correlation_diagonal_defined'][pi],[df[i][i] for i in range(3)]),'three-model undefined diagonal changed')
            checks['model_or_pair_covariance_cells']+=check_matrix(actual['model_or_pair_covariance'][pi],cv,'three-model covariance')
            check_matrix(actual['model_or_pair_correlation'][pi],cr,'three-model correlation')
            require(entry['correlation_defined']==df and entry['defined']==dec['defined'] and entry['draws']==1000 and entry['components']==3,'three-model scope/undefined changed')
            for i in range(3):
                for j in range(3):agree(entry['covariance'][i][j],cv[i][j],'JSON three-model covariance');agree(entry['correlation'][i][j],cr[i][j],'JSON three-model correlation');checks['json_fields']+=2
            for f in ['observed_mean_variance','diagonal_contribution','cross_covariance_contribution','reconstructed_variance','signed_identity_residual','diagonal_only_counterfactual_variance','sd_ratio_to_diagonal_only']:
                agree(entry[f],dec[f],'three-model '+f);checks['json_fields']+=1
            checks['model_or_pair_decompositions']+=1
        print('independent dependence endpoint passed:',ei,key,flush=True)
    require(checks['model_or_pair_decompositions']==7920,'three-model scope incomplete')
    nat=load(out/'natural-components.npz');description=read(out/'natural-components.json');require(description['natural_order']==datasets[:4] and len(description['records'])==360 and nat['values'].shape==(360,1000,4),'natural scope/count changed')
    ri=0
    for fi,f in enumerate(fields[:4]):
        for seed in range(3):
            for mi,m in enumerate(methods):
                key=f'ranking/{f}/seed{seed}/{m}';x=np.concatenate([b['ranking'][:,seed,:4,mi,fi] for b in blocks]);require(np.array_equal(nat['values'][ri],x,equal_nan=True),'natural ranking vector changed')
                macro=dict(endpoints)[f'ranking/natural_macro/{f}'][:,seed,mi];require(np.allclose(x.mean(1),macro,rtol=0,atol=1e-12,equal_nan=True),'natural ranking macro changed')
                record=description['records'][ri];require(record['key']==key and record['index']==ri,'natural ranking order changed')
                cc,mm,cv,cr,df,dec=scalar_decomposition(x)
                require(record['finite_counts']==cc and nat['finite_counts'][ri].tolist()==cc and np.array_equal(nat['correlation_defined'][ri],df),'natural masks/counts changed')
                check_matrix(nat['means'][ri],mm,'natural means');checks['natural_covariance_cells']+=check_matrix(nat['covariance'][ri],cv,'natural covariance');check_matrix(nat['correlation'][ri],cr,'natural correlation')
                require(record['correlation_defined']==df and record['natural_datasets']==datasets[:4] and record['defined']==dec['defined'],'natural report scope changed')
                for i in range(4):
                    for j in range(4):agree(record['covariance'][i][j],cv[i][j],'JSON natural covariance');agree(record['correlation'][i][j],cr[i][j],'JSON natural correlation');checks['json_fields']+=2
                for field in ['observed_mean_variance','diagonal_contribution','cross_covariance_contribution','reconstructed_variance','signed_identity_residual','diagonal_only_counterfactual_variance','sd_ratio_to_diagonal_only']:agree(record[field],dec[field],'natural '+field);checks['json_fields']+=1
                checks['natural_decompositions']+=1;ri+=1
    for ti,t in enumerate(targets):
        for seed in range(3):
            for mi,m in enumerate(methods):
                key=f'operating/{t:g}/seed{seed}/{m}';x=np.concatenate([b['ood_acceptance'][:,seed,mi,ti,:4] for b in blocks]);require(np.array_equal(nat['values'][ri],x,equal_nan=True),'natural operational vector changed')
                macro=dict(endpoints)[f'operating/{t:g}/natural_ood_acceptance_macro'][:,seed,mi];require(np.allclose(x.mean(1),macro,rtol=0,atol=1e-12,equal_nan=True),'natural operational macro changed')
                record=description['records'][ri];require(record['key']==key and record['index']==ri,'natural operation order changed')
                cc,mm,cv,cr,df,dec=scalar_decomposition(x)
                require(record['finite_counts']==cc and nat['finite_counts'][ri].tolist()==cc and np.array_equal(nat['correlation_defined'][ri],df),'natural masks/counts changed')
                check_matrix(nat['means'][ri],mm,'natural means');checks['natural_covariance_cells']+=check_matrix(nat['covariance'][ri],cv,'natural covariance');check_matrix(nat['correlation'][ri],cr,'natural correlation')
                require(record['correlation_defined']==df and record['natural_datasets']==datasets[:4] and record['defined']==dec['defined'],'natural report scope changed')
                for i in range(4):
                    for j in range(4):agree(record['covariance'][i][j],cv[i][j],'JSON natural covariance');agree(record['correlation'][i][j],cr[i][j],'JSON natural correlation');checks['json_fields']+=2
                for field in ['observed_mean_variance','diagonal_contribution','cross_covariance_contribution','reconstructed_variance','signed_identity_residual','diagonal_only_counterfactual_variance','sd_ratio_to_diagonal_only']:agree(record[field],dec[field],'natural '+field);checks['json_fields']+=1
                checks['natural_decompositions']+=1;ri+=1
    require(ri==360==checks['natural_decompositions'],'natural scope incomplete')
    for name,h in manifest['original_artifacts'].items():require(digest(ROOT/name)==h,'source changed during independent audit '+name)
    result={'state':'passed','audited_at':now(),'analysis_inputs_sha256':digest(RUN/f'{attempt}-inputs.json'),'raw_manifest_sha256':digest(out/'raw-manifest.json'),
            'independent_source_sha256':digest(__file__),'checks':checks,'bound_input_count':len(manifest['original_artifacts']),
            'boundary':'Every saved draw/variable/endpoint and every covariance/correlation/decomposition recomputed by scalar Python arithmetic; no pairwise deletion or method selection'}
    write(RUN/'raw-audit.json',result);print(result,flush=True)

if __name__=='__main__':main()
