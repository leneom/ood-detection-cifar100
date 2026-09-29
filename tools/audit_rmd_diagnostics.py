"""Independent frozen subtraction identity and Python moments/contingency reconstruction."""
from collections import Counter
import math
import statistics
from pathlib import Path
import numpy as np
from geometry_io import ROOT, digest, read, write, load, require, now, METHODS, ADDITIONS, DATASETS, FOLDERS

RUN = ROOT / 'runs/rmd-diagnostics'
GEOMETRY = ROOT / 'runs/rmd-phase4'
OUT = ROOT / 'report/rmd-diagnostics'
POPULATIONS = ['validation_id','test_id'] + DATASETS
NEW, REFERENCES, TARGETS = ['rmd'], METHODS+ADDITIONS+['mahalanobis_cpu'], [.8,.9,.95,.99]
COMPONENTS = ['mahalanobis_cpu','background_distance','rmd']
STATES = ['both_accept','rmd_only_accept','reference_only_accept','neither_accept']
QUANTILES = [.01,.05,.25,.5,.75,.95,.99]
ALL = METHODS + ADDITIONS + ['rmd','mahalanobis_cpu']


def main():
    status = read(RUN/'status.json')
    require(status['state'] in ['raw_complete_pending_independent_audit','complete'] and status['completed_seeds'] == [0,1,2], 'producer incomplete')
    attempt = status['output_name']
    manifest_path = RUN/f'{attempt}-inputs.json'
    manifest = read(manifest_path)
    for name,h in manifest['original_artifacts'].items():
        require(digest(ROOT/name) == h,'frozen input changed: '+name)
    for name,h in manifest['source_hashes'].items():
        require(digest(ROOT/name) == digest(RUN/f'{attempt}-source'/Path(name).name) == h,'executed source changed')
    require(read(RUN/'predeclaration.json')['declared_at'] < manifest['prepared_at'],'diagnostics predate scope declaration')
    counts = dict(component_arrays=0,moment_fields=0,quantile_fields=0,joint_fields=0,mean_difference_fields=0,
                  sampled_indices=0,transition_code_arrays=0,transition_fields=0,per_class_cells=0,original_label_arrays=0,per_class_denominators=0)
    hashes = {}
    def close(saved,actual,label,tol=1e-11):
        require(saved is None if actual is None else saved is not None and abs(saved-actual) <= 1e-8+1e-10*abs(actual),label)
    def fields(saved,actual,label,kind):
        for key,value in actual.items():
            close(saved[key],value,label+'/'+key)
            counts[kind]+=1
    def vector_summary(array):
        a=array.tolist()
        ordered=sorted(a)
        mean=statistics.fmean(a)
        quantiles={}
        for q in QUANTILES:
            position=q*(len(a)-1)
            lower=math.floor(position); upper=math.ceil(position); fraction=position-lower
            lo,hi=ordered[lower],ordered[upper]
            quantiles[str(q)]=lo+(hi-lo)*fraction if fraction < .5 else hi-(hi-lo)*(1-fraction)
        return dict(n=len(a),mean=mean,population_sd=statistics.pstdev(a),min=ordered[0],max=ordered[-1]),quantiles
    for seed in [0,1,2]:
        directory=RUN/f'seed{seed}'/attempt
        result=read(directory/'results.json')
        require(result['state']=='complete' and result['seed']==seed and result['inputs_sha256']==digest(manifest_path) and
                result['computed_at']>manifest['prepared_at'],'seed diagnostic provenance changed')
        require(result['population_order']==POPULATIONS and result['new_methods']==NEW and result['references']==REFERENCES and
                result['targets']==TARGETS and result['transition_states']==STATES,'diagnostic axes changed')
        for name,h in result['output_file_sha256'].items():
            require(digest(directory/name)==h,'diagnostic artifact changed')
            hashes[str((directory/name).relative_to(ROOT)).replace('\\','/')]=h
        hashes[str((directory/'results.json').relative_to(ROOT)).replace('\\','/')]=digest(directory/'results.json')
        original = ROOT/f'runs/cifar100-seed{seed}/eval-final'
        id_data = load(original/'id_logits.npz')
        split = load(ROOT/'runs/calibration-phase2/calibration_split.npz')
        correct=id_data['logits'].argmax(1)==id_data['labels']
        fit=read(GEOMETRY/f'seed{seed}/fit.json')
        require(result['checkpoint_sha256']==fit['checkpoint_sha256'] and result['fit_json_sha256']==digest(GEOMETRY/f'seed{seed}/fit.json'),'fit mismatch')
        require(set(result['population_statistics'])==set(POPULATIONS),'population summary omitted')
        require(set(result['mean_component_differences'])=={f'{d}_minus_test_id' for d in ['validation_id']+DATASETS},'signed difference scope omitted')
        populations={}
        for ordinal,pop in enumerate(POPULATIONS):
            raw=load(directory/f'{pop}_components.npz')
            key='id' if pop=='test_id' else 'validation' if pop=='validation_id' else pop
            middle=load(GEOMETRY/f'seed{seed}/evaluation/{key}_intermediates.npz')
            part='id' if pop=='test_id' else 'val' if pop=='validation_id' else 'ood'
            dataset='cifar10' if pop in ['test_id','validation_id'] else pop
            saved=load(GEOMETRY/f'seed{seed}/evaluation/{dataset}_scores.npz')
            for k in COMPONENTS:
                require(raw[k].dtype==np.float64 and np.array_equal(raw[k],middle[k]),'frozen component changed: '+k)
                counts['component_arrays']+=1
            require(np.array_equal(raw['rmd'],saved[f'rmd_{part}']) and np.array_equal(raw['mahalanobis_cpu'],saved[f'mahalanobis_cpu_{part}']),'frozen scores changed')
            require(np.allclose(raw['rmd'],raw['mahalanobis_cpu']-raw['background_distance'],atol=1e-8,rtol=1e-10),'signed component identity failed')
            n=len(raw['rmd'])
            require(n=={'validation_id':5000,'test_id':10000,'cifar10':10000,'mnist':10000,'svhn':26032,'textures':1880,'noise':10000}[pop],'population count changed')
            require(np.array_equal(raw['original_rows'],np.arange(n,dtype=np.int64)),'original row order changed')
            if pop in ['test_id','validation_id']:
                labels=id_data['labels'] if pop=='test_id' else split['validation_labels']
                predictions=id_data['logits'].argmax(1) if pop=='test_id' else load(original/'validation_logits.npz')['logits'].argmax(1)
                require(np.array_equal(raw['labels'],labels) and np.array_equal(raw['original_predictions'],predictions),'original ID labels/predictions changed')
                counts['original_label_arrays']+=2
            expected=np.sort(np.random.default_rng(6829+100*seed+ordinal).choice(n,size=min(1000,n),replace=False))
            require(np.array_equal(raw['sample_rows'],expected),'fixed sample rows changed')
            counts['sampled_indices']+=1
            independent={}
            for key in COMPONENTS:
                stats,q=vector_summary(raw[key])
                fields(result['population_statistics'][pop][key],stats,'independent population moments','moment_fields')
                fields(result['population_statistics'][pop][key]['quantiles'],q,'sorted linear quantile','quantile_fields')
                independent[key]=stats
            v,e,s=[raw[k].tolist() for k in ['mahalanobis_cpu','background_distance','rmd']]
            vm,em=statistics.fmean(v),statistics.fmean(e)
            vv,ev,sv=statistics.pvariance(v),statistics.pvariance(e),statistics.pvariance(s)
            cov=math.fsum((a-vm)*(b-em) for a,b in zip(v,e))/len(v)
            variance_sum=vv+ev-2*cov
            joint=dict(n=len(v),MD_variance=vv,background_variance=ev,rmd_variance=sv,covariance_MD_background=cov,
                       pearson_MD_background=cov/math.sqrt(vv*ev) if vv>0 and ev>0 else None,
                       variance_component_difference=variance_sum,variance_identity_error=abs(sv-variance_sum),
                       score_identity_max_error=max(abs(a-(b-c)) for a,b,c in zip(s,v,e)))
            fields(result['population_statistics'][pop]['joint'],joint,'independent covariance/identity','joint_fields')
            require(abs(sv-variance_sum)<=1e-8+1e-10*abs(sv),'independent variance identity failed')
            populations[pop]=independent
        for comparison,saved in result['mean_component_differences'].items():
            a,b=comparison.split('_minus_')
            values={k:populations[a][k]['mean']-populations[b][k]['mean'] for k in ['mahalanobis_cpu','background_distance','rmd']}
            values['MD_minus_background']=values['mahalanobis_cpu']-values['background_distance']
            values['identity_error']=abs(values['rmd']-values['MD_minus_background'])
            fields(saved,values,'independent signed mean differences','mean_difference_fields')
        codes=load(directory/'transitions.npz')
        frozen=load(GEOMETRY/f'seed{seed}/evaluation/decisions.npz')
        for key,actual in dict(id_labels=id_data['labels'],id_predictions=id_data['logits'].argmax(1),id_correct=correct).items():
            require(np.array_equal(codes[key],actual),'ID labels/predictions/correctness changed')
            counts['original_label_arrays']+=1
        denom_counts=Counter(zip(id_data['labels'].tolist(),correct.astype(int).tolist()))
        require(codes['per_class_ID_denominators'].shape==(100,2),'denominator axes changed')
        for cls in range(100):
            for truth in range(2):
                require(codes['per_class_ID_denominators'][cls,truth]==denom_counts[(cls,truth)],'class correctness denominator changed')
                counts['per_class_denominators']+=1
        require(codes['per_class_ID_counts'].shape==(1,11,4,100,4,2),'count tensor axes changed')
        for pop,key in [('test_id','id')]+[(d,d) for d in DATASETS]:
            require(codes[f'{pop}_codes'].shape==(1,11,4,frozen[f'{key}_accepted'].shape[-1]),'transition tensor axes changed')
        records={(r['addition'],r['reference'],r['target']):r for r in result['transitions']}
        require(len(records)==len(result['transitions'])==44,'missing/duplicate transition comparison')
        for ai,addition in enumerate(NEW):
            for ri,reference in enumerate(REFERENCES):
                for ti,target in enumerate(TARGETS):
                    record=records[(addition,reference,target)]
                    for pop,key in [('test_id','id')]+[(d,d) for d in DATASETS]:
                        aa=frozen[f'{key}_accepted'][ALL.index(addition),ti].tolist()
                        rr=frozen[f'{key}_accepted'][ALL.index(reference),ti].tolist()
                        expected=[0 if a and r else 1 if a else 2 if r else 3 for a,r in zip(aa,rr)]
                        actual=codes[f'{pop}_codes'][ai,ri,ti]
                        require(actual.dtype==np.uint8 and np.array_equal(actual,expected),'transition row codes mismatch')
                        counts['transition_code_arrays']+=1
                        histogram=Counter(expected)
                        values=dict(n=len(aa),addition_acceptance=sum(aa)/len(aa),reference_acceptance=sum(rr)/len(rr),
                                    acceptance_change=(histogram[1]-histogram[2])/len(aa))
                        saved=record['populations'][pop]
                        fields(saved,values,'decision transition endpoint','transition_fields')
                        fields(saved['state_counts'],{s:histogram[i] for i,s in enumerate(STATES)},'state counts','transition_fields')
                        fields(saved['state_fractions'],{s:histogram[i]/len(aa) for i,s in enumerate(STATES)},'state fractions','transition_fields')
                        if pop=='test_id':
                            for name,flag in [('correct',True),('incorrect',False)]:
                                subset=[x for x,c in zip(expected,correct.tolist()) if c==flag]
                                c=Counter(subset);total=len(subset)
                                fields(saved[name],dict(n=total),'original correctness subgroup','transition_fields')
                                fields(saved[name]['state_counts'],{s:c[i] for i,s in enumerate(STATES)},'correctness state counts','transition_fields')
                                fields(saved[name]['state_fractions'],{s:c[i]/total if total else None for i,s in enumerate(STATES)},'correctness fractions','transition_fields')
                            class_counts=Counter(zip(id_data['labels'].tolist(),expected,correct.astype(int).tolist()))
                            saved_class=codes['per_class_ID_counts'][ai,ri,ti]
                            require(saved_class.shape==(100,4,2) and saved_class.dtype==np.int64,'per-class axes/dtype mismatch')
                            for cls in range(100):
                                for state in range(4):
                                    for truth in range(2):
                                        require(saved_class[cls,state,truth]==class_counts[(cls,state,truth)],'independent per-class contingency mismatch')
                                        counts['per_class_cells']+=1
        print('seed',seed,'independent component/moment/transition audit passed',flush=True)

    audit={'state':'passed','audited_at':now(),'output_name':attempt,'checks':counts,
           'original_input_hash_checks':len(manifest['original_artifacts']),'output_artifacts':hashes,
           'inputs_sha256':digest(manifest_path),'independent_source_sha256':digest(__file__),
           'boundary':'posthoc descriptive components and fixed original masks; all seeds/populations/comparisons verified'}
    write(OUT/'audit.json',audit)
    print(__import__('json').dumps({k:v for k,v in audit.items() if k!='output_artifacts'}),flush=True)


if __name__=='__main__':
    main()
