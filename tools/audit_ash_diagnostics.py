"""Independent frozen ASH support/scale/head identity and Python moments/contingency reconstruction."""
from collections import Counter
import math
import statistics
from pathlib import Path
import numpy as np
from geometry_io import ROOT, digest, read, write, load, require, now, METHODS, ADDITIONS, DATASETS, FOLDERS

RUN = ROOT / 'runs/ash-diagnostics'
GEOMETRY = ROOT / 'runs/ash-phase5'
OUT = ROOT / 'report/ash-diagnostics'
POPULATIONS = ['validation_id','test_id'] + DATASETS
NEW=['ash_p_energy','ash_b_energy','ash_s_energy']
REFERENCES=METHODS+ADDITIONS+['rmd','mahalanobis_cpu']
TARGETS=[.8,.9,.95,.99]
SCALARS=['original_sum','retained_sum','retained_fraction','dropped_fraction','ratio','exp_ratio','binary_fill','cutoff','cutoff_equal_count','cutoff_greater_count','cutoff_selected_equal_count']
COMPONENTS=SCALARS+['affine_energy']+NEW+[m+'_minus_affine' for m in NEW]
CLASS_STATES=['both_correct','shaped_only_correct','original_only_correct','neither_correct']
STATES = ['both_accept','ash_only_accept','reference_only_accept','neither_accept']
QUANTILES = [.01,.05,.25,.5,.75,.95,.99]
ALL=REFERENCES+NEW


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
                  sampled_indices=0,transition_code_arrays=0,transition_fields=0,per_class_cells=0,original_label_arrays=0,per_class_denominators=0,classification_code_arrays=0,classification_cells=0,classification_fields=0,independent_head_arrays=0)
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
        fit=read(GEOMETRY/f'seed{seed}/preflight.json')
        bank=load(ROOT/f'runs/geometry-phase3/seed{seed}/training-bank.npz');weight,bias=bank['weight'].astype(float),bank['bias'].astype(float)
        require(result['checkpoint_sha256']==fit['checkpoint_sha256'] and result['preflight_json_sha256']==digest(GEOMETRY/f'seed{seed}/preflight.json'),'fit mismatch')
        require(set(result['population_statistics'])==set(POPULATIONS),'population summary omitted')
        require(set(result['mean_component_differences'])=={f'{d}_minus_test_id' for d in ['validation_id']+DATASETS},'signed difference scope omitted')
        populations={}
        sampling_rng=np.random.default_rng(7859+seed)
        for ordinal,pop in enumerate(POPULATIONS):
            raw=load(directory/f'{pop}_components.npz')
            key='id' if pop=='test_id' else 'validation' if pop=='validation_id' else pop
            middle=load(GEOMETRY/f'seed{seed}/evaluation/{key}_intermediates.npz')
            part='id' if pop=='test_id' else 'val' if pop=='validation_id' else 'ood'
            dataset='cifar10' if pop in ['test_id','validation_id'] else pop
            saved=load(GEOMETRY/f'seed{seed}/evaluation/{dataset}_scores.npz')
            n=len(raw['ratio'])
            fsource=load(original/'features.npz')[key] if pop in ['test_id','validation_id'] else load(ROOT/f'runs/cifar100-seed{seed}'/FOLDERS[pop]/f'{pop}_features.npz')['features']
            f=fsource.astype(float)
            # Independent manual full-row sums and einsum head; never imports producer cores.
            sums=np.asarray([math.fsum(row.tolist()) for row in f]);kept=np.asarray([math.fsum(row.tolist()) for row in middle['selected_values']])
            expected={k:middle[k] for k in SCALARS if k not in ['retained_fraction','dropped_fraction']}
            expected['retained_fraction']=kept/sums;expected['dropped_fraction']=1-expected['retained_fraction']
            require(np.allclose(sums,raw['original_sum'],atol=1e-8,rtol=1e-10) and np.allclose(kept,raw['retained_sum'],atol=1e-8,rtol=1e-10),'independent activation sums differ')
            for m in ['affine_energy']+NEW:expected[m]=saved[m+'_'+part]
            for m in NEW:expected[m+'_minus_affine']=expected[m]-expected['affine_energy']
            for k in COMPONENTS:
                require(raw[k].dtype==expected[k].dtype and raw[k].shape==(n,) and np.isfinite(raw[k]).all() and np.allclose(raw[k],expected[k],atol=1e-8,rtol=1e-10),'frozen diagnostic component changed '+k)
                if k not in ['retained_fraction','dropped_fraction']:require(np.array_equal(raw[k],expected[k]),'copied frozen vector changed '+k)
                counts['component_arrays']+=1
            require(np.allclose(raw['retained_fraction']*raw['ratio'],1,atol=1e-8,rtol=1e-10),'fraction ratio identity failed')
            indices=middle['selected_indices'];values=middle['selected_values'];p=np.zeros_like(f);np.put_along_axis(p,indices,values,axis=1)
            binary=np.zeros_like(f);np.put_along_axis(binary,indices,np.broadcast_to((sums/179)[:,None],values.shape),axis=1)
            scale=np.asarray([math.exp(a/b) for a,b in zip(sums,kept)])
            from scipy.special import logsumexp
            for m,shaped in zip(NEW,[p,binary,p*scale[:,None]]):
                z=np.einsum('ni,ci->nc',shaped,weight,optimize=False)+bias
                require(np.allclose(z,middle[m+'_logits'],atol=1e-8,rtol=1e-10),'independent frozen shaped head mismatch')
                require(np.array_equal(z.argmax(1),middle[m+'_logits'].argmax(1)),'independent shaped argmax changed')
                require(np.allclose(-logsumexp(z,axis=1),raw[m],atol=1e-8,rtol=1e-10),'independent Energy audit mismatch')
                counts['independent_head_arrays']+=1
            zp,zs=middle['ash_p_energy_logits'],middle['ash_s_energy_logits'];cc=raw['exp_ratio'][:,None]
            predicted=cc*zp+(1-cc)*bias
            require(np.allclose(zs,predicted,atol=1e-8,rtol=1e-10),'bias-aware scaled logits identity failed')
            require(n=={'validation_id':5000,'test_id':10000,'cifar10':10000,'mnist':10000,'svhn':26032,'textures':1880,'noise':10000}[pop],'population count changed')
            require(np.array_equal(raw['original_rows'],np.arange(n,dtype=np.int64)),'original row order changed')
            if pop in ['test_id','validation_id']:
                labels=id_data['labels'] if pop=='test_id' else split['validation_labels']
                predictions=id_data['logits'].argmax(1) if pop=='test_id' else load(original/'validation_logits.npz')['logits'].argmax(1)
                require(np.array_equal(raw['labels'],labels) and np.array_equal(raw['original_predictions'],predictions),'original ID labels/predictions changed')
                counts['original_label_arrays']+=2
            expected=np.sort(sampling_rng.choice(n,size=min(1000,n),replace=False))
            require(np.array_equal(raw['sample_rows'],expected),'fixed sample rows changed')
            counts['sampled_indices']+=1
            independent={}
            for key in COMPONENTS:
                stats,q=vector_summary(raw[key])
                fields(result['population_statistics'][pop][key],stats,'independent population moments','moment_fields')
                fields(result['population_statistics'][pop][key]['quantiles'],q,'sorted linear quantile','quantile_fields')
                independent[key]=stats
            q=raw['retained_fraction'].tolist();qm=statistics.fmean(q);qv=statistics.pvariance(q)
            for m in NEW:
                delta=raw[m+'_minus_affine'].tolist();dm=statistics.fmean(delta);dv=statistics.pvariance(delta)
                cov=math.fsum((x-qm)*(y-dm) for x,y in zip(q,delta))/n
                joint=dict(n=n,fraction_variance=qv,delta_energy_variance=dv,covariance_fraction_delta=cov,pearson_fraction_delta=cov/math.sqrt(qv*dv) if qv>0 and dv>0 else None)
                fields(result['population_statistics'][pop]['joint'][m],joint,'independent covariance/Pearson','joint_fields')
            identities={'fraction_ratio_max_error':float(np.abs(raw['retained_fraction']*raw['ratio']-1).max()),'sum_reconstruction_max_error':float(np.abs(raw['original_sum']-raw['retained_sum']/raw['retained_fraction']).max()),'bias_aware_logits_max_error':float(np.abs(zs-predicted).max())}
            fields(result['population_statistics'][pop]['identities'],identities,'independent exact identities','joint_fields')
            populations[pop]=independent
            if pop in ['test_id','validation_id']:
                classraw=load(directory/f'{pop}_classification.npz')
                oc=predictions==labels
                for k,actual in [('labels',labels),('original_predictions',predictions),('original_correct',oc)]:require(np.array_equal(classraw[k],actual),'original classification input changed')
                denom=Counter(zip(labels.tolist(),oc.astype(int).tolist()))
                require(classraw['per_class_original_denominators'].shape==(100,2),'classification denominator axes changed')
                for cls in range(100):
                    for truth in range(2):require(classraw['per_class_original_denominators'][cls,truth]==denom[(cls,truth)],'classification denominator changed');counts['per_class_denominators']+=1
                require(classraw['classification_codes'].shape==(3,n) and classraw['per_class_correctness_counts'].shape==(3,100,4),'classification axes changed')
                for mi,m in enumerate(NEW):
                    pp=middle[m+'_logits'].argmax(1);flags=pp==labels
                    require(np.array_equal(classraw['shaped_predictions'][mi],pp),'shaped predictions changed')
                    expected_codes=[0 if a and b else 1 if a else 2 if b else 3 for a,b in zip(flags.tolist(),oc.tolist())]
                    require(classraw['classification_codes'].dtype==np.uint8 and np.array_equal(classraw['classification_codes'][mi],expected_codes),'classification code mismatch');counts['classification_code_arrays']+=1
                    h=Counter(expected_codes);description={'n':n,'original_accuracy':float(oc.mean()),'shaped_accuracy':float(flags.mean()),'accuracy_change':(h[1]-h[2])/n,'argmax_changes':int((pp!=predictions).sum())}
                    savedclass=result['classification_statistics'][pop][m]
                    fields(savedclass,description,'classification endpoint','classification_fields')
                    fields(savedclass['state_counts'],{name:h[i] for i,name in enumerate(CLASS_STATES)},'classification states','classification_fields')
                    fields(savedclass['state_fractions'],{name:h[i]/n for i,name in enumerate(CLASS_STATES)},'classification fractions','classification_fields')
                    perclass=Counter(zip(labels.tolist(),expected_codes))
                    for cls in range(100):
                        for state in range(4):require(classraw['per_class_correctness_counts'][mi,cls,state]==perclass[(cls,state)],'classification class cell changed');counts['classification_cells']+=1
                dis=(classraw['shaped_predictions'][0]!=classraw['shaped_predictions'][2]).astype(np.uint8)
                require(np.array_equal(classraw['P_S_disagreement'],dis),'P/S disagreement vector changed')
                fields(result['classification_statistics'][pop]['P_S_argmax'],{'n':n,'disagreement_count':int(dis.sum()),'disagreement_fraction':float(dis.mean())},'P/S disagreement','classification_fields')
        for comparison,saved_difference in result['mean_component_differences'].items():
            a,b=comparison.split('_minus_')
            values={k:populations[a][k]['mean']-populations[b][k]['mean'] for k in COMPONENTS}
            fields(saved_difference,values,'independent signed mean differences','mean_difference_fields')
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
        require(codes['per_class_ID_counts'].shape==(3,12,4,100,4,2),'count tensor axes changed')
        for pop,key in [('test_id','id')]+[(d,d) for d in DATASETS]:
            require(codes[f'{pop}_codes'].shape==(3,12,4,frozen[f'{key}_accepted'].shape[-1]),'transition tensor axes changed')
        records={(r['addition'],r['reference'],r['target']):r for r in result['transitions']}
        require(len(records)==len(result['transitions'])==144,'missing/duplicate transition comparison')
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
                            ar=sum(a and not c for a,c in zip(aa,correct.tolist()))/sum(aa) if sum(aa) else None
                            rrisk=sum(r and not c for r,c in zip(rr,correct.tolist()))/sum(rr) if sum(rr) else None
                            fields(saved,dict(addition_retained_error_risk=ar,reference_retained_error_risk=rrisk,retained_error_risk_change=ar-rrisk if ar is not None and rrisk is not None else None),'original-classifier risk endpoints','transition_fields')
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
