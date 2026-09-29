"""Independent full-training-ID CPU torch shaping and manual mathematical checks."""
import math
from pathlib import Path
import numpy as np
import torch
from geometry_io import ROOT,read,write,load,digest,require,now
RUN=ROOT/'runs/ash-phase5'
METHODS=['ash_p_energy','ash_b_energy','ash_s_energy']


def main():
    require(not torch.cuda.is_available(),'CUDA must be hidden before CPU preflight audit')
    torch.set_num_threads(4)
    require(read(RUN/'preflight-status.json')['state']=='complete','preflight incomplete')
    manifest=read(RUN/'preflight-inputs.json')
    for name,h in manifest['original_artifacts'].items():require(digest(ROOT/name)==h,'original input changed')
    for name,h in manifest['source_hashes'].items():require(digest(ROOT/name)==digest(RUN/'preflight-source'/Path(name).name)==h,'executed source changed')
    require(read(RUN/'predeclaration.json')['declared_at']<manifest['prepared_at'],'preflight predates declaration')
    records=[]
    for seed in [0,1,2]:
        out=RUN/f'seed{seed}';meta=read(out/'preflight.json');saved=load(out/'preflight.npz')
        require(meta['seed']==seed and meta['training_rows']==45000 and meta['percentile']==65 and meta['k']==179 and meta['inputs_sha256']==digest(RUN/'preflight-inputs.json') and meta['raw_sha256']==digest(out/'preflight.npz'),'preflight settings/raw changed')
        bank=load(ROOT/f'runs/geometry-phase3/seed{seed}/training-bank.npz')
        original=load(ROOT/f'runs/cifar100-seed{seed}/split_indices.npz')
        require(np.array_equal(saved['indices'],bank['indices']) and np.array_equal(saved['labels'],bank['labels']) and np.array_equal(saved['indices'],original['train']) and not np.intersect1d(saved['indices'],original['val']).size,'train/validation isolation failed')
        f=torch.from_numpy(bank['features'].astype(float));w=torch.from_numpy(bank['weight'].astype(float));b=torch.from_numpy(bank['bias'].astype(float))
        require(tuple(f.shape)==(45000,512) and bool(torch.isfinite(f).all()) and bool((f>=0).all()),'training constraint failed')
        k=512-round(512*65/100)
        indices=torch.argsort(f,dim=1,descending=True,stable=True)[:,:k]
        values=f.gather(1,indices);s1=f.sum(1);s2=values.sum(1)
        require(bool((s1>0).all()) and bool((s2>0).all()),'positive sums failed')
        require(np.array_equal(indices.numpy(),saved['selected_indices']) and np.array_equal(values.numpy(),saved['selected_values']),'stable selected support/value mismatch')
        # Independent scalar sorting and summation on fixed first 256 training rows, including exact tie policy.
        for row in range(256):
            a=bank['features'][row].tolist();idx=sorted(range(512),key=lambda j:(-a[j],j))[:k]
            require(idx==saved['selected_indices'][row].tolist(),'manual sort/tie mismatch')
            require(math.isclose(math.fsum(a),saved['original_sum'][row],abs_tol=1e-8,rel_tol=1e-10),'manual original sum failed')
            require(math.isclose(math.fsum(a[j] for j in idx),saved['retained_sum'][row],abs_tol=1e-8,rel_tol=1e-10),'manual retained sum failed')
        ratio=s1/s2;scale=torch.exp(ratio);fill=s1/k
        for name,a in [('original_sum',s1),('retained_sum',s2),('ratio',ratio),('exp_ratio',scale),('binary_fill',fill)]:
            require(np.allclose(saved[name],a.numpy(),atol=1e-8,rtol=1e-10),'scalar shaping mismatch: '+name)
        cutoff=values[:,-1];eq=(f==cutoff[:,None]).sum(1);gt=(f>cutoff[:,None]).sum(1)
        for name,a in [('cutoff',cutoff),('cutoff_equal_count',eq),('cutoff_greater_count',gt),('cutoff_selected_equal_count',k-gt)]:require(np.array_equal(saved[name],a.numpy()),'cutoff tie record mismatch')
        p=torch.zeros_like(f).scatter(1,indices,values);binary=torch.zeros_like(f).scatter(1,indices,fill[:,None].expand_as(values))
        require(torch.allclose(binary.sum(1),s1,atol=1e-8,rtol=1e-10),'ASH-B sum not preserved')
        maximum=0.
        for method,shaped in zip(METHODS,[p,binary,p*scale[:,None]]):
            logits=torch.nn.functional.linear(shaped,w,b);energy=-torch.logsumexp(logits,dim=1)
            require(bool(torch.isfinite(logits).all()) and np.allclose(saved[method+'_logits'],logits.numpy(),atol=1e-8,rtol=1e-10) and np.allclose(saved[method],energy.numpy(),atol=1e-8,rtol=1e-10),'independent CPU torch shaped output failed')
            maximum=max(maximum,float(np.max(np.abs(saved[method+'_logits']-logits.numpy()))))
        affine=torch.nn.functional.linear(f,w,b)
        require(np.allclose(saved['affine_logits'],affine.numpy(),atol=1e-8,rtol=1e-10),'original CPU float64 head identity failed')
        head=read(ROOT/'runs/geometry-phase3/head-audit.json')['per_seed'][seed]
        require(meta['training_bank_sha256']==digest(ROOT/f'runs/geometry-phase3/seed{seed}/training-bank.npz')==head['bank_sha256'] and meta['checkpoint_sha256']==head['checkpoint_sha256'],'native head/bank provenance failed')
        records.append({'seed':seed,'training_rows':45000,'support_coordinates_checked':45000*k,'manual_sorted_rows':256,'max_shaped_logit_error':maximum,'preflight_json_sha256':digest(out/'preflight.json'),'raw_sha256':digest(out/'preflight.npz')})
        print('seed',seed,'independent full training-ID ASH preflight passed',flush=True)
        del bank,saved,f,w,b,p,binary,affine,logits,shaped
    audit={'state':'passed','audited_at':now(),'per_seed':records,'training_rows_checked':135000,'manual_sorted_rows':768,'methods':METHODS,'original_input_hash_checks':len(manifest['original_artifacts']),'inputs_sha256':digest(RUN/'preflight-inputs.json'),'independent_source_sha256':digest(__file__),'CUDA_is_available':False,'boundary':'independent CPU torch and manual sort/math; training ID only; no OOD scores or fitted statistics'}
    write(RUN/'preflight-audit.json',audit)
    write(RUN/'status.json',{'state':'ready_for_evaluation','completed_preflight_seeds':[0,1,2],'preflight_audit_sha256':digest(RUN/'preflight-audit.json'),'ASH_OOD_scoring_started':False,'updated_at':now()})
    print(__import__('json').dumps(audit),flush=True)


if __name__=='__main__':main()
