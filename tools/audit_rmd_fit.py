"""Independent fsum means, chunked covariance, SciPy Cholesky and training-row checks."""
import math
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import cholesky,cho_solve,solve_triangular,eigvalsh
from geometry_io import ROOT,read,write,digest,load,require,now

RUN=ROOT/'runs/rmd-phase4'


def main():
    require(read(RUN/'fit-status.json')['state']=='complete','fit producer incomplete')
    manifest_path=RUN/'fit-inputs.json';manifest=read(manifest_path)
    for name,h in manifest['original_artifacts'].items():require(digest(ROOT/name)==h,'fit input changed')
    for name,h in manifest['source_hashes'].items():
        require(digest(ROOT/name)==digest(RUN/'fit-source'/Path(name).name)==h,'fit source changed')
    require(read(RUN/'predeclaration.json')['declared_at']<manifest['prepared_at'],'fit predates declaration')
    reports=[]
    for seed in [0,1,2]:
        directory=RUN/f'seed{seed}';fit,state=read(directory/'fit.json'),load(directory/'fit-state.npz')
        require(fit['seed']==seed and fit['state']=='complete' and fit['fit_inputs_sha256']==digest(manifest_path) and
                fit['fit_state_sha256']==digest(directory/'fit-state.npz') and fit['fitted_at']>manifest['prepared_at'],'fit provenance mismatch')
        require(fit['training_count']==45000 and fit['class_count']==100 and fit['feature_dimensions']==512 and
                fit['shrinkage']==.01 and fit['covariance_denominator']=='N' and fit['feature_normalization']=='none' and
                fit['background_subtraction_weight']==1. and fit['negative_distance_tolerance']==1e-8,'fixed fit settings changed')
        bank_path=ROOT/f'runs/geometry-phase3/seed{seed}/training-bank.npz'
        require(digest(bank_path)==fit['training_bank_sha256'],'training bank changed')
        with np.load(bank_path,allow_pickle=False) as bank:f,y,indices=bank['features'].astype(float),bank['labels'],bank['indices']
        split=load(ROOT/f'runs/cifar100-seed{seed}/split_indices.npz')
        require(np.array_equal(indices,split['train']) and not np.intersect1d(indices,split['val']).size,'training split overlap')
        require(np.array_equal(state['classes'],np.arange(100)) and np.array_equal(state['class_counts'],np.full(100,450)),'class population mismatch')
        means=[]
        for k in range(100):
            group=f[y==k]
            require(len(group)==450,'incorrect class count')
            means.append([math.fsum(col)/450 for col in group.T])
        means=np.array(means)
        background_mean=np.array([math.fsum(col)/45000 for col in f.T])
        require(np.allclose(means,state['class_means'],atol=1e-10,rtol=0) and
                np.allclose(background_mean,state['background_mean'],atol=1e-10,rtol=0),'independent means differ')
        covs={k:np.zeros((512,512)) for k in ['within','background']}
        for start in range(0,len(f),1024):
            chunk=f[start:start+1024]
            residual=chunk-means[y[start:start+1024]]
            bg=chunk-background_mean
            covs['within']+=np.einsum('ni,nj->ij',residual,residual,optimize=True)
            covs['background']+=np.einsum('ni,nj->ij',bg,bg,optimize=True)
        report={'seed':seed,'fit_json_sha256':digest(directory/'fit.json'),'fit_state_sha256':digest(directory/'fit-state.npz'),
                'training_rows':len(f),'means_max_error':float(np.max(np.abs(means-state['class_means']))),'covariances':{}}
        for name,raw in covs.items():
            raw/=len(f)
            scale=max(math.fsum(raw.diagonal())/512,1e-12)
            regularized=.99*raw+.01*scale*np.eye(512)
            require(np.allclose(raw,state[name+'_covariance_raw'],atol=1e-10,rtol=0) and
                    np.allclose(regularized,state[name+'_covariance'],atol=1e-10,rtol=0),'independent covariance/shrinkage mismatch')
            require(abs(scale-fit[name+'_trace_scale'])<1e-10,'training trace scale mismatch')
            # Use the archived covariance for an independent Cholesky inverse/form check.
            cov=state[name+'_covariance'];lower=cholesky(cov,lower=True)
            precision=cho_solve((lower,True),np.eye(512))
            eigen=eigvalsh(cov,driver='evr');raw_eigen=eigvalsh(state[name+'_covariance_raw'],driver='evr')
            require(eigen[0]>0 and np.allclose(eigen,state[name+'_eigenvalues'],atol=1e-10,rtol=1e-9) and
                    np.allclose(raw_eigen,state[name+'_raw_eigenvalues'],atol=1e-10,rtol=1e-9),'eigenvalues/SPD mismatch')
            require(np.allclose(precision,state[name+'_precision'],atol=1e-8,rtol=1e-9),'independent inverse mismatch')
            require(np.allclose(state[name+'_precision']@cov,np.eye(512),atol=1e-8,rtol=1e-8),'inverse identity mismatch')
            require(abs(float(eigen[-1]/eigen[0])-fit[name+'_condition_number'])<1e-7,'condition number mismatch')
            residual=f-state['class_means'][y] if name=='within' else f-state['background_mean']
            white=solve_triangular(lower,residual.T,lower=True).T
            q=np.einsum('ni,ni->n',white,white)
            direct=np.einsum('ni,ni->n',residual@state[name+'_precision'],residual)
            require(np.allclose(q,direct,atol=1e-8,rtol=1e-9),'training Cholesky/quadratic distance mismatch')
            report['covariances'][name]={'raw_covariance_max_error':float(np.max(np.abs(raw-state[name+'_covariance_raw']))),
                                        'precision_max_error':float(np.max(np.abs(precision-state[name+'_precision']))),
                                        'training_distance_max_error':float(np.max(np.abs(q-direct))),
                                        'regularized_min_eigenvalue':float(eigen[0]),'condition_number':fit[name+'_condition_number']}
        reports.append(report)
        print('seed',seed,'training-only RMD fit independently passed',flush=True)
    audit={'state':'passed','audited_at':now(),'per_seed':reports,'original_artifact_hash_checks':len(manifest['original_artifacts']),
           'training_distance_rows_checked':270000,'fit_inputs_sha256':digest(manifest_path),'numpy':np.__version__,'scipy':scipy.__version__,
           'source_sha256':digest(__file__),'boundary':'training ID only; no validation/test/OOD fit or scoring'}
    write(RUN/'fit-audit.json',audit)
    write(RUN/'status.json',{'state':'ready_for_evaluation','completed_fit_seeds':[0,1,2],'fit_audit_sha256':digest(RUN/'fit-audit.json'),
                            'RMD_OOD_scoring_started':False,'updated_at':now()})
    print(__import__('json').dumps(audit),flush=True)


if __name__=='__main__':main()
