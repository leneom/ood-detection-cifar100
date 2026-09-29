"""One CPU controller for all fixed saved-draw dependence endpoints."""
import argparse,os,re,shutil,time
from pathlib import Path
import numpy as np
from geometry_io import ROOT,read,write,digest,save_arrays,require,now
from dependence_io import RUN,guard,saved_vectors
from dependence_core import moments,decomposition,json_finite

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='analysis');args=parser.parse_args()
    require(re.fullmatch(r'analysis(?:-attempt-\d{3})?',args.output_name),'invalid analysis name')
    import msvcrt
    with (RUN/'controller.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            pre=guard();out=RUN/args.output_name;require(not out.exists(),'retained analysis exists; use a fresh attempt name')
            source_files=[Path(__file__),ROOT/'tools/dependence_io.py',ROOT/'tools/dependence_core.py',ROOT/'tools/geometry_io.py',ROOT/'tests/test_dependence.py']
            manifest_path=RUN/f'{args.output_name}-inputs.json';require(not manifest_path.exists(),'source manifest already exists')
            snap=RUN/f'{args.output_name}-source';snap.mkdir(exist_ok=False)
            for path in source_files:shutil.copy2(path,snap/path.name)
            manifest={'state':'source_frozen_before_covariance_calculation','prepared_at':now(),'predeclaration_sha256':digest(RUN/'predeclaration.json'),
                      'original_artifacts':pre['original_artifacts'],'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in source_files},
                      'output_name':args.output_name,'ddof':1,'covariance_audit_rtol':1e-9,'covariance_audit_atol':1e-12,
                      'identity_audit_rtol':1e-9,'identity_audit_atol':1e-12,'correlation_audit_rtol':1e-9,'correlation_audit_atol':1e-12}
            write(manifest_path,manifest);out.mkdir(exist_ok=False)
            write(RUN/'status.json',{'state':'running','output_name':args.output_name,'pid':os.getpid(),'stage':'source-bound CPU saved-bootstrap covariance','completed_endpoints':0,'updated_at':now()})
            endpoints,natural,bootstrap_summary=saved_vectors();methods=pre['methods'];pairs=bootstrap_summary['pairs']
            index=[];record_count=0;start=time.monotonic()
            for ei,(key,raw) in enumerate(endpoints):
                require(raw.shape==(1000,3,15),'endpoint draw shape changed')
                vector=raw.reshape(1000,45);m=moments(vector)
                names=[f'seed{s}/{method}' for s in range(3) for method in methods]
                record={'endpoint':key,'variables':names,'finite_counts':m['finite_counts'].tolist(),
                        'means':json_finite(m['means']),'marginal_variances':json_finite(np.diag(m['covariance'])),
                        'correlation_undefined_count':int((~m['correlation_defined']).sum()),'methods':{},'pairs':{}}
                pair_cov=np.full((120,3,3),np.nan);pair_corr=np.full_like(pair_cov,np.nan)
                pair_count=np.empty((120,3),dtype=np.int64);pair_defined=np.empty((120,3),dtype=bool)
                for idx in range(120):
                    if idx<15:kind='methods';name=methods[idx];component=raw[:,:,idx]
                    else:
                        kind='pairs';p=pairs[idx-15];name=p['key'];component=raw[:,:,methods.index(p['later'])]-raw[:,:,methods.index(p['earlier'])]
                    partial,d=decomposition(component)
                    if d['defined']:require(abs(d['signed_identity_residual'])<=1e-12+1e-9*max(abs(d['observed_mean_variance']),abs(d['reconstructed_variance'])),'three-model variance identity failed')
                    pair_cov[idx]=partial['covariance'];pair_corr[idx]=partial['correlation'];pair_count[idx]=partial['finite_counts'];pair_defined[idx]=partial['correlation_defined'].diagonal()
                    record[kind][name]={**d,'finite_counts':partial['finite_counts'].tolist(),'covariance':json_finite(partial['covariance']),
                                        'correlation':json_finite(partial['correlation']),'correlation_defined':partial['correlation_defined'].tolist()}
                    record_count+=1
                save_arrays(out/f'endpoint-{ei:03d}.npz',values=vector,finite_mask=m['finite_mask'],finite_counts=m['finite_counts'],means=m['means'],
                            covariance=m['covariance'],correlation=m['correlation'],correlation_defined=m['correlation_defined'],
                            model_or_pair_covariance=pair_cov,model_or_pair_correlation=pair_corr,model_or_pair_finite_counts=pair_count,
                            model_or_pair_correlation_diagonal_defined=pair_defined)
                write(out/f'endpoint-{ei:03d}.json',record);index.append({'index':ei,'endpoint':key,'npz':f'endpoint-{ei:03d}.npz','json':f'endpoint-{ei:03d}.json'})
                if (ei+1)%10==0 or ei+1==66:
                    write(RUN/'status.json',{'state':'running','output_name':args.output_name,'pid':os.getpid(),'stage':'source-bound CPU saved-bootstrap covariance','completed_endpoints':ei+1,'expected_endpoints':66,'updated_at':now()})
                    print({'completed_endpoints':ei+1,'elapsed_seconds':round(time.monotonic()-start,2)},flush=True)
            require(record_count==66*120 and len(natural)==360,'method/pair scope incomplete')
            natural_values=np.stack([v for _,v in natural]);natural_means=np.full((360,4),np.nan);natural_cov=np.full((360,4,4),np.nan);natural_corr=np.full_like(natural_cov,np.nan)
            natural_count=np.empty((360,4),dtype=np.int64);natural_corr_defined=np.empty((360,4,4),dtype=bool);natural_records=[]
            by_key={k:v for k,v in endpoints}
            for ni,(key,component) in enumerate(natural):
                prefix,metric,seedpart,method=key.split('/');seed=int(seedpart.replace('seed',''));mi=methods.index(method)
                macrokey=f'ranking/natural_macro/{metric}' if prefix=='ranking' else f'operating/{metric}/natural_ood_acceptance_macro'
                require(np.allclose(component.mean(1),by_key[macrokey][:,seed,mi],rtol=0,atol=1e-12,equal_nan=True),'four-natural original macro changed')
                m,d=decomposition(component)
                if d['defined']:require(abs(d['signed_identity_residual'])<=1e-12+1e-9*max(abs(d['observed_mean_variance']),abs(d['reconstructed_variance'])),'four-natural variance identity failed')
                natural_means[ni]=m['means'];natural_cov[ni]=m['covariance'];natural_corr[ni]=m['correlation'];natural_count[ni]=m['finite_counts'];natural_corr_defined[ni]=m['correlation_defined']
                natural_records.append({'index':ni,'key':key,'natural_datasets':['cifar10','mnist','svhn','textures'],**d,'finite_counts':m['finite_counts'].tolist(),
                                        'covariance':json_finite(m['covariance']),'correlation':json_finite(m['correlation']),
                                        'correlation_defined':m['correlation_defined'].tolist()})
            save_arrays(out/'natural-components.npz',values=natural_values,means=natural_means,covariance=natural_cov,correlation=natural_corr,
                        finite_counts=natural_count,correlation_defined=natural_corr_defined)
            write(out/'natural-components.json',{'state':'raw_saved_pending_audit','records':natural_records,'natural_order':['cifar10','mnist','svhn','textures']})
            write(out/'endpoint-index.json',{'state':'raw_saved_pending_audit','ordering':'seed-major then method-minor, all 1000 original replicate rows','methods':methods,'pairs':pairs,'endpoints':index,'natural_component_count':360})
            for n,h in manifest['original_artifacts'].items():require(digest(ROOT/n)==h,'frozen source changed during covariance '+n)
            for n,h in manifest['source_hashes'].items():require(digest(ROOT/n)==h,'producer source changed during covariance '+n)
            hashes={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sorted(out.glob('*')) if p.is_file()}
            write(out/'raw-manifest.json',{'state':'raw_complete_pending_independent_audit','completed_at':now(),'analysis_inputs_sha256':digest(manifest_path),
                                           'source_hashes':manifest['source_hashes'],'outputs':hashes,'endpoints':66,'method_or_pair_decompositions':record_count,'natural_decompositions':360,
                                           'boundary':'Conditional covariance of saved image draws only; zero-variance correlations undefined, no pairwise deletion or training-seed resampling'})
            write(RUN/'status.json',{'state':'raw_complete_pending_independent_audit','output_name':args.output_name,'completed_endpoints':66,'method_or_pair_decompositions':record_count,
                                     'natural_decompositions':360,'raw_manifest_sha256':digest(out/'raw-manifest.json'),'updated_at':now()})
            print({'state':'raw_complete_pending_independent_audit','endpoints':66,'method_or_pair_decompositions':record_count,'natural_decompositions':360},flush=True)
        except Exception as e:
            write(RUN/'status.json',{'state':'failed','output_name':args.output_name,'pid':os.getpid(),'error':repr(e),'updated_at':now()});raise
        finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':main()
