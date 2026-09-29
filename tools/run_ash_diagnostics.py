"""CPU-only posthoc summaries and mask transitions from immutable phase-4 inputs."""
import argparse
import os
from pathlib import Path
import re
import shutil
import numpy as np
from geometry_io import ROOT, digest, read, write, load, save_arrays, require, now, METHODS, ADDITIONS, DATASETS, FOLDERS
from ash_diagnostics_core import describe, mean_difference, transitions, STATE_NAMES, QUANTILES, assemble, classification, CLASS_STATES

RUN = ROOT / 'runs/ash-diagnostics'
GEOMETRY = ROOT / 'runs/ash-phase5'
POPULATIONS = ['validation_id','test_id'] + DATASETS
NEW = ['ash_p_energy','ash_b_energy','ash_s_energy']
REFERENCES = METHODS + ADDITIONS + ['rmd','mahalanobis_cpu']
TARGETS = [.8,.9,.95,.99]
ALL = REFERENCES + NEW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='analysis')
    args = parser.parse_args()
    require(re.fullmatch(r'analysis(?:-attempt-\d{3})?', args.output_name), 'invalid attempt name')
    import msvcrt
    with (RUN / 'analysis.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            require(read(GEOMETRY / 'status.json')['state'] == 'complete', 'geometry producer incomplete')
            prereg = read(RUN / 'predeclaration.json')
            require(prereg['expected_seeds'] == [0,1,2] and prereg['population_order'] == POPULATIONS and
                    prereg['additions'] == NEW and prereg['references'] == REFERENCES and prereg['targets'] == TARGETS and
                    prereg['quantiles'] == QUANTILES and prereg['transition_states']==STATE_NAMES and prereg['scatter_rng_base']==7859 and prereg['classification_states']==CLASS_STATES and prereg['seed_comparison_target_settings']==432, 'diagnostic settings changed')
            inputs = dict(prereg['original_artifacts'])
            inputs['runs/ash-diagnostics/predeclaration.json'] = digest(RUN / 'predeclaration.json')
            for name, expected in inputs.items():
                require(digest(ROOT / name) == expected, 'frozen diagnostic input changed: '+name)
            files = [Path(__file__), ROOT / 'tools/ash_diagnostics_core.py', ROOT / 'tools/geometry_io.py']
            sources = {str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files}
            manifest_path = RUN / f'{args.output_name}-inputs.json'
            if manifest_path.exists():
                manifest = read(manifest_path)
                require(manifest['original_artifacts'] == inputs and manifest['source_hashes'] == sources, 'incompatible diagnostic reuse')
            else:
                snapshot = RUN / f'{args.output_name}-source'
                snapshot.mkdir(exist_ok=True)
                for path in files:
                    shutil.copy2(path, snapshot / path.name)
                manifest = {'state':'inputs_frozen_before_diagnostics','prepared_at':now(),'original_artifacts':inputs,
                            'source_hashes':sources,'output_name':args.output_name,'numpy':np.__version__,
                            'seed_order':[0,1,2],'population_order':POPULATIONS,'new_methods':NEW,'references':REFERENCES,
                            'targets':TARGETS,'transition_states':STATE_NAMES,'boundary':'posthoc descriptive outputs; no fitting/threshold selection'}
                write(manifest_path,manifest)
            split = load(ROOT / 'runs/calibration-phase2/calibration_split.npz')
            for seed in [0,1,2]:
                write(RUN / 'status.json',{'state':'running','stage':'CPU components and transitions','pid':os.getpid(),
                                          'completed_seeds':list(range(seed)),'output_name':args.output_name,'updated_at':now()})
                output = RUN / f'seed{seed}' / args.output_name
                if (output / 'results.json').exists():
                    previous = read(output / 'results.json')
                    require(previous['inputs_sha256'] == digest(manifest_path), 'diagnostic input manifest changed')
                    for name, expected in previous['output_file_sha256'].items():
                        require(digest(output / name) == expected, 'completed diagnostic output changed')
                    continue
                require(not output.exists() or not any(output.iterdir()), 'partial attempt retained; use fresh output name')
                output.mkdir(parents=True,exist_ok=True)
                source = GEOMETRY / f'seed{seed}'
                fit=read(source/'preflight.json')
                bank=load(ROOT/f'runs/geometry-phase3/seed{seed}/training-bank.npz')
                class_records={}
                canonical=ROOT/f'runs/cifar100-seed{seed}/eval-final'
                original_id=load(canonical/'id_logits.npz')
                correct=original_id['logits'].argmax(1)==original_id['labels']
                sampling_rng=np.random.default_rng(7859+seed)
                descriptions={}
                for ordinal,population in enumerate(POPULATIONS):
                    key='id' if population=='test_id' else 'validation' if population=='validation_id' else population
                    middle=load(source/f'evaluation/{key}_intermediates.npz')
                    part='id' if population=='test_id' else 'val' if population=='validation_id' else 'ood'
                    dataset='cifar10' if population in ['test_id','validation_id'] else population
                    saved=load(source/f'evaluation/{dataset}_scores.npz')
                    arrays,logit_error=assemble(middle,saved,part,bank['bias'].astype(float))
                    n=len(arrays['ratio'])
                    sample=np.sort(sampling_rng.choice(n,size=min(1000,n),replace=False))
                    descriptions[population]=describe(arrays,logit_error)
                    extras={'sample_rows':sample,'original_rows':np.arange(n,dtype=np.int64)}
                    if population in ['test_id','validation_id']:
                        extras['labels']=middle['original_labels']
                        extras['original_predictions']=middle['original_predictions']
                        labels=middle['original_labels'];original=middle['original_predictions']
                        cc={'labels':labels,'original_predictions':original,'original_correct':original==labels}
                        preds=np.stack([middle[m+'_logits'].argmax(1) for m in NEW]);cc['shaped_predictions']=preds
                        per_class_codes=np.zeros((3,100,4),dtype=np.int64);all_codes=[];class_records[population]={}
                        for mi,m in enumerate(NEW):
                            description,codes=classification(preds[mi],original,labels)
                            class_records[population][m]=description;all_codes.append(codes)
                            np.add.at(per_class_codes[mi],(labels,codes),1)
                        cc['classification_codes']=np.stack(all_codes);cc['per_class_correctness_counts']=per_class_codes
                        cc['per_class_original_denominators']=np.stack([np.bincount(labels[original!=labels],minlength=100),np.bincount(labels[original==labels],minlength=100)],axis=1)
                        cc['P_S_disagreement']=(preds[0]!=preds[2]).astype(np.uint8)
                        class_records[population]['P_S_argmax']={'n':n,'disagreement_count':int(cc['P_S_disagreement'].sum()),'disagreement_fraction':float(cc['P_S_disagreement'].mean())}
                        save_arrays(output/f'{population}_classification.npz',**cc)

                    save_arrays(output/f'{population}_components.npz',**arrays,**extras)
                differences={f'{pop}_minus_test_id':mean_difference(descriptions[pop],descriptions['test_id']) for pop in ['validation_id']+DATASETS}
                frozen_masks = load(source / 'evaluation/decisions.npz')
                require(np.array_equal(frozen_masks['id_correct'],correct),'original correctness changed')
                transition_arrays = {'id_labels':original_id['labels'], 'id_predictions':original_id['logits'].argmax(1), 'id_correct':correct}
                for pop,key in [('test_id','id')]+[(d,d) for d in DATASETS]:
                    transition_arrays[f'{pop}_codes'] = np.empty((3,12,4,frozen_masks[f'{key}_accepted'].shape[-1]),dtype=np.uint8)
                per_class = np.zeros((3,12,4,100,4,2),dtype=np.int64)
                records = []
                for ai, addition in enumerate(NEW):
                    for ri, reference in enumerate(REFERENCES):
                        for ti,target in enumerate(TARGETS):
                            populations = {}
                            for pop,key in [('test_id','id')]+[(d,d) for d in DATASETS]:
                                masks = frozen_masks[f'{key}_accepted']
                                record,codes = transitions(masks[ALL.index(addition),ti], masks[ALL.index(reference),ti], correct if pop == 'test_id' else None)
                                populations[pop] = record
                                transition_arrays[f'{pop}_codes'][ai,ri,ti] = codes
                                if pop == 'test_id':
                                    np.add.at(per_class[ai,ri,ti],(original_id['labels'],codes,correct.astype(int)),1)
                            records.append({'addition':addition,'reference':reference,'target':target,'populations':populations})
                transition_arrays['per_class_ID_counts'] = per_class
                transition_arrays['per_class_ID_denominators'] = np.stack([np.bincount(original_id['labels'][~correct],minlength=100),np.bincount(original_id['labels'][correct],minlength=100)],axis=1)
                save_arrays(output / 'transitions.npz',**transition_arrays)
                result = {'state':'complete','seed':seed,'computed_at':now(),'inputs_sha256':digest(manifest_path),
                          'preflight_json_sha256':digest(source / 'preflight.json'),'checkpoint_sha256':fit['checkpoint_sha256'],
                          'population_order':POPULATIONS,'population_statistics':descriptions,'mean_component_differences':differences,
                          'new_methods':NEW,'references':REFERENCES,'targets':TARGETS,'transition_states':STATE_NAMES,
                          'transition_axes':['addition','reference','target','row'],
                          'per_class_count_axes':['addition','reference','target','ID_class','transition_state','correctness_0_incorrect_1_correct'],
                          'classification_statistics':class_records,'classification_states':CLASS_STATES,'transitions':records,'output_file_sha256':{p.name:digest(p) for p in output.glob('*.npz')},
                          'boundary':'posthoc descriptive diagnostics; fixed methods/masks; no fitting, tuning or causal inference'}
                write(output / 'results.json',result)
                print(__import__('json').dumps({'seed':seed,'state':'raw_diagnostics_saved','populations':len(descriptions),
                                              'comparisons':len(records),'existing_settings_unchanged':True}),flush=True)

            for name, expected in inputs.items():
                require(digest(ROOT/name) == expected,'input changed during diagnostics')
            for name, expected in sources.items():
                require(digest(ROOT/name) == expected,'source changed during diagnostics')
            write(RUN/'status.json',{'state':'raw_complete_pending_independent_audit','completed_seeds':[0,1,2],
                                    'output_name':args.output_name,'original_artifacts_unchanged':True,'completed_at':now()})
        except Exception as error:
            write(RUN/'status.json',{'state':'failed','stage':'CPU diagnostic producer','pid':os.getpid(),'error':repr(error),'updated_at':now()})
            raise
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__ == '__main__':
    main()
