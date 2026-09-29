"""CPU-only posthoc summaries and mask transitions from immutable phase-3 inputs."""
import argparse
import os
from pathlib import Path
import re
import shutil
import numpy as np
from geometry_io import ROOT, digest, read, write, load, save_arrays, require, now, METHODS, ADDITIONS, DATASETS, FOLDERS
from geometry_diagnostics_core import describe, mean_difference, transitions, STATE_NAMES, QUANTILES

RUN = ROOT / 'runs/geometry-diagnostics'
GEOMETRY = ROOT / 'runs/geometry-phase3'
POPULATIONS = ['training_id','validation_id','test_id'] + DATASETS
NEW = ADDITIONS[:3]
REFERENCES = ['affine_energy','energy','msp']
TARGETS = [.8,.9,.95,.99]
ALL = METHODS + ADDITIONS


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
                    prereg['quantiles'] == QUANTILES, 'diagnostic settings changed')
            inputs = dict(prereg['original_artifacts'])
            inputs['runs/geometry-diagnostics/predeclaration.json'] = digest(RUN / 'predeclaration.json')
            for name, expected in inputs.items():
                require(digest(ROOT / name) == expected, 'frozen diagnostic input changed: '+name)
            files = [Path(__file__), ROOT / 'tools/geometry_diagnostics_core.py', ROOT / 'tools/geometry_io.py']
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
                fit, state = read(source / 'fit.json'), load(source / 'fit-state.npz')
                canonical = ROOT / f'runs/cifar100-seed{seed}/eval-final'
                original_id = load(canonical / 'id_logits.npz')
                features = load(canonical / 'features.npz')
                correct = original_id['logits'].argmax(1) == original_id['labels']
                bank = load(source / 'training-bank.npz')
                descriptions = {}
                for ordinal, population in enumerate(POPULATIONS):
                    labels = None
                    if population == 'training_id':
                        f = bank['features'].astype(float)
                        affine = f @ bank['weight'].astype(float).T + bank['bias'].astype(float)
                        energy = -np.logaddexp.reduce(affine,axis=1)
                        residual = state['training_residual']
                        virtual = fit['alpha'] * residual
                        score = virtual + energy
                        labels = bank['labels']
                    else:
                        key = 'id' if population == 'test_id' else 'validation' if population == 'validation_id' else population
                        middle = load(source / f'evaluation/{key}_intermediates.npz')
                        if population in ['test_id','validation_id']:
                            part = 'id' if population == 'test_id' else 'val'
                            saved = load(source / 'evaluation/cifar10_scores.npz')
                            f = features['id' if population == 'test_id' else 'validation'].astype(float)
                            labels = original_id['labels'] if population == 'test_id' else split['validation_labels']
                        else:
                            part = 'ood'
                            saved = load(source / f'evaluation/{population}_scores.npz')
                            f = load(ROOT / f'runs/cifar100-seed{seed}' / FOLDERS[population] / f'{population}_features.npz')['features'].astype(float)
                        residual, virtual = middle['residual_norm'], middle['virtual_logit']
                        energy, score = saved[f'affine_energy_{part}'], saved[f'vim_{part}']
                        require(np.allclose(virtual,fit['alpha']*residual,atol=2e-9,rtol=0),'saved scale identity failed')
                    norm = np.linalg.norm(f,axis=1)
                    arrays = dict(residual=residual,virtual_logit=virtual,affine_energy=energy,vim=score,feature_norm=norm)
                    sample = np.sort(np.random.default_rng(4313+100*seed+ordinal).choice(len(f),size=min(1000,len(f)),replace=False))
                    descriptions[population] = describe(arrays)
                    extras = {'sample_rows':sample}
                    if labels is not None:
                        extras['labels'] = labels
                    save_arrays(output / f'{population}_components.npz',**arrays,**extras)
                differences = {f'{pop}_minus_test_id':mean_difference(descriptions[pop],descriptions['test_id']) for pop in DATASETS}
                differences.update({f'{pop}_minus_training_id':mean_difference(descriptions[pop],descriptions['training_id'])
                                    for pop in ['validation_id','test_id']})
                frozen_masks = load(source / 'evaluation/decisions.npz')
                require(np.array_equal(frozen_masks['id_correct'],correct),'original correctness changed')
                transition_arrays = {'id_labels':original_id['labels'], 'id_predictions':original_id['logits'].argmax(1), 'id_correct':correct}
                for pop,key in [('test_id','id')]+[(d,d) for d in DATASETS]:
                    transition_arrays[f'{pop}_codes'] = np.empty((3,3,4,frozen_masks[f'{key}_accepted'].shape[-1]),dtype=np.uint8)
                per_class = np.zeros((3,3,4,100,4,2),dtype=np.int64)
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
                save_arrays(output / 'transitions.npz',**transition_arrays)
                result = {'state':'complete','seed':seed,'computed_at':now(),'inputs_sha256':digest(manifest_path),
                          'fit_json_sha256':digest(source / 'fit.json'),'checkpoint_sha256':fit['checkpoint_sha256'],
                          'population_order':POPULATIONS,'population_statistics':descriptions,'mean_component_differences':differences,
                          'new_methods':NEW,'references':REFERENCES,'targets':TARGETS,'transition_states':STATE_NAMES,
                          'transition_axes':['addition','reference','target','row'],
                          'per_class_count_axes':['addition','reference','target','ID_class','transition_state','correctness_0_incorrect_1_correct'],
                          'transitions':records,'output_file_sha256':{p.name:digest(p) for p in output.glob('*.npz')},
                          'boundary':'posthoc descriptive diagnostics; fixed methods/masks; no fitting, tuning or causal inference'}
                write(output / 'results.json',result)
                print(__import__('json').dumps({'seed':seed,'state':'raw_diagnostics_saved','populations':len(descriptions),
                                              'comparisons':len(records),'existing_settings_unchanged':True}),flush=True)
                del bank,features
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
