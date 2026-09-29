"""Sequential frozen-classifier training-ID feature export; never trains or fits detectors."""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from geometry_io import ROOT, RUN, digest, read, write, load, save_arrays, require, now, frozen_inputs


def other_gpu_python():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [int(row[0]) for row in csv.reader(text.splitlines())
            if len(row) == 2 and 'python' in row[1].lower() and int(row[0]) != os.getpid()]


def main():
    import msvcrt
    with (RUN / 'bank-export.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            require(not other_gpu_python(), 'another Python GPU workload is active')
            inputs = frozen_inputs()
            files = [Path(__file__), ROOT / 'tools/geometry_io.py'] + list((ROOT / 'src/ood').glob('*.py'))
            sources = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files}
            manifest_path = RUN / 'bank-export-inputs.json'
            if manifest_path.exists():
                manifest = read(manifest_path)
                require(manifest['original_artifacts'] == inputs and manifest['source_hashes'] == sources, 'bank resume source/input mismatch')
            else:
                snapshot = RUN / 'bank-export-source'
                snapshot.mkdir(exist_ok=True)
                for path in files:
                    shutil.copy2(path, snapshot / path.name)
                manifest = {'state': 'inputs_frozen_before_training_bank_export', 'prepared_at': now(),
                            'original_artifacts': inputs, 'source_hashes': sources,
                            'population': '45000 original ID training split only', 'batch_size': 256,
                            'amp': False, 'augmentation': False, 'fit_performed': False}
                write(manifest_path, manifest)
            write(RUN / 'bank-export-status.json', {'state': 'running', 'pid': os.getpid(), 'started_at': now()})
            sys.path.insert(0, str(ROOT / 'src'))
            import torch
            from torch import nn
            from ood.data import load_cifar, index_hash
            from ood.features import extract
            from ood.models import build_model
            from ood.runtime import seed_all, environment
            from ood.evaluate import image_fingerprint
            require(torch.cuda.is_available(), 'verified CUDA environment unavailable')
            seed_all(0)
            images, labels = load_cifar(ROOT / 'data', 'cifar100', True)
            require(digest(ROOT / 'data/cifar-100-python/train', 'md5') == '16019d7e3df5f24257cddd939b257f8d', 'official train MD5 failed')
            for seed in [0, 1, 2]:
                directory = RUN / f'seed{seed}'
                directory.mkdir(exist_ok=True)
                meta_path, bank_path = directory / 'bank.json', directory / 'training-bank.npz'
                if meta_path.exists():
                    previous = read(meta_path)
                    require(previous['bank_sha256'] == digest(bank_path) and previous['export_inputs_sha256'] == digest(manifest_path),
                            'completed bank changed')
                    continue
                require(not bank_path.exists(), 'orphaned bank exists; retain and inspect before reuse')
                require(not other_gpu_python(), 'another GPU Python appeared before seed export')
                selected = read(ROOT / f'runs/cifar100-seed{seed}/eval-final/results.json')
                checkpoint_path = Path(selected['checkpoint'])
                require(digest(checkpoint_path) == selected['checkpoint_sha256'], 'selected checkpoint changed')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                require(checkpoint['epoch'] == selected['checkpoint_epoch'], 'selected epoch mismatch')
                indices = load(ROOT / f'runs/cifar100-seed{seed}/split_indices.npz')
                train, validation = indices['train'], indices['val']
                require(len(train) == 45000 and len(validation) == 5000 and not np.intersect1d(train, validation).size, 'split leakage')
                require(index_hash(train) == checkpoint['split']['train_indices_sha256'], 'checkpoint training split mismatch')
                require(np.array_equal(np.bincount(labels[train].numpy(), minlength=100), np.full(100, 450)), 'training class balance mismatch')
                training_pixels = image_fingerprint(images[train])
                original_pixels = read(ROOT / f'runs/cifar100-seed{seed}/eval-extension-textures/results.json')['input_pixels']['id_train']
                require(training_pixels == original_pixels, 'training input pixels differ from audited original')
                model = build_model('resnet18', 100).to('cuda').to(memory_format=torch.channels_last)
                model.load_state_dict(checkpoint['model'])
                # Match the old first evaluation batch shape; only its fixed first 64 rows check identity.
                check_logits, check_features = extract(model, images[validation[:256]], 256, 'cuda')
                check_logits, check_features = check_logits[:64].numpy(), check_features[:64].numpy()
                old_logits = load(ROOT / f'runs/cifar100-seed{seed}/eval-final/validation_logits.npz')['logits'][:64]
                old_features = load(ROOT / f'runs/cifar100-seed{seed}/eval-final/features.npz')['validation'][:64]
                save_arrays(directory / 'extraction-identity.npz', logits=check_logits, features=check_features,
                            expected_logits=old_logits, expected_features=old_features, validation_indices=validation[:64])
                require(np.allclose(check_logits, old_logits, atol=1e-6, rtol=1e-6), 'validation logit identity check failed')
                require(np.allclose(check_features, old_features, atol=1e-6, rtol=1e-6), 'validation feature identity check failed')
                started = time.perf_counter()
                logits, features = extract(model, images[train], 256, 'cuda')
                head = [layer for layer in model.modules() if isinstance(layer, nn.Linear)][-1]
                save_arrays(bank_path, features=features.numpy(), logits=logits.numpy(), labels=labels[train].numpy(),
                            indices=train, weight=head.weight.detach().cpu().numpy(), bias=head.bias.detach().cpu().numpy())
                require(features.shape == (45000, 512) and logits.shape == (45000, 100), 'bank shapes differ from protocol')
                write(meta_path, {'state': 'complete', 'seed': seed, 'exported_at': now(), 'seconds': time.perf_counter() - started,
                                 'export_inputs_sha256': digest(manifest_path), 'bank_sha256': digest(bank_path),
                                 'identity_sha256': digest(directory / 'extraction-identity.npz'),
                                 'checkpoint_sha256': selected['checkpoint_sha256'], 'checkpoint_epoch': selected['checkpoint_epoch'],
                                 'training_indices_sha256': index_hash(train), 'training_count': 45000,
                                 'training_pixels': training_pixels, 'environment': environment(),
                                 'max_identity_feature_error': float(np.max(np.abs(check_features - old_features))),
                                 'max_identity_logit_error': float(np.max(np.abs(check_logits - old_logits))),
                                 'forward_identity_rows': 64, 'forward_batch_shape': 256, 'amp': False,
                                 'augmentation': False, 'feature_dtype': str(features.numpy().dtype),
                                 'boundary': 'training ID export only; validation identity rows never enter detector fitting'})
                print(f'completed training bank seed {seed}: 45000 x 512', flush=True)
                del model, checkpoint, logits, features
                torch.cuda.empty_cache()
                write(RUN / 'bank-export-status.json', {'state': 'running', 'pid': os.getpid(), 'completed_seeds': list(range(seed + 1)), 'updated_at': now()})
            for name, expected in sources.items():
                require(digest(ROOT / name) == expected, 'export source changed while running')
            for name, expected in inputs.items():
                require(digest(ROOT / name) == expected, 'original artifact changed during export')
            write(RUN / 'bank-export-status.json', {'state': 'complete', 'pid': os.getpid(), 'completed_seeds': [0, 1, 2],
                                                   'original_artifacts_unchanged': True, 'completed_at': now()})
        except Exception as error:
            write(RUN / 'bank-export-status.json', {'state': 'failed', 'pid': os.getpid(), 'error': repr(error), 'updated_at': now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
