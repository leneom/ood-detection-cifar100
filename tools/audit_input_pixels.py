"""Independently reconstruct real Textures-phase input hashes using CPU NumPy.

CIFAR pickles are accepted only after official file MD5 checks. No torch,
classifier inference, JPEG redecoding or detector fitting is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Verified against installed torchvision 0.28.0 CIFAR100 integrity metadata.
CIFAR_MD5 = {'train': '16019d7e3df5f24257cddd939b257f8d',
             'test': 'f0ef6b0ae62326f3e7ffdfab6717acfc'}


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_pixels(record, pixels):
    require(pixels.dtype == np.uint8, 'reference pixels must be uint8')
    expected = {'sha256': hashlib.sha256(pixels.tobytes(order='C')).hexdigest(),
                'shape': list(pixels.shape), 'dtype': 'uint8', 'layout': 'NHWC'}
    require(record == expected, 'executed input fingerprint differs from independent reference pixels')
    return expected


def load_reference(root):
    arrays, provenance = {}, {}
    for name, md5 in CIFAR_MD5.items():
        source = root / 'cifar-100-python' / name
        require(file_hash(source, 'md5') == md5, f'official CIFAR {name} MD5 mismatch')
        with source.open('rb') as stream:
            payload = pickle.load(stream, encoding='latin1')
        flat = payload['data']
        count = 50000 if name == 'train' else 10000
        require(flat.dtype == np.uint8 and flat.shape == (count, 3072), 'CIFAR reference shape mismatch')
        arrays[name] = flat.reshape(count, 3, 32, 32).transpose(0, 2, 3, 1).copy()
        provenance[name] = {'path': str(source.resolve()), 'md5': md5, 'sha256': file_hash(source)}
    manifest_path = root / 'textures-test1.manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    audit_path = root.parent / 'runs' / 'textures-conversion-audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    target = root / 'textures-test1.npz'
    cache_hash = file_hash(target)
    require(audit['state'] == 'passed' and audit['all_pixel_equality_checks'] == 1880,
            'full DTD conversion audit missing')
    require(file_hash(manifest_path) == audit['manifest_sha256'], 'DTD manifest no longer matches pixel audit')
    require(cache_hash == audit['cache_sha256'] == manifest['cache_sha256'], 'DTD cached input hash mismatch')
    with np.load(target, allow_pickle=False) as cache:
        arrays['textures'] = cache['images']
    require(arrays['textures'].shape == (1880, 32, 32, 3) and arrays['textures'].dtype == np.uint8,
            'DTD pixel reference schema mismatch')
    provenance['textures'] = {'path': str(target.resolve()), 'cache_sha256': cache_hash,
                              'conversion_audit_sha256': file_hash(audit_path)}
    return arrays, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', nargs='+', type=Path, required=True)
    parser.add_argument('--expected-seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(ROOT), 'audit output must stay inside ood_detection')
    require(not args.output.exists(), 'existing input audit retained; choose a new output')
    require(len(args.expected_seeds) == len(set(args.expected_seeds)), 'duplicate expected seeds')
    references, source_provenance, checks = {}, {}, []
    seeds = []
    for evaluation in args.evaluations:
        result = json.loads((evaluation / 'results.json').read_text(encoding='utf-8'))
        config = result['training_config']
        require(config['dataset'] == 'cifar100' and config['epochs'] == 100, 'requires real full-schedule CIFAR classifier')
        status = json.loads((evaluation.parent / 'status.json').read_text(encoding='utf-8'))
        require(status['state'] == 'complete' and status['epoch'] == 100, 'classifier training incomplete')
        require(set(result['datasets']) == {'textures'}, 'input audit is for the separately declared Textures phase')
        root = Path(config['data_root']).resolve()
        require(root.is_relative_to(ROOT), 'reference data must stay inside ood_detection')
        if root not in references:
            references[root], source_provenance[str(root)] = load_reference(root)
        arrays = references[root]
        with np.load(evaluation.parent / 'split_indices.npz', allow_pickle=False) as split:
            train, val = split['train'], split['val']
        require(train.shape == (45000,) and val.shape == (5000,), 'ID split counts differ')
        require(np.array_equal(np.sort(np.concatenate((train, val))), np.arange(50000)), 'ID split overlaps or is incomplete')
        pixels = result['input_pixels']
        verified = {
            'id_train': verify_pixels(pixels['id_train'], arrays['train'][train]),
            'id_validation': verify_pixels(pixels['id_validation'], arrays['train'][val]),
            'id_test': verify_pixels(pixels['id_test'], arrays['test']),
            'textures': verify_pixels(pixels['ood']['textures'], arrays['textures']),
        }
        seeds.append(config['seed'])
        checks.append({'seed': config['seed'], 'evaluation': str(evaluation.resolve()),
                       'results_sha256': file_hash(evaluation / 'results.json'), 'populations': verified})
    require(len(seeds) == len(set(seeds)) and sorted(seeds) == sorted(args.expected_seeds),
            'all expected seeds must be present exactly once')
    record = {'state': 'passed', 'seeds': sorted(seeds), 'population_pixel_hash_checks': 4 * len(seeds),
              'per_seed': checks, 'sources': source_provenance, 'tool_sha256': file_hash(__file__),
              'python': platform.python_version(), 'numpy': np.__version__,
              'audited_at': datetime.now(timezone.utc).isoformat(),
              'boundary': 'verifies captured executed raw inputs; does not retroactively add hashes to earlier phases'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'state': 'passed', 'seeds': record['seeds'],
                      'population_pixel_hash_checks': record['population_pixel_hash_checks'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
