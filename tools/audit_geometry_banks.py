"""Pure NumPy/official-data audit before any geometry fit or OOD scoring."""
import hashlib
import pickle
import numpy as np
from geometry_io import ROOT, RUN, read, load, digest, write, require, now


def main():
    require(read(RUN / 'bank-export-status.json')['state'] == 'complete', 'bank exporter unfinished')
    manifest = read(RUN / 'bank-export-inputs.json')
    for name, expected in manifest['original_artifacts'].items():
        require(digest(ROOT / name) == expected, 'original export input changed')
    for name, expected in manifest['source_hashes'].items():
        require(digest(ROOT / name) == digest(RUN / 'bank-export-source' / name.split('/')[-1]) == expected, 'export source changed')
    heads = read(RUN / 'head-audit.json')
    require(heads['state'] == 'passed' and heads['source_sha256'] == digest(ROOT / 'tools/audit_geometry_heads.py'), 'head audit missing/stale')
    official_path = ROOT / 'data/cifar-100-python/train'
    require(digest(official_path, 'md5') == '16019d7e3df5f24257cddd939b257f8d', 'official training MD5 failed')
    with official_path.open('rb') as stream:
        official = pickle.load(stream, encoding='latin1')
    labels = np.asarray(official['fine_labels'], dtype=np.int64)
    images = official['data'].reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
    records = []
    for seed in [0, 1, 2]:
        directory = RUN / f'seed{seed}'
        bank, meta = load(directory / 'training-bank.npz'), read(directory / 'bank.json')
        require(meta['state'] == 'complete' and meta['seed'] == seed, 'bank incomplete')
        require(meta['bank_sha256'] == digest(directory / 'training-bank.npz') == heads['per_seed'][seed]['bank_sha256'], 'bank hash changed')
        require(meta['export_inputs_sha256'] == digest(RUN / 'bank-export-inputs.json'), 'bank export inputs changed')
        split = load(ROOT / f'runs/cifar100-seed{seed}/split_indices.npz')
        require(np.array_equal(bank['indices'], split['train']) and len(bank['indices']) == 45000, 'training bank indices mismatch')
        require(not np.intersect1d(bank['indices'], split['val']).size, 'ID-validation leakage')
        require(np.array_equal(bank['labels'], labels[bank['indices']]), 'training label/order mismatch')
        require(np.array_equal(np.bincount(bank['labels'], minlength=100), np.full(100, 450)), 'class balance mismatch')
        pixel_hash = hashlib.sha256(images[bank['indices']].copy().tobytes()).hexdigest()
        require(pixel_hash == meta['training_pixels']['sha256'], 'training pixels/order mismatch')
        require(bank['features'].shape == (45000, 512) and bank['logits'].shape == (45000, 100), 'bank dimensions mismatch')
        require(bank['weight'].shape == (100, 512) and bank['bias'].shape == (100,), 'head dimensions mismatch')
        require(all(np.isfinite(a).all() for a in bank.values()), 'nonfinite bank')
        require(bank['features'].dtype == bank['logits'].dtype == np.float32, 'native forward precision mismatch')
        reconstructed = bank['features'].astype(np.float64) @ bank['weight'].astype(np.float64).T + bank['bias'].astype(np.float64)
        require(np.allclose(reconstructed, bank['logits'], atol=2e-5, rtol=2e-6), 'training affine-logit reconstruction failed')
        require(meta['identity_sha256'] == digest(directory / 'extraction-identity.npz'), 'extraction identity changed')
        identity = load(directory / 'extraction-identity.npz')
        old_logits = load(ROOT / f'runs/cifar100-seed{seed}/eval-final/validation_logits.npz')['logits'][:64]
        old_features = load(ROOT / f'runs/cifar100-seed{seed}/eval-final/features.npz')['validation'][:64]
        require(np.array_equal(identity['expected_logits'], old_logits) and np.array_equal(identity['expected_features'], old_features), 'identity reference changed')
        require(np.array_equal(identity['validation_indices'], split['val'][:64]), 'identity rows changed')
        for name, expected in [('logits', old_logits), ('features', old_features)]:
            require(np.allclose(identity[name], expected, atol=1e-6, rtol=1e-6), 'extraction identity gate failed')
        records.append({'seed': seed, 'bank_sha256': meta['bank_sha256'], 'bank_metadata_sha256': digest(directory / 'bank.json'),
                        'checkpoint_sha256': meta['checkpoint_sha256'], 'training_count': 45000,
                        'training_pixel_sha256': pixel_hash, 'max_affine_logit_error': float(np.max(np.abs(reconstructed - bank['logits']))),
                        'affine_argmax_changes': int((reconstructed.argmax(1) != bank['logits'].argmax(1)).sum()),
                        'max_identity_feature_error': float(np.max(np.abs(identity['features'] - old_features))),
                        'max_identity_logit_error': float(np.max(np.abs(identity['logits'] - old_logits)))})
    result = {'state': 'passed', 'audited_at': now(), 'per_seed': records, 'training_rows_checked': 135000,
              'original_artifact_hash_checks': len(manifest['original_artifacts']), 'head_audit_sha256': digest(RUN / 'head-audit.json'),
              'export_inputs_sha256': digest(RUN / 'bank-export-inputs.json'), 'source_sha256': digest(__file__),
              'boundary': 'training banks only; no detector fitting or OOD score evaluation'}
    write(RUN / 'bank-audit.json', result)
    print(__import__('json').dumps(result), flush=True)


if __name__ == '__main__':
    main()
