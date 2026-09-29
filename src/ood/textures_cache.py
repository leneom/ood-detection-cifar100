"""Staged audited DTD cache loader; integrate only after existing queues exit.

NumPy-only: reads fixed RGB cache pixels without JPEG decoding, SciPy, torch,
normalization, model fitting or OOD-dependent selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

DTD_TEST_MD5 = 'fff73e5086ae6bdbea199a49dfb8a4c1'
TEXTURES_TRANSFORM = {
    'decode': 'Pillow Image.open -> convert RGB; no EXIF transpose',
    'resize': 'shorter edge 32; longer edge floor(32 * longer / shorter)',
    'resampling': 'Pillow BILINEAR (PIL input antialiasing)',
    'crop': 'center 32x32; each offset round((resized_dimension - 32) / 2)',
    'output': 'contiguous NHWC uint8; no normalization or augmentation',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_textures_cache(root, audit_path=None):
    root = Path(root)
    target, manifest_path = root / 'textures-test1.npz', root / 'textures-test1.manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {'schema_version': 1, 'dataset': 'textures', 'release': 'r1.0.1',
                'split': 'test', 'partition': 1, 'count': 1880, 'source_md5': DTD_TEST_MD5,
                'shape': [1880, 32, 32, 3], 'dtype': 'uint8', 'transform': TEXTURES_TRANSFORM}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Textures manifest differs from the predeclared official test1 protocol')
    audit_path = Path(audit_path) if audit_path is not None else root.parent / 'runs' / 'textures-conversion-audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    required_audit = {'state': 'passed', 'dataset': 'textures', 'split': 'test', 'partition': 1,
                      'all_pixel_equality_checks': 1880, 'class_label_and_order_checks': 1880,
                      'original_split_isolation': True, 'source_md5': DTD_TEST_MD5}
    if any(audit.get(key) != value for key, value in required_audit.items()):
        raise ValueError('Textures requires the complete independent pixel/isolation audit')
    if sha256(manifest_path) != audit.get('manifest_sha256') or manifest.get('source_sha256') != audit.get('source_sha256'):
        raise ValueError('Textures manifest/source does not match the audited provenance')
    cache_hash = sha256(target)
    if cache_hash != manifest.get('cache_sha256') or cache_hash != audit.get('cache_sha256'):
        raise ValueError('Textures cache hash mismatch')
    if audit.get('canonical_decoder', {}).get('pillow') != manifest.get('pillow'):
        raise ValueError('Textures canonical decode environment differs from its audit')
    with np.load(target, allow_pickle=False) as source:
        images, labels = source['images'], source['labels']
        filenames, classes, dimensions = source['filenames'], source['class_names'], source['original_hw']
    if images.shape != (1880, 32, 32, 3) or images.dtype != np.uint8:
        raise ValueError('Textures image shape/dtype mismatch')
    if hashlib.sha256(images.tobytes()).hexdigest() != manifest.get('images_sha256'):
        raise ValueError('Textures pixel-byte hash mismatch')
    if labels.shape != (1880,) or labels.dtype != np.int64 or not np.isin(labels, np.arange(47)).all():
        raise ValueError('Textures label shape/dtype/range mismatch')
    if hashlib.sha256(labels.tobytes()).hexdigest() != manifest.get('labels_sha256'):
        raise ValueError('Textures label-byte hash mismatch')
    if filenames.shape != (1880,) or filenames.dtype.kind != 'U' or classes.shape != (47,) or classes.dtype.kind != 'U':
        raise ValueError('Textures filename/class schema mismatch')
    names, class_names = filenames.tolist(), classes.tolist()
    if len(set(names)) != 1880 or names != manifest.get('selected_filenames'):
        raise ValueError('Textures sample order differs from the audited test list')
    if class_names != sorted(set(class_names)) or class_names != manifest.get('class_names'):
        raise ValueError('Textures category mapping mismatch')
    mapping = {name: index for index, name in enumerate(class_names)}
    expected_labels = []
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) != 2 or any(part in ('.', '..') for part in parts) or name != '/'.join(parts):
            raise ValueError('Textures filename path is invalid')
        if parts[0] not in mapping or not parts[1].endswith('.jpg'):
            raise ValueError('Textures filename category/JPEG mismatch')
        expected_labels.append(mapping[parts[0]])
    if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int64)):
        raise ValueError('Textures labels do not match official filename categories')
    if not np.array_equal(np.bincount(labels, minlength=47), np.full(47, 40)) or manifest.get('class_counts') != [40] * 47:
        raise ValueError('Textures category balance differs from full official test1')
    if dimensions.shape != (1880, 2) or dimensions.dtype != np.int64 or not (dimensions > 0).all():
        raise ValueError('Textures original dimension metadata mismatch')
    provenance = dict(manifest)
    provenance['independent_audit'] = {
        'path': str(audit_path.resolve()), 'sha256': sha256(audit_path),
        'all_pixel_equality_checks': 1880, 'canonical_decoder': audit['canonical_decoder'],
    }
    return np.ascontiguousarray(images), labels, provenance
