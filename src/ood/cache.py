"""Read CPU-prepared arrays without adding SciPy to the training environment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SVHN_TEST_MD5 = 'eb5a983be6a315427106f1b164d9cef3'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_svhn_cache(root):
    root = Path(root)
    archive, manifest_path = root / 'svhn-test.npz', root / 'svhn-test.manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {'schema_version': 1, 'dataset': 'svhn', 'split': 'test', 'count': 26032,
                'source_md5': SVHN_TEST_MD5, 'dtype': 'uint8', 'shape': [26032, 32, 32, 3]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('SVHN cache manifest differs from the official test schema')
    if sha256(archive) != manifest.get('cache_sha256'):
        raise ValueError('SVHN converted cache hash mismatch')
    with np.load(archive, allow_pickle=False) as cache:
        images, labels = cache['images'], cache['labels']
    if images.dtype != np.uint8 or images.shape != tuple(expected['shape']):
        raise ValueError('SVHN cached images have incorrect shape or dtype')
    if labels.shape != (26032,) or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, range(10)).all():
        raise ValueError('SVHN cached labels are invalid')
    return images, labels, manifest
