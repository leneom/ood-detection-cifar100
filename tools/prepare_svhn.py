"""Prepare verified SVHN test arrays using a CPU Python with SciPy.

No packages are installed into the active GPU training environment. Uses the
official torchvision-listed Stanford URL and MD5, and preserves the full test
split. Completed download ranges can be reused after an interruption.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import urllib.request

import numpy as np
import scipy
from scipy.io import loadmat

URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
MD5 = 'eb5a983be6a315427106f1b164d9cef3'
SIZE = 64275384
PART_SIZE = 2 * 1024 * 1024


def digest(path, algorithm='sha256'):
    hasher = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def fetch(root, workers=4):
    archive = root / 'test_32x32.mat'
    if archive.exists() and digest(archive, 'md5') == MD5:
        return archive
    parts = root / 'download_parts' / 'svhn-test'
    parts.mkdir(parents=True, exist_ok=True)
    count = (SIZE + PART_SIZE - 1) // PART_SIZE
    def part(index):
        begin, end = index * PART_SIZE, min(SIZE, (index + 1) * PART_SIZE) - 1
        target = parts / f'{index:03d}.part'
        if target.exists() and target.stat().st_size == end - begin + 1:
            return
        temporary = target.with_suffix('.tmp')
        for attempt in range(3):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={begin}-{end}'})
                with urllib.request.urlopen(request, timeout=30) as response, temporary.open('wb') as stream:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {begin}-{end}/{SIZE}':
                        raise RuntimeError('server did not honor requested byte range')
                    while block := response.read(64 * 1024):
                        stream.write(block)
                if temporary.stat().st_size != end - begin + 1:
                    raise RuntimeError('download range has incorrect length')
                temporary.replace(target)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, future in enumerate(as_completed([pool.submit(part, index) for index in range(count)]), 1):
            future.result()
            print(f'SVHN ranges completed: {completed}/{count}', flush=True)
    combined = root / 'svhn-test-download.tmp'
    with combined.open('wb') as output:
        for index in range(count):
            with (parts / f'{index:03d}.part').open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    output.write(block)
    if digest(combined, 'md5') != MD5:
        raise RuntimeError('official SVHN MD5 mismatch; data not accepted')
    combined.replace(archive)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('workers must lie between 1 and 8')
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    archive = fetch(root, args.workers)
    source = loadmat(archive)
    images = np.ascontiguousarray(source['X'].transpose(3, 0, 1, 2))
    labels = np.asarray(source['y']).reshape(-1).astype(np.int64)
    labels[labels == 10] = 0
    if images.dtype != np.uint8 or images.shape != (26032, 32, 32, 3) or labels.shape != (26032,) or not np.isin(labels, range(10)).all():
        raise ValueError('SVHN arrays differ from the official test schema')
    target = root / 'svhn-test.npz'
    temporary = root / 'svhn-test.tmp.npz'
    np.savez_compressed(temporary, images=images, labels=labels)
    temporary.replace(target)
    manifest = {
        'schema_version': 1, 'dataset': 'svhn', 'split': 'test', 'count': len(images),
        'shape': list(images.shape), 'dtype': str(images.dtype), 'source_url': URL,
        'source_md5': MD5, 'source_sha256': digest(archive), 'cache_sha256': digest(target),
        'label_rule': 'original digit 10 remapped to 0; all test samples retained',
        'transform': 'X[H,W,C,N] -> images[N,H,W,C]; no resizing or normalization',
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'prepared_at': datetime.now(timezone.utc).isoformat(),
    }
    (root / 'svhn-test.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Official SVHN test verified and converted: {len(images)} images; {target}', flush=True)


if __name__ == '__main__':
    main()
