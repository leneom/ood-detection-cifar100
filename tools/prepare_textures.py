"""CPU-only official DTD partition-1 test preparation; no torch dependency.

The population/transform are fixed in docs/TEXTURES_EXTENSION.md. Archive
members are read directly, never extracted to paths supplied by the archive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import tarfile
import time
import urllib.request

import numpy as np
import PIL
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
URL = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
MD5 = 'fff73e5086ae6bdbea199a49dfb8a4c1'
SIZE = 625239812
PART_SIZE = 4 * 1024 * 1024
TRANSFORM = {
    'decode': 'Pillow Image.open -> convert RGB; no EXIF transpose',
    'resize': 'shorter edge 32; longer edge floor(32 * longer / shorter)',
    'resampling': 'Pillow BILINEAR (PIL input antialiasing)',
    'crop': 'center 32x32; each offset round((resized_dimension - 32) / 2)',
    'output': 'contiguous NHWC uint8; no normalization or augmentation',
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def status(path, state, **details):
    atomic_json(path, {'state': state, 'pid': os.getpid(),
                      'updated_at': datetime.now(timezone.utc).isoformat(), **details})


def fetch(root, state_path, workers):
    archive = root / 'dtd-r1.0.1.tar.gz'
    if archive.exists():
        if archive.stat().st_size != SIZE or digest(archive, 'md5') != MD5:
            raise ValueError('existing DTD archive failed integrity; retained without overwrite')
        return archive
    parts = root / 'download_parts' / 'dtd-r1.0.1'
    parts.mkdir(parents=True, exist_ok=True)
    count = (SIZE + PART_SIZE - 1) // PART_SIZE

    def part(index):
        begin, end = index * PART_SIZE, min(SIZE, (index + 1) * PART_SIZE) - 1
        target = parts / f'{index:03d}.part'
        if target.exists() and target.stat().st_size == end - begin + 1:
            return
        temporary = target.with_suffix('.tmp')
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={begin}-{end}'})
                with urllib.request.urlopen(request, timeout=30) as response, temporary.open('wb') as stream:
                    expected = f'bytes {begin}-{end}/{SIZE}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError('server did not honor exact DTD byte range')
                    while block := response.read(64 * 1024):
                        stream.write(block)
                if temporary.stat().st_size != end - begin + 1:
                    raise RuntimeError('incorrect DTD range length')
                temporary.replace(target)
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    status(state_path, 'downloading', total_ranges=count, completed_ranges=0)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(part, index) for index in range(count)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 10 == 0 or completed == count:
                status(state_path, 'downloading', total_ranges=count, completed_ranges=completed)
                print(f'DTD ranges completed: {completed}/{count}', flush=True)
    combined = root / 'dtd-download.tmp'
    status(state_path, 'verifying_archive')
    with combined.open('wb') as output:
        for index in range(count):
            with (parts / f'{index:03d}.part').open('rb') as stream:
                while block := stream.read(1024 * 1024):
                    output.write(block)
    if combined.stat().st_size != SIZE or digest(combined, 'md5') != MD5:
        raise RuntimeError('full official DTD MD5 mismatch; archive not accepted, ranges retained')
    combined.replace(archive)
    return archive


def parse_split(payload):
    paths = payload.decode('utf-8').splitlines()
    if len(paths) != 1880 or len(set(paths)) != 1880:
        raise ValueError('DTD split must contain exactly 1880 unique images')
    counts = {}
    for path in paths:
        components = PurePosixPath(path).parts
        if len(components) != 2 or any(part in ('.', '..') for part in components):
            raise ValueError('unexpected DTD image path')
        category, filename = components
        if path != f'{category}/{filename}' or not filename.endswith('.jpg'):
            raise ValueError('DTD split path must be canonical category/JPEG')
        counts[category] = counts.get(category, 0) + 1
    if len(counts) != 47 or set(counts.values()) != {40}:
        raise ValueError('DTD split must retain all 47 classes, 40 samples per class')
    return paths


def transform_image(image):
    image = image.convert('RGB')
    width, height = image.size
    if width <= height:
        size = (32, int(32 * height / width))
    else:
        size = (int(32 * width / height), 32)
    image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = round((size[0] - 32) / 2), round((size[1] - 32) / 2)
    return np.asarray(image.crop((left, top, left + 32, top + 32)), dtype=np.uint8).copy()


def verify_reuse(target, manifest_path, archive):
    if not target.exists() and not manifest_path.exists():
        return False
    if not target.exists() or not manifest_path.exists():
        raise ValueError('incomplete existing DTD output retained; inspect before re-running')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {'schema_version': 1, 'dataset': 'textures', 'split': 'test',
                'partition': 1, 'count': 1880, 'source_md5': MD5, 'transform': TRANSFORM}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('existing DTD manifest protocol mismatch; retained')
    if digest(archive) != manifest['source_sha256'] or digest(target) != manifest['cache_sha256']:
        raise ValueError('existing DTD source/cache SHA-256 mismatch; retained')
    with np.load(target, allow_pickle=False) as cache:
        images, labels = cache['images'], cache['labels']
        if images.shape != (1880, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError('existing DTD cache image schema mismatch')
        if labels.shape != (1880,) or labels.dtype != np.int64 or not np.array_equal(np.bincount(labels, minlength=47), np.full(47, 40)):
            raise ValueError('existing DTD cache label schema mismatch')
        if hashlib.sha256(images.tobytes()).hexdigest() != manifest['images_sha256']:
            raise ValueError('existing DTD pixel hash mismatch')
    return True


def convert(archive, target, manifest_path, state_path):
    status(state_path, 'converting')
    # Two sequential gzip passes avoid repeated backward decompression when
    # the official test-list order differs from the archive member order.
    lists, member_names = {}, set()
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            if member.name in member_names:
                raise ValueError('duplicate archive member paths')
            member_names.add(member.name)
            for split in ('train', 'val', 'test'):
                if member.name == f'dtd/labels/{split}1.txt':
                    if not member.isfile() or member.size > 1024 * 1024:
                        raise ValueError('unexpected DTD split-list member')
                    with source.extractfile(member) as stream:
                        lists[split] = stream.read()
    if set(lists) != {'train', 'val', 'test'}:
        raise ValueError('official partition-1 split lists missing')
    paths = {split: parse_split(payload) for split, payload in lists.items()}
    sets = {split: set(items) for split, items in paths.items()}
    if any(sets[a] & sets[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))):
        raise ValueError('DTD partition-1 train/validation/test overlap')
    if len(set.union(*sets.values())) != 5640:
        raise ValueError('DTD partition-1 combined count mismatch')
    classes = sorted({path.split('/')[0] for path in paths['test']})
    class_to_idx = {name: index for index, name in enumerate(classes)}
    selected = {f'dtd/images/{path}': index for index, path in enumerate(paths['test'])}
    images = np.empty((1880, 32, 32, 3), dtype=np.uint8)
    labels = np.asarray([class_to_idx[path.split('/')[0]] for path in paths['test']], dtype=np.int64)
    original_hw = np.empty((1880, 2), dtype=np.int64)
    jpeg_hashes = [None] * 1880
    completed = 0
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            if member.name not in selected:
                continue
            if not member.isfile() or member.size > 20 * 1024 * 1024:
                raise ValueError('unexpected non-regular or oversize DTD JPEG member')
            index = selected[member.name]
            with source.extractfile(member) as stream:
                payload = stream.read()
            jpeg_hashes[index] = hashlib.sha256(payload).hexdigest()
            with Image.open(io.BytesIO(payload)) as image:
                original_hw[index] = (image.height, image.width)
                images[index] = transform_image(image)
            completed += 1
            if completed % 200 == 0:
                print(f'DTD test images converted: {completed}/1880', flush=True)
    if completed != 1880 or any(value is None for value in jpeg_hashes):
        raise ValueError('official DTD test JPEG population incomplete')
    filenames = np.asarray(paths['test'])
    temporary = target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, images=images, labels=labels, filenames=filenames,
                        class_names=np.asarray(classes), original_hw=original_hw)
    temporary.replace(target)
    manifest = {
        'schema_version': 1, 'dataset': 'textures', 'release': 'r1.0.1',
        'split': 'test', 'partition': 1, 'count': 1880,
        'shape': list(images.shape), 'dtype': str(images.dtype),
        'class_names': classes, 'class_counts': np.bincount(labels, minlength=47).tolist(),
        'source_url': URL, 'source_bytes': SIZE, 'source_md5': MD5,
        'source_sha256': digest(archive), 'cache_sha256': digest(target),
        'images_sha256': hashlib.sha256(images.tobytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
        'split_list_sha256': {split: hashlib.sha256(payload).hexdigest() for split, payload in lists.items()},
        'selected_filenames': paths['test'], 'selected_jpeg_sha256': jpeg_hashes,
        'transform': TRANSFORM, 'labels_usage': 'provenance/annotations only; never fitting',
        'split_isolation': 'all three lists pairwise disjoint; 5640 unique combined images',
        'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
        'tool_sha256': digest(__file__), 'prepared_at': datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(manifest_path, manifest)
    if not verify_reuse(target, manifest_path, archive):
        raise AssertionError('new cache could not be independently reloaded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(PROJECT / 'data'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    if not root.is_relative_to(PROJECT) or not 1 <= args.workers <= 4:
        raise ValueError('data root must stay inside ood_detection; workers must lie in 1..4')
    root.mkdir(parents=True, exist_ok=True)
    # OS byte-range lock prevents duplicate prep; automatically released on exit.
    import msvcrt
    lock_path = root / 'dtd-preparation.lock'
    with lock_path.open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError('another DTD preparation process holds the lock') from error
        state_path = root / 'textures-preparation.status.json'
        try:
            archive = fetch(root, state_path, args.workers)
            target, manifest = root / 'textures-test1.npz', root / 'textures-test1.manifest.json'
            reused = verify_reuse(target, manifest, archive)
            if not reused:
                convert(archive, target, manifest, state_path)
            status(state_path, 'complete', count=1880, cache=str(target), reused=reused)
            print(f'DTD partition-1 test verified: 1880 images; {target}; reused={reused}', flush=True)
        except Exception as error:
            status(state_path, 'failed', error=repr(error))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    main()
