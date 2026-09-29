"""Independently verify every cached DTD test pixel with torchvision on CPU.

Run with the existing torchvision Python and CUDA_VISIBLE_DEVICES=''. Does not
import the preparation implementation and never creates a model/GPU tensor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import struct
import tarfile

import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class CanonicalDecoder:
    def __init__(self, executable):
        self.executable = str(executable)
        self.worker_path = Path(__file__).with_name('jpeg_decoder_worker.py')

    def __enter__(self):
        self.worker = subprocess.Popen(
            [self.executable, '-u', str(self.worker_path)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        length = struct.unpack('<I', self._read(4))[0]
        self.metadata = json.loads(self._read(length))
        return self

    def _read(self, count):
        result = self.worker.stdout.read(count)
        if len(result) != count:
            raise EOFError('canonical CPU decoder returned an incomplete response')
        return result

    def decode(self, payload):
        self.worker.stdin.write(struct.pack('<I', len(payload)) + payload)
        self.worker.stdin.flush()
        height, width, length = struct.unpack('<III', self._read(12))
        require(length == height * width * 3 and length <= 48 * 1024 * 1024, 'invalid decoder response shape')
        return np.frombuffer(self._read(length), dtype=np.uint8).reshape(height, width, 3)

    def __exit__(self, exc_type, exc_value, traceback):
        self.worker.stdin.close()
        try:
            code = self.worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.worker.terminate()  # only the helper created by this audit
            code = self.worker.wait(timeout=10)
        error = self.worker.stderr.read().decode('utf-8', errors='replace')
        self.worker.stdout.close()
        self.worker.stderr.close()
        if exc_type is None:
            require(code == 0, 'canonical decoder failed: ' + error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder-python', required=True, help='CPU Pillow Python used to prepare the cache')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('runs/textures-conversion-audit.json'))
    args = parser.parse_args()
    root = args.data_root
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), 'CUDA was initialized before the CPU-only audit')
    archive = root / 'dtd-r1.0.1.tar.gz'
    target = root / 'textures-test1.npz'
    manifest_path = root / 'textures-test1.manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(file_hash(archive, 'md5') == 'fff73e5086ae6bdbea199a49dfb8a4c1', 'official archive MD5 mismatch')
    require(archive.stat().st_size == 625239812, 'official archive size mismatch')
    require(file_hash(archive) == manifest['source_sha256'], 'archive SHA-256 mismatch')
    require(file_hash(target) == manifest['cache_sha256'], 'cache SHA-256 mismatch')
    require(manifest['dataset'] == 'textures' and manifest['split'] == 'test' and manifest['partition'] == 1, 'population mismatch')
    require(manifest['count'] == 1880 and manifest['schema_version'] == 1, 'count/schema mismatch')
    with np.load(target, allow_pickle=False) as source:
        images = source['images']
        labels = source['labels']
        filenames = source['filenames'].tolist()
        classes = source['class_names'].tolist()
        original_hw = source['original_hw']
    require(images.shape == (1880, 32, 32, 3) and images.dtype == np.uint8, 'pixel schema mismatch')
    require(labels.shape == (1880,) and labels.dtype == np.int64, 'label schema mismatch')
    require(hashlib.sha256(images.tobytes()).hexdigest() == manifest['images_sha256'], 'pixel-byte hash mismatch')
    require(hashlib.sha256(labels.tobytes()).hexdigest() == manifest['labels_sha256'], 'label-byte hash mismatch')
    lists = {}
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            for split in ('train', 'val', 'test'):
                if member.name == f'dtd/labels/{split}1.txt':
                    require(member.isfile(), 'split is not a regular archive file')
                    with source.extractfile(member) as stream:
                        payload = stream.read()
                    require(hashlib.sha256(payload).hexdigest() == manifest['split_list_sha256'][split], 'split-list hash mismatch')
                    lists[split] = payload.decode('utf-8').splitlines()
    require(set(lists) == {'train', 'val', 'test'}, 'missing official split list')
    split_sets = [set(lists[split]) for split in ('train', 'val', 'test')]
    require(all(len(items) == 1880 for items in split_sets), 'split sample count mismatch')
    require(all(not split_sets[i] & split_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))), 'split overlap')
    require(len(set.union(*split_sets)) == 5640, 'combined population mismatch')
    require(filenames == lists['test'] == manifest['selected_filenames'], 'cache order differs from official test1.txt')
    expected_classes = sorted({name.split('/')[0] for name in filenames})
    require(classes == expected_classes == manifest['class_names'] and len(classes) == 47, 'class mapping mismatch')
    expected_labels = np.array([classes.index(name.split('/')[0]) for name in filenames], dtype=np.int64)
    require(np.array_equal(labels, expected_labels), 'labels differ from official test paths')
    require(np.array_equal(np.bincount(labels, minlength=47), np.full(47, 40)), 'class balance mismatch')
    transform = Compose([Resize(32, interpolation=InterpolationMode.BILINEAR), CenterCrop(32)])
    # Check differing aspect ratios and odd crop margins in addition to real JPEGs.
    geometry_shapes = [(19, 32), (32, 19), (47, 89), (89, 47), (51, 51)]
    for height, width in geometry_shapes:
        grid = np.arange(height * width * 3, dtype=np.int64).reshape(height, width, 3).astype(np.uint8)
        reference = np.asarray(transform(Image.fromarray(grid)))
        resized = (32, int(32 * height / width)) if width <= height else (int(32 * width / height), 32)
        pil = Image.fromarray(grid).resize(resized, Image.Resampling.BILINEAR)
        left, top = round((resized[0] - 32) / 2), round((resized[1] - 32) / 2)
        require(np.array_equal(reference, np.asarray(pil.crop((left, top, left + 32, top + 32)))), 'reference geometry disagreement')
    selected = {f'dtd/images/{name}': index for index, name in enumerate(filenames)}
    seen = set()
    with CanonicalDecoder(args.decoder_python) as decoder:
        decoder_metadata = decoder.metadata.copy()
        require(decoder_metadata['pillow'] == manifest['pillow'], 'canonical decoder Pillow differs from preparation')
        with tarfile.open(archive, 'r|gz') as source:
            for member in source:
                if member.name not in selected:
                    continue
                index = selected[member.name]
                require(index not in seen and member.isfile(), 'duplicate or non-regular selected JPEG')
                with source.extractfile(member) as stream:
                    payload = stream.read()
                require(hashlib.sha256(payload).hexdigest() == manifest['selected_jpeg_sha256'][index], 'JPEG hash mismatch')
                # Decode with the recorded preparation environment. JPEG libraries
                # in different Pillow builds can disagree before any resize occurs.
                # The actual independent Resize/CenterCrop reference still comes
                # from torchvision, applied to identical canonical RGB pixels.
                canonical_rgb = decoder.decode(payload)
                require(np.array_equal(original_hw[index], canonical_rgb.shape[:2]), 'original dimensions mismatch')
                reference = np.asarray(transform(Image.fromarray(canonical_rgb)), dtype=np.uint8)
                require(np.array_equal(images[index], reference), f'torchvision pixel mismatch at official index {index}')
                seen.add(index)
                if len(seen) % 400 == 0:
                    print(f'Independent torchvision pixel checks: {len(seen)}/1880', flush=True)
    require(len(seen) == 1880, 'incomplete independent pixel audit')
    require(not torch.cuda.is_initialized(), 'CPU-only audit initialized CUDA')
    record = {
        'state': 'passed', 'dataset': 'textures', 'split': 'test', 'partition': 1,
        'all_pixel_equality_checks': len(seen), 'geometry_reference_checks': len(geometry_shapes),
        'original_split_isolation': True, 'class_label_and_order_checks': 1880,
        'source_md5': 'fff73e5086ae6bdbea199a49dfb8a4c1',
        'source_sha256': file_hash(archive), 'cache_sha256': file_hash(target),
        'manifest_sha256': file_hash(manifest_path), 'audit_tool_sha256': file_hash(__file__),
        'reference': 'independent torchvision Resize(32, BILINEAR) then CenterCrop(32) on canonical RGB decoded in the preparation Pillow environment',
        'canonical_decoder': decoder_metadata,
        'decoder_worker_sha256': file_hash(Path(__file__).with_name('jpeg_decoder_worker.py')),
        'cross_environment_boundary': 'direct JPEG decoding by a different Pillow/libjpeg build is not assumed pixel-identical; no numeric tolerance was relaxed',
        'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'cuda_initialized': False, 'audited_at': datetime.now(timezone.utc).isoformat(),
        'boundary': 'data preparation verified; no classifier inference or OOD performance result',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), 'existing audit record retained; choose a new output')
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
