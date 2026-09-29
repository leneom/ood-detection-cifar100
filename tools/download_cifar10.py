"""Download the official CIFAR-10 archive in verified HTTP ranges.

Completed parts can be reused after interruption. Final integrity is the
torchvision-published MD5. No datasets are usable until verification succeeds.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import urllib.request

from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
SIZE = 170498071
PART_SIZE = 4 * 1024 * 1024


def download_part(index):
    begin = index * PART_SIZE
    end = min(SIZE, begin + PART_SIZE) - 1
    target = DATA / 'download_parts' / 'cifar10' / f'{index:03d}.part'
    if target.exists() and target.stat().st_size == end - begin + 1:
        return index
    temp = target.with_suffix('.tmp')
    for attempt in range(3):
        try:
            request = urllib.request.Request(URL, headers={'Range': f'bytes={begin}-{end}'})
            with urllib.request.urlopen(request, timeout=30) as response, temp.open('wb') as stream:
                expected = f'bytes {begin}-{end}/{SIZE}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise RuntimeError('server did not honor requested range')
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break
                    stream.write(block)
            if temp.stat().st_size != end - begin + 1:
                raise RuntimeError('partial range has wrong size')
            temp.replace(target)
            return index
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    target = DATA / CIFAR10.filename
    if check_integrity(str(target), CIFAR10.tgz_md5):
        print('official archive already verified', flush=True)
        return
    parts = DATA / 'download_parts' / 'cifar10'
    parts.mkdir(parents=True, exist_ok=True)
    indices = range((SIZE + PART_SIZE - 1) // PART_SIZE)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index in pool.map(download_part, indices):
            print(f'verified-length range {index + 1}/{len(indices)}', flush=True)
    merged = DATA / 'cifar10-merged.tmp'
    with merged.open('wb') as output:
        for index in indices:
            with (parts / f'{index:03d}.part').open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
    if not check_integrity(str(merged), CIFAR10.tgz_md5):
        raise RuntimeError('official CIFAR-10 MD5 mismatch; archive not installed')
    merged.replace(target)
    CIFAR10(str(DATA), train=False, download=True)
    print('CIFAR-10 test split verified and extracted', flush=True)


if __name__ == '__main__':
    main()
