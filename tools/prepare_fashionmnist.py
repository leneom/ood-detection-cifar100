"""Decode verified Fashion-MNIST test IDX and match frozen MNIST preprocessing."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fashionmnist-extension'
DATA=ROOT/'data/fashionmnist-test'
CACHE=ROOT/'data/fashionmnist-test-32x32.npz'
sys.path.insert(0,str(ROOT))
from src.ood.data import load_cifar


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,record):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def main():
    if CACHE.exists() or (RUN/'cache-manifest.json').exists():
        raise FileExistsError('existing Fashion cache needs review')
    declaration=read(RUN/'declaration.json')
    downloaded=read(RUN/'download-status.json')
    if (declaration['protocol_sha256']!=sha(ROOT/'docs/FASHIONMNIST_EXTENSION.md') or
            downloaded['state']!='verified_test_archives_downloaded' or
            downloaded['protocol_sha256']!=declaration['protocol_sha256']):
        raise ValueError('frozen protocol or verified official test files differ')
    source=RUN/'executed_cache_source.py'
    shutil.copyfile(__file__,source)
    image_archive=DATA/'t10k-images-idx3-ubyte.gz'
    label_archive=DATA/'t10k-labels-idx1-ubyte.gz'
    for p in (image_archive,label_archive):
        if sha(p)!=downloaded['files'][p.name]['sha256']:
            raise ValueError('downloaded source archive changed')
    raw_images=gzip.decompress(image_archive.read_bytes())
    raw_labels=gzip.decompress(label_archive.read_bytes())
    if len(raw_images)<16 or struct.unpack('>4I',raw_images[:16])!=(2051,10000,28,28):
        raise ValueError('official 10k 28x28 IDX image header differs')
    if len(raw_labels)<8 or struct.unpack('>2I',raw_labels[:8])!=(2049,10000):
        raise ValueError('official 10k IDX label header differs')
    if len(raw_images)!=16+10000*28*28 or len(raw_labels)!=8+10000:
        raise ValueError('official IDX payload length differs')
    gray=np.frombuffer(raw_images,dtype=np.uint8,offset=16).reshape(10000,28,28).copy()
    labels=np.frombuffer(raw_labels,dtype=np.uint8,offset=8).copy()
    counts=np.bincount(labels.astype(np.int64),minlength=10)
    if labels.max()>9 or not np.array_equal(counts,np.full(10,1000)):
        raise ValueError('official ten-class test-label census differs')
    images=np.empty((10000,32,32,3),dtype=np.uint8)
    for start in range(0,10000,500):
        block=torch.from_numpy(gray[start:start+500])
        up=F.interpolate(block[:,None].float(),size=(32,32),mode='bilinear',align_corners=False)
        rgb=up.round().clamp(0,255).to(torch.uint8).repeat(1,3,1,1).permute(0,2,3,1).contiguous()
        images[start:start+len(block)]=rgb.numpy()
    pixel_hashes=np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in images])
    train,_=load_cifar(ROOT/'data','cifar100',True)
    test,_=load_cifar(ROOT/'data','cifar100',False)
    id_hashes={hashlib.sha256(np.asarray(row).tobytes()).hexdigest()
               for source_array in (train,test) for row in source_array}
    duplicates=set(pixel_hashes)&id_hashes
    if duplicates:
        raise ValueError(f'{len(duplicates)} exact pixel duplicates with CIFAR-100 train/test')
    temp=CACHE.with_suffix('.npz.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,images=images,labels=labels,source_rows=np.arange(10000,dtype=np.int64),
                            pixel_sha256=pixel_hashes)
    temp.replace(CACHE)
    manifest={'state':'verified_source_and_pixels_pending_independent_audit',
              'created_at':datetime.now(timezone.utc).isoformat(),
              'count':10000,'labels_per_class':counts.tolist(),
              'image_shape':[10000,32,32,3],'dtype':'uint8',
              'archive_sha256':{p.name:sha(p) for p in (image_archive,label_archive)},
              'decompressed_idx_sha256':{'images':hashlib.sha256(raw_images).hexdigest(),
                                         'labels':hashlib.sha256(raw_labels).hexdigest()},
              'cache_sha256':sha(CACHE),'source_sha256':sha(__file__),
              'source_snapshot_sha256':sha(source),
              'protocol_sha256':declaration['protocol_sha256'],
              'mnist_transform_source_sha256':sha(ROOT/'src/ood/datasets.py'),
              'cifar100_exact_pixel_duplicates':0,
              'torch_version':torch.__version__,
              'boundary':'Source and pixel integrity only; no classifier/OOD score has been computed.'}
    write(RUN/'cache-manifest.json',manifest)
    write(RUN/'cache-status.json',{'state':manifest['state'],
                                  'manifest_sha256':sha(RUN/'cache-manifest.json')})
    print(json.dumps({'state':manifest['state'],'count':10000,
                      'duplicate_cifar_pixels':0}),flush=True)

if __name__=='__main__': main()
