"""Independent full IDX, NumPy bilinear and exact-overlap audit."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fashionmnist-extension'
DATA=ROOT/'data/fashionmnist-test'
CACHE=ROOT/'data/fashionmnist-test-32x32.npz'
sys.path.insert(0,str(ROOT))
from src.ood.data import load_cifar


def digest(path,kind='sha256'):
    h=hashlib.new(kind)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def require(ok,why):
    if not ok: raise AssertionError(why)


def independently_resize(gray):
    # Half-pixel 28 -> 32 bilinear map, separately implemented in NumPy.
    coordinate=(np.arange(32,dtype=np.float32)+np.float32(.5))*np.float32(28/32)-np.float32(.5)
    coordinate=np.clip(coordinate,0,27)
    lower=np.floor(coordinate).astype(np.int64)
    upper=np.minimum(lower+1,27)
    high=(coordinate-lower).astype(np.float32)
    low=np.float32(1)-high
    x=(gray[:,:,lower].astype(np.float32)*low[None,None,:]+
       gray[:,:,upper].astype(np.float32)*high[None,None,:])
    y=x[:,lower,:]*low[None,:,None]+x[:,upper,:]*high[None,:,None]
    return np.rint(np.clip(y,0,255)).astype(np.uint8)


def main():
    declaration=read(RUN/'declaration.json')
    downloaded=read(RUN/'download-status.json')
    manifest=read(RUN/'cache-manifest.json')
    status=read(RUN/'cache-status.json')
    require(status['manifest_sha256']==digest(RUN/'cache-manifest.json') and
            manifest['cache_sha256']==digest(CACHE) and
            manifest['protocol_sha256']==declaration['protocol_sha256']==digest(ROOT/'docs/FASHIONMNIST_EXTENSION.md') and
            manifest['source_sha256']==digest(ROOT/'tools/prepare_fashionmnist.py')==
            manifest['source_snapshot_sha256']==digest(RUN/'executed_cache_source.py'),
            'frozen cache and executed-source binding')
    originals=[]
    for filename,expected in (('t10k-images-idx3-ubyte.gz','bef4ecab320f06d8554ea6380940ec79'),
                              ('t10k-labels-idx1-ubyte.gz','bb300cfdad3c16e7a12a480ee83cd310')):
        path=DATA/filename
        require(digest(path,'md5')==expected and
                digest(path)==manifest['archive_sha256'][filename]==downloaded['files'][filename]['sha256'],
                'official archive bytes '+filename)
        originals.append(gzip.decompress(path.read_bytes()))
    images_idx,labels_idx=originals
    require(struct.unpack('>4I',images_idx[:16])==(2051,10000,28,28) and
            struct.unpack('>2I',labels_idx[:8])==(2049,10000) and
            len(images_idx)==16+10000*784 and len(labels_idx)==8+10000,
            'full IDX headers and payloads')
    require(hashlib.sha256(images_idx).hexdigest()==manifest['decompressed_idx_sha256']['images'] and
            hashlib.sha256(labels_idx).hexdigest()==manifest['decompressed_idx_sha256']['labels'],
            'decompressed IDX hashes')
    gray=np.frombuffer(images_idx,dtype=np.uint8,offset=16).reshape(10000,28,28)
    labels=np.frombuffer(labels_idx,dtype=np.uint8,offset=8)
    with np.load(CACHE,allow_pickle=False) as z:
        cached={k:z[k] for k in z.files}
    require(set(cached)=={'images','labels','source_rows','pixel_sha256'} and
            cached['images'].shape==(10000,32,32,3) and cached['images'].dtype==np.uint8 and
            np.array_equal(cached['labels'],labels) and
            np.array_equal(cached['source_rows'],np.arange(10000)) and
            np.array_equal(np.bincount(labels,minlength=10),np.full(10,1000)),
            'cached full row and label population')
    equal=0
    for start in range(0,10000,250):
        expected=independently_resize(gray[start:start+250])
        actual=cached['images'][start:start+250]
        require(np.array_equal(actual[:,:,:,0],expected) and
                np.array_equal(actual[:,:,:,1],expected) and
                np.array_equal(actual[:,:,:,2],expected),
                'all full-resolution transformed pixels')
        equal+=len(expected)
    pixels=cached['images']
    hashes=np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in pixels])
    require(np.array_equal(hashes,cached['pixel_sha256']),'all 10k row pixel hashes')
    train,_=load_cifar(ROOT/'data','cifar100',True)
    test,_=load_cifar(ROOT/'data','cifar100',False)
    id_hashes={hashlib.sha256(np.asarray(row).tobytes()).hexdigest()
               for source_array in (train,test) for row in source_array}
    require(len(set(hashes)&id_hashes)==0,'no exact CIFAR100 pixel duplication')
    path=RUN/'cache-audit.json'
    require(not path.exists(),'cache audit already exists')
    record={'state':'passed','count':10000,'independent_numpy_pixels_checked':equal,
            'labels_per_class':np.bincount(labels,minlength=10).tolist(),
            'cifar100_exact_pixel_duplicates':0,
            'archive_sha256':manifest['archive_sha256'],
            'cache_sha256':digest(CACHE),
            'manifest_sha256':digest(RUN/'cache-manifest.json'),
            'audit_source_sha256':digest(__file__),
            'created_at':datetime.now(timezone.utc).isoformat(),
            'boundary':'Archive/label/pixel integrity only; no classifier or OOD score has been inspected.'}
    path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':'passed','pixels_checked':equal,'duplicates':0}),flush=True)

if __name__=='__main__': main()
