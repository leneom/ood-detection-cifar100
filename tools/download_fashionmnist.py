"""Download only verified Fashion-MNIST test IDX gzip files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fashionmnist-extension'
DATA=ROOT/'data/fashionmnist-test'
BASE='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
FILES={
 't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
 't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310',
}

def digest(path,kind='sha256'):
    h=hashlib.new(kind)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,record):
    path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')

def main():
    declaration=read(RUN/'declaration.json')
    if (declaration['state']!='frozen_before_any_fashion_data_access' or
        declaration['protocol_sha256']!=digest(ROOT/'docs/FASHIONMNIST_EXTENSION.md')):
        raise ValueError('frozen population declaration missing or changed')
    DATA.mkdir(parents=True,exist_ok=True)
    status=RUN/'download-status.json'
    if status.exists(): raise FileExistsError('download status already exists')
    write(status,{'state':'downloading','started_at':datetime.now(timezone.utc).isoformat(),
                  'source_sha256':digest(__file__)})
    evidence={}
    for filename,expected_md5 in FILES.items():
        destination=DATA/filename
        if destination.exists():
            if digest(destination,'md5')!=expected_md5:
                raise ValueError('existing local archive has wrong official MD5: '+filename)
        else:
            temp=destination.with_suffix(destination.suffix+'.partial')
            if temp.exists(): raise FileExistsError('partial download needs review: '+str(temp))
            request=Request(BASE+filename,headers={'User-Agent':'OOD-local-research/1.0'})
            with urlopen(request,timeout=90) as response, temp.open('wb') as f:
                while True:
                    block=response.read(1<<20)
                    if not block: break
                    f.write(block)
            if digest(temp,'md5')!=expected_md5:
                raise ValueError('downloaded archive MD5 differs from official metadata: '+filename)
            temp.replace(destination)
        evidence[filename]={'url':BASE+filename,'bytes':destination.stat().st_size,
                            'md5':digest(destination,'md5'),'sha256':digest(destination)}
    record={'state':'verified_test_archives_downloaded',
            'files':evidence,'protocol_sha256':declaration['protocol_sha256'],
            'source_sha256':digest(__file__),
            'completed_at':datetime.now(timezone.utc).isoformat(),
            'boundary':'Only official test image/label files downloaded; no Fashion-MNIST preprocessing or scoring yet.'}
    write(status,record)
    print(json.dumps({'state':record['state'],'files':{k:v['bytes'] for k,v in evidence.items()}}),flush=True)

if __name__=='__main__': main()
