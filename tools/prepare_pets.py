"""Build all official Oxford-IIIT Pet test pixels from MD5-verified archives."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib,io,json,re,shutil,sys,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image,__version__ as pillow_version
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/oxford-pets-extension'
DATA=ROOT/'data'
CACHE=DATA/'pets-test.npz'
sys.path.insert(0,str(ROOT))
from src.ood.data import load_cifar

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def write(path,record):
 temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def require(ok,why):
 if not ok:raise ValueError(why)
def safe(member):
 path=PurePosixPath(member.name)
 require(not path.is_absolute() and '..' not in path.parts and
         '\\' not in member.name and ':' not in path.parts[0] and
         (member.isfile() or member.isdir()),'unsafe tar member '+member.name)
 return path

def annotation_rows(path):
 with tarfile.open(path,'r:gz') as archive:
  members=archive.getmembers();names=[m.name for m in members]
  require(len(names)==len(set(names)),'duplicate annotation archive path')
  found=[]
  for member in members:
   member_path=safe(member)
   if member.isfile() and member_path.name=='test.txt':found.append(member)
  require(len(found)==1,'exact official test.txt required')
  raw=archive.extractfile(found[0]).read()
 rows={}
 for line in raw.decode('utf-8').splitlines():
  parts=line.split()
  require(len(parts)==4,'invalid test annotation row')
  name=parts[0]
  require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*_[0-9]+',name) is not None,
          'unsafe image id '+name)
  require(name not in rows,'duplicate test image id')
  label,species,breed=map(int,parts[1:])
  require(1<=label<=37 and 1<=species<=2 and 1<=breed<=37,
          'invalid test category metadata')
  rows[name]=(label,species,breed)
 require(len(rows)==3669 and len({v[0] for v in rows.values()})==37,
         'official 3669/37 test census changed')
 return raw,rows

def main():
 d,status=read(RUN/'declaration.json'),read(RUN/'download-status.json')
 require(d['state']=='protocol_frozen_before_pet_archive_access' and
         d['protocol_sha256']==sha(ROOT/'docs/OXFORD_PETS_EXTENSION.md') and
         status['state']=='verified_archives_pending_pixel_cache' and
         status['declaration_sha256']==sha(RUN/'declaration.json'),
         'frozen declared archives required')
 for name,expected in d['input_sha256'].items():require(sha(ROOT/name)==expected,'frozen ID input '+name)
 for role in ('images','annotations'):
  item=status['files'][role]
  require(item['sha256']==sha(ROOT/item['relative_path']) and
          item['md5']==d['expected_md5'][role],'official archive hash '+role)
 require(not any(p.exists() for p in (CACHE,RUN/'cache-status.json',RUN/'cache-manifest.json',
                                      RUN/'executed_cache_source.py')),'cache attempt already exists')
 annotation,rows=annotation_rows(DATA/'pets-annotations.tar.gz')
 names=sorted(rows);position={name:i for i,name in enumerate(names)}
 images=np.empty((3669,32,32,3),dtype=np.uint8)
 dimensions=np.empty((3669,2),dtype=np.int32)
 jpeg_hashes=['']*3669;pixel_hashes=['']*3669
 seen=set();all_jpeg=set()
 snapshot=RUN/'executed_cache_source.py';shutil.copyfile(__file__,snapshot)
 write(RUN/'cache-status.json',{'state':'preparing_verified_pet_pixels','started_at':now(),
       'declaration_sha256':sha(RUN/'declaration.json'),
       'source_sha256':sha(__file__)})
 try:
  with tarfile.open(DATA/'pets-images.tar.gz','r|gz') as archive:
   for member in archive:
    path=safe(member)
    if not member.isfile() or path.suffix.lower()!='.jpg':continue
    name=path.stem
    require(name not in all_jpeg,'duplicate JPEG basename '+name)
    all_jpeg.add(name)
    if name not in position:continue
    i=position[name]
    raw=archive.extractfile(member).read()
    jpeg_hashes[i]=hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as picture:
     picture.load();dimensions[i]=(picture.height,picture.width)
     pixels=np.asarray(picture.convert('RGB').resize(
         (32,32),Image.Resampling.BILINEAR),dtype=np.uint8)
    require(pixels.shape==(32,32,3),'resized RGB pixel shape')
    images[i]=pixels
    pixel_hashes[i]=hashlib.sha256(pixels.tobytes()).hexdigest()
    seen.add(name)
    if len(seen)%500==0:print(json.dumps({'decoded':len(seen),'total':3669}),flush=True)
  require(seen==set(names),'missing official test JPEG IDs')
  train,_=load_cifar(DATA,'cifar100',True)
  test,_=load_cifar(DATA,'cifar100',False)
  id_hashes={hashlib.sha256(np.asarray(row).tobytes()).hexdigest()
             for population in (train,test) for row in population}
  overlap=set(pixel_hashes)&id_hashes
  require(not overlap,f'{len(overlap)} exact CIFAR-100 pixel overlaps')
  within=len(pixel_hashes)-len(set(pixel_hashes))
  temp=CACHE.with_suffix('.npz.tmp')
  with temp.open('xb') as stream:
   np.savez_compressed(stream,images=images,filenames=np.asarray([n+'.jpg' for n in names]),
      labels=np.asarray([rows[n][0] for n in names],dtype=np.int16),
      species=np.asarray([rows[n][1] for n in names],dtype=np.int8),
      breed_metadata=np.asarray([rows[n][2] for n in names],dtype=np.int16),
      source_hw=dimensions,jpeg_sha256=np.asarray(jpeg_hashes),
      resized_sha256=np.asarray(pixel_hashes))
  temp.replace(CACHE)
  manifest={'state':'verified_source_and_pixels_pending_independent_audit',
      'created_at':now(),'count':3669,'categories':37,
      'label_counts':{str(k):v for k,v in sorted(Counter(rows[n][0] for n in names).items())},
      'image_shape':[3669,32,32,3],'dtype':'uint8',
      'image_archive_sha256':status['files']['images']['sha256'],
      'annotation_archive_sha256':status['files']['annotations']['sha256'],
      'annotation_test_sha256':hashlib.sha256(annotation).hexdigest(),
      'cache_sha256':sha(CACHE),'pillow_version':pillow_version,
      'source_sha256':sha(__file__),'source_snapshot_sha256':sha(snapshot),
      'declaration_sha256':sha(RUN/'declaration.json'),
      'cifar100_exact_pixel_duplicates':0,'within_ood_resized_pixel_duplicates':within,
      'boundary':'Official test source/pixel integrity only; no model inference or OOD score.'}
  write(RUN/'cache-manifest.json',manifest)
  write(RUN/'cache-status.json',{'state':manifest['state'],'completed_at':now(),
        'manifest_sha256':sha(RUN/'cache-manifest.json')})
  print(json.dumps({'state':manifest['state'],'count':3669,'within_ood_duplicates':within}),flush=True)
 except Exception as exc:
  write(RUN/'cache-status.json',{'state':'cache_failed_preserve_evidence','failed_at':now(),
      'error':repr(exc),'source_sha256':sha(__file__)})
  raise
if __name__=='__main__':main()