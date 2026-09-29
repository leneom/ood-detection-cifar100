"""Independent full-JPEG and full-pixel audit of Oxford-IIIT Pet test cache."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/oxford-pets-extension'
DATA=ROOT/'data'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(ok,why):
 if not ok:raise AssertionError(why)
def official_test():
 matches=[]
 with tarfile.open(DATA/'pets-annotations.tar.gz','r|gz') as stream:
  for member in stream:
   name=PurePosixPath(member.name)
   require(not name.is_absolute() and '..' not in name.parts and '\\' not in member.name,
           'unsafe annotation tar path')
   if member.isfile() and name.name=='test.txt':matches.append(stream.extractfile(member).read())
 require(len(matches)==1,'one official test.txt')
 rows={}
 for text in matches[0].decode('utf-8').splitlines():
  name,label,species,breed=text.split()
  require(name not in rows,'unique test image id')
  rows[name]=(int(label),int(species),int(breed))
 require(len(rows)==3669 and len({x[0] for x in rows.values()})==37,'official test census')
 return matches[0],rows

def main():
 out=RUN/'cache-audit.json'
 require(not out.exists(),'cache audit exists')
 d=read(RUN/'declaration.json');download=read(RUN/'download-status.json')
 manifest=read(RUN/'cache-manifest.json');status=read(RUN/'cache-status.json')
 require(d['protocol_sha256']==sha(ROOT/'docs/OXFORD_PETS_EXTENSION.md') and
         download['state']=='verified_archives_pending_pixel_cache' and
         status['manifest_sha256']==sha(RUN/'cache-manifest.json') and
         manifest['count']==3669 and manifest['source_sha256']==
         sha(RUN/'executed_cache_source.py') and manifest['cache_sha256']==
         sha(DATA/'pets-test.npz'),'declared cache binding')
 for role in ('images','annotations'):
  item=download['files'][role]
  require(item['sha256']==sha(ROOT/item['relative_path']) and
          item['md5']==d['expected_md5'][role],'official archive unchanged '+role)
 raw_ann,rows=official_test()
 require(manifest['annotation_test_sha256']==hashlib.sha256(raw_ann).hexdigest(),
         'test annotation bytes')
 with np.load(DATA/'pets-test.npz',allow_pickle=False) as z:
  cached={k:z[k] for k in z.files}
 names=sorted(rows);expected_names=np.asarray([n+'.jpg' for n in names])
 require(np.array_equal(cached['filenames'],expected_names) and
         cached['images'].shape==(3669,32,32,3) and
         cached['images'].dtype==np.uint8 and
         np.array_equal(cached['labels'],[rows[n][0] for n in names]) and
         np.array_equal(cached['species'],[rows[n][1] for n in names]) and
         np.array_equal(cached['breed_metadata'],[rows[n][2] for n in names]),
         'cache census labels and order')
 expected_index={name:i for i,name in enumerate(names)}
 observed=set();all_jpeg=set();duplicates=0
 with tarfile.open(DATA/'pets-images.tar.gz','r|gz') as stream:
  for member in stream:
   path=PurePosixPath(member.name)
   require(not path.is_absolute() and '..' not in path.parts and
           '\\' not in member.name,'unsafe image tar path')
   if not member.isfile() or path.suffix.lower()!='.jpg':continue
   name=path.stem
   require(name not in all_jpeg,'unique JPEG basename')
   all_jpeg.add(name)
   if name not in expected_index:continue
   i=expected_index[name];raw=stream.extractfile(member).read()
   require(hashlib.sha256(raw).hexdigest()==str(cached['jpeg_sha256'][i]),
           'original JPEG hash mismatch '+name)
   with Image.open(io.BytesIO(raw)) as picture:
    picture.load();hw=(picture.height,picture.width)
    rgb=np.asarray(picture.convert('RGB').resize((32,32),Image.Resampling.BILINEAR),dtype=np.uint8)
   require(np.array_equal(cached['source_hw'][i],hw) and
           np.array_equal(cached['images'][i],rgb) and
           hashlib.sha256(rgb.tobytes()).hexdigest()==str(cached['resized_sha256'][i]),
           'independent resized pixel mismatch '+name)
   observed.add(name)
 require(observed==set(names),'all test images audited')
 counts=Counter(map(int,cached['labels']))
 require({str(k):v for k,v in sorted(counts.items())}==manifest['label_counts'],
         'full category counts')
 duplicate_count=3669-len(set(map(str,cached['resized_sha256'])))
 require(duplicate_count==manifest['within_ood_resized_pixel_duplicates'],
         'within pet duplicate count')
 record={'state':'passed','audited_at':datetime.now(timezone.utc).isoformat(),
         'images_checked':3669,'labels_checked':3669,'jpeg_hashes_checked':3669,
         'resized_pixel_arrays_checked':3669,'categories_checked':37,
         'within_ood_resized_duplicates':duplicate_count,
         'cache_sha256':sha(DATA/'pets-test.npz'),
         'image_archive_sha256':sha(DATA/'pets-images.tar.gz'),
         'annotation_archive_sha256':sha(DATA/'pets-annotations.tar.gz'),
         'manifest_sha256':sha(RUN/'cache-manifest.json'),
         'source_sha256':sha(__file__),
         'boundary':'Official test/JPEG/resized-pixel integrity only; no model or OOD score examined.'}
 out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'state':'passed','images_checked':3669,
                   'within_ood_duplicates':duplicate_count}))
if __name__=='__main__':main()