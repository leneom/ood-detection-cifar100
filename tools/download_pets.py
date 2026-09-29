"""Download the two MD5-pinned official Oxford-IIIT Pet archives locally."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import sys,urllib.request

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/oxford-pets-extension'
DATA=ROOT/'data'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,record):
 path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
def main():
 d=json.loads((RUN/'declaration.json').read_text(encoding='utf-8'))
 if d['state']!='protocol_frozen_before_pet_archive_access' or d['protocol_sha256']!=sha(ROOT/'docs/OXFORD_PETS_EXTENSION.md'):
  raise ValueError('frozen protocol mismatch')
 for name,expected in d['input_sha256'].items():
  if sha(ROOT/name)!=expected:raise ValueError('frozen input changed '+name)
 status_path=RUN/'download-status.json'
 if status_path.exists():raise FileExistsError('download status exists; inspect before any restart')
 status={'state':'downloading_official_archives','pid':os.getpid(),'started_at':now(),
         'declaration_sha256':sha(RUN/'declaration.json'),'source_sha256':sha(Path(__file__)),
         'files':{}}
 write(status_path,status)
 for role,filename in (('annotations','pets-annotations.tar.gz'),('images','pets-images.tar.gz')):
  url=d['source_urls'][role];target=DATA/filename;part=DATA/(filename+'.part')
  if target.exists() or part.exists():raise FileExistsError('archive or partial already exists '+filename)
  request=urllib.request.Request(url,headers={'User-Agent':'OOD-Detection-Research/1.0'})
  md5=hashlib.md5();h=hashlib.sha256();size=0
  with urllib.request.urlopen(request,timeout=90) as response,part.open('xb') as output:
   if response.status!=200:raise RuntimeError('unexpected HTTP status '+str(response.status))
   resolved=response.geturl()
   if not resolved.startswith('https://') or not resolved.endswith('/pets/'+filename.replace('pets-','')):
    raise RuntimeError('unexpected official redirect '+resolved)
   content_length=response.headers.get('Content-Length')
   while True:
    block=response.read(1<<20)
    if not block:break
    output.write(block);md5.update(block);h.update(block);size+=len(block)
   output.flush();os.fsync(output.fileno())
  if content_length and size!=int(content_length):raise RuntimeError('truncated archive '+role)
  if md5.hexdigest()!=d['expected_md5'][role]:raise RuntimeError('official MD5 mismatch '+role)
  os.replace(part,target)
  status['files'][role]={'state':'verified_archive','relative_path':target.relative_to(ROOT).as_posix(),
      'bytes':size,'md5':md5.hexdigest(),'sha256':h.hexdigest(),
      'url':resolved,'declared_url':url}
  status['updated_at']=now();write(status_path,status)
  print(json.dumps({'role':role,'bytes':size,'md5':md5.hexdigest()}),flush=True)
 status['state']='verified_archives_pending_pixel_cache';status['completed_at']=now();write(status_path,status)
 print(json.dumps({'state':status['state'],'roles':list(status['files'])}),flush=True)
if __name__=='__main__':main()