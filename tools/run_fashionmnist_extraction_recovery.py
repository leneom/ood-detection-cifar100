"""Resume Fashion-MNIST extraction only after repeated low-load GPU checks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/fashionmnist-extension'
SOURCE=ROOT/'tools/extract_fashionmnist.py'
MAX_USED_MIB=4096
MAX_UTIL_PERCENT=55
MAX_MEMORY_UTIL_PERCENT=10
MAX_POWER_W=45
SAMPLES=3
SAMPLE_GAP_SECONDS=2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):
            h.update(part)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def now(): return datetime.now(timezone.utc).isoformat()

def write(path,record):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def inside(path,root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())

def gpu_samples():
    result=[]
    for attempt in range(SAMPLES):
        output=subprocess.check_output(
            ['nvidia-smi','--query-gpu=memory.used,utilization.gpu,utilization.memory,power.draw,memory.total',
             '--format=csv,noheader,nounits'],text=True,timeout=20).strip()
        rows=[line.strip() for line in output.splitlines() if line.strip()]
        if len(rows)!=1: raise RuntimeError('exactly one verified GPU required')
        used,util,mem_util,power,total=(part.strip() for part in rows[0].split(','))
        result.append({'used_mib':int(used),'util_percent':int(util),
                       'memory_util_percent':int(mem_util),'power_w':float(power),
                       'total_mib':int(total)})
        if attempt+1<SAMPLES: time.sleep(SAMPLE_GAP_SECONDS)
    return result

def idle(samples):
    return all(item['used_mib']<=MAX_USED_MIB and
               item['util_percent']<=MAX_UTIL_PERCENT and
               item['memory_util_percent']<=MAX_MEMORY_UTIL_PERCENT and
               item['power_w']<=MAX_POWER_W for item in samples)

def check_complete(seed,source_hash):
    folder=RUN/f'extract-seed{seed}'
    if not folder.exists(): return False
    status=read(folder/'status.json')
    manifest=read(folder/'manifest.json')
    if not (status['state']=='extracted_pending_independent_audit_and_scoring' and
            status['manifest_sha256']==sha(folder/'manifest.json') and
            manifest['seed']==seed and
            manifest['source_hashes']['tools/extract_fashionmnist.py']==source_hash and
            all(sha(folder/name)==value for name,value in manifest['outputs'].items())):
        raise ValueError(f'existing seed {seed} folder is not verified complete')
    return True

def main():
    lock_path=RUN/'extraction-controller.lock'
    with lock_path.open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0');lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            original=read(RUN/'extraction-launch.json')
            source_hash=original['extract_source_sha256']
            if sha(SOURCE)!=source_hash or any(
                    sha(ROOT/name)!=digest for name,digest in
                    original['source_dependencies_sha256'].items()):
                raise ValueError('executed frozen extraction source changed')
            previous=read(RUN/'extraction-status.json')
            if previous['state']!='extraction_failed_preserve_partial':
                raise ValueError('original failed-attempt record differs')
            recovery_path=RUN/'recovery-launch.json'
            if not recovery_path.exists():
                if any((RUN/f'extract-seed{seed}').exists() for seed in (1,2)):
                    raise ValueError('unexpected nonzero seed result before recovery')
                partial=RUN/'extract-seed0'
                if not partial.is_dir() or read(partial/'status.json')['state']!='extracting':
                    raise ValueError('expected interrupted seed-0 folder missing')
                if any((partial/name).exists() for name in ('raw.npz','T1_scores.npz','feature_scores.npz','manifest.json')):
                    raise ValueError('partial result contains output requiring separate review')
                samples=gpu_samples()
                if not idle(samples):
                    write(RUN/'recovery-status.json',
                          {'state':'deferred_gpu_busy_no_mutation','samples':samples,'updated_at':now()})
                    print(json.dumps({'state':'deferred_gpu_busy_no_mutation','samples':samples}),flush=True)
                    return
                archive=RUN/'interrupted-attempt-1'
                if archive.exists(): raise FileExistsError('interrupted attempt archive exists')
                if not (inside(partial,ROOT) and inside(archive,ROOT) and
                        inside(partial,RUN) and inside(archive,RUN)):
                    raise ValueError('resolved archive path outside intended workspace')
                archive.mkdir()
                partial.rename(archive/'extract-seed0')
                for name in ('extraction-launch.json','extraction-status.json',
                             'extraction-controller.stdout.log','extraction-controller.stderr.log',
                             'extract-seed0.stdout.log','extract-seed0.stderr.log'):
                    path=RUN/name
                    if path.exists(): shutil.copyfile(path,archive/name)
                write(recovery_path,
                      {'state':'launched_after_failed_gpu_contention','launched_at':now(),
                       'pid':os.getpid(),'python':sys.executable,'seeds':[0,1,2],
                       'original_launch_sha256':sha(RUN/'extraction-launch.json'),
                       'original_failure_sha256':sha(RUN/'extraction-status.json'),
                       'archived_partial_status_sha256':sha(archive/'extract-seed0/status.json'),
                       'extract_source_sha256':source_hash,
                       'source_dependencies_sha256':original['source_dependencies_sha256'],
                       'gpu_guard':{'max_used_mib':MAX_USED_MIB,
                                    'max_util_percent':MAX_UTIL_PERCENT,
                                    'max_memory_util_percent':MAX_MEMORY_UTIL_PERCENT,
                                    'max_power_w':MAX_POWER_W,
                                    'samples':SAMPLES,'gap_seconds':SAMPLE_GAP_SECONDS},
                       'prelaunch_gpu_samples':samples})
            else:
                recovery=read(recovery_path)
                if recovery['extract_source_sha256']!=source_hash:
                    raise ValueError('recovery source changed')
            completed=[]
            for seed in range(3):
                if check_complete(seed,source_hash):
                    completed.append(seed)
                    continue
                samples=gpu_samples()
                if not idle(samples):
                    write(RUN/'recovery-status.json',
                          {'state':'deferred_gpu_busy','next_seed':seed,
                           'completed_seeds':completed,'samples':samples,'updated_at':now()})
                    print(json.dumps({'state':'deferred_gpu_busy','next_seed':seed,
                                      'completed_seeds':completed,'samples':samples}),flush=True)
                    return
                write(RUN/'recovery-status.json',
                      {'state':'extracting','seed':seed,'completed_seeds':completed,
                       'pid':os.getpid(),'gpu_samples':samples,'updated_at':now()})
                env=os.environ.copy()
                env.pop('CUDA_VISIBLE_DEVICES',None)
                env.update({'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
                            'OPENBLAS_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
                stdout=RUN/f'recovery-seed{seed}.stdout.log'
                stderr=RUN/f'recovery-seed{seed}.stderr.log'
                with stdout.open('w',encoding='utf-8') as out,stderr.open('w',encoding='utf-8') as err:
                    subprocess.run([sys.executable,'-X','utf8','tools/extract_fashionmnist.py',
                                    '--seed',str(seed)],cwd=ROOT,env=env,
                                   stdout=out,stderr=err,check=True)
                if not check_complete(seed,source_hash):
                    raise ValueError(f'seed {seed} exited without complete audited outputs')
                completed.append(seed)
            write(RUN/'recovery-status.json',
                  {'state':'three_seeds_extracted_pending_audit_and_scoring',
                   'completed_seeds':completed,
                   'recovery_launch_sha256':sha(recovery_path),
                   'updated_at':now()})
            print(json.dumps({'state':'three_seeds_extracted_pending_audit_and_scoring',
                              'completed_seeds':completed}),flush=True)
        except Exception as exc:
            write(RUN/'recovery-status.json',
                  {'state':'recovery_failed_preserve_partial','error':repr(exc),
                   'traceback':traceback.format_exc(),
                   'pid':os.getpid(),'updated_at':now()})
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__': main()
