"""One native CPU job with append-only measured cost and hard process limits."""
from __future__ import annotations
import argparse
import json
import math
import fcntl
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--id',required=True); p.add_argument('--stage',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    lock_path=ROOT/'results/.queue.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    queue_lock=lock_path.open('a+')
    try:fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('another C1 CPU job owns the serialized queue')
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text())
    ledger=ROOT/'results/JOB_LEDGER.jsonl'
    rows=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    reserves_path=ROOT/'results/RESOURCE_RESERVES.json'
    reserves=json.loads(reserves_path.read_text()) if reserves_path.exists() else {'cpu_seconds':0,'wall_seconds':0,'cpu_by_stage':{}}
    if any(r['id']==a.id for r in rows): raise SystemExit('duplicate run ID refused')
    starts_path=ROOT/'results/JOB_STARTS.jsonl'
    starts=[json.loads(s) for s in starts_path.read_text().splitlines()] if starts_path.exists() else []
    done={r['id'] for r in rows}
    if any(r['id'] not in done for r in starts):raise SystemExit('unfinished prior job; reconcile charged reservation before launch')
    if a.stage not in conf['cpu_stage_caps']: raise SystemExit('unknown stage')
    cpu_remaining=min(conf['cpu_seconds_cap']-reserves['cpu_seconds']-sum(r['cpu_seconds'] for r in rows),
                      conf['cpu_stage_caps'][a.stage]-reserves['cpu_by_stage'].get(a.stage,0)-sum(r['cpu_seconds'] for r in rows if r['stage']==a.stage))
    wall_remaining=conf['wall_seconds_cap']-reserves['wall_seconds']-sum(r['wall_seconds'] for r in rows)
    if min(cpu_remaining,wall_remaining)<1: raise SystemExit('budget exhausted before launch')
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    dest=ROOT/'results/jobs'/a.id
    dest.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='',C1_DEVICE='cpu')
    def limits():
        os.nice(15)
        n=max(1,math.floor(cpu_remaining))
        resource.setrlimit(resource.RLIMIT_CPU,(n,n))
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    w=time.perf_counter()
    with starts_path.open('a') as f:
        f.write(json.dumps(dict(id=a.id,stage=a.stage,cpu_reservation=cpu_remaining,
                               wall_reservation=wall_remaining,command=command))+'\n')
    code=-1; error=None
    with (dest/'stdout.log').open('w') as out,(dest/'stderr.log').open('w') as err:
        try:
            run=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,
                                 preexec_fn=limits,start_new_session=True)
            try:code=run.wait(timeout=max(1,wall_remaining))
            except subprocess.TimeoutExpired:
                os.killpg(run.pid,signal.SIGKILL);run.wait();raise
        except subprocess.TimeoutExpired:
            error='wall limit'; code=-9
        except OSError as e:
            error=type(e).__name__+': '+str(e);code=-127
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    row=dict(id=a.id,stage=a.stage,status='PASS' if code==0 else 'FAILED_CHARGED',returncode=code,error=error,
             command=command,wall_seconds=time.perf_counter()-w,
             cpu_seconds=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime),gpu_seconds=0,
             maxrss_raw=after.ru_maxrss,scope='child process including imports and failures',
             cpu_limit_at_launch=cpu_remaining)
    with ledger.open('a') as f:f.write(json.dumps(row)+'\n')
    (dest/'receipt.json').write_text(json.dumps(row,indent=2)+'\n')
    queue_lock.close()
    print(json.dumps(row))
    print((dest/'stdout.log').read_text()[-2000:])
    if code: print((dest/'stderr.log').read_text()[-4000:]);raise SystemExit(1)
if __name__=='__main__':main()
