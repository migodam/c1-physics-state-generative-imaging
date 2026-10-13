"""Seal after the last job's terminal receipt exists; no science or physical reruns."""
from pathlib import Path
import json
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    start_cpu,start_wall=time.process_time(),time.perf_counter()
    reserve_path=ROOT/'results/RESOURCE_RESERVES.json';reserve=json.loads(reserve_path.read_text())
    if not reserve.get('post_job_seal_charge'):
        reserve['cpu_seconds']+=3;reserve['wall_seconds']+=4;reserve['cpu_by_stage']['diagnostics']+=3
        reserve['post_job_seal_charge']=dict(cpu_seconds=3,wall_seconds=4,
            scope='conservative charge for final metadata refresh and archive sealing after last measured job exits')
        reserve_path.write_text(json.dumps(reserve,indent=2)+'\n')
    jobs=[json.loads(s) for s in (ROOT/'results/JOB_LEDGER.jsonl').read_text().splitlines()]
    starts=[json.loads(s) for s in (ROOT/'results/JOB_STARTS.jsonl').read_text().splitlines()]
    done={j['id'] for j in jobs}
    if any(s['id'] not in done for s in starts):raise RuntimeError('cannot seal unfinished queue')
    cost=json.loads((ROOT/'C1_RESOURCE_LEDGER.json').read_text())
    cost.update(measured_cpu_seconds=sum(j['cpu_seconds'] for j in jobs),
        conservative_cpu_reserve_seconds=reserve['cpu_seconds'],charged_cpu_seconds=sum(j['cpu_seconds'] for j in jobs)+reserve['cpu_seconds'],
        measured_job_wall_seconds=sum(j['wall_seconds'] for j in jobs),conservative_wall_reserve_seconds=reserve['wall_seconds'],
        charged_job_wall_seconds=sum(j['wall_seconds'] for j in jobs)+reserve['wall_seconds'],
        failed_job_attempts=sum(j['status']!='PASS' for j in jobs),snapshot='after last completed job, including post-job seal reserve')
    for stage,c in cost['stage_costs'].items():
        c['measured_cpu_seconds']=sum(j['cpu_seconds'] for j in jobs if j['stage']==stage)
        c['reserve_cpu_seconds']=reserve['cpu_by_stage'].get(stage,0)
        if c['measured_cpu_seconds']+c['reserve_cpu_seconds']>c['cap']:raise RuntimeError('stage cap exceeded')
    if cost['charged_cpu_seconds']>cost['cpu_cap'] or cost['charged_job_wall_seconds']>cost['wall_cap']:raise RuntimeError('global cap exceeded')
    (ROOT/'C1_RESOURCE_LEDGER.json').write_text(json.dumps(cost,indent=2)+'\n')
    source=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    source['source_files']=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size)
        for folder in ['src','tools','tests','configs'] for p in sorted((ROOT/folder).rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts]
    (ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(source,indent=2)+'\n')
    entries=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        if any(x in {'vendor','__pycache__','.pytest_cache'} for x in p.relative_to(ROOT).parts):continue
        if p.suffix=='.zip' or p.name=='.queue.lock':continue
        entries.append(p)
    dest=ROOT/'C1_MINIMAL_RESEARCH_V2.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in entries:z.write(p,'C1_V2/'+str(p.relative_to(ROOT)))
    receipt=dict(status='LOCAL_DELIVERY_COMPLETE_UNPUBLISHED',date='2026-10-13',
        decision='NO_GO_SCALE_PHYSICS_STATE_MEMORY',archive=dest.name,archive_bytes=dest.stat().st_size,
        archive_members=len(entries),registered_jobs=len(jobs),unfinished_jobs=0,
        physics=cost['physics'],charged_cpu_seconds=cost['charged_cpu_seconds'],charged_job_wall_seconds=cost['charged_job_wall_seconds'],
        local_gpu_seconds=0,b1_changed=False,sha256_checks=False,
        algebra_tests='12 package checks',implementation_tests='77 full suite;1 decoder-specific check after precision fix',
        numerical_sample_precision='7 extreme U0 tails unresolved after float64; primary preserved',
        final_seal_internal_cpu=time.process_time()-start_cpu,final_seal_internal_wall=time.perf_counter()-start_wall,
        charge_scope='internal seal time is covered by3CPU/4wall reserve, not added twice')
    if receipt['final_seal_internal_cpu']>3 or receipt['final_seal_internal_wall']>4:
        raise RuntimeError('seal reserve insufficient, record actual cost before completion')
    (ROOT/'C1_DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
