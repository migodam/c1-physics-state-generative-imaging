"""Correct query timing: exclude diagnostic ledger-summary scans, retain backend costs."""
import json
import time
import numpy as np
import torch
from .accounting import ROOT,dump
from .data import load_split
from .evaluate import load_model,tensor
from .material import z_to_x
from .problems import DipoleProblem

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    path=ROOT/'results/p0'
    if (path/'QUERY_TIMING_CORRECTED.json').exists():raise RuntimeError('completed timing refused')
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text());freeze=json.loads((path/'MODELS_FROZEN.json').read_text())
    d=load_split('p0','test');Y=tensor(d['inputs_norm']);S=tensor(d['states_norm']);norm=np.load(path/'normalization.npz')
    dip=DipoleProblem();dip.scale=np.load(path/'scale.npy');kinds=freeze['models']
    models={k:load_model('p0',k,conf['training_seeds'][0]) for k in kinds};rows=[]
    with torch.no_grad():
        for repeat in range(3):
            for kind in kinds[repeat:]+kinds[:repeat]:
                before=dip.ledger.summary();w,c=time.perf_counter(),time.process_time();pc=0.
                if kind in ['mdn_state','mdn_memory']:
                    c0=time.process_time();ss=dip.context(d['y'][0],d['sigma'][0]);pc=time.process_time()-c0
                    ss=tensor(((ss-norm['sm'])/norm['ss'])[None])
                else:ss=S[:1]
                gen=torch.Generator(device='cpu').manual_seed(261013800+repeat)
                x=z_to_x(models[kind].sample_z(Y[:1],ss,1024,generator=gen).numpy())
                eps=2+x+1j*.02
                elapsed_w,elapsed_c=time.perf_counter()-w,time.process_time()-c
                # Audit aggregation is outside the actual inference interval.
                after=dip.ledger.summary()
                rows.append(dict(model=kind,repeat=repeat,wall_seconds=elapsed_w,cpu_seconds=elapsed_c,
                    context_cpu=pc,material_frequency=after['used']-before['used'],
                    forward_rhs=after['forward_rhs']-before['forward_rhs'],
                    derivative_rhs=after['derivative_rhs']-before['derivative_rhs'],samples=eps.shape[1],
                    first_backend_use=(kind=='mdn_state' and repeat==0)))
    summary={k:dict(median_cpu_seconds=float(np.median([r['cpu_seconds'] for r in rows if r['model']==k])),
                    median_wall_seconds=float(np.median([r['wall_seconds'] for r in rows if r['model']==k])),
                    mf_per_new_query=2 if k in ['mdn_state','mdn_memory'] else 0,
                    RHS_per_new_query=12 if k in ['mdn_state','mdn_memory'] else 0) for k in kinds}
    dump(path/'QUERY_TIMING_CORRECTED.json',dict(rows=rows,summary=summary,new_material_frequency=12,
        reason='original DIAGNOSTICS timing includes post-query ledger aggregation; retain original as failed timing receipt',
        scope='in-memory model and preassembled fixed geometry; candidate contexts not cached; decode included; interpreter/model setup outside query median',
        full_pipeline_cost='C1_RESOURCE_LEDGER.json; setup, reference, states, training and both timing attempts remain charged',
        quality_at_required_tolerance='FAIL; no cost-advantage promotion',gpu_seconds=0))
    dump(ROOT/'results/PHYSICS_SUMMARY.json',dip.ledger.summary());print(json.dumps(summary))

if __name__=='__main__':main()
