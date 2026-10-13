"""Equal-data, near-equal-capacity conditional density training; no test reads."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .accounting import ROOT,dump
from .data import load_split
from .material import x_to_z
from .density import Density

def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cpu')

def train_one(problem,kind,seed,updates=None,tag='primary'):
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text())
    steps=conf[problem].get('updates',conf['updates']) if updates is None else updates
    folder=ROOT/'results'/problem/'models';folder.mkdir(exist_ok=True,parents=True)
    if (folder.parent/'MODELS_FROZEN.json').exists():raise RuntimeError('training frozen; new registered development experiment required')
    name=f'{kind}-{seed}' if tag=='primary' else f'{kind}-{seed}-{tag}'
    if (folder/f'{name}.pt').exists() or (folder/f'{name}.json').exists():raise RuntimeError('completed training refused')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    train=load_split(problem,'train');val=load_split(problem,'validation')
    y=tensor(train['inputs_norm']);s=tensor(train['states_norm']);z=tensor(x_to_z(train['x']))
    vy=tensor(val['inputs_norm']);vs=tensor(val['states_norm']);vz=tensor(x_to_z(val['x']))
    model=Density(kind,y.shape[1],s.shape[-1],target_parameters=conf['target_parameters']).cpu()
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if abs(params/conf['target_parameters']-1)>conf['parameter_relative_tolerance'] or params>conf['max_parameters']:
        raise RuntimeError(f'parameter matching failed {params}')
    optimizer=torch.optim.Adam(model.parameters(),lr=conf['learning_rate'])
    history=[];best=np.inf;best_state=None;best_step=None
    wall0,cpu0=time.perf_counter(),time.process_time()
    for step in range(1,steps+1):
        ii=rng.integers(len(z),size=conf['batch_size'])
        optimizer.zero_grad(set_to_none=True)
        loss=-model.log_prob_z(z[ii],y[ii],s[ii]).mean()
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite training loss at {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%conf['checkpoint_every']==0 or step==steps:
            model.eval()
            with torch.no_grad():vl=float(-model.log_prob_z(vz,vy,vs).mean())
            if vl<best:best=vl;best_step=step;best_state=copy.deepcopy(model.state_dict())
            history.append(dict(step=step,train_loss=float(loss.detach()),validation_log_score_z=vl,
                                elapsed_cpu=time.process_time()-cpu0))
            model.train()
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        log_score_x=float(-model.log_prob_x(tensor(val['x']),vy,vs).mean())
    torch.save(dict(state_dict=best_state,kind=kind,y_dim=y.shape[1],state_dim=s.shape[-1],
                    target_parameters=conf['target_parameters']),folder/f'{name}.pt')
    receipt=dict(problem=problem,kind=kind,seed=seed,tag=tag,updates_executed=steps,
                 checkpoint_step=best_step,trainable_parameters=params,validation_score_x=log_score_x,
                 validation_score_z=best,history=history,wall_seconds=time.perf_counter()-wall0,
                 cpu_seconds=time.process_time()-cpu0,device='cpu',gpu_seconds=0,
                 topology_labels_used=False,truth_fields_used=False,test_read=False,
                 physics_calls_inside_training=0,context='precomputed at candidate, frozen for generation',
                 timing_scope='optimization + validation + model save; imports/preparation in outer job')
    dump(folder/f'{name}.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['problem','kind','seed','trainable_parameters','validation_score_x','cpu_seconds']}),flush=True)
    return receipt

def train_all(problem):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text());receipts=[]
    kinds=conf[problem].get('models',conf['models'])
    for seed in conf['training_seeds']:
        for kind in kinds:receipts.append(train_one(problem,kind,seed))
    baselines={}
    for seed in conf['training_seeds']:
        pair=[r for r in receipts if r['seed']==seed and r['kind'] in ['mdn','measurement']]
        baselines[str(seed)]=min(pair,key=lambda r:r['validation_score_x'])['kind']
    dump(ROOT/'results'/problem/'MODELS_FROZEN.json',dict(status='FROZEN_BEFORE_TEST',
         seeds=conf['training_seeds'],models=kinds,baseline_selected_by_validation=baselines,
         selection_rule='minimum validation continuous material log score between MDN and measurement flow',
         test_read=False,receipt_paths=[f'models/{r["kind"]}-{r["seed"]}.json' for r in receipts]))
    print(json.dumps(dict(problem=problem,freeze='PASS',baseline=baselines)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('problem',choices=['u0','p0'])
    train_all(p.parse_args().problem)
