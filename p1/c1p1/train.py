import argparse
import copy
import json
import time
import numpy as np
import torch
from .common import P1, ORIGINAL, config, dump, zero_physics_guard
from .reference import grid,split,to_z
from .density import FullCovDensity,numpy_parameters
from .mixture import logpdf_z,one
from src.c1.density import Density

def model_for(method,y_dim,s_dim,condition='measurement'):
    if method.startswith('full'):
        return FullCovDensity(condition,y_dim,s_dim)
    kind={'measurement':'mdn','state':'mdn_state','memory':'mdn_memory'}[condition]
    return Density(kind,y_dim,s_dim)

def expected_score(model,Y,S,valbank):
    p=numpy_parameters(model,Y,S)
    z=to_z(valbank['x'])
    jac=(np.log(3)-np.log(valbank['x']+1.5)-np.log(1.5-valbank['x'])).sum(-1)
    values=np.array([-valbank['mass'][i]@(logpdf_z(z,one(p,i))+jac) for i in range(len(Y))])
    return float(values.mean())

def train_one(problem,method,seed,condition='measurement'):
    conf=config();folder=P1/'results'/problem/'models';folder.mkdir(parents=True,exist_ok=True)
    name=f'{method}-{condition}-{seed}'
    if (folder/f'{name}.json').exists() or (folder/f'{name}.pt').exists():raise RuntimeError('completed P1 run not overwritten')
    if condition!='measurement' and problem!='p0':raise RuntimeError('state ablation restricted to cached P0')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    d=split(problem,'train');v=split(problem,'validation')
    Y=torch.tensor(d['inputs_norm']);S=torch.tensor(d['states_norm']);Z=torch.tensor(to_z(d['x']),dtype=torch.float32)
    VY=torch.tensor(v['inputs_norm']);VS=torch.tensor(v['states_norm']);VX=torch.tensor(v['x'],dtype=torch.float32)
    valbank=dict(np.load(P1/f'results/{problem}/validation_teacher.npz'))
    teacher=np.load(P1/f'results/{problem}/teacher_cdf.npy',mmap_mode='r') if method.endswith('distill') else None
    tz=to_z(grid(problem,conf['teacher_orders'][problem])['x']) if teacher is not None else None
    model=model_for(method,Y.shape[1],S.shape[-1],condition)
    optimizer=torch.optim.Adam(model.parameters(),lr=conf['learning_rate'])
    cpu,wall=time.process_time(),time.perf_counter()
    best=np.inf;best_truth=np.inf;best_state=None;history=[];truth_state=None
    for step in range(1,conf['updates'][problem]+1):
        ii=rng.integers(len(Y),size=conf['batch'])
        if teacher is None:targets=Z[ii]
        else:
            uu=rng.random(len(ii))
            nn=np.array([np.searchsorted(teacher[i],u,side='right') for i,u in zip(ii,uu)])
            targets=torch.tensor(tz[nn],dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss=-model.log_prob_z(targets,Y[ii],S[ii]).mean()
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss {name} {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%conf['checkpoint_every']==0:
            model.eval()
            with torch.no_grad():observed=float(-model.log_prob_x(VX,VY,VS).mean())
            expected=expected_score(model,VY,VS,valbank)
            if expected<best:
                best=expected;best_step=step;best_state=copy.deepcopy(model.state_dict())
            if observed<best_truth:
                best_truth=observed;truth_step=step;truth_state=copy.deepcopy(model.state_dict())
            history.append(dict(step=step,training_batch_NLL_z=float(loss.detach()),
                       validation_expected_NLL_x=expected,validation_observed_NLL_x=observed,
                       elapsed_cpu=time.process_time()-cpu))
            model.train()
    metadata=dict(problem=problem,method=method,seed=seed,condition=condition,
                  y_dim=Y.shape[1],state_dim=S.shape[-1],target_parameters=30000)
    torch.save(dict(**metadata,state_dict=best_state),folder/f'{name}.pt')
    torch.save(dict(**metadata,state_dict=truth_state),folder/f'{name}-observed-selection.pt')
    receipt=dict(**metadata,parameters=sum(p.numel() for p in model.parameters()),
         updates=conf['updates'][problem],checkpoint=best_step,observed_selection_checkpoint=truth_step,
         validation_expected_NLL_x=best,validation_best_observed_NLL_x=best_truth,history=history,
         training_objective='joint sample NLL' if teacher is None else 'continuous posterior quadrature weighted NLL (random nodes)',
         validation_selection='shared expected posterior log score; historical observed selector retained separately',
         observations=len(Y),parents=len(np.unique(d['parent'])),topology_input=False,
         reference_input_at_inference=False,physics_calls=0,gpu_seconds=0,
         cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall)
    dump(folder/f'{name}.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['problem','method','condition','seed','checkpoint','validation_expected_NLL_x','cpu_seconds']}),flush=True)
    return receipt

def main(problem,ablation=False):
    zero_physics_guard();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not ablation:
        rows=[]
        for seed in config()['seeds']:
            for method in config()['methods']:rows.append(train_one(problem,method,seed))
        dump(P1/f'results/{problem}/PRIMARY_MODELS_SELECTED.json',dict(status='SELECTED_BEFORE_FRESH_U0_PANEL',
             configurations=[{k:r[k] for k in ['method','seed','condition','checkpoint']} for r in rows],
             old_tests='exposed regression only',new_Maxwell=0,new_B1=0))
    else:
        note=json.loads((P1/'STATE_ABLATION_RECORD.json').read_text())
        if note['problem']!=problem:raise RuntimeError('registered ablation target mismatch')
        rows=[]
        for seed in config()['seeds']:
            for condition in ['state','memory']:rows.append(train_one(problem,note['method'],seed,condition))
        dump(P1/f'results/{problem}/STATE_MODELS_SELECTED.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('problem',choices=['u0','p0']);p.add_argument('--ablation',action='store_true')
    a=p.parse_args();main(a.problem,a.ablation)
