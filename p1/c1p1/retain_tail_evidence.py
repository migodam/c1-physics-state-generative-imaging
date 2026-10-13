import json
import numpy as np
import torch
from .common import P1,dump,zero_physics_guard
from .reference import split
from .evaluate import load_model,fresh_u0
from .density import numpy_parameters
from .mixture import one,sample,logpdf_z

def main():
    zero_physics_guard();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    failed=[]
    for file in sorted((P1/'results').glob('*/evaluation/*-OBSERVATIONS.jsonl')):
        for line in file.read_text().splitlines():
            row=json.loads(line)
            if row['boundary_fraction']>0:failed.append(row)
    models={};datasets={};retained=[]
    for r in failed:
        pr,pa,method,seed,condition=r['problem'],r['panel'],r['method'],r['seed'],r['condition']
        key=(pr,method,seed,condition)
        if key not in models:models[key]=load_model(pr,method,seed,condition)
        if (pr,pa) not in datasets:datasets[pr,pa]=fresh_u0() if pa=='fresh_final' else split(pr,'test')
        d=datasets[pr,pa];i=r['observation']
        p=one(numpy_parameters(models[key],d['inputs_norm'][i:i+1],d['states_norm'][i:i+1]),0)
        x,z=sample(p,2048,np.random.default_rng(261013600+seed+i))
        ix=np.flatnonzero((np.abs(x)>=1.5).any(1))
        retained.append(dict(problem=pr,panel=pa,method=method,condition=condition,seed=seed,observation=i,
            original_failed_count=round(r['boundary_fraction']*2048),reproduced_failed_count=len(ix),
            generator_seed=261013600+seed+i,
            draws=[dict(index=int(j),z=z[j].tolist(),decoded_x=x[j].tolist(),
                   log_density_z=float(logpdf_z(z[j:j+1],p)[0]),log_density_decoded_x='-inf: rounded endpoint outside open support') for j in ix],
            action='preserve failed original law; no clipping/resampling or physical call'))
    dump(P1/'results/TAIL_FAILURE_DRAWS.json',dict(scope='same frozen models and exact original evaluation RNG, only already recorded failed observations',
        new_models=0,new_data=0,new_physics=0,results_unchanged=True,records=retained))
    print(json.dumps(dict(failed_observations=len(failed),retained_draws=sum(len(r['draws']) for r in retained),
                         new_training=0,new_physics=0)))

if __name__=='__main__':main()
