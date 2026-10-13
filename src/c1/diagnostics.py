"""No training: reference-score, same-family, input-reliance and cold-cost checks."""
import json
import time
import numpy as np
import torch
from scipy.special import logsumexp
from .accounting import ROOT,dump,PhysicsLedger
from .data import load_split,reference,quadrature
from .evaluate import load_model,tensor,paired_bootstrap
from .material import prior_logpdf,quadrant
from .probability import gaussian_loglike
from .problems import DipoleProblem,toy_forward

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text());path=ROOT/'results/p0'
    if (path/'DIAGNOSTICS.json').exists():raise RuntimeError('completed diagnostics refused')
    data=load_split('p0','test');val=load_split('p0','validation');Y=tensor(data['inputs_norm']);S=tensor(data['states_norm'])
    vy=tensor(val['inputs_norm']);vs=tensor(val['states_norm']);vx=tensor(val['x'])
    rows=[json.loads(s) for s in (path/'OBSERVATION_METRICS.jsonl').read_text().splitlines()]
    freeze=json.loads((path/'MODELS_FROZEN.json').read_text());grids=[dict(np.load(path/f'grid_{n}.npz')) for n in conf['p0']['quadrature_orders']]
    coarse=grids[0];gx=tensor(coarse['x']);refs=[reference(coarse,y,s,problem='p0') for y,s in zip(data['y'],data['sigma'])]
    score_refinement={};feature_reliance={};paired={}
    rng=np.random.default_rng(261013789);perm=rng.permutation(len(vs))
    with torch.no_grad():
        for seed in conf['training_seeds']:
            for kind in freeze['models']:
                key=f'{kind}-{seed}';model=load_model('p0',kind,seed);C=model.c(Y,S)
                group=[r for r in rows if r['seed']==seed and r['model']==kind];values=[]
                for i,r in enumerate(group):
                    logq=model.log_prob_x_context(gx,C[i:i+1].expand(len(gx),-1)).numpy().astype(float)
                    values.append(float(refs[i]['weights']@(refs[i]['logdensity']-logq)))
                diff=np.array([r['KL_reference'] for r in group])-values
                score_refinement[key]=dict(mean_KL_coarse=float(np.mean(values)),mean_KL_fine=float(np.mean([r['KL_reference'] for r in group])),
                    mean_fine_minus_coarse=float(np.mean(diff)),max_abs_observation_difference=float(np.max(np.abs(diff))),
                    warning='Score integration is unstable at the desired0.02 scale' if abs(np.mean(diff))>.02 else 'mean refinement within0.02; empirical only')
                if kind in ['mdn_state','mdn_memory']:
                    feature_reliance[key]=dict(validation_log_score_original=float(-model.log_prob_x(vx,vy,vs).mean()),
                        validation_log_score_zero_state=float(-model.log_prob_x(vx,vy,torch.zeros_like(vs)).mean()),
                        validation_log_score_permuted_state=float(-model.log_prob_x(vx,vy,vs[perm]).mean()),
                        scope='post-freeze reliance diagnostic, not newly trained ablation or independent likelihood')
            for kind in ['mdn_state','mdn_memory']:
                for metric in ['truth_log_score','KL_reference','event_mass_MAE','expected_Brier_excess']:
                    a=[r[metric] for r in rows if r['seed']==seed and r['model']==kind]
                    for base in ['mdn',freeze['baseline_selected_by_validation'][str(seed)]]:
                        b=[r[metric] for r in rows if r['seed']==seed and r['model']==base]
                        key=f'{kind}-vs-{base}-{seed}-{metric}'
                        paired[key]=paired_bootstrap(np.array(a)-b,data['parent'],conf['bootstrap_seed'],conf['bootstrap_repeats'])
    # Cold query: models already in memory; physics contexts genuinely recomputed.
    dip=DipoleProblem();dip.scale=np.load(path/'scale.npy');norm=np.load(path/'normalization.npz');timing=[]
    first=conf['training_seeds'][0]
    models={kind:load_model('p0',kind,first) for kind in freeze['models']}
    with torch.no_grad():
        for repeat in range(3):
            for kind in freeze['models'][repeat:]+freeze['models'][:repeat]:
                model=models[kind];before=dip.ledger.summary();wall,cpu=time.perf_counter(),time.process_time()
                context_cpu=0.
                if kind in ['mdn_state','mdn_memory']:
                    c=time.process_time();ss=dip.context(data['y'][0],data['sigma'][0]);context_cpu=time.process_time()-c
                    ss=tensor(((ss-norm['sm'])/norm['ss'])[None])
                else:ss=S[:1]
                generator=torch.Generator(device='cpu').manual_seed(261013800+repeat)
                model.sample_z(Y[:1],ss,1024,generator=generator)
                after=dip.ledger.summary()
                timing.append(dict(model=kind,repeat=repeat,wall_seconds=time.perf_counter()-wall,
                    cpu_seconds=time.process_time()-cpu,physics_context_cpu=context_cpu,
                    charged_material_frequency=after['used']-before['used'],
                    charged_forward_rhs=after['forward_rhs']-before['forward_rhs'],
                    charged_derivative_rhs=after['derivative_rhs']-before['derivative_rhs'],
                    samples=1024,scope='model in memory, uncached candidate contexts for physical models'))
    # Deliberately wrong rules, on a fixed analytical case, with known one-likelihood target.
    x,w=quadrature(128);y=np.array([1.2,.9,-.3]);sigma=.25;k=.6
    ll=gaussian_loglike(toy_forward(x,k),y,sigma);lp=prior_logpdf(x)+np.log(w)
    def event_mass(logw):
        weights=np.exp(logw-logsumexp(logw));return np.bincount(quadrant(x),weights=weights,minlength=4)
    correct=event_mass(lp+ll);reused=event_mass(lp+3*ll)
    wrong_copy=event_mass(lp+ll+np.where(quadrant(x)==1,np.log(5.),0.))
    controls=dict(correct_events=correct.tolist(),repeated_likelihood_wrong_events=reused.tolist(),
                  five_copies_without_mass_split_wrong_events=wrong_copy.tolist(),
                  repeat_max_mass_error=float(np.max(np.abs(correct-reused))),
                  duplicate_max_mass_error=float(np.max(np.abs(correct-wrong_copy))),
                  meaning='intentional invalid controls; never used by the trained generator')
    dump(path/'DIAGNOSTICS.json',dict(score_refinement=score_refinement,validation_state_reliance=feature_reliance,
         paired_comparisons=paired,cold_query_timing=timing,wrong_rule_controls=controls,
         iso_parameter_iso_data=True,wall_cost_matching='not established: actual costs and quality differ; no Pareto promotion',
         local_gpu_seconds=0,additional_training_runs=0,additional_material_frequency=12))
    dump(ROOT/'results/PHYSICS_SUMMARY.json',dip.ledger.summary())
    print(json.dumps(dict(diagnostics='PASS',new_physics=12,charged_total=dip.ledger.summary()['used'],score_refinement=score_refinement)))

if __name__=='__main__':main()
