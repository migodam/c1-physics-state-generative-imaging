"""Registered offline family-fit and prior/likelihood diagnostics, no physics."""
import json
import numpy as np
import torch
from scipy.special import logsumexp
from .common import P1,config,dump,zero_physics_guard
from .reference import grid,split,posterior,logprior,to_z
from .refinement import refined_grid
from .mixture import logpdf_z,logpdf_x,events
from .evaluate import load_model
from .density import numpy_parameters

def em(z,mass,full,seed,steps=60):
    rng=np.random.default_rng(seed)
    mu=z[rng.choice(len(z),8,p=mass)].copy()
    p=dict(weights=np.full(8,.125),means=mu,stds=np.full((8,2),.8),rho=np.zeros(8))
    best_score=np.inf;best=None;history=[]
    for step in range(steps):
        lp=logpdf_z(z,p,components=True)
        total=logsumexp(lp,axis=-1)
        score=float(-mass@total)
        if score<best_score:
            best_score=score;best={k:v.copy() for k,v in p.items()}
        resp=mass[:,None]*np.exp(lp-total[:,None])
        nk=resp.sum(0);den=np.maximum(nk,1e-15)
        mu=resp.T@z/den[:,None]
        u=z[:,None,:]-mu
        variance=np.einsum('nk,nki->ki',resp,u*u)/den[:,None]
        std=np.clip(np.sqrt(np.maximum(variance,0)),np.exp(-5),np.exp(5))
        cross=np.einsum('nk,nk,nk->k',resp,u[...,0],u[...,1])/den
        rho=np.clip(cross/(std[:,0]*std[:,1]),-.994999,.994999) if full else np.zeros(8)
        p=dict(weights=nk/nk.sum(),means=mu,stds=std,rho=rho)
        history.append(score)
    return best,history

def main():
    zero_physics_guard();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    oracle=[];mechanisms=[]
    for problem in ['u0','p0']:
        d=split(problem,'validation');g=grid(problem,64)
        gf=grid('u0',128) if problem=='u0' else refined_grid(128)
        prior_event=np.bincount(((g['x'][:,0]>=0).astype(int)*2+(g['x'][:,1]>=0)).astype(int),
                  weights=np.exp(logprior(g['x']))*g['w'],minlength=4)
        for i in range(len(d['y'])):
            r=posterior(gf,d['y'][i],d['sigma'][i],d['kappa'][i],problem)
            flat=posterior(gf,d['y'][i],d['sigma'][i],d['kappa'][i],problem,flat_prior=True)
            pe=r['events'];positive=pe>0
            mechanisms.append(dict(problem=problem,observation=i,sigma=float(d['sigma'][i]),kappa=float(d['kappa'][i]),
                posterior_prior_KL=float(r['mass']@(r['logpdf']-logprior(gf['x']))),
                event_posterior_prior_KL=float(np.sum(pe[positive]*np.log(pe[positive]/prior_event[positive]))),
                true_prior_events=pe.tolist(),counterfactual_flat_prior_events=flat['events'].tolist(),
                flat_prior_event_departure_from_quarter=float(np.max(np.abs(flat['events']-.25))),
                prior_event_change=float(np.max(np.abs(pe-flat['events']))),
                scope='counterfactual diagnostic only; actual training/reference prior unchanged'))
        for i in range(8):
            r=posterior(g,d['y'][i],d['sigma'][i],d['kappa'][i],problem)
            rf=posterior(gf,d['y'][i],d['sigma'][i],d['kappa'][i],problem)
            z=to_z(g['x'])
            for full in [False,True]:
                attempts=[]
                for start in range(2):
                    p,h=em(z,r['mass'],full,261013710+i+100*start)
                    fit=float(r['mass']@(r['logpdf']-logpdf_x(g['x'],p)))
                    independent=float(rf['mass']@(rf['logpdf']-logpdf_x(gf['x'],p)))
                    attempts.append(dict(start=start,fit_KL=fit,independent_KL=independent,training_NLL_z=h,
                          event_MAE=float(np.abs(events(p)-rf['events']).mean())))
                selected=min(attempts,key=lambda a:a['fit_KL'])
                method='full_distill' if full else 'diag_distill'
                amortized=[]
                for seed in config()['seeds']:
                    m=load_model(problem,method,seed)
                    p={k:v[0] for k,v in numpy_parameters(m,d['inputs_norm'][i:i+1],d['states_norm'][i:i+1]).items()}
                    amortized.append(dict(seed=seed,KL=float(rf['mass']@(rf['logpdf']-logpdf_x(gf['x'],p)))))
                oracle.append(dict(problem=problem,validation_observation=i,covariance='full' if full else 'diagonal',
                    **selected,all_starts=attempts,amortized_distill=amortized,
                    scope='offline non-amortized reference fit, not a deployable generator or proof of global family optimum'))
    dump(P1/'results/FAMILY_CAPACITY_DIAGNOSTIC.json',dict(round=1,observations='fixed first8 validation per problem',
        starts=2,steps=60,new_Maxwell=0,topology_initialization=False,results=oracle))
    dump(P1/'results/PRIOR_LIKELIHOOD_DIAGNOSTIC.json',mechanisms)
    for problem in ['u0','p0']:
        for cov in ['diagonal','full']:
            rr=[r for r in oracle if r['problem']==problem and r['covariance']==cov]
            print(json.dumps(dict(problem=problem,covariance=cov,non_amortized_KL=np.mean([r['independent_KL'] for r in rr]),
                  amortized_KL=np.mean([a['KL'] for r in rr for a in r['amortized_distill']]))),flush=True)

if __name__=='__main__':main()
