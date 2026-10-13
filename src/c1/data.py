"""Independent simulated parents, frozen candidate contexts, continuous references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import logsumexp
from .accounting import ROOT, dump, PhysicsLedger
from .material import prior_sample,prior_logpdf,quadrant
from .problems import DipoleProblem,toy_forward,toy_context
from .probability import gaussian_loglike

def quadrature(order):
    """order is total nodes per axis, split at 0 for fixed offline events."""
    if order%2:raise ValueError('need even total order')
    t,w=leggauss(order//2)
    axis=np.r_[.75*t-.75,.75*t+.75]; weights=np.r_[.75*w,.75*w]
    xx,yy=np.meshgrid(axis,axis,indexing='ij');ww,vv=np.meshgrid(weights,weights,indexing='ij')
    return np.c_[xx.ravel(),yy.ravel()],(ww*vv).ravel()

def reference(grid,y,sigma,kappa=0.,problem='u0'):
    x=grid['x'];w=grid['w']
    pred=toy_forward(x,kappa) if problem=='u0' else grid['f']
    logdensity=prior_logpdf(x)+gaussian_loglike(pred,y,sigma)
    logZ=logsumexp(logdensity+np.log(w))
    probability=np.exp(logdensity+np.log(w)-logZ)
    event=np.bincount(quadrant(x),weights=probability,minlength=4)
    return dict(weights=probability,logdensity=logdensity-logZ,events=event,logZ=logZ)

def independent_toy_reference():
    # Fixed cases selected before data/training, including coupled and ambiguous cases.
    cases=[([1.,.8,0.],0.,.12),([1.7,.4,.2],.25,.25),([1.2,.9,-.3],.6,.25)]
    x,w=quadrature(96);grid={'x':x,'w':w};rows=[]
    for y,k,s in cases:
        y=np.asarray(y);masses=[]
        for a,b in [(-1.5,0.),(0.,1.5)]:
            for c,d in [(-1.5,0.),(0.,1.5)]:
                def outer(xx):
                    def inner(yy):
                        z=np.array([xx,yy]);return np.exp(prior_logpdf(z)+gaussian_loglike(toy_forward(z,k),y,s))
                    return quad(inner,c,d,epsabs=1e-12,epsrel=1e-10,limit=160)[0]
                masses.append(quad(outer,a,b,epsabs=1e-12,epsrel=1e-10,limit=160)[0])
        masses=np.array(masses);masses/=masses.sum()
        p=reference(grid,y,s,k)['events']
        rows.append(dict(y=y.tolist(),kappa=k,sigma=s,adaptive_events=masses.tolist(),
                         max_event_difference=float(np.max(np.abs(p-masses)))))
    return dict(cases=rows,max_difference=max(r['max_event_difference'] for r in rows),
                gate='PASS' if max(r['max_event_difference'] for r in rows)<1e-7 else 'FAIL',
                kind='independent adaptive vs split Gauss empirical check')

def prepare(problem):
    conf=json.loads((ROOT/'configs/protocol_v2.json').read_text());cfg=conf[problem]
    path=ROOT/'results'/problem
    if (path/'PREPARED.json').exists():raise RuntimeError('completed preparation refused; no ledger reset')
    path.mkdir(exist_ok=True,parents=True)
    wall0,cpu0=time.perf_counter(),time.process_time()
    dipole=DipoleProblem() if problem=='p0' else None
    if dipole:dipole.set_scale();np.save(path/'scale.npy',dipole.scale)
    times={};metadata={};parent_manifest={}
    for split in ['train','validation','test']:
        rng=np.random.default_rng(cfg['seeds'][split]);n=cfg['parents'][split];reps=cfg['noise_replicates'][split]
        px=prior_sample(rng,n);x=np.repeat(px,reps,axis=0);parents=np.repeat(np.arange(n),reps)
        # Truth fields are immediately discarded. Only F and supervised material targets stored.
        t,c=time.perf_counter(),time.process_time()
        if dipole:
            pf=np.stack([dipole.forward(a,kind='truth_data_'+split)['f'] for a in px]);kappa=np.zeros(len(x))
            means=np.repeat(pf,reps,axis=0)
        else:
            kappa=rng.choice(cfg['kappa'],len(x));means=toy_forward(x,kappa)
        sigma=rng.choice(cfg['sigma'],len(x));y=means+rng.normal(size=means.shape)*sigma[:,None]
        truth_time=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
        t,c=time.perf_counter(),time.process_time()
        if dipole:states=np.stack([dipole.context(a,s) for a,s in zip(y,sigma)])
        else:states=np.stack([toy_context(a,k,s) for a,k,s in zip(y,kappa,sigma)])
        state_time=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c,
                        material_frequency_calls=2*len(x) if dipole else 0)
        inputs=np.c_[y,np.log(sigma)] if dipole else np.c_[y,kappa,np.log(sigma)]
        np.savez_compressed(path/f'{split}.npz',x=x,y=y,inputs=inputs,states=states,
                            sigma=sigma,kappa=kappa,parent=parents)
        parent_manifest[split]=dict(seed=cfg['seeds'][split],parent_ids=[f'{problem}-{split}-{j:04d}' for j in range(n)],
                                    noise_replicates=reps,material_parents=n,observations=len(x))
        times[split]=dict(truth_generation=truth_time,candidate_context_generation=state_time)
        if split=='train':
            ym=inputs.mean(axis=0);ys=np.maximum(inputs.std(axis=0),1e-5)
            sm=states.mean(axis=(0,1));ss=np.maximum(states.std(axis=(0,1)),1e-4)
            np.savez(path/'normalization.npz',ym=ym,ys=ys,sm=sm,ss=ss)
    t,c=time.perf_counter(),time.process_time()
    for order in cfg['quadrature_orders']:
        x,w=quadrature(order)
        kwargs=dict(x=x,w=w)
        if dipole:kwargs['f']=np.stack([dipole.forward(a,kind='reference_'+str(order))['f'] for a in x])
        np.savez_compressed(path/f'grid_{order}.npz',**kwargs)
    times['reference_preparation']=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
    if problem=='u0':dump(path/'INDEPENDENT_REFERENCE.json',independent_toy_reference())
    dump(path/'PARENT_SPLITS.json',parent_manifest)
    dump(path/'PREPARED.json',dict(problem=problem,prior_id=conf['prior']['id'],
                                 solver_id=dipole.solver_id if dipole else 'square-plus-weak-linear-v1',
                                 truth_fields_stored=False,context_function_accepts_truth=False,
                                 topology_input=False,baseline_inputs='full measurement and noise/acquisition metadata',
                                 times=times,wall_seconds=time.perf_counter()-wall0,cpu_seconds=time.process_time()-cpu0,
                                 cpu_nested_in_job=True,sha256_checks=False,
                                 support='restricted continuous two-coordinate prior, not full voxel posterior',
                                 source_count=2 if dipole else None))
    if dipole:dump(ROOT/'results/PHYSICS_SUMMARY.json',dipole.ledger.summary())
    print(json.dumps(dict(prepared=problem,physics_used=dipole.ledger.used if dipole else 0)))

def load_split(problem,split,require_frozen=True):
    if split=='test':
        freeze_path=ROOT/'results'/problem/'MODELS_FROZEN.json'
        if not require_frozen or not freeze_path.exists():
            raise RuntimeError('test evaluation forbidden before complete model freeze')
        freeze=json.loads(freeze_path.read_text())
        if freeze.get('status')!='FROZEN_BEFORE_TEST':
            raise RuntimeError('test evaluation forbidden before complete model freeze')
        for receipt in freeze.get('receipt_paths',[]):
            if not (freeze_path.parent/receipt).exists() or not (freeze_path.parent/receipt.replace('.json','.pt')).exists():
                raise RuntimeError('frozen model receipt/checkpoint missing')
    path=ROOT/'results'/problem
    d=dict(np.load(path/f'{split}.npz'));n=np.load(path/'normalization.npz')
    d['inputs_norm']=((d['inputs']-n['ym'])/n['ys']).astype(np.float32)
    d['states_norm']=((d['states']-n['sm'])/n['ss']).astype(np.float32)
    return d

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('problem',choices=['u0','p0'])
    prepare(p.parse_args().problem)
