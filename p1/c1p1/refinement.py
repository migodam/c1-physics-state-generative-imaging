"""Empirical polynomial interpolation of existing P0 banks; no forward calls."""
import numpy as np
from numpy.polynomial.chebyshev import chebvander
from .common import P1,config,dump,zero_physics_guard
from .reference import grid,split,posterior,reference_features

def coefficients(g,degree):
    n=int(np.sqrt(len(g['x'])))
    axis=g['x'].reshape(n,n,2)[:,0,0]/1.5
    v=chebvander(axis,degree)
    f=g['f'].reshape(n,n,4)
    a=np.linalg.lstsq(v,f.reshape(n,-1),rcond=None)[0].reshape(degree+1,n,4)
    c=np.linalg.lstsq(v,a.transpose(1,0,2).reshape(n,-1),rcond=None)[0]
    return c.reshape(degree+1,degree+1,4).transpose(1,0,2)

def responses(x,c):
    vx=chebvander(x[:,0]/1.5,len(c)-1)
    vy=chebvander(x[:,1]/1.5,len(c)-1)
    return np.einsum('na,nb,abk->nk',vx,vy,c,optimize=True)

def refined_grid(order,bank=64,degree=12):
    g=grid('u0',order)
    c=np.load(P1/f'results/p0/response-cheb-{bank}-{degree}.npy')
    g['f']=responses(g['x'],c)
    return g

def main():
    zero_physics_guard();folder=P1/'results/p0'
    banks={n:grid('p0',n) for n in [40,64]};errors=[]
    for n in [40,64]:
        for degree in [8,12]:
            c=coefficients(banks[n],degree)
            np.save(folder/f'response-cheb-{n}-{degree}.npy',c)
            other=banks[104-n]
            diff=responses(other['x'],c)-other['f']
            errors.append(dict(fit_bank=n,degree=degree,independent_bank=104-n,
               max_absolute_response_error=float(np.abs(diff).max()),rms_response_error=float(np.sqrt(np.mean(diff*diff)))))
    gs=[refined_grid(96,40),refined_grid(96,64),refined_grid(128,64)]
    rows=[]
    for name in ['train','validation']:
        d=split('p0',name)
        for i in range(len(d['y'])):
            refs=[posterior(g,d['y'][i],d['sigma'][i],problem='p0') for g in gs]
            old=posterior(banks[64],d['y'][i],d['sigma'][i],problem='p0')
            a,b=[reference_features(g,r) for g,r in zip(gs[1:],refs[1:])]
            rows.append(dict(split=name,observation=i,
                 independent_bank_event_difference=float(np.max(np.abs(refs[0]['events']-refs[1]['events']))),
                 old64_event_difference=float(np.max(np.abs(old['events']-refs[-1]['events']))),
                 old64_evidence_difference=abs(old['logZ']-refs[-1]['logZ']),
                 refined_event_difference=float(np.max(np.abs(refs[1]['events']-refs[-1]['events']))),
                 refined_evidence_difference=abs(refs[1]['logZ']-refs[-1]['logZ']),
                 refined_cdf_difference=float(np.max(np.abs(np.asarray(a['cdf'])-b['cdf']))),
                 refined_tail_difference=float(np.max(np.abs(np.asarray(a['tail'])-b['tail']))),
                 refined_quantile_difference=float(np.max(np.abs(np.asarray(a['quantiles'])-b['quantiles']))),
                 refined_cdf_nonmonotone=b['cdf_nonmonotone_max']))
    summary={k:max(r[k] for r in rows) for k in rows[0] if 'difference' in k or 'nonmonotone' in k}
    dump(folder/'REFERENCE_NUMERICAL_REFINEMENT.json',dict(
        method='degree12 tensor Chebyshev response approximation, both banks independently fitted',
        original_reference_retained=True,original_teacher_order=64,primary_statistical_target_unchanged=True,
        approximation_scope='empirical bank cross-check and integral refinement; not full-solver uniform certificate',
        new_Maxwell=0,response_errors=errors,summary=summary,per_observation=rows))
    print(__import__('json').dumps(dict(response_errors=errors,summary=summary)),flush=True)

if __name__=='__main__':main()
