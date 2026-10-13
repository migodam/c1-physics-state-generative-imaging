"""Continuous prior/likelihood quadrature; P0 only reads existing responses."""
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander, legint, legval
from scipy.special import logsumexp, expit
from .common import ORIGINAL

BOUND = 1.5
TILT = np.array([.4, -.2])

def logprior(x):
    x = np.asarray(x, dtype=np.float64)
    value = (x*TILT-np.log(2*np.sinh(BOUND*TILT)/TILT)).sum(-1)
    return np.where((np.abs(x)<BOUND).all(-1), value, -np.inf)

def sampleprior(rng, n):
    return np.log(np.exp(-BOUND*TILT)+rng.random((n,2))*(np.exp(BOUND*TILT)-np.exp(-BOUND*TILT)))/TILT

def to_z(x):
    x = np.asarray(x, dtype=np.float64)
    if np.any(np.abs(x)>=BOUND):
        raise ValueError('open material support required; no clipping')
    return np.log(x+BOUND)-np.log(BOUND-x)

def to_x(z):
    return 3*expit(np.asarray(z, dtype=np.float64))-BOUND

def event(x):
    return (x[...,0]>=0).astype(int)*2+(x[...,1]>=0).astype(int)

def toy(x, kappa):
    return np.stack([x[...,0]**2,x[...,1]**2,np.asarray(kappa)*(x[...,0]+.5*x[...,1])],-1)

def grid(problem, order):
    if problem == 'p0':
        if order not in (40,64):
            raise ValueError('P0 forbids new forward-bank orders')
        return dict(np.load(ORIGINAL/f'results/p0/grid_{order}.npz'))
    t,w = leggauss(order//2)
    axis = np.r_[.75*t-.75,.75*t+.75]
    aw = np.r_[.75*w,.75*w]
    xx,yy = np.meshgrid(axis,axis,indexing='ij')
    ww,vv = np.meshgrid(aw,aw,indexing='ij')
    return {'x':np.c_[xx.ravel(),yy.ravel()], 'w':(ww*vv).ravel()}

def split(problem, name):
    d = dict(np.load(ORIGINAL/f'results/{problem}/{name}.npz'))
    n = np.load(ORIGINAL/f'results/{problem}/normalization.npz')
    d['inputs_norm'] = ((d['inputs']-n['ym'])/n['ys']).astype(np.float32)
    d['states_norm'] = ((d['states']-n['sm'])/n['ss']).astype(np.float32)
    return d

def posterior(g, y, sigma, kappa=0., problem='u0', flat_prior=False):
    pred = g['f'] if problem=='p0' else toy(g['x'],kappa)
    ll = -.5*(((pred-y)/sigma)**2).sum(-1)-len(y)*np.log(sigma*np.sqrt(2*np.pi))
    lp = np.full(len(ll),-2*np.log(3.)) if flat_prior else logprior(g['x'])
    density = lp+ll
    logZ = logsumexp(density+np.log(g['w']))
    mass = np.exp(density+np.log(g['w'])-logZ)
    return {'mass':mass, 'logpdf':density-logZ, 'logZ':float(logZ),
            'events':np.bincount(event(g['x']),weights=mass,minlength=4)}

def marginal_cdf(g, mass, thresholds):
    """Integrate interpolated marginal density, not a step CDF of Gauss nodes.

    The interpolation has no guarantee of positivity; callers must retain any
    negative/nonmonotone/above-one result and report order refinement.
    """
    order = int(np.sqrt(len(g['x'])))
    half = order//2
    t,w = leggauss(half)
    aw = np.r_[.75*w,.75*w]
    pm = mass.reshape(order,order)
    marginals = np.stack([pm.sum(1)/aw,pm.sum(0)/aw])
    v = legvander(t,half-1)
    factors = (2*np.arange(half)+1)/2
    th = np.asarray(thresholds)
    out = np.zeros((2,len(th)))
    for j in range(2):
        for k,(lo,hi) in enumerate([(-1.5,0.),(0.,1.5)]):
            vals = marginals[j,k*half:(k+1)*half]
            coeff = factors*(v.T@(w*vals))
            integral = legint(coeff)
            scaled = np.clip(2*(th-lo)/(hi-lo)-1,-1,1)
            out[j] += .75*(legval(scaled,integral)-legval(-1.,integral))
    return out

def reference_features(g,r):
    x = g['x'];p = r['mass']
    thresholds = np.linspace(-1.5,1.5,61)
    cdf = marginal_cdf(g,p,thresholds)
    mean = p@x
    second = p@(x*x)
    qth = np.linspace(-1.5,1.5,601)
    dense = marginal_cdf(g,p,qth)
    # Quantiles are provisional if this reference is nonmonotonic; no repair.
    quantiles = np.array([[np.interp(q,dense[j],qth) for q in [.05,.5,.95]] for j in range(2)])
    tail_cdf = marginal_cdf(g,p,[-1.4,1.4])
    tail = tail_cdf[:,0]+1-tail_cdf[:,1]
    shape = []
    ids = event(x)
    for k in range(4):
        w = p*(ids==k);m=w.sum()
        mu = w@x/max(m,1e-300)
        cov = np.einsum('n,ni,nj->ij',w,x-mu,x-mu)/max(m,1e-300)
        shape.append({'mass':float(m),'mean':mu.tolist(),'cov':cov.tolist()})
    return dict(events=r['events'].tolist(),mean=mean.tolist(),variance=(second-mean*mean).tolist(),
                cdf=cdf.tolist(),thresholds=thresholds.tolist(),quantiles=quantiles.tolist(),tail=tail.tolist(),
                within_region=shape,cdf_min=float(dense.min()),cdf_max=float(dense.max()),
                cdf_nonmonotone_max=float(max(0.,-np.min(np.diff(dense,axis=1)))),
                posterior_prior_KL=float(p@(r['logpdf']-logprior(x))))

