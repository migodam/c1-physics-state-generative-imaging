"""Float64 Gaussian-mixture evaluation and deterministic probability summaries."""
from functools import lru_cache
import numpy as np
from scipy.special import ndtr, logsumexp
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.hermite import hermgauss
from .reference import to_z, to_x

def one(p,i):
    return {key:value[i] for key,value in p.items()}

def logpdf_z(z,p,components=False):
    z=np.asarray(z,dtype=np.float64)
    u=(z[:,None,:]-p['means'])/p['stds']
    rho=p['rho'];v=1-rho*rho
    quad=u[...,0]**2+(u[...,1]-rho*u[...,0])**2/v
    lp=-.5*quad-np.log(p['stds']).sum(-1)-.5*np.log(v)-np.log(2*np.pi)
    with np.errstate(divide='ignore'):
        lp=lp+np.log(p['weights'])
    return lp if components else logsumexp(lp,axis=-1)

def logpdf_x(x,p):
    x=np.asarray(x,dtype=np.float64)
    inside=(np.abs(x)<1.5).all(-1)
    out=np.full(len(x),-np.inf)
    if np.any(inside):
        a=x[inside]
        jac=(np.log(3)-np.log(a+1.5)-np.log(1.5-a)).sum(-1)
        out[inside]=logpdf_z(to_z(a),p)+jac
    return out

@lru_cache(None)
def leg(order):return leggauss(order)

def events(p,order=64):
    """P(Z1<0,Z2<0) via Plackett's integral, then other sign events.

    d Phi2(a,b,rho)/d rho is the bivariate normal pdf at (a,b).
    rho=sin(theta) removes its endpoint square-root denominator.
    """
    a,b=(-p['means']/p['stds']).T
    fa,fb=ndtr(a),ndtr(b)
    rho=p['rho'];t,w=leg(order)
    angle=np.arcsin(rho)
    theta=.5*angle[:,None]*(t[None,:]+1)
    numerator=a[:,None]**2-2*a[:,None]*b[:,None]*np.sin(theta)+b[:,None]**2
    integrand=np.exp(-numerator/(2*np.cos(theta)**2))
    p00=fa*fb+.5*angle*(integrand@w)/(2*np.pi)
    comp=np.stack([p00,fa-p00,fb-p00,1-fa-fb+p00],-1)
    return p['weights']@comp

def cdf(p,thresholds):
    th=np.asarray(thresholds,dtype=float)
    out=np.zeros((2,len(th)))
    inside=(th>-1.5)&(th<1.5)
    z=np.log(th[inside]+1.5)-np.log(1.5-th[inside])
    for j in range(2):
        out[j,inside]=ndtr((z[:,None]-p['means'][:,j])/p['stds'][:,j])@p['weights']
    out[:,th>=1.5]=1.
    return out

def quantiles(p,levels=(.05,.5,.95)):
    out=np.empty((2,len(levels)))
    for j in range(2):
        for k,level in enumerate(levels):
            lo=float(np.min(p['means'][:,j]-12*p['stds'][:,j]))
            hi=float(np.max(p['means'][:,j]+12*p['stds'][:,j]))
            for _ in range(60):
                mid=(lo+hi)/2
                val=ndtr((mid-p['means'][:,j])/p['stds'][:,j])@p['weights']
                if val<level:lo=mid
                else:hi=mid
            out[j,k]=to_x(np.array([(lo+hi)/2]))[0]
    return out

@lru_cache(None)
def herm(order):return hermgauss(order)

def moments(p,order=64):
    t,w=herm(order);w=w/np.sqrt(np.pi)
    z=p['means'][:,:,None]+np.sqrt(2)*p['stds'][:,:,None]*t
    x=to_x(z)
    mean=p['weights']@(x@w)
    second=p['weights']@((x*x)@w)
    return mean,second-mean*mean

def moments_cdf(p,order=256):
    # Integration by parts on the bounded material domain avoids a slowly
    # convergent Hermite approximation of a very broad logit Gaussian.
    t,w=leg(order);x=1.5*t;dx=1.5*w;f=cdf(p,x)
    mean=1.5-f@dx
    second=2.25-2*(f*x)@dx
    return mean,second-mean*mean

def sample(p,n,rng):
    ii=rng.choice(len(p['weights']),size=n,p=p['weights'])
    noise=rng.normal(size=(n,2))
    rho=p['rho'][ii]
    noise[:,1]=rho*noise[:,0]+np.sqrt(1-rho*rho)*noise[:,1]
    z=p['means'][ii]+p['stds'][ii]*noise
    return to_x(z),z

