"""Continuous material coordinates and offline events, without branch labels."""
import numpy as np
from scipy.special import expit
from scipy.ndimage import label, binary_opening

BOUND=1.5
TILT=np.array([0.4,-0.2])

def prior_logpdf(x):
    x=np.asarray(x)
    logz=np.log(2*np.sinh(BOUND*TILT)/TILT)
    value=np.sum(x*TILT-logz,axis=-1)
    return np.where(np.all(np.abs(x)<BOUND,axis=-1),value,-np.inf)

def prior_sample(rng,n):
    u=rng.random((n,2))
    return np.log(np.exp(-BOUND*TILT)+u*(np.exp(BOUND*TILT)-np.exp(-BOUND*TILT)))/TILT

def x_to_z(x):
    x=np.asarray(x)
    if np.any(np.abs(x)>=BOUND):raise ValueError('material coordinate outside open support')
    return np.log(x+BOUND)-np.log(BOUND-x)

def z_to_x(z):
    # Keep material decoding in float64: float32 expit rounds moderate logit tails to endpoints.
    return 2*BOUND*expit(np.asarray(z,dtype=np.float64))-BOUND

def quadrant(x):
    x=np.asarray(x)
    return (x[...,0]>=0).astype(int)*2+(x[...,1]>=0).astype(int)

def passive_grid(u,loss_u=None):
    """Full-rank grid coordinates can represent threshold holes of any count."""
    er=0.5+3*expit(np.asarray(u))
    loss=np.zeros_like(er) if loss_u is None else np.logaddexp(0,np.asarray(loss_u))
    return er+1j*loss

def finite_scale_holes(material,threshold=2.0,opening_pixels=1):
    """Post-generation planar enclosed low-material components; not 3D cavities."""
    low=np.asarray(material).real<threshold
    if opening_pixels>1:low=binary_opening(low,structure=np.ones((opening_pixels,opening_pixels)))
    labs,n=label(low)
    touching=set(np.r_[labs[0],labs[-1],labs[:,0],labs[:,-1]])
    return sum(i not in touching for i in range(1,n+1))
