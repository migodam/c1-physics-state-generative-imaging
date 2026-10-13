"""Probability conventions used by continuous reference integration."""
import numpy as np
from scipy.special import logsumexp

def gaussian_loglike(pred,y,sigma):
    pred=np.asarray(pred);y=np.asarray(y)
    return -0.5*np.sum(((pred-y)/sigma)**2,axis=-1)-pred.shape[-1]*np.log(sigma*np.sqrt(2*np.pi))

def proper_complex_whiten(residual,sigma_c):
    r=np.asarray(residual)
    return np.concatenate([r.real,r.imag],axis=-1)*np.sqrt(2)/sigma_c

def tempering_weights(log_prior,log_likelihood,beta):
    b=np.asarray(beta)
    if b[0]!=0 or b[-1]!=1 or np.any(np.diff(b)<0):raise ValueError('invalid single-likelihood schedule')
    logw=np.asarray(log_prior)+np.sum(np.diff(b)[:,None]*np.asarray(log_likelihood)[None,:],axis=0)
    return np.exp(logw-logsumexp(logw))

def split_weight(weight,fractions):
    a=np.asarray(fractions)
    if np.any(a<0) or not np.isclose(a.sum(),1,atol=1e-12,rtol=0):raise ValueError('split fractions must conserve mass')
    return weight*a
