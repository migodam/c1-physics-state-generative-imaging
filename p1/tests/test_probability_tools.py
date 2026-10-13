import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from c1p1.mixture import events,cdf,logpdf_x,logpdf_z,quantiles,moments_cdf
from c1p1.reference import grid,posterior,marginal_cdf,logprior,to_z,to_x

def pars(rho):
    return dict(weights=np.array([1.]),means=np.array([[.4,-.7]]),
                stds=np.array([[.8,1.2]]),rho=np.array([rho]))

@pytest.mark.parametrize('rho',[-.994,-.6,0.,.7,.994])
def test_event_integral_agrees_with_independent_conditional_normal_quad(rho):
    p=pars(rho);a,b=(-p['means']/p['stds'])[0]
    value=quad(lambda u:np.exp(-u*u/2)/np.sqrt(2*np.pi)*ndtr((b-rho*u)/np.sqrt(1-rho*rho)),
               -np.inf,a,epsabs=1e-12,epsrel=1e-12)[0]
    e=events(p)
    np.testing.assert_allclose(e[0],value,atol=2e-10,rtol=0)
    np.testing.assert_allclose(e.sum(),1.,atol=1e-14)
    assert e.min()>-1e-12
    np.testing.assert_allclose(events(p,32),e,atol=2e-8)

def test_material_log_jacobian_and_quantiles_are_exact():
    p=pars(.7);x=np.array([[.2,-.4],[-1.499,1.498]])
    jac=(np.log(3)-np.log(x+1.5)-np.log(1.5-x)).sum(-1)
    np.testing.assert_allclose(logpdf_x(x,p),logpdf_z(to_z(x),p)+jac,atol=1e-12)
    q=quantiles(p)
    for j in range(2):
        for k,level in enumerate([.05,.5,.95]):
            np.testing.assert_allclose(cdf(p,[q[j,k]])[j,0],level,atol=2e-12)
    assert np.all(np.isneginf(logpdf_x(np.array([[1.5,0.],[-2.,0.]]),p)))

def test_uniform_continuous_reference_cdf_and_volume_weights():
    g=grid('u0',64);g['f']=np.zeros((len(g['x']),4))
    r=posterior(g,np.zeros(4),1.,problem='p0',flat_prior=True)
    thresholds=np.linspace(-1.5,1.5,13)
    np.testing.assert_allclose(marginal_cdf(g,r['mass'],thresholds),
                               np.tile((thresholds+1.5)/3,(2,1)),atol=2e-14)
    np.testing.assert_allclose(r['events'],np.full(4,.25),atol=1e-14)
    np.testing.assert_allclose(g['w'].sum(),9.,atol=2e-13)
    np.testing.assert_allclose(np.exp(logprior(g['x']))@g['w'],1.,atol=2e-14)

def test_extreme_float64_decode_remains_explicitly_unresolved():
    z=np.array([[45.,-45.]])
    x=to_x(z)
    assert np.any(np.abs(x)>=1.5)
    assert np.isneginf(logpdf_x(x,pars(0.))[0])
    assert np.isfinite(logpdf_z(z,pars(0.))[0])

def test_zero_physics_guard_blocks_before_historical_ledger_creation():
    from c1p1.common import zero_physics_guard
    zero_physics_guard()
    from src.c1.accounting import PhysicsLedger
    from src.c1.problems import DipoleProblem
    with pytest.raises(RuntimeError,match='ZERO'):
        PhysicsLedger()
    with pytest.raises(RuntimeError,match='ZERO'):
        DipoleProblem()
    from c1.accounting import PhysicsLedger as AliasLedger
    with pytest.raises(RuntimeError,match='ZERO'):
        AliasLedger()

def test_bounded_material_moments_agree_with_independent_gaussian_integrals():
    p=pars(.7)
    mean,var=moments_cdf(p)
    for j in range(2):
        mu,sd=p['means'][0,j],p['stds'][0,j]
        def expectation(power):
            return quad(lambda z:to_x(np.array([mu+sd*z]))[0]**power*np.exp(-z*z/2)/np.sqrt(2*np.pi),
                        -np.inf,np.inf,epsabs=1e-12)[0]
        m=expectation(1);v=expectation(2)-m*m
        np.testing.assert_allclose(mean[j],m,atol=3e-10)
        np.testing.assert_allclose(var[j],v,atol=3e-10)
