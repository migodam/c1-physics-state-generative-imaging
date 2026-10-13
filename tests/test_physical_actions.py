"""Small independent operator-action checks, all actual RHS charged canonically."""
import numpy as np
import pytest
from c1.problems import DipoleProblem

@pytest.fixture(scope='module')
def candidate():
    p=DipoleProblem();p.set_scale();x=np.array([.7,-.4])
    s=p.forward(x,derivative=True,kind='operator_action_candidate')
    return p,x,s

def test_adjoint_matches_material_jacobian_gradient(candidate):
    p,x,s=candidate;y=s['f']+np.array([.01,-.03,.04,-.02]);sigma=.2
    g=p.adjoint_gradient(s,x,y,sigma)
    np.testing.assert_allclose(g,s['J'].T@(s['f']-y)/sigma**2,rtol=2e-12,atol=1e-13)

def test_matrix_free_real_material_probes_match_explicit_small_jacobian(candidate):
    p,x,s=candidate
    for h in [np.array([.6,.8]),np.array([-.8,.6])]:
        np.testing.assert_allclose(p.jvp(s,x,h),s['J']@h,rtol=2e-12,atol=1e-13)

def test_wrong_candidate_cache_refused_before_extra_action_charge(candidate):
    p,x,s=candidate;before=p.ledger.summary()['used']
    with pytest.raises(ValueError,match='different candidate'):p.jvp(s,x+.01,[1.,0])
    assert p.ledger.summary()['used']==before
