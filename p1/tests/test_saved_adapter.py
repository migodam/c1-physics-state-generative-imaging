import numpy as np
from c1p1.adapter import saved_p0_benchmark
from c1p1.benchmark import CostRecord
import pytest

def test_real_saved_c1_benchmark_constructs_without_any_forward():
    b=saved_p0_benchmark()
    assert b.forward is None
    assert b.target.observation_id=='p0-validation-0'
    assert b.target.noise_id.startswith('real-iid-sigma-')
    assert b.target.solver_id=='c1-vector-dipole13-rr-v1'
    assert b.reference.nodes.shape==(64**2,2)
    np.testing.assert_allclose(b.reference.probability_weights.sum(),1.,atol=1e-14)
    materials=b.material_space.field(np.array([[.2,-.4]]))
    np.testing.assert_allclose(materials.real,[[2.2,1.6]])
    assert np.all(materials.imag>=0)

def test_interface_cost_keeps_derivative_rhs_separate():
    r=CostRecord('candidate',material_frequency_evaluations=1,forward_rhs=2,derivative_rhs=4,adjoint_rhs=0)
    assert r.forward_rhs+r.derivative_rhs+r.adjoint_rhs==6
    with pytest.raises(ValueError,match='physical counts'):
        CostRecord('candidate',derivative_rhs=-1)

