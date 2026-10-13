import numpy as np
from c1.material import z_to_x,x_to_z

def test_float32_latent_tails_decode_inside_open_material_support_in_float64():
    z=np.array([[-18.,18.],[-12.,12.]],dtype=np.float32)
    x=z_to_x(z)
    assert x.dtype==np.float64 and np.all(np.abs(x)<1.5)
    np.testing.assert_allclose(x_to_z(x),z,atol=1e-7,rtol=0)
