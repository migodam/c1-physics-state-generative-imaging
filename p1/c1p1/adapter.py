"""Runnable read-only C1 adapter demonstrating the solver-independent contract."""
import numpy as np
from .common import zero_physics_guard
from .reference import split,grid,posterior,logprior,sampleprior
from .benchmark import (TargetIdentity,BoxSupport,MaterialSpace,ProperPrior,
    MeasurementMetadata,Observation,QuadratureReference,PosteriorBenchmark)

def saved_p0_benchmark(observation_index=0,split_name='validation',order=64):
    zero_physics_guard()
    d=split('p0',split_name);g=grid('p0',order)
    i=observation_index;sigma=float(d['sigma'][i])
    identity=TargetIdentity(
        material_space_id='c1-two-region-latent-v1',support_id='open-square-bound-1.5',
        base_measure_id='Lebesgue-dx1-dx2',prior_id='connected-exponential-tilt-v1',
        likelihood_id='c1-p0-intensity-real-Gaussian-v1',noise_id=f'real-iid-sigma-{sigma:g}',
        nuisance_id='none-fixed-geometry',acquisition_id='two-source-four-intensity-k4-v1',
        solver_id='c1-vector-dipole13-rr-v1',reference_target_id='original-discrete-DDA-continuous-prior-posterior-v1',
        observation_id=f'p0-{split_name}-{i}')
    metadata=MeasurementMetadata(identity.likelihood_id,identity.noise_id,identity.nuisance_id,
        identity.acquisition_id,packing='four real intensity readings',units='known-scale-normalized intensity',
        noise={'sigma':sigma,'distribution':'independent real Gaussian'},nuisance={'policy':'fixed'},
        acquisition={'source_count':2,'readout_count':4,'background_wavenumber':4.})
    observation=Observation(identity.observation_id,d['y'][i],metadata,
        exposure='exposed regression' if split_name=='test' else 'existing train/validation')
    space=MaterialSpace(identity.material_space_id,identity.support_id,identity.base_measure_id,
        BoxSupport(np.full(2,-1.5),np.full(2,1.5)),('core_contrast','shell_contrast'),
        decoder=lambda x:2+np.asarray(x)+1j*.02,
        field_description='two continuous regional permittivities; fixed13cell geometry, not a general field')
    prior=ProperPrior(identity.prior_id,identity.base_measure_id,logprior,sampleprior,
        'analytic product exponential-tilt normalization on open square')
    r=posterior(g,d['y'][i],sigma,problem='p0')
    reference=QuadratureReference(identity,f'original-P0-Gauss-{order}',g['x'],g['w'],r['logpdf'],
        'continuous density quadrature, original saved bank; not discrete material prior or continuum certificate')
    return PosteriorBenchmark(identity,space,prior,metadata,reference,forward=None,observation=observation)

