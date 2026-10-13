"""Necessary density mathematics checks; no Maxwell imports or training.

Run only within the parent-controlled charged P2 job.  These checks establish
software probability consistency; they do not establish posterior calibration.
"""
import math

import numpy as np
import unittest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
import torch

from c1p2.density import (BoundedMDN, LOWER, UPPER, from_transformed,
                          log_prob_batch, radial_event_ids, sample_parameters)


def constant_mixture(model):
    """Use a finite, known four-dimensional mixture with actual correlations."""
    with torch.no_grad():
        model.head.weight.zero_()
        raw = torch.zeros((model.components, model.component_size), dtype=model.head.weight.dtype)
        raw[:, 0] = torch.linspace(-.8, .8, model.components, dtype=raw.dtype)
        raw[:, 1:5] = (torch.linspace(-.3, .3, model.components, dtype=raw.dtype)[:, None]
                       + torch.tensor([.2, -.2, .1, -.1], dtype=raw.dtype))
        if model.covariance == "full":
            values = raw[:, 5:]
            values[:, model.diagonal_entries] = 5 * math.atanh(math.log(.7) / 5)
            values[:, ~model.diagonal_entries] = torch.tensor([.15, -.10, .12, .05, -.08, .2], dtype=raw.dtype)
        else:
            raw[:, 5:] = 5 * math.atanh(math.log(.7) / 5)
        model.head.bias.copy_(raw.ravel())


def test_full_mixture_and_exact_material_jacobian_match_independent_scipy():
    model = BoundedMDN().double()
    constant_mixture(model)
    condition = torch.zeros((1, 34), dtype=torch.float64)
    parameters = model.export_parameters(condition)
    z = np.array([[.3, -.2, .6, -.7], [2., -1., -2., 1.5], [-8., 6., 5., -7.]])
    decoded = from_transformed(z)
    x = torch.as_tensor(decoded["latent"])
    component = []
    for k in range(model.components):
        factor = parameters["cholesky"][0, k]
        component.append(parameters["log_weights"][0, k] +
                         multivariate_normal.logpdf(z, mean=parameters["means"][0, k], cov=factor @ factor.T))
    dz_log_density = logsumexp(np.stack(component, axis=-1), axis=-1)
    expected_jacobian = (-np.log(np.asarray(UPPER) - LOWER)
                         + np.logaddexp(0, z) + np.logaddexp(0, -z)).sum(-1)
    expected = dz_log_density + expected_jacobian
    np.testing.assert_allclose(model.log_prob(x, condition).detach(), expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(log_prob_batch(x.numpy(), parameters), expected, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(model.log_prob_transformed(torch.as_tensor(z), condition).detach(), dz_log_density, atol=2e-12, rtol=2e-12)
    differentiable = x[0].clone().requires_grad_()
    transform = lambda point: (point - model.lower).log() - (model.upper - point).log()
    jacobian = torch.autograd.functional.jacobian(transform, differentiable)
    sign, actual = torch.linalg.slogdet(jacobian)
    assert sign == 1
    np.testing.assert_allclose(float(actual), expected_jacobian[0], atol=2e-12)


def test_diagonal_material_pushforward_has_unit_mass_by_independent_quadrature():
    model = BoundedMDN(covariance="diagonal", components=1).double()
    constant_mixture(model)
    p = model.export_parameters(torch.zeros((1, 34), dtype=torch.float64))
    nodes, weights = np.polynomial.legendre.leggauss(192)
    lo, hi = np.array(LOWER), np.array(UPPER)
    midpoint = (lo + hi) / 2
    masses = []
    for coordinate in range(4):
        x = np.tile(midpoint, (len(nodes), 1))
        x[:, coordinate] = midpoint[coordinate] + (hi[coordinate] - lo[coordinate]) * nodes / 2
        # Divide independent other-coordinate densities; this is 1D marginal
        # integration, not numerical normalization of the learned density.
        factor = 1.
        for j in range(4):
            if j != coordinate:
                factor *= norm.pdf(0, loc=p["means"][0, 0, j], scale=p["cholesky"][0, 0, j, j]) * 4 / (hi[j] - lo[j])
        mass = np.exp(log_prob_batch(x, p)) @ (weights * (hi[coordinate] - lo[coordinate]) / 2) / factor
        masses.append(mass)
    np.testing.assert_allclose(masses, np.ones(4), atol=1e-10, rtol=1e-10)
    assert abs(np.prod(masses) - 1) < 5e-10


def test_cholesky_sampling_matches_four_dimensional_covariance_and_marginal_cdf():
    model = BoundedMDN(components=1).double()
    constant_mixture(model)
    y = torch.zeros((1, 34), dtype=torch.float64)
    parameters = model.export_parameters(y)
    with torch.no_grad():
        draws = model.sample(y, 40000, torch.Generator().manual_seed(26017))
    z = draws.transformed[0].numpy()
    mean, factor = parameters["means"][0, 0], parameters["cholesky"][0, 0]
    covariance = factor @ factor.T
    np.testing.assert_allclose(z.mean(0), mean, atol=.02, rtol=0)
    np.testing.assert_allclose(np.cov(z, rowvar=False), covariance, atol=.012, rtol=.05)
    for j in range(4):
        thresholds = mean[j] + np.sqrt(covariance[j, j]) * np.array([-1.5, -.5, .5, 1.5])
        expected = norm.cdf(thresholds, mean[j], np.sqrt(covariance[j, j]))
        empirical = (z[:, j, None] <= thresholds).mean(0)
        np.testing.assert_allclose(empirical, expected, atol=.012, rtol=0)
    torch.testing.assert_close(draws.transformed, model.sample(y, 40000, torch.Generator().manual_seed(26017)).transformed)
    ndraws = sample_parameters(parameters, 40000, np.random.default_rng(26018))
    np.testing.assert_allclose(ndraws["transformed"][0].mean(0), mean, atol=.02, rtol=0)
    assert not draws.rounded_endpoint.any()


def test_extreme_endpoint_rounding_keeps_raw_coordinates_without_clipping():
    model = BoundedMDN(components=1).double()
    constant_mixture(model)
    with torch.no_grad():
        model.head.bias[1:5] = torch.tensor([80., -80., 80., -80.], dtype=model.head.bias.dtype)
    y = torch.zeros((1, 34), dtype=torch.float64)
    with torch.no_grad():
        draws = model.sample(y, 12, torch.Generator().manual_seed(33))
    assert draws.rounded_endpoint.all()
    assert torch.isfinite(draws.transformed).all()
    assert torch.isneginf(model.log_prob(draws.latent, y)).all()
    assert torch.isfinite(model.log_prob_draws(draws, y)).all()
    p = model.export_parameters(y)
    raw_density = log_prob_batch(draws.transformed[0].numpy(), p, transformed=True)
    np.testing.assert_allclose(raw_density, model.log_prob_draws(draws, y)[0].detach(), rtol=1e-12, atol=1e-12)


def test_train_only_normalization_exact_bounds_and_measurement_gradient():
    model = BoundedMDN()
    data = torch.randn((20, 34), generator=torch.Generator().manual_seed(43))
    model.fit_normalization(data)
    torch.testing.assert_close(model.condition_mean, data.mean(0))
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "already fitted"):
        model.fit_normalization(50 * data)
    p = model.export_parameters(data[:2])
    np.testing.assert_array_equal(p["lower"], LOWER)
    np.testing.assert_array_equal(p["upper"], UPPER)
    coordinate = torch.tensor([[-.8, -.5, 2.45, .11], [-1.8, -2.1, 2.65, .08]], requires_grad=True)
    condition = data[:2].clone().requires_grad_()
    loss = -model.log_prob(coordinate, condition).mean()
    loss.backward()
    assert condition.grad is None
    assert coordinate.grad is not None and torch.isfinite(coordinate.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    cholesky = p["cholesky"]
    assert np.all(np.diagonal(cholesky, axis1=-2, axis2=-1) > 0)
    with unittest.TestCase().assertRaises(TypeError):
        model.log_prob(coordinate, condition, topology=torch.zeros(2))


def test_derived_events_partition_and_prior_solid_mass_are_not_generator_labels():
    rng = np.random.default_rng(26101334)
    latent = rng.uniform(LOWER, UPPER, (120000, 4))
    event = radial_event_ids(latent)
    assert event.shape == (len(latent),) and np.all((event >= 0) & (event < 12))
    solid = event == 0
    np.testing.assert_array_equal(solid, (latent[:, 0] >= -1) & (latent[:, 1] >= -1))
    assert abs(solid.mean() - .25) < .006
    atom_latent = np.array([[-1., -1., 2.5, .1], [0., .5, 2.5, .1]])
    np.testing.assert_array_equal(radial_event_ids(atom_latent), [0, 0])

if __name__ == '__main__':
    import json,time
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter();rows=[]
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn();rows.append(dict(name=name,status='PASS'))
    print(json.dumps(dict(checks=rows,new_physics=0,wall_seconds=time.perf_counter()-start)))
