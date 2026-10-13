"""Independent density mathematics checks; no training or physical calls.

These tests are supplied for the parent-controlled, charged execution job.
No topology classes or unknown true-material fields condition any density.
"""

import inspect
import math

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
import torch

from c1.density import Density
from c1.material import z_to_x
from c1p1.density import FullCovDensity, numpy_parameters


CONDITIONS = ("measurement", "state", "memory")


def _inputs(batch=3, state_dim=18):
    generator = torch.Generator().manual_seed(821)
    y = torch.randn((batch, 5), dtype=torch.float64, generator=generator)
    states = torch.randn((batch, 2, state_dim), dtype=torch.float64, generator=generator)
    z = torch.randn((batch, 2), dtype=torch.float64, generator=generator)
    return z, y, states


def _constant_mixture(model, means=None, stds=None, rho=None):
    """Set a moderate known mixture independent of encoder outputs."""
    raw = torch.zeros((8, 6), dtype=torch.float64)
    raw[:, 0] = torch.linspace(-1, 1, 8)
    raw[:, 1] = torch.linspace(-0.8, 0.8, 8)
    raw[:, 2] = torch.linspace(0.4, -0.4, 8)
    raw[:, 3:5] = 5 * torch.atanh(torch.tensor([-0.25, 0.15]) / 5)
    raw[:, 5] = torch.atanh(torch.linspace(-0.7, 0.7, 8) / 0.995)
    if means is not None:
        raw[:, 1:3] = torch.tensor(means, dtype=torch.float64)
    if stds is not None:
        logs = torch.tensor(stds, dtype=torch.float64).log()
        raw[:, 3:5] = 5 * torch.atanh(logs / 5)
    if rho is not None:
        raw[:, 5] = math.atanh(rho / 0.995)
    with torch.no_grad():
        model.mdn_head.weight.zero_()
        model.mdn_head.bias.copy_(raw.flatten())


@pytest.mark.parametrize("condition", CONDITIONS)
@pytest.mark.parametrize("state_dim", (18, 38))
def test_real_connected_parameter_counts_match_legacy_mdn(condition, state_dim):
    model = FullCovDensity(condition, 5, state_dim).double()
    legacy_kind = {"measurement": "mdn", "state": "mdn_state", "memory": "mdn_memory"}[condition]
    legacy = Density(legacy_kind, 5, state_dim)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == model.parameter_count
    assert abs(count - 30000) <= 0.05 * 30000
    assert abs(count - legacy.parameter_count) <= 0.05 * 30000
    assert all(p.device.type == "cpu" for p in model.parameters())
    z, y, states = _inputs(state_dim=state_dim)
    (-model.log_prob_z(z, y, states).mean()).backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.mdn_head.weight.grad[5::6].abs().sum() > 0
    if condition == "memory":
        assert model.memory.weight_ih_l0.grad.abs().sum() > 0


def test_too_small_target_and_bad_condition_are_rejected():
    with pytest.raises(ValueError, match="parameter target"):
        FullCovDensity("measurement", 5, 18, target_parameters=1000)
    with pytest.raises(ValueError, match="condition"):
        FullCovDensity("topology", 5, 18)


def test_fullcov_logpdf_agrees_with_independent_scipy_mixture():
    model = FullCovDensity("measurement", 5, 18).double()
    _constant_mixture(model)
    z, y, states = _inputs(batch=7)
    parameters = numpy_parameters(model, y, states)
    component_logpdfs = []
    for component in range(8):
        stds = parameters["stds"][0, component]
        rho = parameters["rho"][0, component]
        covariance = np.diag(stds ** 2)
        covariance[0, 1] = covariance[1, 0] = rho * stds[0] * stds[1]
        component_logpdfs.append(
            np.log(parameters["weights"][0, component])
            + multivariate_normal.logpdf(z.numpy(), mean=parameters["means"][0, component],
                                        cov=covariance)
        )
    expected = logsumexp(np.stack(component_logpdfs, axis=1), axis=1)
    np.testing.assert_allclose(model.log_prob_z(z, y, states).detach().numpy(), expected,
                               rtol=1e-12, atol=1e-12)


def test_mixture_normalization_is_checked_by_independent_quadrature():
    model = FullCovDensity("measurement", 5, 18).double()
    _constant_mixture(model)
    _, y, states = _inputs(batch=1)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    axis = torch.as_tensor(10 * nodes, dtype=torch.float64)
    grid = torch.cartesian_prod(axis, axis)
    mass_weights = torch.as_tensor(np.outer(10 * weights, 10 * weights).ravel(),
                                   dtype=torch.float64)
    context = model.c(y, states).expand(len(grid), -1)
    with torch.no_grad():
        mass = (model.log_prob_z_context(grid, context).exp() * mass_weights).sum()
    assert abs(float(mass) - 1.0) < 1e-8


def test_material_density_and_jacobian_match_change_of_variables():
    model = FullCovDensity("measurement", 5, 18).double()
    _constant_mixture(model)
    _, y, states = _inputs(batch=2)
    x = torch.tensor([[-1.49999, -0.2], [0.4, 1.49999]], dtype=torch.float64,
                     requires_grad=True)
    z = (x + 1.5).log() - (1.5 - x).log()
    log_jacobian = (math.log(3) - (x + 1.5).log() - (1.5 - x).log()).sum(-1)
    jacobian = torch.autograd.functional.jacobian(
        lambda material: (material + 1.5).log() - (1.5 - material).log(), x[0]
    )
    _, actual_logdet = torch.linalg.slogdet(jacobian)
    torch.testing.assert_close(actual_logdet, log_jacobian[0])
    actual = model.log_prob_x(x, y, states)
    torch.testing.assert_close(actual, model.log_prob_z(z, y, states) + log_jacobian,
                               rtol=1e-12, atol=1e-12)
    assert torch.isfinite(torch.autograd.grad(actual.sum(), x)[0]).all()
    outside = torch.tensor([[-1.5, 0.0], [0.0, 1.5], [-1.6, 0.1], [math.nan, 0.0]],
                           dtype=torch.float64)
    context = model.c(y[:1], states[:1]).expand(len(outside), -1)
    assert torch.isneginf(model.log_prob_x_context(outside, context)).all()


def test_normalized_material_pushforward_integral():
    model = FullCovDensity("measurement", 5, 18).double()
    _constant_mixture(model, means=(0.05, -0.1), stds=(0.7, 0.8), rho=0.45)
    _, y, states = _inputs(batch=1)
    nodes, weights = np.polynomial.legendre.leggauss(128)
    axis = torch.as_tensor(1.5 * nodes, dtype=torch.float64)
    grid = torch.cartesian_prod(axis, axis)
    mass_weights = torch.as_tensor(np.outer(1.5 * weights, 1.5 * weights).ravel(),
                                   dtype=torch.float64)
    context = model.c(y, states).expand(len(grid), -1)
    with torch.no_grad():
        mass = (model.log_prob_x_context(grid, context).exp() * mass_weights).sum()
    assert abs(float(mass) - 1.0) < 3e-7


def test_covariance_strictly_positive_definite_for_extreme_raw_parameters():
    model = FullCovDensity("measurement", 5, 18).double()
    with torch.no_grad():
        model.mdn_head.weight.zero_()
        raw = torch.zeros((8, 6), dtype=torch.float64)
        raw[:, 3] = 1e4
        raw[:, 4] = -1e4
        raw[:, 5] = torch.tensor([-1e6, 1e6] * 4, dtype=torch.float64)
        model.mdn_head.bias.copy_(raw.ravel())
    z, y, states = _inputs(batch=3)
    parameters = numpy_parameters(model, y, states)
    assert np.max(np.abs(parameters["rho"])) <= 0.995
    for stds, rho in zip(parameters["stds"][0], parameters["rho"][0]):
        covariance = np.array([[stds[0] ** 2, rho * stds.prod()],
                               [rho * stds.prod(), stds[1] ** 2]])
        assert np.linalg.eigvalsh(covariance).min() > 0
        assert np.linalg.det(covariance) > 0
    tail = torch.tensor([[0.0, 0.0], [100.0, -100.0], [-100.0, 100.0]], dtype=torch.float64)
    logpdf = model.log_prob_z(tail, y, states)
    assert torch.isfinite(logpdf).all()
    (-logpdf.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_correlated_cholesky_sampling_and_normal_marginals():
    model = FullCovDensity("measurement", 5, 18).double()
    mean, std, rho = np.array([0.4, -0.6]), np.array([0.6, 1.1]), 0.73
    _constant_mixture(model, means=mean, stds=std, rho=rho)
    _, y, states = _inputs(batch=1)
    with torch.no_grad():
        sample = model.sample_z(y, states, 40000,
                                torch.Generator().manual_seed(911))[0].numpy()
    np.testing.assert_allclose(sample.mean(axis=0), mean, rtol=0, atol=0.018)
    covariance = np.diag(std ** 2)
    covariance[0, 1] = covariance[1, 0] = rho * std.prod()
    np.testing.assert_allclose(np.cov(sample, rowvar=False), covariance, rtol=0.045, atol=0.01)
    for coordinate in range(2):
        thresholds = mean[coordinate] + std[coordinate] * np.array([-1.5, -0.5, 0.5, 1.5])
        empirical = (sample[:, coordinate, None] <= thresholds).mean(axis=0)
        expected = norm.cdf(thresholds, loc=mean[coordinate], scale=std[coordinate])
        np.testing.assert_allclose(empirical, expected, rtol=0, atol=0.012)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_context_reuses_only_saved_features_and_preserves_parameter_gradients(condition):
    model = FullCovDensity(condition, 5, 18).double()
    z, y, states = _inputs(batch=2)
    first, last = states.clone(), states.clone()
    first[:, 0] += 0.8
    last[:, -1] += 0.8
    context = model.c(y, states)
    assert context.shape == (2, 32)
    for coordinate in range(5):
        changed = y.clone()
        changed[:, coordinate] += 0.7
        assert not torch.allclose(context, model.c(changed, states))
    if condition != "memory":
        torch.testing.assert_close(context, model.c(y, first), rtol=0, atol=0)
    else:
        assert not torch.allclose(context, model.c(y, first))
    if condition == "measurement":
        torch.testing.assert_close(context, model.c(y, last), rtol=0, atol=0)
    else:
        assert not torch.allclose(context, model.c(y, last))
    y.requires_grad_()
    states.requires_grad_()
    z.requires_grad_()
    (-model.log_prob_z(z, y, states).mean()).backward()
    assert y.grad is None and states.grad is None
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert model.encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("condition", CONDITIONS)
def test_sampling_generator_empty_batch_and_single_context_encoding(condition):
    model = FullCovDensity(condition, 5, 18).double()
    _, y, states = _inputs()
    calls = []
    hook = model.encoder.register_forward_hook(lambda module, inputs, output: calls.append(1))
    try:
        first = model.sample_z(y, states, 71, torch.Generator().manual_seed(412))
    finally:
        hook.remove()
    assert len(calls) == 1
    second = model.sample_z(y, states, 71, torch.Generator().manual_seed(412))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (3, 71, 2) and torch.isfinite(first).all()
    assert model.sample_z(y, states, 0).shape == (3, 0, 2)
    assert model.sample_z(y[:0], states[:0], 3).shape == (0, 3, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        model.sample_z(y, states, -1)


def test_float64_decode_retains_moderate_tail_interior_and_exposes_extreme_rounding():
    z = np.array([[-20.0, 20.0], [-25.0, 25.0]], dtype=np.float32)
    x = z_to_x(z)
    assert x.dtype == np.float64
    assert np.all(np.abs(x) < 1.5)
    model = FullCovDensity("measurement", 5, 18).double()
    _, y, states = _inputs(batch=2)
    assert torch.isfinite(model.log_prob_x(torch.as_tensor(x), y, states)).all()
    # Finite machine arithmetic still cannot decode every real logit to an
    # interior float. The endpoint is detected and never silently repaired.
    extreme = z_to_x(np.array([[1000.0, -1000.0]]))
    assert np.any(np.abs(extreme) == 1.5)
    assert torch.isneginf(model.log_prob_x(torch.as_tensor(extreme), y[:1], states[:1])).all()


@pytest.mark.parametrize("legacy_kind", ("mdn", "mdn_state", "mdn_memory"))
def test_numpy_export_supports_unchanged_diagonal_mdn(legacy_kind):
    model = Density(legacy_kind, 5, 18).double()
    _, y, states = _inputs(batch=3)
    result = numpy_parameters(model, y.numpy(), states.numpy())
    assert set(result) == {"weights", "means", "stds", "rho"}
    assert result["weights"].shape == (3, 8)
    assert result["means"].shape == result["stds"].shape == (3, 8, 2)
    assert result["rho"].shape == (3, 8)
    assert all(array.dtype == np.float64 for array in result.values())
    np.testing.assert_allclose(result["weights"].sum(-1), 1, rtol=0, atol=1e-12)
    assert np.all(result["rho"] == 0)
    with torch.no_grad():
        logits, means, logs = model._mixture_parameters(model.c(y, states))
    np.testing.assert_array_equal(result["means"], means.numpy())
    np.testing.assert_allclose(result["stds"], logs.exp().numpy(), rtol=1e-14, atol=0)


def test_numpy_export_rejects_flow_and_encoded_density_matches_public_api():
    _, y, states = _inputs()
    with pytest.raises(ValueError, match="MDN"):
        numpy_parameters(Density("measurement", 5, 18).double(), y, states)
    model = FullCovDensity("measurement", 5, 18).double()
    z = torch.tensor([[0.0, 1.0], [-2.0, 0.4], [0.8, -1.1]], dtype=torch.float64)
    x = torch.as_tensor(z_to_x(z.numpy()))
    context = model.c(y, states)
    torch.testing.assert_close(model.log_prob_z(z, y, states), model.log_prob_z_context(z, context))
    torch.testing.assert_close(model.log_prob_x(x, y, states), model.log_prob_x_context(x, context))


def test_public_interface_has_no_topology_truth_or_solver_inputs():
    assert tuple(inspect.signature(FullCovDensity).parameters) == (
        "condition", "y_dim", "state_dim", "target_parameters"
    )
    assert tuple(inspect.signature(FullCovDensity.c).parameters) == ("self", "y", "states")
    assert tuple(inspect.signature(FullCovDensity.log_prob_x).parameters) == ("self", "x", "y", "states")
    assert tuple(inspect.signature(FullCovDensity.sample_z).parameters) == (
        "self", "y", "states", "n", "generator"
    )
    model = FullCovDensity("measurement", 5, 18)
    with pytest.raises(TypeError):
        model.sample_z(torch.zeros(1, 5), torch.zeros(1, 2, 18), 1, topology=0)
    with pytest.raises(TypeError):
        model.c(torch.zeros(1, 5), torch.zeros(1, 2, 18), truth=torch.zeros(1, 2))
