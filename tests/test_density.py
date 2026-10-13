"""Density algebra and interface checks; no training or physical solver calls."""

import inspect
import math

import numpy as np
import pytest
import torch

from c1.density import Density


KINDS = ("mdn", "measurement", "state", "memory")
FLOW_KINDS = KINDS[1:]


def _inputs(batch=3, state_dim=18):
    generator = torch.Generator().manual_seed(113)
    y = torch.randn((batch, 5), dtype=torch.float64, generator=generator)
    states = torch.randn((batch, 2, state_dim), dtype=torch.float64, generator=generator)
    z = torch.randn((batch, 2), dtype=torch.float64, generator=generator)
    return z, y, states


def _nontrivial_flow(model):
    generator = torch.Generator().manual_seed(191)
    with torch.no_grad():
        for coupling in model.couplings:
            final = coupling.conditioner[-1]
            final.weight.copy_(0.03 * torch.randn(final.weight.shape, dtype=final.weight.dtype, generator=generator))
            final.bias.copy_(0.03 * torch.randn(final.bias.shape, dtype=final.bias.dtype, generator=generator))


@pytest.mark.parametrize("state_dim", (18, 38))
def test_actual_parameter_counts_are_matched_without_padding(state_dim):
    counts = []
    for kind in KINDS:
        model = Density(kind, y_dim=5, state_dim=state_dim)
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        assert count == model.parameter_count
        assert abs(count - 30000) <= 0.05 * 30000
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        counts.append(count)
    assert max(counts) - min(counts) <= 0.05 * 30000


def test_target_too_small_for_fixed_flow_is_rejected():
    with pytest.raises(ValueError, match="parameter target"):
        Density("measurement", 5, 18, target_parameters=1000)


@pytest.mark.parametrize("kind", FLOW_KINDS)
def test_zero_initialized_couplings_are_identity(kind):
    model = Density(kind, 5, 18).double()
    z, y, states = _inputs()
    assert len(model.couplings) == 6
    assert [block.conditioned_coordinate for block in model.couplings] == [0, 1, 0, 1, 0, 1]
    for block in model.couplings:
        assert block.conditioner[0].out_features == 48
        assert block.conditioner[2].out_features == 48
        assert torch.count_nonzero(block.conditioner[-1].weight) == 0
        assert torch.count_nonzero(block.conditioner[-1].bias) == 0
    eps, logdet = model.forward_to_base(z, y, states)
    recovered, inverse_logdet = model.inverse_from_base(eps, y, states)
    torch.testing.assert_close(eps, z, rtol=0, atol=0)
    torch.testing.assert_close(recovered, z, rtol=0, atol=0)
    torch.testing.assert_close(logdet, torch.zeros_like(logdet), rtol=0, atol=0)
    torch.testing.assert_close(inverse_logdet, -logdet, rtol=0, atol=0)
    expected = -0.5 * z.square().sum(dim=-1) - math.log(2 * math.pi)
    torch.testing.assert_close(model.log_prob_z(z, y, states), expected)


@pytest.mark.parametrize("kind", FLOW_KINDS)
def test_nontrivial_inverse_and_logdet_match_autograd(kind):
    model = Density(kind, 5, 18).double()
    _nontrivial_flow(model)
    z, y, states = _inputs(batch=2)
    eps, logdet = model.forward_to_base(z, y, states)
    recovered, inverse_logdet = model.inverse_from_base(eps, y, states)
    torch.testing.assert_close(recovered, z, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(logdet + inverse_logdet, torch.zeros_like(logdet), rtol=0, atol=1e-10)
    for row in range(z.shape[0]):
        jacobian = torch.autograd.functional.jacobian(
            lambda point: model.forward_to_base(point[None], y[row:row + 1], states[row:row + 1])[0][0],
            z[row],
        )
        sign, actual_logdet = torch.linalg.slogdet(jacobian)
        assert sign > 0
        torch.testing.assert_close(actual_logdet, logdet[row], rtol=1e-9, atol=1e-10)
    expected = -0.5 * eps.square().sum(dim=-1) - math.log(2 * math.pi) + logdet
    torch.testing.assert_close(model.log_prob_z(z, y, states), expected)


def test_mdn_matches_a_normalized_unlabeled_gaussian_mixture():
    model = Density("mdn", 5, 18).double()
    raw = torch.zeros((8, 5), dtype=torch.float64)
    raw[:, 0] = torch.linspace(-1, 1, 8)
    raw[:, 1] = torch.linspace(-0.8, 0.8, 8)
    raw[:, 2] = torch.linspace(0.4, -0.4, 8)
    raw[:, 3] = torch.linspace(-0.3, 0.3, 8)
    raw[:, 4] = 0.15
    with torch.no_grad():
        model.mdn_head.weight.zero_()
        model.mdn_head.bias.copy_(raw.flatten())
    z, y, states = _inputs(batch=7)
    log_stds = 5 * torch.tanh(raw[:, 3:5] / 5)
    distribution = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=raw[:, 0]),
        torch.distributions.Independent(torch.distributions.Normal(raw[:, 1:3], log_stds.exp()), 1),
    )
    torch.testing.assert_close(model.log_prob_z(z, y, states), distribution.log_prob(z), rtol=1e-12, atol=1e-12)
    # An independent integral checks mixture normalization beyond pointwise algebra.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    axis = torch.as_tensor(10 * nodes, dtype=torch.float64)
    grid = torch.cartesian_prod(axis, axis)
    mass_weights = torch.as_tensor(np.outer(10 * weights, 10 * weights).reshape(-1), dtype=torch.float64)
    mass = (model.log_prob_z(grid, y[:1].expand(grid.shape[0], -1),
                             states[:1].expand(grid.shape[0], -1, -1)).exp() * mass_weights).sum()
    torch.testing.assert_close(mass, torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_exact_logit_jacobian_and_open_support(kind):
    model = Density(kind, 5, 18).double()
    if kind != "mdn":
        _nontrivial_flow(model)
    _, y, states = _inputs(batch=2)
    x = torch.tensor([[-1.49, -0.2], [0.4, 1.49]], dtype=torch.float64, requires_grad=True)
    z = (x + 1.5).log() - (1.5 - x).log()
    log_jacobian = (math.log(3) - (x + 1.5).log() - (1.5 - x).log()).sum(dim=-1)
    actual = model.log_prob_x(x, y, states)
    torch.testing.assert_close(actual, model.log_prob_z(z, y, states) + log_jacobian, rtol=1e-12, atol=1e-12)
    gradient = torch.autograd.grad(actual.sum(), x)[0]
    assert torch.isfinite(gradient).all()
    outside = torch.tensor([[-1.5, 0.0], [0.0, 1.6]], dtype=torch.float64)
    assert torch.isneginf(model.log_prob_x(outside, y, states)).all()


def test_logit_density_integrates_to_one_on_material_support():
    model = Density("measurement", 5, 18).double()
    _, y, states = _inputs(batch=1)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    axis = torch.as_tensor(1.5 * nodes, dtype=torch.float64)
    grid = torch.cartesian_prod(axis, axis)
    mass_weights = torch.as_tensor(np.outer(1.5 * weights, 1.5 * weights).reshape(-1), dtype=torch.float64)
    mass = (model.log_prob_x(grid, y.expand(grid.shape[0], -1),
                            states.expand(grid.shape[0], -1, -1)).exp() * mass_weights).sum()
    torch.testing.assert_close(mass, torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=2e-6)


def test_context_uses_full_y_and_only_requested_state_history():
    _, y, states = _inputs(batch=2)
    altered_first = states.clone()
    altered_first[:, 0] += 0.8
    altered_last = states.clone()
    altered_last[:, 1] += 0.8
    for kind in KINDS:
        model = Density(kind, 5, 18).double()
        context = model.c(y, states)
        assert context.shape == (2, 32)
        for coordinate in range(y.shape[1]):
            changed_y = y.clone()
            changed_y[:, coordinate] += 0.7
            assert not torch.allclose(context, model.c(changed_y, states))
        if kind in ("mdn", "measurement", "state"):
            torch.testing.assert_close(context, model.c(y, altered_first), rtol=0, atol=0)
        if kind in ("mdn", "measurement"):
            torch.testing.assert_close(context, model.c(y, altered_last), rtol=0, atol=0)
        else:
            assert not torch.allclose(context, model.c(y, altered_last))
        if kind == "memory":
            assert not torch.allclose(context, model.c(y, altered_first))


@pytest.mark.parametrize("kind", KINDS)
def test_frozen_features_preserve_parameter_gradients(kind):
    model = Density(kind, 5, 18).double()
    if kind != "mdn":
        _nontrivial_flow(model)
    z, y, states = _inputs()
    z.requires_grad_()
    y.requires_grad_()
    states.requires_grad_()
    (-model.log_prob_z(z, y, states).mean()).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert y.grad is None and states.grad is None
    assert model.encoder[0].weight.grad.abs().sum() > 0
    if kind == "memory":
        assert model.memory.weight_ih_l0.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", KINDS)
def test_sampling_shape_seed_and_empty_draws(kind):
    model = Density(kind, 5, 18).double()
    if kind != "mdn":
        _nontrivial_flow(model)
    _, y, states = _inputs(batch=3)
    first = model.sample_z(y, states, 11, torch.Generator().manual_seed(71))
    second = model.sample_z(y, states, 11, torch.Generator().manual_seed(71))
    assert first.shape == (3, 11, 2)
    assert first.device.type == "cpu" and first.dtype == torch.float64
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert model.sample_z(y, states, 0).shape == (3, 0, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        model.sample_z(y, states, -1)


def test_memory_sampling_encodes_once_per_observation_batch():
    model = Density("memory", 5, 18).double()
    _, y, states = _inputs(batch=3)
    calls = {"memory": 0, "encoder": 0}

    def counted(name):
        def hook(module, inputs, result):
            calls[name] += 1
        return hook

    memory_hook = model.memory.register_forward_hook(counted("memory"))
    encoder_hook = model.encoder.register_forward_hook(counted("encoder"))
    try:
        samples = model.sample_z(y, states, 1024, torch.Generator().manual_seed(2))
    finally:
        memory_hook.remove()
        encoder_hook.remove()
    assert samples.shape == (3, 1024, 2)
    assert calls == {"memory": 1, "encoder": 1}


def test_public_interface_has_no_truth_or_topology_inputs():
    assert tuple(inspect.signature(Density).parameters) == ("kind", "y_dim", "state_dim", "target_parameters")
    assert tuple(inspect.signature(Density.c).parameters) == ("self", "y", "states")
    assert tuple(inspect.signature(Density.log_prob_z).parameters) == ("self", "z", "y", "states")
    assert tuple(inspect.signature(Density.log_prob_x).parameters) == ("self", "x", "y", "states")
    assert tuple(inspect.signature(Density.sample_z).parameters) == ("self", "y", "states", "n", "generator")
    model = Density("measurement", 5, 18)
    with pytest.raises(TypeError):
        model.sample_z(torch.zeros(1, 5), torch.zeros(1, 2, 18), 1, topology=0)
    with pytest.raises(TypeError):
        model.c(torch.zeros(1, 5), torch.zeros(1, 2, 18), truth=torch.zeros(1, 2))


def test_mdn_does_not_claim_a_single_invertible_flow():
    model = Density("mdn", 5, 18).double()
    z, y, states = _inputs()
    with pytest.raises(ValueError, match="only for flow"):
        model.forward_to_base(z, y, states)
    with pytest.raises(ValueError, match="only for flow"):
        model.inverse_from_base(z, y, states)

@pytest.mark.parametrize('kind',KINDS)
def test_encoded_context_density_agrees_with_public_density(kind):
    model=Density(kind,5,18).double()
    if kind!='mdn':_nontrivial_flow(model)
    z,y,states=_inputs()
    x=3*torch.sigmoid(z)-1.5
    context=model.c(y,states)
    torch.testing.assert_close(model.log_prob_z_context(z,context),model.log_prob_z(z,y,states))
    torch.testing.assert_close(model.log_prob_x_context(x,context),model.log_prob_x(x,y,states))

@pytest.mark.parametrize('kind',['mdn_state','mdn_memory'])
@pytest.mark.parametrize('state_dim',[18,38])
def test_mdn_state_and_memory_are_matched_normalized_unlabeled_heads(kind,state_dim):
    model=Density(kind,5,state_dim).double()
    assert abs(model.parameter_count/30000-1)<=.05
    assert model.is_mdn and len(model.couplings)==0
    z,y,states=_inputs(batch=3,state_dim=state_dim)
    c=model.c(y,states)
    logits,means,logs=model._mixture_parameters(c)
    dist=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(means,logs.exp()),1))
    torch.testing.assert_close(model.log_prob_z(z,y,states),dist.log_prob(z))
    sample=model.sample_z(y,states,32,torch.Generator().manual_seed(17))
    assert sample.shape==(3,32,2) and torch.isfinite(sample).all()
    changed=states.clone();changed[:,-1]+=1
    assert not torch.allclose(c,model.c(y,changed))
    (-model.log_prob_z(z,y,states).mean()).backward()
    assert model.encoder[0].weight.grad.abs().sum()>0
