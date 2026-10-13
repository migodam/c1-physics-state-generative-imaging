"""Full-covariance conditional MDN on C1's unchanged open material support.

The eight mixture components are exchangeable density components, not material
or topology classes. Inputs are measurement vectors and optional already saved
candidate features. This module creates no physical states and calls no solver.

The model is normalized with respect to two-dimensional Lebesgue measure in z.
Material coordinates x = 3 sigmoid(z) - 1.5 have the corresponding normalized
pushforward density, including its exact Jacobian. Numerical normalization over
a finite quadrature grid remains an evaluation concern, not a change of prior.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_CONTEXT_DIM = 32
_MEMORY_DIM = 16
_COMPONENTS = 8
_RHO_BOUND = 0.995
_LOG_2PI = math.log(2.0 * math.pi)
_BOUND = 1.5
_WIDTH = 3.0


class FullCovDensity(nn.Module):
    """Compact full-covariance MDN with measurement/state/memory conditioning.

    ``state`` uses only the final supplied state; ``memory`` encodes both saved
    states with a 16-dimensional GRU. Supplied features are detached, while the
    feature encoders and mixture parameters remain trainable. Sampling encodes
    the supplied context once, independently of subsequent production draws.

    Each component predicts a logit, two means, two bounded log standard
    deviations, and rho = 0.995 tanh(raw_rho). Thus all covariance matrices are
    strictly positive definite for finite network output. Correlation rows are
    initialized to zero to begin in the nested diagonal-Gaussian family.
    Encoder widths are selected from actual connected parameters, not padding.
    """

    def __init__(
        self,
        condition: str,
        y_dim: int,
        state_dim: int,
        target_parameters: int = 30000,
    ) -> None:
        super().__init__()
        if condition not in {"measurement", "state", "memory"}:
            raise ValueError("condition must be measurement, state, or memory")
        for name, value in (("y_dim", y_dim), ("state_dim", state_dim),
                            ("target_parameters", target_parameters)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.condition = condition
        self.kind = "fullcov" if condition == "measurement" else f"fullcov_{condition}"
        self.is_mdn = True
        self.uses_state = condition == "state"
        self.uses_memory = condition == "memory"
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.target_parameters = target_parameters

        encoder_input = y_dim
        memory_parameters = 0
        if self.uses_state:
            encoder_input += state_dim
        elif self.uses_memory:
            encoder_input += _MEMORY_DIM
            memory_parameters = 3 * _MEMORY_DIM * (state_dim + _MEMORY_DIM + 2)
        head_parameters = (_CONTEXT_DIM + 1) * (_COMPONENTS * 6)
        fixed_parameters = head_parameters + memory_parameters
        self.encoder_hidden = self._choose_width(encoder_input, fixed_parameters)

        self.memory = (
            nn.GRU(state_dim, _MEMORY_DIM, batch_first=True, device="cpu")
            if self.uses_memory else None
        )
        hidden = self.encoder_hidden
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, hidden, device="cpu"),
            nn.SiLU(),
            nn.Linear(hidden, hidden, device="cpu"),
            nn.SiLU(),
            nn.Linear(hidden, _CONTEXT_DIM, device="cpu"),
        )
        self.mdn_head = nn.Linear(_CONTEXT_DIM, _COMPONENTS * 6, device="cpu")
        with torch.no_grad():
            self.mdn_head.weight[5::6].zero_()
            self.mdn_head.bias[5::6].zero_()
        if abs(self.parameter_count - target_parameters) > 0.05 * target_parameters:
            raise ValueError("the fixed architecture cannot meet the parameter target")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def _choose_width(self, input_dim: int, fixed_parameters: int) -> int:
        # The encoder contributes h^2 + (input_dim + 34)h + 32 parameters.
        linear = input_dim + _CONTEXT_DIM + 2
        remaining = self.target_parameters - fixed_parameters - _CONTEXT_DIM
        root = (-linear + math.sqrt(max(0, linear * linear + 4 * remaining))) / 2
        lower = max(1, math.floor(root))

        def count(width: int) -> int:
            return fixed_parameters + width * width + linear * width + _CONTEXT_DIM

        width = min((lower, lower + 1),
                    key=lambda h: (abs(count(h) - self.target_parameters), h))
        if abs(count(width) - self.target_parameters) > 0.05 * self.target_parameters:
            raise ValueError("the fixed architecture cannot meet the parameter target")
        return width

    def _check_context(self, y: torch.Tensor, states: torch.Tensor) -> None:
        if not isinstance(y, torch.Tensor) or not isinstance(states, torch.Tensor):
            raise TypeError("y and states must be torch tensors")
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            raise ValueError(f"y must have shape [N, {self.y_dim}]")
        if states.shape != (y.shape[0], 2, self.state_dim):
            raise ValueError(f"states must have shape [N, 2, {self.state_dim}]")
        if y.device.type != "cpu" or states.device.type != "cpu":
            raise ValueError("C1-P1 densities require CPU tensors")
        dtype = self.encoder[0].weight.dtype
        if y.dtype != dtype or states.dtype != dtype:
            raise ValueError("y and states must have the model's floating dtype")

    def _check_encoded(self, points: torch.Tensor, context: torch.Tensor) -> None:
        if not isinstance(points, torch.Tensor) or not isinstance(context, torch.Tensor):
            raise TypeError("coordinates and context must be torch tensors")
        if context.ndim != 2 or context.shape[1] != _CONTEXT_DIM:
            raise ValueError("context must have shape [N, 32]")
        if points.shape != (context.shape[0], 2):
            raise ValueError("coordinates must have shape [N, 2]")
        if points.device.type != "cpu" or context.device.type != "cpu":
            raise ValueError("coordinates and context must be CPU tensors")
        if points.dtype != context.dtype or context.dtype != self.mdn_head.weight.dtype:
            raise ValueError("coordinates and context must have the model's floating dtype")

    def c(self, y: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Encode saved features without reading any production material draw."""
        self._check_context(y, states)
        features = y.detach()
        if self.uses_state:
            features = torch.cat((features, states[:, -1].detach()), dim=-1)
        elif self.uses_memory:
            if y.shape[0] == 0:
                summary = states.new_empty((0, _MEMORY_DIM))
            else:
                _, hidden = self.memory(states.detach())
                summary = hidden[-1]
            features = torch.cat((features, summary), dim=-1)
        return self.encoder(features)

    def _mixture_parameters(self, context: torch.Tensor):
        raw = self.mdn_head(context).reshape(-1, _COMPONENTS, 6)
        logits = raw[:, :, 0]
        means = raw[:, :, 1:3]
        log_stds = 5.0 * torch.tanh(raw[:, :, 3:5] / 5.0)
        rho = _RHO_BOUND * torch.tanh(raw[:, :, 5])
        return logits, means, log_stds, rho

    def log_prob_z_context(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Normalized bivariate-mixture log density for already encoded context."""
        self._check_encoded(z, context)
        logits, means, log_stds, rho = self._mixture_parameters(context)
        standardized = (z[:, None, :] - means) * (-log_stds).exp()
        u, v = standardized.unbind(dim=-1)
        conditional_variance = 1.0 - rho.square()
        # This Cholesky form avoids subtraction of two similar squared terms.
        quadratic = u.square() + (v - rho * u).square() / conditional_variance
        component_log_prob = (
            -0.5 * quadratic - log_stds.sum(dim=-1)
            -0.5 * torch.log1p(-rho.square()) - _LOG_2PI
        )
        return torch.logsumexp(F.log_softmax(logits, dim=-1) + component_log_prob, dim=-1)

    def log_prob_z(self, z: torch.Tensor, y: torch.Tensor,
                   states: torch.Tensor) -> torch.Tensor:
        return self.log_prob_z_context(z, self.c(y, states))

    def log_prob_x_context(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Material density; boundary/outside points have log probability -inf.

        No clipping, projection, resampling, or point-mass correction is applied.
        For accurate near-boundary evaluation, convert the model and inputs to
        float64; already rounded float32 endpoints cannot be recovered here.
        """
        self._check_encoded(x, context)
        inside = ((x > -_BOUND) & (x < _BOUND)).all(dim=-1)
        safe_x = torch.where(inside[:, None], x, torch.zeros_like(x))
        lower_distance = safe_x + _BOUND
        upper_distance = _BOUND - safe_x
        z = lower_distance.log() - upper_distance.log()
        log_jacobian = (
            math.log(_WIDTH) - lower_distance.log() - upper_distance.log()
        ).sum(dim=-1)
        result = self.log_prob_z_context(z, context) + log_jacobian
        return torch.where(inside, result, torch.full_like(result, -torch.inf))

    def log_prob_x(self, x: torch.Tensor, y: torch.Tensor,
                   states: torch.Tensor) -> torch.Tensor:
        return self.log_prob_x_context(x, self.c(y, states))

    def sample_z(
        self,
        y: torch.Tensor,
        states: torch.Tensor,
        n: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw [N, n, 2] using the component Cholesky factor, with CPU RNG."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        context = self.c(y, states)
        batch = y.shape[0]
        if n == 0 or batch == 0:
            return context.new_empty((batch, n, 2))
        logits, means, log_stds, rho = self._mixture_parameters(context)
        component = torch.multinomial(logits.softmax(dim=-1), n, replacement=True,
                                      generator=generator)
        indices = component[:, :, None].expand(-1, -1, 2)
        selected_means = means.gather(1, indices)
        selected_stds = log_stds.exp().gather(1, indices)
        selected_rho = rho.gather(1, component)
        eps = torch.randn((batch, n, 2), dtype=context.dtype, device="cpu",
                          generator=generator)
        correlated = torch.stack(
            (eps[:, :, 0], selected_rho * eps[:, :, 0]
             + torch.sqrt(1.0 - selected_rho.square()) * eps[:, :, 1]), dim=-1
        )
        return selected_means + selected_stds * correlated


def numpy_parameters(model: nn.Module, Y, S) -> dict[str, np.ndarray]:
    """Export weights/means/stds/rho as float64 NumPy arrays without sampling.

    Supports this full-covariance model and the unchanged legacy ``Density``
    MDN family. Legacy diagonal mixtures receive rho=0. No flow-to-mixture
    approximation is performed. Y and S may be NumPy arrays or CPU tensors.
    """
    if not getattr(model, "is_mdn", False):
        raise ValueError("numpy_parameters requires an MDN density")
    parameter = next(model.parameters())
    if parameter.device.type != "cpu":
        raise ValueError("numpy_parameters requires a CPU model")
    y = torch.as_tensor(Y, dtype=parameter.dtype, device="cpu").detach()
    states = torch.as_tensor(S, dtype=parameter.dtype, device="cpu").detach()
    with torch.no_grad():
        parameters = model._mixture_parameters(model.c(y, states))
        if len(parameters) == 3:
            logits, means, log_stds = parameters
            rho = torch.zeros_like(logits)
        elif len(parameters) == 4:
            logits, means, log_stds, rho = parameters
        else:
            raise ValueError("unsupported MDN parameter structure")
        return {
            "weights": logits.double().softmax(dim=-1).numpy(),
            "means": means.double().numpy(),
            "stds": log_stds.double().exp().numpy(),
            "rho": rho.double().numpy(),
        }
