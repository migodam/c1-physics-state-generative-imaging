"""Normalized conditional densities on the connected C1 material support.

The supplied observations and two pilot states are frozen features.  This
module neither constructs physical states nor calls a forward solver.  All
components and coupling blocks act on unlabeled continuous coordinates.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


_CONTEXT_DIM = 32
_COUPLING_HIDDEN = 48
_COUPLING_BLOCKS = 6
_MEMORY_DIM = 16
_COMPONENTS = 8
_LOG_2PI = math.log(2.0 * math.pi)
_BOUND = 1.5
_WIDTH = 2.0 * _BOUND


class _AffineCoupling(nn.Module):
    """A triangular map with one unchanged coordinate and bounded log scale."""

    def __init__(self, conditioned_coordinate: int) -> None:
        super().__init__()
        self.conditioned_coordinate = conditioned_coordinate
        self.conditioner = nn.Sequential(
            nn.Linear(_CONTEXT_DIM + 1, _COUPLING_HIDDEN, device="cpu"),
            nn.SiLU(),
            nn.Linear(_COUPLING_HIDDEN, _COUPLING_HIDDEN, device="cpu"),
            nn.SiLU(),
            nn.Linear(_COUPLING_HIDDEN, 2, device="cpu"),
        )
        # The full flow starts at the identity, including its determinant.
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.zeros_(self.conditioner[-1].bias)

    def _parameters_for(self, points: torch.Tensor, context: torch.Tensor):
        unchanged = points[:, self.conditioned_coordinate : self.conditioned_coordinate + 1]
        raw = self.conditioner(torch.cat((unchanged, context), dim=-1))
        log_scale = 2.0 * torch.tanh(raw[:, 0])
        return log_scale, raw[:, 1]

    def forward(self, points: torch.Tensor, context: torch.Tensor):
        log_scale, shift = self._parameters_for(points, context)
        coordinate = 1 - self.conditioned_coordinate
        changed = points[:, coordinate] * log_scale.exp() + shift
        if coordinate == 0:
            result = torch.stack((changed, points[:, 1]), dim=-1)
        else:
            result = torch.stack((points[:, 0], changed), dim=-1)
        return result, log_scale

    def inverse(self, points: torch.Tensor, context: torch.Tensor):
        log_scale, shift = self._parameters_for(points, context)
        coordinate = 1 - self.conditioned_coordinate
        changed = (points[:, coordinate] - shift) * (-log_scale).exp()
        if coordinate == 0:
            result = torch.stack((changed, points[:, 1]), dim=-1)
        else:
            result = torch.stack((points[:, 0], changed), dim=-1)
        return result, -log_scale


class Density(nn.Module):
    """C1's MDN and three conditional affine flows, using CPU tensors.

    ``mdn`` and ``measurement`` encode the entire ``y`` vector.  ``state``
    appends the last pilot state, and ``memory`` appends a GRU summary of both
    supplied states.  Raw conditioning inputs are detached to keep physical
    context fixed; gradients still train the encoders and density parameters.

    The flow convention is ``eps = forward_to_base(z)``.  Its reported log
    determinant is therefore added to the standard-normal base log density.
    Encoder width is chosen from real connected weights, with no unused
    parameter padding, to stay within 5% of ``target_parameters``.
    """

    def __init__(
        self,
        kind: str,
        y_dim: int,
        state_dim: int,
        target_parameters: int = 30000,
    ) -> None:
        super().__init__()
        if kind not in {"mdn", "measurement", "state", "memory", "mdn_state", "mdn_memory"}:
            raise ValueError("kind must be mdn, measurement, state, or memory")
        for name, value in (("y_dim", y_dim), ("state_dim", state_dim),
                            ("target_parameters", target_parameters)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.kind = kind
        self.is_mdn = kind.startswith('mdn')
        self.uses_state = kind in {'state','mdn_state'}
        self.uses_memory = kind in {'memory','mdn_memory'}
        self.y_dim = y_dim
        self.state_dim = state_dim
        self.target_parameters = target_parameters

        encoder_input = y_dim
        memory_parameters = 0
        if self.uses_state:
            encoder_input += state_dim
        elif self.uses_memory:
            encoder_input += _MEMORY_DIM
            # One GRU layer: input weights, recurrent weights, and two biases.
            memory_parameters = 3 * _MEMORY_DIM * (state_dim + _MEMORY_DIM + 2)

        if self.is_mdn:
            density_parameters = (_CONTEXT_DIM + 1) * (_COMPONENTS * 5)
        else:
            per_block = (
                (_CONTEXT_DIM + 2) * _COUPLING_HIDDEN
                + (_COUPLING_HIDDEN + 1) * _COUPLING_HIDDEN
                + (_COUPLING_HIDDEN + 1) * 2
            )
            density_parameters = _COUPLING_BLOCKS * per_block
        fixed_parameters = density_parameters + memory_parameters
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
        self.mdn_head = (
            nn.Linear(_CONTEXT_DIM, _COMPONENTS * 5, device="cpu")
            if self.is_mdn else None
        )
        self.couplings = nn.ModuleList(
            [] if self.is_mdn else
            [_AffineCoupling(block % 2) for block in range(_COUPLING_BLOCKS)]
        )
        if abs(self.parameter_count - target_parameters) > 0.05 * target_parameters:
            raise ValueError("the fixed architecture cannot meet the parameter target")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def _choose_width(self, input_dim: int, fixed_parameters: int) -> int:
        # Two hidden layers and a context output use h^2 + (d + 34)h + 32.
        linear = input_dim + _CONTEXT_DIM + 2
        remaining = self.target_parameters - fixed_parameters - _CONTEXT_DIM
        root = (-linear + math.sqrt(max(0, linear * linear + 4 * remaining))) / 2
        lower = max(1, math.floor(root))
        widths = (lower, lower + 1)

        def count(width: int) -> int:
            return fixed_parameters + width * width + linear * width + _CONTEXT_DIM

        width = min(widths, key=lambda candidate: (abs(count(candidate) - self.target_parameters), candidate))
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
            raise ValueError("C1 densities require CPU tensors")
        dtype = self.encoder[0].weight.dtype
        if y.dtype != dtype or states.dtype != dtype:
            raise ValueError("y and states must have the model's floating dtype")

    def _check_points(self, points: torch.Tensor, y: torch.Tensor) -> None:
        if not isinstance(points, torch.Tensor):
            raise TypeError("coordinates must be a torch tensor")
        if points.shape != (y.shape[0], 2):
            raise ValueError("coordinates must have shape [N, 2]")
        if points.device.type != "cpu" or points.dtype != y.dtype:
            raise ValueError("coordinates must have the context's CPU floating dtype")

    def c(self, y: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Encode frozen supplied features, without using a production draw."""
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

    def _forward_with_context(self, z: torch.Tensor, context: torch.Tensor):
        result = z
        logdet = z.new_zeros(z.shape[0])
        for coupling in self.couplings:
            result, change = coupling(result, context)
            logdet = logdet + change
        return result, logdet

    def _inverse_with_context(self, eps: torch.Tensor, context: torch.Tensor):
        result = eps
        logdet = eps.new_zeros(eps.shape[0])
        for coupling in reversed(self.couplings):
            result, change = coupling.inverse(result, context)
            logdet = logdet + change
        return result, logdet

    def forward_to_base(self, z: torch.Tensor, y: torch.Tensor, states: torch.Tensor):
        """Return base coordinates and log|d eps / d z| for a flow."""
        if self.is_mdn:
            raise ValueError("forward_to_base is available only for flow models")
        context = self.c(y, states)
        self._check_points(z, y)
        return self._forward_with_context(z, context)

    def inverse_from_base(self, eps: torch.Tensor, y: torch.Tensor, states: torch.Tensor):
        """Return continuous coordinates and log|d z / d eps| for a flow."""
        if self.is_mdn:
            raise ValueError("inverse_from_base is available only for flow models")
        context = self.c(y, states)
        self._check_points(eps, y)
        return self._inverse_with_context(eps, context)

    def _mixture_parameters(self, context: torch.Tensor):
        raw = self.mdn_head(context).reshape(-1, _COMPONENTS, 5)
        logits = raw[:, :, 0]
        means = raw[:, :, 1:3]
        # Smoothly bound standard deviations without changing normalization.
        log_stds = 5.0 * torch.tanh(raw[:, :, 3:5] / 5.0)
        return logits, means, log_stds

    def log_prob_z(self, z: torch.Tensor, y: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Evaluate a normalized density on unbounded logit coordinates."""
        context = self.c(y, states)
        self._check_points(z, y)
        return self.log_prob_z_context(z, context)

    def log_prob_z_context(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Use an already encoded frozen context for many integration points."""
        if self.is_mdn:
            logits, means, log_stds = self._mixture_parameters(context)
            standardized = (z[:, None, :] - means) * (-log_stds).exp()
            component_log_prob = (
                -0.5 * standardized.square().sum(dim=-1)
                - log_stds.sum(dim=-1) - _LOG_2PI
            )
            return torch.logsumexp(F.log_softmax(logits, dim=-1) + component_log_prob, dim=-1)
        eps, logdet = self._forward_with_context(z, context)
        return -0.5 * eps.square().sum(dim=-1) - _LOG_2PI + logdet

    def log_prob_x_context(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Exact material density with encoded context; never evaluates physics."""
        inside = ((x > -_BOUND) & (x < _BOUND)).all(dim=-1)
        safe_x = torch.where(inside[:, None], x, torch.zeros_like(x))
        lower_distance=safe_x+_BOUND;upper_distance=_BOUND-safe_x
        z=lower_distance.log()-upper_distance.log()
        log_jacobian=(math.log(_WIDTH)-lower_distance.log()-upper_distance.log()).sum(dim=-1)
        result=self.log_prob_z_context(z,context)+log_jacobian
        return torch.where(inside,result,torch.full_like(result,-torch.inf))

    def log_prob_x(self, x: torch.Tensor, y: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Evaluate q(x) including the exact logit Jacobian on (-1.5, 1.5)^2.

        Points on or outside the open support have log density ``-inf``.
        Interior coordinates are never clipped or projected.
        """
        self._check_context(y, states)
        self._check_points(x, y)
        inside = ((x > -_BOUND) & (x < _BOUND)).all(dim=-1)
        safe_x = torch.where(inside[:, None], x, torch.zeros_like(x))
        lower_distance = safe_x + _BOUND
        upper_distance = _BOUND - safe_x
        z = lower_distance.log() - upper_distance.log()
        log_jacobian = (math.log(_WIDTH) - lower_distance.log() - upper_distance.log()).sum(dim=-1)
        result = self.log_prob_z(z, y, states) + log_jacobian
        return torch.where(inside, result, torch.full_like(result, -torch.inf))

    def sample_z(
        self,
        y: torch.Tensor,
        states: torch.Tensor,
        n: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw [N, n, 2] samples, encoding each observation exactly once."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        context = self.c(y, states)
        batch = y.shape[0]
        if n == 0 or batch == 0:
            return context.new_empty((batch, n, 2))
        eps = torch.randn((batch, n, 2), dtype=context.dtype, device="cpu", generator=generator)
        if self.is_mdn:
            logits, means, log_stds = self._mixture_parameters(context)
            component = torch.multinomial(logits.softmax(dim=-1), n, replacement=True, generator=generator)
            index = component[:, :, None].expand(-1, -1, 2)
            selected_means = means.gather(1, index)
            selected_stds = log_stds.exp().gather(1, index)
            return selected_means + selected_stds * eps
        expanded_context = context[:, None, :].expand(-1, n, -1).reshape(batch * n, _CONTEXT_DIM)
        samples, _ = self._inverse_with_context(eps.reshape(batch * n, 2), expanded_context)
        return samples.reshape(batch, n, 2)
