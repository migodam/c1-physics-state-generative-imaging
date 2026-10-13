"""Normalized bounded Gaussian-mixture densities for a common material target.

The model learns density with respect to Lebesgue measure in the continuous
material latent x.  Exchangeable mixture components are not topology classes.
The only condition is a measured vector and declared measurement metadata.

For each coordinate, x = lower + width * sigmoid(z).  The Gaussian-mixture
density in z is normalized, and the exact logistic Jacobian gives its density
in x.  No clipping, truncated-Gaussian approximation, or numerical grid
renormalization changes this law.  A numerical endpoint caused by rounding is
reported together with its original z coordinate, rather than repaired.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from scipy.special import expit, logsumexp
import torch
from torch import nn
from torch.nn import functional as F


LOWER = (-3.0, -3.0, 2.2, 0.04)
UPPER = (1.0, 1.0, 2.8, 0.16)
DEFAULT_SEEDS = (26101321, 26101322, 26101323)
METHODS = ("full_distill", "diag_distill", "gaussian_distill", "full_joint", "diag_joint")
_LOG_2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class MaterialDraws:
    """Samples with their exact transformed-coordinate provenance retained.

    ``latent`` may contain a rounded support endpoint in floating arithmetic.
    Such a draw is not an atom in the mathematical density.  Evaluate density
    using ``transformed`` when preserving very extreme tails matters.
    """

    latent: torch.Tensor
    transformed: torch.Tensor
    component: torch.Tensor
    rounded_endpoint: torch.Tensor


def _bounds(lower, upper) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    if (lo.ndim != 1 or hi.shape != lo.shape or not len(lo)
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo >= hi)):
        raise ValueError("finite, strictly increasing coordinate bounds required")
    return lo, hi


def to_transformed(latent, lower=LOWER, upper=UPPER) -> np.ndarray:
    x = np.asarray(latent, dtype=np.float64)
    lo, hi = _bounds(lower, upper)
    if x.shape[-1] != len(lo):
        raise ValueError("latent dimension mismatch")
    if not np.all((x > lo) & (x < hi)):
        raise ValueError("interior latent points required; endpoints are not clipped")
    return np.log(x - lo) - np.log(hi - x)


def from_transformed(z, lower=LOWER, upper=UPPER) -> dict[str, np.ndarray]:
    raw = np.asarray(z, dtype=np.float64)
    lo, hi = _bounds(lower, upper)
    if raw.shape[-1] != len(lo) or not np.isfinite(raw).all():
        raise ValueError("finite transformed coordinates with matching dimension required")
    x = lo + (hi - lo) * expit(raw)
    return {"latent": x, "transformed": raw.copy(),
            "rounded_endpoint": np.any((x <= lo) | (x >= hi), axis=-1)}


class BoundedMDN(nn.Module):
    """Dimension-aware CPU MDN with two 96-unit measurement encoder layers.

    ``covariance='full'`` uses a triangular Cholesky factor with bounded positive
    diagonal and unconstrained finite off-diagonal entries.  Diagonal covariance
    is its explicit nested baseline.  ``components=1`` is a single Gaussian.
    Feature centering and scaling are fitted on training observations only and
    stored as model buffers.  No material, topology, state or memory conditions
    are accepted by the public interface.
    """

    def __init__(self, condition_dim=34, lower=LOWER, upper=UPPER,
                 covariance="full", components=8, hidden=96,
                 target_identity=""):
        super().__init__()
        lo, hi = _bounds(lower, upper)
        if covariance not in {"full", "diagonal"}:
            raise ValueError("covariance must be full or diagonal")
        for name, value in (("condition_dim", condition_dim), ("components", components),
                            ("hidden", hidden)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        self.condition_dim, self.dimension = condition_dim, len(lo)
        self.components, self.covariance, self.hidden = components, covariance, hidden
        self.target_identity = str(target_identity)
        # Semantic bounds remain FP64 even when the trainable encoder is FP32.
        # Otherwise 0.04 and 2.2 would silently change the declared support.
        self.register_buffer("lower", torch.as_tensor(lo, dtype=torch.float64))
        self.register_buffer("upper", torch.as_tensor(hi, dtype=torch.float64))
        self.register_buffer("condition_mean", torch.zeros(condition_dim, device="cpu"))
        self.register_buffer("condition_scale", torch.ones(condition_dim, device="cpu"))
        self.register_buffer("normalization_fitted", torch.tensor(False, device="cpu"))
        self.encoder = nn.Sequential(nn.Linear(condition_dim, hidden, device="cpu"), nn.SiLU(),
                                     nn.Linear(hidden, hidden, device="cpu"), nn.SiLU())
        self.covariance_size = (self.dimension * (self.dimension + 1) // 2
                                if covariance == "full" else self.dimension)
        self.component_size = 1 + self.dimension + self.covariance_size
        self.head = nn.Linear(hidden, components * self.component_size, device="cpu")
        rows, cols = torch.tril_indices(self.dimension, self.dimension)
        self.register_buffer("tril_rows", rows)
        self.register_buffer("tril_cols", cols)
        self.register_buffer("diagonal_entries", rows == cols)
        with torch.no_grad():
            raw = self.head.bias.reshape(components, self.component_size)
            raw[:, 0].zero_()
            raw[:, 1:1 + self.dimension].zero_()
            raw[:, 1 + self.dimension:].zero_()
            if covariance == "full":
                raw[:, 1 + self.dimension:][:, self.diagonal_entries] = 0.3
                # Begin in the diagonal family; correlations remain trainable.
                indices = torch.arange(components)[:, None] * self.component_size
                indices = indices + 1 + self.dimension + torch.arange(self.covariance_size)
                off_diagonal = indices[:, ~self.diagonal_entries].ravel()
                self.head.weight[off_diagonal].zero_()
            else:
                raw[:, 1 + self.dimension:] = 0.3

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit_normalization(self, train_condition):
        """Fit once using the explicitly supplied training split only."""
        if bool(self.normalization_fitted):
            raise RuntimeError("condition normalization is already fitted")
        self._check_condition(train_condition)
        if not len(train_condition):
            raise ValueError("nonempty training split required")
        with torch.no_grad():
            self.condition_mean.copy_(train_condition.mean(0))
            self.condition_scale.copy_(train_condition.std(0, unbiased=False).clamp_min(1e-8))
            self.normalization_fitted.fill_(True)

    def _check_condition(self, condition):
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch tensor")
        if condition.ndim != 2 or condition.shape[1] != self.condition_dim:
            raise ValueError(f"condition must have shape [N, {self.condition_dim}]")
        if condition.device.type != "cpu" or condition.dtype != self.head.weight.dtype:
            raise ValueError("condition must be CPU and use the model floating dtype")
        if not torch.isfinite(condition).all():
            raise ValueError("condition must be finite")

    def mixture_parameters(self, condition):
        self._check_condition(condition)
        encoded = self.encoder((condition.detach() - self.condition_mean) / self.condition_scale)
        raw = self.head(encoded).reshape(-1, self.components, self.component_size)
        log_weights = raw[:, :, 0].log_softmax(-1)
        means = raw[:, :, 1:1 + self.dimension]
        cov_raw = raw[:, :, 1 + self.dimension:]
        cholesky = cov_raw.new_zeros((len(condition), self.components,
                                     self.dimension, self.dimension))
        if self.covariance == "full":
            values = cov_raw.clone()
            diagonal = values[:, :, self.diagonal_entries]
            values[:, :, self.diagonal_entries] = (5 * torch.tanh(diagonal / 5)).exp()
            cholesky[:, :, self.tril_rows, self.tril_cols] = values
        else:
            diagonal = torch.arange(self.dimension)
            cholesky[:, :, diagonal, diagonal] = (5 * torch.tanh(cov_raw / 5)).exp()
        return log_weights, means, cholesky

    def _shape_points(self, points, batch):
        if not isinstance(points, torch.Tensor) or points.device.type != "cpu":
            raise TypeError("latent coordinates must be CPU torch tensors")
        if points.dtype not in {torch.float32, torch.float64} or points.shape[-1] != self.dimension:
            raise ValueError("latent dimension/dtype must match the model")
        if points.ndim == 2 and points.shape[0] == batch:
            return points[:, None, :], True
        if points.ndim == 2 and batch == 1:
            return points[None, :, :], False
        if points.ndim == 3 and points.shape[0] == batch:
            return points, False
        raise ValueError("latent must be [B,D] or [B,N,D], or [N,D] for one condition")

    def _log_prob_parameters(self, z, parameters):
        log_weights, means, cholesky = parameters
        difference = z.to(means.dtype)[:, :, None, :] - means[:, None, :, :]
        solved = torch.linalg.solve_triangular(cholesky[:, None, :, :, :],
                                                difference.unsqueeze(-1), upper=False).squeeze(-1)
        logdet = cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        component = (-0.5 * solved.square().sum(-1) - logdet[:, None, :]
                     - 0.5 * self.dimension * _LOG_2PI)
        return torch.logsumexp(log_weights[:, None, :] + component, dim=-1)

    def log_prob_transformed(self, z, condition):
        """Normalized Gaussian-mixture density with respect to dz."""
        parameters = self.mixture_parameters(condition)
        points, paired = self._shape_points(z, len(condition))
        result = self._log_prob_parameters(points, parameters)
        return result[:, 0] if paired else (result[0] if z.ndim == 2 else result)

    def log_prob(self, latent, condition):
        """Normalized material-latent density with respect to dx.

        Exact boundaries, outside support and NaNs have log density -inf.  They
        are never moved to interior material points.  A draw's retained raw z
        can be evaluated through ``log_prob_draws`` despite machine rounding.
        """
        parameters = self.mixture_parameters(condition)
        points, paired = self._shape_points(latent, len(condition))
        inside = ((points > self.lower) & (points < self.upper)).all(-1)
        safe = torch.where(inside[..., None], points, (self.lower + self.upper) / 2)
        low_distance, high_distance = safe - self.lower, self.upper - safe
        z = low_distance.log() - high_distance.log()
        jacobian = ((self.upper - self.lower).log() - low_distance.log()
                    - high_distance.log()).sum(-1)
        result = self._log_prob_parameters(z, parameters) + jacobian
        result = torch.where(inside, result, torch.full_like(result, -torch.inf))
        return result[:, 0] if paired else (result[0] if latent.ndim == 2 else result)

    def log_prob_draws(self, draws: MaterialDraws, condition):
        """dx log density evaluated from retained z, without endpoint inversion."""
        z = draws.transformed
        log_dx_dz = ((self.upper - self.lower).log() - F.softplus(z) - F.softplus(-z)).sum(-1)
        return self.log_prob_transformed(z, condition) - log_dx_dz

    def sample(self, condition, n, generator=None) -> MaterialDraws:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        log_weights, means, cholesky = self.mixture_parameters(condition)
        batch = len(condition)
        if n == 0 or batch == 0:
            z = means.new_empty((batch, n, self.dimension))
            components = torch.empty((batch, n), dtype=torch.long)
        else:
            components = torch.multinomial(log_weights.exp(), n, replacement=True,
                                           generator=generator)
            rows = torch.arange(batch)[:, None]
            selected_means = means[rows, components]
            selected_cholesky = cholesky[rows, components]
            eps = torch.randn((batch, n, self.dimension), dtype=means.dtype,
                              device="cpu", generator=generator)
            z = selected_means + (selected_cholesky @ eps[..., None]).squeeze(-1)
        # Decode in FP64 even when the encoder is FP32; retain z regardless.
        raw = z.double()
        lo, hi = self.lower.double(), self.upper.double()
        latent = lo + (hi - lo) * raw.sigmoid()
        rounded = ((latent <= lo) | (latent >= hi)).any(-1)
        return MaterialDraws(latent, z, components, rounded)

    def export_parameters(self, condition) -> dict[str, np.ndarray]:
        parameter = self.head.weight
        y = torch.as_tensor(condition, dtype=parameter.dtype, device="cpu")
        with torch.no_grad():
            log_weights, means, cholesky = self.mixture_parameters(y)
        return {"weights": log_weights.double().exp().numpy(),
                "log_weights": log_weights.double().numpy(),
                "means": means.double().numpy(), "cholesky": cholesky.double().numpy(),
                "lower": self.lower.double().numpy(), "upper": self.upper.double().numpy()}

    def configuration(self):
        return dict(condition_dim=self.condition_dim, lower=self.lower.tolist(),
                    upper=self.upper.tolist(), covariance=self.covariance,
                    components=self.components, hidden=self.hidden,
                    target_identity=self.target_identity)


def model_for(method, condition_dim=34, target_identity=""):
    if method not in METHODS:
        raise ValueError("unknown P2 method")
    return BoundedMDN(condition_dim=condition_dim,
                      covariance="diagonal" if method.startswith("diag") else "full",
                      components=1 if method.startswith("gaussian") else 8,
                      target_identity=target_identity)


def one_parameters(parameters: Mapping[str, np.ndarray], index: int):
    """Keep the explicit one-condition axis in a portable parameter export."""
    return {key: (np.asarray(value)[index:index + 1] if key in
                   {"weights", "log_weights", "means", "cholesky"} else np.asarray(value))
            for key, value in parameters.items()}


def log_prob_batch(points, parameters, *, transformed=False, material_density=True,
                   chunk_size=16384):
    """NumPy dx density for shared [N,D] or per-condition [B,N,D] points.

    Returns [N] for a one-condition export, otherwise [B,N].  Set transformed
    for retained z points; ``material_density=False`` then returns dz density.
    Densities are normalized analytically, never normalized on the supplied
    evaluation nodes.  This function invokes no model, reference or physics.
    """
    x = np.asarray(points, dtype=np.float64)
    means = np.asarray(parameters["means"], dtype=np.float64)
    factors = np.asarray(parameters["cholesky"], dtype=np.float64)
    log_weights = np.asarray(parameters["log_weights"], dtype=np.float64)
    lo, hi = _bounds(parameters["lower"], parameters["upper"])
    batch, components, dim = means.shape
    if factors.shape != (batch, components, dim, dim) or log_weights.shape != (batch, components):
        raise ValueError("invalid exported mixture dimensions")
    if x.ndim not in (2, 3) or x.shape[-1] != dim or (x.ndim == 3 and x.shape[0] != batch):
        raise ValueError("points must be [N,D] or [B,N,D]")
    count = x.shape[-2]
    result = np.empty((batch, count), dtype=np.float64)
    inverse = np.linalg.inv(factors)
    logdet = np.log(np.diagonal(factors, axis1=-2, axis2=-1)).sum(-1)
    for b in range(batch):
        for start in range(0, count, chunk_size):
            p = x[start:start + chunk_size] if x.ndim == 2 else x[b, start:start + chunk_size]
            if transformed:
                inside = np.isfinite(p).all(-1)
                z = np.where(inside[:, None], p, 0)
                jacobian = (np.log(hi - lo) - np.logaddexp(0, z)
                            - np.logaddexp(0, -z)).sum(-1)
                jacobian = -jacobian if material_density else np.zeros(len(z))
            else:
                inside = ((p > lo) & (p < hi)).all(-1)
                safe = np.where(inside[:, None], p, (lo + hi) / 2)
                low, high = safe - lo, hi - safe
                z = np.log(low) - np.log(high)
                jacobian = (np.log(hi - lo) - np.log(low) - np.log(high)).sum(-1)
            standardized = np.einsum("kij,nkj->nki", inverse[b], z[:, None, :] - means[b])
            component_lp = (-0.5 * np.sum(standardized ** 2, axis=-1)
                            - logdet[b] - 0.5 * dim * _LOG_2PI)
            density = logsumexp(log_weights[b] + component_lp, axis=-1) + jacobian
            result[b, start:start + len(p)] = np.where(inside, density, -np.inf)
    return result[0] if batch == 1 else result


def sample_parameters(parameters, count, rng: np.random.Generator):
    """Portable NumPy samples, retaining raw z and endpoint diagnostics."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    means, factors = np.asarray(parameters["means"]), np.asarray(parameters["cholesky"])
    weights = np.asarray(parameters["weights"], dtype=np.float64)
    batch, components, dim = means.shape
    z = np.empty((batch, count, dim), dtype=np.float64)
    choices = np.empty((batch, count), dtype=np.int64)
    for b in range(batch):
        # tiny softmax roundoff is corrected solely for categorical RNG, not
        # to renormalize model density on a restricted material grid.
        probability = weights[b] / weights[b].sum()
        choices[b] = rng.choice(components, count, p=probability)
        eps = rng.standard_normal((count, dim))
        z[b] = means[b, choices[b]] + np.einsum("nij,nj->ni", factors[b, choices[b]], eps)
    decoded = from_transformed(z, parameters["lower"], parameters["upper"])
    decoded["component"] = choices
    return decoded


def radial_event_ids(latent):
    """Frozen post-generation 4×3 event partition; never an encoder feature."""
    x = np.asarray(latent, dtype=np.float64)
    if x.shape[-1] != 4:
        raise ValueError("four radial material coordinates required")
    active_c, active_s = x[..., 0] < -1, x[..., 1] < -1
    rc = np.zeros(x.shape[:-1])
    hs = np.zeros(x.shape[:-1])
    rc[active_c] = .45 * np.sqrt(1 - np.sqrt(-1 / x[..., 0][active_c]))
    hs[active_s] = .18 * np.sqrt(1 - np.sqrt(-1 / x[..., 1][active_s]))
    cbin = np.where(active_c, 1 + (rc > .10) + (rc > .20), 0)
    sbin = np.where(active_s, 1 + (hs > .06), 0)
    return (3 * cbin + sbin).astype(np.int64)
