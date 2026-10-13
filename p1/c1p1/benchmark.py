"""Small solver-agnostic posterior contracts; importing this never runs physics.

Densities are with respect to the named material base measure. Quadrature
integration weights and posterior probability weights are different objects.
Events are evaluation callbacks, not generator inputs or discrete prior labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Callable, Mapping, Protocol

import numpy as np

Array = np.ndarray
LogDensity = Callable[[Array], Array]


@dataclass(frozen=True)
class TargetIdentity:
    """Versioned semantic IDs, not hashes or proof of scientific correctness."""

    material_space_id: str
    support_id: str
    base_measure_id: str
    prior_id: str
    likelihood_id: str
    noise_id: str
    nuisance_id: str
    acquisition_id: str
    solver_id: str
    reference_target_id: str
    observation_id: str

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"explicit {item.name} required")


def require_same_target(left: TargetIdentity, right: TargetIdentity) -> None:
    """Reject comparisons across targets, including a changed numerical forward."""
    mismatch = [item.name for item in fields(left)
                if getattr(left, item.name) != getattr(right, item.name)]
    if mismatch:
        raise ValueError("posterior target mismatch: " + ", ".join(mismatch))


class MaterialSupport(Protocol):
    """Allows non-box or unbounded latent supports without a solver dependency."""

    dimension: int

    def contains(self, latent: Array) -> Array: ...


@dataclass(frozen=True)
class BoxSupport:
    """A declared coordinate support; it makes no claim about general geometry."""

    lower: Array
    upper: Array
    open_boundary: bool = True

    def __post_init__(self) -> None:
        lo = np.asarray(self.lower, dtype=np.float64)
        hi = np.asarray(self.upper, dtype=np.float64)
        if (lo.ndim != 1 or hi.shape != lo.shape or lo.size == 0
                or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
                or np.any(lo >= hi)):
            raise ValueError("finite increasing coordinate bounds required")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @property
    def dimension(self) -> int:
        return len(self.lower)

    def contains(self, latent: Array) -> Array:
        x = np.asarray(latent, dtype=np.float64)
        if x.ndim == 0 or x.shape[-1] != len(self.lower):
            raise ValueError("material coordinate dimension mismatch")
        if self.open_boundary:
            return np.all((x > self.lower) & (x < self.upper), axis=-1)
        return np.all((x >= self.lower) & (x <= self.upper), axis=-1)


@dataclass(frozen=True)
class MaterialSpace:
    space_id: str
    support_id: str
    base_measure_id: str
    support: MaterialSupport
    coordinate_names: tuple[str, ...]
    decoder: Callable[[Array], Array] | None = None
    field_description: str = "latent coordinates; no field decoder supplied"

    def __post_init__(self) -> None:
        if len(self.coordinate_names) != self.support.dimension:
            raise ValueError("coordinate names must describe every latent dimension")

    def field(self, latent: Array) -> Array:
        if self.decoder is None:
            raise ValueError("material field decoder is not supplied")
        if not np.all(self.support.contains(latent)):
            raise ValueError("material outside declared support")
        return np.asarray(self.decoder(np.asarray(latent, dtype=np.float64)))


@dataclass(frozen=True)
class ProperPrior:
    """The provider supplies its normalized density and the normalization basis.

    This declaration is not a numerical or mathematical normalization proof.
    A pushforward field can be lower dimensional; its latent density must not
    be reported as a Lebesgue density over every material voxel.
    """

    prior_id: str
    base_measure_id: str
    log_prob: LogDensity
    sample: Callable[[np.random.Generator, int], Array]
    normalization_basis: str


@dataclass(frozen=True)
class MeasurementMetadata:
    likelihood_id: str
    noise_id: str
    nuisance_id: str
    acquisition_id: str
    packing: str
    units: str
    noise: Mapping[str, object] = field(default_factory=dict)
    nuisance: Mapping[str, object] = field(default_factory=dict)
    acquisition: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    observation_id: str
    values: Array
    metadata: MeasurementMetadata
    exposure: str


@dataclass(frozen=True)
class CostRecord:
    """Actual costs, including setup, failures and reused feature preparation.

    Cached historical costs can be recorded as a separate phase instead of
    being claimed free. No budget enforcement or solver is implemented here.
    """

    phase: str
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    material_frequency_evaluations: int = 0
    forward_rhs: int = 0
    derivative_rhs: int = 0
    adjoint_rhs: int = 0
    gpu_seconds: float = 0.0
    b1_physical_calls: int = 0
    measured: bool = True

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("cost phase required")
        for name in ("cpu_seconds", "wall_seconds", "gpu_seconds"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError("cost must be finite and nonnegative")
        for name in ("material_frequency_evaluations", "forward_rhs", "derivative_rhs", "adjoint_rhs",
                     "b1_physical_calls"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("physical counts must be nonnegative integers")


@dataclass(frozen=True)
class ForwardEvaluation:
    prediction: Array
    cost: CostRecord


class MaxwellForwardAdapter(Protocol):
    """Lazy, optional adapter. No C1/B1 solver is imported or reimplemented.

    An implementation must charge its own complete physical solve before use.
    P1 has zero new physical-call authorization, so it must not invoke this.
    """

    solver_id: str

    def predict(self, latent: Array, observation: Observation,
                nuisance: Array | None = None) -> ForwardEvaluation: ...


class ConditionalMaterialDensity(Protocol):
    target: TargetIdentity
    normalization_basis: str

    def log_prob(self, latent: Array, observation: Observation) -> Array: ...

    def sample(self, rng: np.random.Generator, count: int,
               observation: Observation) -> Array: ...


@dataclass(frozen=True)
class CandidateFeatures:
    candidate_latent: Array
    observation_id: str
    values: Array
    origin: str
    cost: CostRecord
    history: tuple[Array, ...] = ()

    def __post_init__(self) -> None:
        if self.origin not in {"candidate_and_observation", "saved_candidate_pilot"}:
            raise ValueError("features must come from a candidate, not oracle truth")


@dataclass(frozen=True)
class MaterialEvent:
    """A measurable post-generation event in the same material coordinates."""

    event_id: str
    description: str
    indicator: Callable[[Array], Array]


@dataclass(frozen=True)
class ContinuousReference:
    target: TargetIdentity
    reference_id: str
    log_prob: LogDensity
    normalization_basis: str
    approximation_note: str
    numerical_errors: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleReference:
    target: TargetIdentity
    reference_id: str
    samples: Array
    sampling_method: str
    approximation_note: str
    # No ESS is fabricated from the number of MCMC samples.
    validated_effective_sample_size: float | None = None


@dataclass(frozen=True)
class QuadratureReference:
    """Approximation of a continuous posterior, never a discrete material prior.

    integration_weights integrate the named base measure. log_density is the
    continuous posterior log density with respect to that measure at nodes.
    Re-normalization below belongs only to quadrature expectation estimates;
    it does not change the declared prior or model density.
    """

    target: TargetIdentity
    reference_id: str
    nodes: Array
    integration_weights: Array
    log_density: Array
    approximation_note: str
    numerical_errors: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = np.asarray(self.nodes, dtype=np.float64)
        w = np.asarray(self.integration_weights, dtype=np.float64)
        lp = np.asarray(self.log_density, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or w.shape != (len(x),) or lp.shape != w.shape:
            raise ValueError("quadrature nodes/density/weight shape mismatch")
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(w))
                or np.any(w <= 0) or np.any(np.isnan(lp)) or np.any(np.isposinf(lp))
                or not np.any(np.isfinite(lp))):
            raise ValueError("invalid continuous quadrature data")
        object.__setattr__(self, "nodes", x)
        object.__setattr__(self, "integration_weights", w)
        object.__setattr__(self, "log_density", lp)

    @property
    def log_integral(self) -> float:
        z = self.log_density + np.log(self.integration_weights)
        maximum = np.max(z)
        return float(maximum + np.log(np.exp(z - maximum).sum()))

    @property
    def probability_weights(self) -> Array:
        return np.exp(self.log_density + np.log(self.integration_weights)
                      - self.log_integral)


@dataclass(frozen=True)
class PosteriorBenchmark:
    """Bind matching definitions without loading a bank or invoking a solver."""

    target: TargetIdentity
    material_space: MaterialSpace
    prior: ProperPrior
    metadata: MeasurementMetadata
    reference: ContinuousReference | SampleReference | QuadratureReference
    forward: MaxwellForwardAdapter | None = None
    observation: Observation | None = None

    def __post_init__(self) -> None:
        expected = {
            "material_space_id": self.material_space.space_id,
            "support_id": self.material_space.support_id,
            "base_measure_id": self.material_space.base_measure_id,
            "prior_id": self.prior.prior_id,
            "likelihood_id": self.metadata.likelihood_id,
            "noise_id": self.metadata.noise_id,
            "nuisance_id": self.metadata.nuisance_id,
            "acquisition_id": self.metadata.acquisition_id,
        }
        for name, value in expected.items():
            if getattr(self.target, name) != value:
                raise ValueError("benchmark definition mismatch: " + name)
        if self.prior.base_measure_id != self.target.base_measure_id:
            raise ValueError("prior and density base measures differ")
        require_same_target(self.target, self.reference.target)
        if self.observation is not None:
            if self.observation.observation_id != self.target.observation_id:
                raise ValueError("reference/model observation_id mismatch")
            for name in ("likelihood_id", "noise_id", "nuisance_id", "acquisition_id"):
                if getattr(self.observation.metadata, name) != getattr(self.metadata, name):
                    raise ValueError("observation metadata mismatch: " + name)
        if isinstance(self.reference, QuadratureReference):
            if not np.all(self.material_space.support.contains(self.reference.nodes)):
                raise ValueError("reference node outside material support")
        if self.forward is not None and self.forward.solver_id != self.target.solver_id:
            raise ValueError("forward solver_id mismatch")


@dataclass(frozen=True)
class ProbabilityEvaluation:
    """Numerical summaries; a small error is not a calibration certificate."""

    reference_integral: float
    model_integral: float
    reference_kl: float
    conditional_log_score: float
    reference_events: Mapping[str, float]
    model_events: Mapping[str, float]
    reference_mean: Array
    model_mean_integral: Array
    reference_variance: Array
    cost: CostRecord


def evaluate_quadrature(reference: QuadratureReference, model_target: TargetIdentity,
                        model_log_prob: LogDensity,
                        events: tuple[MaterialEvent, ...] = (),
                        cost: CostRecord | None = None) -> ProbabilityEvaluation:
    """Evaluate a provided density without sampling or invoking any forward.

    Model probability is not silently normalized on the nodes: missing mass
    remains visible in model_integral/events. KL is provisional until reference
    and model integrals/log-score converge under an independent refinement.
    """
    require_same_target(reference.target, model_target)
    logq = np.asarray(model_log_prob(reference.nodes), dtype=np.float64)
    if logq.shape != reference.log_density.shape or np.any(np.isnan(logq)):
        raise ValueError("model log density must have one non-NaN value per node")
    if np.any(np.isposinf(logq)):
        raise ValueError("infinite positive node density cannot be evaluated")
    p = reference.probability_weights
    with np.errstate(over="ignore"):
        qmass = np.exp(logq + np.log(reference.integration_weights))
    if not np.all(np.isfinite(qmass)):
        raise ValueError("model quadrature mass overflow; numerical evaluation unresolved")
    positive = p > 0
    log_score = float(np.sum(p[positive] * logq[positive]))
    kl = float(np.sum(p[positive] *
                      (reference.log_density[positive] - logq[positive])))
    ref_events, model_events = {}, {}
    for event in events:
        if event.event_id in ref_events:
            raise ValueError("duplicate event ID")
        indicator = np.asarray(event.indicator(reference.nodes))
        if indicator.shape != p.shape or indicator.dtype.kind != "b":
            raise ValueError("event indicator must return one boolean per node")
        ref_events[event.event_id] = float(p[indicator].sum())
        model_events[event.event_id] = float(qmass[indicator].sum())
    mean = p @ reference.nodes
    variance = p @ ((reference.nodes - mean) ** 2)
    return ProbabilityEvaluation(
        reference_integral=float(np.exp(reference.log_integral)),
        model_integral=float(qmass.sum()), reference_kl=kl,
        conditional_log_score=log_score, reference_events=ref_events,
        model_events=model_events, reference_mean=mean,
        model_mean_integral=qmass @ reference.nodes, reference_variance=variance,
        cost=cost or CostRecord("evaluation_time_not_measured", measured=False),
    )
