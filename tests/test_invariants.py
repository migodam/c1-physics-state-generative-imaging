"""Implementation invariants, not scientific validation or an advantage claim.

The parent must execute this suite through the budgeted job launcher. A complete
single run uses 18 new material-frequency evaluations in the canonical ledger:
normalization 1, derivative base points 2, centered differences 8, support corners
4, zero contrast 1, and two candidate-pilot rounds 2. All tmp-path ledger tests
are mock accounting only: they call no physical forward or linear solver.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1.accounting import PhysicsLedger
from c1 import material, problems
from c1.probability import (
    gaussian_loglike,
    proper_complex_whiten,
    split_weight,
    tempering_weights,
)

PROTOCOL = json.loads((ROOT / "configs/protocol_v2.json").read_text())


def test_mock_cap_refuses_before_body(tmp_path):
    ledger = PhysicsLedger(tmp_path, cap=0)
    entered = False
    with pytest.raises(RuntimeError, match="before solve"):
        with ledger.reserve("mock_would_solve"):
            entered = True
    assert not entered
    assert ledger.summary()["used"] == 0
    assert [r["event"] for r in ledger.rows()] == ["budget_refusal"]


def test_mock_failure_is_charged_and_rethrown(tmp_path):
    ledger = PhysicsLedger(tmp_path, cap=3)
    with pytest.raises(ArithmeticError, match="mock failure"):
        with ledger.reserve("mock_failure", material_frequency=2,
                            forward_rhs=3, derivative_rhs=4, adjoint_rhs=1):
            raise ArithmeticError("mock failure")
    rows = ledger.rows()
    assert [r["event"] for r in rows] == ["charge", "result"]
    assert rows[1]["status"] == "FAILED_CHARGED"
    assert rows[1]["id"] == rows[0]["id"]
    assert rows[1]["error"] == "ArithmeticError"
    assert rows[1]["cpu_seconds"] >= 0 and rows[1]["wall_seconds"] >= 0
    assert ledger.summary() == {
        "cap": 3, "used": 2, "forward_rhs": 3, "derivative_rhs": 4,
        "adjoint_rhs": 1, "failures": 1, "by_kind": {"mock_failure": 2},
    }


def test_mock_reopened_ledger_preserves_cost_and_cap(tmp_path):
    first = PhysicsLedger(tmp_path, cap=2)
    with first.reserve("mock_first", forward_rhs=0):
        pass
    reopened = PhysicsLedger(tmp_path, cap=2)
    assert reopened.summary()["used"] == 1
    with reopened.reserve("mock_second", forward_rhs=0):
        pass
    with pytest.raises(RuntimeError, match="before solve"):
        with reopened.reserve("mock_forbidden", forward_rhs=0):
            pytest.fail("completed cost was reset")
    assert reopened.summary()["used"] == 2
    charges = [r for r in reopened.rows() if r["event"] == "charge"]
    assert len({r["id"] for r in charges}) == 2


def test_mock_stale_instances_and_repeated_refusals_do_not_reset_or_double_count(tmp_path):
    first = PhysicsLedger(tmp_path, cap=1)
    stale = PhysicsLedger(tmp_path, cap=1)
    with first.reserve("mock_first", forward_rhs=0):
        pass
    for _ in range(2):
        with pytest.raises(RuntimeError, match="before solve"):
            with stale.reserve("mock_stale_refused", forward_rhs=0):
                pytest.fail("a stale handle bypassed the cap")
    rows = stale.rows()
    assert len([r for r in rows if r["event"] == "charge"]) == 1
    assert [r["used"] for r in rows if r["event"] == "budget_refusal"] == [1, 1]
    assert stale.summary()["used"] == 1


@pytest.mark.parametrize("field", ["material_frequency", "forward_rhs", "derivative_rhs", "adjoint_rhs"])
def test_mock_negative_cost_cannot_create_budget(tmp_path, field):
    ledger = PhysicsLedger(tmp_path, cap=1)
    with pytest.raises(ValueError, match="negative"):
        with ledger.reserve("mock_negative", **{field: -1}):
            pytest.fail("negative costs were accepted")
    assert ledger.summary()["used"] == 0


def test_mock_live_handle_detects_ledger_truncation(tmp_path):
    ledger = PhysicsLedger(tmp_path, cap=2)
    with ledger.reserve("mock_first", forward_rhs=0):
        pass
    ledger.path.write_text("")
    with pytest.raises(RuntimeError, match="truncation"):
        with ledger.reserve("mock_after_truncation", forward_rhs=0):
            pytest.fail("a live handle accepted a truncated ledger")


def test_duplicate_job_id_refused_before_any_process_launch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("c1_invariant_job", ROOT / "tools/job.py")
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "configs/protocol_v2.json").write_text(json.dumps(PROTOCOL))
    fake_ledger = tmp_path / "results/JOB_LEDGER.jsonl"
    previous = json.dumps({"id": "already-used", "status": "FAILED_CHARGED"}) + "\n"
    fake_ledger.write_text(previous)
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["job.py", "--id", "already-used", "--stage", "tests", "--", "never-launch"])

    def forbidden_launch(*args, **kwargs):
        pytest.fail("duplicate ID reached subprocess launch")

    monkeypatch.setattr(job.subprocess, "run", forbidden_launch)
    with pytest.raises(SystemExit, match="duplicate run ID refused"):
        job.main()
    assert fake_ledger.read_text() == previous
    assert not (tmp_path / "results/jobs").exists()


def test_proper_prior_integrates_on_the_declared_continuous_support():
    bound = PROTOCOL["prior"]["bounds"][1]
    assert material.BOUND == bound
    np.testing.assert_array_equal(material.TILT, PROTOCOL["prior"]["tilt"])
    nodes, weights = leggauss(24)
    x1, x2 = np.meshgrid(bound * nodes, bound * nodes, indexing="ij")
    points = np.stack([x1, x2], axis=-1)
    integral = np.sum(np.exp(material.prior_logpdf(points)) * np.outer(weights, weights)) * bound**2
    np.testing.assert_allclose(integral, 1.0, rtol=0, atol=2e-13)
    assert np.isneginf(material.prior_logpdf([[bound, 0], [-bound, 0], [0, bound + .01]])).all()


def test_prior_sampler_has_continuous_support_without_category_inputs():
    samples = material.prior_sample(np.random.default_rng(261013), 1024)
    assert samples.shape == (1024, 2)
    assert np.isfinite(material.prior_logpdf(samples)).all()
    assert (np.abs(samples) < material.BOUND).all()
    assert set(np.unique(material.quadrant(samples))) == {0, 1, 2, 3}
    assert set(inspect.signature(material.prior_sample).parameters) == {"rng", "n"}


def test_quadrants_are_offline_events_not_sampler_or_decoder_conditions(monkeypatch):
    points = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    np.testing.assert_array_equal(material.quadrant(points), [0, 1, 2, 3])

    def forbidden_label(*args, **kwargs):
        pytest.fail("prior or decoder requested an offline quadrant label")

    monkeypatch.setattr(material, "quadrant", forbidden_label)
    material.prior_sample(np.random.default_rng(91), 8)
    eps = material.passive_grid(points)
    assert eps.shape == points.shape
    assert set(inspect.signature(material.passive_grid).parameters) == {"u", "loss_u"}


@pytest.mark.parametrize("centers,expected", [((), 0), (((0., 0.),), 1), (((-.5, 0.), (.5, 0.)), 2)])
def test_one_continuous_grid_decoder_represents_threshold_hole_events(centers, expected):
    axis = np.linspace(-1, 1, 41)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    u = np.full(xx.shape, 3.)
    for cx, cy in centers:
        u -= 6 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * .16**2))
    loss_u = np.linspace(-3., 1., u.size).reshape(u.shape)
    eps = material.passive_grid(u, loss_u)
    assert eps.shape == u.shape
    assert np.all((eps.real > .5) & (eps.real < 3.5))
    assert (eps.imag >= 0).all()
    assert material.finite_scale_holes(eps, threshold=2., opening_pixels=1) == expected
    # These are enclosed threshold events, not strict-vacuum atoms or recovered images.


def test_full_grid_coefficients_independently_affect_material_pixels():
    u = np.zeros((3, 4))
    base = material.passive_grid(u)
    for pixel in np.ndindex(u.shape):
        changed = u.copy()
        changed[pixel] += .01
        delta = material.passive_grid(changed) - base
        assert delta[pixel].real > 0
        mask = np.ones(u.shape, dtype=bool)
        mask[pixel] = False
        np.testing.assert_array_equal(delta[mask], 0)
    np.testing.assert_array_equal(base.imag, 0)


def test_real_gaussian_likelihood_uses_standard_deviation_and_full_normalizer():
    pred = np.array([[.2, -.5, 1.], [.7, -.2, .4]])
    y = np.array([.1, -.3, .8])
    sigma = .25
    expected = multivariate_normal(mean=y, cov=sigma**2 * np.eye(3)).logpdf(pred)
    np.testing.assert_allclose(gaussian_loglike(pred, y, sigma), expected, rtol=0, atol=2e-14)
    log_at_mean = gaussian_loglike(y, y, sigma)
    np.testing.assert_allclose(gaussian_loglike(pred, y, sigma) - log_at_mean,
                               -.5 * np.sum(((pred - y) / sigma)**2, axis=-1))


def test_proper_complex_whitening_has_half_variance_in_each_real_component():
    residual = np.array([[1 + 2j, -.5 + .3j], [.2 - .4j, -.8 - .6j]])
    sigma_c = .7  # E|noise|^2 = sigma_c^2; each real component variance = sigma_c^2/2.
    white = proper_complex_whiten(residual, sigma_c)
    assert white.shape == (2, 4)
    np.testing.assert_allclose(np.sum(white**2, axis=-1),
                               2 * np.sum(np.abs(residual)**2, axis=-1) / sigma_c**2)
    dimension = residual.shape[-1]
    log_density = gaussian_loglike(white, np.zeros_like(white), 1.) + dimension * np.log(2 / sigma_c**2)
    expected = -np.sum(np.abs(residual)**2, axis=-1) / sigma_c**2 - dimension * np.log(np.pi * sigma_c**2)
    np.testing.assert_allclose(log_density, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("beta", [[0., 1.], [0., .1, .35, .35, .7, 1.]])
def test_tempering_telescopes_to_exactly_one_likelihood(beta):
    log_prior = np.log([.15, .25, .6])
    log_likelihood = np.array([-4., -.8, -2.])
    final_logw = log_prior + log_likelihood
    expected = np.exp(final_logw - final_logw.max())
    expected /= expected.sum()
    np.testing.assert_allclose(tempering_weights(log_prior, log_likelihood, beta), expected, rtol=2e-14, atol=0)


@pytest.mark.parametrize("beta", [[.1, 1.], [0., .9], [0., .6, .5, 1.], [0., 1.1, 1.]])
def test_tempering_rejects_schedules_that_reuse_or_omit_evidence(beta):
    with pytest.raises(ValueError, match="single-likelihood"):
        tempering_weights(np.log([.5, .5]), np.array([-1., -2.]), beta)


@pytest.mark.parametrize("fractions", [[.5, .6], [-.1, 1.1], [0., 0.]])
def test_weight_splits_refuse_creation_or_destruction_of_parent_mass(fractions):
    with pytest.raises(ValueError, match="conserve mass"):
        split_weight(.4, fractions)


def test_copied_candidate_mass_is_invariant_before_and_after_one_likelihood():
    x = np.array([[-.4, -.8], [-.2, .5], [.6, -.3], [.7, .4]])
    prior_mass = np.array([.1, .2, .3, .4])
    y = np.array([.3, .25, .05])
    loglike = gaussian_loglike(problems.toy_forward(x, .25), y, .25)
    baseline = tempering_weights(np.log(prior_mass), loglike, [0., 1.])
    split = split_weight(prior_mass[2], [.25, .25, .5])
    copied_x = np.concatenate([x[:2], np.repeat(x[2:3], 3, axis=0), x[3:]])
    copied_prior = np.r_[prior_mass[:2], split, prior_mass[3:]]
    copied_loglike = gaussian_loglike(problems.toy_forward(copied_x, .25), y, .25)
    copied = tempering_weights(np.log(copied_prior), copied_loglike, [0., .3, 1.])
    np.testing.assert_allclose(copied_prior.sum(), prior_mass.sum(), rtol=0, atol=1e-16)
    np.testing.assert_allclose(
        np.bincount(material.quadrant(copied_x), weights=copied, minlength=4),
        np.bincount(material.quadrant(x), weights=baseline, minlength=4), rtol=0, atol=2e-15,
    )
    np.testing.assert_allclose(split_weight(.4, [0., .25, .75]), [0., .1, .3], rtol=0, atol=1e-16)


@pytest.mark.parametrize("problem", ["u0", "p0"])
def test_locked_test_payload_cannot_be_loaded_before_model_freeze(tmp_path, monkeypatch, problem):
    from c1 import data
    monkeypatch.setattr(data, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="forbidden before complete model freeze"):
        data.load_split(problem, "test")
    assert not list(tmp_path.iterdir())


def test_candidate_context_apis_accept_no_truth_or_posterior_keywords():
    expected = [(problems.toy_context, {"y", "kappa", "sigma", "initial"}),
                (problems.DipoleProblem.context, {"self", "y", "sigma", "initial"})]
    for function, names in expected:
        signature = inspect.signature(function)
        assert set(signature.parameters) == names
        assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        supplied = {name: .2 for name in names}
        for forbidden in ["truth", "truth_material", "truth_fields", "posterior", "quadrant", "H"]:
            with pytest.raises(TypeError):
                signature.bind(**supplied, **{forbidden: np.zeros(2)})


def test_toy_pilots_start_from_the_fixed_candidate_and_whiten_once(monkeypatch):
    candidates = []
    original = problems.toy_forward

    def tracked_forward(x, kappa):
        candidates.append(np.array(x, copy=True))
        return original(x, kappa)

    monkeypatch.setattr(problems, "toy_forward", tracked_forward)
    y, kappa, sigma = np.array([.8, .4, .1]), .25, .2
    states = problems.toy_context(y, kappa, sigma)
    assert len(candidates) == 2 and states.shape == (2, 18)
    np.testing.assert_array_equal(candidates[0], PROTOCOL["pilot"]["initial"])
    for c in states:
        np.testing.assert_allclose(c[5:8], (c[2:5] - y) / sigma)
        J = c[10:16].reshape(3, 2)
        np.testing.assert_allclose(c[8:10], J.T @ c[5:8])
        np.testing.assert_allclose(c[16:18], np.sum(J**2, axis=0))


@pytest.fixture(scope="session")
def dda_problem():
    # Deliberately default canonical ledger. No temp/private physical ledger.
    problem = problems.DipoleProblem()
    assert problem.ledger.path == ROOT / "results/PHYSICS_LEDGER.jsonl"
    problem.set_scale()  # 1 charged evaluation; fixed reference material.
    return problem


@pytest.fixture(scope="session")
def dda_derivative_states(dda_problem):
    points = [np.array([.4, -.6]), np.array([1.25, -.25])]
    return [(x, dda_problem.forward(x, derivative=True, kind="invariant_jacobian_base")) for x in points]


def test_dda_intensity_jacobian_matches_centered_material_differences(dda_problem, dda_derivative_states):
    step = 1e-5
    for x, state in dda_derivative_states:
        assert state["J"].shape == (4, 2)
        for coordinate in range(2):
            shift = np.zeros(2)
            shift[coordinate] = step
            plus = dda_problem.forward(x + shift, kind="invariant_fd_plus")["f"]
            minus = dda_problem.forward(x - shift, kind="invariant_fd_minus")["f"]
            np.testing.assert_allclose(state["J"][:, coordinate], (plus - minus) / (2 * step),
                                       rtol=5e-5, atol=2e-8)


def test_dda_constitutive_equation_and_two_source_forward_state(dda_problem, dda_derivative_states):
    for x, state in dda_derivative_states:
        eps, alpha, _ = dda_problem.polarizability(x)
        repeated = np.repeat(alpha, 3)[:, None]
        np.testing.assert_allclose(state["eps"], eps)
        np.testing.assert_allclose(state["E"], dda_problem.inc + dda_problem.G @ state["p"], rtol=2e-13, atol=1e-14)
        np.testing.assert_allclose(state["p"], repeated * state["E"], rtol=2e-12, atol=1e-15)
        np.testing.assert_allclose(state["L"] @ state["p"], repeated * dda_problem.inc, rtol=2e-12, atol=1e-15)
        np.testing.assert_allclose(state["f"], np.abs(state["amplitude"])**2 / dda_problem.scale)
        assert state["constitutive_residual"] < 1e-12
        assert state["source_count"] == 2 and state["factorization_count"] == 1
        assert state["p"].shape == (3 * dda_problem.n, 2)


def test_dda_reciprocity_uses_transpose_bilinear_pairing(dda_problem, dda_derivative_states):
    # Reciprocity for lossy reciprocal media is a transpose, not a Hermitian identity.
    np.testing.assert_allclose(dda_problem.G, dda_problem.G.T, rtol=1e-13, atol=1e-12)
    for _, state in dda_derivative_states:
        lhs = dda_problem.inc[:, 0] @ state["p"][:, 1]
        rhs = dda_problem.inc[:, 1] @ state["p"][:, 0]
        np.testing.assert_allclose(lhs, rhs, rtol=2e-12, atol=1e-15)


def test_dda_declared_support_is_passive_and_shared_by_both_sources(dda_problem):
    bound = material.BOUND - .01
    for x in [(-bound, -bound), (-bound, bound), (bound, -bound), (bound, bound)]:
        state = dda_problem.forward(x, kind="invariant_support_corner")
        assert state["passive"] and np.isfinite(state["f"]).all()
        np.testing.assert_allclose(state["eps"].imag, .02)
        assert ((state["eps"].real > .5) & (state["eps"].real < 3.5)).all()
        np.testing.assert_allclose(state["eps"].real[dda_problem.core], 2 + x[0])
        np.testing.assert_allclose(state["eps"].real[~dda_problem.core], 2 + x[1])
        assert state["constitutive_residual"] < 1e-12


def test_dda_zero_contrast_background_has_no_scattered_field(dda_problem):
    # The protocol background is epsilon_b=2; this is the no-scattering limit,
    # not a claim that epsilon=2 is physical vacuum or that sigmoid has atoms.
    state = dda_problem.forward([0., 0.], loss=0., kind="invariant_zero_contrast")
    np.testing.assert_array_equal(state["p"], 0)
    np.testing.assert_array_equal(state["amplitude"], 0)
    np.testing.assert_array_equal(state["f"], 0)
    np.testing.assert_array_equal(state["E"], dda_problem.inc)
    assert state["constitutive_residual"] == 0


def test_dda_two_round_candidate_context_is_charged_and_internally_consistent(dda_problem, dda_derivative_states):
    y = dda_derivative_states[0][1]["f"] + np.array([.01, .03, -.02, .04])
    sigma = .25
    before = dda_problem.ledger.summary()["used"]
    states = dda_problem.context(y, sigma)
    assert dda_problem.ledger.summary()["used"] - before == 2
    assert states.shape == (2, 38)
    np.testing.assert_array_equal(states[0, :2], PROTOCOL["pilot"]["initial"])
    assert np.linalg.norm(states[1, :2] - states[0, :2]) <= PROTOCOL["pilot"]["step_norm_cap"] + 1e-14
    for c in states:
        np.testing.assert_allclose(c[6:10], (c[2:6] - y) / sigma)
        J = c[12:20].reshape(4, 2)
        np.testing.assert_allclose(c[10:12], J.T @ c[6:10])
        np.testing.assert_allclose(c[20:22], np.sum(J**2, axis=0))
        assert np.isfinite(c).all()
