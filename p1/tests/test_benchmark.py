"""Pure NumPy interface tests; no physical solver, B1 arrays, or GPU imports."""
from dataclasses import replace
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from c1p1.benchmark import (
    BoxSupport, CandidateFeatures, CostRecord, MaterialEvent, MaterialSpace,
    MeasurementMetadata, Observation, PosteriorBenchmark, ProperPrior, QuadratureReference,
    TargetIdentity, evaluate_quadrature, require_same_target,
)


def target() -> TargetIdentity:
    return TargetIdentity("unit-interval", "open-01", "Lebesgue-dx", "uniform-01",
                          "unit-likelihood", "none", "none", "unit-acquisition",
                          "algebraic-fixture", "uniform-continuous-target", "fixture-0")


def reference(order=12) -> QuadratureReference:
    x, w = leggauss(order)
    return QuadratureReference(target(), "independent-uniform-fixture",
                               ((x + 1) / 2)[:, None], w / 2,
                               np.zeros(order), "Gauss integration of dx")


class BenchmarkInterfaceTests(unittest.TestCase):
    def test_uniform_density_uses_integration_measure(self):
        ref = reference()
        event = MaterialEvent("low", "x < 0.5, after generation", lambda x: x[:, 0] < .5)
        result = evaluate_quadrature(ref, target(), lambda x: np.zeros(len(x)), (event,))
        self.assertAlmostEqual(result.reference_integral, 1.0)
        self.assertAlmostEqual(result.model_integral, 1.0)
        self.assertAlmostEqual(result.reference_kl, 0.0)
        self.assertAlmostEqual(result.reference_events["low"], .5)
        self.assertAlmostEqual(result.model_events["low"], .5)
        np.testing.assert_allclose(result.reference_mean, [.5], atol=1e-14)
        np.testing.assert_allclose(result.reference_variance, [1 / 12], atol=1e-14)

    def test_each_target_identity_difference_is_rejected(self):
        baseline = target()
        for name in baseline.__dataclass_fields__:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    require_same_target(baseline, replace(baseline, **{name: "different"}))

    def test_target_ids_cannot_be_unknown_or_blank(self):
        with self.assertRaises(ValueError):
            replace(target(), prior_id=" ")

    def test_nonuniform_node_weights_are_not_a_discrete_uniform_prior(self):
        # These nodes integrate dx; the resulting posterior node weights are
        # unequal even though the continuous material density is exactly one.
        ref = reference(6)
        self.assertFalse(np.allclose(ref.probability_weights, 1 / 6))
        result = evaluate_quadrature(ref, target(), lambda x: np.zeros(len(x)))
        self.assertAlmostEqual(result.reference_mean[0], .5)

    def test_model_mass_is_never_silently_renormalized(self):
        ref = reference()
        result = evaluate_quadrature(ref, target(), lambda x: np.full(len(x), np.log(.5)))
        self.assertAlmostEqual(result.model_integral, .5)
        self.assertAlmostEqual(result.reference_kl, np.log(2))

    def test_missing_model_support_reports_infinite_kl(self):
        ref = reference()
        result = evaluate_quadrature(
            ref, target(), lambda x: np.where(x[:, 0] < .5, 0.0, -np.inf))
        self.assertTrue(np.isposinf(result.reference_kl))
        self.assertTrue(np.isneginf(result.conditional_log_score))

    def test_support_dimension_and_open_boundary(self):
        support = BoxSupport([0, -1], [1, 1])
        np.testing.assert_array_equal(support.contains([[.5, 0], [1, 0], [0, 0]]),
                                      [True, False, False])
        with self.assertRaises(ValueError):
            support.contains([[.5]])
        with self.assertRaises(ValueError):
            BoxSupport([1], [0])

    def test_field_decode_does_not_invoke_forward(self):
        space = MaterialSpace("one", "box", "dx", BoxSupport([0], [1]), ("x",),
                              decoder=lambda x: 2 + x + .02j, field_description="passive epsilon")
        np.testing.assert_allclose(space.field([[.5]]), [[2.5 + .02j]])
        with self.assertRaises(ValueError):
            space.field([[1]])

    def test_proper_prior_and_metadata_are_explicit(self):
        prior = ProperPrior("uniform-01", "Lebesgue-dx", lambda x: np.zeros(len(x)),
                            lambda rng, n: rng.uniform(size=(n, 1)),
                            "uniform density one over unit interval")
        metadata = MeasurementMetadata("unit-likelihood", "none", "none",
                                       "unit-acquisition", "real-vector", "dimensionless")
        observation = Observation("fixture-0", np.array([0.0]), metadata,
                                  "synthetic test fixture")
        self.assertEqual(prior.base_measure_id, target().base_measure_id)
        self.assertEqual(observation.metadata.likelihood_id, target().likelihood_id)

    def test_bound_benchmark_rejects_mismatches_without_calling_forward(self):
        class UncallableForward:
            solver_id = "algebraic-fixture"

            def predict(self, *args, **kwargs):
                raise AssertionError("interface must not invoke a physical solver")

        space = MaterialSpace("unit-interval", "open-01", "Lebesgue-dx",
                              BoxSupport([0], [1]), ("x",))
        prior = ProperPrior("uniform-01", "Lebesgue-dx", lambda x: np.zeros(len(x)),
                            lambda rng, n: rng.uniform(size=(n, 1)), "unit interval")
        metadata = MeasurementMetadata("unit-likelihood", "none", "none",
                                       "unit-acquisition", "real", "dimensionless")
        bundle = PosteriorBenchmark(target(), space, prior, metadata, reference(),
                                    UncallableForward())
        self.assertIsNotNone(bundle.forward)
        with self.assertRaisesRegex(ValueError, "nuisance_id"):
            replace(bundle, metadata=replace(metadata, nuisance_id="marginalized-gain"))
        with self.assertRaisesRegex(ValueError, "base measures"):
            replace(bundle, prior=replace(prior, base_measure_id="uniform-node-count"))
        wrong_observation = Observation("another-y", np.array([1.0]), metadata, "fixture")
        with self.assertRaisesRegex(ValueError, "observation_id"):
            replace(bundle, observation=wrong_observation)

    def test_missing_cost_is_explicitly_unmeasured(self):
        result = evaluate_quadrature(reference(), target(), lambda x: np.zeros(len(x)))
        self.assertFalse(result.cost.measured)

    def test_candidate_features_reject_oracle_origin(self):
        cost = CostRecord("historical_candidate_preparation", .1, .2)
        feature = CandidateFeatures(np.array([.5]), "observation-0", np.array([1.]),
                                    "saved_candidate_pilot", cost)
        self.assertEqual(feature.origin, "saved_candidate_pilot")
        with self.assertRaises(ValueError):
            replace(feature, origin="true_material_oracle_fields")

    def test_cost_rejects_negative_or_fractional_physical_counts(self):
        with self.assertRaises(ValueError):
            CostRecord("training", cpu_seconds=-1)
        with self.assertRaises(ValueError):
            CostRecord("forward", material_frequency_evaluations=.5)
        with self.assertRaises(ValueError):
            CostRecord("density", wall_seconds=np.nan)

    def test_bad_quadrature_weights_and_events_are_rejected(self):
        ref = reference()
        with self.assertRaises(ValueError):
            replace(ref, integration_weights=-ref.integration_weights)
        with self.assertRaises(ValueError):
            replace(ref, log_density=np.full(len(ref.nodes), np.nan))
        bad_event = MaterialEvent("bad", "must be boolean", lambda x: np.ones(len(x)))
        with self.assertRaises(ValueError):
            evaluate_quadrature(ref, target(), lambda x: np.zeros(len(x)), (bad_event,))
        with self.assertRaisesRegex(ValueError, "overflow"):
            evaluate_quadrature(ref, target(), lambda x: np.full(len(x), 1000.0))


if __name__ == "__main__":
    unittest.main()
