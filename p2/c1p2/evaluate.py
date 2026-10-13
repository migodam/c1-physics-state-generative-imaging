"""Zero-physics density comparison on explicitly supplied common references.

Raw proposal samples are model samples, not full-Mie-corrected posterior draws.
Reference uncertainty is supplied separately; binomial MC errors below quantify
only iid generator sampling.  A reference without log density cannot yield KL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .density import LOWER, UPPER, log_prob_batch, one_parameters, radial_event_ids, sample_parameters
from .train import load_model, write_json


def physical_coordinates(latent):
    """Post-decoder material coordinates, including the radius-zero atoms."""
    x = np.asarray(latent, dtype=np.float64)
    if x.shape[-1] != 4:
        raise ValueError("four common radial material coordinates required")
    value = np.array(x, copy=True)
    for j, scale in ((0, .45), (1, .18)):
        active = x[..., j] < -1
        geometry = np.zeros(x.shape[:-1], dtype=np.float64)
        geometry[active] = scale * np.sqrt(np.maximum(1 - 1 / np.sqrt(-x[..., j][active]), 0))
        value[..., j] = geometry
    return value


PHYSICAL_COORDINATE_NAMES = ("central_radius_over_R", "annular_halfwidth_over_R", "epsilon_real", "epsilon_imag")
PHYSICAL_LOWER = (0., 0., 2.2, .04)
PHYSICAL_UPPER = (.45 * np.sqrt(1 - 1 / np.sqrt(3)), .18 * np.sqrt(1 - 1 / np.sqrt(3)), 2.8, .16)
SIGNIFICANT_REFERENCE_MASS = .05
MASS_AWARE_DISCOVERY = .01
FALSE_EXCLUSION_MASS = .005
FALSE_CONFIDENCE_MASS = .90


def weighted_cdf(points, weights, thresholds):
    order = np.argsort(points, kind="stable")
    values = np.asarray(points)[order]
    cumulative = np.r_[0., np.cumsum(np.asarray(weights)[order])]
    return cumulative[np.searchsorted(values, thresholds, side="right")]


def weighted_quantile(points, weights, probabilities=(.05, .5, .95)):
    order = np.argsort(points, kind="stable")
    values = np.asarray(points)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    return values[np.minimum(np.searchsorted(cumulative, probabilities, side="left"), len(values) - 1)]


def _summary(points, weights):
    mean = weights @ points
    variance = weights @ ((points - mean) ** 2)
    return dict(mean=mean.tolist(), sd=np.sqrt(variance).tolist(),
                q05_q50_q95=np.stack([weighted_quantile(points[:, j], weights) for j in range(4)]).tolist())


def evaluate_parameters(parameters, reference_points, reference_weights,
                        reference_log_density=None, integration_weights=None,
                        event_ids=None, sample_count=8192, rng=None,
                        *, reference_event_mass=None, truth=None):
    """Evaluate one condition without silently normalizing proposal grid mass."""
    if sample_count < 8192:
        raise ValueError("P2 reported generator event masses require at least 8192 draws")
    x, p = np.asarray(reference_points, dtype=np.float64), np.asarray(reference_weights, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != 4 or p.shape != (len(x),) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(p).all()
            or np.any(p < 0) or abs(p.sum() - 1) > 1e-6):
        raise ValueError("reference coordinates and normalized probability weights required")
    positive = p > 0
    p = p / p.sum()  # Only posterior expectation roundoff, never model renormalization.
    events = radial_event_ids(x) if event_ids is None else np.asarray(event_ids, dtype=np.int64)
    if events.shape != p.shape or np.any((events < 0) | (events >= 12)):
        raise ValueError("reference event IDs must implement the frozen 12-event partition")
    q_log = log_prob_batch(x, parameters)
    if q_log.shape != p.shape or not np.isfinite(q_log[positive]).all():
        raise ValueError("finite proposal density on positive reference support required")
    score = float(np.sum(p[positive] * q_log[positive]))
    kl = None
    if reference_log_density is not None:
        logp = np.asarray(reference_log_density, dtype=np.float64)
        if logp.shape != p.shape or not np.isfinite(logp[positive]).all():
            raise ValueError("normalized dx reference log density required for KL")
        kl = float(np.sum(p[positive] * (logp[positive] - q_log[positive])))
    model_integral = None
    if integration_weights is not None:
        w = np.asarray(integration_weights, dtype=np.float64)
        if w.shape != p.shape or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("positive base-measure integration weights required")
        model_integral = float(np.exp(q_log) @ w)
        if not np.isfinite(model_integral):
            raise ValueError("model quadrature mass overflow")
    draw = sample_parameters(parameters, sample_count, rng or np.random.default_rng(26101331))
    sample = draw["latent"][0]
    sample_events = radial_event_ids(sample)
    counts = np.bincount(sample_events, minlength=12)
    mass = counts / sample_count
    sample_reference_mass = np.bincount(events, weights=p, minlength=12)
    reference_mass = sample_reference_mass
    if reference_event_mass is not None:
        reference_mass = np.asarray(reference_event_mass, dtype=np.float64)
        if (reference_mass.shape != (12,) or not np.isfinite(reference_mass).all()
                or np.any(reference_mass < 0) or abs(reference_mass.sum() - 1) > 1e-6):
            raise ValueError("twelve normalized primary reference event masses required")
    standard_error = np.sqrt(mass * (1 - mass) / sample_count)
    # A zero count does not imply a truly zero model mass: report the binomial
    # one-sided 95% upper bound rather than assigning a certainty to exclusion.
    zero_count_upper95 = 1 - .05 ** (1 / sample_count)
    thresholds = np.stack([np.linspace(LOWER[j], UPPER[j], 201) for j in range(4)])
    generated_weights = np.full(sample_count, 1 / sample_count)
    reference_cdf = np.stack([weighted_cdf(x[:, j], p, thresholds[j]) for j in range(4)])
    generated_cdf = np.stack([weighted_cdf(sample[:, j], generated_weights, thresholds[j]) for j in range(4)])
    physical_reference = physical_coordinates(x)
    physical_sample = physical_coordinates(sample)
    physical_thresholds = np.stack([np.linspace(PHYSICAL_LOWER[j], PHYSICAL_UPPER[j], 201) for j in range(4)])
    physical_reference_cdf = np.stack([weighted_cdf(physical_reference[:, j], p, physical_thresholds[j]) for j in range(4)])
    physical_model_cdf = np.stack([weighted_cdf(physical_sample[:, j], generated_weights, physical_thresholds[j]) for j in range(4)])
    conditional = []
    for event in range(12):
        ref = events == event
        gen = sample_events == event
        row = dict(event_id=event, reference_mass=float(reference_mass[event]),
                   sample_reference_mass=float(sample_reference_mass[event]),
                   model_mass=float(mass[event]), generator_count=int(counts[event]))
        if ref.any() and sample_reference_mass[event] > 0:
            rp = p[ref] / sample_reference_mass[event]
            row["reference_material"] = _summary(x[ref], rp)
            row["reference_physical_material"] = _summary(physical_reference[ref], rp)
            if gen.any():
                gp = np.full(gen.sum(), 1 / gen.sum())
                row["generated_material"] = _summary(sample[gen], gp)
                row["generated_physical_material"] = _summary(physical_sample[gen], gp)
                rcdf = np.stack([weighted_cdf(x[ref, j], rp, thresholds[j]) for j in range(4)])
                gcdf = np.stack([weighted_cdf(sample[gen, j], gp, thresholds[j]) for j in range(4)])
                row["conditional_cdf_max_abs"] = np.max(np.abs(rcdf - gcdf), axis=-1).tolist()
                prcdf = np.stack([weighted_cdf(physical_reference[ref, j], rp, physical_thresholds[j]) for j in range(4)])
                pgcdf = np.stack([weighted_cdf(physical_sample[gen, j], gp, physical_thresholds[j]) for j in range(4)])
                row["conditional_physical_cdf_max_abs"] = np.max(np.abs(prcdf - pgcdf), axis=-1).tolist()
            else:
                row["generated_material"] = None
                row["conditional_cdf_max_abs"] = None
                row["generated_physical_material"] = None
                row["conditional_physical_cdf_max_abs"] = None
        conditional.append(row)
    significant = reference_mass >= SIGNIFICANT_REFERENCE_MASS
    discovered = mass >= MASS_AWARE_DISCOVERY
    false_excluded = significant & (mass < FALSE_EXCLUSION_MASS)
    jeffreys_mass = (counts + .5) / (sample_count + 12 * .5)
    finite_odds = significant & (reference_mass > 0) & (reference_mass < 1)
    log_odds_error = np.abs((np.log(reference_mass[finite_odds]) - np.log1p(-reference_mass[finite_odds]))
                            - (np.log(jeffreys_mass[finite_odds]) - np.log1p(-jeffreys_mass[finite_odds])))
    order = np.argsort(-mass, kind="stable")
    top_k = min(12, 1 + int(np.searchsorted(np.cumsum(mass[order]), .95, side="left")))
    selected = order[:top_k]
    selected_model_mass = float(mass[selected].sum())
    selected_reference_mass = float(reference_mass[selected].sum())
    significant_mass = float(reference_mass[significant].sum())
    result = dict(
        proposal_expected_log_score_x=score, reference_KL_to_proposal=kl,
        kl_status="reference log density supplied; numerical target only" if kl is not None else "NOT_AVAILABLE: reference log density absent",
        model_quadrature_integral_unrenormalized=model_integral,
        reference_event_mass=reference_mass.tolist(), model_event_mass=mass.tolist(),
        primary_event_reference="supplied continuous quadrature mass" if reference_event_mass is not None else "saved reference probability weights",
        sample_reference_event_mass=sample_reference_mass.tolist(),
        sample_reference_event_mass_MAE=float(np.mean(np.abs(sample_reference_mass - reference_mass))),
        event_mass_MC_standard_error=standard_error.tolist(), generator_event_counts=counts.tolist(),
        zero_count_mass_upper95=float(zero_count_upper95), sample_count=sample_count,
        event_mass_MAE=float(np.mean(np.abs(reference_mass - mass))),
        significant_event_recall=float(np.mean(discovered[significant])) if significant.any() else None,
        significant_event_count_recall=float(np.mean(discovered[significant])) if significant.any() else None,
        significant_event_mass_recall=float(reference_mass[significant & discovered].sum() / significant_mass) if significant_mass > 0 else None,
        sample_hit_event_count_recall=float(np.mean(counts[significant] > 0)) if significant.any() else None,
        significant_event_count=int(significant.sum()),
        false_exclusion=bool(false_excluded.any()), false_exclusion_event_ids=np.flatnonzero(false_excluded).tolist(),
        false_excluded_reference_mass=float(reference_mass[false_excluded].sum()),
        false_confidence=bool(mass.max() >= FALSE_CONFIDENCE_MASS and significant.sum() >= 2),
        failure_definitions=dict(significant_reference_mass=SIGNIFICANT_REFERENCE_MASS,
                                 mass_aware_discovery_model_mass=MASS_AWARE_DISCOVERY,
                                 false_exclusion_model_mass_below=FALSE_EXCLUSION_MASS,
                                 false_confidence_max_model_mass=FALSE_CONFIDENCE_MASS,
                                 false_confidence_min_significant_reference_events=2),
        important_event_max_abs_error=float(np.max(np.abs(reference_mass[significant] - mass[significant]))) if significant.any() else None,
        important_event_log_odds_MAE=float(log_odds_error.mean()) if len(log_odds_error) else None,
        important_event_log_odds_max_abs=float(log_odds_error.max()) if len(log_odds_error) else None,
        event_log_odds_status="reference quadrature mass versus Jeffreys-stabilized generator counts; exact zero/one reference odds excluded and not certified",
        reference_material=_summary(x, p), generated_material=_summary(sample, generated_weights),
        physical_coordinate_names=list(PHYSICAL_COORDINATE_NAMES),
        reference_physical_material=_summary(physical_reference, p),
        generated_physical_material=_summary(physical_sample, generated_weights),
        cdf_max_abs=np.max(np.abs(reference_cdf - generated_cdf), axis=-1).tolist(),
        physical_cdf_max_abs=np.max(np.abs(physical_reference_cdf - physical_model_cdf), axis=-1).tolist(),
        top_k=top_k, top_k_selected_event_ids=selected.tolist(),
        top_k_represented_model_mass=selected_model_mass, top_k_represented_reference_mass=selected_reference_mass,
        top_k_unshown_modeled_mass=max(0., 1 - selected_model_mass),
        top_k_unshown_reference_mass=max(0., 1 - selected_reference_mass),
        conditional_material=conditional, numerical_endpoint_draws=int(draw["rounded_endpoint"].sum()),
        MC_error_scope="iid proposal draws only; excludes reference/solver/model error",
        posterior_status="raw learned proposal; not full-likelihood corrected",
        reference_summary_scope="consistent saved reference probability weights; quadrature or sample approximation is external",
    )
    if truth is not None:
        true_x = np.asarray(truth, dtype=np.float64)
        if true_x.shape != (4,) or not np.isfinite(true_x).all():
            raise ValueError("one finite four-dimensional truth required")
        true_material = physical_coordinates(true_x[None])[0]
        true_event = int(radial_event_ids(true_x[None])[0])
        onehot = np.eye(12)[true_event]
        # Explicit finite MC estimator for event log score.  Raw zero counts
        # stay visible; Jeffreys count stabilization is not density correction.
        result.update(truth_latent=true_x.tolist(), truth_physical_material=true_material.tolist(),
                      truth_event_id=true_event, event_Brier_score=float(np.sum((mass - onehot) ** 2)),
                      reference_event_Brier_score=float(np.sum((reference_mass - onehot) ** 2)),
                      truth_event_log_score_raw_MC=float(np.log(mass[true_event])) if mass[true_event] > 0 else None,
                      truth_event_log_score_MC_Jeffreys=float(np.log(jeffreys_mass[true_event])),
                      truth_event_log_score_MC_status="iid counts with explicit Jeffreys .5 per event for finite score; raw zero retained",
                      reference_truth_event_log_score=float(np.log(reference_mass[true_event])) if reference_mass[true_event] > 0 else None)
        for label, value in (("latent", true_x), ("physical", true_material)):
            gen_key = "generated_material" if label == "latent" else "generated_physical_material"
            ref_key = "reference_material" if label == "latent" else "reference_physical_material"
            gq = np.asarray(result[gen_key]["q05_q50_q95"])
            rq = np.asarray(result[ref_key]["q05_q50_q95"])
            covered = (value >= gq[:, 0]) & (value <= gq[:, 2])
            reference_covered = (value >= rq[:, 0]) & (value <= rq[:, 2])
            result[f"truth_90pct_marginal_coverage_{label}"] = covered.tolist()
            result[f"truth_90pct_simultaneous_coverage_{label}"] = bool(covered.all())
            result[f"reference_truth_90pct_marginal_coverage_{label}"] = reference_covered.tolist()
        result["credible_coverage_scope"] = "equal-tail marginal 90% intervals; physical radius-zero atoms retained; joint box is not a 90% joint credible region"
    arrays = dict(sample_latent=sample, sample_transformed=draw["transformed"][0],
                  sample_event=sample_events, reference_cdf=reference_cdf,
                  model_cdf=generated_cdf, cdf_thresholds=thresholds,
                  sample_physical_material=physical_sample, reference_physical_cdf=physical_reference_cdf,
                  model_physical_cdf=physical_model_cdf, physical_cdf_thresholds=physical_thresholds)
    return result, arrays


def load_final_data(path):
    """Portable final file.  Training cannot load this file through train.py."""
    with np.load(path, allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    if "target_identity" not in data or "y" not in data:
        raise ValueError("final data needs target_identity and y")
    identity = str(np.asarray(data["target_identity"]).item())
    y = np.asarray(data["y"], dtype=np.float32)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("finite final measurement conditions required")
    if "reference_points" in data and "reference_weights" in data:
        x, weights = np.asarray(data["reference_points"]), np.asarray(data["reference_weights"])
        if x.ndim == 2:
            x = np.broadcast_to(x, (len(y), *x.shape))
    elif "reference_samples" in data:
        x = np.asarray(data["reference_samples"])
        weights = np.full(x.shape[:2], 1 / x.shape[1])
    else:
        raise ValueError("supply weighted reference points or reference_samples")
    if x.ndim != 3 or x.shape[0] != len(y) or weights.shape != x.shape[:2]:
        raise ValueError("reference coordinates/weights shape mismatch")
    parent = np.asarray(data.get("parent_ids", data.get("parent", np.arange(len(y)))))
    if parent.shape != (len(y),):
        raise ValueError("one parent ID per observation required")
    return data, identity, y, x, weights, parent


def evaluate_final(model_paths, dataset, output, sample_count=8192, save_samples=False):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    output = Path(output)
    if (output / "EVALUATION_SUMMARY.json").exists():
        raise RuntimeError("completed P2 density evaluation is not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    data, identity, y, x, weights, parent = load_final_data(dataset)
    rows = []
    for model_path in model_paths:
        model, metadata = load_model(model_path, expected_target=identity)
        parameters = model.export_parameters(y)
        name = Path(model_path).stem
        np.savez_compressed(output / f"{name}-parameters.npz", **parameters,
                            target_identity=np.asarray(identity))
        for i in range(len(y)):
            lp = data["reference_log_density"][i] if "reference_log_density" in data else None
            iw = data.get("integration_weights")
            iw = iw[i] if iw is not None and iw.ndim == 2 else iw
            ev = data.get("event_ids")
            ev = ev[i] if ev is not None and ev.ndim == 2 else ev
            result, arrays = evaluate_parameters(one_parameters(parameters, i), x[i], weights[i],
                                                lp, iw, ev, sample_count,
                                                np.random.default_rng(metadata["seed"] + 100000 + i),
                                                reference_event_mass=data["reference_event_mass"][i] if "reference_event_mass" in data else None,
                                                truth=data["truth"][i] if "truth" in data else None)
            result.update(method=metadata["method"], seed=int(metadata["seed"]),
                          parent_id=str(parent[i]), observation_index=i, target_identity=identity)
            if "truth" in data:
                result["observed_truth_log_score_x"] = float(log_prob_batch(data["truth"][i:i + 1], one_parameters(parameters, i))[0])
            rows.append(result)
            if save_samples:
                np.savez_compressed(output / f"{name}-observation-{i:03d}.npz", **arrays,
                                    target_identity=np.asarray(identity))
        write_json(output / f"{name}-rows.json", [row for row in rows if row["method"] == metadata["method"] and row["seed"] == metadata["seed"]])
        print(json.dumps(dict(model=name, observations=len(y), material_frequency_evaluations=0)), flush=True)
    summary = dict(status="PROPOSAL_EVALUATION", target_identity=identity,
                   observations=len(y), parents=len(np.unique(parent)), models=len(model_paths),
                   rows=rows, sample_count_per_condition=sample_count, material_frequency_evaluations=0,
                   cpu_seconds=time.process_time() - start_cpu, wall_seconds=time.perf_counter() - start_wall,
                   calibration_status="not established by one reference comparison")
    write_json(output / "EVALUATION_SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("p2/results/evaluation"))
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--save-samples", action="store_true")
    args = parser.parse_args()
    evaluate_final(args.models, args.dataset, args.output, args.sample_count, args.save_samples)


if __name__ == "__main__":
    main()
