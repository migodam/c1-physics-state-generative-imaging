"""Compare frozen proposal draws to direct truncated-CDF reference integrals.

This is a pure postprocessor: saved mixture parameters reproduce the original
8192 draws with their original RNG seed.  It trains nothing, selects no models,
invokes no Maxwell/reference solver, and leaves all original evaluation files
unchanged.  The fixed-grid CDF maximum is not a continuous supremum bound.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .density import one_parameters, radial_event_ids, sample_parameters
from .evaluate import PHYSICAL_COORDINATE_NAMES, PHYSICAL_LOWER, PHYSICAL_UPPER, physical_coordinates, weighted_cdf
from .summarize import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap_pair
from .train import write_json


def _parent_means(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_id"]].append(row["metrics"])
    result = {}
    for parent, items in groups.items():
        keys = {key for item in items for key in item}
        result[parent] = {}
        for key in keys:
            values = [item[key] for item in items if item.get(key) is not None]
            result[parent][key] = float(np.mean(values)) if values else None
    return result


def _group_summary(rows):
    parents = _parent_means(rows)
    keys = {key for item in parents.values() for key in item}
    means, count = {}, {}
    for key in sorted(keys):
        values = [item[key] for item in parents.values() if item.get(key) is not None]
        means[key] = float(np.mean(values)) if values else None
        count[key] = len(values)
    return dict(observations=len(rows), parents=len(parents), parent_equal_weight_means=means,
                available_parent_counts=count)


def continuous_comparison(reference_path, evaluation_folder, output):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    reference_path, evaluation_folder, output = map(Path, (reference_path, evaluation_folder, output))
    if (output / "CONTINUOUS_COMPARISON.json").exists():
        raise RuntimeError("completed continuous comparison is preserved")
    with np.load(reference_path, allow_pickle=False) as file:
        reference = {key: file[key] for key in file.files}
    identity = str(np.asarray(reference["target_identity"]).item())
    grid = np.asarray(reference["grid"], dtype=np.float64)
    cdf = np.asarray(reference["CDF"], dtype=np.float64)
    conditional_cdf = np.asarray(reference["conditional_CDF"], dtype=np.float64)
    means, standard_deviations = np.asarray(reference["mean"]), np.asarray(reference["SD"])
    event_mass = np.asarray(reference["event_mass"])
    n = cdf.shape[0]
    if (grid.shape != (4, 65) or cdf.shape != (n, 4, 65)
            or conditional_cdf.shape != (n, 12, 4, 65)
            or means.shape != (n, 4) or standard_deviations.shape != (n, 4)
            or event_mass.shape != (n, 12)):
        raise ValueError("direct reference shapes do not match the frozen common panel")
    if not all(np.isfinite(value).all() for value in (grid, cdf, conditional_cdf, means, standard_deviations, event_mass)):
        raise ValueError("finite continuous reference arrays required")
    original = json.loads((evaluation_folder / "EVALUATION_SUMMARY.json").read_text())
    if original["target_identity"] != identity or original["observations"] != n:
        raise ValueError("continuous and original reference target/panel mismatch")
    direct_audit_path = reference_path.with_name("FRESH_CONTINUOUS_AUDIT.json")
    direct_audit = json.loads(direct_audit_path.read_text()) if direct_audit_path.exists() else None
    direct_quantiles = (direct_audit or {}).get("first_eight_direct_quantile_summaries", [])
    widths = np.asarray(PHYSICAL_UPPER) - PHYSICAL_LOWER
    rows, parameter_cache = [], {}
    for old in original["rows"]:
        method, seed, i = old["method"], int(old["seed"]), int(old["observation_index"])
        model_id = f"{method}-{seed}"
        if model_id not in parameter_cache:
            with np.load(evaluation_folder / f"{model_id}-parameters.npz", allow_pickle=False) as saved:
                parameters = {key: saved[key] for key in saved.files}
            if str(parameters["target_identity"].item()) != identity:
                raise ValueError("saved model parameter target mismatch")
            parameter_cache[model_id] = parameters
        count = int(old["sample_count"])
        if count != 8192:
            raise ValueError("the registered original evaluator used 8192 samples")
        draw = sample_parameters(one_parameters(parameter_cache[model_id], i), count,
                                 np.random.default_rng(seed + 100000 + i))
        sample = physical_coordinates(draw["latent"][0])
        events = radial_event_ids(draw["latent"][0])
        counts = np.bincount(events, minlength=12)
        if not np.array_equal(counts, old["generator_event_counts"]):
            raise RuntimeError("original model draw reproduction differs; no new MC experiment is substituted")
        weights = np.full(count, 1 / count)
        model_cdf = np.stack([weighted_cdf(sample[:, j], weights, grid[j]) for j in range(4)])
        cdf_error = np.max(np.abs(model_cdf - cdf[i]), axis=-1)
        model_mean = sample.mean(0)
        model_sd = sample.std(0, ddof=0)
        mean_error_fraction = np.abs(model_mean - means[i]) / widths
        sd_error_fraction = np.abs(model_sd - standard_deviations[i]) / widths
        relative_sd = np.abs(model_sd - standard_deviations[i]) / np.maximum(standard_deviations[i], widths * 1e-6)
        conditional, unresolved_mass = [], 0.
        for event in np.flatnonzero(event_mass[i] >= .05):
            selected = events == event
            if not selected.any():
                unresolved_mass += event_mass[i, event]
                conditional.append(dict(event_id=int(event), reference_mass=float(event_mass[i, event]),
                                        model_count=0, grid_CDF_max_abs=None))
                continue
            probability = np.full(selected.sum(), 1 / selected.sum())
            mcdf = np.stack([weighted_cdf(sample[selected, j], probability, grid[j]) for j in range(4)])
            error = np.max(np.abs(mcdf - conditional_cdf[i, event]), axis=-1)
            conditional.append(dict(event_id=int(event), reference_mass=float(event_mass[i, event]),
                                    model_count=int(selected.sum()), grid_CDF_max_abs=error.tolist()))
        available_conditional = [max(item["grid_CDF_max_abs"]) for item in conditional if item["grid_CDF_max_abs"] is not None]
        metrics = dict(max_physical_direct_grid_CDF_error=float(cdf_error.max()),
                       important_conditional_direct_grid_CDF_max=max(available_conditional) if available_conditional else None,
                       important_conditional_CDF_unresolved_reference_mass=float(unresolved_mass),
                       max_physical_mean_error_prior_fraction=float(mean_error_fraction.max()),
                       max_physical_SD_error_prior_fraction=float(sd_error_fraction.max()),
                       max_physical_SD_relative_error=float(relative_sd.max()),
                       old_node_CDF_diagnostic=float(max(old["physical_cdf_max_abs"])))
        for j in range(4):
            metrics[f"physical_direct_grid_CDF_coordinate{j}"] = float(cdf_error[j])
            metrics[f"physical_mean_error_prior_fraction_coordinate{j}"] = float(mean_error_fraction[j])
            metrics[f"physical_SD_relative_error_coordinate{j}"] = float(relative_sd[j])
        row = dict(method=method, seed=seed, observation_index=i, parent_id=old["parent_id"],
                   target_identity=identity, metrics=metrics, physical_grid_CDF_max_abs=cdf_error.tolist(),
                   model_physical_mean=model_mean.tolist(), reference_physical_mean=means[i].tolist(),
                   model_physical_SD=model_sd.tolist(), reference_physical_SD=standard_deviations[i].tolist(),
                   conditional_CDF=conditional, exact_original_event_counts_reproduced=True,
                   truth_90pct_marginal_coverage_physical=old.get("truth_90pct_marginal_coverage_physical"),
                   model_quantiles_unchanged=old["generated_physical_material"]["q05_q50_q95"],
                   old_reference_quantile_status="NODE_QUANTILE_DIAGNOSTIC; not an inverted continuous CDF")
        if i < len(direct_quantiles):
            quantiles = np.asarray(direct_quantiles[i]["quantiles_05_50_95"])
            model_quantiles = np.asarray(row["model_quantiles_unchanged"])
            if quantiles.shape != (4, 3):
                raise ValueError("first-eight direct quantiles must have shape [4,3]")
            row["first_eight_direct_reference_quantiles"] = quantiles.tolist()
            row["first_eight_quantile_error_prior_fraction"] = (np.abs(model_quantiles - quantiles) / widths[:, None]).tolist()
            if "truth_physical_material" in old:
                truth = np.asarray(old["truth_physical_material"])
                row["first_eight_direct_reference_truth_90pct_marginal_coverage"] = ((truth >= quantiles[:, 0]) & (truth <= quantiles[:, 2])).tolist()
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"])].append(row)
    seed_separated = [dict(method=method, seed=seed, **_group_summary(group))
                      for (method, seed), group in sorted(grouped.items())]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired, pooled = [], defaultdict(list)
    for seed in sorted({seed for method, seed in grouped if method == "full_distill"}):
        if ("diag_distill", seed) not in grouped:
            continue
        left, right = _parent_means(grouped[("full_distill", seed)]), _parent_means(grouped[("diag_distill", seed)])
        paired.append(dict(seed=seed, metrics=bootstrap_pair(left, right, rng)))
        for parent in set(left) & set(right):
            pooled[parent].append({key: left[parent][key] - right[parent][key] for key in set(left[parent]) & set(right[parent])
                                   if left[parent][key] is not None and right[parent][key] is not None})
    pooled_left, pooled_right = {}, {}
    for parent, items in pooled.items():
        keys = {key for item in items for key in item}
        pooled_left[parent] = {key: float(np.mean([item[key] for item in items if key in item])) for key in keys}
        pooled_right[parent] = {key: 0. for key in keys}
    report = dict(status="DIRECT_CONTINUOUS_REFERENCE_COMPARISON", target_identity=identity,
                  reference=str(reference_path.resolve()), original_evaluation=str(evaluation_folder.resolve()),
                  observations=n, models=len(grouped), rows=rows, seed_separated=seed_separated,
                  physical_coordinate_names=list(PHYSICAL_COORDINATE_NAMES),
                  paired_full_vs_diagonal_distillation=paired,
                  paired_seed_mean_parent_bootstrap=bootstrap_pair(pooled_left, pooled_right, rng),
                  bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES, unit="parent",
                                 scope="conditional on fixed original MC draws and numerical direct-integral reference"),
                  old_node_metrics="NODE_CDF_DIAGNOSTIC, preserved; not a continuous CDF error",
                  CDF_error_scope="maximum on fixed 65-point physical-coordinate grids, not uniform continuous supremum",
                  reference_error_scope=("positive partition CDF, all128 moment/event13/17 checks, three rows upgraded17 and checked21; direct inverse quantiles on fixed8; empirical not uniform/full-Mie/continuum certificate" if "reference_orders" in reference else "order13 integrals, first-eight order13/17 refinement and direct inverse quantiles; empirical not full-Mie/continuum certificate"),
                  sampler_scope="exact reproduction of original frozen proposal draws, no model selection or fresh re-sampling experiment",
                  new_physics=0, gpu_seconds=0, wall_seconds=time.perf_counter() - start_wall,
                  cpu_seconds=time.process_time() - start_cpu)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "CONTINUOUS_COMPARISON.json", report)
    lines = ["# C1 P2：直接连续积分的材料 CDF 复核", "",
             "使用已冻结模型和原始8192次 draws 的相同 RNG；原始节点 CDF 文件保留。", "",
             "| method | seed | physical grid-CDF error | important-event conditional grid-CDF | mean/prior-width error | SD/prior-width error | raw node diagnostic |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    def fmt(value):
        return "NOT_AVAILABLE" if value is None else f"{value:.5g}"
    for summary in seed_separated:
        m = summary["parent_equal_weight_means"]
        lines.append("| " + " | ".join([summary["method"], str(summary["seed"]), *[fmt(m.get(key)) for key in
                     ("max_physical_direct_grid_CDF_error", "important_conditional_direct_grid_CDF_max",
                      "max_physical_mean_error_prior_fraction", "max_physical_SD_error_prior_fraction", "old_node_CDF_diagnostic")]]) + " |")
    lines += ["", "每列先对两个噪声观测按 parent 求均值，再等权平均64个 parents。",
              "固定65点 grid-CDF 最大值不是连续 supremum；不能据此认证所有阈值都达到.01。",
              "关键事件的模型计数为零时，条件 CDF 记不可用并单列 unresolved reference mass。",
              "完整直接反解 quantiles 与13/17误差只在前八个观测提供；其余旧 quantiles 不改称连续参考。",
              "模型90%边际区间及 truth coverage 沿用原始 frozen evaluation；半径为零的原子保留。",
              "配对 parent bootstrap2000、seed26101329 的全部结果在 CONTINUOUS_COMPARISON.json；没有全波或现实概率校准声明。", ""]
    (output / "CONTINUOUS_COMPARISON.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--evaluation", type=Path, default=Path("p2/results/evaluation"))
    parser.add_argument("--output", type=Path, default=Path("p2/results/continuous"))
    args = parser.parse_args()
    result = continuous_comparison(args.reference, args.evaluation, args.output)
    print(json.dumps(dict(status=result["status"], observations=result["observations"], models=result["models"], new_physics=0)))


if __name__ == "__main__":
    main()
