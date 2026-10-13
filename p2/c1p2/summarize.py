"""Summarize actual frozen-proposal rows; parent is the bootstrap unit.

This reads one completed evaluator output without generating new density or
physical samples.  Seeds remain separate; pooled seed comparisons first
average paired differences within each parent.  Claims concern the declared
numerical proxy target, never continuum or real-world posterior calibration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from .train import write_json


BOOTSTRAP_SEED = 26101329
BOOTSTRAP_REPLICATES = 2000


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def row_metrics(row):
    """Extract parallel metrics while retaining unresolved conditional CDFs."""
    result = {}
    direct = {
        "event_mass_MAE": "event_mass_MAE",
        "reference_KL_to_proposal": "reference_KL_to_proposal",
        "proposal_expected_log_score_x": "proposal_expected_log_score_x",
        "truth_continuous_log_score_x": "observed_truth_log_score_x",
        "truth_event_Brier_score": "event_Brier_score",
        "reference_truth_event_Brier_score": "reference_event_Brier_score",
        "truth_event_log_score_MC_Jeffreys": "truth_event_log_score_MC_Jeffreys",
        "reference_truth_event_log_score": "reference_truth_event_log_score",
        "significant_event_count_recall": "significant_event_count_recall",
        "significant_event_mass_recall": "significant_event_mass_recall",
        "sample_hit_event_count_recall": "sample_hit_event_count_recall",
        "false_exclusion_rate": "false_exclusion",
        "false_confidence_rate": "false_confidence",
        "false_excluded_reference_mass": "false_excluded_reference_mass",
        "top_k": "top_k",
        "top_k_represented_model_mass": "top_k_represented_model_mass",
        "top_k_represented_reference_mass": "top_k_represented_reference_mass",
        "important_event_log_odds_MAE": "important_event_log_odds_MAE",
        "sample_reference_event_mass_MAE": "sample_reference_event_mass_MAE",
    }
    for destination, source in direct.items():
        result[destination] = _finite(row.get(source))
    for label in ("physical", "latent"):
        cdf = row.get("physical_cdf_max_abs" if label == "physical" else "cdf_max_abs")
        result[f"max_{label}_CDF_error"] = float(np.max(cdf)) if cdf is not None else None
        coverage = row.get(f"truth_90pct_marginal_coverage_{label}")
        reference_coverage = row.get(f"reference_truth_90pct_marginal_coverage_{label}")
        if coverage is not None:
            result[f"truth_90pct_marginal_coverage_{label}_mean"] = float(np.mean(coverage))
            for j, item in enumerate(coverage):
                result[f"truth_90pct_coverage_{label}_coordinate{j}"] = float(item)
        if reference_coverage is not None:
            for j, item in enumerate(reference_coverage):
                result[f"reference_truth_90pct_coverage_{label}_coordinate{j}"] = float(item)
        result[f"truth_90pct_simultaneous_coverage_{label}"] = _finite(row.get(f"truth_90pct_simultaneous_coverage_{label}"))
    important_cdf = []
    unresolved_mass = 0.
    for conditional in row.get("conditional_material", []):
        if conditional["reference_mass"] >= .05:
            values = conditional.get("conditional_physical_cdf_max_abs")
            if values is None:
                unresolved_mass += conditional["reference_mass"]
            else:
                important_cdf.append(float(np.max(values)))
    result["important_conditional_physical_CDF_max"] = max(important_cdf) if important_cdf else None
    result["important_conditional_CDF_unresolved_reference_mass"] = unresolved_mass
    return result


def _mean_present(values):
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def parent_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_id"])].append(row_metrics(row))
    result = {}
    for parent, items in grouped.items():
        keys = sorted({key for item in items for key in item})
        result[parent] = {key: _mean_present([item.get(key) for item in items]) for key in keys}
    return result


def summarize_group(rows):
    parents = parent_metrics(rows)
    keys = sorted({key for values in parents.values() for key in values})
    metrics = {}
    available_counts = {}
    for key in keys:
        values = [item.get(key) for item in parents.values()]
        metrics[key] = _mean_present(values)
        available_counts[key] = sum(value is not None for value in values)
    physical_names = rows[0].get("physical_coordinate_names")
    return dict(observations=len(rows), parents=len(parents),
                physical_coordinate_names=physical_names, parent_equal_weight_means=metrics,
                available_parent_counts=available_counts,
                aggregation="two noise observations averaged within parent, then equal-weight parents",
                raw_zero_count_truth_event_log_scores=sum(row.get("truth_event_log_score_raw_MC") is None for row in rows),
                proposal_status="raw learned proposal versus stated proxy target; not full-Mie corrected",
                interval_status="marginal 90% equal-tail coverage; no nominal claim for joint coordinate box",
                conditional_CDF_status="unresolved important-event mass reported separately; available CDF mean does not hide missing events")


def bootstrap_pair(left, right, rng):
    """Paired full-minus-diagonal delta, with parent-resampling 95% intervals."""
    common = sorted(set(left) & set(right))
    keys = sorted({key for parent in common for key in set(left[parent]) & set(right[parent])})
    comparisons = {}
    for key in keys:
        available = [parent for parent in common if left[parent].get(key) is not None and right[parent].get(key) is not None]
        if not available:
            comparisons[key] = dict(delta=None, interval95=None, parents=0)
            continue
        delta = np.array([left[parent][key] - right[parent][key] for parent in available])
        indices = rng.integers(len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
        draws = delta[indices].mean(-1)
        comparisons[key] = dict(delta=float(delta.mean()), interval95=np.quantile(draws, [.025, .975]).tolist(),
                                parents=len(available), sign="full_distill minus diag_distill")
    return comparisons


def summarize(evaluation_path, output):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    payload = json.loads(Path(evaluation_path).read_text())
    rows = payload["rows"]
    if not rows:
        raise ValueError("actual evaluator rows required")
    identity = payload["target_identity"]
    if any(row["target_identity"] != identity for row in rows):
        raise ValueError("different posterior targets cannot share a summary")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["seed"]))].append(row)
    seed_summaries = []
    for (method, seed), group in sorted(grouped.items()):
        seed_summaries.append(dict(method=method, seed=seed, **summarize_group(group)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired = []
    pooled_parent_deltas = defaultdict(list)
    seeds = sorted({seed for method, seed in grouped if method == "full_distill"})
    for seed in seeds:
        if ("diag_distill", seed) not in grouped:
            continue
        left = parent_metrics(grouped[("full_distill", seed)])
        right = parent_metrics(grouped[("diag_distill", seed)])
        paired.append(dict(seed=seed, metrics=bootstrap_pair(left, right, rng)))
        for parent in set(left) & set(right):
            delta = {key: left[parent][key] - right[parent][key] for key in set(left[parent]) & set(right[parent])
                     if left[parent][key] is not None and right[parent][key] is not None}
            pooled_parent_deltas[parent].append(delta)
    # Keep parents independent: repeated model seeds are averaged before the
    # bootstrap, rather than treated as 3×64 new independent experimental units.
    pooled_left, pooled_right = {}, {}
    for parent, values in pooled_parent_deltas.items():
        keys = {key for value in values for key in value}
        pooled_left[parent] = {key: _mean_present([value.get(key) for value in values]) for key in keys}
        pooled_right[parent] = {key: 0. for key in keys}
    pooled_comparison = bootstrap_pair(pooled_left, pooled_right, rng) if pooled_left else {}
    report = dict(target_identity=identity, status="ACTUAL_RAW_PROPOSAL_COMPARISON",
                  seed_separated=seed_summaries, paired_full_vs_diagonal_distillation=paired,
                  paired_seed_mean_parent_bootstrap=pooled_comparison,
                  bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES,
                                 unit="parent", independent_seed_claim=False,
                                 uncertainty_scope="conditional on saved reference and fixed proposal MC draws; excludes reference/solver error"),
                  scope="same stated numerical proxy target and same teacher supervision for the paired distillation comparison",
                  calibration_status="not a continuum/full-Mie/real-data calibration claim",
                  source=str(Path(evaluation_path).resolve()), material_frequency_evaluations=0,
                  wall_seconds=time.perf_counter() - start_wall, cpu_seconds=time.process_time() - start_cpu)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "METHOD_SUMMARY.json").exists():
        raise RuntimeError("completed P2 statistical summary not overwritten")
    write_json(output / "METHOD_SUMMARY.json", report)
    header = ["method", "seed", "parent", "event MAE", "KL", "mass recall", "false exclusion", "false confidence", "Top-K reference mass", "Brier", "truth log score", "physical 90% coverage"]
    markdown = ["# C1 P2：共同 target 的实际生成器结果", "", "同一数值 proxy target；这些是未做 full-likelihood correction 的模型分布结果。", "",
                "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    def fmt(value):
        return "NOT_AVAILABLE" if value is None else f"{value:.5g}"
    for row in seed_summaries:
        m = row["parent_equal_weight_means"]
        values = [row["method"], str(row["seed"]), str(row["parents"]),
                  *[fmt(m.get(key)) for key in ("event_mass_MAE", "reference_KL_to_proposal", "significant_event_mass_recall",
                                               "false_exclusion_rate", "false_confidence_rate", "top_k_represented_reference_mass",
                                               "truth_event_Brier_score", "truth_continuous_log_score_x", "truth_90pct_marginal_coverage_physical_mean")]]
        markdown.append("| " + " | ".join(values) + " |")
    markdown.extend(["", "配对比较使用相同 seed 的 full/diagonal distillation；两个噪声观测先按 parent 合并。",
                     "三个 seed 仍分列，汇总 bootstrap 先对每个 parent 的 seed 差求均值，避免伪增独立样本数。",
                     "2000次 parent 配对 bootstrap、seed 26101329 的全部区间在 METHOD_SUMMARY.json。",
                     "CDF/quantile 的参考来自保存的连续 target 节点或样本；主要事件质量优先使用独立提供的积分质量。",
                     "事件 log score 的零计数使用明确的 Jeffreys .5/count 稳定估计；原始零计数保留。",
                     "半径为零的几何原子保留在材料 CDF 中；坐标区间的同时覆盖不是90%联合可信域。", ""])
    (output / "METHOD_SUMMARY.md").write_text("\n".join(markdown))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("p2/results/summary"))
    args = parser.parse_args()
    result = summarize(args.evaluation, args.output)
    print(json.dumps(dict(target_identity=result["target_identity"],
                          models=len(result["seed_separated"]), material_frequency_evaluations=0)))


if __name__ == "__main__":
    main()
