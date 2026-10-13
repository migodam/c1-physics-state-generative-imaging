"""Render C1-P1 scientific figures from completed CSV and training receipts.

This module reads saved scalar results only.  It never imports a model, a
reference evaluator, a forward adapter, Torch, or the original C1 backend.
Execute inside the parent's controlled job; writing this file runs nothing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


P1 = Path(__file__).resolve().parents[1]
METHODS = ("diag_joint", "full_joint", "diag_distill", "full_distill")
LABELS = {
    "diag_joint": "Diagonal\njoint",
    "full_joint": "Full covariance\njoint",
    "diag_distill": "Diagonal\ndistilled",
    "full_distill": "Full covariance\ndistilled",
}
PANELS = (
    ("u0", "exposed", "U0 exposed regression"),
    ("p0", "exposed", "P0 exposed regression"),
    ("u0", "fresh_final", "U0 fresh final panel"),
)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def number(row, field):
    value = row.get(field)
    if value in (None, "", "None", "null"):
        return math.nan
    return float(value)


def load_results(path):
    required = {
        "problem", "panel", "condition", "method", "seed", "KL", "event_MAE",
        "CDF_MAE", "tail_MAE", "coverage90", "region_mean_error", "region_cov_error",
    }
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Results CSV lacks fields: {sorted(missing)}")
        # Optional state/memory/historical rows are not primary factorial data.
        records = [row for row in reader
                   if row["condition"] == "measurement" and row["method"] in METHODS]
    indexed = {}
    for row in records:
        seed = int(row["seed"])
        key = (row["problem"], row["panel"], row["method"], seed)
        if key in indexed:
            raise ValueError(f"Duplicate primary result row: {key}")
        indexed[key] = row
    seeds = sorted({key[-1] for key in indexed})
    if not seeds:
        raise ValueError("No primary measurement results in CSV")
    missing_rows = [
        (problem, panel, method, seed)
        for problem, panel, _ in PANELS for method in METHODS for seed in seeds
        if (problem, panel, method, seed) not in indexed
    ]
    if missing_rows:
        raise ValueError(f"Incomplete paired primary panels: {missing_rows}")
    return indexed, seeds


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 180,
    })


def decorate(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)


def seed_legend(seeds):
    return [Line2D([], [], color=COLORS[i % len(COLORS)], marker="o",
                   linewidth=1.2, markersize=4, label=f"Training seed {seed}")
            for i, seed in enumerate(seeds)]


def paired_lines(ax, records, seeds, problem, panel, metric, refined=None):
    unavailable = 0
    all_values = []
    for i, seed in enumerate(seeds):
        color = COLORS[i % len(COLORS)]
        offset = 0.065 * (i - (len(seeds)-1)/2)
        xs = [j + offset for j in range(len(METHODS))]
        ys = [number(records[(problem, panel, m, seed)], metric) for m in METHODS]
        unavailable += sum(not math.isfinite(v) for v in ys)
        all_values.extend(v for v in ys if math.isfinite(v))
        ax.plot(xs, ys, color=color, marker="o", markersize=4.5, linewidth=1.2)
        if refined:
            ry = [number(records[(problem, panel, m, seed)], refined) for m in METHODS]
            unavailable += sum(not math.isfinite(v) for v in ry)
            all_values.extend(v for v in ry if math.isfinite(v))
            ax.plot(xs, ry, color=color, marker="D", markerfacecolor="none",
                    markersize=5, linestyle="--", linewidth=0.85, alpha=0.8)
    if unavailable:
        ax.text(0.02, 0.96, f"Missing/nonfinite points: {unavailable}",
                transform=ax.transAxes, va="top", color="#9D2525", fontsize=8)
    return all_values


def save_pair(fig, output, stem):
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def density_figure(records, seeds, output):
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.9), squeeze=False)
    for col, (problem, panel, title) in enumerate(PANELS):
        axes[0, col].set_title(title)
        paired_lines(axes[0, col], records, seeds, problem, panel, "KL", "KL_refined")
        paired_lines(axes[1, col], records, seeds, problem, panel, "event_MAE", "event_MAE_refined")
        decorate(axes[0, col], "Reference KL (nats)")
        decorate(axes[1, col], "Event mass MAE")
    handles = seed_legend(seeds) + [
        Line2D([], [], color="#333333", marker="o", linewidth=1.2,
               label="Original saved-bank quadrature"),
        Line2D([], [], color="#333333", marker="D", markerfacecolor="none",
               linestyle="--", linewidth=0.85, label="Refined numerical cross-check"),
    ]
    fig.suptitle("C1-P1: posterior density and event probability, every seed retained", y=0.98, fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
               ncol=min(len(handles), 5), frameon=False)
    fig.text(0.06, 0.045,
             "Lines pair the same training seed across all four primary methods; no pooled seed CI.\n"
             "P0 refined values reuse the saved response bank by interpolation; original-bank KL is retained.\n"
             "Fresh U0 was generated after model selection and is now exposed. These are restricted material-coordinate benchmarks.",
             fontsize=8.5, va="bottom")
    fig.subplots_adjust(top=0.83, bottom=0.22, left=0.075, right=0.99, hspace=0.44, wspace=0.29)
    return save_pair(fig, output, "P1_DENSITY_EVENT_PAIRED")


def training_figure(seeds, receipts_root, output):
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 13.0), squeeze=False)
    for row, method in enumerate(METHODS):
        for i, seed in enumerate(seeds):
            path = receipts_root / "p0" / "models" / f"{method}-measurement-{seed}.json"
            receipt = json.loads(path.read_text())
            history = sorted(receipt["history"], key=lambda item: int(item["step"]))
            if not history:
                raise ValueError(f"Empty training history: {path}")
            steps = [int(h["step"]) for h in history]
            train = [float(h["training_batch_NLL_z"]) for h in history]
            val = [float(h["validation_expected_NLL_x"]) for h in history]
            color = COLORS[i % len(COLORS)]
            axes[row, 0].plot(steps, train, color=color, marker=".", markersize=4, linewidth=1)
            axes[row, 1].plot(steps, val, color=color, marker=".", markersize=4, linewidth=1)
            selected = int(receipt["checkpoint"])
            for step, score in zip(steps, val):
                if step == selected:
                    axes[row, 1].plot(step, score, color=color, marker="o", markersize=7,
                                      markerfacecolor="none", markeredgewidth=1.3)
            axes[row, 1].axvline(selected, color=color, alpha=0.18, linewidth=0.8)
            for col, values in enumerate((train, val)):
                bad = sum(not math.isfinite(value) for value in values)
                if bad:
                    axes[row, col].text(0.02, 0.96, f"Seed {seed}: {bad} nonfinite values",
                                        transform=axes[row, col].transAxes, va="top", fontsize=8)
        short = LABELS[method].replace("\n", " ")
        axes[row, 0].set_title(f"{short}\nTraining minibatch, z coordinates")
        axes[row, 1].set_title(f"{short}\nValidation posterior expectation, x coordinates")
        axes[row, 0].set_ylabel("Training batch NLL_z (nats)")
        axes[row, 1].set_ylabel("Validation expected NLL_x (nats)")
        for ax in axes[row]:
            ax.set_xlabel("Optimizer update")
            ax.grid(axis="y", alpha=0.22, linewidth=0.6)
            ax.set_axisbelow(True)
    handles = seed_legend(seeds) + [
        Line2D([], [], color="#333333", marker="o", markerfacecolor="none",
               linestyle="none", markersize=7, label="Selected expected-NLL checkpoint"),
    ]
    fig.suptitle("P0 exposed regression: optimization histories, all three seeds", y=0.985, fontsize=13)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=4, frameon=False)
    fig.text(0.06, 0.023,
             "Left and right use different variables and separate y axes: NLL_z and NLL_x differ by the transformation Jacobian.\n"
             "Training values are individual stochastic minibatches; validation values are posterior expectations. Absolute gaps are not comparable.\n"
             "No averaged-seed confidence band and no new model or physical evaluation are used.", fontsize=8.5, va="bottom")
    fig.subplots_adjust(top=0.89, bottom=0.105, left=0.08, right=0.99, hspace=0.65, wspace=0.29)
    return save_pair(fig, output, "P1_P0_TRAINING_HISTORIES")


def probability_figure(records, seeds, output):
    metrics = (
        ("CDF_MAE", "Continuous CDF MAE"),
        ("tail_MAE", "Tail probability MAE"),
        ("coverage90", "90% interval coverage"),
        ("region_mean_error", "Within-region mean L2 error"),
        ("region_cov_error", "Within-region covariance error"),
    )
    fig, axes = plt.subplots(len(metrics), 3, figsize=(14.0, 14.5), squeeze=False)
    for col, (problem, panel, title) in enumerate(PANELS):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            values = paired_lines(ax, records, seeds, problem, panel, metric)
            decorate(ax, ylabel)
            if row == 0:
                ax.set_title(title)
            if metric == "coverage90":
                ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1)
                ax.text(0.98, 0.90, "target 0.90", ha="right", va="bottom", fontsize=8,
                        transform=ax.get_yaxis_transform())
                if values and all(0 <= value <= 1 for value in values):
                    lo = max(0, min(values + [0.90]) - 0.045)
                    hi = min(1, max(values + [0.90]) + 0.045)
                    ax.set_ylim(lo, hi)
    fig.suptitle("C1-P1: distribution quality beyond candidate diversity", y=0.985, fontsize=13)
    fig.legend(handles=seed_legend(seeds), loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=min(len(seeds), 5), frameon=False)
    fig.text(0.055, 0.018,
             "CDF, tails and within-region target moments use numerically refined references; P0 refinement is bank-interpolated.\n"
             "Within-region sample moments use 2048 iid draws per observation; their finite Monte Carlo error is recorded in the result tables.\n"
             "Old U0/P0 panels are exposed regression; the independent final U0 panel is now exposed for future work. No arbitrary-topology imaging claim.",
             fontsize=8.5, va="bottom")
    fig.subplots_adjust(top=0.90, bottom=0.105, left=0.085, right=0.99, hspace=0.57, wspace=0.31)
    return save_pair(fig, output, "P1_PROBABILITY_QUALITY")


def main(results=None, output=None, receipts_root=None):
    results = Path(results) if results is not None else P1 / "C1_P1_RESULTS.csv"
    output = Path(output) if output is not None else P1 / "figures"
    receipts_root = Path(receipts_root) if receipts_root is not None else P1 / "results"
    records, seeds = load_results(results)
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = density_figure(records, seeds, output)
    artifacts += training_figure(seeds, receipts_root, output)
    artifacts += probability_figure(records, seeds, output)
    manifest = {
        "source_csv": str(results), "source_receipts_root": str(receipts_root),
        "methods": list(METHODS), "condition": "measurement",
        "seeds": seeds, "panels": [list(p[:2]) for p in PANELS],
        "artifacts": [str(path) for path in artifacts],
        "scope": "Saved scalar metrics and JSON histories only; no models or physics executed.",
        "seed_display": "Every seed plotted individually, paired by method; no seed-averaged CI.",
        "KL_display": "Original-bank KL plus distinct refined numerical cross-check.",
        "exposure": "Old panels exposed; independent final U0 is now exposed after evaluation.",
        "imaging_scope": "Restricted continuous material-coordinate benchmark, not unknown-topology imaging.",
    }
    (output / "P1_PLOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": [str(path) for path in artifacts], "new_physics": 0}))
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=P1 / "C1_P1_RESULTS.csv")
    parser.add_argument("--output", type=Path, default=P1 / "figures")
    parser.add_argument("--receipts-root", type=Path, default=P1 / "results")
    args = parser.parse_args()
    main(args.results, args.output, args.receipts_root)
