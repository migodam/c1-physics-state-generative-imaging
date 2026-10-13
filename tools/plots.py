"""Static plots from frozen C1 results; no training or physical forward calls.

Execute only in the parent's diagnostics job after locked evaluation. Missing
results are labelled, never replaced with zero metrics or invented samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from c1.data import reference
from c1.density import Density
from c1.material import z_to_x

DEFAULT_MODELS = ("mdn", "measurement", "state", "memory")
LABELS = {"mdn": "M-MDN", "measurement": "M-Y", "state": "M-S", "memory": "M-R",
          "mdn_state": "M-MDN-S", "mdn_memory": "M-MDN-R"}
COLORS = {"mdn": "#35618d", "measurement": "#c8752f", "state": "#43826b", "memory": "#926792",
          "mdn_state": "#43826b", "mdn_memory": "#926792"}
SEED_COLORS = ("#35618d", "#c8752f", "#43826b")
SEED_MARKERS = ("o", "s", "^")
CONF = json.loads((ROOT / "configs/protocol_v2.json").read_text())
FIRST_SEED = CONF["training_seeds"][0]
OUT = ROOT / "figures"


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def models_for(problem):
    """Frozen per-problem names precede config; never infer names from scores."""
    frozen = read_json(ROOT / "results" / problem / "MODELS_FROZEN.json")
    if frozen is not None and frozen.get("models"):
        return tuple(frozen["models"])
    per_problem = CONF.get("models_by_problem", {}).get(problem) or CONF.get(problem, {}).get("models")
    if per_problem:
        return tuple(per_problem)
    shared = CONF.get("models", DEFAULT_MODELS)
    return tuple(shared.get(problem, DEFAULT_MODELS) if isinstance(shared, dict) else shared)


def model_label(kind):
    return LABELS.get(kind, kind.replace("_", " "))


def model_color(kind):
    return COLORS.get(kind, "#35618d")


def ref_label(problem, evaluation):
    if evaluation is None:
        return "REFERENCE_UNRESOLVED / NOT_RUN" if problem == "p0" else "NOT_RUN"
    return "PASS (empirical)" if evaluation.get("reference_status") == "PASS" else "REFERENCE_UNRESOLVED"


def assert_frozen(problem):
    path = ROOT / "results" / problem / "MODELS_FROZEN.json"
    frozen = read_json(path)
    if frozen is None or frozen.get("status") != "FROZEN_BEFORE_TEST":
        raise RuntimeError(f"{problem.upper()} model freeze absent; locked test data will not be read")
    return frozen


def first_observation(problem):
    """Read only required observation/context arrays after checking the freeze."""
    assert_frozen(problem)
    folder = ROOT / "results" / problem
    with np.load(folder / "test.npz", allow_pickle=False) as data:
        obs = {key: np.array(data[key][0], copy=True) for key in ("y", "sigma", "kappa", "inputs", "states")}
    with np.load(folder / "normalization.npz", allow_pickle=False) as norm:
        obs["inputs_norm"] = ((obs["inputs"] - norm["ym"]) / norm["ys"]).astype(np.float32)
        obs["states_norm"] = ((obs["states"] - norm["sm"]) / norm["ss"]).astype(np.float32)
    # Material target x, truth fields, and residual rankings are not accessed.
    return obs


def load_density(problem, kind):
    checkpoint = ROOT / "results" / problem / "models" / f"{kind}-{FIRST_SEED}.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Density(saved["kind"], saved["y_dim"], saved["state_dim"], saved["target_parameters"]).cpu()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model


def metadata_line(y, sigma, kappa=None):
    text = "y = [" + ", ".join(f"{v:.3g}" for v in np.asarray(y).ravel()) + f"], sigma = {float(sigma):.3g}"
    if kappa is not None:
        text += f", kappa = {float(kappa):.3g}"
    return text


def save_figure(fig, name):
    paths = []
    for suffix in ("png", "pdf"):
        path = OUT / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def placeholder(name, message):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_axis_off()
    ax.text(.5, .65, "NOT_RUN", ha="center", va="center", fontsize=18, transform=ax.transAxes)
    ax.text(.5, .35, message, ha="center", va="center", fontsize=10, wrap=True, transform=ax.transAxes)
    return save_figure(fig, name)


def u0_density_scatter(evaluations, draw_count):
    evaluation = evaluations["u0"]
    if evaluation is None:
        raise RuntimeError("U0 EVALUATION.json missing")
    obs = first_observation("u0")
    models = models_for("u0")
    audit=read_json(ROOT/'results/u0/REFERENCE_AUDIT.json')
    order = (audit or {}).get('orders',CONF["u0"]["quadrature_orders"])[-1]
    grid_path = ROOT / "results/u0" / f"grid_{order}.npz"
    with np.load(grid_path, allow_pickle=False) as stored:
        grid = {key: np.array(stored[key], copy=True) for key in ("x", "w")}
    # Reuses the production cached-grid reference routine; zero physical solves.
    posterior = reference(grid, obs["y"], obs["sigma"], obs["kappa"], problem="u0")
    n = int(np.sqrt(len(grid["x"])))
    if n * n != len(grid["x"]):
        raise ValueError("U0 cached reference is not a tensor grid")
    xy = grid["x"].reshape(n, n, 2)
    density = np.exp(posterior["logdensity"]).reshape(n, n).T
    if not np.isfinite(density).all():
        raise ValueError("U0 cached reference density is nonfinite")
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(3.5 * (len(models) + 1), 4.2), sharex=True, sharey=True)
    image = axes[0].pcolormesh(xy[:, 0, 0], xy[0, :, 1], density, cmap="magma", shading="nearest", rasterized=True)
    axes[0].set_title("Cached-grid reference density")
    fig.colorbar(image, ax=axes[0], fraction=.046, pad=.035, label="Density in material coordinates")
    y = torch.as_tensor(obs["inputs_norm"][None], dtype=torch.float32)
    states = torch.as_tensor(obs["states_norm"][None], dtype=torch.float32)
    scatter_seed = 261013900 + FIRST_SEED
    for ax, kind in zip(axes[1:], models):
        model = load_density("u0", kind)
        generator = torch.Generator(device="cpu").manual_seed(scatter_seed)
        with torch.no_grad():
            sample = z_to_x(model.sample_z(y, states, draw_count, generator=generator)[0].numpy())
        if sample.shape != (draw_count, 2) or not np.isfinite(sample).all():
            raise ValueError(f"invalid U0 diagnostic q draws for {kind}")
        ax.scatter(sample[:, 0], sample[:, 1], s=3, alpha=.35, color=model_color(kind), edgecolors="none", rasterized=True)
        ax.set_title(f"{model_label(kind)}: {draw_count} iid q draws")
    bound = CONF["prior"]["bounds"][1]
    for ax in axes:
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect("equal")
        ax.axhline(0, color=".7", linewidth=.7)
        ax.axvline(0, color=".7", linewidth=.7)
        ax.set_xlabel("Material coordinate x1")
    axes[0].set_ylabel("Material coordinate x2")
    fig.suptitle(f"U0 | observation 0 | training seed {FIRST_SEED} | reference: {ref_label('u0', evaluation)}\n"
                 + metadata_line(obs["y"], obs["sigma"], obs["kappa"]), fontsize=12, y=1.02)
    fig.text(.5, .015, "Same frozen observation/context in every model. New iid diagnostic draws; no residual selection.\n"
             "Axes at zero define offline coordinate-sign events. Density values are not point probabilities.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.19, top=.83, wspace=.28)
    paths = save_figure(fig, "U0_REFERENCE_Q_SAMPLES")
    return dict(paths=paths, observation=0, training_seed=FIRST_SEED, scatter_seed=scatter_seed,
                draws_per_model=draw_count, reference_grid=str(grid_path.relative_to(ROOT)),
                sample_scope="new diagnostic iid q draws; evaluation metrics are not recomputed")


def model_metrics(evaluations):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    missing = []
    metrics = (("KL_reference", "Mean reference KL [nats]"), ("event_mass_MAE", "Mean event-mass MAE"))
    seeds = CONF["training_seeds"]
    offsets = np.linspace(-.16, .16, len(seeds))
    for row, problem in enumerate(("u0", "p0")):
        evaluation = evaluations[problem]
        models = models_for(problem)
        results = {} if evaluation is None else evaluation.get("models", {})
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            present = 0
            absent_models = set()
            for j, seed in enumerate(seeds):
                xs, ys = [], []
                for x, kind in enumerate(models):
                    value = results.get(f"{kind}-{seed}", {}).get(metric)
                    if value is None or not np.isfinite(value):
                        missing.append(f"{problem}/{kind}-{seed}/{metric}")
                        absent_models.add(kind)
                        continue
                    xs.append(x + offsets[j])
                    ys.append(float(value))
                if ys:
                    ax.scatter(xs, ys, s=48, color=SEED_COLORS[j], marker=SEED_MARKERS[j], label=str(seed), zorder=3)
                    present += len(ys)
            if not present:
                ax.text(.5, .5, "NOT_RUN: metrics absent", ha="center", transform=ax.transAxes)
            elif absent_models:
                ax.text(.02, .98, "Missing entries: " + ", ".join(model_label(k) for k in models if k in absent_models),
                        fontsize=8, va="top", transform=ax.transAxes)
            ax.set_xticks(range(len(models)), [model_label(k) for k in models])
            ax.set_xlim(-.5, len(models) - .5)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{problem.upper()} | reference: {ref_label(problem, evaluation)}")
            ax.grid(axis="y", alpha=.2)
            if present:
                ax.legend(title="Training seed", fontsize=8, title_fontsize=8)
    fig.suptitle("Frozen evaluation: separate results for three training seeds", fontsize=14)
    fig.text(.5, .015, "Dots are recorded means across test observations; training seeds are not pooled into a CI.\n"
             "Event reference refinement passes; large flow KL values are tail-sensitive and NOT converged. No general topology recovery.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.14, top=.89, hspace=.38, wspace=.26)
    return dict(paths=save_figure(fig, "MODEL_METRICS"), missing_entries=missing)


def paired_kl(evaluations):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3))
    missing = []
    for ax, problem in zip(axes, ("u0", "p0")):
        evaluation = evaluations[problem]
        state_models = [kind for kind in models_for(problem) if kind not in ("mdn", "measurement")]
        comparisons = {} if evaluation is None else evaluation.get("paired_comparisons", {})
        diagnostics=read_json(ROOT/'results/p0/DIAGNOSTICS.json') if problem=='p0' else None
        labels = []
        for row, (seed, kind) in enumerate((s, k) for s in CONF["training_seeds"] for k in state_models):
            if problem=='p0' and diagnostics is not None:
                item=diagnostics['paired_comparisons'].get(f'{kind}-vs-mdn-{seed}-KL_reference')
                if item is not None:item={**item,'baseline':'mdn'}
            else:item = comparisons.get(f"{kind}-{seed}")
            base = "NOT_RUN" if item is None else model_label(item.get("baseline", "UNKNOWN"))
            labels.append(f"{model_label(kind)} vs {base} | {seed}")
            if item is None:
                missing.append(f"{problem}/{kind}-{seed}")
                ax.text(.97, row, "NOT_RUN", va="center", ha="right", fontsize=9, transform=ax.get_yaxis_transform())
                continue
            mean = float(item["mean"])
            low, high = np.asarray(item["ci95"], dtype=float)
            if not np.isfinite([mean, low, high]).all() or low > high:
                raise ValueError(f"invalid paired CI: {problem}/{kind}-{seed}")
            # Draw endpoints directly: percentile CIs need not contain the mean.
            ax.hlines(row, low, high, color=model_color(kind), linewidth=2)
            ax.plot([low, high], [row, row], "|", color=model_color(kind), markersize=9)
            ax.plot(mean, row, "o", color=model_color(kind), markersize=6)
        ax.axvline(0, color=".4", linewidth=1, linestyle="--")
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_ylim(len(labels) - .5, -.5)
        ax.set_xlabel("KL(model) - KL(M-MDN) [nats]")
        ax.set_title(f"{problem.upper()} | " + ('same-family MDN ablation' if problem=='p0' else 'flow scores provisional'))
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Paired KL differences and recorded 95% confidence intervals", fontsize=14)
    fig.text(.5, .02, "Parent-paired bootstrap; noise is averaged within each parent. Training seeds remain separate.\n"
             "P0 compares the same MDN family; the original validation-selected flow comparison is retained in EVALUATION.json.\n"
             "U0 flow tails are numerically unstable; its CIs are provisional. Full cost and calibration still prevent promotion.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.17, right=.98, bottom=.16, top=.86, wspace=.70)
    return dict(paths=save_figure(fig, "PAIRED_KL_DIFFERENCES"), missing_entries=missing)


def material_pixels(x):
    """Exact fixed 13-dipole planar geometry, with exterior background epsilon=2."""
    x = np.asarray(x, dtype=float)
    if x.shape != (2,) or not np.isfinite(x).all():
        raise ValueError("gallery material must contain two finite continuous coefficients")
    pixels = np.full((5, 5), 2.)
    occupied = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            radius_squared = i*i + j*j
            if radius_squared <= 4:
                pixels[j + 2, i + 2] = 2 + x[0 if radius_squared <= 1 else 1]
                occupied.append((i, j))
    if len(occupied) != 13:
        raise RuntimeError("fixed dipole geometry mismatch")
    return pixels, occupied


def p0_gallery(evaluations):
    evaluation = evaluations["p0"]
    if evaluation is None:
        raise RuntimeError("P0 REFERENCE_UNRESOLVED / NOT_RUN: EVALUATION.json missing")
    assert_frozen("p0")
    models = models_for("p0")
    source = ROOT / "results/p0/GALLERY_SAMPLES.npz"
    with np.load(source, allow_pickle=False) as stored:
        samples = {kind: np.array(stored[kind], copy=True) for kind in models}
        y, sigma = np.array(stored["y"], copy=True), float(stored["sigma"])
    if any(a.shape != (6, 2) for a in samples.values()):
        raise ValueError("P0 gallery requires exactly six stored iid draws for each primary model")
    fig, axes = plt.subplots(len(models), 6, figsize=(13.8, 2.0 * len(models) + 1.1), squeeze=False)
    norm = Normalize(.5, 3.5)
    for row, kind in enumerate(models):
        for draw in range(6):
            ax = axes[row, draw]
            x = samples[kind][draw]
            pixels, occupied = material_pixels(x)
            image = ax.imshow(pixels, origin="lower", extent=(-.20, .20, -.20, .20),
                              cmap="viridis", norm=norm, interpolation="nearest")
            for i, j in occupied:
                ax.add_patch(Rectangle((i*.08-.04, j*.08-.04), .08, .08, fill=False, linewidth=.35, edgecolor=".25"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"core {2+x[0]:.2f} | shell {2+x[1]:.2f}", fontsize=8)
            if row == 0:
                ax.set_title(f"iid draw {draw+1}", fontsize=10)
            if draw == 0:
                ax.set_ylabel(model_label(kind), fontsize=11)
    color_ax = fig.add_axes([.92, .23, .018, .53])
    fig.colorbar(image, cax=color_ax, label="Real relative permittivity")
    fig.suptitle(f"P0 | observation 0 | training seed {FIRST_SEED} | reference: {ref_label('p0', evaluation)}\n"
                 + metadata_line(y, sigma), fontsize=12, y=.98)
    fig.text(.5, .032, "Fixed 13 dipoles on a 5 x 5 pixel display; exterior background = 2; active-cell loss = 0.02.\n"
             "Six stored iid q draws per model in original order; no residual selection. Every exact continuous image has point mass 0.\n"
             "Region masses appear in a separate figure. Restricted two-region support and intensity readout; no general topology recovery.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.895, bottom=.15, top=.86, hspace=.32, wspace=.20)
    return dict(paths=save_figure(fig, "P0_MATERIAL_GALLERY"), source=str(source.relative_to(ROOT)),
                observation=0, training_seed=FIRST_SEED, draw_order="stored first six iid draws; unchanged",
                point_mass=0, physical_geometry="13 dipoles; spacing 0.08; core 5 cells and shell 8 cells")


def p0_event_masses(evaluations):
    evaluation = evaluations["p0"]
    if evaluation is None:
        raise RuntimeError("P0 REFERENCE_UNRESOLVED / NOT_RUN: EVALUATION.json missing")
    assert_frozen("p0")
    models = models_for("p0")
    source = ROOT / "results/p0/OBSERVATION_METRICS.jsonl"
    rows = {}
    with source.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item["observation"] == 0 and item["seed"] == FIRST_SEED and item["model"] in models:
                rows[item["model"]] = item
    if set(rows) != set(models):
        raise ValueError("P0 observation-0 first-seed event masses are incomplete")
    q = np.array([rows[k]["q_event_masses"] for k in models], dtype=float)
    p = np.array([rows[k]["reference_event_masses"] for k in models], dtype=float)
    if q.shape != (len(models), 4) or not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(axis=1), 1):
        raise ValueError("invalid P0 learned event-mass records")
    if p.shape != (len(models), 4) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1) or not np.allclose(p, p[0]):
        raise ValueError("P0 models do not share the same observation reference events")
    n = CONF["evaluation_draws"]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    x = np.arange(4)
    width = .8 / (len(models) + 1)
    for j, kind in enumerate(models):
        se = np.sqrt(q[j] * (1-q[j]) / n)
        ax.bar(x + (j-len(models)/2)*width, q[j], width, color=model_color(kind), label=model_label(kind), yerr=1.96*se,
               capsize=2, error_kw={"linewidth": .8})
    unresolved = evaluation.get("reference_status") != "PASS"
    ax.bar(x + len(models)/2*width, p[0], width, color=".80", edgecolor=".25", hatch="//" if unresolved else None,
           label="Reference (provisional)" if unresolved else "Reference (empirical)")
    ax.set_xticks(x, ["(--)", "(-+)", "(+-)", "(++)"])
    ax.set_xlabel("Offline signs of (x_core, x_shell), relative to background permittivity 2")
    ax.set_ylabel("Conditional region mass")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=len(models) + 1, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.21))
    ax.set_title(f"P0 | observation 0 | training seed {FIRST_SEED} | reference: {ref_label('p0', evaluation)}")
    fig.text(.5, .015, f"Recorded q masses use {n} iid evaluation draws; error bars are binomial MC +/- 1.96 SE.\n"
             "These are event masses, not per-image probabilities or topology classes. Six gallery draws do not determine these masses.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.30, top=.88)
    return dict(paths=save_figure(fig, "P0_EVENT_MASSES"), source=str(source.relative_to(ROOT)),
                observation=0, training_seed=FIRST_SEED, evaluation_draws=n,
                q_event_masses={k: rows[k]["q_event_masses"] for k in models})


def physics_snapshot():
    counts = dict(material_frequency=0, forward_rhs=0, derivative_rhs=0, adjoint_rhs=0)
    path = ROOT / "results/PHYSICS_LEDGER.jsonl"
    if path.exists():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["event"] == "charge":
                    for key in counts:
                        counts[key] += row.get(key, 0)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scatter-draws", type=int, default=CONF["evaluation_draws"])
    args = parser.parse_args()
    if not 1 <= args.scatter_draws <= CONF["evaluation_draws"]:
        parser.error("scatter draws must be between 1 and the declared evaluation draw count")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    OUT.mkdir(parents=True, exist_ok=True)
    before = physics_snapshot()
    evaluations = {p: read_json(ROOT / "results" / p / "EVALUATION.json") for p in ("u0", "p0")}
    output = {}
    tasks = [("U0_REFERENCE_Q_SAMPLES", lambda: u0_density_scatter(evaluations, args.scatter_draws)),
             ("MODEL_METRICS", lambda: model_metrics(evaluations)),
             ("P0_MATERIAL_GALLERY", lambda: p0_gallery(evaluations)),
             ("P0_EVENT_MASSES", lambda: p0_event_masses(evaluations)),
             ("PAIRED_KL_DIFFERENCES", lambda: paired_kl(evaluations))]
    for name, operation in tasks:
        try:
            output[name] = dict(status="WRITTEN", **operation())
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            plt.close("all")
            output[name] = dict(status="NOT_RUN", error=f"{type(error).__name__}: {error}",
                                paths=placeholder(name, str(error)))
    after = physics_snapshot()
    if before != after:
        raise RuntimeError("physical ledger changed while plotting; parent must review concurrent activity")
    complete = all(v["status"] == "WRITTEN" and not v.get("missing_entries") for v in output.values())
    manifest = dict(status="COMPLETE" if complete else "PARTIAL", figures=output,
                    reference_status={p: ref_label(p, evaluations[p]) for p in evaluations},
                    primary_models={p: list(models_for(p)) for p in evaluations},
                    physics_before=before, physics_after=after, new_physical_evaluations=0,
                    source_schema="src/c1/evaluate.py V2", figures_are="visualizations of recorded or frozen-model outputs, not scientific acceptance",
                    p0_scope="restricted continuous core/shell material; real intensity readout; no general topology recovery",
                    missing_values="explicit NOT_RUN; no invented metrics, samples, or confidence intervals")
    (OUT / "PLOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"plot_status": manifest["status"], "manifest": "figures/PLOT_MANIFEST.json",
                      "new_physical_evaluations": 0, "reference_status": manifest["reference_status"]}))


if __name__ == "__main__":
    main()
