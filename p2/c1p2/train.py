"""One CPU-only, jointly budgeted P2 density training job on paid B1 arrays.

Every method and seed receives the same actual update count.  All checkpoint
selection uses the same posterior-expected validation log score; no final test
file or test observation is accepted here.  This imports no physical solver.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .density import DEFAULT_SEEDS, METHODS, LOWER, UPPER, model_for


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _weights(data, key, shape):
    w = np.asarray(data[key], dtype=np.float64) if key in data else np.ones(shape)
    if w.shape != shape or not np.isfinite(w).all() or np.any(w < 0) or np.any(w.sum(-1) <= 0):
        raise ValueError(key + " must contain valid posterior probability weights")
    return w / w.sum(-1, keepdims=True)


def load_training_data(path):
    with np.load(path, allow_pickle=False) as bank:
        data = {key: bank[key] for key in bank.files}
    required = ("train_y", "train_x", "train_teacher", "validation_y",
                "validation_teacher", "target_identity")
    if any(key not in data for key in required):
        raise ValueError("missing training dataset keys: " + str(required))
    y, vy = np.asarray(data["train_y"]), np.asarray(data["validation_y"])
    x, teacher, val_teacher = (np.asarray(data[key]) for key in
                               ("train_x", "train_teacher", "validation_teacher"))
    if y.ndim != 2 or vy.ndim != 2 or y.shape[1] != vy.shape[1] or not len(y) or not len(vy):
        raise ValueError("nonempty [N, condition_dim] train/validation observations required")
    if x.shape != (len(y), 4) or teacher.ndim != 3 or teacher.shape[0] != len(y) or teacher.shape[-1] != 4:
        raise ValueError("train coordinates/teacher must have [N,4]/[N,T,4]")
    if val_teacher.ndim != 3 or val_teacher.shape[0] != len(vy) or val_teacher.shape[-1] != 4:
        raise ValueError("validation teacher must have [V,T,4]")
    for name, values in (("train_y", y), ("validation_y", vy), ("train_x", x),
                         ("train_teacher", teacher), ("validation_teacher", val_teacher)):
        if not np.isfinite(values).all():
            raise ValueError(name + " contains nonfinite values")
        if name not in {"train_y", "validation_y"} and not np.all((values > LOWER) & (values < UPPER)):
            raise ValueError(name + " contains a support endpoint; no silent clipping allowed")
    identity = str(np.asarray(data["target_identity"]).item())
    if not identity.strip():
        raise ValueError("explicit target identity required")
    tw = _weights(data, "train_teacher_weights", teacher.shape[:2])
    vw = _weights(data, "validation_teacher_weights", val_teacher.shape[:2])
    teacher_cdf = np.cumsum(tw, -1)
    teacher_cdf[:, -1] = 1.0
    return dict(y=torch.tensor(y, dtype=torch.float32), vy=torch.tensor(vy, dtype=torch.float32),
                x=torch.tensor(x, dtype=torch.float32),
                teacher=torch.tensor(teacher, dtype=torch.float32),
                val_teacher=torch.tensor(val_teacher, dtype=torch.float32),
                teacher_cdf=teacher_cdf,
                val_weights=torch.tensor(vw, dtype=torch.float64),
                target_identity=identity, condition_dim=y.shape[1],
                parent_counts={name: int(len(np.unique(data[name]))) for name in
                               ("train_parent", "validation_parent") if name in data})


def expected_validation_score(model, data):
    model.eval()
    with torch.no_grad():
        values = model.log_prob(data["val_teacher"], data["vy"])
        score = -(values.double() * data["val_weights"]).sum(-1).mean()
    model.train()
    if not torch.isfinite(score):
        raise RuntimeError("nonfinite common posterior-expected validation NLL")
    return float(score)


def train_job(dataset, output, methods=METHODS, seeds=DEFAULT_SEEDS,
              max_steps=3000, budget_seconds=600, batch=128,
              learning_rate=1e-3, checkpoint_every=250):
    """Run matched CPU training; return actual counts and all failures intact."""
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    output = Path(output)
    if output.exists() and (list(output.glob("*.pt")) or (output / "TRAINING_SUMMARY.json").exists()):
        raise RuntimeError("existing P2 training artifacts are not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    data = load_training_data(dataset)
    runs = []
    for seed in seeds:
        for method in methods:
            torch.manual_seed(seed)
            model = model_for(method, data["condition_dim"], data["target_identity"])
            model.fit_normalization(data["y"])
            runs.append(dict(method=method, seed=int(seed), model=model,
                             optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
                             rng=np.random.default_rng(seed), history=[], best=float("inf"),
                             best_state=None, best_step=None, training_wall=0.0, training_cpu=0.0,
                             completed_updates=0))
    stop_reason, failure, steps = "MAX_UPDATES", None, 0
    max_cycle, max_validation = 0.0, 0.0

    def checkpoint(actual_step):
        nonlocal max_validation
        before = time.perf_counter()
        for run in runs:
            score = expected_validation_score(run["model"], data)
            run["history"].append(dict(step=actual_step, validation_expected_NLL_x=score,
                                       batch_training_NLL_x=run.get("last_loss")))
            if score < run["best"]:
                run["best"], run["best_step"] = score, actual_step
                run["best_state"] = copy.deepcopy(run["model"].state_dict())
        max_validation = max(max_validation, time.perf_counter() - before)
        print(json.dumps(dict(event="checkpoint", steps=actual_step,
                              elapsed_wall=time.perf_counter() - start_wall,
                              best_scores={f"{r['method']}-{r['seed']}": r["best"] for r in runs})), flush=True)

    try:
        for step in range(1, max_steps + 1):
            # Reserve completion/validation time; the common cycle is indivisible.
            reserve = max(10.0, 3 * max_cycle + 2 * max_validation)
            if time.perf_counter() - start_wall + reserve >= budget_seconds:
                stop_reason = "COMMON_WALL_BUDGET"
                break
            before = time.perf_counter()
            # Rotate execution order without changing per-run RNG or supervision.
            ordered = runs[(step - 1) % len(runs):] + runs[:(step - 1) % len(runs)]
            for run in ordered:
                wall, cpu = time.perf_counter(), time.process_time()
                ii = run["rng"].integers(len(data["y"]), size=batch)
                if run["method"].endswith("distill"):
                    uniform = run["rng"].random(batch)
                    indices = np.array([np.searchsorted(data["teacher_cdf"][i], u, side="right")
                                        for i, u in zip(ii, uniform)])
                    targets = data["teacher"][ii, indices]
                else:
                    targets = data["x"][ii]
                optimizer, model = run["optimizer"], run["model"]
                optimizer.zero_grad(set_to_none=True)
                loss = -model.log_prob(targets, data["y"][ii]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss: {run['method']} seed {run['seed']} step {step}")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not torch.isfinite(gradient):
                    raise RuntimeError("nonfinite density gradient")
                optimizer.step()
                run["completed_updates"] += 1
                run["last_loss"] = float(loss.detach())
                run["training_wall"] += time.perf_counter() - wall
                run["training_cpu"] += time.process_time() - cpu
            steps = step
            max_cycle = max(max_cycle, time.perf_counter() - before)
            if step % checkpoint_every == 0:
                checkpoint(step)
        if steps == 0:
            raise RuntimeError("training wall budget exhausted before one common update")
        if steps % checkpoint_every:
            checkpoint(steps)
    except Exception as exc:
        stop_reason, failure = "FAILED", dict(type=type(exc).__name__, message=str(exc), common_completed_steps=steps)
        write_json(output / "FAILURE.json", failure)
    receipts = []
    for run in runs:
        name = f"{run['method']}-{run['seed']}"
        if run["best_state"] is None:
            # Preserve a failed fit, explicitly excluded from selected models.
            torch.save(dict(configuration=run["model"].configuration(),
                            state_dict=run["model"].state_dict(), failed=True), output / f"{name}-FAILED.pt")
            continue
        receipt = dict(method=run["method"], seed=run["seed"], updates=run["completed_updates"],
                       common_completed_updates=steps,
                       selected_checkpoint=run["best_step"], validation_expected_NLL_x=run["best"],
                       selection="common posterior-expected validation log score",
                       objective="posterior distillation" if run["method"].endswith("distill") else "joint truth NLL",
                       observations=len(data["y"]), validation_observations=len(data["vy"]),
                       teacher_draws_per_observation=data["teacher"].shape[1],
                       parent_counts=data["parent_counts"], parameters=run["model"].parameter_count,
                       history=run["history"], training_wall_seconds=run["training_wall"],
                       training_cpu_seconds=run["training_cpu"], material_frequency_evaluations=0,
                       topology_input=False, state_input=False, memory_input=False,
                       normalization="train observations only", target_identity=data["target_identity"],
                       teacher_cost="paid B1 bank; setup/amortized cost is in the shared B1 ledger",
                       failed_job=bool(failure))
        torch.save(dict(configuration=run["model"].configuration(), state_dict=run["best_state"],
                        method=run["method"], seed=run["seed"], updates=run["completed_updates"],
                        selected_checkpoint=run["best_step"], target_identity=data["target_identity"]),
                   output / f"{name}.pt")
        write_json(output / f"{name}.json", receipt)
        receipts.append(receipt)
    summary = dict(status="FAILED" if failure else "MODELS_FROZEN_BEFORE_FRESH_TEST",
                   stop_reason=stop_reason, failure=failure, common_updates=steps,
                   methods=list(methods), seeds=list(seeds), dataset=str(Path(dataset).resolve()),
                   target_identity=data["target_identity"], runs=receipts,
                   wall_budget_seconds=budget_seconds, wall_seconds=time.perf_counter() - start_wall,
                   cpu_seconds=time.process_time() - start_cpu, material_frequency_evaluations=0,
                   gpu_seconds=0, final_test_access=False,
                   actual_updates_by_run={f"{r['method']}-{r['seed']}": r["completed_updates"] for r in runs})
    write_json(output / "TRAINING_SUMMARY.json", summary)
    return summary


def load_model(path, *, expected_target=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    from .density import BoundedMDN
    configuration = saved["configuration"]
    if expected_target is not None and configuration["target_identity"] != expected_target:
        raise ValueError("density and reference target_identity mismatch")
    model = BoundedMDN(**configuration)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("p2/results/models"))
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    args = parser.parse_args()
    result = train_job(args.dataset, args.output, max_steps=args.max_steps,
                       budget_seconds=args.wall_seconds, checkpoint_every=args.checkpoint_every)
    print(json.dumps({key: result[key] for key in
                      ("status", "common_updates", "wall_seconds", "cpu_seconds", "material_frequency_evaluations")}))
    if result["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
