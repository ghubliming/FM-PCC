#!/usr/bin/env python3
"""VF-only ODE integrator benchmark for FM-PCC Flow Matching v3.

This script benchmarks ODE integration behavior inside the FM vector-field
sampling path only (legacy_euler vs torchdiffeq methods).
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import yaml

import flow_matcher_v3_ode_selectable.utils as utils
from flow_matcher_v3_ode_selectable.sampling.policies import Policy


class Parser(utils.Parser):
    dataset: str = "avoiding-d3il"
    config: str = "config.avoiding-d3il"


def parse_solver_specs(specs: str) -> List[Dict[str, object]]:
    """Parse --solver-spec entries.

    Format:
    - backend:method
    - backend:method:rtol:atol:step_size
    Use "none" for optional values.
    """
    parsed: List[Dict[str, object]] = []
    entries = [e.strip() for e in specs.split(",") if e.strip()]
    for entry in entries:
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) not in (2, 5):
            raise ValueError(
                f"Invalid solver spec '{entry}'. Expected 'backend:method' "
                "or 'backend:method:rtol:atol:step_size'."
            )

        backend = parts[0]
        method = parts[1]
        rtol = None
        atol = None
        step_size = None

        if len(parts) == 5:
            rtol = None if parts[2].lower() == "none" else float(parts[2])
            atol = None if parts[3].lower() == "none" else float(parts[3])
            step_size = None if parts[4].lower() == "none" else float(parts[4])

        parsed.append(
            {
                "backend": backend,
                "method": method,
                "rtol": rtol,
                "atol": atol,
                "step_size": step_size,
                "label": f"{backend}:{method}",
            }
        )
    return parsed


def aggregate_vf_only_metrics(trials: List[Dict[str, float]]) -> Dict[str, float]:
    if not trials:
        return {}

    inference_ms = np.array([t["avg_inference_ms"] for t in trials], dtype=np.float32)
    smoothness = np.array([t["traj_smoothness"] for t in trials], dtype=np.float32)
    goal_dist = np.array([t["final_goal_dist"] for t in trials], dtype=np.float32)
    final_xy_std = np.array([t["batch_final_xy_std"] for t in trials], dtype=np.float32)

    return {
        "avg_inference_ms": float(inference_ms.mean()),
        "avg_traj_smoothness": float(smoothness.mean()),
        "avg_final_goal_dist": float(goal_dist.mean()),
        "avg_batch_final_xy_std": float(final_xy_std.mean()),
        "n_trials": int(len(trials)),
    }


def write_summary_csv(summary_rows: List[Dict[str, object]], csv_path: str) -> None:
    if not summary_rows:
        return

    fieldnames = list(summary_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def render_optional_plots(summary_rows: List[Dict[str, object]], out_dir: str, trials_by_solver: Dict[str, List[Dict[str, float]]]) -> None:
    """Render optional benchmark plots when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] Skipping plots because matplotlib is unavailable: {exc}")
        return

    if not summary_rows:
        print("[plot] No summary rows available; skipping plot generation.")
        return

    labels = [f"{row['backend']}:{row['method']}" for row in summary_rows]
    x = np.arange(len(labels))
    inference_ms = [float(row["avg_inference_ms"]) for row in summary_rows]
    final_goal_dist = [float(row["avg_final_goal_dist"]) for row in summary_rows]
    smoothness = [float(row["avg_traj_smoothness"]) for row in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(x, inference_ms)
    axes[0].set_title("Average Inference Time (ms)")
    axes[1].bar(x, final_goal_dist)
    axes[1].set_title("Average Final Goal Distance")
    axes[2].bar(x, smoothness)
    axes[2].set_title("Average Trajectory Smoothness")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "benchmark_summary_plots.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.scatter(inference_ms, final_goal_dist)
    for i, label in enumerate(labels):
        ax2.annotate(label, (inference_ms[i], final_goal_dist[i]), xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("Average Inference Time (ms)")
    ax2.set_ylabel("Average Final Goal Distance")
    ax2.set_title("VF ODE Trade-off: Speed vs Goal Proximity")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "benchmark_tradeoff_scatter.png"), dpi=160, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    for solver_label, trials in trials_by_solver.items():
        if not trials:
            continue
        ms = [float(t["avg_inference_ms"]) for t in trials]
        ax3.plot(ms, label=solver_label)
    ax3.set_xlabel("Trial Index")
    ax3.set_ylabel("Average Inference Time (ms)")
    ax3.set_title("Per-Trial Inference Time by Solver (VF-only)")
    ax3.legend(loc="best")
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "benchmark_inference_per_trial.png"), dpi=160, bbox_inches="tight")
    plt.close(fig3)

    print("[plot] Saved benchmark plots:")
    print(f"  - {os.path.join(out_dir, 'benchmark_summary_plots.png')}")
    print(f"  - {os.path.join(out_dir, 'benchmark_tradeoff_scatter.png')}")
    print(f"  - {os.path.join(out_dir, 'benchmark_inference_per_trial.png')}")


def resolve_model_seed(cli_seed: int, proj_cfg: Dict[str, object]) -> int:
    """Resolve model seed from CLI or projection config."""
    if cli_seed is not None:
        return int(cli_seed)

    cfg_seeds = proj_cfg.get("seeds")
    if isinstance(cfg_seeds, list) and len(cfg_seeds) > 0:
        return int(cfg_seeds[0])

    # Final fallback: deterministic default if config does not define seeds.
    return 0


def sample_condition_observations(dataset, n_trials: int, seed: int) -> List[np.ndarray]:
    """Sample real condition observations from dataset episodes (no env interaction)."""
    rng = np.random.default_rng(seed)
    obs_pool = dataset.fields.observations
    path_lengths = dataset.path_lengths
    candidates: List[np.ndarray] = []
    n_episodes = int(dataset.n_episodes)

    for _ in range(n_trials):
        ep = int(rng.integers(0, n_episodes))
        path_len = max(int(path_lengths[ep]), 1)
        t = int(rng.integers(0, path_len))
        candidates.append(np.array(obs_pool[ep, t], copy=True))

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure VF ODE integrator benchmark for FM-v3")
    parser.add_argument("--seed", type=int, default=None, help="Model seed/checkpoint seed. If omitted, uses first seed from config/projection_eval.yaml.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of condition observations per solver option.")
    parser.add_argument("--horizon", type=int, default=None, help="Override rollout horizon; defaults to plan config.")
    parser.add_argument("--flow-steps", type=int, default=10, help="Flow rollout steps for all solver options.")
    parser.add_argument("--vf-batch-size", type=int, default=16, help="Batch size for vf_only trajectory sampling metrics.")
    parser.add_argument(
        "--solver-spec",
        type=str,
        default="legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint",
        help="Comma-separated solver specs: backend:method or backend:method:rtol:atol:step_size",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device from config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for logs and metrics.")
    parser.add_argument("--plot", action="store_true", help="Enable optional plot generation across tested solver options.")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(project_root, "config", "projection_eval.yaml"), "r", encoding="utf-8") as f:
        proj_cfg = yaml.safe_load(f)
    model_seed = resolve_model_seed(args.seed, proj_cfg)

    parser_args = Parser().parse_args(experiment="plan_fm_v3_ode_selectable", seed=model_seed)
    if args.device is not None:
        parser_args.device = args.device

    fm_experiment = utils.load_diffusion(
        parser_args.loadbase,
        parser_args.dataset,
        parser_args.diffusion_loadpath,
        str(parser_args.seed),
        epoch=parser_args.diffusion_epoch,
        device=parser_args.device,
    )
    fm_model = fm_experiment.diffusion
    dataset = fm_experiment.dataset

    horizon = int(args.horizon if args.horizon is not None else parser_args.horizon)
    obs_dim = int(dataset.observation_dim)
    goal_dim = int(fm_model.goal_dim)
    if goal_dim > 0 and goal_dim < obs_dim:
        xy_indices = [max(0, obs_dim - goal_dim - 2), max(0, obs_dim - goal_dim - 1)]
        goal_xy_indices = [obs_dim - goal_dim, obs_dim - goal_dim + 1]
    else:
        xy_indices = [0, 1] if obs_dim >= 2 else [0, 0]
        goal_xy_indices = xy_indices

    conditions_pool = sample_condition_observations(dataset, int(args.n_trials), model_seed)

    solver_options = parse_solver_specs(args.solver_spec)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = os.path.join(os.path.dirname(__file__), "benchmark_outputs", f"{timestamp}_seed{model_seed}_vf_only")
    out_dir = args.output_dir if args.output_dir is not None else default_out
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "timestamp": timestamp,
        "experiment": str(parser_args.dataset),
        "seed": int(model_seed),
        "n_trials": int(args.n_trials),
        "vf_batch_size": int(args.vf_batch_size),
        "horizon": horizon,
        "flow_steps_v3": int(args.flow_steps),
        "ode_inference_steps_v3": int(args.flow_steps),
        "benchmark_mode": "vf_only",
        "device": str(parser_args.device),
        "solver_options": solver_options,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 90)
    print("FM-v3 VF ODE Integrator Benchmark")
    print(f"mode=vf_only | seed={model_seed} | trials={args.n_trials} | horizon={horizon}")
    print(f"outputs={out_dir}")
    print("=" * 90)

    summary_rows: List[Dict[str, object]] = []
    trials_by_solver: Dict[str, List[Dict[str, float]]] = {}

    for solver_idx, solver in enumerate(solver_options):
        backend = str(solver["backend"])
        method = str(solver["method"])

        fm_model.flow_steps_v3 = int(args.flow_steps)
        fm_model.ode_inference_steps_v3 = int(args.flow_steps)
        fm_model.ode_solver_backend_v3 = backend
        fm_model.ode_solver_method_v3 = method
        fm_model.ode_solver_rtol_v3 = solver["rtol"]
        fm_model.ode_solver_atol_v3 = solver["atol"]
        fm_model.ode_solver_step_size_v3 = solver["step_size"]

        policy = Policy(
            model=fm_model,
            normalizer=dataset.normalizer,
            preprocess_fns=parser_args.preprocess_fns,
            test_ret=parser_args.test_ret,
            projector=None,
            trajectory_selection="random",
        )

        print("-" * 90)
        print(
            f"[{solver_idx + 1}/{len(solver_options)}] {backend}:{method} "
            f"rtol={solver['rtol']} atol={solver['atol']} step_size={solver['step_size']}"
        )

        trials: List[Dict[str, float]] = []
        for trial in range(args.n_trials):
            torch.manual_seed(trial)
            np.random.seed(trial)

            obs = conditions_pool[trial]
            t0 = time.time()
            _action, samples = policy(
                conditions={0: obs},
                batch_size=int(args.vf_batch_size),
                horizon=horizon,
                disable_projection=True,
            )
            elapsed_ms = (time.time() - t0) * 1000.0

            sample_obs = samples.observations  # [batch, horizon, obs_dim]
            xy = sample_obs[:, :, xy_indices]
            goal_xy = sample_obs[:, :, goal_xy_indices]

            if xy.shape[1] > 1:
                traj_smoothness = float(np.mean(np.linalg.norm(xy[:, 1:, :] - xy[:, :-1, :], axis=-1)))
            else:
                traj_smoothness = 0.0
            final_goal_dist = float(np.mean(np.linalg.norm(xy[:, -1, :] - goal_xy[:, -1, :], axis=-1)))
            batch_final_xy_std = float(np.mean(np.std(xy[:, -1, :], axis=0)))

            trial_record = {
                "trial": int(trial),
                "avg_inference_ms": float(elapsed_ms),
                "traj_smoothness": traj_smoothness,
                "final_goal_dist": final_goal_dist,
                "batch_final_xy_std": batch_final_xy_std,
            }
            trials.append(trial_record)

            print(
                f"  trial={trial:03d} inf_ms={trial_record['avg_inference_ms']:.2f} "
                f"smooth={trial_record['traj_smoothness']:.4f} "
                f"goal_dist={trial_record['final_goal_dist']:.4f}"
            )

        solver_summary = aggregate_vf_only_metrics(trials)
        solver_summary.update(
            {
                "backend": backend,
                "method": method,
                "rtol": solver["rtol"],
                "atol": solver["atol"],
                "step_size": solver["step_size"],
                "flow_steps_v3": int(args.flow_steps),
                "ode_inference_steps_v3": int(args.flow_steps),
            }
        )
        summary_rows.append(solver_summary)

        safe_label = f"{backend}_{method}".replace("/", "_")
        with open(os.path.join(out_dir, f"trials_{safe_label}.json"), "w", encoding="utf-8") as f:
            json.dump(trials, f, indent=2)
        trials_by_solver[f"{backend}:{method}"] = trials

        print(
            f"  summary: avg_inf_ms={solver_summary['avg_inference_ms']:.2f} "
            f"avg_goal_dist={solver_summary['avg_final_goal_dist']:.4f} "
            f"avg_smooth={solver_summary['avg_traj_smoothness']:.4f}"
        )

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    csv_path = os.path.join(out_dir, "summary.csv")
    write_summary_csv(summary_rows, csv_path)

    if args.plot:
        render_optional_plots(summary_rows, out_dir, trials_by_solver)

    print("=" * 90)
    print("Benchmark completed")
    print(f"Saved summary JSON: {os.path.join(out_dir, 'summary.json')}")
    print(f"Saved summary CSV : {csv_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
