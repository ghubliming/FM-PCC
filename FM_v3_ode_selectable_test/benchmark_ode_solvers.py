#!/usr/bin/env python3
"""Standalone ODE-solver benchmark on a synthetic vector field.

Compares wall-clock inference time of different ODE backends/methods
on *exactly the same* deterministic dynamics so the only variable
is the solver itself.

No FM model, no dataset, no env rollout — pure numerics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# 1. Synthetic vector field
# ---------------------------------------------------------------------------

def spiral_vf(x: np.ndarray, alpha: float, omega: float, beta: float) -> np.ndarray:
    """Stable spiral with nonlinear damping.

    Parameters
    ----------
    x : ndarray, shape [batch, dim]   (dim must be even)
    alpha, omega, beta : scalar VF parameters
    """
    b, d = x.shape
    assert d >= 2 and d % 2 == 0, f"dim must be positive even, got {d}"

    dx = np.empty_like(x)
    for k in range(0, d, 2):
        u, v = x[:, k], x[:, k + 1]
        r2 = u * u + v * v
        damp = alpha + beta * r2
        dx[:, k]     = -damp * u - omega * v
        dx[:, k + 1] =  omega * u - damp * v
    return dx


# Default VF parameters (shared everywhere so every solver sees the same field)
VF_ALPHA, VF_OMEGA, VF_BETA = 0.35, 1.25, 0.12

def _default_rhs(x: np.ndarray) -> np.ndarray:
    return spiral_vf(x, VF_ALPHA, VF_OMEGA, VF_BETA)


# ---------------------------------------------------------------------------
# 2. Integrators
# ---------------------------------------------------------------------------

def euler_integrate(
    x0: np.ndarray, rhs: Callable, n_steps: int, t0: float, t1: float,
) -> np.ndarray:
    """Simple forward-Euler on numpy arrays."""
    dt = (t1 - t0) / n_steps
    x = x0.copy()
    for _ in range(n_steps):
        x = x + dt * rhs(x)
    return x


def torchdiffeq_integrate(
    x0: np.ndarray,
    method: str,
    n_steps: int,
    t0: float,
    t1: float,
    rtol: float,
    atol: float,
) -> np.ndarray:
    """Integrate with torchdiffeq, return numpy result."""
    import torch
    from torchdiffeq import odeint

    device = torch.device("cpu")
    x0_t = torch.from_numpy(x0).float().to(device)

    def rhs_torch(_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        dx_np = _default_rhs(x_t.detach().cpu().numpy())
        return torch.from_numpy(dx_np).to(dtype=x_t.dtype, device=x_t.device)

    # Fixed-step methods need an explicit time grid; adaptive ones need tol
    FIXED = {"euler", "midpoint", "rk4", "heun2", "heun3",
             "explicit_adams", "implicit_adams", "fixed_adams"}

    if method in FIXED:
        ts = torch.linspace(t0, t1, n_steps + 1, device=device)
        traj = odeint(rhs_torch, x0_t, ts, method=method)
    else:
        ts = torch.tensor([t0, t1], dtype=torch.float32, device=device)
        traj = odeint(rhs_torch, x0_t, ts, method=method, rtol=rtol, atol=atol)

    return traj[-1].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Solver-spec parsing
# ---------------------------------------------------------------------------

def parse_solvers(spec: str) -> List[Dict[str, str]]:
    """Parse comma-separated solver entries.

    Accepted formats
    ----------------
    legacy_euler          ->  backend=legacy_euler  method=euler
    legacy_euler:euler    ->  backend=legacy_euler  method=euler   (also OK)
    torchdiffeq:dopri5    ->  backend=torchdiffeq   method=dopri5
    """
    solvers: List[Dict[str, str]] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" in entry:
            backend, method = entry.split(":", 1)
        else:
            backend, method = entry, entry  # legacy_euler -> legacy_euler/legacy_euler
        backend, method = backend.strip(), method.strip()
        # Normalise the legacy shortcut
        if backend == "legacy_euler":
            method = "euler"
        solvers.append({"backend": backend, "method": method})
    if not solvers:
        raise ValueError("--solver-spec produced an empty list")
    return solvers


# ---------------------------------------------------------------------------
# 4. Statistics helper
# ---------------------------------------------------------------------------

def compute_stats(times_ms: List[float]) -> Dict[str, float]:
    a = np.asarray(times_ms, dtype=np.float64)
    return {
        "avg_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


# ---------------------------------------------------------------------------
# 5. Plotting (optional, only when --plot is passed)
# ---------------------------------------------------------------------------

def make_plots(summary: List[Dict[str, Any]], out_dir: str) -> None:
    """Generate bar charts for each timing metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['backend']}:{r['method']}" for r in summary]
    metrics = ["avg_ms", "std_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    # -- Individual bar charts --
    for metric, color in zip(metrics, colors):
        vals = [r[metric] for r in summary]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
        bars = ax.bar(labels, vals, color=color, edgecolor="white", linewidth=0.6)
        ax.set_ylabel(metric)
        ax.set_title(f"ODE Solver Benchmark — {metric}")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"plot_{metric}.png"), dpi=150)
        plt.close(fig)

    # -- Combined overview --
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, metric, color in zip(axes.flat, metrics, colors):
        vals = [r[metric] for r in summary]
        bars = ax.bar(labels, vals, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(metric, fontsize=10)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.suptitle("ODE Solver Benchmark — All Metrics", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_overview.png"), dpi=150)
    plt.close(fig)
    print(f"  Plots saved to {out_dir}/plot_*.png")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark ODE solvers on a synthetic vector field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-trials",    type=int,   default=100,  help="Repetitions per solver")
    ap.add_argument("--batch-size",  type=int,   default=128,  help="Batch of initial states")
    ap.add_argument("--state-dim",   type=int,   default=8,    help="State dimension (must be even)")
    ap.add_argument("--t0",          type=float, default=0.0)
    ap.add_argument("--t1",          type=float, default=1.0)
    ap.add_argument("--steps",       type=int,   default=20,   help="Steps for fixed-step methods")
    ap.add_argument("--rtol",        type=float, default=1e-5, help="Rel tol for adaptive methods")
    ap.add_argument("--atol",        type=float, default=1e-6, help="Abs tol for adaptive methods")
    ap.add_argument("--solver-spec", type=str,
                    default="legacy_euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint",
                    help="Comma-separated list: legacy_euler, torchdiffeq:<method>")
    ap.add_argument("--output-dir",  type=str,   default=None)
    ap.add_argument("--plot",        action="store_true", help="Generate bar-chart PNGs")
    args = ap.parse_args()

    # ---- Validation ----
    assert args.state_dim >= 2 and args.state_dim % 2 == 0, "--state-dim must be even ≥ 2"
    assert args.steps >= 1, "--steps must be ≥ 1"
    assert args.t1 > args.t0, "--t1 must be > --t0"

    np.random.seed(args.seed)
    solvers = parse_solvers(args.solver_spec)

    # ---- Output directory ----
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__),
                               "benchmark_outputs", f"{ts}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Save run metadata ----
    meta = {
        "seed": args.seed,
        "n_trials": args.n_trials,
        "batch_size": args.batch_size,
        "state_dim": args.state_dim,
        "t0": args.t0, "t1": args.t1,
        "steps": args.steps,
        "rtol": args.rtol, "atol": args.atol,
        "solvers": solvers,
    }
    _dump_json(os.path.join(out_dir, "run_meta.json"), meta)

    hdr = (f"Synthetic VF ODE Benchmark | trials={args.n_trials} "
           f"batch={args.batch_size} dim={args.state_dim} steps={args.steps}")
    print("=" * len(hdr))
    print(hdr)
    print(f"output → {out_dir}")
    print("=" * len(hdr))

    # ---- Run each solver ----
    all_summary: List[Dict[str, Any]] = []

    for i, sol in enumerate(solvers, 1):
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n[{i}/{len(solvers)}] {tag}")

        trial_times: List[float] = []

        for t in range(args.n_trials):
            x0 = np.random.randn(args.batch_size, args.state_dim).astype(np.float32)
            t_start = time.perf_counter()

            if backend == "legacy_euler":
                euler_integrate(x0, _default_rhs, args.steps, args.t0, args.t1)
            elif backend == "torchdiffeq":
                torchdiffeq_integrate(x0, method, args.steps, args.t0, args.t1,
                                      args.rtol, args.atol)
            else:
                raise ValueError(f"Unknown backend '{backend}'")

            ms = (time.perf_counter() - t_start) * 1000.0
            trial_times.append(ms)
            print(f"  trial {t:03d}  {ms:8.3f} ms")

        stats = compute_stats(trial_times)
        row = {"backend": backend, "method": method, "n_trials": args.n_trials, **stats}
        all_summary.append(row)

        # Per-solver trial dump
        _dump_json(
            os.path.join(out_dir, f"trials_{backend}_{method}.json"),
            [{"trial": k, "ms": v} for k, v in enumerate(trial_times)],
        )
        print(f"  → avg={stats['avg_ms']:.3f}  std={stats['std_ms']:.3f}  "
              f"p50={stats['p50_ms']:.3f}  p95={stats['p95_ms']:.3f}")

    # ---- Write summary ----
    _dump_json(os.path.join(out_dir, "summary.json"), all_summary)
    _dump_csv(os.path.join(out_dir, "summary.csv"), all_summary)

    # ---- Optional plots ----
    if args.plot:
        make_plots(all_summary, out_dir)

    print(f"\nDone. Results in {out_dir}")


# ---------------------------------------------------------------------------
# 7. IO helpers
# ---------------------------------------------------------------------------

def _dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _dump_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
