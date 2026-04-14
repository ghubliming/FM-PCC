#!/usr/bin/env python3
from __future__ import annotations

"""Standalone ODE method benchmark in a synthetic VF test environment.

This script is intentionally independent from FM model loading and dataset code.
It benchmarks ODE solvers on a deterministic vector field only.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Callable, Dict, List


def parse_solver_specs(specs: str) -> List[Dict[str, str]]:
    """Parse solver list.

    Supported entry formats:
    - legacy_euler
    - backend:method (example: torchdiffeq:dopri5)
    """
    out: List[Dict[str, str]] = []
    entries = [e.strip() for e in specs.split(",") if e.strip()]
    if not entries:
        raise ValueError("--solver-spec must not be empty")

    for entry in entries:
        if ":" not in entry:
            if entry != "legacy_euler":
                raise ValueError(f"Unknown solver entry '{entry}'")
            out.append({"backend": "legacy_euler", "method": "euler"})
            continue

        parts = [p.strip() for p in entry.split(":")]
        if len(parts) != 2:
            raise ValueError(f"Invalid solver entry '{entry}'. Use 'backend:method'.")
        out.append({"backend": parts[0], "method": parts[1]})

    return out


def vf_rhs(x: np.ndarray, alpha: float, omega: float, beta: float) -> np.ndarray:
    """Synthetic vector field: stable spiral + nonlinear damping.

    x shape: [batch, dim], dim must be even (paired coordinates).
    """
    batch, dim = x.shape
    if dim % 2 != 0:
        raise ValueError("state dimension must be even")

    y = np.zeros_like(x)
    for i in range(0, dim, 2):
        xi = x[:, i]
        yi = x[:, i + 1]
        r2 = xi * xi + yi * yi
        damp = alpha + beta * r2
        y[:, i] = -damp * xi - omega * yi
        y[:, i + 1] = omega * xi - damp * yi

    return y


def integrate_legacy_euler(
    x0: np.ndarray,
    n_steps: int,
    t0: float,
    t1: float,
    rhs: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    dt = (t1 - t0) / float(n_steps)
    x = x0.copy()
    for _ in range(n_steps):
        x = x + dt * rhs(x)
    return x


def integrate_torchdiffeq(
    x0: np.ndarray,
    method: str,
    t0: float,
    t1: float,
    n_steps: int,
    rtol: float,
    atol: float,
) -> np.ndarray:
    try:
        import torch
        from torchdiffeq import odeint
    except Exception as exc:
        raise RuntimeError(
            "torchdiffeq backend requested but torch/torchdiffeq is unavailable"
        ) from exc

    device = torch.device("cpu")
    x0_t = torch.tensor(x0, dtype=torch.float32, device=device)

    def rhs_t(_t, x_t):
        x_np = x_t.detach().cpu().numpy()
        dx_np = vf_rhs(x_np, alpha=0.35, omega=1.25, beta=0.12)
        return torch.tensor(dx_np, dtype=x_t.dtype, device=x_t.device)

    if method in {"euler", "midpoint", "rk4", "heun2", "heun3", "explicit_adams", "implicit_adams", "fixed_adams"}:
        t_eval = torch.linspace(float(t0), float(t1), int(n_steps) + 1, device=device)
        traj = odeint(rhs_t, x0_t, t_eval, method=method)
    else:
        t_eval = torch.tensor([float(t0), float(t1)], dtype=torch.float32, device=device)
        traj = odeint(rhs_t, x0_t, t_eval, method=method, rtol=float(rtol), atol=float(atol))

    x1 = traj[-1].detach().cpu().numpy()
    return x1


def summarize(values: List[float]) -> Dict[str, float]:
    import numpy as np

    arr = np.array(values, dtype=np.float32)
    return {
        "avg_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main() -> None:
    import numpy as np

    parser = argparse.ArgumentParser(description="Standalone ODE benchmark in synthetic VF env")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=8, help="Must be even")
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20, help="Step count for fixed-step methods")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--solver-spec",
        type=str,
        default="legacy_euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint",
        help="Comma list of solver entries",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.state_dim <= 0 or args.state_dim % 2 != 0:
        raise ValueError("--state-dim must be a positive even integer")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.t1 <= args.t0:
        raise ValueError("--t1 must be greater than --t0")

    np.random.seed(int(args.seed))
    solvers = parse_solver_specs(args.solver_spec)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = os.path.join(
        os.path.dirname(__file__),
        "benchmark_outputs",
        f"{timestamp}_seed{args.seed}_synthetic_vf_ode",
    )
    out_dir = args.output_dir or default_out
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "seed": int(args.seed),
        "n_trials": int(args.n_trials),
        "batch_size": int(args.batch_size),
        "state_dim": int(args.state_dim),
        "t0": float(args.t0),
        "t1": float(args.t1),
        "steps": int(args.steps),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "solver_options": solvers,
        "env": "synthetic_vf_only",
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 88)
    print("Standalone Synthetic VF ODE Benchmark")
    print(f"trials={args.n_trials} batch={args.batch_size} dim={args.state_dim} steps={args.steps}")
    print(f"output={out_dir}")
    print("=" * 88)

    summary_rows: List[Dict[str, object]] = []

    for idx, solver in enumerate(solvers):
        backend = solver["backend"]
        method = solver["method"]
        print("-" * 88)
        print(f"[{idx + 1}/{len(solvers)}] backend={backend} method={method}")

        trial_times: List[float] = []
        trial_rows: List[Dict[str, float]] = []

        for trial in range(int(args.n_trials)):
            x0 = np.random.randn(int(args.batch_size), int(args.state_dim)).astype(np.float32)
            start = time.perf_counter()

            if backend == "legacy_euler":
                _ = integrate_legacy_euler(
                    x0=x0,
                    n_steps=int(args.steps),
                    t0=float(args.t0),
                    t1=float(args.t1),
                    rhs=lambda x: vf_rhs(x, alpha=0.35, omega=1.25, beta=0.12),
                )
            elif backend == "torchdiffeq":
                _ = integrate_torchdiffeq(
                    x0=x0,
                    method=method,
                    t0=float(args.t0),
                    t1=float(args.t1),
                    n_steps=int(args.steps),
                    rtol=float(args.rtol),
                    atol=float(args.atol),
                )
            else:
                raise ValueError(f"Unsupported backend '{backend}'")

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            trial_times.append(elapsed_ms)
            trial_rows.append({"trial": float(trial), "inference_ms": float(elapsed_ms)})
            print(f"  trial={trial:03d} inf_ms={elapsed_ms:.3f}")

        stats = summarize(trial_times)
        row: Dict[str, object] = {
            "backend": backend,
            "method": method,
            "n_trials": int(args.n_trials),
            **stats,
        }
        summary_rows.append(row)

        per_trial_path = os.path.join(out_dir, f"trials_{backend}_{method}.json".replace("/", "_"))
        with open(per_trial_path, "w", encoding="utf-8") as f:
            json.dump(trial_rows, f, indent=2)

        print(
            f"  summary: avg={row['avg_ms']:.3f}ms std={row['std_ms']:.3f} "
            f"p95={row['p95_ms']:.3f}"
        )

    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    summary_csv = os.path.join(out_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 88)
    print("Benchmark completed")
    print(f"summary json: {summary_json}")
    print(f"summary csv : {summary_csv}")
    print("=" * 88)


if __name__ == "__main__":
    main()
