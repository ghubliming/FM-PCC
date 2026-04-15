#!/usr/bin/env python3
"""Standalone ODE-solver benchmark (v2) on synthetic vector fields & actual U-Net Models.

V2 Differences:
- Supports loading a real trained Flow Matcher U-Net from /logs/ via `--vf-mode flow_matcher`
- Adds `--integration-mode` (continuous vs chunked) to faithfully replicate eval pipeline overheads
- Uses properly scoped `torch.no_grad()` surrounding torchdiffeq calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch

# Dynamically add the project root to sys.path if not present so we can import flow_matcher modules
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ---------------------------------------------------------------------------
# 1. Synthetic vector field (Spiral)
# ---------------------------------------------------------------------------

def spiral_vf(x: np.ndarray, alpha: float, omega: float, beta: float) -> np.ndarray:
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

VF_ALPHA, VF_OMEGA, VF_BETA = 0.35, 1.25, 0.12

def _default_rhs(x: np.ndarray) -> np.ndarray:
    return spiral_vf(x, VF_ALPHA, VF_OMEGA, VF_BETA)


# ---------------------------------------------------------------------------
# 1b. Real Flow Matcher Vector Field
# ---------------------------------------------------------------------------
class UNetVF:
    """Wrapper to make the actual Flow_matcher_U_Net_v2 callable with same signature."""
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        
        # Determine shapes from the loaded model fallback
        self.transition_dim = getattr(model, 'transition_dim', 4)  
        self.observation_dim = getattr(model, 'observation_dim', self.transition_dim)
        self.cond = None
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def set_dummy_cond(self, batch_size: int, horizon: int):
        """Creates dummy observations to satisfy the model's condition requirement."""
        dummy_obs = torch.zeros(batch_size, horizon, self.observation_dim, device=self.device)
        self.cond = {0: dummy_obs}

    def __call_torch__(self, t_scalar: torch.Tensor, x_t: torch.Tensor):
        """Evaluate the U-Net VF: torch tensor in → torch tensor out."""
        with torch.no_grad():
            batch = x_t.shape[0]
            # U-Net _predict_velocity expects time to be a batch tensor
            t_batch = torch.ones(batch, device=self.device, dtype=torch.float32) * t_scalar
            return self.model._predict_velocity(x_t, self.cond, t_batch, returns=None)

# ---------------------------------------------------------------------------
# Global VF dispatcher — set by main() before any integration
# ---------------------------------------------------------------------------
_active_rhs: Callable = _default_rhs
_active_unet_vf: UNetVF | None = None


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


def euler_integrate_torch(
    x0: np.ndarray, unet_vf: UNetVF, n_steps: int, t0: float, t1: float,
) -> np.ndarray:
    """Forward-Euler using PyTorch tensors directly."""
    import torch
    with torch.no_grad():
        dt = (t1 - t0) / n_steps
        x = torch.from_numpy(x0).float().to(unet_vf.device)
        t_val = torch.tensor(t0, device=unet_vf.device)
        for _ in range(n_steps):
            x = x + dt * unet_vf.__call_torch__(t_val, x)
            t_val = t_val + dt
        return x.cpu().numpy()


def torchdiffeq_integrate_v2(
    x0: np.ndarray,
    method: str,
    n_steps: int,
    t0: float,
    t1: float,
    rtol: float,
    atol: float,
    device: str = "cpu",
    integration_mode: str = "chunked"
) -> np.ndarray:
    """Integrate with torchdiffeq, chunked or continuous."""
    import torch
    from torchdiffeq import odeint

    dev = torch.device(device)
    x0_t = torch.from_numpy(x0).float().to(dev)

    if _active_unet_vf is not None:
        def rhs_torch(_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            return _active_unet_vf.__call_torch__(_t, x_t)
    else:
        def rhs_torch(_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            dx_np = _active_rhs(x_t.detach().cpu().numpy())
            return torch.from_numpy(dx_np).to(dtype=x_t.dtype, device=x_t.device)

    FIXED = {"euler", "midpoint", "rk4", "heun2", "heun3",
             "explicit_adams", "implicit_adams", "fixed_adams"}

    x = x0_t
    with torch.no_grad():  # Crucial V2 fix: do not track gradient history!
        if integration_mode == "continuous":
            if method in FIXED:
                ts = torch.linspace(t0, t1, n_steps + 1, device=dev)
                traj = odeint(rhs_torch, x, ts, method=method)
            else:
                ts = torch.tensor([t0, t1], dtype=torch.float32, device=dev)
                traj = odeint(rhs_torch, x, ts, method=method, rtol=rtol, atol=atol)
            x_out = traj[-1]
            
        elif integration_mode == "chunked":
            dt = (t1 - t0) / n_steps
            for i in range(n_steps):
                chunk_t0 = t0 + i * dt
                chunk_t1 = chunk_t0 + dt
                t_span = torch.tensor([chunk_t0, chunk_t1], dtype=torch.float32, device=dev)
                kwargs = {"method": method}
                if method not in FIXED:
                    kwargs["rtol"] = rtol
                    kwargs["atol"] = atol
                traj = odeint(rhs_torch, x, t_span, **kwargs)
                x = traj[-1]
            x_out = x
            
    return x_out.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Solver-spec parsing
# ---------------------------------------------------------------------------

def parse_solvers(spec: str) -> List[Dict[str, str]]:
    solvers: List[Dict[str, str]] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry: continue
        if ":" in entry:
            backend, method = entry.split(":", 1)
        else:
            backend, method = entry, entry
        backend, method = backend.strip(), method.strip()
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
        "avg_ms":   float(a.mean()),
        "std_ms":   float(a.std()),
        "p50_ms":   float(np.percentile(a, 50)),
        "p95_ms":   float(np.percentile(a, 95)),
        "min_ms":   float(a.min()),
        "max_ms":   float(a.max()),
    }


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def make_plots(summary: List[Dict[str, Any]], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['backend']}:{r['method']}" for r in summary]
    metrics = ["avg_ms", "std_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    for metric, color in zip(metrics, colors):
        vals = [r[metric] for r in summary]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
        bars = ax.bar(labels, vals, color=color, edgecolor="white", linewidth=0.6)
        ax.set_ylabel(metric)
        ax.set_title(f"ODE Benchmark V2 — {metric}")
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"plot_{metric}.png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, metric, color in zip(axes.flat, metrics, colors):
        vals = [r[metric] for r in summary]
        bars = ax.bar(labels, vals, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(metric, fontsize=10)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.suptitle("ODE Benchmark V2 — All Metrics", fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "plot_overview.png"), dpi=150)
    plt.close(fig)
    print(f"  Plots saved to {out_dir}/plot_*.png")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-trials",    type=int,   default=50,   help="Repetitions per solver")
    ap.add_argument("--batch-size",  type=int,   default=128,  help="Batch of initial states")
    ap.add_argument("--state-dim",   type=int,   default=8,    help="State dimension for synthetic math")
    ap.add_argument("--t0",          type=float, default=0.0)
    ap.add_argument("--t1",          type=float, default=1.0)
    ap.add_argument("--steps",       type=int,   default=20,   help="Steps for fixed methods / Chunks")
    ap.add_argument("--rtol",        type=float, default=1e-5, help="Rel tol for adaptive")
    ap.add_argument("--atol",        type=float, default=1e-6, help="Abs tol for adaptive")
    ap.add_argument("--solver-spec", type=str,
                    default="legacy_euler,torchdiffeq:dopri5,torchdiffeq:rk4",
                    help="Comma-separated list")
    
    # V2 Specific args
    ap.add_argument("--vf-mode",     type=str,   default="spiral", choices=["spiral", "flow_matcher"])
    ap.add_argument("--loadbase",    type=str,   default="logs", help="Base directory for logs")
    ap.add_argument("--dataset",     type=str,   default="avoiding-d3il", help="Dataset/task name")
    ap.add_argument("--diffusion-loadpath", type=str, default="", help="Experiment path e.g. flow_matching/H8_K20_Dmodels.diffusion.GaussianDiffusion")
    ap.add_argument("--diffusion-seed", type=int, default=0, help="Seed folder inside the diffusion loadpath")
    ap.add_argument("--diffusion-epoch", type=str, default="latest", help="Epoch to load (e.g. 'latest' or '1000')")
    ap.add_argument("--integration-mode", type=str, default="chunked", choices=["continuous", "chunked"])
    ap.add_argument("--horizon",     type=int,   default=128, help="Sequence length for U-Net dummy conditioning")
    
    ap.add_argument("--device",      type=str,   default="cpu", help="Hardware: cpu, cuda")
    ap.add_argument("--output-dir",  type=str,   default=None)
    ap.add_argument("--plot",        action="store_true")
    args = ap.parse_args()

    # ---- Validation ----
    assert args.steps >= 1, "--steps must be ≥ 1"
    assert args.t1 > args.t0, "--t1 must be > --t0"
    
    np.random.seed(args.seed)
    solvers = parse_solvers(args.solver_spec)

    # ---- Set up VF mode (V2 Logic) ----
    global _active_rhs, _active_unet_vf
    model_loaded = False
    
    if args.vf_mode == "flow_matcher":
        if not args.diffusion_loadpath:
            raise ValueError("Must provide --diffusion-loadpath when using --vf-mode flow_matcher")
            
        try:
            from flow_matcher_v3_ode_selectable.utils import serialization as utils_serialization
        except ImportError:
            raise ImportError(f"Could not import utils.serialization. DPCC root must be in path: {_PROJ_ROOT}")

        # Loading the exact same way as eval_flow_matching_v3_ode_selectable.py
        fm_exp = utils_serialization.load_diffusion(
            args.loadbase, 
            args.dataset, 
            args.diffusion_loadpath, 
            str(args.diffusion_seed), 
            epoch=args.diffusion_epoch, 
            device=args.device
        )
        
        unet_vf = UNetVF(fm_exp.diffusion, device=args.device)
        unet_vf.set_dummy_cond(args.batch_size, args.horizon)
            
        _active_unet_vf = unet_vf
        _active_rhs = unet_vf
        model_loaded = True
        
        # Override state dimension for reporting based on real U-Net Transition dim
        args.state_dim = unet_vf.transition_dim * args.horizon 
        vf_label = f"U-Net ({unet_vf.n_params:,} params) [horizon={args.horizon}, transition_dim={unet_vf.transition_dim}]"
        
        print(f"[Info] Loaded Real Model. Auto-detected transition_dim={unet_vf.transition_dim}.")
    else:
        _active_unet_vf = None
        _active_rhs = _default_rhs
        vf_label = "spiral (analytic)"
        assert args.state_dim >= 2 and args.state_dim % 2 == 0, "--state-dim must be even ≥ 2 for spiral"


    # ---- Base Directories ----
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs_v2", f"{ts}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "seed": args.seed,
        "n_trials": args.n_trials,
        "batch_size": args.batch_size,
        "state_dim": args.state_dim,
        "vf_mode": args.vf_mode,
        "integration_mode": args.integration_mode,
        "t0": args.t0, "t1": args.t1,
        "steps": args.steps,
        "rtol": args.rtol, "atol": args.atol,
        "solvers": solvers,
    }
    _dump_json(os.path.join(out_dir, "run_meta.json"), meta)

    hdr = (f"ODE Benchmark V2 | device={args.device} | vf={vf_label} | mode={args.integration_mode} | trials={args.n_trials} \n"
           f"batch={args.batch_size} dim={args.state_dim} chunks/steps={args.steps}")
    print("=" * 60)
    print(hdr)
    print(f"output → {out_dir}")
    print("=" * 60)

    # ---- Tensor Shape Selection ----
    if model_loaded:
        x0_shape = (args.batch_size, args.horizon, _active_unet_vf.transition_dim)
    else:
        x0_shape = (args.batch_size, args.state_dim)

    # ---- Run each solver ----
    all_summary: List[Dict[str, Any]] = []

    for i, sol in enumerate(solvers, 1):
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n[{i}/{len(solvers)}] {tag}")

        trial_times: List[float] = []

        # WARM-UP
        x0_warm = np.random.randn(*x0_shape).astype(np.float32)
        if backend == "legacy_euler":
            if model_loaded:
                _ = euler_integrate_torch(x0_warm, _active_unet_vf, int(args.steps), float(args.t0), float(args.t1))
            else:
                _ = euler_integrate(x0_warm, _active_rhs, int(args.steps), float(args.t0), float(args.t1))
        elif backend == "torchdiffeq":
            _ = torchdiffeq_integrate_v2(x0_warm, method, int(args.steps), float(args.t0), float(args.t1),
                                      float(args.rtol), float(args.atol), device=args.device,
                                      integration_mode=args.integration_mode)

        for trial in range(int(args.n_trials)):
            x0 = np.random.randn(*x0_shape).astype(np.float32)
            t_start = time.perf_counter()

            if backend == "legacy_euler":
                if model_loaded:
                    euler_integrate_torch(x0, _active_unet_vf, args.steps, args.t0, args.t1)
                else:
                    euler_integrate(x0, _active_rhs, args.steps, args.t0, args.t1)
            elif backend == "torchdiffeq":
                torchdiffeq_integrate_v2(x0, method, args.steps, args.t0, args.t1,
                                      args.rtol, args.atol, device=args.device,
                                      integration_mode=args.integration_mode)
            else:
                raise ValueError(f"Unknown backend '{backend}'")

            ms = (time.perf_counter() - t_start) * 1000.0
            trial_times.append(ms)
            print(f"  trial {trial:03d}  {ms:8.3f} ms")

        stats = compute_stats(trial_times)
        row = {"backend": backend, "method": method, "n_trials": args.n_trials, **stats}
        all_summary.append(row)

        _dump_json(os.path.join(out_dir, f"trials_{backend}_{method}.json"),
            [{"trial": k, "ms": v} for k, v in enumerate(trial_times)])
        print(f"  → avg={stats['avg_ms']:.3f}  std={stats['std_ms']:.3f}  "
              f"p50={stats['p50_ms']:.3f}  p95={stats['p95_ms']:.3f}")

    _dump_json(os.path.join(out_dir, "summary.json"), all_summary)
    _dump_csv(os.path.join(out_dir, "summary.csv"), all_summary)

    if args.plot:
        make_plots(all_summary, out_dir)

    print(f"\nDone. Results in {out_dir}")

def _dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _dump_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()
