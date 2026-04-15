#!/usr/bin/env python3
"""Standalone ODE-solver benchmark (v2) on synthetic vector fields & actual U-Net Models.

V2 Differences:
- Retains V1's synthetic testing (`spiral` pure math, and `neural` 1.5M Param MLP).
- ADDS a new feature to load a real trained Flow Matcher U-Net from /logs/ via `--vf-mode flow_matcher`
  which wires into the actual `p_sample_loop` in FMPCC to guarantee exact measuring of 
  production environment's overhead and chunked integration structure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import sys

# --- MINARI MOCK (fixes ModuleNotFoundError on systems without dataset libs) ---
if 'minari' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['minari'] = MagicMock()
# -------------------------------------------------------------------------------

# Dynamically add the project root to sys.path if not present so we can import flow_matcher modules
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ---------------------------------------------------------------------------
# 1a. Synthetic vector field (Spiral Math Baseline)
# ---------------------------------------------------------------------------

def spiral_vf(x: np.ndarray, alpha: float, omega: float, beta: float) -> np.ndarray:
    b, d = x.shape
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
# 1b. Synthetic Neural VF (V1 MLP)
# ---------------------------------------------------------------------------
class NeuralVF:
    """A PyTorch MLP that approximates the computational cost of a real model pass.
    Total ~1.5M parameters.
    """
    def __init__(self, state_dim: int, device: str = "cpu"):
        import torch.nn as nn
        self.device = torch.device(device)
        self.state_dim = state_dim
        embed_dim = 128

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.Mish(), nn.Linear(512, embed_dim),
        ).to(self.device)

        self.trunk = nn.Sequential(
            nn.Linear(state_dim + embed_dim, 512), nn.Mish(),
            nn.Linear(512, 1024), nn.Mish(),
            nn.Linear(1024, 512), nn.Mish(),
            nn.Linear(512, state_dim),
        ).to(self.device)

        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(-np.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / half)
        self._freqs = freqs.to(self.device)

        self.n_params = sum(p.numel() for p in self.time_mlp.parameters()) + \
                        sum(p.numel() for p in self.trunk.parameters())

    def __call_torch__(self, t_scalar, x_t):
        with torch.no_grad():
            batch = x_t.shape[0]
            args = t_scalar * self._freqs
            time_embed = torch.cat([args.sin(), args.cos()], dim=-1)
            time_embed = self.time_mlp(time_embed)
            time_embed = time_embed.unsqueeze(0).expand(batch, -1)
            inp = torch.cat([x_t, time_embed], dim=-1)
            return self.trunk(inp)


_active_rhs: Callable = _default_rhs
_active_neural_vf: NeuralVF | None = None

# ---------------------------------------------------------------------------
# 2. Integrators (Synthetic)
# ---------------------------------------------------------------------------

def euler_integrate(x0: np.ndarray, rhs: Callable, n_steps: int, t0: float, t1: float) -> np.ndarray:
    """Simple forward-Euler on numpy arrays."""
    dt = (t1 - t0) / n_steps
    x = x0.copy()
    for _ in range(n_steps):
        x = x + dt * rhs(x)
    return x

def euler_integrate_torch(x0: np.ndarray, neural_vf: NeuralVF, n_steps: int, t0: float, t1: float) -> np.ndarray:
    """Forward-Euler using PyTorch tensors directly (no numpy bridge)."""
    with torch.no_grad():
        dt = (t1 - t0) / n_steps
        x = torch.from_numpy(x0).float().to(neural_vf.device)
        t_val = torch.tensor(t0, device=neural_vf.device)
        for _ in range(n_steps):
            x = x + dt * neural_vf.__call_torch__(t_val, x)
            t_val = t_val + dt
        return x.cpu().numpy()

def torchdiffeq_integrate_synthetic(
    x0: np.ndarray, method: str, n_steps: int, t0: float, t1: float,
    rtol: float, atol: float, device: str = "cpu",
) -> np.ndarray:
    """Integrate synthetic math or MLP with torchdiffeq."""
    from torchdiffeq import odeint
    dev = torch.device(device)
    x0_t = torch.from_numpy(x0).float().to(dev)

    if _active_neural_vf is not None:
        def rhs_torch(_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            return _active_neural_vf.__call_torch__(_t, x_t)
    else:
        def rhs_torch(_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
            dx_np = _active_rhs(x_t.detach().cpu().numpy())
            return torch.from_numpy(dx_np).to(dtype=x_t.dtype, device=x_t.device)

    FIXED = {"euler", "midpoint", "rk4", "heun2", "heun3", "explicit_adams", "implicit_adams", "fixed_adams"}
    with torch.no_grad():
        if method in FIXED:
            ts = torch.linspace(t0, t1, n_steps + 1, device=dev)
            traj = odeint(rhs_torch, x0_t, ts, method=method)
        else:
            ts = torch.tensor([t0, t1], dtype=torch.float32, device=dev)
            traj = odeint(rhs_torch, x0_t, ts, method=method, rtol=rtol, atol=atol)
            
    return traj[-1].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Solver-spec parsing & Stats Helper
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
    return solvers

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
# 4. Plotting
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
# 5. Main Evaluator
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-trials",    type=int,   default=50,   help="Repetitions per solver")
    ap.add_argument("--batch-size",  type=int,   default=128,  help="Batch of initial states")
    ap.add_argument("--state-dim",   type=int,   default=8,    help="State dimension for synthetic math/MLP")
    ap.add_argument("--t0",          type=float, default=0.0)
    ap.add_argument("--t1",          type=float, default=1.0)
    ap.add_argument("--steps",       type=int,   default=20,   help="Steps for fixed methods / Chunks")
    ap.add_argument("--rtol",        type=float, default=1e-5, help="Rel tol for adaptive")
    ap.add_argument("--atol",        type=float, default=1e-6, help="Abs tol for adaptive")
    ap.add_argument("--solver-spec", type=str, default="legacy_euler,torchdiffeq:dopri5,torchdiffeq:rk4")
    
    # V2 Specific args
    ap.add_argument("--vf-mode",     type=str,   default="spiral", choices=["spiral", "neural", "flow_matcher"])
    ap.add_argument("--loadbase",    type=str,   default="logs", help="Base directory for logs")
    ap.add_argument("--dataset",     type=str,   default="avoiding-d3il", help="Dataset/task name")
    ap.add_argument("--diffusion-loadpath", type=str, default="", help="Experiment path e.g. flow_matching/H8_K20_... ")
    ap.add_argument("--diffusion-seed", type=int, default=0, help="Seed folder inside the diffusion loadpath")
    ap.add_argument("--diffusion-epoch", type=str, default="latest", help="Epoch to load")
    ap.add_argument("--horizon",     type=int,   default=128, help="Sequence length for U-Net cond shape matching")
    
    ap.add_argument("--device",      type=str,   default="cpu", help="Hardware: cpu, cuda")
    ap.add_argument("--output-dir",  type=str,   default=None)
    ap.add_argument("--plot",        action="store_true")
    ap.add_argument("--include-bridge-tax", action="store_true", help="Include NumPy<->Torch conversion in timing (matches eval.py logic)")
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    solvers = parse_solvers(args.solver_spec)

    # ---- Setup Logging ----
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs_v2", f"{ts}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # OPTION A: REAL FLOW MATCHER MODE (Uses FMPCC Native Loader & Loop)
    # =========================================================================
    if args.vf_mode == "flow_matcher":
        if not args.diffusion_loadpath:
            raise ValueError("Must provide --diffusion-loadpath when using --vf-mode flow_matcher")
            
        try:
            from flow_matcher_v3_ode_selectable.utils import serialization as utils_serialization
        except ImportError:
            raise ImportError(f"Could not import utils.serialization. DPCC root must be in path: {_PROJ_ROOT}")

        print("[Info] WIRING INTO REAL FMPCC MODEL & NATIVE p_sample_loop...")
        fm_exp = utils_serialization.load_diffusion(
            args.loadbase, args.dataset, args.diffusion_loadpath, 
            str(args.diffusion_seed), epoch=args.diffusion_epoch, device=args.device
        )
        fm_model = fm_exp.diffusion
        fm_model.eval()
        
        transition_dim = getattr(fm_model, 'transition_dim', 4)
        observation_dim = getattr(fm_model, 'observation_dim', transition_dim)
        
        vf_label = f"U-Net P_Sample_Loop Engine [horizon={args.horizon}, transition_dim={transition_dim}]"
        
        # Setup exactly matching what Policy does
        shape = (args.batch_size, args.horizon, transition_dim)
        dummy_obs = torch.zeros(args.batch_size, observation_dim, device=args.device)
        cond = {0: dummy_obs}

        hdr = f"ODE Benchmark V2 (REAL PIPELINE) | {vf_label} | steps={args.steps}"
        print("=" * 70); print(hdr); print("=" * 70)
        
        all_summary: List[Dict[str, Any]] = []
        for i, sol in enumerate(solvers, 1):
            backend, method = sol["backend"], sol["method"]
            tag = f"{backend}:{method}"
            print(f"\n[{i}/{len(solvers)}] {tag}")
            
            # Wire settings directly into the FMPCC GaussianDiffusion config
            fm_model.ode_solver_backend_v3 = backend
            fm_model.ode_solver_method_v3 = method
            fm_model.flow_steps_v3 = args.steps
            fm_model.ode_solver_rtol_v3 = args.rtol
            fm_model.ode_solver_atol_v3 = args.atol

            # Warm-up (loads caches, JIT, torchdiffeq init)
            with torch.no_grad():
                _ = fm_model.p_sample_loop(shape, cond)
            if "cuda" in args.device: torch.cuda.synchronize()

            trial_times = []
            for trial in range(args.n_trials):
                # If requested, prepare input on CPU to measure transfer cost
                if args.include_bridge_tax:
                    dummy_obs_cpu = dummy_obs.cpu().numpy()

                if "cuda" in args.device: torch.cuda.synchronize()
                t_start = time.perf_counter()
                
                with torch.no_grad():
                    if args.include_bridge_tax:
                        # Re-upload and run
                        cond_repack = {0: torch.from_numpy(dummy_obs_cpu).to(args.device)}
                        res = fm_model.p_sample_loop(shape, cond_repack)
                        # Sync and move back to CPU (matches utils.to_np in eval.py)
                        _ = res.cpu().numpy()
                    else:
                        fm_model.p_sample_loop(shape, cond)
                
                if "cuda" in args.device: torch.cuda.synchronize()
                ms = (time.perf_counter() - t_start) * 1000.0
                trial_times.append(ms)
                print(f"  trial {trial:03d}  {ms:8.3f} ms")

            stats = compute_stats(trial_times)
            row = {"backend": backend, "method": method, "n_trials": args.n_trials, **stats}
            all_summary.append(row)
            _dump_json(os.path.join(out_dir, f"trials_{backend}_{method}.json"), [{"trial": k, "ms": v} for k, v in enumerate(trial_times)])
            print(f"  → p50={stats['p50_ms']:.3f} ms | p95={stats['p95_ms']:.3f} ms")

        _dump_json(os.path.join(out_dir, "summary.json"), all_summary)
        _dump_csv(os.path.join(out_dir, "summary.csv"), all_summary)
        if args.plot: make_plots(all_summary, out_dir)
        print(f"\nDone. Results -> {out_dir}")
        return


    # =========================================================================
    # OPTION B: SYNTHETIC MATH OR NEURAL (V1 Baseline testing without loading real models)
    # =========================================================================
    assert args.state_dim >= 2 and args.state_dim % 2 == 0, "--state-dim must be even ≥ 2 for spiral"
    
    global _active_rhs, _active_neural_vf
    if args.vf_mode == "neural":
        neural_vf = NeuralVF(args.state_dim, device=args.device)
        _active_neural_vf = neural_vf
        _active_rhs = neural_vf  # callable numpy-in/numpy-out
        vf_label = f"neural MLP ({neural_vf.n_params:,} params)"
    else:
        _active_neural_vf = None
        _active_rhs = _default_rhs
        vf_label = "spiral (analytic)"
        
    hdr = f"ODE Benchmark V2 (SYNTHETIC) | device={args.device} | vf={vf_label} | steps={args.steps}"
    print("=" * 70); print(hdr); print("=" * 70)
    
    all_summary: List[Dict[str, Any]] = []
    x0_shape = (args.batch_size, args.state_dim)

    for i, sol in enumerate(solvers, 1):
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n[{i}/{len(solvers)}] {tag}")

        trial_times: List[float] = []

        # WARM-UP
        x0_warm = np.random.randn(*x0_shape).astype(np.float32)
        if backend == "legacy_euler":
            if _active_neural_vf is not None:
                _ = euler_integrate_torch(x0_warm, _active_neural_vf, int(args.steps), float(args.t0), float(args.t1))
            else:
                _ = euler_integrate(x0_warm, _active_rhs, int(args.steps), float(args.t0), float(args.t1))
        elif backend == "torchdiffeq":
            _ = torchdiffeq_integrate_synthetic(x0_warm, method, int(args.steps), float(args.t0), float(args.t1), float(args.rtol), float(args.atol), device=args.device)

        for trial in range(int(args.n_trials)):
            x0 = np.random.randn(*x0_shape).astype(np.float32)
            if "cuda" in args.device: torch.cuda.synchronize()
            t_start = time.perf_counter()

            if backend == "legacy_euler":
                if _active_neural_vf is not None:
                    _ = euler_integrate_torch(x0, _active_neural_vf, args.steps, args.t0, args.t1)
                else:
                    _ = euler_integrate(x0, _active_rhs, args.steps, args.t0, args.t1)
            elif backend == "torchdiffeq":
                _ = torchdiffeq_integrate_synthetic(x0, method, args.steps, args.t0, args.t1, args.rtol, args.atol, device=args.device)

            if "cuda" in args.device: torch.cuda.synchronize()
            ms = (time.perf_counter() - t_start) * 1000.0
            trial_times.append(ms)
            print(f"  trial {trial:03d}  {ms:8.3f} ms")

        stats = compute_stats(trial_times)
        row = {"backend": backend, "method": method, "n_trials": args.n_trials, **stats}
        all_summary.append(row)
        _dump_json(os.path.join(out_dir, f"trials_{backend}_{method}.json"), [{"trial": k, "ms": v} for k, v in enumerate(trial_times)])
        print(f"  → p50={stats['p50_ms']:.3f} ms | p95={stats['p95_ms']:.3f} ms")

    _dump_json(os.path.join(out_dir, "summary.json"), all_summary)
    _dump_csv(os.path.join(out_dir, "summary.csv"), all_summary)
    if args.plot: make_plots(all_summary, out_dir)
    print(f"\nDone. Results -> {out_dir}")

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
