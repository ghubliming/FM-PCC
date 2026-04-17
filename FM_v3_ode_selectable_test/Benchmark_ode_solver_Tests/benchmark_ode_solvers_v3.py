#!/usr/bin/env python3
"""Standalone ODE-solver benchmark (v3) on synthetic vector fields & actual U-Net Models.

V3 Differences:
- RESOLVES the "Performance Paradox": Ensures 100% fair pathing between all solvers.
- Introduces --mode {math, production} to explicitly choose the benchmark environment.
- Standardizes naming conventions (legacy:euler, torchdiffeq:rk4, etc.)
- Retains all V2 features (Mixed synthetic/real testing, plotting, grid support).
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

# --- MINARI MOCK ---
if 'minari' not in sys.modules:
    from unittest.mock import MagicMock
    sys.modules['minari'] = MagicMock()

# Dynamically add the project root to sys.path
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# ---------------------------------------------------------------------------
# 1. Synthetic Vector Fields (Baseline Logic)
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

def _default_rhs(x: np.ndarray) -> np.ndarray:
    return spiral_vf(x, 0.35, 1.25, 0.12)

class NeuralVF:
    def __init__(self, state_dim: int, device: str = "cpu"):
        import torch.nn as nn
        self.device = torch.device(device)
        self.state_dim = state_dim
        embed_dim = 128
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, 512), nn.Mish(), nn.Linear(512, embed_dim)).to(self.device)
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
# 2. Integrators (Synthetic Path)
# ---------------------------------------------------------------------------

def euler_integrate_torch(x0: torch.Tensor, model: Any, n_steps: int, t0: float, t1: float) -> torch.Tensor:
    dt = (t1 - t0) / n_steps
    x = x0.clone()
    t_val = torch.tensor(t0, device=x.device)
    for _ in range(n_steps):
        v = model.__call_torch__(t_val, x) if hasattr(model, '__call_torch__') else model._predict_velocity(x, {0: torch.zeros_like(x[:,0,:])}, torch.ones(x.shape[0], device=x.device)*t_val)
        x = x + dt * v
        t_val = t_val + dt
    return x

def torchdiffeq_integrate_synthetic(x0: np.ndarray, method: str, n_steps: int, t0: float, t1: float, rtol: float, atol: float, device: str = "cpu") -> np.ndarray:
    from torchdiffeq import odeint
    dev = torch.device(device)
    x0_t = torch.from_numpy(x0).float().to(dev)
    def rhs_torch(_t, x_t):
        if _active_neural_vf: return _active_neural_vf.__call_torch__(_t, x_t)
        dx_np = _active_rhs(x_t.detach().cpu().numpy())
        return torch.from_numpy(dx_np).to(dtype=x_t.dtype, device=x_t.device)
    
    with torch.no_grad():
        if method in {"euler", "midpoint", "rk4"}:
            ts = torch.linspace(t0, t1, n_steps + 1, device=dev)
            traj = odeint(rhs_torch, x0_t, ts, method=method)
        else:
            ts = torch.tensor([t0, t1], dtype=torch.float32, device=dev)
            traj = odeint(rhs_torch, x0_t, ts, method=method, rtol=rtol, atol=atol)
    return traj[-1].detach().cpu().numpy()

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def parse_solvers(spec: str) -> List[Dict[str, str]]:
    solvers = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry: continue
        if ":" in entry: backend, method = entry.split(":", 1)
        else: backend, method = entry, entry
        backend, method = backend.strip(), method.strip()
        if backend == "legacy_euler": backend, method = "legacy", "euler"
        elif backend.startswith("legacy_"): 
            method = backend.replace("legacy_", "")
            backend = "legacy"
        solvers.append({"backend": backend, "method": method})
    return solvers

def compute_stats(times_ms: List[float]) -> Dict[str, float]:
    a = np.asarray(times_ms, dtype=np.float64)
    return {"avg_ms": float(a.mean()), "std_ms": float(a.std()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "max_ms": float(a.max())}

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
        ax.set_ylabel(metric); ax.set_title(f"ODE Benchmark V3 — {metric}"); ax.bar_label(bars, fmt="%.2f", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"plot_{metric}.png"), dpi=150); plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Main Evaluator
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode",        type=str,   default="math", choices=["math", "production"], help="Fairness mode.")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-trials",    type=int,   default=50)
    ap.add_argument("--batch-size",  type=int,   default=128)
    ap.add_argument("--state-dim",   type=int,   default=8)
    ap.add_argument("--t0",          type=float, default=0.0)
    ap.add_argument("--t1",          type=float, default=1.0)
    ap.add_argument("--steps",       type=int,   default=20)
    ap.add_argument("--rtol",        type=float, default=1e-5)
    ap.add_argument("--atol",        type=float, default=1e-6)
    ap.add_argument("--solver-spec", type=str,   default="legacy:euler,torchdiffeq:rk4")
    ap.add_argument("--vf-mode",     type=str,   default="spiral", choices=["spiral", "neural", "flow_matcher"])
    ap.add_argument("--loadbase",    type=str,   default="logs")
    ap.add_argument("--dataset",     type=str,   default="avoiding-d3il")
    ap.add_argument("--diffusion-loadpath", type=str, default="")
    ap.add_argument("--diffusion-seed", type=int, default=0)
    ap.add_argument("--diffusion-epoch", type=str, default="latest")
    ap.add_argument("--horizon",     type=int,   default=128)
    ap.add_argument("--device",      type=str,   default="cpu")
    ap.add_argument("--output-dir",  type=str,   default=None)
    ap.add_argument("--plot",        action="store_true")
    ap.add_argument("--include-bridge-tax", action="store_true")
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    solvers = parse_solvers(args.solver_spec)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.path.dirname(__file__), "benchmark_outputs_v3", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Load real model if requested
    fm_model = None
    if args.vf_mode == "flow_matcher":
        from flow_matcher_v3_ode_selectable.utils import serialization as utils_serialization
        fm_exp = utils_serialization.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.diffusion_seed), epoch=args.diffusion_epoch, device=args.device)
        fm_model = fm_exp.diffusion
        fm_model.eval()
        transition_dim = getattr(fm_model, 'transition_dim', 4)
        observation_dim = getattr(fm_model, 'observation_dim', transition_dim)
        shape = (args.batch_size, args.horizon, transition_dim)
        dummy_obs = torch.zeros(args.batch_size, observation_dim, device=args.device)
        cond = {0: dummy_obs}
    
    hdr = f"ODE Benchmark V3 | mode={args.mode} | device={args.device} | vf={args.vf_mode} | steps={args.steps}"
    print("=" * 70); print(hdr); print("=" * 70)
    
    all_summary = []
    for i, sol in enumerate(solvers, 1):
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n[{i}/{len(solvers)}] {tag}")
        
        # Warm-up (3 cycles for stability)
        for _ in range(3):
            with torch.no_grad():
                if args.vf_mode == "flow_matcher":
                    fm_model.ode_solver_backend_v3 = backend
                    fm_model.ode_solver_method_v3 = method
                    fm_model.flow_steps_v3 = args.steps
                    _ = fm_model.p_sample_loop(shape, cond)
                else: 
                    _ = torchdiffeq_integrate_synthetic(np.zeros((args.batch_size, args.state_dim)), method, args.steps, args.t0, args.t1, args.rtol, args.atol, args.device)
        if "cuda" in args.device: torch.cuda.synchronize()

        trial_times = []
        for trial in range(args.n_trials):
            t_start = time.perf_counter()
            with torch.no_grad():
                if args.mode == "math" and backend == "legacy" and args.vf_mode == "flow_matcher":
                    # FAIR MATH PATH (Common loop for ALL legacy solvers)
                    x = 0.5 * torch.randn(shape, device=args.device)
                    dt = 1.0 / args.steps
                    for s in range(args.steps):
                        t_b = torch.ones(args.batch_size, device=args.device) * (s * dt)
                        if method == "euler": v = fm_model._predict_velocity(x, cond, t_b)
                        elif method == "midpoint":
                            v1 = fm_model._predict_velocity(x, cond, t_b)
                            v = fm_model._predict_velocity(x + v1*(dt*0.5), cond, t_b + (dt*0.5))
                        elif method == "rk4":
                            v1 = fm_model._predict_velocity(x, cond, t_b)
                            v2 = fm_model._predict_velocity(x + v1*(dt*0.5), cond, t_b + (dt*0.5))
                            v3 = fm_model._predict_velocity(x + v2*(dt*0.5), cond, t_b + (dt*0.5))
                            v4 = fm_model._predict_velocity(x + v3*dt, cond, t_b + dt)
                            v = (v1 + 2*v2 + 2*v3 + v4) / 6.0
                        x = x + v * dt
                    if args.include_bridge_tax: _ = x.cpu().numpy()
                elif args.vf_mode == "flow_matcher":
                    # PRODUCTION PATH (Unified p_sample_loop for everyone)
                    fm_model.ode_solver_backend_v3 = backend
                    fm_model.ode_solver_method_v3 = method
                    fm_model.flow_steps_v3 = args.steps
                    res = fm_model.p_sample_loop(shape, cond)
                    if args.include_bridge_tax: _ = res.cpu().numpy()
                else:
                    # Synthetic Baseline
                    _ = torchdiffeq_integrate_synthetic(np.random.randn(args.batch_size, args.state_dim), method, args.steps, args.t0, args.t1, args.rtol, args.atol, args.device)

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

def _dump_json(p, o):
    with open(p, "w") as f: json.dump(o, f, indent=2)
def _dump_csv(p, r):
    if not r: return
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(r[0].keys()))
        w.writeheader(); w.writerows(r)

if __name__ == "__main__":
    main()
