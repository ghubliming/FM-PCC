#!/usr/bin/env python3
"""Standalone ODE-solver benchmark (v4) on synthetic vector fields & actual U-Net Models.

V4 Differences:
- DETERMINISTIC MATH MODE: Moves noise initialization OUTSIDE the trial loop for 'math' audits.
- SPLIT-LOGIC STRATEGY: 'math' uses locked noise basis; 'production' maintains random per-trial robustness.
- Preserves all V3 features (Production mirrors, Legacy math fixes, Plotting).
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
    if len(a) == 0: return {"avg_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
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
        ax.set_ylabel(metric); ax.set_title(f"ODE Benchmark V4 — {metric}"); ax.bar_label(bars, fmt="%.2f", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"plot_{metric}.png"), dpi=150); plt.close(fig)

    # Combined Overview Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, metric, color in zip(axes.flat, metrics, colors):
        vals = [r[metric] for r in summary]
        bars = ax.bar(labels, vals, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(metric, fontsize=10)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.suptitle("ODE Benchmark V4 — All Metrics Summary", fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "plot_overview.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Fair Production Mirror (V4 deterministic support)
# ---------------------------------------------------------------------------

def p_sample_loop_v4_fair(fm_model, shape, cond, method, backend, steps, args, projector=None, constraints=None, x_init=None, return_trajectory=False):
    """
    MIRROR of flow_matcher_v3_ode_selectable/models/diffusion.py :: p_sample_loop
    Standardized for V4 to accept an optional x_init for deterministic audits.
    """
    from flow_matcher_v3_ode_selectable.models.helpers import apply_conditioning
    device = args.device
    batch_size = shape[0]
    
    # 1. Initial State
    if x_init is not None:
        x = x_init.clone()
    else:
        x = 0.5 * torch.randn(shape, device=device)
    
    x = apply_conditioning(x, cond, fm_model.action_dim, goal_dim=fm_model.goal_dim)
    
    traj = [x.clone()]
    dt = 1.0 / max(steps, 1)
    
    # 2. Integration Loop
    for i in range(steps):
        t_scalar = float(i) / max(steps, 1)
        t_cont = torch.full((batch_size,), t_scalar, device=device, dtype=torch.float32)
        
        if backend == "torchdiffeq":
            from torchdiffeq import odeint
            t_span = torch.tensor([t_scalar, t_scalar + dt], device=device)
            def ode_rhs(t_s, state):
                t_b = torch.ones(batch_size, device=device) * t_s
                return fm_model._predict_velocity(state, cond, t_b)
            x = odeint(ode_rhs, x, t_span, method=method)[-1]
        else:
            # LEGACY FAIR PATH
            if method == "euler":
                v = fm_model._predict_velocity(x, cond, t_cont)
                x = x + v * dt
            elif method == "midpoint":
                v1 = fm_model._predict_velocity(x, cond, t_cont)
                x_mid = x + v1 * (dt * 0.5)
                v2 = fm_model._predict_velocity(x_mid, cond, t_cont + (dt * 0.5))
                x = x + v2 * dt
            elif method == "rk4":
                v1 = fm_model._predict_velocity(x, cond, t_cont)
                v2 = fm_model._predict_velocity(x + v1*(dt*0.5), cond, t_cont + (dt*0.5))
                v3 = fm_model._predict_velocity(x + v2*(dt*0.5), cond, t_cont + (dt*0.5))
                v4 = fm_model._predict_velocity(x + v3*dt, cond, t_cont + dt)
                x = x + (v1 + 2*v2 + 2*v3 + v4) * (dt/6.0)
            elif method == "dopri5":
                v1 = fm_model._predict_velocity(x, cond, t_cont)
                v2 = fm_model._predict_velocity(x + v1*(dt/5), cond, t_cont + (dt/5))
                v3 = fm_model._predict_velocity(x + v1*(3/40)*dt + v2*(9/40)*dt, cond, t_cont + (3/10)*dt)
                v4 = fm_model._predict_velocity(x + v1*(44/45)*dt - v2*(56/15)*dt + v3*(32/9)*dt, cond, t_cont + (4/5)*dt)
                v5 = fm_model._predict_velocity(x + v1*(19372/6561)*dt - v2*(25360/2187)*dt + v3*(64448/6561)*dt - v4*(212/729)*dt, cond, t_cont + (8/9)*dt)
                v6 = fm_model._predict_velocity(x + v1*(9017/3168)*dt - v2*(355/33)*dt + v3*(46732/5247)*dt + v4*(49/176)*dt - v5*(5103/18656)*dt, cond, t_cont + dt)
                x = x + (35/384*v1 + 500/1113*v3 + 125/192*v4 - 2187/6784*v5 + 11/84*v6) * dt
        
        # 3. Post-Velocity Boilerplate
        x = apply_conditioning(x, cond, fm_model.action_dim, goal_dim=fm_model.goal_dim)
        x = apply_conditioning(x, cond, fm_model.action_dim, goal_dim=fm_model.goal_dim)
        if return_trajectory: traj.append(x.clone())
        
    return torch.stack(traj) if return_trajectory else x

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
    ap.add_argument("--vf-mode",     type=str,   default="flow_matcher", choices=["flow_matcher"])
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
    torch.manual_seed(args.seed)
    solvers = parse_solvers(args.solver_spec)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(os.path.dirname(__file__), "benchmark_outputs_v4", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Load real model
    from flow_matcher_v3_ode_selectable.utils import serialization as utils_serialization
    fm_exp = utils_serialization.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.diffusion_seed), epoch=args.diffusion_epoch, device=args.device)
    fm_model = fm_exp.diffusion
    fm_model.eval()
    t_dim = getattr(fm_model, 'transition_dim', 4)
    o_dim = getattr(fm_model, 'observation_dim', t_dim)
    shape = (args.batch_size, args.horizon, t_dim)
    cond = {0: torch.zeros(args.batch_size, o_dim, device=args.device)}
    
    # --- V4 DETERMINISTIC BASIS ---
    global_x_init = 0.5 * torch.randn(shape, device=args.device)

    print("=" * 70)
    print(f"ODE Benchmark V4 | mode={args.mode} | steps={args.steps}")
    print("=" * 70)
    
    all_summary = []
    for i, sol in enumerate(solvers, 1):
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n[{i}/{len(solvers)}] {tag}")
        
        # Warm-up (3 cycles)
        for _ in range(3):
            with torch.no_grad():
                if args.mode == "math":
                    if backend == "legacy":
                        _ = fm_model._predict_velocity(global_x_init, cond, torch.ones(args.batch_size, device=args.device)*0.5)
                    elif backend == "torchdiffeq":
                        def rhs_w(ts, s): return fm_model._predict_velocity(s, cond, torch.ones(args.batch_size, device=args.device)*ts)
                        from torchdiffeq import odeint
                        _ = odeint(rhs_w, global_x_init, torch.tensor([0.0, 0.1], device=args.device), method="euler")
                else: 
                    _ = p_sample_loop_v4_fair(fm_model, shape, cond, method, backend, args.steps, args)

        if "cuda" in args.device: torch.cuda.synchronize()
        trial_times = []
        for trial in range(args.n_trials):
            current_x_init = global_x_init if args.mode == "math" else None
            t_start = time.perf_counter()
            with torch.no_grad():
                if args.mode == "math":
                    if backend == "legacy":
                        x = current_x_init.clone()
                        dt = 1.0 / args.steps
                        if method == "dopri5":
                            for s_idx in range(args.steps):
                                t_v_b = torch.ones(args.batch_size, device=args.device) * (s_idx * dt)
                                k1 = fm_model._predict_velocity(x, cond, t_v_b)
                                k2 = fm_model._predict_velocity(x + dt*(1/5)*k1, cond, t_v_b + dt*(1/5))
                                k3 = fm_model._predict_velocity(x + dt*(3/40*k1 + 9/40*k2), cond, t_v_b + dt*(3/10))
                                k4 = fm_model._predict_velocity(x + dt*(44/45*k1 - 56/15*k2 + 32/9*k3), cond, t_v_b + dt*(4/5))
                                k5 = fm_model._predict_velocity(x + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - k4*(212/729)*dt), cond, t_v_b + dt*(8/9)) # wait k4 typo fix
                                k5 = fm_model._predict_velocity(x + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4), cond, t_v_b + dt*(8/9))
                                k6 = fm_model._predict_velocity(x + dt*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5), cond, t_v_b + dt)
                                x = x + dt*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
                        else:
                            for s in range(args.steps):
                                t_b = torch.ones(args.batch_size, device=args.device) * (s * dt)
                                if method == "euler": v = fm_model._predict_velocity(x, cond, t_b)
                                elif method == "midpoint":
                                    v1 = fm_model._predict_velocity(x, cond, t_b)
                                    v = fm_model._predict_velocity(x + v1*(dt*0.5), cond, t_b + (dt*0.5))
                                elif method == "rk4":
                                    v1 = fm_model._predict_velocity(x, cond, t_b)
                                    v2 = fm_model._predict_velocity(x+v1*(dt*0.5), cond, t_b+(dt*0.5))
                                    v3 = fm_model._predict_velocity(x+v2*(dt*0.5), cond, t_b+(dt*0.5))
                                    v4 = fm_model._predict_velocity(x+v3*dt, cond, t_b+dt)
                                    v = (v1 + 2*v2 + 2*v3 + v4) / 6.0
                                x = x + v * dt
                        if args.include_bridge_tax: _ = x.cpu().numpy()
                    elif backend == "torchdiffeq":
                        from torchdiffeq import odeint
                        def rhs_l(ts, s): return fm_model._predict_velocity(s, cond, torch.ones(args.batch_size, device=args.device)*ts)
                        ts_span = torch.linspace(args.t0, args.t1, args.steps+1, device=args.device) if method in {"euler", "midpoint", "rk4"} else torch.tensor([args.t0, args.t1], device=args.device)
                        res = odeint(rhs_l, current_x_init, ts_span, method=method, rtol=args.rtol, atol=args.atol)
                        if args.include_bridge_tax: _ = res[-1].cpu().numpy()
                else:
                    res = p_sample_loop_v4_fair(fm_model, shape, cond, method, backend, args.steps, args, x_init=current_x_init)
                    if args.include_bridge_tax: _ = res.cpu().numpy()

            if "cuda" in args.device: torch.cuda.synchronize()
            ms = (time.perf_counter() - t_start) * 1000.0
            trial_times.append(ms); print(f"  trial {trial:03d}  {ms:8.3f} ms")

        stats = compute_stats(trial_times)
        all_summary.append({"backend": backend, "method": method, "n_trials": args.n_trials, **stats})
        _dump_json(os.path.join(out_dir, f"trials_{backend}_{method}.json"), [{"trial": k, "ms": v} for k, v in enumerate(trial_times)])
    
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
