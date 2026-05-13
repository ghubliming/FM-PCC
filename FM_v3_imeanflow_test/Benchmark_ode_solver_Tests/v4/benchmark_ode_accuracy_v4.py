#!/usr/bin/env python3
"""ODE Accuracy Audit (v4): High-resolution drift tracking with deterministic batch basis.

V4 Differences:
- INTEGRATION: Imports from benchmark_ode_solvers_v4 for unified loop logic.
- NOISE CONSISTENCY: Enforces that Oracle and all Candidates use the SAME global noise basis.
- DRIFT TRACKING: Retains V3's per-step trajectory drift analysis.
"""
import os
import sys
import argparse
import time
import json
import csv
from datetime import datetime
import torch
import numpy as np

# Import V4 logic
from benchmark_ode_solvers_v4 import parse_solvers, p_sample_loop_v4_fair

def compute_stats(times_ms):
    a = np.asarray(times_ms, dtype=np.float64)
    if len(a) == 0: return {"avg_ms": 0.0, "std_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {"avg_ms": float(a.mean()), "std_ms": float(a.std()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()), "max_ms": float(a.max())}

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode",        type=str,   default="math", choices=["math", "production"], help="Fairness mode.")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-trials",    type=int,   default=1)
    ap.add_argument("--batch-size",  type=int,   default=128)
    ap.add_argument("--state-dim",   type=int,   default=8)
    ap.add_argument("--steps",       type=int,   default=10, help="Candidate ODE steps to test.")
    ap.add_argument("--rtol",        type=float, default=1e-5)
    ap.add_argument("--atol",        type=float, default=1e-6)
    ap.add_argument("--solver-spec", type=str,   default="legacy:euler,legacy:rk4")
    ap.add_argument("--vf-mode",     type=str,   default="flow_matcher", choices=["flow_matcher"])
    ap.add_argument("--loadbase",    type=str,   default="logs")
    ap.add_argument("--dataset",     type=str,   default="avoiding-d3il")
    ap.add_argument("--diffusion-loadpath", type=str, default="")
    ap.add_argument("--diffusion-seed", type=int, default=0)
    ap.add_argument("--diffusion-epoch", type=str, default="latest")
    ap.add_argument("--horizon",     type=int,   default=128)
    ap.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-dir",  type=str,   default=None)
    ap.add_argument("--plot",        action="store_true")
    ap.add_argument("--track-trajectory", action="store_true", help="Monitor drift at every sub-integration step.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    solvers = parse_solvers(args.solver_spec)
    
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(
        os.path.dirname(__file__), "benchmark_outputs_v4", f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 1. LOAD VF MODEL
    from flow_matcher_v3_ode_selectable.utils import serialization as utils_serialization
    fm_exp = utils_serialization.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.diffusion_seed), epoch=args.diffusion_epoch, device=args.device)
    fm_model = fm_exp.diffusion
    fm_model.eval()
    t_dim = getattr(fm_model, 'transition_dim', 4)
    o_dim = getattr(fm_model, 'observation_dim', t_dim)
    shape = (args.batch_size, args.horizon, t_dim)
    cond = {0: torch.zeros(args.batch_size, o_dim, device=args.device)}

    print("=" * 80)
    print(f"ODE Accuracy Audit V4 | Mode={args.mode} | Candidate Steps={args.steps}")
    print("=" * 80)

    # 2. GENERATE GLOBAL NOISE BASIS
    global_noise = 0.5 * torch.randn(shape, device=args.device)

    # 3. THE ORACLE RUN
    print(f"\n[1] Generating ORACLE Ground Truth (Dopri5 @ 1e-10)...")
    
    with torch.no_grad():
        if args.mode == "math":
            from torchdiffeq import odeint
            def ode_rhs_oracle(t_s, state):
                return fm_model._predict_velocity(state, cond, torch.ones(args.batch_size, device=args.device)*t_s)
            ts_lin = torch.linspace(0.0, 1.0, args.steps+1, device=args.device)
            oracle_traj = odeint(ode_rhs_oracle, global_noise.clone(), ts_lin, method="dopri5", atol=1e-10, rtol=1e-10)
            oracle_x = oracle_traj[-1]
        else:
            oracle_traj = p_sample_loop_v4_fair(fm_model, shape, cond, "dopri5", "torchdiffeq", steps=100, args=args, x_init=global_noise, return_trajectory=True)
            oracle_x = oracle_traj[-1]

    # 4. CANDIDATE EVALUATION
    all_summary = []
    for sol in solvers:
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n    -> Simulating {tag}")
        
        trial_times, l2_dist_metric, step_drifts = [], 0.0, []
        for trial in range(args.n_trials):
            t_start = time.perf_counter()
            with torch.no_grad():
                if args.mode == "math":
                    if backend == "legacy":
                        x_test = global_noise.clone()
                        traj_l = [x_test.clone()]
                        dt = 1.0 / args.steps
                        for i in range(args.steps):
                            t_b = torch.ones(args.batch_size, device=args.device) * (i * dt)
                            if method == "euler":
                                v = fm_model._predict_velocity(x_test, cond, t_b)
                            elif method == "midpoint":
                                v1 = fm_model._predict_velocity(x_test, cond, t_b)
                                v = fm_model._predict_velocity(x_test + v1 * (dt * 0.5), cond, t_b + (dt * 0.5))
                            elif method == "rk4":
                                v1 = fm_model._predict_velocity(x_test, cond, t_b)
                                v2 = fm_model._predict_velocity(x_test + v1 * (dt * 0.5), cond, t_b + (dt * 0.5))
                                v3 = fm_model._predict_velocity(x_test + v2 * (dt * 0.5), cond, t_b + (dt * 0.5))
                                v4 = fm_model._predict_velocity(x_test + v3 * dt, cond, t_b + dt)
                                v = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
                            elif method == "dopri5":
                                k1 = fm_model._predict_velocity(x_test, cond, t_b)
                                k2 = fm_model._predict_velocity(x_test + dt * (1/5) * k1, cond, t_b + dt * (1/5))
                                k3 = fm_model._predict_velocity(x_test + dt * (3/40 * k1 + 9/40 * k2), cond, t_b + dt * (3/10))
                                k4 = fm_model._predict_velocity(x_test + dt * (44/45 * k1 - 56/15 * k2 + 32/9 * k3), cond, t_b + dt * (4/5))
                                k5 = fm_model._predict_velocity(x_test + dt * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4), cond, t_b + dt * (8/9))
                                k6 = fm_model._predict_velocity(x_test + dt * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5), cond, t_b + dt)
                                v = (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
                            x_test = x_test + v * dt
                            traj_l.append(x_test.clone())
                        candidate_traj = torch.stack(traj_l)
                    elif backend == "torchdiffeq":
                        from torchdiffeq import odeint
                        def t_rhs(ts, s): return fm_model._predict_velocity(s, cond, torch.ones(args.batch_size, device=args.device)*ts)
                        t_span = torch.linspace(0.0, 1.0, args.steps+1, device=args.device)
                        candidate_traj = odeint(t_rhs, global_noise.clone(), t_span, method=method)
                        x_test = candidate_traj[-1]
                else:
                    candidate_traj = p_sample_loop_v4_fair(fm_model, shape, cond, method, backend, args.steps, args, x_init=global_noise, return_trajectory=True)
                    x_test = candidate_traj[-1]
            
            if "cuda" in args.device: torch.cuda.synchronize()
            ms = (time.perf_counter() - t_start) * 1000.0
            trial_times.append(ms)
            
            if trial == 0:
                diffs = x_test - oracle_x
                l2_dist_metric = torch.linalg.vector_norm(diffs.contiguous().view(args.batch_size, -1), dim=1).mean().item()
                # simplified step drift for briefing logic
                for s in range(len(candidate_traj)):
                    oidx = min(s * ((len(oracle_traj)-1)//(len(candidate_traj)-1)), len(oracle_traj)-1) if len(candidate_traj)>1 else 0
                    step_drifts.append(float(torch.linalg.vector_norm((candidate_traj[s]-oracle_traj[oidx]).contiguous().view(args.batch_size,-1), dim=1).mean().item()))

        stats = compute_stats(trial_times)
        all_summary.append({"backend": backend, "method": method, "steps": args.steps, "l2_distance_nm": l2_dist_metric, "step_drifts": step_drifts, **stats})
        print(f"      ✅ L2 Drift: {l2_dist_metric:.6f}")

    with open(os.path.join(out_dir, "accuracy_summary.json"), 'w') as f: json.dump(all_summary, f, indent=4)
        
    print(f"\n================ FINAL V4 ACCURACY SUMMARY ================")
    sorted_res = sorted(all_summary, key=lambda x: x["l2_distance_nm"])
    for i, r in enumerate(sorted_res):
         print(f" {i+1}. {r['backend']}:{r['method']:<10} | Steps: {r['steps']:<2} | Drift L2: {r['l2_distance_nm']:.6f}")
    print("===========================================================\n")

    # --- PLOTTING ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            # 1. DRIFT BAR CHART
            labels = [f"{r['backend']}:{r['method']}" for r in all_summary]
            drifts = [r['l2_distance_nm'] for r in all_summary]
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, drifts, color='coral', edgecolor='black')
            ax.set_title(f"V4 ODE Math Deviation (Locked Batch, Steps={args.steps})", fontsize=14)
            ax.set_ylabel("L2 Drift")
            plt.xticks(rotation=15, ha='right')
            ax.bar_label(bars, fmt='%.4f', padding=3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "accuracy_drift_plot.png"), dpi=150)
            plt.close(fig)

            # 2. DRIFT ACCUMULATION
            fig, ax = plt.subplots(figsize=(10, 6))
            for r in all_summary:
                tag = f"{r['backend']}:{r['method']}"
                ax.plot(range(len(r['step_drifts'])), r['step_drifts'], marker='o', markersize=4, label=tag)
            ax.set_title(f"V4 Mean L2 Drift Accumulation (Steps={args.steps})", fontsize=14)
            ax.set_yscale('log'); ax.legend(); ax.grid(True, which="both", linestyle='--', alpha=0.5)
            fig.savefig(os.path.join(out_dir, "accuracy_drift_accumulation.png"), dpi=150)
            plt.close(fig)
            print(f"✅ Plots saved to {out_dir}")
        except ImportError:
            print("⚠️ Matplotlib not installed, skipping plots.")


if __name__ == "__main__":
    main()
