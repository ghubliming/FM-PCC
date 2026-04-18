import os
import sys
import argparse
import time
import json
import csv
from datetime import datetime
import torch
import numpy as np

# Lightweight wrapper: Import the proven Fair-Loop and Utilities
from benchmark_ode_solvers_v3 import parse_solvers, p_sample_loop_v3_fair

def compute_stats(times_ms):
    a = np.asarray(times_ms, dtype=np.float64)
    return {
        "avg_ms": float(a.mean()) if len(a) > 0 else 0.0,
        "std_ms": float(a.std()) if len(a) > 1 else 0.0,
        "p50_ms": float(np.percentile(a, 50)) if len(a) > 0 else 0.0,
        "p95_ms": float(np.percentile(a, 95)) if len(a) > 0 else 0.0,
        "min_ms": float(a.min()) if len(a) > 0 else 0.0,
        "max_ms": float(a.max()) if len(a) > 0 else 0.0
    }

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode",        type=str,   default="math", choices=["math", "production"], help="Fairness mode.")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-trials",    type=int,   default=1,    help="For accuracy, 1 trial is enough. Increase to 20+ only for speed stats.")
    ap.add_argument("--batch-size",  type=int,   default=128)
    ap.add_argument("--state-dim",   type=int,   default=8)
    ap.add_argument("--t0",          type=float, default=0.0)
    ap.add_argument("--t1",          type=float, default=1.0)
    ap.add_argument("--steps",       type=int,   default=10, help="Candidate ODE steps to test.")
    ap.add_argument("--rtol",        type=float, default=1e-5)
    ap.add_argument("--atol",        type=float, default=1e-6)
    ap.add_argument("--solver-spec", type=str,   default="legacy:euler,legacy:rk4")
    ap.add_argument("--vf-mode",     type=str,   default="flow_matcher", choices=["spiral", "neural", "flow_matcher"])
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
    ap.add_argument("--include-bridge-tax", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    solvers = parse_solvers(args.solver_spec)
    
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(
        os.path.dirname(__file__), "benchmark_outputs_v3", f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 1. LOAD VF MODEL
    fm_model = None
    if args.vf_mode == "flow_matcher":
        from flow_matcher_v3_ode_selectable.utils import serialization as utils_serialization
        fm_exp = utils_serialization.load_diffusion(
            args.loadbase, args.dataset, args.diffusion_loadpath, 
            str(args.diffusion_seed), epoch=args.diffusion_epoch, device=args.device
        )
        fm_model = fm_exp.diffusion
        fm_model.eval()
        transition_dim = getattr(fm_model, 'transition_dim', 4)
        observation_dim = getattr(fm_model, 'observation_dim', transition_dim)
        shape = (args.batch_size, args.horizon, transition_dim)
        dummy_obs = torch.zeros(args.batch_size, observation_dim, device=args.device)
        cond = {0: dummy_obs}
    else:
        print("❌ Error: Accuracy Audit only supports --vf-mode flow_matcher.")
        sys.exit(1)

    print("=" * 80)
    print(f"ODE Accuracy Audit V3 | Mode={args.mode} | Device={args.device} | Candidate Steps={args.steps} | Trials={args.n_trials}")
    print("=" * 80)

    # 2. GENERATE GLOBAL NOISE BASIS
    global_noise = 0.5 * torch.randn(shape, device=args.device)

    # 3. THE ORACLE RUN
    print(f"\n[1] Generating ORACLE Ground Truth (ONCE)...")
    oracle_steps = 100 
    
    with torch.no_grad():
        if args.mode == "math":
            from torchdiffeq import odeint
            def ode_rhs_oracle(t_s, state):
                t_b = torch.ones(args.batch_size, device=args.device) * t_s
                return fm_model._predict_velocity(state, cond, t_b)
            
            oracle_t = torch.linspace(0.0, 1.0, args.steps + 1, device=args.device)
            oracle_traj = odeint(ode_rhs_oracle, global_noise.clone(), oracle_t, method="dopri5", atol=1e-10, rtol=1e-10)
            oracle_x = oracle_traj[-1]
        elif args.mode == "production":
            oracle_traj = p_sample_loop_v3_fair(
                fm_model, shape, cond, "dopri5", "torchdiffeq", 
                steps=oracle_steps, args=args, projector=None, return_trajectory=True
            )
            oracle_x = oracle_traj[-1]

    print(f"    ✅ Oracle Generated. (Terminal Norm: {torch.norm(oracle_x):.4f})")

    # 4. CANDIDATE EVALUATION
    all_summary = []
    
    for sol in solvers:
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n    -> Simulating {tag}")
        
        trial_times = []
        l2_dist_metric = 0.0
        mse_metric = 0.0
        step_drifts = []
        
        for trial in range(args.n_trials):
            t_start = time.perf_counter()
            with torch.no_grad():
                if args.mode == "math":
                    if backend == "legacy":
                        x_test = global_noise.clone()
                        traj_list = [x_test.clone()]
                        dt = 1.0 / args.steps
                        for i in range(args.steps):
                            t_s = float(i) / args.steps
                            t_b = torch.ones(args.batch_size, device=args.device) * t_s
                            
                            if method == "euler":
                                v = fm_model._predict_velocity(x_test, cond, t_b)
                                x_test = x_test + v * dt
                            elif method == "rk4":
                                v1 = fm_model._predict_velocity(x_test, cond, t_b)
                                v2 = fm_model._predict_velocity(x_test + v1*(dt*0.5), cond, t_b + (dt*0.5))
                                v3 = fm_model._predict_velocity(x_test + v2*(dt*0.5), cond, t_b + (dt*0.5))
                                v4 = fm_model._predict_velocity(x_test + v3*dt, cond, t_b + dt)
                                x_test = x_test + (v1 + 2*v2 + 2*v3 + v4) * (dt/6.0)
                            elif method == "midpoint":
                                v1 = fm_model._predict_velocity(x_test, cond, t_b)
                                x_mid = x_test + v1 * (dt * 0.5)
                                v2 = fm_model._predict_velocity(x_mid, cond, t_b + (dt * 0.5))
                                x_test = x_test + v2 * dt
                            else: raise ValueError(f"Method '{method}' missing.")
                            traj_list.append(x_test.clone())
                        candidate_traj = torch.stack(traj_list)
                    elif backend == "torchdiffeq":
                        from torchdiffeq import odeint
                        def test_rhs(t_s, state):
                            t_b = torch.ones(args.batch_size, device=args.device) * t_s
                            return fm_model._predict_velocity(state, cond, t_b)
                        t_span = torch.linspace(0.0, 1.0, args.steps + 1, device=args.device)
                        candidate_traj = odeint(test_rhs, global_noise.clone(), t_span, method=method)
                        x_test = candidate_traj[-1]
    
                elif args.mode == "production":
                    candidate_traj = p_sample_loop_v3_fair(
                        fm_model, shape, cond, method, backend, args.steps, args, 
                        projector=None, return_trajectory=True
                    )
                    x_test = candidate_traj[-1]
            
            if "cuda" in args.device: torch.cuda.synchronize()
            ms = (time.perf_counter() - t_start) * 1000.0
            trial_times.append(ms)
            
            if trial == 0:
                mse_metric = torch.nn.functional.mse_loss(x_test, oracle_x).item()
                diffs = x_test - oracle_x
                distances = torch.linalg.vector_norm(diffs.contiguous().view(args.batch_size, -1), dim=1)
                l2_dist_metric = distances.mean().item()
                l2_dist_std_metric = distances.std().item()

                # --- PER-STEP DRIFT ACCUMULATION ---
                n_cand = len(candidate_traj)
                n_oral = len(oracle_traj)
                if n_cand == n_oral:
                    for s in range(n_cand):
                        d = candidate_traj[s] - oracle_traj[s]
                        dist_s = torch.linalg.vector_norm(d.contiguous().view(args.batch_size, -1), dim=1)
                        step_drifts.append(float(dist_s.mean().item()))
                else:
                    ratio = (n_oral - 1) // (n_cand - 1)
                    for s in range(n_cand):
                        oral_idx = min(s * ratio, n_oral - 1)
                        d = candidate_traj[s] - oracle_traj[oral_idx]
                        dist_s = torch.linalg.vector_norm(d.contiguous().view(args.batch_size, -1), dim=1)
                        step_drifts.append(float(dist_s.mean().item()))

        stats = compute_stats(trial_times)
        row = {
            "backend": backend, "method": method, "steps": args.steps,
            "mse": mse_metric, "l2_distance_nm": l2_dist_metric,
            "l2_std_nm": l2_dist_std_metric,
            "step_drifts": step_drifts, "n_trials": args.n_trials, **stats
        }
        all_summary.append(row)
        print(f"      ✅ L2 Drift: {l2_dist_metric:.6f} | p50_ms: {stats['p50_ms']:.2f}")

    # 5. OUTPUT
    json_path = os.path.join(out_dir, "accuracy_summary.json")
    with open(json_path, 'w') as f: json.dump(all_summary, f, indent=4)
        
    print(f"\n================ FINAL ACCURACY SUMMARY ================")
    sorted_res = sorted(all_summary, key=lambda x: x["l2_distance_nm"])
    for i, r in enumerate(sorted_res):
         print(f" {i+1}. {r['backend']}:{r['method']:<10} | Steps: {r['steps']:<2} | p50_ms: {r['p50_ms']:>6.2f} | Drift L2: {r['l2_distance_nm']:.6f}")
    print("========================================================\n")
    print(f"✅ Benchmark Complete -> {json_path}")

    # 6. PLOTTING
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            # --- PLOT 1: DRIFT BAR CHART ---
            print("[Generating Accuracy Bar Chart...]")
            labels = [f"{r['backend']}:{r['method']}" for r in all_summary]
            drifts = [r['l2_distance_nm'] for r in all_summary]
            stds = [r['l2_std_nm'] for r in all_summary]
            
            sorted_idx = np.argsort(drifts)
            labels = [labels[i] for i in sorted_idx]
            drifts = [drifts[i] for i in sorted_idx]
            stds = [stds[i] for i in sorted_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(labels, drifts, yerr=stds, capsize=8, color='coral', edgecolor='black', linewidth=1.2)
            ax.bar_label(bars, fmt='%.3f', padding=10, fontsize=10)
            ax.set_title(f"ODE Math Deviation (Mean Drift across Batch, Steps={args.steps})", fontsize=14, pad=15)
            ax.set_ylabel("L2 Euclidean Distance (Lower is Better)", fontsize=12)
            ax.set_xlabel("Solver Configuration", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            plt.xticks(rotation=15, ha='right')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "accuracy_drift_plot.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)

            # --- PLOT 2: DRIFT ACCUMULATION ---
            print("[Generating Drift Accumulation Plot...]")
            fig, ax = plt.subplots(figsize=(10, 6))
            for r in all_summary:
                tag = f"{r['backend']}:{r['method']}"
                ax.plot(range(len(r['step_drifts'])), r['step_drifts'], marker='o', markersize=4, label=tag, linewidth=2)
            
            ax.set_title(f"Mean L2 Drift Accumulation (Steps={args.steps})", fontsize=14)
            ax.set_xlabel("Integration Step Index", fontsize=12)
            ax.set_ylabel("L2 Drift (vs Oracle)", fontsize=12)
            ax.set_yscale('log')
            ax.legend(); ax.grid(True, which="both", linestyle='--', alpha=0.5)
            fig.savefig(os.path.join(out_dir, "accuracy_drift_accumulation.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            print("✅ All Plots Saved successfully.")
            
        except ImportError:
            print("⚠️ Matplotlib not installed, skipping plot generation.")

if __name__ == "__main__":
    main()