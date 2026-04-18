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
        "avg_ms": float(a.mean()),
        "std_ms": float(a.std()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max())
    }

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode",        type=str,   default="math", choices=["math", "production"], help="Fairness mode.")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n-trials",    type=int,   default=20, help="Number of trials for speed variance reliability.")
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

    # 2. GENERATE GLOBAL NOISE BASIS (Single Golden Start Point)
    global_noise = 0.5 * torch.randn(shape, device=args.device)

    # 3. THE ORACLE RUN
    print(f"\n[1] Generating ORACLE Ground Truth (ONCE)...")
    oracle_steps = 100 
    print(f"    -> Running torchdiffeq:dopri5 natively with tight tolerances (Steps={oracle_steps})")
    
    with torch.no_grad():
        if args.mode == "math":
            from torchdiffeq import odeint
            def ode_rhs_oracle(t_s, state):
                t_b = torch.ones(args.batch_size, device=args.device) * t_s
                return fm_model._predict_velocity(state, cond, t_b)
            
            oracle_t = torch.tensor([0.0, 1.0], device=args.device)
            oracle_x = odeint(ode_rhs_oracle, global_noise.clone(), oracle_t, method="dopri5", atol=1e-10, rtol=1e-10)[-1]
        elif args.mode == "production":
            oracle_x = p_sample_loop_v3_fair(
                fm_model, shape, cond, "dopri5", "torchdiffeq", 
                steps=oracle_steps, args=args, projector=None
            )

    print(f"    ✅ Oracle Generated. (Terminal Norm: {torch.norm(oracle_x):.4f})")

    # 4. CANDIDATE EVALUATION (With n-trials multi-loop)
    print(f"\n[2] Evaluating Candidate Solvers over {args.n_trials} trials (Steps = {args.steps})")
    all_summary = []
    
    for sol in solvers:
        backend, method = sol["backend"], sol["method"]
        tag = f"{backend}:{method}"
        print(f"\n    -> Simulating {tag}")
        
        trial_times = []
        l2_dist_metric = 0.0
        mse_metric = 0.0
        
        for trial in range(args.n_trials):
            t_start = time.perf_counter()
            with torch.no_grad():
                if args.mode == "math":
                    if backend == "legacy":
                        x_test = global_noise.clone()
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
                            else:
                                raise ValueError(f"Legacy '{method}' missing.")
                    elif backend == "torchdiffeq":
                        from torchdiffeq import odeint
                        def test_rhs(t_s, state):
                            t_b = torch.ones(args.batch_size, device=args.device) * t_s
                            return fm_model._predict_velocity(state, cond, t_b)
                        t_span = torch.linspace(0.0, 1.0, args.steps + 1, device=args.device)
                        x_test = odeint(test_rhs, global_noise.clone(), t_span, method=method)[-1]
    
                elif args.mode == "production":
                    # We pass global_noise correctly here by modifying p_sample_loop_v3_fair temporarily,
                    # but since the wrapper doesn't take 'noise', we monkey-patch random.randn.
                    original_randn = torch.randn
                    def fixed_randn(*n_args, **n_kwargs): return global_noise.clone() * 2.0
                    torch.randn = fixed_randn
                    x_test = p_sample_loop_v3_fair(
                        fm_model, shape, cond, method, backend, args.steps, args, projector=None
                    )
                    torch.randn = original_randn # Restore
            
            if "cuda" in args.device: torch.cuda.synchronize()
            ms = (time.perf_counter() - t_start) * 1000.0
            trial_times.append(ms)
            
            # Record math drift (identically constant across deterministic trials)
            if trial == 0:
                mse_metric = torch.nn.functional.mse_loss(x_test, oracle_x).item()
                
                # Calculate True Statistical Variance by measuring L2 deviation PER BATCH ITEM
                diffs = x_test - oracle_x
                # Flatten horizon and dimensions to get 1 Euclidean distance per element in the batch
                distances = torch.linalg.vector_norm(diffs.contiguous().view(args.batch_size, -1), dim=1)
                
                l2_dist_metric = distances.mean().item()
                l2_dist_std_metric = distances.std().item()

        stats = compute_stats(trial_times)
        row = {
            "backend": backend, 
            "method": method, 
            "steps": args.steps,
            "mse": mse_metric,
            "l2_distance_nm": l2_dist_metric,
            "l2_std_nm": l2_dist_std_metric,
            "n_trials": args.n_trials,
            **stats
        }
        all_summary.append(row)
        print(f"      ✅ L2 Drift: {l2_dist_metric:.6f} | p50_ms: {stats['p50_ms']:.2f}")

    # 6. OUTPUT SAVING
    json_path = os.path.join(out_dir, "accuracy_summary.json")
    with open(json_path, 'w') as f:
        json.dump(all_summary, f, indent=4)
        
    print(f"\n================ FINAL ACCURACY SUMMARY ================")
    sorted_res = sorted(all_summary, key=lambda x: x["l2_distance_nm"])
    for i, r in enumerate(sorted_res):
         print(f" {i+1}. {r['backend']}:{r['method']:<10} | Steps: {r['steps']:<2} | p50_ms: {r['p50_ms']:>6.2f} | Drift L2: {r['l2_distance_nm']:.6f}")
    print("========================================================\n")
    print(f"✅ Benchmark Matrix Saved -> {json_path}")

    # 7. VISUALIZATION (Plotting)
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
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
            ax.set_title(f"ODE Math Deviation (Mean ± Std across Batch, Steps={args.steps}) [{args.mode.upper()}]", fontsize=14, pad=15)
            ax.set_ylabel("L2 Euclidean Distance (Lower is Better)", fontsize=12)
            ax.set_xlabel("Solver Configuration", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            plt.xticks(rotation=15, ha='right')
            fig.tight_layout()
            
            plot_path = os.path.join(out_dir, "accuracy_drift_plot.png")
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Accuracy Plot Saved -> {plot_path}")
            
        except ImportError:
            print("⚠️ Matplotlib not installed, skipping plot generation.")

if __name__ == "__main__":
    main()
