#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import itertools
import json
import csv
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Accuracy Grid Search Automated Runner V3")
    ap.add_argument("--mode", type=str, default="math", choices=["math", "production"], help="Fairness mode to pass to v3 script")
    ap.add_argument("--vf-mode", type=str, default="flow_matcher")
    ap.add_argument("--loadbase", type=str, default="logs")
    ap.add_argument("--dataset", type=str, default="avoiding-d3il")
    ap.add_argument("--diffusion-loadpath", type=str, default="flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion")
    ap.add_argument("--diffusion-seed", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--solver-spec", type=str, default="legacy:euler,legacy:rk4")
    
    # Grid sweeps
    ap.add_argument("--grid-batch", type=str, default="4,32")
    ap.add_argument("--grid-steps", type=str, default="5,10,20")
    ap.add_argument("--grid-horizon", type=str, default="8,16")
    
    ap.add_argument("--base-out", type=str, default="FM_v3_ode_selectable_test/benchmark_accuracy_grid_v3", help="Root folder for results")
    args = ap.parse_args()

    # Parse grids
    batch_grid = [int(b) for b in args.grid_batch.split(",") if b.strip()]
    steps_grid = [int(s) for s in args.grid_steps.split(",") if s.strip()]
    horizon_grid = [int(h) for h in args.grid_horizon.split(",") if h.strip()]

    print(f"=== Starting ODE Accuracy Grid Search V3 [Mode: {args.mode}] ===")
    
    total_runs = len(batch_grid) * len(steps_grid) * len(horizon_grid)
    current_run = 1

    script_path = os.path.join(os.path.dirname(__file__), "benchmark_ode_accuracy_v3.py")
    generated_folders = []

    # 1. RUN ALL GRID SUBPROCESSES
    for horizon, batch, steps in itertools.product(horizon_grid, batch_grid, steps_grid):
        run_name = f"acc_mode_{args.mode}_h{horizon}_b{batch}_s{steps}"
        out_dir = os.path.join(args.base_out, run_name)
        generated_folders.append((out_dir, horizon, batch, steps))
        
        print(f"\n[{current_run}/{total_runs}] Running accuracy pass: {run_name}")
        cmd = [
            sys.executable, script_path,
            "--mode", args.mode,
            "--vf-mode", args.vf_mode,
            "--loadbase", args.loadbase,
            "--dataset", args.dataset,
            "--diffusion-loadpath", args.diffusion_loadpath,
            "--diffusion-seed", str(args.diffusion_seed),
            "--horizon", str(horizon),
            "--device", args.device,
            "--solver-spec", args.solver_spec,
            "--batch-size", str(batch),
            "--steps", str(steps),
            "--output-dir", out_dir,
            "--plot" # We ask the base script to plot itself too just in case
        ]
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed! {e}")
            
        current_run += 1
        
    # 2. AGGREGATE
    print("\n[Aggregating Results...]")
    master_data = []
    for (out_dir, horizon, batch, steps) in generated_folders:
        json_path = os.path.join(out_dir, "accuracy_summary.json")
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f:
            runs = json.load(f)
        for r in runs:
            row = {
                "mode": args.mode,
                "horizon": horizon,
                "batch_size": batch,
                "steps": steps,
                "backend_method": f"{r['backend']}:{r['method']}",
                **r
            }
            master_data.append(row)
            
    os.makedirs(args.base_out, exist_ok=True)
    csv_path = os.path.join(args.base_out, f"MASTER_ACCURACY_MATRIX_V3_{args.mode}.csv")
    if master_data:
        keys = master_data[0].keys()
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, list(keys))
            w.writeheader(); w.writerows(master_data)
        print(f"✅ Aggregated Data -> {csv_path}")
        
        # Print a quick console summary
        print("\n================ FINAL RESULTS SUMMARY ================")
        print("🥇 TOP 5 MOST ACCURATE CONFIGURATIONS (Lowest L2 Drift):")
        sorted_data = sorted(master_data, key=lambda x: x["l2_distance_nm"])
        for i, d in enumerate(sorted_data[:5]):
            print(f" {i+1}. {d['backend_method']:<18} | H={d['horizon']:<2} S={d['steps']:<2} B={d['batch_size']:<2} | Drift L2: {d['l2_distance_nm']:.6f}")
        print("=======================================================\n")
        
        # 3. MACRO PLOTTING using Matplotlib
        print("[Generating Macro Plots...]")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            
            solvers = list(set([r["backend_method"] for r in master_data]))
            
            # Plot 1: Math Drift vs Steps (O(h) vs O(h^4))
            fig, ax = plt.subplots(figsize=(10,6))
            for solver in solvers:
                # Group by constant batch and horizon
                points = [r for r in master_data if r["backend_method"]==solver and r["batch_size"]==batch_grid[0] and r["horizon"]==horizon_grid[0]]
                points = sorted(points, key=lambda x: x["steps"])
                if not points: continue
                # We expect exponential decay here for RK4!
                ax.plot([p["steps"] for p in points], [p["l2_distance_nm"] for p in points], marker='^', label=solver, linewidth=2)
                
            ax.set_title(f"Accuracy vs ODE Steps (h={horizon_grid[0]}, b={batch_grid[0]}) [{args.mode.upper()}]")
            ax.set_xlabel("Integration Steps ($S$)")
            ax.set_ylabel("L2 Euclidean Drift (Lower is better)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            # Log scale y-axis handles exponential differences much better visually
            ax.set_yscale('log')
            fig.savefig(os.path.join(args.base_out, f"macroplot_ACCURACY_vs_STEPS_v3_{args.mode}.png"), dpi=200, bbox_inches='tight')
            plt.close(fig)
            
            # Plot 2: Math Drift vs Horizon
            fig, ax = plt.subplots(figsize=(10,6))
            for solver in solvers:
                points = [r for r in master_data if r["backend_method"]==solver and r["batch_size"]==batch_grid[0] and r["steps"]==steps_grid[-1]]
                points = sorted(points, key=lambda x: x["horizon"])
                if not points: continue
                ax.plot([p["horizon"] for p in points], [p["l2_distance_nm"] for p in points], marker='s', label=solver, linewidth=2)
                
            ax.set_title(f"Trajectory Elongation Cost: Accuracy vs Horizon (b={batch_grid[0]}, s={steps_grid[-1]}) [{args.mode.upper()}]")
            ax.set_xlabel("Horizon Length ($H$)")
            ax.set_ylabel("L2 Euclidean Drift (Lower is better)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            fig.savefig(os.path.join(args.base_out, f"macroplot_ACCURACY_vs_HORIZON_v3_{args.mode}.png"), dpi=150)
            plt.close(fig)

            print("✅ Accuracy Macro Plots (Steps scale, Horizon scale) generated successfully!")
        except ImportError:
            print("⚠️ Matplotlib not installed, skipping macro plots.")

if __name__ == "__main__":
    main()
