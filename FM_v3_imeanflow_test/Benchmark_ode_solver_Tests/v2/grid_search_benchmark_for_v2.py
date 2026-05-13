#!/usr/bin/env python3
import argparse
import subprocess
import os
import itertools
import json
import csv

def main():
    ap = argparse.ArgumentParser(description="Grid Search Automated Benchmark Runner")
    # Base fixed parameters
    ap.add_argument("--vf-mode", type=str, default="flow_matcher")
    ap.add_argument("--loadbase", type=str, default="logs")
    ap.add_argument("--dataset", type=str, default="avoiding-d3il")
    ap.add_argument("--diffusion-loadpath", type=str, default="flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion")
    ap.add_argument("--diffusion-seed", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--solver-spec", type=str, default="legacy_euler,torchdiffeq:euler,torchdiffeq:midpoint,torchdiffeq:rk4,torchdiffeq:dopri5")
    
    # Grid Search Lists (Comma separated strings)
    ap.add_argument("--grid-batch", type=str, default="4,32,128", help="Batch sizes to sweep")
    ap.add_argument("--grid-steps", type=str, default="10,20", help="Integration steps to sweep")
    ap.add_argument("--grid-horizon", type=str, default="8,16,32", help="Sequence horizons to sweep")
    ap.add_argument("--grid-bridge", action="store_true", help="If set, toggles testing both WITH and WITHOUT the bridge tax")
    
    ap.add_argument("--base-out", type=str, default="FM_v3_ode_selectable_test/benchmark_grid_search_outputs", help="Root folder for results")
    args = ap.parse_args()

    # Parse grids
    batch_grid = [int(b) for b in args.grid_batch.split(",") if b.strip()]
    steps_grid = [int(s) for s in args.grid_steps.split(",") if s.strip()]
    horizon_grid = [int(h) for h in args.grid_horizon.split(",") if h.strip()]
    bridge_grid = [False, True] if args.grid_bridge else [False]

    print(f"=== Starting ODE Benchmark Grid Search ===")
    print(f"Sweeping Batch Sizes: {batch_grid}")
    print(f"Sweeping Steps:       {steps_grid}")
    print(f"Sweeping Horizons:    {horizon_grid}")
    print(f"Sweeping Bridge Tax:  {bridge_grid}")
    print(f"Target Solvers:       {args.solver_spec}")
    
    total_runs = len(batch_grid) * len(steps_grid) * len(horizon_grid) * len(bridge_grid)
    current_run = 1

    # Base path to the actual executable
    script_path = os.path.join(os.path.dirname(__file__), "benchmark_ode_solvers_v2.py")
    
    # Track paths for final aggregate
    generated_folders = []

    # 1. RUN ALL GRID SUBPROCESSES
    for horizon, batch, steps, bridge in itertools.product(horizon_grid, batch_grid, steps_grid, bridge_grid):
        bridge_str = "with_tax" if bridge else "pure_gpu"
        run_name = f"h{horizon}_b{batch}_s{steps}_{bridge_str}"
        out_dir = os.path.join(args.base_out, run_name)
        generated_folders.append((out_dir, horizon, batch, steps, bridge))
        
        print(f"\n[{current_run}/{total_runs}] Running combination: {run_name}")
        cmd = [
            "python", script_path,
            "--vf-mode", args.vf_mode,
            "--loadbase", args.loadbase,
            "--dataset", args.dataset,
            "--diffusion-loadpath", args.diffusion_loadpath,
            "--diffusion-seed", str(args.diffusion_seed),
            "--horizon", str(horizon),
            "--n-trials", str(args.n_trials),
            "--device", args.device,
            "--solver-spec", args.solver_spec,
            "--batch-size", str(batch),
            "--steps", str(steps),
            "--output-dir", out_dir,
            "--plot"
        ]
        if bridge: cmd.append("--include-bridge-tax")
            
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed! Exited with error {e}")
            
        current_run += 1
        
    # 2. AGGREGATE RESULTS INTO MEGA MATRIX
    print("\n[Aggregating Results into Gigantic Matrix...]")
    master_data = []
    
    for (out_dir, horizon, batch, steps, bridge) in generated_folders:
        json_path = os.path.join(out_dir, "summary.json")
        if not os.path.exists(json_path): continue
        
        with open(json_path, 'r') as f:
            runs = json.load(f)
            
        for r in runs:
            # We construct a flattened master JSON containing BOTH parameters and metrics
            row = {
                "horizon": horizon,
                "batch_size": batch,
                "steps": steps,
                "bridge_tax_included": bridge,
                "backend_method": f"{r['backend']}:{r['method']}",
                **r
            }
            master_data.append(row)
            
    # Save Master File
    os.makedirs(args.base_out, exist_ok=True)
    csv_path = os.path.join(args.base_out, "MASTER_MATRIX_RESULTS.csv")
    if master_data:
        keys = master_data[0].keys()
        with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(master_data)
        print(f"✅ Saved Aggregated Data to {csv_path}")
        
        # Print a quick console summary
        print("\n================ FINAL RESULTS SUMMARY ================")
        print("TOP 5 FASTEST CONFIGURATIONS:")
        sorted_data = sorted(master_data, key=lambda x: x["avg_ms"])
        for i, d in enumerate(sorted_data[:5]):
            tax_status = "Tax: YES" if d['bridge_tax_included'] else "Tax: NO"
            print(f" {i+1}. {d['backend_method']:<20} | H={d['horizon']:<2} S={d['steps']:<2} | {tax_status} | {d['avg_ms']:>6.2f} ms")
        print("=======================================================\n")
        
    # 3. MACRO PLOTTING using Matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        # Sort data
        solvers = list(set([r["backend_method"] for r in master_data]))
        
        # Plot 1: Batch Scalability
        fig, ax = plt.subplots(figsize=(10,6))
        for solver in solvers:
            # Filter for specific baseline parameters to isolate batch growth
            points = [r for r in master_data if r["backend_method"]==solver and r["horizon"]==horizon_grid[0] and r["steps"]==steps_grid[0] and r["bridge_tax_included"]==False]
            points = sorted(points, key=lambda x: x["batch_size"])
            if not points: continue
            ax.plot([p["batch_size"] for p in points], [p["avg_ms"] for p in points], marker='o', label=solver)
        ax.set_title(f"Scalability: Solver latency vs Batch Size (h={horizon_grid[0]}, s={steps_grid[0]})")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Average Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3) # Subtle grid
        fig.savefig(os.path.join(args.base_out, "macroplot_batch_influence.png"), dpi=150)
        plt.close(fig)

        # Plot 2: Horizon Complexity 
        fig, ax = plt.subplots(figsize=(10,6))
        for solver in solvers:
            points = [r for r in master_data if r["backend_method"]==solver and r["batch_size"]==batch_grid[0] and r["steps"]==steps_grid[0] and r["bridge_tax_included"]==False]
            points = sorted(points, key=lambda x: x["horizon"])
            if not points: continue
            ax.plot([p["horizon"] for p in points], [p["avg_ms"] for p in points], marker='s', label=solver)
        ax.set_title(f"Sequence Cost: Solver latency vs Horizon (b={batch_grid[0]}, s={steps_grid[0]})")
        ax.set_xlabel("Horizon Length")
        ax.set_ylabel("Average Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(args.base_out, "macroplot_horizon_influence.png"), dpi=150)
        plt.close(fig)

        # Plot 3: Steps Influence (The user's densest sweep)
        fig, ax = plt.subplots(figsize=(10,6))
        for solver in solvers:
            points = [r for r in master_data if r["backend_method"]==solver and r["batch_size"]==batch_grid[0] and r["horizon"]==horizon_grid[0] and r["bridge_tax_included"]==False]
            points = sorted(points, key=lambda x: x["steps"])
            if not points: continue
            ax.plot([p["steps"] for p in points], [p["avg_ms"] for p in points], marker='^', label=solver)
        ax.set_title(f"Step Cost: Solver latency vs ODE Steps (h={horizon_grid[0]}, b={batch_grid[0]})")
        ax.set_xlabel("Integration Steps")
        ax.set_ylabel("Average Latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(args.base_out, "macroplot_steps_influence.png"), dpi=150)
        plt.close(fig)
        
        print("✅ Macro Summary Plots (Batch, Horizon, Steps) generated successfully!")
    except ImportError:
        print("⚠️ Matplotlib not installed, skipping macro plots.")

    print(f"\n🎉 Meta-Analysis Complete! All matrices saved to {args.base_out}")

if __name__ == "__main__":
    main()
