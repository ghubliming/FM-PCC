#!/usr/bin/env python3
import argparse
import subprocess
import os
import itertools
import json
import csv
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(description="Grid Search Automated Benchmark Runner V3")
    ap.add_argument("--mode", type=str, default="math", choices=["math", "production"], help="Fairness mode to pass to v3 script")
    ap.add_argument("--vf-mode", type=str, default="flow_matcher")
    ap.add_argument("--loadbase", type=str, default="logs")
    ap.add_argument("--dataset", type=str, default="avoiding-d3il")
    ap.add_argument("--diffusion-loadpath", type=str, default="flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion")
    ap.add_argument("--diffusion-seed", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--solver-spec", type=str, default="legacy:euler,torchdiffeq:euler,torchdiffeq:rk4")
    
    ap.add_argument("--grid-batch", type=str, default="4,32,128")
    ap.add_argument("--grid-steps", type=str, default="10,20")
    ap.add_argument("--grid-horizon", type=str, default="8,16,32")
    ap.add_argument("--grid-bridge", action="store_true")
    
    ap.add_argument("--base-out", type=str, default="FM_v3_ode_selectable_test/benchmark_grid_search_v3", help="Root folder for results")
    args = ap.parse_args()

    # Parse grids
    batch_grid = [int(b) for b in args.grid_batch.split(",") if b.strip()]
    steps_grid = [int(s) for s in args.grid_steps.split(",") if s.strip()]
    horizon_grid = [int(h) for h in args.grid_horizon.split(",") if h.strip()]
    bridge_grid = [False, True] if args.grid_bridge else [False]

    print(f"=== Starting ODE Benchmark Grid Search V3 [Mode: {args.mode}] ===")
    
    total_runs = len(batch_grid) * len(steps_grid) * len(horizon_grid) * len(bridge_grid)
    current_run = 1

    script_path = os.path.join(os.path.dirname(__file__), "benchmark_ode_solvers_v3.py")
    generated_folders = []

    # 1. RUN ALL GRID SUBPROCESSES
    for horizon, batch, steps, bridge in itertools.product(horizon_grid, batch_grid, steps_grid, bridge_grid):
        bridge_str = "with_tax" if bridge else "pure_gpu"
        run_name = f"mode_{args.mode}_h{horizon}_b{batch}_s{steps}_{bridge_str}"
        out_dir = os.path.join(args.base_out, run_name)
        generated_folders.append((out_dir, horizon, batch, steps, bridge))
        
        print(f"\n[{current_run}/{total_runs}] Running combination: {run_name}")
        cmd = [
            "python", script_path,
            "--mode", args.mode,
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
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed! {e}")
            
        current_run += 1
        
    # 2. AGGREGATE
    print("\n[Aggregating Results...]")
    master_data = []
    for (out_dir, horizon, batch, steps, bridge) in generated_folders:
        json_path = os.path.join(out_dir, "summary.json")
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f:
            runs = json.load(f)
        for r in runs:
            row = {
                "mode": args.mode,
                "horizon": horizon,
                "batch_size": batch,
                "steps": steps,
                "bridge_tax_included": bridge,
                "backend_method": f"{r['backend']}:{r['method']}",
                **r
            }
            master_data.append(row)
            
    os.makedirs(args.base_out, exist_ok=True)
    csv_path = os.path.join(args.base_out, f"MASTER_MATRIX_V3_{args.mode}.csv")
    if master_data:
        keys = master_data[0].keys()
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader(); w.writerows(master_data)
        print(f"✅ Aggregated Data -> {csv_path}")

if __name__ == "__main__":
    main()
