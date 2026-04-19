#!/usr/bin/env python3
"""Grid Search Automated Benchmark Runner (V4).
Ensures deterministic seed propagation for the V4 benchmarking suite.
"""
import argparse
import subprocess
import os
import sys
import itertools
import json
import csv

def main():
    ap = argparse.ArgumentParser(description="Grid Search Automated Benchmark Runner V4")
    ap.add_argument("--mode", type=str, default="math", choices=["math", "production"])
    ap.add_argument("--vf-mode", type=str, default="flow_matcher")
    ap.add_argument("--loadbase", type=str, default="logs")
    ap.add_argument("--dataset", type=str, default="avoiding-d3il")
    ap.add_argument("--diffusion-loadpath", type=str, default="flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion")
    ap.add_argument("--diffusion-seed", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--solver-spec", type=str, default="legacy:euler,torchdiffeq:euler,torchdiffeq:rk4")
    ap.add_argument("--seed", type=int, default=0)
    
    ap.add_argument("--grid-batch", type=str, default="4,32,128")
    ap.add_argument("--grid-steps", type=str, default="10,20")
    ap.add_argument("--grid-horizon", type=str, default="8,16,32")
    
    ap.add_argument("--base-out", type=str, default="FM_v3_ode_selectable_test/benchmark_grid_search_v4")
    args = ap.parse_args()

    batch_grid = [int(b) for b in args.grid_batch.split(",")]
    steps_grid = [int(s) for s in args.grid_steps.split(",")]
    horizon_grid = [int(h) for h in args.grid_horizon.split(",")]

    script_path = os.path.join(os.path.dirname(__file__), "benchmark_ode_solvers_v4.py")
    generated_folders = []

    for horizon, batch, steps in itertools.product(horizon_grid, batch_grid, steps_grid):
        run_name = f"mode_{args.mode}_h{horizon}_b{batch}_s{steps}"
        out_dir = os.path.join(args.base_out, run_name)
        generated_folders.append((out_dir, horizon, batch, steps))
        
        cmd = [
            sys.executable, script_path,
            "--mode", args.mode, "--seed", str(args.seed),
            "--vf-mode", args.vf_mode, "--loadbase", args.loadbase, "--dataset", args.dataset,
            "--diffusion-loadpath", args.diffusion_loadpath, "--diffusion-seed", str(args.diffusion_seed),
            "--horizon", str(horizon), "--n-trials", str(args.n_trials),
            "--device", args.device, "--solver-spec", args.solver_spec,
            "--batch-size", str(batch), "--steps", str(steps), "--output-dir", out_dir
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError: pass
        
    master_data = []
    for (out_dir, horizon, batch, steps) in generated_folders:
        json_path = os.path.join(out_dir, "summary.json")
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f:
            runs = json.load(f)
        for r in runs:
            master_data.append({"mode": args.mode, "horizon": horizon, "batch_size": batch, "steps": steps, "backend_method": f"{r['backend']}:{r['method']}", **r})
            
    os.makedirs(args.base_out, exist_ok=True)
    if master_data:
        with open(os.path.join(args.base_out, f"MASTER_MATRIX_V4_{args.mode}.csv"), 'w', newline='') as f:
            w = csv.DictWriter(f, master_data[0].keys())
            w.writeheader(); w.writerows(master_data)

if __name__ == "__main__":
    main()
