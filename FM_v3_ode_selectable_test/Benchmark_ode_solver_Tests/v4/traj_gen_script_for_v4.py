#!/usr/bin/env python3
"""
traj_gen_script_for_v4.py
Standalone script to generate diffuser-style trajectory plots from V4 ODE solver benchmarks.
Directly utilizes existing flow_matcher utils and config/projection_eval.yaml.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import yaml

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dynamically add the project root to sys.path
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import flow_matcher_v3_ode_selectable.utils as utils
from flow_matcher_v3_ode_selectable.utils import serialization as utils_serialization

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", type=str, required=True, help="Path to the v4 benchmark outputs directory containing .npy trajectory files")
    ap.add_argument("--dataset", type=str, default="avoiding-d3il")
    ap.add_argument("--seed", type=int, default=0, help="Seed used for loading the diffusion model normalizer")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--plot-batch-limit", type=int, default=4, help="Maximum number of trajectories from the batch to plot")
    args = ap.parse_args()

    benchmark_dir = os.path.abspath(args.benchmark_dir)
    if not os.path.isdir(benchmark_dir):
        print(f"Error: {benchmark_dir} is not a valid directory.")
        sys.exit(1)

    traj_files = glob.glob(os.path.join(benchmark_dir, "traj_*.npy"))
    if not traj_files:
        print(f"No traj_*.npy files found in {benchmark_dir}. Did you run the benchmark with --datalog-for-traj?")
        sys.exit(1)

    print(f"Found {len(traj_files)} trajectory file(s).")

    # 1. Load normalizer by initializing a dummy model container
    print(f"Loading normalizer from dataset '{args.dataset}' (seed {args.seed})...")
    fm_exp = utils_serialization.load_diffusion("logs", args.dataset, "", str(args.seed), epoch="latest", device=args.device)
    normalizer = fm_exp.dataset.normalizer

    # 2. Load configurations for constraints
    config_path = os.path.join(_PROJ_ROOT, "config", "projection_eval.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp = args.dataset
    obs_indices = config["observation_indices"]["avoiding"]
    x_idx = obs_indices["x"]
    y_idx = obs_indices["y"]

    # We use the default 'top-left-hard' configuration similar to eval defaults for simplicity, 
    # but could be iterated over if needed.
    halfspace_variant = "top-left-hard"
    if halfspace_variant == "top-left-hard":
        polytopic_constraints = [config["halfspace_constraints"][exp][0]]
        obstacle_constraints = [config["obstacle_constraints"][exp][3]]
    else:
        polytopic_constraints = config["halfspace_constraints"][exp]
        obstacle_constraints = config["obstacle_constraints"][exp]

    ax_limits = config["ax_limits"][exp]
    constraint_types = config["constraint_types"]

    # 3. Process each trajectory file
    for file_path in traj_files:
        basename = os.path.basename(file_path)
        print(f"\n[{basename}] Processing...")
        
        traj_np = np.load(file_path) # [batch_size, horizon, t_dim]
        
        # Unnormalize
        traj_unnorm = normalizer.unnormalize(traj_np, "observations")

        # Console Output of parameters
        batch_size = traj_unnorm.shape[0]
        print(f"[{basename}] Console Output of Trajectory Parameters (X, Y):")
        for b in range(batch_size):
            xs = traj_unnorm[b, :, x_idx]
            ys = traj_unnorm[b, :, y_idx]
            print(f"  Batch {b} X: {xs}")
            print(f"  Batch {b} Y: {ys}")

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot Trajectories
        plot_limit = min(batch_size, args.plot_batch_limit)
        for b in range(plot_limit):
            ax.plot(traj_unnorm[b, :, x_idx], traj_unnorm[b, :, y_idx], "b")
            ax.plot(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], "go", label="Start" if b == 0 else "")

        ax.set_xlim(ax_limits[0])
        ax.set_ylim(ax_limits[1])
        ax.set_title(f"Trajectory Visualization: {basename.replace('.npy', '')}")

        # Overlay Constraints using real code directly
        utils.plot_environment_constraints(exp, ax)
        if "halfspace" in constraint_types:
            utils.plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits)
        if "obstacles" in constraint_types:
            for constraint in obstacle_constraints:
                ax.add_patch(matplotlib.patches.Circle(
                    constraint["center"], constraint["radius"], color="b", alpha=0.2
                ))
        
        ax.legend()
        out_png = os.path.join(benchmark_dir, basename.replace(".npy", ".png"))
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"[{basename}] Saved plot to {out_png}")

if __name__ == "__main__":
    main()
