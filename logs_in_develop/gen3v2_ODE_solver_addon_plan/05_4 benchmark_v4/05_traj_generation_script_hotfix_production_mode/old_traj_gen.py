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
    ap.add_argument("--loadbase", type=str, default="logs")
    ap.add_argument("--diffusion-loadpath", type=str, default="", help="Subpath to the model (e.g. flow_matching_v3/...)")
    ap.add_argument("--diffusion-seed", type=int, default=0, help="Seed used for loading the diffusion model normalizer")
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
    print(f"Loading normalizer from dataset '{args.dataset}' (seed {args.diffusion_seed})...")
    fm_exp = utils_serialization.load_diffusion(args.loadbase, args.dataset, args.diffusion_loadpath, str(args.diffusion_seed), epoch="latest", device=args.device)
    normalizer = fm_exp.dataset.normalizer

    # 2. Load configurations for constraints
    config_path = os.path.join(_PROJ_ROOT, "config", "projection_eval.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp = args.dataset
    obs_indices = config["observation_indices"]["avoiding"]
    x_idx = obs_indices["x"]
    y_idx = obs_indices["y"]

    # 2. Load Environment Constraints
    exp = args.dataset
    polytopic_constraints = config["halfspace_constraints"][exp]
    obstacle_constraints = config["obstacle_constraints"][exp]

    ax_limits = config["ax_limits"][exp]
    constraint_types = config["constraint_types"]

    # 3. Initialize Comparison Plot and Dynamic Colors
    fig_all, ax_all = plt.subplots(1, 1, figsize=(12, 12))
    
    # Use a high-contrast color palette for distinctness
    distinct_palette = plt.get_cmap("tab10").colors 
    color_map = {}
    color_idx = 0
    
    # Pre-identify the Oracle to reserve Red for it if needed
    # But we'll just handle it in the loop for simplicity.

    # 4. Process each trajectory file
    for file_path in traj_files:
        basename = os.path.basename(file_path)
        print(f"\n[{basename}] Processing...")
        
        traj_np = np.load(file_path) # [batch_size, horizon, t_dim]
        
        # Unnormalize (only the observation dimensions if t_dim > obs_dim)
        obs_dim = normalizer.normalizers["observations"].mins.shape[0]
        traj_obs = traj_np[:, :, :obs_dim]
        traj_unnorm = normalizer.unnormalize(traj_obs, "observations")

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
            ax.plot(traj_unnorm[b, :, x_idx], traj_unnorm[b, :, y_idx], "b", alpha=0.6, linewidth=1.0)
            ax.plot(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], "go", markersize=6, label="Start" if b == 0 else "")
            ax.plot(traj_unnorm[b, -1, x_idx], traj_unnorm[b, -1, y_idx], "rx", markersize=8, label="End" if b == 0 else "")
            # Label batch number
            ax.text(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], f"B{b}", fontsize=9, fontweight='bold')

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
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[{basename}] Saved plot to {out_png}")

        # Add to Comparison Plot with Logic: Oracle=Red, Others=Distinct
        if basename not in color_map:
            if "dopri5" in basename.lower():
                color_map[basename] = "red"
            else:
                # Pick a color from tab10 that isn't red-like (tab10 index 3 is red-ish)
                # We'll just cycle through and skip index 3 if we want to be strictly different from red.
                c = distinct_palette[color_idx % 10]
                if color_idx % 10 == 3: # Skip the red-ish color in tab10
                    color_idx += 1
                    c = distinct_palette[color_idx % 10]
                color_map[basename] = c
                color_idx += 1
        
        current_color = color_map[basename]
        
        for b in range(plot_limit):
            ax_all.plot(traj_unnorm[b, :, x_idx], traj_unnorm[b, :, y_idx], color=current_color, alpha=0.7, linewidth=1.0, 
                        label=basename.replace("traj_", "").replace(".npy", "") if b == 0 else "")
            ax_all.plot(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], "go", markersize=3, alpha=0.5)
            ax_all.plot(traj_unnorm[b, -1, x_idx], traj_unnorm[b, -1, y_idx], "rx", markersize=4, alpha=0.5)
            
            # Only mark batch numbers for the Oracle (dopri5) for clarity
            if "dopri5" in basename.lower():
                 ax_all.text(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], f"B{b}", fontsize=8, fontweight='bold', alpha=0.8)

    # 5. Finalize and save comparison plot
    ax_all.set_xlim(ax_limits[0])
    ax_all.set_ylim(ax_limits[1])
    ax_all.set_title(f"Solver Comparison: All Batches ({exp})")
    utils.plot_environment_constraints(exp, ax_all)
    
    # Symbols legend
    ax_all.plot([], [], 'go', markersize=6, label='Start Point')
    ax_all.plot([], [], 'rx', markersize=8, label='End Point')
    
    ax_all.legend(loc='upper right', fontsize='x-small', ncol=2)
    
    # Save as high-res PNG
    comparison_png = os.path.join(benchmark_dir, "solver_comparison_all.png")
    fig_all.savefig(comparison_png, dpi=300, bbox_inches="tight")
    
    # Save as SVG for maximum quality
    comparison_svg = os.path.join(benchmark_dir, "solver_comparison_all.svg")
    fig_all.savefig(comparison_svg, bbox_inches="tight")
    plt.close(fig_all)
    print(f"\n[Comparison] Saved master comparison plot to {comparison_png} (and .svg)")

    # 6. Per-Batch Comparison (Requested Add-on)
    # We re-iterate through the data we already have to create individual batch audits
    print(f"\n[Per-Batch] Generating individual batch audits...")
    all_trajs = {}
    for f in traj_files:
        bn = os.path.basename(f)
        data = np.load(f)
        obs_dim = normalizer.normalizers["observations"].mins.shape[0]
        unnorm = normalizer.unnormalize(data[:, :, :obs_dim], "observations")
        all_trajs[bn] = unnorm

    plot_limit = min(all_trajs[os.path.basename(traj_files[0])].shape[0], args.plot_batch_limit)
    
    for b in range(plot_limit):
        fig_b, ax_b = plt.subplots(1, 1, figsize=(10, 10))
        for basename, traj_unnorm in all_trajs.items():
            current_color = color_map[basename]
            label = basename.replace("traj_", "").replace(".npy", "")
            ax_b.plot(traj_unnorm[b, :, x_idx], traj_unnorm[b, :, y_idx], color=current_color, alpha=0.8, linewidth=1.2, label=label)
            ax_b.plot(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], "go", markersize=4)
            ax_b.plot(traj_unnorm[b, -1, x_idx], traj_unnorm[b, -1, y_idx], "rx", markersize=5)
            
        ax_b.set_xlim(ax_limits[0])
        ax_b.set_ylim(ax_limits[1])
        ax_b.set_title(f"Per-Batch Comparison: Batch {b} (Shared Noise Basis)")
        utils.plot_environment_constraints(exp, ax_b)
        
        ax_b.legend(loc='upper right', fontsize='small')
        out_b = os.path.join(benchmark_dir, f"batch_comparison_B{b}.png")
        fig_b.savefig(out_b, dpi=300, bbox_inches="tight")
        plt.close(fig_b)
        print(f"  [B{b}] Saved to {out_b}")

if __name__ == "__main__":
    main()