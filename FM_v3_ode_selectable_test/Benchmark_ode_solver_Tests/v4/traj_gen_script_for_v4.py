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
import json

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

    polytopic_constraints = config["halfspace_constraints"][exp]
    obstacle_constraints = config["obstacle_constraints"][exp]
    ax_limits = config["ax_limits"][exp]
    constraint_types = config["constraint_types"]

    # 3. Load Metadata and True Conditions
    metadata_path = os.path.join(benchmark_dir, "traj_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        n_init_points = metadata.get("n_init_points", 1)
        batch_size_per_init = metadata.get("batch_size_per_init", None)
    else:
        n_init_points = 1
        batch_size_per_init = None

    cond_path = os.path.join(benchmark_dir, "cond_true_start.npy")
    if os.path.exists(cond_path):
        true_cond_norm = np.load(cond_path)
    else:
        true_cond_norm = None

    # Load all trajectory files into memory
    print("Loading all trajectory files into memory...")
    all_trajs = {}
    for file_path in traj_files:
        basename = os.path.basename(file_path)
        traj_np = np.load(file_path) # [total_batch_size, horizon, t_dim] or [steps+1, total_batch_size, horizon, t_dim]
        
        # If the file contains the full ODE evolution (4D), take the final generated plan
        if traj_np.ndim == 4:
            traj_np = traj_np[-1]
            
        action_dim = getattr(fm_exp.diffusion, 'action_dim', 0)
        obs_dim = normalizer.normalizers["observations"].mins.shape[0]
        traj_obs = traj_np[..., action_dim : action_dim + obs_dim]
        all_trajs[basename] = normalizer.unnormalize(traj_obs, "observations")
        if batch_size_per_init is None:
            batch_size_per_init = traj_np.shape[0] // n_init_points
    
    # [V4 STRICT SAFETY] Abort if the solver start does not match the True Start (Yellow Star)
    # We check this in NORMALIZED space to ensure 100% mathematical parity.
    for basename in all_trajs.keys():
        # Load the raw normalized data for verification
        raw_norm = np.load(os.path.join(benchmark_dir, basename))
        if raw_norm.ndim == 4: raw_norm = raw_norm[-1]
        
        # Sliced normalized start vs true cond start
        actual_obs_start_norm = raw_norm[:, 0, action_dim : action_dim + obs_dim]
        if true_cond_norm is not None:
            if not np.allclose(actual_obs_start_norm, true_cond_norm, atol=1e-4):
                raise AssertionError(f"CRITICAL: Production Anchoring Drift detected in {basename}! Plotting aborted to prevent misinformation.")

    # Unnormalize the True Start for plotting
    if true_cond_norm is not None:
        true_start_unnorm_all = normalizer.unnormalize(true_cond_norm, "observations")
    else:
        true_start_unnorm_all = normalizer.unnormalize(np.zeros((n_init_points * batch_size_per_init, obs_dim)), "observations")

    # Set up dynamic colors
    distinct_palette = plt.get_cmap("tab10").colors 
    color_map = {}
    color_idx = 0
    for basename in all_trajs.keys():
        if "dopri5" in basename.lower():
            color_map[basename] = "red"
        else:
            c = distinct_palette[color_idx % 10]
            if color_idx % 10 == 3: # skip red-ish
                color_idx += 1
                c = distinct_palette[color_idx % 10]
            color_map[basename] = c
            color_idx += 1

    # 4. Outer loop for Initialization Points
    for init_idx in range(n_init_points):
        print(f"\n{'='*50}\nProcessing Initialization Point [{init_idx+1}/{n_init_points}]\n{'='*50}")
        start_idx = init_idx * batch_size_per_init
        end_idx = start_idx + batch_size_per_init
        plot_limit = min(batch_size_per_init, args.plot_batch_limit)
        
        # Comparison plot for this init_idx
        fig_all, ax_all = plt.subplots(1, 1, figsize=(12, 12))
        
        for basename, traj_unnorm_full in all_trajs.items():
            traj_unnorm = traj_unnorm_full[start_idx:end_idx]
            true_start_unnorm = true_start_unnorm_all[start_idx:end_idx]
            
            print(f"[{basename}] Console Output of Trajectory Parameters (X, Y):")
            for b in range(plot_limit):
                xs = traj_unnorm[b, :, x_idx]
                ys = traj_unnorm[b, :, y_idx]
                print(f"  Batch {b} X: {xs}")
                print(f"  Batch {b} Y: {ys}")
                if true_start_unnorm is not None:
                    print(f"  --> Yellow Star X: {true_start_unnorm[b, x_idx]}, Y: {true_start_unnorm[b, y_idx]}")

            # Plotting Per-Solver
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            current_color = color_map[basename]
            for b in range(plot_limit):
                ax.plot(traj_unnorm[b, :, x_idx], traj_unnorm[b, :, y_idx], color=current_color, alpha=0.7, linewidth=1.0, zorder=10,
                        label="Solver Traj" if b == 0 else "")
                ax.plot(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], "go", markersize=3, alpha=0.5, zorder=11, label="Solver Start" if b == 0 else "")
                ax.plot(traj_unnorm[b, -1, x_idx], traj_unnorm[b, -1, y_idx], "rx", markersize=4, alpha=0.5, zorder=11, label="Solver End" if b == 0 else "")
                ax.plot(true_start_unnorm[b, x_idx], true_start_unnorm[b, y_idx], "y*", markersize=8, alpha=0.9, zorder=12, label="True Start (Cond)" if b == 0 else "")
                ax.text(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], f"B{b}", fontsize=7, alpha=0.7, zorder=13)

            ax.set_xlim(ax_limits[0])
            ax.set_ylim(ax_limits[1])
            ax.set_title(f"Trajectory Visualization: {basename.replace('.npy', '')} (Init {init_idx})")

            # Clean constraints only
            utils.plot_environment_constraints(exp, ax)
            
            ax.legend()
            suffix = f"_init{init_idx}" if n_init_points > 1 else ""
            out_png = os.path.join(benchmark_dir, basename.replace(".npy", f"{suffix}.png"))
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[{basename}] Saved plot to {out_png}")

            # Add to Comparison Plot
            current_color = color_map[basename]
            for b in range(plot_limit):
                ax_all.plot(traj_unnorm[b, :, x_idx], traj_unnorm[b, :, y_idx], color=current_color, alpha=0.7, linewidth=1.0, 
                            label=basename.replace("traj_", "").replace(".npy", "") if b == 0 else "")
                ax_all.plot(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], "go", markersize=3, alpha=0.5)
                ax_all.plot(traj_unnorm[b, -1, x_idx], traj_unnorm[b, -1, y_idx], "rx", markersize=4, alpha=0.5)
                
                if basename == list(all_trajs.keys())[0]:
                    ax_all.plot(true_start_unnorm[b, x_idx], true_start_unnorm[b, y_idx], "y*", markersize=8, alpha=0.9)
                
                if "dopri5" in basename.lower():
                     ax_all.text(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], f"B{b}", fontsize=8, fontweight='bold', alpha=0.8)

        # Finalize and save comparison plot
        ax_all.set_xlim(ax_limits[0])
        ax_all.set_ylim(ax_limits[1])
        ax_all.set_title(f"Solver Comparison: All Batches ({exp}) - Init {init_idx}")
        utils.plot_environment_constraints(exp, ax_all)
        
        ax_all.plot([], [], 'go', markersize=6, label='Solver Start Point')
        ax_all.plot([], [], 'rx', markersize=8, label='Solver End Point')
        ax_all.plot([], [], 'y*', markersize=10, label='True Start (Cond)')
        ax_all.legend(loc='upper right', fontsize='x-small', ncol=2)
        
        comparison_png = os.path.join(benchmark_dir, f"solver_comparison_all{suffix}.png")
        fig_all.savefig(comparison_png, dpi=300, bbox_inches="tight")
        comparison_svg = os.path.join(benchmark_dir, f"solver_comparison_all{suffix}.svg")
        fig_all.savefig(comparison_svg, bbox_inches="tight")
        plt.close(fig_all)
        print(f"\n[Comparison] Saved master comparison plot to {comparison_png}")

        # Per-Batch Comparison
        print(f"\n[Per-Batch] Generating individual batch audits for Init {init_idx}...")
        for b in range(plot_limit):
            fig_b, ax_b = plt.subplots(1, 1, figsize=(10, 10))
            for basename, traj_unnorm_full in all_trajs.items():
                traj_unnorm = traj_unnorm_full[start_idx:end_idx]
                true_start_unnorm = true_start_unnorm_all[start_idx:end_idx]
                current_color = color_map[basename]
                label = basename.replace("traj_", "").replace(".npy", "")
                ax_b.plot(traj_unnorm[b, :, x_idx], traj_unnorm[b, :, y_idx], color=current_color, alpha=0.8, linewidth=1.2, label=label)
                ax_b.plot(traj_unnorm[b, 0, x_idx], traj_unnorm[b, 0, y_idx], "go", markersize=4)
                ax_b.plot(traj_unnorm[b, -1, x_idx], traj_unnorm[b, -1, y_idx], "rx", markersize=5)
                
            ax_b.plot(true_start_unnorm[b, x_idx], true_start_unnorm[b, y_idx], "y*", markersize=12, label="True Start (Cond)")
            ax_b.set_xlim(ax_limits[0])
            ax_b.set_ylim(ax_limits[1])
            ax_b.set_title(f"Per-Batch Comparison: Init {init_idx} Batch {b}")
            utils.plot_environment_constraints(exp, ax_b)
            
            ax_b.legend(loc='upper right', fontsize='small')
            out_b = os.path.join(benchmark_dir, f"batch_comparison_init{init_idx}_B{b}.png" if n_init_points > 1 else f"batch_comparison_B{b}.png")
            fig_b.savefig(out_b, dpi=300, bbox_inches="tight")
            plt.close(fig_b)
            print(f"  [B{b}] Saved to {out_b}")

if __name__ == "__main__":
    main()
