#!/usr/bin/env python3
"""
iMeanFlow Evaluation Script

Evaluates trained dual-velocity models on D3IL tasks.
Tests multiple variants (NFE, solvers) and logs metrics.

Usage:
    python eval_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
import json

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flow_matcher_v3_imeanflow.models.imf_velocity import TimeConditionedDualVelocity
from flow_matcher_v3_imeanflow.sampling.imf_ode_solvers import (
    EulerSolver,
    RK4Solver,
    DopriSolver,
)
from flow_matcher_v3_imeanflow.sampling.imf_trajectory_sampler import (
    SingleStepSampler,
    DualStepSampler,
)
from flow_matcher_v3_imeanflow.utils.imf_metrics import (
    ImfMetricsTracker,
    compute_path_length,
    compute_smoothness,
)


class ImfEvaluator:
    """End-to-end evaluator for iMeanFlow models."""
    
    def __init__(
        self,
        device: str = 'cuda',
        state_dim: int = 28,
    ):
        self.device = device
        self.state_dim = state_dim
        self.results = {}
    
    def load_checkpoint(self, ckpt_path: Path) -> TimeConditionedDualVelocity:
        """Load model from checkpoint."""
        model = TimeConditionedDualVelocity(
            state_dim=self.state_dim,
            hidden_dim=256,
            time_dim=128,
            include_jvp=False,
        ).to(self.device)
        
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✓ Loaded checkpoint from {ckpt_path}")
        else:
            print(f"⚠ Checkpoint not found: {ckpt_path}. Using random initialization.")
        
        model.eval()
        return model
    
    def generate_evaluation_trajectories(
        self,
        num_trajectories: int = 50,
        seq_length: int = 20,
    ) -> torch.Tensor:
        """Generate synthetic evaluation trajectories."""
        trajectories = []
        
        for _ in range(num_trajectories):
            start_pos = torch.randn(self.state_dim, device=self.device) * 0.5
            traj = [start_pos.clone()]
            vel = torch.randn(self.state_dim, device=self.device) * 0.1
            
            for t in range(seq_length - 1):
                vel = 0.8 * vel + torch.randn(self.state_dim, device=self.device) * 0.05
                pos = start_pos + vel * 0.1
                traj.append(pos.clone())
            
            traj = torch.stack(traj)
            trajectories.append(traj)
        
        return torch.stack(trajectories)
    
    def evaluate_variant(
        self,
        model: TimeConditionedDualVelocity,
        trajectories: torch.Tensor,
        solver_name: str = 'euler',
        nfe: int = 1,
        variant_name: str = None,
    ) -> dict:
        """Evaluate a specific model variant."""
        variant_name = variant_name or f'{solver_name}_nfe{nfe}'
        
        # Initialize solver
        if solver_name == 'euler':
            solver = EulerSolver()
        elif solver_name == 'rk4':
            solver = RK4Solver()
        elif solver_name == 'dopri5':
            solver = DopriSolver()
        else:
            raise ValueError(f"Unknown solver: {solver_name}")
        
        # Initialize sampler
        if nfe == 1:
            sampler = SingleStepSampler(model=model, solver=solver)
        elif nfe == 2:
            sampler = DualStepSampler(model=model, solver=solver)
        else:
            raise ValueError(f"NFE must be 1 or 2, got {nfe}")
        
        metrics_tracker = ImfMetricsTracker()
        sampled_trajectories = []
        
        print(f"  Evaluating {variant_name}...")
        
        for idx in tqdm(range(len(trajectories)), leave=False, desc=variant_name):
            traj_true = trajectories[idx]
            start_pos = traj_true[0]
            
            # Sample trajectory
            with torch.no_grad():
                if nfe == 1:
                    traj_pred, u_trajectory, v_trajectory = sampler.sample(
                        x_0=start_pos,
                        num_steps=traj_true.shape[0],
                    )
                else:
                    traj_pred, u_trajectory, v_trajectory = sampler.sample(
                        x_0=start_pos,
                        num_steps=traj_true.shape[0],
                    )
            
            sampled_trajectories.append({
                'trajectory': traj_pred.cpu().numpy(),
                'u_trajectory': u_trajectory.cpu().numpy() if u_trajectory is not None else None,
                'v_trajectory': v_trajectory.cpu().numpy() if v_trajectory is not None else None,
            })
            
            # Compute metrics
            # Trajectory matching error
            if traj_pred.shape == traj_true.shape:
                traj_error = torch.norm(traj_pred - traj_true) / torch.norm(traj_true)
                metrics_tracker.compute_trajectory_error(traj_pred, traj_true)
            
            # Path-based metrics
            path_length = compute_path_length(traj_pred)
            smoothness = compute_smoothness(traj_pred)
            
            metrics_tracker.trajectory_length_list.append(path_length.item())
            metrics_tracker.smoothness_list.append(smoothness.item())
        
        # Aggregate metrics
        summary = metrics_tracker.get_summary()
        summary['variant'] = variant_name
        summary['sampled_trajectories'] = sampled_trajectories
        
        return summary
    
    def evaluate_seed(
        self,
        seed: int,
        checkpoint_dir: Path = None,
        solvers: list = None,
        nfe_values: list = None,
    ) -> dict:
        """Evaluate all variants for a seed."""
        solvers = solvers or ['euler', 'rk4', 'dopri5']
        nfe_values = nfe_values or [1, 2]
        
        # Setup checkpoint path
        if checkpoint_dir is None:
            checkpoint_dir = Path(f'checkpoints_seed{seed}')
        
        ckpt_path = checkpoint_dir / 'state_best.pt'
        
        # Load model
        model = self.load_checkpoint(ckpt_path)
        
        # Generate eval data
        print(f"  Generating evaluation trajectories...")
        eval_trajectories = self.generate_evaluation_trajectories(
            num_trajectories=50,
            seq_length=20,
        )
        
        # Evaluate all variants
        results = {'seed': seed}
        
        for solver_name in solvers:
            for nfe in nfe_values:
                variant_name = f'{solver_name}_nfe{nfe}'
                print(f"\nSeed {seed}: {variant_name}")
                
                variant_result = self.evaluate_variant(
                    model=model,
                    trajectories=eval_trajectories,
                    solver_name=solver_name,
                    nfe=nfe,
                    variant_name=variant_name,
                )
                
                results[variant_name] = variant_result
                
                # Print summary
                print(f"    Trajectory Error: {variant_result.get('trajectory_error', np.nan):.4f}")
                print(f"    Path Length (mean): {np.mean(variant_result.get('trajectory_length_list', [])):.4f}")
                print(f"    Smoothness (mean): {np.mean(variant_result.get('smoothness_list', [])):.4f}")
        
        return results
    
    def save_results(self, results: dict, output_path: Path):
        """Save results to .npz file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_dict = {'seed': results['seed']}
        
        for variant_key, variant_data in results.items():
            if variant_key == 'seed':
                continue
            
            variant_results = {}
            
            # Save scalars
            for key in ['trajectory_error', 'u_error', 'v_error']:
                if key in variant_data:
                    variant_results[f'{variant_key}_{key}'] = variant_data[key]
            
            # Save lists (convert to arrays)
            for key in ['trajectory_length_list', 'smoothness_list', 'u_error_list', 'v_error_list']:
                if key in variant_data:
                    variant_results[f'{variant_key}_{key}'] = np.array(variant_data[key])
            
            save_dict.update(variant_results)
        
        np.savez(output_path, **save_dict)
        print(f"✓ Results saved to {output_path}")
    
    def evaluate(
        self,
        seeds: list,
        checkpoint_base_dir: Path = None,
        output_dir: Path = None,
    ):
        """Run full evaluation."""
        if checkpoint_base_dir is None:
            checkpoint_base_dir = Path('checkpoints')
        
        if output_dir is None:
            output_dir = Path('evaluation_results')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for seed in seeds:
            print(f"\n{'='*80}")
            print(f"Evaluating Seed {seed}")
            print(f"{'='*80}\n")
            
            seed_checkpoint_dir = checkpoint_base_dir / f'seed_{seed}'
            results = self.evaluate_seed(
                seed=seed,
                checkpoint_dir=seed_checkpoint_dir,
            )
            
            all_results[seed] = results
            
            # Save per-seed results
            seed_output = output_dir / f'results_seed_{seed}.npz'
            self.save_results(results, seed_output)
        
        # Save aggregate results as JSON
        aggregate_path = output_dir / 'aggregate_results.json'
        self._save_aggregate_results(all_results, aggregate_path)
        
        return all_results
    
    def _save_aggregate_results(self, all_results: dict, output_path: Path):
        """Save aggregated results across seeds."""
        # Summarize across seeds
        summary = {}
        
        if not all_results:
            return
        
        # Get all variants from first seed
        first_seed_results = next(iter(all_results.values()))
        variants = [k for k in first_seed_results.keys() if k != 'seed']
        
        for variant in variants:
            variant_scores = []
            
            for seed, results in all_results.items():
                if variant in results:
                    score = results[variant].get('trajectory_error', np.nan)
                    variant_scores.append(score)
            
            if variant_scores:
                summary[variant] = {
                    'mean': float(np.nanmean(variant_scores)),
                    'std': float(np.nanstd(variant_scores)),
                    'scores': variant_scores,
                }
        
        # Save as JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Aggregate results saved to {output_path}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate iMF models')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='Seeds to evaluate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Base checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--solvers', nargs='+', default=['euler', 'rk4', 'dopri5'],
                       help='Solvers to test')
    parser.add_argument('--nfe-values', nargs='+', type=int, default=[1, 2],
                       help='NFE values to test')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("iMeanFlow Evaluation")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Seeds: {args.seeds}")
    print(f"Solvers: {args.solvers}")
    print(f"NFE values: {args.nfe_values}")
    print()
    
    evaluator = ImfEvaluator(device=args.device)
    results = evaluator.evaluate(
        seeds=args.seeds,
        checkpoint_base_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
    )
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
