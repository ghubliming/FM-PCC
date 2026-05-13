#!/usr/bin/env python3
"""
Evaluate iMeanFlow (iMF) models on D3IL avoiding-d3il validation dataset.

Standard FM-PCC evaluation pattern (inherited from FMv3ODE):
1. Load trained checkpoint
2. Run inference on validation split
3. Compute MSE error + visualize results

Usage:
    python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Standard FM-PCC imports
import diffuser.utils as utils
from diffuser.utils import Parser


def evaluate_seed(seed, results_dir='evaluation_results'):
    """
    Evaluate a single seed using standard FM-PCC pattern.
    
    Args:
        seed: Random seed
        results_dir: Output directory
        
    Returns:
        dict with evaluation metrics or None if failed
    """
    print(f"[ eval ] Seed {seed}")
    
    # Parse config to locate checkpoint using the standard FM-PCC handoff.
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        parser = Parser(exe_name='eval')
        args = parser.parse_args(experiment='eval', seed=seed)
    finally:
        sys.argv = original_argv
    
    print(f"[ eval ] Checkpoint: {args.savepath}")
    
    # Load trained diffusion model (standard FM-PCC function)
    try:
        diffusion_experiment = utils.load_diffusion(
            args.savepath,
            epoch='best',
            device=args.device,
        )
    except Exception as e:
        print(f"[ eval ] ERROR: Failed to load seed {seed}: {e}")
        return None
    
    model = diffusion_experiment.model
    diffusion = diffusion_experiment.diffusion
    dataset = diffusion_experiment.dataset
    
    # Get validation split indices
    split_idx = int(len(dataset) * 0.85)  # 85% train, 15% val
    val_indices = list(range(split_idx, len(dataset)))[:50]  # Up to 50 validation samples
    
    errors = []
    
    # Evaluate on validation split
    print(f"[ eval ] Evaluating on {len(val_indices)} validation samples...")
    
    for idx in val_indices:
        try:
            sample = dataset[idx]
            
            # Unpack trajectory (state, action, cond, mask)
            trajectory = sample[0]  # Full trajectory
            
            # Run inference: sample prediction
            with torch.no_grad():
                trajectory_tensor = torch.from_numpy(trajectory).float().unsqueeze(0).to(args.device)
                
                # iMF sample: single forward pass
                sampled = diffusion.sample(batch_size=1)
                
                # Compute MSE error
                error = torch.nn.functional.mse_loss(sampled, trajectory_tensor).item()
                errors.append(error)
        except Exception as e:
            print(f"[ eval ] Warning: Failed to evaluate sample {idx}: {e}")
    
    # Compute statistics
    if errors:
        results = {
            'seed': seed,
            'mse_error': float(np.mean(errors)),
            'mse_std': float(np.std(errors)),
            'num_samples': len(errors),
        }
    else:
        results = None
    
    if results:
        print(f"[ eval ] Seed {seed} | MSE: {results['mse_error']:.6f} ± {results['mse_std']:.6f}")
    
    return results


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(description='Evaluate iMF')
    parser.add_argument('--seed', type=int, help='Single seed.')
    parser.add_argument('--seeds', type=int, nargs='+', help='List of seeds.')
    parser.add_argument('--results-dir', type=str, default='evaluation_results', help='Output directory.')
    args, _ = parser.parse_known_args()
    
    # Resolve seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = [6, 7, 8, 9, 10]  # Default
    
    print("=" * 80)
    print("[ eval ] iMeanFlow Evaluation (iMF-PCC)")
    print(f"[ eval ] Seeds: {seeds}")
    print("=" * 80)
    print()
    
    os.makedirs(args.results_dir, exist_ok=True)
    all_results = {}
    
    # Evaluate each seed
    for seed in seeds:
        result = evaluate_seed(seed, args.results_dir)
        if result:
            all_results[seed] = result
        print()
    
    # Save results to JSON
    results_file = os.path.join(args.results_dir, 'eval_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 80)
    print(f"[ eval ] Results saved to: {results_file}")
    print(f"[ eval ] Evaluated {len(all_results)}/{len(seeds)} seeds successfully")
    print("=" * 80)


if __name__ == '__main__':
    main()
