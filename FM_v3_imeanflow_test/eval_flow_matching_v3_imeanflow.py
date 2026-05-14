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
from pathlib import Path

import numpy as np
import torch

# Standard FM-PCC imports
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'avoiding-d3il'
    config: str = 'config.avoiding-d3il'


def load_diffusion_robust(checkpoint_dir, epoch='best', device='cuda'):
    """Load diffusion checkpoint with legacy-config compatibility in a single pass."""
    print(f'\n[ utils/serialization ] Loading model from {checkpoint_dir}\n')

    dataset_config = utils.load_config(checkpoint_dir, 'dataset_config.pkl')
    model_config = utils.load_config(checkpoint_dir, 'model_config.pkl')
    diffusion_config = utils.load_config(checkpoint_dir, 'diffusion_config.pkl')
    trainer_config = utils.load_config(checkpoint_dir, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = checkpoint_dir

    # Some old checkpoints serialized `model` inside diffusion kwargs,
    # which collides with the positional `model` argument at instantiation.
    if isinstance(getattr(diffusion_config, '_dict', None), dict) and 'model' in diffusion_config._dict:
        print('[ eval ] INFO: Removing legacy diffusion_config["model"] compatibility key.')
        diffusion_config._dict.pop('model', None)

    dataset = dataset_config()
    model = model_config().to(device)
    diffusion = diffusion_config(model).to(device)
    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)

    if epoch == 'latest':
        epoch = utils.get_latest_epoch((checkpoint_dir,))

    trainer.load(epoch)
    losses = utils.load_losses(checkpoint_dir, 'losses.pkl')

    return utils.DiffusionExperiment(
        dataset,
        trainer.model.model,
        trainer.model,
        trainer,
        epoch,
        losses,
    )


def resolve_checkpoint_dir(seed, experiment='flow_matching_v3_imeanflow', checkpoint_root=None):
    """Resolve a seed checkpoint directory from config experiment or explicit root."""
    if checkpoint_root:
        return os.path.join(checkpoint_root, str(seed))

    # Parse config to locate checkpoint using the standard FM-PCC handoff.
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        parser = Parser(exe_name='eval')
        args = parser.parse_args(experiment=experiment, seed=seed)
    finally:
        sys.argv = original_argv

    return args.savepath


def evaluate_seed(seed, results_dir='evaluation_results', experiment='flow_matching_v3_imeanflow', checkpoint_root=None):
    """
    Evaluate a single seed using standard FM-PCC pattern.
    
    Args:
        seed: Random seed
        results_dir: Output directory
        
    Returns:
        dict with evaluation metrics or None if failed
    """
    print(f"[ eval ] Seed {seed}")
    
    checkpoint_dir = resolve_checkpoint_dir(
        seed=seed,
        experiment=experiment,
        checkpoint_root=checkpoint_root,
    )
    print(f"[ eval ] Checkpoint: {checkpoint_dir}")

    missing = [
        name
        for name in ('dataset_config.pkl', 'model_config.pkl', 'diffusion_config.pkl', 'trainer_config.pkl')
        if not os.path.exists(os.path.join(checkpoint_dir, name))
    ]
    if missing:
        print(
            f"[ eval ] ERROR: Missing checkpoint files for seed {seed}: {missing} | "
            f"path={checkpoint_dir}"
        )
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained diffusion model (with legacy pickle compatibility)
    try:
        diffusion_experiment = load_diffusion_robust(checkpoint_dir, epoch='best', device=device)
    except Exception as e:
        print(f"[ eval ] ERROR: Failed to load seed {seed}: {e}")
        return None
    
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

            # Namedtuple from SequenceDataset: trajectories, conditions, (optional) returns
            trajectory = sample.trajectories
            conditions = sample.conditions
            returns = getattr(sample, 'returns', None)
            
            # Run inference: sample prediction
            with torch.no_grad():
                trajectory_tensor = torch.from_numpy(trajectory).float().unsqueeze(0).to(device)
                
                # Convert conditions to batched tensors for sampler compatibility
                cond_tensors = {}
                for t, val in conditions.items():
                    cond_tensor = torch.as_tensor(val, dtype=torch.float32, device=device)
                    if cond_tensor.ndim == 1:
                        cond_tensor = cond_tensor.unsqueeze(0)
                    cond_tensors[t] = cond_tensor

                returns_tensor = None
                if returns is not None:
                    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
                    if returns_tensor.ndim == 1:
                        returns_tensor = returns_tensor.unsqueeze(0)

                # iMF sample conditioned on current observation (and optional return)
                sampled = diffusion.sample(
                    batch_size=1,
                    conditions=cond_tensors,
                    returns=returns_tensor,
                )
                
                # Compute MSE error
                error = torch.nn.functional.mse_loss(sampled.to(device), trajectory_tensor).item()
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


def resolve_results_dir(explicit_results_dir, checkpoint_dir):
    """Resolve the evaluation output directory next to the experiment logs."""
    if explicit_results_dir is not None:
        return str(Path(explicit_results_dir).expanduser())

    checkpoint_path = Path(checkpoint_dir)
    return str(checkpoint_path.parent / 'evaluation_results' / 'imf')


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(description='Evaluate iMF')
    parser.add_argument('--seed', type=int, help='Single seed.')
    parser.add_argument('--seeds', type=int, nargs='+', help='List of seeds.')
    parser.add_argument('--results-dir', type=str, default=None, help='Output directory. Defaults to <experiment-root>/evaluation_results/imf.')
    parser.add_argument(
        '--experiment',
        type=str,
        default='flow_matching_v3_imeanflow',
        help='Config experiment key used to resolve checkpoint savepath.',
    )
    parser.add_argument(
        '--checkpoint-root',
        type=str,
        default=None,
        help='Optional explicit checkpoint root. If set, eval loads <checkpoint-root>/<seed>.',
    )
    args, _ = parser.parse_known_args()

    if args.checkpoint_root is not None:
        args.checkpoint_root = str(Path(args.checkpoint_root).expanduser())
    
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

    if args.checkpoint_root is not None:
        base_checkpoint_dir = Path(args.checkpoint_root).expanduser()
        if args.seed is not None:
            base_checkpoint_dir = base_checkpoint_dir / str(args.seed)
    else:
        base_checkpoint_dir = Path(resolve_checkpoint_dir(seeds[0], experiment=args.experiment))

    results_dir = resolve_results_dir(args.results_dir, base_checkpoint_dir)
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}
    
    # Evaluate each seed
    for seed in seeds:
        result = evaluate_seed(
            seed,
            results_dir,
            experiment=args.experiment,
            checkpoint_root=args.checkpoint_root,
        )
        if result:
            all_results[seed] = result
        print()
    
    # Save results to JSON
    results_file = os.path.join(results_dir, 'eval_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 80)
    print(f"[ eval ] Results saved to: {results_file}")
    print(f"[ eval ] Evaluated {len(all_results)}/{len(seeds)} seeds successfully")
    print("=" * 80)


if __name__ == '__main__':
    main()
