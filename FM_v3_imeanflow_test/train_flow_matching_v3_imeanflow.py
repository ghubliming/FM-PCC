#!/usr/bin/env python3
"""
iMeanFlow (Improved Mean Flows) Training Script

Standardized training script for Improved Mean Flows models.
Aligned with the FMv3-ODE pipeline ground truth.

Usage:
    python train_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10 --use-wandb
"""

import os
import sys
import glob
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import flow_matcher_v3_imeanflow.utils as utils

# Try to import W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

class Parser(utils.Parser):
    dataset: str = 'avoiding-d3il'
    config: str = 'config.avoiding-d3il'
    
    # Training Overrides
    batch_size: int = 32
    learning_rate: float = 5e-4
    n_train_steps: int = 100000
    device: str = 'cuda'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add support for the specific arguments found in the user's cluster scripts
        self.add_argument('--batch-size', type=int, help='Batch size')
        self.add_argument('--learning-rate', type=float, help='Learning rate')
        self.add_argument('--num-epochs', type=int, help='Number of epochs (maps to n_train_steps)')
        self.add_argument('--device', type=str, help='Device')

def parse_top_level_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--num-seeds', type=int, default=1)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', default='FMPCC-iMF')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-group', default=None)
    parser.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--resume-seed', type=int, default=None)
    parser.add_argument('--resume-step', type=int, default=None)
    parser.add_argument('--auto-resume', action='store_true')
    return parser.parse_known_args()

def resolve_seed_list(cli_args):
    if cli_args.seeds is not None:
        return cli_args.seeds, "manual_list"
    
    base_seeds = [6, 7, 8, 9, 10]
    if cli_args.num_seeds > len(base_seeds):
        selected = base_seeds + list(range(11, 11 + cli_args.num_seeds - len(base_seeds)))
        return selected, "extended_range"
    
    return base_seeds[:cli_args.num_seeds], "default_subset"

def sanitize_wandb_env():
    for key in list(os.environ.keys()):
        if key.startswith('WANDB_'):
            del os.environ[key]

def find_latest_checkpoint_step(results_dir):
    pattern = os.path.join(results_dir, 'state_*.pt')
    steps = []
    for checkpoint_path in glob.glob(pattern):
        filename = os.path.basename(checkpoint_path)
        step_token = filename.replace('state_', '').replace('.pt', '')
        try:
            steps.append(int(step_token))
        except ValueError:
            continue
    return max(steps) if len(steps) > 0 else None

def main():
    cli_args, parser_remaining = parse_top_level_args()
    selected_seeds, seed_source = resolve_seed_list(cli_args)
    
    print("=" * 80)
    print(f"[ train ] iMeanFlow Training (Real Data: {Parser.dataset})")
    print(f"[ train ] Seeds: {selected_seeds} ({seed_source})")
    print("=" * 80)
    
    # Save original argv to restore for each seed's Parser
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0], *parser_remaining]
    
    for seed in selected_seeds:
        # Each call to parse_args sets the seed and creates the savepath
        parser = Parser()
        args = parser.parse_args(experiment='flow_matching_v3_imeanflow', seed=seed)
        
        # Handle the num-epochs -> n_train_steps mapping if provided
        if hasattr(args, 'num_epochs') and args.num_epochs is not None:
            # Simple heuristic: 1 epoch = n_steps_per_epoch
            args.n_train_steps = args.num_epochs * args.n_steps_per_epoch
            
        torch.manual_seed(args.seed)
        
        run = None
        if cli_args.use_wandb and cli_args.wandb_mode != 'disabled':
            sanitize_wandb_env()
            savepath_rel = os.path.relpath(args.savepath, args.logbase)
            wandb_name = savepath_rel.replace('/', '-').replace('flow_matcher_v3_imeanflow.models.', '').replace('models.', '')
            
            # Group seeds
            name_parts = wandb_name.split('-')
            if name_parts[-1].isdigit():
                name_parts[-1] = f'S{name_parts[-1]}'
            wandb_name = '-'.join(name_parts)
            default_group = '-'.join(name_parts[:-1]) if len(name_parts) > 1 else wandb_name
            wandb_group = cli_args.wandb_group if cli_args.wandb_group is not None else default_group
            
            run = wandb.init(
                project=cli_args.wandb_project,
                entity=cli_args.wandb_entity,
                group=wandb_group,
                name=wandb_name,
                mode=cli_args.wandb_mode,
                config={**vars(args), 'seed': seed}
            )
            
        # Get dataset
        dataset_config = utils.Config(
            args.loader,
            savepath=(args.savepath, 'dataset_config.pkl'),
            env=args.dataset,
            horizon=args.horizon,
            normalizer=args.normalizer,
            preprocess_fns=args.preprocess_fns,
            use_padding=args.use_padding,
            max_path_length=args.max_path_length,
            include_returns=args.include_returns,
            returns_scale=args.returns_scale,
            discount=args.discount,
        )
        dataset = dataset_config()
        observation_dim = dataset.observation_dim
        action_dim = dataset.action_dim
        
        # Model & Trainer
        model_config = utils.Config(
            args.model,
            savepath=(args.savepath, 'model_config.pkl'),
            state_dim=observation_dim + action_dim,
            time_dim=args.time_dim,
            hidden_dim=args.hidden_dim,
            include_jvp=False, # Standard training usually doesn't need JVP in forward
        )
        
        diffusion_config = utils.Config(
            args.diffusion,
            savepath=(args.savepath, 'diffusion_config.pkl'),
            horizon=args.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            goal_dim=dataset.goal_dim,
            u_loss_weight=args.u_loss_weight,
            v_loss_weight=args.v_loss_weight,
            jvp_weight=args.jvp_weight,
            loss_type=args.loss_type,
            loss_schedule=args.loss_schedule,
            n_train_steps=args.n_train_steps,
            device=args.device,
        )
        
        trainer_config = utils.Config(
            utils.Trainer,
            savepath=(args.savepath, 'trainer_config.pkl'),
            train_test_split=args.train_test_split,
            ema_decay=args.ema_decay,
            n_train_steps=args.n_train_steps,
            n_steps_per_epoch=args.n_steps_per_epoch,
            train_batch_size=args.batch_size,
            train_lr=args.learning_rate,
            gradient_accumulate_every=args.gradient_accumulate_every,
            results_folder=args.savepath,
        )
        
        model = model_config()
        diffusion = diffusion_config(model)
        trainer = trainer_config(diffusion, dataset)
        
        # Resume logic
        resume_step = None
        if cli_args.auto_resume:
            resume_step = find_latest_checkpoint_step(args.savepath)
        if cli_args.resume_seed == seed and cli_args.resume_step is not None:
            resume_step = cli_args.resume_step
            
        if resume_step is not None:
            print(f'[ train ] Resuming seed {seed} from step {resume_step}')
            trainer.load(resume_step)
            
        trainer.train()
        
        if run:
            run.finish()
            
    print("\n[ train ] All training jobs complete.")

if __name__ == '__main__':
    main()
