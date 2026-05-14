# FM_v3 ODE-selectable version of train_FM.py for Vision
import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime

import torch
import wandb
import ddpm_encdec_vision.utils as utils

exp = 'aligning-d3il-visual'
DEFAULT_SEEDS = [5, 6, 7, 8, 9]

def sanitize_wandb_env():
    """Clear malformed W&B service tokens."""
    service_env_keys = ('WANDB_SERVICE', 'WANDB__SERVICE')
    for env_key in service_env_keys:
        token = os.environ.get(env_key)
        if token is not None and len(token.split('-')) != 5:
            os.environ.pop(env_key, None)

def log_wandb_curves_from_losses(losses_path, run):
    if not os.path.exists(losses_path): return
    with open(losses_path, 'rb') as f:
        losses = pickle.load(f)
    for step, train_loss in losses.get('training_losses', []):
        log_dict = {'train/loss': train_loss}
        run.log(log_dict, step=step)

def upload_wandb_artifact(run, seed, args):
    artifact = wandb.Artifact(name=f'{args.dataset}-seed-{seed}-model', type='model')
    for filename in ['losses.pkl', 'args.json']:
        filepath = os.path.join(args.savepath, filename)
        if os.path.exists(filepath): artifact.add_file(filepath)
    run.log_artifact(artifact)

class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

def parse_top_level_args():
    parser = argparse.ArgumentParser(description='Train FM Vision model.')
    parser.add_argument('--seed', type=int, help='Train a single seed.')
    parser.add_argument('--auto-resume', action='store_true', help='Auto-resume latest checkpoint.')
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B.')
    args, remaining = parser.parse_known_args()
    return args, remaining

def resolve_seed_list(cli_args):
    if cli_args.seed is not None: return [cli_args.seed], 'cli'
    return list(DEFAULT_SEEDS), 'default'

cli_args, parser_remaining = parse_top_level_args()
selected_seeds, seed_source = resolve_seed_list(cli_args)
sys.argv = [sys.argv[0], *parser_remaining]

for seed in selected_seeds:
    args = Parser().parse_args(experiment='ddpm_encdec_vision', seed=seed)
    torch.manual_seed(args.seed)
    
    # --- FM-PCC Bone Replication (Multi-stage Config) ---
    
    # 1. Dataset
    from d3il.environments.dataset.aligning_dataset import Aligning_Img_Dataset
    dataset_config = utils.Config(
        Aligning_Img_Dataset,
        savepath=(args.savepath, 'dataset_config.pkl'),
        data_directory='environments/dataset/data/aligning/train_files.pkl',
        device='cpu', # Keep on CPU to avoid CUDA fork issues
        obs_dim=3,
        action_dim=3,
        window_size=args.horizon,
        max_len_data=256
    )
    dataset = dataset_config()
    
    # 2. Model (Backbone + Vision Encoder)
    from ddpm_encdec_vision.models.visual_unet import VisualUNet
    model_config = utils.Config(
        VisualUNet,
        savepath=(args.savepath, 'model_config.pkl'),
        config=args,
    )
    model = model_config()
    
    # 3. Diffusion (Engine)
    from ddpm_encdec_vision.models.visual_gaussian_diffusion import VisualGaussianDiffusion
    diffusion_config = utils.Config(
        VisualGaussianDiffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        model=model,
        horizon=args.horizon,
        observation_dim=3,
        action_dim=3,
        goal_dim=0,
        n_timesteps=getattr(args, 'n_diffusion_steps', 20),
        loss_type=args.loss_type,
        action_weight=getattr(args, 'action_weight', 1.0),
        time_beta_alpha_v3=getattr(args, 'time_beta_alpha_v3', 1.5),
        time_beta_beta_v3=getattr(args, 'time_beta_beta_v3', 1.0),
        flow_steps_v3=getattr(args, 'flow_steps_v3', 10),
        device=args.device,
    )
    diffusion = diffusion_config(model)
    
    # 4. Trainer
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
        train_device=args.device,
    )
    trainer = trainer_config(diffusion, dataset)
    
    # --- W&B Setup ---
    run = None
    if cli_args.use_wandb:
        run = wandb.init(project='fm-pcc-vision', name=f'seed-{seed}', config=vars(args))
    
    # --- Train ---
    trainer.train()
    
    if run:
        log_wandb_curves_from_losses(os.path.join(args.savepath, 'losses.pkl'), run)
        run.finish()

print('Vision FM_v3 training completed.')