#!/usr/bin/env python3
"""
Train iMeanFlow (iMF) trajectory models on D3IL avoiding-d3il dataset.

Engine: iMeanFlow (Improved Mean Flows)
- Dual-velocity decomposition: u (mean) + v (instantaneous)  
- Curriculum training: u_first schedule
- Official repo: github.com/Lyy-iiis/imeanflow

Evolution: DPCC → FMPCC → iMF-PCC (ML engine upgrade)

Usage:
    python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10 --use-wandb
"""

import os
import sys
import argparse
import pickle

import torch

# Standard FM-PCC imports
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'avoiding-d3il'
    config: str = 'config.avoiding-d3il'

DEFAULT_SEEDS = [6, 7, 8, 9, 10]


def sanitize_wandb_env():
    """Clear malformed W&B tokens."""
    service_env_keys = ('WANDB_SERVICE', 'WANDB__SERVICE')
    for env_key in service_env_keys:
        token = os.environ.get(env_key)
        if token and len(token.split('-')) != 5:
            print(f'[ train ] Clearing malformed {env_key}')
            os.environ.pop(env_key, None)


def log_wandb_from_losses(losses_path, run):
    """Reconstruct W&B logs from losses.pkl (standard FM-PCC pattern)."""
    if not os.path.exists(losses_path):
        return
    
    with open(losses_path, 'rb') as f:
        losses_data = pickle.load(f)
    
    training_losses = losses_data.get('training_losses', [])
    test_losses = losses_data.get('test_losses', [])
    test_by_step = {step: value for step, value in test_losses}
    
    for step, train_loss in training_losses:
        log_dict = {'train/loss': train_loss}
        if step in test_by_step:
            log_dict['test/loss'] = test_by_step[step]
        run.log(log_dict, step=step)


def upload_wandb_artifact(run, seed, savepath):
    """Upload checkpoint artifacts to W&B."""
    import wandb
    artifact = wandb.Artifact(
        name=f'imf-seed-{seed}-model',
        type='model',
        metadata={'seed': seed, 'savepath': savepath},
    )
    for filename in ['losses.pkl', 'args.json']:
        filepath = os.path.join(savepath, filename)
        if os.path.exists(filepath):
            artifact.add_file(filepath)
    run.log_artifact(artifact)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train iMF')
    parser.add_argument('--seed', type=int, help='Single seed.')
    parser.add_argument('--seeds', type=int, nargs='+', help='List of seeds.')
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument('--wandb-project', type=str, default='FMPCC-iMF', help='W&B project.')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity.')
    parser.add_argument('--wandb-group', type=str, default=None, help='W&B group.')
    args, remaining = parser.parse_known_args()
    return args, remaining


def resolve_seeds(cli_args):
    """Resolve which seeds to train."""
    if cli_args.seed is not None:
        return [cli_args.seed], 'cli --seed'
    elif cli_args.seeds is not None:
        return cli_args.seeds, 'cli --seeds'
    else:
        return DEFAULT_SEEDS, 'default'


if __name__ == '__main__':
    sanitize_wandb_env()
    
    cli_args, remaining = parse_args()
    seeds, seed_source = resolve_seeds(cli_args)
    
    print("=" * 80)
    print("[ train ] iMeanFlow Training (iMF-PCC)")
    print("[ train ] Engine: Improved Mean Flows (dual-velocity decomposition)")
    print("[ train ] Repo: github.com/Lyy-iiis/imeanflow")
    print(f"[ train ] Seeds: {seeds} ({seed_source})")
    print(f"[ train ] W&B: {cli_args.use_wandb} (project: {cli_args.wandb_project})")
    print("=" * 80)
    print()
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]] + remaining
    try:
        for seed in seeds:
            print(f"[ train ] Seed {seed}")

            # Parse config using standard FM-PCC Parser
            # Uses config/avoiding-d3il.py:flow_matching_v3_imeanflow
            parser = Parser(exe_name='train')
            args = parser.parse_args(experiment='flow_matching_v3_imeanflow', seed=seed)
            device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            predict_epsilon = getattr(args, 'predict_epsilon', True)

            # Build dataset, model, diffusion, trainer using the standard FM-PCC pattern.
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
            goal_dim = getattr(dataset, 'goal_dim', 0)
            transition_dim = observation_dim + action_dim

            model_config = utils.Config(
                args.model,
                savepath=(args.savepath, 'model_config.pkl'),
                state_dim=transition_dim,
                seq_len=args.horizon,
                freq_dim=args.freq_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_dim=args.mlp_dim,
                time_dim=args.time_dim,
                dropout_rate=args.dropout_rate,
                device=device,
            )

            diffusion_config = utils.Config(
                args.diffusion,
                savepath=(args.savepath, 'diffusion_config.pkl'),
                model=model_config,
                horizon=args.horizon,
                observation_dim=observation_dim,
                action_dim=action_dim,
                goal_dim=goal_dim,
                loss_type=args.loss_type,
                clip_denoised=args.clip_denoised,
                predict_epsilon=predict_epsilon,
                action_weight=args.action_weight,
                loss_discount=args.discount,
                returns_condition=args.include_returns,
                condition_guidance_w=args.condition_guidance_w,
                u_loss_weight=args.u_loss_weight,
                v_loss_weight=args.v_loss_weight,
                loss_schedule=args.loss_schedule,
                warmup_epochs=args.warmup_epochs,
                transition_epochs=args.transition_epochs,
                ode_inference_steps_v3=args.ode_inference_steps_v3,
            )

            trainer_config = utils.Config(
                utils.Trainer,
                savepath=(args.savepath, 'trainer_config.pkl'),
                train_test_split=getattr(args, 'train_test_split', 0.9),
                ema_decay=args.ema_decay,
                train_batch_size=args.batch_size,
                train_lr=args.learning_rate,
                gradient_accumulate_every=getattr(args, 'gradient_accumulate_every', 1),
                n_train_steps=args.n_train_steps,
                n_steps_per_epoch=getattr(args, 'n_steps_per_epoch', 1000),
                results_folder=args.savepath,
                train_device=device,
            )

            model = model_config()
            diffusion = diffusion_config(model)
            trainer = trainer_config(diffusion, dataset)

            # Setup W&B
            run = None
            if cli_args.use_wandb:
                try:
                    import wandb
                    run = wandb.init(
                        project=cli_args.wandb_project,
                        entity=cli_args.wandb_entity,
                        group=cli_args.wandb_group,
                        name=f'iMF-seed-{seed}',
                        config=vars(args),
                        reinit=True,
                    )
                except Exception as e:
                    print(f"[ train ] W&B init failed: {e}")

            # Train
            print(f"[ train ] Starting training (steps: {trainer.n_train_steps})")
            trainer.train()

            # Log to W&B
            if run is not None:
                log_wandb_from_losses(os.path.join(args.savepath, 'losses.pkl'), run)
                upload_wandb_artifact(run, seed, args.savepath)
                run.finish()

            print(f"[ train ] Seed {seed} complete → {args.savepath}")
            print()
    finally:
        sys.argv = original_argv
    
    print("=" * 80)
    print("[ train ] Training complete for all seeds")
    print("="* 80)

