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
    """Clear malformed W&B service tokens that can crash wandb.init in Colab."""
    service_env_keys = ('WANDB_SERVICE', 'WANDB__SERVICE')

    for env_key in service_env_keys:
        token = os.environ.get(env_key)
        if token is None:
            continue

        # W&B expects exactly 5 '-' separated token parts.
        if len(token.split('-')) != 5:
            print(f'[ train ] Clearing malformed {env_key}: {token}')
            os.environ.pop(env_key, None)

def log_wandb_curves_from_losses(losses_path, run):
    if not os.path.exists(losses_path):
        return

    with open(losses_path, 'rb') as f:
        losses = pickle.load(f)

    training_losses = losses.get('training_losses', [])
    test_losses = losses.get('test_losses', [])
    training_a0_losses = losses.get('training_a0_losses', [])
    test_a0_losses = losses.get('test_a0_losses', [])

    test_by_step = {step: value for step, value in test_losses}
    train_a0_by_step = {step: value for step, value in training_a0_losses}
    test_a0_by_step = {step: value for step, value in test_a0_losses}

    for step, train_loss in training_losses:
        log_dict = {'train/loss': train_loss}
        if step in test_by_step:
            log_dict['test/loss'] = test_by_step[step]
        if step in train_a0_by_step:
            log_dict['train/a0_loss'] = train_a0_by_step[step]
        if step in test_a0_by_step:
            log_dict['test/a0_loss'] = test_a0_by_step[step]
        run.log(log_dict, step=step)

    if len(training_losses) > 0:
        run.summary['final_train_loss'] = training_losses[-1][1]
    if len(test_losses) > 0:
        run.summary['final_test_loss'] = test_losses[-1][1]

def upload_wandb_artifact(run, seed, args):
    artifact = wandb.Artifact(
        name=f'{args.dataset}-seed-{seed}-model',
        type='model',
        metadata={
            'dataset': args.dataset,
            'seed': seed,
            'savepath': args.savepath,
            'n_train_steps': args.n_train_steps,
        },
    )

    files_to_add = ['losses.pkl', 'args.json']
    for filename in files_to_add:
        filepath = os.path.join(args.savepath, filename)
        if os.path.exists(filepath):
            artifact.add_file(filepath)

    run.log_artifact(artifact)


class Parser(utils.Parser):
    dataset: str = exp
    config: str = 'config.' + exp

def parse_top_level_args():
    parser = argparse.ArgumentParser(description='Train FM Vision model with configurable seeds.')
    parser.add_argument('--seed', type=int, help='Train a single seed.')
    parser.add_argument('--seeds', type=int, nargs='+', help='Train an explicit list of seeds, e.g. --seeds 5 6 7.')
    parser.add_argument('--seeds-from-config', type=str, help='Path to JSON file with `seed_list` or `seeds`.')
    parser.add_argument('--num-seeds', type=int, help='Train only the first N seeds from the resolved seed list.')
    parser.add_argument('--resume-seed', type=int, help='Seed for manual resume step loading.')
    parser.add_argument('--resume-step', type=int, help='Checkpoint step to resume from, e.g. 80000.')
    parser.add_argument('--auto-resume', action='store_true', help='Auto-resume each seed from latest local checkpoint if present.')
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B runs per seed.')
    parser.add_argument('--wandb-project', type=str, default='fm-pcc-flow-matching', help='W&B project name.')
    parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity/team name.')
    parser.add_argument('--wandb-group', type=str, default=None, help='W&B group name for per-seed runs.')
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'], help='W&B mode.')
    parser.add_argument('--log-freq', type=int, default=1000, help='How often to log progress to console/disk.')
    args, remaining = parser.parse_known_args()
    if args.seed is not None and args.seeds is not None:
        raise ValueError('Use either --seed or --seeds, not both.')
    if args.num_seeds is not None and args.num_seeds <= 0:
        raise ValueError('--num-seeds must be > 0.')
    if args.resume_step is not None and args.resume_step < 0:
        raise ValueError('--resume-step must be >= 0.')
    return args, remaining

def load_seeds_from_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        if 'seed_list' in payload:
            seeds = payload['seed_list']
        elif 'seeds' in payload:
            seeds = payload['seeds']
        else:
            raise ValueError(f'No `seed_list` or `seeds` found in {path}.')
    elif isinstance(payload, list):
        seeds = payload
    else:
        raise ValueError(f'Unsupported seed config format in {path}.')
    return [int(seed) for seed in seeds]

def resolve_seed_list(cli_args):
    if cli_args.seed is not None:
        seeds = [cli_args.seed]
        source = 'cli --seed'
    elif cli_args.seeds is not None:
        seeds = [int(seed) for seed in cli_args.seeds]
        source = 'cli --seeds'
    elif cli_args.seeds_from_config is not None:
        seeds = load_seeds_from_config(cli_args.seeds_from_config)
        source = f'config {cli_args.seeds_from_config}'
    else:
        seeds = list(DEFAULT_SEEDS)
        source = 'default'
    if cli_args.num_seeds is not None:
        seeds = seeds[:cli_args.num_seeds]
    if len(seeds) == 0:
        raise ValueError('Resolved seed list is empty.')
    return seeds, source

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

def write_seed_manifest(run_root, selected_seeds, seed_source, cli_args):
    manifest_path = os.path.join(run_root, 'seeds_config.json')
    payload = {
        'generation_date': datetime.utcnow().isoformat() + 'Z',
        'total_seeds': len(selected_seeds),
        'seed_list': selected_seeds,
        'seed_source': seed_source,
        'num_seeds_applied': cli_args.num_seeds,
        'resume_seed': cli_args.resume_seed,
        'resume_step': cli_args.resume_step,
        'auto_resume': cli_args.auto_resume,
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f'[ train ] Saved seed manifest to: {manifest_path}')

def should_apply_manual_resume(seed, selected_seeds, cli_args):
    if cli_args.resume_step is None:
        return False
    if cli_args.resume_seed is not None:
        return seed == cli_args.resume_seed
    return seed == selected_seeds[0]

cli_args, parser_remaining = parse_top_level_args()
selected_seeds, seed_source = resolve_seed_list(cli_args)
print(f'[ train ] Seed source: {seed_source}')
print(f'[ train ] Training seeds: {selected_seeds}')
original_argv = list(sys.argv)
sys.argv = [sys.argv[0], *parser_remaining]
manifest_written = False

for seed in selected_seeds:
    args = Parser().parse_args(experiment='ddpm_encdec_vision', seed=seed)
    torch.manual_seed(args.seed)
    if not manifest_written:
        run_root = os.path.dirname(args.savepath)
        write_seed_manifest(run_root, selected_seeds, seed_source, cli_args)
        manifest_written = True
    
    run = None
    if cli_args.use_wandb and cli_args.wandb_mode != 'disabled':
        sanitize_wandb_env()
        # Build run name exactly like FMv3ODE
        name_parts = [exp, args.exp_name, f'S{seed}']
        wandb_name = '-'.join(name_parts)
        default_group = '-'.join(name_parts[:-1]) if len(name_parts) > 1 else wandb_name
        wandb_group = cli_args.wandb_group if cli_args.wandb_group is not None else default_group

        run = wandb.init(
            project=cli_args.wandb_project,
            entity=cli_args.wandb_entity,
            group=wandb_group,
            name=wandb_name,
            mode=cli_args.wandb_mode,
            config={
                **vars(args),
                'selected_seeds': selected_seeds,
                'seed_source': seed_source,
            },
        )
        
    # --- FM-PCC Bone: Multi-stage Config Instantiation ---
    
    # 1. Dataset
    from d3il.environments.dataset.aligning_dataset import Aligning_Img_Dataset
    dataset_config = utils.Config(
        Aligning_Img_Dataset,
        savepath=(args.savepath, 'dataset_config.pkl'),
        data_directory='environments/dataset/data/aligning/train_files.pkl',
        device='cpu',
        obs_dim=3,
        action_dim=3,
        window_size=args.horizon,
        max_len_data=256
    )
    dataset = dataset_config()
    
    # --- FIX_17: Scaler Initialization ---
    from ddpm_encdec_vision.utils.scaler import Scaler
    # Initialize Scaler with MASKED data (FIX #17 - Prevent zero-padding corruption)
    print(f"[ train ] Calculating dataset statistics (ignoring zero-padding)...")
    all_obs = dataset.get_all_observations()
    all_act = dataset.get_all_actions()
    
    scaler = Scaler(all_obs, all_act, scale_data=True, device=args.device)
    print(f"[ train ] Dataset Stats: Obs Mean {scaler.x_mean.mean().item():.4f}, Act Std {scaler.y_std.mean().item():.4f}")
    import pickle
    scaler_path = os.path.join(args.savepath, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'[ train ] Saved scaler to {scaler_path}')
    
    # 2. Model (Backbone + Vision Encoder)
    from ddpm_encdec_vision.models.visual_unet import VisualUNet
    model_config = utils.Config(
        VisualUNet,
        savepath=(args.savepath, 'model_config.pkl'),
        config=args,
    )
    model = model_config()
    
    # 3. Diffusion (Engine)
    # 3. Diffusion (Engine: Reverted to standard DDPM)
    from ddpm_encdec_vision.models.visual_gaussian_diffusion import VisualGaussianDiffusion
    diffusion_config = utils.Config(
        VisualGaussianDiffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=3,
        action_dim=3,
        goal_dim=0,
        n_timesteps=getattr(args, 'n_diffusion_steps', 100),
        loss_type=args.loss_type,
        clip_denoised=True,
        predict_epsilon=True,
        action_weight=getattr(args, 'action_weight', 10.0),
        device=args.device,
    )
    diffusion = diffusion_config(model)
    
    # 4. Trainer
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        scaler=scaler,
        train_test_split=args.train_test_split,
        ema_decay=args.ema_decay,
        n_train_steps=args.n_train_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        results_folder=args.savepath,
        train_device=args.device,
        log_freq=cli_args.log_freq,
    )
    trainer = trainer_config(diffusion, dataset)
    
    # --- Resume Logic (Replicated from FMv3ODE) ---
    resume_step = None
    if cli_args.auto_resume:
        resume_step = find_latest_checkpoint_step(args.savepath)
    if should_apply_manual_resume(seed, selected_seeds, cli_args):
        resume_step = cli_args.resume_step
    if resume_step is not None:
        checkpoint_path = os.path.join(args.savepath, f'state_{resume_step}.pt')
        if os.path.exists(checkpoint_path):
            print(f'[ train ] Resuming seed {seed} from step {resume_step}')
            trainer.load(resume_step)
        else:
            print(f'[ train ] Resume requested but checkpoint not found: {checkpoint_path}')
            
    trainer.train()
    
    if run is not None:
        losses_path = os.path.join(args.savepath, 'losses.pkl')
        log_wandb_curves_from_losses(losses_path, run)
        upload_wandb_artifact(run, seed, args)
        run.summary['status'] = 'completed'
        run.summary['seed'] = seed
        run.finish()

print('Vision FM_v3 training completed.')