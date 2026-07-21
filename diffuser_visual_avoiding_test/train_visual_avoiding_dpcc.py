# Visual-DPCC (Gen6V4) training script.
# Copy-modified from FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py.
# Core: diffuser_visual_avoiding  |  Dataset: ParityAvoidingDataset (6D, single-cam)
import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime

import torch
import wandb
import diffuser_visual_avoiding.utils as utils

exp = 'avoiding-d3il-visual'
DEFAULT_SEEDS = [5, 6, 7, 8, 9]

# ── helpers (identical boilerplate from fmv3ode train) ────────────────────────

def sanitize_wandb_env():
    for key in ('WANDB_SERVICE', 'WANDB__SERVICE'):
        token = os.environ.get(key)
        if token and len(token.split('-')) != 5:
            os.environ.pop(key, None)

def log_wandb_curves_from_losses(losses_path, run):
    if not os.path.exists(losses_path):
        return
    with open(losses_path, 'rb') as f:
        losses = pickle.load(f)
    training_losses   = losses.get('training_losses', [])
    test_losses       = losses.get('test_losses', [])
    training_a0       = losses.get('training_a0_losses', [])
    test_a0           = losses.get('test_a0_losses', [])
    test_by_step      = {s: v for s, v in test_losses}
    train_a0_by_step  = {s: v for s, v in training_a0}
    test_a0_by_step   = {s: v for s, v in test_a0}
    for step, tloss in training_losses:
        ld = {'train/loss': tloss}
        if step in test_by_step:    ld['test/loss']      = test_by_step[step]
        if step in train_a0_by_step: ld['train/a0_loss'] = train_a0_by_step[step]
        if step in test_a0_by_step:  ld['test/a0_loss']  = test_a0_by_step[step]
        run.log(ld, step=step)
    if training_losses: run.summary['final_train_loss'] = training_losses[-1][1]
    if test_losses:     run.summary['final_test_loss']  = test_losses[-1][1]

def upload_wandb_artifact(run, seed, args):
    artifact = wandb.Artifact(
        name=f'{args.dataset}-seed-{seed}-model', type='model',
        metadata={'dataset': args.dataset, 'seed': seed,
                  'savepath': args.savepath, 'n_train_steps': args.n_train_steps},
    )
    for fn in ['losses.pkl', 'args.json']:
        fp = os.path.join(args.savepath, fn)
        if os.path.exists(fp):
            artifact.add_file(fp)
    run.log_artifact(artifact)

class Parser(utils.Parser):
    dataset: str = exp
    config: str  = 'config.' + exp

def parse_top_level_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--seeds-from-config', type=str)
    p.add_argument('--num-seeds', type=int)
    p.add_argument('--resume-seed', type=int)
    p.add_argument('--resume-step', type=int)
    p.add_argument('--auto-resume', action='store_true')
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='FMPCC-visual-avoiding-dpcc')
    p.add_argument('--wandb-entity', type=str, default=None)
    p.add_argument('--wandb-group', type=str, default=None)
    p.add_argument('--wandb-mode', type=str, default='online',
                   choices=['online', 'offline', 'disabled'])
    p.add_argument('--log-freq', type=int, default=1000)
    args, remaining = p.parse_known_args()
    return args, remaining

def load_seeds_from_config(path):
    with open(path, 'r') as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        seeds = payload.get('seed_list') or payload.get('seeds')
    elif isinstance(payload, list):
        seeds = payload
    else:
        raise ValueError(f'Unsupported seed config: {path}')
    return [int(s) for s in seeds]

def resolve_seed_list(cli_args):
    if cli_args.seed is not None:
        seeds, source = [cli_args.seed], 'cli --seed'
    elif cli_args.seeds is not None:
        seeds, source = [int(s) for s in cli_args.seeds], 'cli --seeds'
    elif cli_args.seeds_from_config is not None:
        seeds, source = load_seeds_from_config(cli_args.seeds_from_config), f'config {cli_args.seeds_from_config}'
    else:
        seeds, source = list(DEFAULT_SEEDS), 'default'
    if cli_args.num_seeds is not None:
        seeds = seeds[:cli_args.num_seeds]
    return seeds, source

def find_latest_checkpoint_step(results_dir):
    steps = []
    for cp in glob.glob(os.path.join(results_dir, 'state_*.pt')):
        try:
            steps.append(int(os.path.basename(cp).replace('state_', '').replace('.pt', '')))
        except ValueError:
            pass
    return max(steps) if steps else None

def write_seed_manifest(run_root, seeds, source, cli_args):
    payload = {
        'generation_date': datetime.utcnow().isoformat() + 'Z',
        'total_seeds': len(seeds), 'seed_list': seeds, 'seed_source': source,
        'num_seeds_applied': cli_args.num_seeds,
        'resume_seed': cli_args.resume_seed, 'resume_step': cli_args.resume_step,
        'auto_resume': cli_args.auto_resume,
    }
    with open(os.path.join(run_root, 'seeds_config.json'), 'w') as f:
        json.dump(payload, f, indent=2)

def should_apply_manual_resume(seed, selected_seeds, cli_args):
    if cli_args.resume_step is None:
        return False
    if cli_args.resume_seed is not None:
        return seed == cli_args.resume_seed
    return seed == selected_seeds[0]

# ── main ──────────────────────────────────────────────────────────────────────

cli_args, parser_remaining = parse_top_level_args()
selected_seeds, seed_source = resolve_seed_list(cli_args)
print(f'[ train ] Seeds: {selected_seeds}  (source: {seed_source})')
sys.argv = [sys.argv[0], *parser_remaining]
manifest_written = False

for seed in selected_seeds:
    args = Parser().parse_args(experiment='visual_avoiding_dpcc', seed=seed)
    torch.manual_seed(args.seed)

    if not manifest_written:
        write_seed_manifest(os.path.dirname(args.savepath), selected_seeds, seed_source, cli_args)
        manifest_written = True

    run = None
    if cli_args.use_wandb and cli_args.wandb_mode != 'disabled':
        sanitize_wandb_env()
        name_parts  = [exp, args.exp_name, f'S{seed}']
        wandb_name  = '-'.join(name_parts)
        wandb_group = (cli_args.wandb_group or '-'.join(name_parts[:-1]))[:128]
        # Tag the run name with the Slurm job id (no-op off-cluster); group stays clean
        if os.environ.get('SLURM_JOB_ID'):
            wandb_name = f"{wandb_name}-slurm-{os.environ['SLURM_JOB_ID']}"
        run = wandb.init(
            project=cli_args.wandb_project, entity=cli_args.wandb_entity,
            group=wandb_group, name=wandb_name, mode=cli_args.wandb_mode,
            config={**vars(args), 'selected_seeds': selected_seeds, 'seed_source': seed_source},
        )

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    # Gen9 Epoch 2: visual-avoiding only. Non-visual avoiding is out of scope —
    # if you need it, port StateOnly*Dataset back into sequence.py first.
    _if_vision = getattr(args, 'if_vision', True)
    if not _if_vision:
        raise NotImplementedError(
            "diffuser_visual_avoiding is visual-only (Gen9 Ep 2). "
            "Set if_vision=True in the visual_avoiding_dpcc config, or implement "
            "a non-visual StateOnlyAvoidingDataset in sequence.py."
        )
    from diffuser_visual_avoiding.datasets.sequence import ParityAvoidingDataset
    _DatasetClass = ParityAvoidingDataset
    print(f'[ train ] dataset=ParityAvoidingDataset (visual, 6D trajectory)')

    dataset_config = utils.Config(
        _DatasetClass,
        savepath=(args.savepath, 'dataset_config.pkl'),
        dataset_path='environments/dataset/data/avoiding/all_data/train_files.pkl',
        horizon=args.horizon,
        max_n_episodes=getattr(args, 'max_path_length', 1000),
    )
    dataset = dataset_config()

    # Save LimitsNormalizers to disk for eval-time denormalization
    obs_norm_path = os.path.join(args.savepath, 'obs_normalizer.pkl')
    act_norm_path = os.path.join(args.savepath, 'act_normalizer.pkl')
    with open(obs_norm_path, 'wb') as f:
        pickle.dump(dataset.obs_normalizer, f)
    with open(act_norm_path, 'wb') as f:
        pickle.dump(dataset.act_normalizer, f)
    print(f'[ train ] Saved obs_normalizer → {obs_norm_path}')
    print(f'[ train ] Saved act_normalizer → {act_norm_path}')
    # Log normalizer statistics so training logs can be cross-checked against eval logs.
    # Near-zero range in any action dim indicates zero-padded frames corrupted the scaler.
    print(f'[ train ] obs_normalizer {dataset.obs_normalizer}')
    print(f'[ train ] act_normalizer {dataset.act_normalizer}')

    # FIX-18: VisualUNet's non-visual branch computes transition_dim from
    # config.obs_dim. Visual variants hardcode obs_dim=6 (the visual obs
    # anchor); for non-visual runs we must override it to 20 so the model
    # builds with the correct 23-D input (3 action + 20 obs). Without this,
    # the first conv expects 9 channels while the dataset feeds 23 → crash.
    if not _if_vision:
        _dataset_obs_dim = dataset.obs_normalizer.mins.shape[0]
        if getattr(args, 'obs_dim', None) != _dataset_obs_dim:
            print(f'[ train ] FIX-18: overriding args.obs_dim '
                  f'{getattr(args, "obs_dim", None)} → {_dataset_obs_dim} '
                  f'(non-visual; from dataset normalizer)')
            args.obs_dim = _dataset_obs_dim

    # ── 2. Model — VisualUNet with hardcoded transition_dim=6 (Gen9 Ep 2) ────
    from diffuser_visual_avoiding.models.visual_unet import VisualUNet

    model_config = utils.Config(
        VisualUNet,
        savepath=(args.savepath, 'model_config.pkl'),
        config=args,
    )
    model = model_config()

    # ── 3. Diffusion engine — VisualGaussianDiffusion ─────────────────────────
    from diffuser_visual_avoiding.models.visual_gaussian_diffusion import VisualGaussianDiffusion

    _n_diff_steps = getattr(args, 'n_diffusion_steps', 100)
    print(f'[ train ] n_diffusion_steps = {_n_diff_steps}  '
          f'(must match eval config to avoid denoising-chain mismatch)')
    # Gen9 Ep 2 Fix-2: visual avoiding obs is 4-D [des_xy(2), c_xy(2)] (NOT 6).
    # Aligning hardcoded _obs_dim=6 because its visual obs was [des_c_pos(3), c_pos(3)].
    # transition_dim = action(2) + obs(4) = 6 for visual avoiding.
    _obs_dim = 4 if _if_vision else 20   # visual=4 (avoiding 2-D), non-visual=20 (out of scope)
    diffusion_config = utils.Config(
        VisualGaussianDiffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=_obs_dim,   # 4D visual avoiding / 20D non-visual (not currently used)
        action_dim=args.action_dim, # 2D act: [dx, dy]
        goal_dim=0,
        n_timesteps=_n_diff_steps,
        loss_type=args.loss_type,
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=getattr(args, 'action_weight', 10.0),
        device=args.device,
    )
    diffusion = diffusion_config(model)

    # ── 4. Trainer — diffuser_visual_avoiding Trainer ─────────────────────────
    # No scaler argument. Trainer calls model.loss(*batch) which unpacks
    # Batch(trajectories, conditions) → loss(trajectories, conditions).
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
        log_freq=cli_args.log_freq,
    )
    trainer = trainer_config(diffusion, dataset)

    # ── Resume logic ──────────────────────────────────────────────────────────
    resume_step = None
    if cli_args.auto_resume:
        resume_step = find_latest_checkpoint_step(args.savepath)
    if should_apply_manual_resume(seed, selected_seeds, cli_args):
        resume_step = cli_args.resume_step
    if resume_step is not None:
        cp = os.path.join(args.savepath, f'state_{resume_step}.pt')
        if os.path.exists(cp):
            print(f'[ train ] Resuming seed {seed} from step {resume_step}')
            trainer.load(resume_step)
        else:
            print(f'[ train ] Resume checkpoint not found: {cp}')

    trainer.train()

    if run is not None:
        log_wandb_curves_from_losses(os.path.join(args.savepath, 'losses.pkl'), run)
        upload_wandb_artifact(run, seed, args)
        run.summary['status'] = 'completed'
        run.summary['seed']   = seed
        run.finish()

print('Visual-DPCC training completed.')
