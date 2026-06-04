# Visual-iMF (Gen8) training script.
# Copy-modified from fm_visual_aligning_test/train_fm_visual_aligning.py (Gen7).
# Core: imf_visual_aligning (iMeanFlow ODE engine)  |  Dataset: ParityAligningDataset (9D)
import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime

import torch
import wandb
import imf_visual_aligning.utils as utils

exp = 'aligning-d3il-visual'  # config file
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
    p.add_argument('--wandb-project', type=str, default='FM-PCC-visual-aligning-gen7')
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
    args = Parser().parse_args(experiment='imf_visual_aligning', seed=seed)
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
        run = wandb.init(
            project=cli_args.wandb_project, entity=cli_args.wandb_entity,
            group=wandb_group, name=wandb_name, mode=cli_args.wandb_mode,
            config={**vars(args), 'selected_seeds': selected_seeds, 'seed_source': seed_source},
        )

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    # Visual (if_vision=True):     ParityAligningDataset  — 9D  [act(3)|des_c_pos(3)|c_pos(3)]
    # Non-visual (if_vision=False): StateOnlyAligningDataset — 23D [act(3)|obs_20D]
    _if_vision = getattr(args, 'if_vision', True)
    if _if_vision:
        from imf_visual_aligning.datasets.sequence import ParityAligningDataset
        _DatasetClass = ParityAligningDataset
        print(f'[ train ] dataset=ParityAligningDataset (visual, 9D trajectory)')
    else:
        from imf_visual_aligning.datasets.sequence import StateOnlyAligningDataset
        _DatasetClass = StateOnlyAligningDataset
        print(f'[ train ] dataset=StateOnlyAligningDataset (non-visual, 23D trajectory)')

    dataset_config = utils.Config(
        _DatasetClass,
        savepath=(args.savepath, 'dataset_config.pkl'),
        dataset_path='environments/dataset/data/aligning/train_files.pkl',
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

    # ── 2. Engine — iMeanFlowEngine (visual: VisualUNet velocity_net via if_vision) ──
    # Gen8: no standalone VisualUNet construction. VisualUNet is wired INSIDE
    # iMFTrajectoryModel when if_vision=True (Option A from PLAN.md §3 Step 2).
    # iMeanFlowEngine → iMFTrajectoryModel(velocity_net=VisualUNet(args))
    from imf_visual_aligning.models.imf_engine import iMeanFlowEngine

    # Gen8 Ep 1: visual aligning obs is 6-D [des_c_pos(3), c_pos(3)].
    # Aligning dims match Gen7 (9-D traj = act(3) + obs(6)); no dim change from FM→iMF.
    _obs_dim = 6 if _if_vision else 20   # visual=6 (aligning 3-D pos×2), non-visual=20
    _transition_dim = args.action_dim + _obs_dim  # 3 + 6 = 9 for visual

    # Wrap in utils.Config so model_config.pkl is written alongside diffusion_config.pkl.
    # eval_imf_visual_aligning.load_diffusion_with_override() loads model_config.pkl to
    # reconstruct the engine before passing it to VisualIMF — same pattern as Gen7 VisualUNet.
    model_config = utils.Config(
        iMeanFlowEngine,
        savepath=(args.savepath, 'model_config.pkl'),
        state_dim=_transition_dim,
        seq_len=args.horizon,
        freq_dim=getattr(args, 'dim', 128),
        dropout_rate=getattr(args, 'condition_dropout', 0.1),
        device=args.device,
        if_vision=_if_vision,
        vis_config=args,
    )
    engine = model_config()

    # ── 3. iMF diffusion wrapper — VisualIMF ──────────────────────────────────
    from imf_visual_aligning.models.visual_imf_diffusion import VisualIMF

    _n_diff_steps = getattr(args, 'n_diffusion_steps', 100)
    print(f'[ train ] n_timesteps (legacy buffer size) = {_n_diff_steps}  '
          f'(iMF uses continuous time; this value only sizes legacy buffers)')
    diffusion_config = utils.Config(
        VisualIMF,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=_obs_dim,   # 6D visual / 20D non-visual
        action_dim=args.action_dim, # 3D act: [dx, dy, dz]
        goal_dim=0,
        n_timesteps=_n_diff_steps,
        loss_type=args.loss_type,
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=getattr(args, 'action_weight', 1.0),
        time_beta_alpha_v3=getattr(args, 'time_beta_alpha_v3', 1.5),
        time_beta_beta_v3=getattr(args, 'time_beta_beta_v3', 1.0),
        flow_steps_v3=getattr(args, 'flow_steps_v3', _n_diff_steps),
        ode_solver_backend_v3=getattr(args, 'ode_solver_backend_v3', 'legacy_euler'),
        ode_solver_method_v3=getattr(args, 'ode_solver_method_v3', 'euler'),
        u_loss_weight=getattr(args, 'u_loss_weight', 1.0),
        v_loss_weight=getattr(args, 'v_loss_weight', 0.1),
        loss_schedule=getattr(args, 'loss_schedule', 'balanced'),
        if_vision=_if_vision,
        device=args.device,
    )
    diffusion = diffusion_config(engine)

    # ── 4. Trainer — imf_visual_aligning Trainer ─────────────────────────────
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

print('Visual-iMF (Gen8) training completed.')
