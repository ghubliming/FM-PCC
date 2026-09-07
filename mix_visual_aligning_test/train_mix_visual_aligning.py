# Visual-Mix-ML (Gen14) training script.
# Copy-modified from fm_visual_aligning_test/train_fm_visual_aligning.py (Gen7 @ HEAD).
# Dataset: ParityAligningDataset (9D visual) / StateOnlyAligningDataset (23D non-visual).
#
# ONE frame, FOUR engines, selected with --engine:
#     diffusion  Gen6V4  diffuser_visual_aligning     VisualGaussianDiffusion
#     fm    Gen7    fm_visual_aligning           VisualFlowMatching        (default/reference)
#     mf    Gen3v6  flow_matcher_v3_meanflow     VisualMeanFlow
#     af    Gen3v7  flow_matcher_v3_alphaflow    VisualAlphaFlow
#
# Everything except the three blocks marked `Gen14` below is Gen7's script verbatim.
# The engine branches live in mix_visual_aligning/models/engine_registry.py.
#
# Usage:  python -m mix_visual_aligning_test.train_mix_visual_aligning --engine mf --seed 6
import argparse
import glob
import inspect
import json
import os
import pickle
import sys
from datetime import datetime

import torch
import wandb
import mix_visual_aligning.utils as utils
# Gen14 — the arm dispatch table. Every engine-specific branch lives there, not here.
from mix_visual_aligning.models.engine_registry import (
    ENGINE_KEYS, ENGINE_INPUT_KEYS, canonical_engine,
    resolve, import_class, get_trainer_cls, describe,
)

exp = 'aligning-d3il-visual'
DEFAULT_SEEDS = [5, 6, 7, 8, 9]

# ── helpers (identical boilerplate from fmv3ode train) ────────────────────────

def sanitize_wandb_env():
    for key in ('WANDB_SERVICE', 'WANDB__SERVICE'):
        token = os.environ.get(key)
        if token and len(token.split('-')) != 5:
            os.environ.pop(key, None)

# ── Gen14 U4 ── the pkl-key -> W&B-key map. Gen14 inherited Gen7's 4-key version, which
# predates the Gen3v6 U9 metric-parity pass; meanwhile the two-time trainer it now uses
# (utils/training_twotime.py, copied from Gen3v7) persists 30+ series. The result was a
# run that wrote every diagnostic to losses.pkl and showed almost none of it.
#
# Read val/raw_mse_u, NOT train/loss or test/loss: on the mf/af arms the adaptive weight
# pins the reported loss near its ceiling by construction (COMPARE §7.1) — job 24124 ended
# at test/loss 0.926 having started around 1.0, while val/raw_mse_u actually moved.
#
# h_mse_b0..b3 are the h-stratified residuals (h==0, (0,0.3), [0.3,0.6), [0.6,1.0]). They
# answer whether the field is bad ONLY at large h — exactly where 1-2-NFE sampling lives.
# Empty buckets are dropped by the trainer, so those series are sparse by design.
#
# Arms that do not produce a given metric (diffusion has no h-buckets, mf has no alpha) simply
# have no pkl key and are skipped. One map serves all four engines.
WANDB_COMPANION_KEYS = {
    'test_losses': 'test/loss',
    'training_a0_losses': 'train/a0_loss',
    'test_a0_losses': 'test/a0_loss',
    'training_raw_mse_losses': 'train/raw_mse',
    'test_raw_mse_losses': 'val/raw_mse',
    'training_aux_losses': 'train/aux_loss',
    'test_aux_losses': 'test/aux_loss',
    'lr_history': 'train/lr',
    # ── Gen3v6 ────────────────────────────────────────────────────────────────────
    'grad_norm_history': 'train/grad_norm',       # pre-clip norm; is gradient_clip biting?
    'training_raw_mse_u_losses': 'train/raw_mse_u',
    'training_raw_mse_v_losses': 'train/raw_mse_v',
    'training_per_dim_rms_u_losses': 'train/per_dim_rms_u',
    'training_h_mse_b0_losses': 'train/h_mse_b0',
    'training_h_mse_b1_losses': 'train/h_mse_b1',
    'training_h_mse_b2_losses': 'train/h_mse_b2',
    'training_h_mse_b3_losses': 'train/h_mse_b3',
    'training_h_mean_losses': 'train/h_mean',
    'training_fm_frac_losses': 'train/fm_frac',
    'test_raw_mse_u_losses': 'val/raw_mse_u',
    'test_raw_mse_v_losses': 'val/raw_mse_v',
    'test_per_dim_rms_u_losses': 'val/per_dim_rms_u',
    'test_h_mse_b0_losses': 'val/h_mse_b0',
    'test_h_mse_b1_losses': 'val/h_mse_b1',
    'test_h_mse_b2_losses': 'val/h_mse_b2',
    'test_h_mse_b3_losses': 'val/h_mse_b3',
    # ── Gen3v7 — alpha schedule telemetry (gate G4), af arm only ───────────────────
    'training_alpha_losses': 'train/alpha',
    'training_discrete_frac_losses': 'train/discrete_frac',
    'training_clamp_frac_losses': 'train/clamp_frac',
    'test_alpha_losses': 'val/alpha',
    'test_discrete_frac_losses': 'val/discrete_frac',
}


def log_wandb_curves_from_losses(losses_path, run, after_step=-1):
    """Replay losses.pkl into W&B (standard FM-PCC pattern).

    Gen14 U4: only logs steps > after_step and returns the last step logged, so it can be
    called incrementally per epoch — curves appear DURING training and survive a SLURM
    timeout, instead of only after a seed fully completes. Visual aligning is ~9.5 h per
    seed, so the previous log-at-the-end behaviour meant a wall-clock kill lost the entire
    W&B record of a run whose losses.pkl on disk was perfectly intact.
    """
    if not os.path.exists(losses_path):
        return after_step
    with open(losses_path, 'rb') as f:
        losses = pickle.load(f)

    training_losses = losses.get('training_losses', [])
    by_step = {
        wandb_key: dict(losses.get(pkl_key, []))
        for pkl_key, wandb_key in WANDB_COMPANION_KEYS.items()
    }

    last_step = after_step
    for step, train_loss in training_losses:
        if step <= after_step:
            continue
        log_dict = {'train/loss': train_loss}
        for wandb_key, series in by_step.items():
            if step in series:
                log_dict[wandb_key] = series[step]
        run.log(log_dict, step=step)
        last_step = max(last_step, step)

    if training_losses:
        run.summary['final_train_loss'] = training_losses[-1][1]
    test_losses = losses.get('test_losses', [])
    if test_losses:
        run.summary['final_test_loss'] = test_losses[-1][1]
    raw_mse_losses = losses.get('test_raw_mse_losses', [])
    if raw_mse_losses:
        run.summary['final_val_raw_mse'] = raw_mse_losses[-1][1]
    # Kill-criterion inputs: the first and final h-stratified residuals. If b3 (h in
    # [0.6,1]) is flat at its step-0 value while b0 (h=0) dropped ~10x, the field is
    # untrained exactly where low-NFE sampling lives — report and stop.
    for bucket in ('h_mse_b0', 'h_mse_b1', 'h_mse_b2', 'h_mse_b3'):
        series = losses.get(f'training_{bucket}_losses', [])
        if series:
            run.summary[f'first_train_{bucket}'] = series[0][1]
            run.summary[f'final_train_{bucket}'] = series[-1][1]
    return last_step

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

def _resume_target(v):
    """--resume-step accepts a step NUMBER or the literal 'best'.

    'best' resolves to state_best.pt, which is a FULL checkpoint, not a weights dump:
    save_best() writes the same _checkpoint_payload() as the periodic saves, so it carries
    step / model / ema / optimizer / lr_scheduler / best_test_loss / loss histories. It is
    therefore a legitimate resume point, and on a run whose periodic saves were lost it is
    the ONLY one. The step it resumes at is whatever step the last val improvement happened
    on -- read from the payload, never assumed.
    """
    if isinstance(v, str) and v.strip().lower() == 'best':
        return 'best'
    try:
        return int(v)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--resume-step wants an integer step or 'best' (got {v!r})")


def parse_top_level_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--seeds-from-config', type=str)
    p.add_argument('--num-seeds', type=int)
    p.add_argument('--resume-seed', type=int)
    p.add_argument('--resume-step', type=_resume_target)
    p.add_argument('--auto-resume', action='store_true')
    p.add_argument('--use-wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='FM-PCC-visual-aligning-gen14')
    p.add_argument('--wandb-entity', type=str, default=None)
    p.add_argument('--wandb-group', type=str, default=None)
    p.add_argument('--wandb-mode', type=str, default='online',
                   choices=['online', 'offline', 'disabled'])
    p.add_argument('--log-freq', type=int, default=1000)
    p.add_argument('--save-every', type=int, default=None,
                   help='checkpoint cadence in steps (default: n_train_steps // 5). Lower it '
                        'on long visual runs so a wall-clock kill costs minutes, not hours.')
    # ── Gen14 ── the arm selector. Picks the config block, the engine classes and
    # the Trainer. Default 'fm' == the Gen7 reference arm, so a bare invocation
    # reproduces Gen7 behaviour.
    # `ENGINE_INPUT_KEYS` = the four canonical keys + deprecated aliases ('ddpm'), so a
    # stale command still runs. canonical_engine() normalises before anything is named.
    p.add_argument('--engine', type=str, default='fm', choices=list(ENGINE_INPUT_KEYS),
                   help='ML engine arm: diffusion (Gen6V4) | fm (Gen7) | mf (Gen3v6) | af (Gen3v7)')
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
    # 🔴 STEP 0 IS NOT PROGRESS. The trainer saves on `step % save_freq == 0`, which fires on
    # the very first iteration, so EVERY run that has ever started owns a state_0.pt. Returning
    # it made --auto-resume load a randomly-initialised network and announce 'Resuming from
    # step 0' -- a fresh run wearing a resume's clothes, and one that silently discards the
    # `resume checkpoint not found` warning that should have fired. state_best.pt is excluded
    # separately (int('best') raises ValueError above), so a run killed before its first
    # periodic save correctly reports None and starts over openly.
    steps = [s for s in steps if s > 0]
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

# ── Gen14 BLOCK 1/3 — resolve the arm ─────────────────────────────────────────
ENGINE      = canonical_engine(cli_args.engine)   # 'ddpm' -> 'diffusion' (U5), with a notice
ENGINE_SPEC = resolve(ENGINE)
EXPERIMENT  = f'mix_visual_aligning_{ENGINE}'     # the config block name
print(f'[ train ] {describe(ENGINE)}')
print(f'[ train ] config block: {EXPERIMENT}')

sys.argv = [sys.argv[0], *parser_remaining]
manifest_written = False

for seed in selected_seeds:
    args = Parser().parse_args(experiment=EXPERIMENT, seed=seed)
    torch.manual_seed(args.seed)

    if not manifest_written:
        write_seed_manifest(os.path.dirname(args.savepath), selected_seeds, seed_source, cli_args)
        manifest_written = True

    run = None
    if cli_args.use_wandb and cli_args.wandb_mode != 'disabled':
        sanitize_wandb_env()
        name_parts  = [exp, ENGINE, args.exp_name, f'S{seed}']   # Gen14: arm in the run name
        wandb_name  = '-'.join(name_parts)
        wandb_group = (cli_args.wandb_group or '-'.join(name_parts[:-1]))[:128]
        # Tag the run name with the Slurm job id (no-op off-cluster); group stays clean
        if os.environ.get('SLURM_JOB_ID'):
            wandb_name = f"{wandb_name}-slurm-{os.environ['SLURM_JOB_ID']}"
        run = wandb.init(
            project=cli_args.wandb_project, entity=cli_args.wandb_entity,
            group=wandb_group, name=wandb_name, mode=cli_args.wandb_mode,
                config={**vars(args), 'engine': ENGINE,
                    'selected_seeds': selected_seeds, 'seed_source': seed_source},
        )

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    # Visual (if_vision=True):     ParityAligningDataset  — 9D  [act(3)|des_c_pos(3)|c_pos(3)]
    # Non-visual (if_vision=False): StateOnlyAligningDataset — 23D [act(3)|obs_20D]
    _if_vision = getattr(args, 'if_vision', True)
    if _if_vision:
        from mix_visual_aligning.datasets.sequence import ParityAligningDataset
        _DatasetClass = ParityAligningDataset
        print(f'[ train ] dataset=ParityAligningDataset (visual, 9D trajectory)')
    else:
        from mix_visual_aligning.datasets.sequence import StateOnlyAligningDataset
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

    # ── Gen14 BLOCK 2/3 — backbone + engine, per arm ──────────────────────────
    # Two shapes only, selected by ENGINE_SPEC['wraps_unet']:
    #   False (diffusion, fm) — VisualUNet is built here and handed to the engine.
    #   True  (mf, af)   — an engine wrapper is built here; it constructs
    #                      VisualUNetTwoTime internally as its velocity_net.
    # model_config.pkl therefore describes the U-Net for diffusion/fm and the ENGINE
    # for mf/af; eval's load_diffusion_with_override reconstructs whichever it finds.
    _obs_dim        = 6 if _if_vision else 20   # UF-17: non-visual uses 20D obs
    _transition_dim = args.action_dim + _obs_dim
    _n_diff_steps   = getattr(args, 'n_diffusion_steps', 100)

    # ── Gen14 Fix_9 ── FiLM conditioning backbone. Set per arm in the config block
    # (config/aligning-d3il-visual.py, `film_mode`); reaches the backbone through
    # `config=args` (diffusion/fm -> VisualUNet) or `vis_config=args` (mf/af ->
    # VisualUNetTwoTime), both of which do getattr(config, 'film_mode', 'v1').
    # It is an ARCHITECTURE + PATH key: it is in args_to_watch_mix_visual_train, so v1 and
    # v2 train into parallel '..._film{film_mode}_E..' trees and their state_dicts are NOT
    # interchangeable. Validated and printed here so a v1/v2 mix-up is visible at the top of
    # the log rather than only in a directory name — and so an unknown mode dies in seconds
    # instead of after the dataset load.
    _film_mode = getattr(args, 'film_mode', 'v1') or 'v1'
    if _film_mode not in ('v1', 'v2'):
        raise SystemExit(
            f"[ train ] ERROR: film_mode='{_film_mode}' is not a known mode (want 'v1' or 'v2').")
    print(f"[ train ] film_mode = {_film_mode} "
          f"({'TRUE FiLM — per-block gamma scale + beta shift' if _film_mode == 'v2' else 'additive-bias FiLM (default)'})"
          f" — architecture key; v1/v2 checkpoints are NOT interchangeable")

    # ── Gen14 U8 ── ML BONE (generative backbone for the two-time arms).
    # Set per arm in the config block via _mix_bone_keys() / MIX_BONE_<ARM>. It is an
    # ARCHITECTURE + PATH key ('B{ml_bone}' in args_to_watch_mix_visual_train), so each bone
    # trains into its own tree and state_dicts are NOT interchangeable across bones.
    # Validated and printed here so a bone mix-up is visible at the top of the log rather
    # than only in a directory name.
    _ML_BONE_BY_ARM = {'mf': ('unet', 'mf_dit', 'dit'), 'af': ('unet', 'sit', 'dit')}
    _ml_bone = getattr(args, 'ml_bone', 'unet') or 'unet'
    if ENGINE_SPEC['two_time']:
        _allowed = _ML_BONE_BY_ARM[ENGINE]
        if _ml_bone not in _allowed:
            raise SystemExit(
                f"[ train ] ERROR: ml_bone='{_ml_bone}' is not valid for the '{ENGINE}' arm "
                f"(want one of {list(_allowed)}).")
        if _ml_bone == 'unet':
            print(f"[ train ] ml_bone = unet — VisualUNetTwoTime (FiLM {_film_mode}); "
                  f"the Gen14 baseline bone")
        else:
            # 🔴 film_mode must NOT be defined on a DiT block: FiLM is a U-Net concept and the
            # fragment would put a lying '_film..' in the checkpoint path. _mix_bone_keys()
            # deletes it; this catches a hand-edited config that put it back.
            if 'film_mode' in getattr(args, '_dict', {}) or hasattr(args, 'film_mode'):
                if getattr(args, 'film_mode', None) is not None:
                    raise SystemExit(
                        f"[ train ] ERROR: ml_bone='{_ml_bone}' is a transformer bone but the "
                        f"config block still defines film_mode={getattr(args,'film_mode')!r}. "
                        f"FiLM is a U-Net concept; leave the key out (see _mix_bone_keys).")
            print(f"[ train ] ml_bone = {_ml_bone} — VisualDiTTwoTime, visual latent enters as "
                  f"ONE PREPENDED TOKEN (hidden={getattr(args,'dit_hidden_size',160)}, "
                  f"depth={getattr(args,'dit_depth',8)}); "
                  f"parameter-matched to the ~4.0M U-Net — see Gen14/U8 PLAN section 8")
    elif _ml_bone != 'unet':
        raise SystemExit(
            f"[ train ] ERROR: ml_bone='{_ml_bone}' set on the '{ENGINE}' arm, which is "
            f"single-time and has no transformer bone. Only mf/af accept a DiT (PLAN section 11).")

    ModelCls     = import_class(ENGINE_SPEC['model'])
    DiffusionCls = import_class(ENGINE_SPEC['diffusion'])

    if ENGINE_SPEC['wraps_unet']:
        # mf / af — two-time (u, v) engine wrapper.
        model_config = utils.Config(
            ModelCls,
            savepath=(args.savepath, 'model_config.pkl'),
            state_dim=_transition_dim,
            seq_len=args.horizon,
            freq_dim=getattr(args, 'dim', 128),
            dropout_rate=getattr(args, 'condition_dropout', 0.1),
            device=args.device,
            if_vision=_if_vision,
            vis_config=args,
            # 🔴 FIDELITY: Gen3v6 and Gen3v7 BOTH ship dual_head=True (config/avoiding-d3il.py:635,
            # :754 — "the v head carries a FULL loss, not a stabiliser", FIX-4). With dual_head=False
            # the v target is regressed by an orphan MLP on raw x that shares nothing with the
            # backbone, which silently guts the second half of the objective. Do not default this off.
            dual_head=getattr(args, 'dual_head', True),
            # 🔴 interval_cfg=False in both Gen3v6 and Gen3v7 (no CFG in either). On the UNet arm
            # it changes the state_dict, so flipping it makes checkpoints non-interchangeable.
            interval_cfg=getattr(args, 'interval_cfg', False),
            # ── Gen14 U8 ── the bone selector + its sizing. Before U8 these never reached the
            # engine at all, so imf_backbone was stuck at its 'unet' default and the DiT/SiT
            # ports were unreachable from a visual run. Because they are constructor kwargs of
            # model_config, they are written into model_config.pkl and the eval loader
            # reconstructs the right bone for free (eval_mix_visual_aligning.py:2291/2355).
            imf_backbone=_ml_bone,
            dit_hidden_size=getattr(args, 'dit_hidden_size', 160),
            dit_depth=getattr(args, 'dit_depth', 8),
            dit_num_heads=getattr(args, 'dit_num_heads', 4),
            dit_patch_size=getattr(args, 'dit_patch_size', 1),
            dit_aux_head_depth=getattr(args, 'dit_aux_head_depth', 2),
            dit_condition_on_t=getattr(args, 'dit_condition_on_t', False),
        )
    else:
        # diffusion / fm — Gen6V4/Gen7 shape, unchanged.
        model_config = utils.Config(
            ModelCls,
            savepath=(args.savepath, 'model_config.pkl'),
            config=args,
        )
    model = model_config()

    # Engine kwargs: the shared core, then only what THIS arm's engine accepts.
    _engine_kwargs = dict(
        horizon=args.horizon,
        observation_dim=_obs_dim,   # 6D visual / 20D non-visual
        action_dim=args.action_dim, # 3D act: [dx, dy, dz]
        goal_dim=0,
        n_timesteps=_n_diff_steps,
        loss_type=args.loss_type,
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=getattr(args, 'action_weight', 10.0),
    )

    if ENGINE == 'diffusion':
        # Gen6V4 DDPM: a real discrete denoising chain — n_diffusion_steps is LIVE here.
        print(f'[ train ] n_diffusion_steps = {_n_diff_steps} (DDPM: live denoising chain)')
        _engine_kwargs.update(
            loss_discount=getattr(args, 'loss_discount', 1.0),
        )
    else:
        # fm / mf / af: continuous time. n_diffusion_steps only sizes a legacy buffer.
        print(f'[ train ] n_timesteps (legacy buffer size) = {_n_diff_steps}  '
              f'(continuous-time engine; does not affect training dynamics)')
        _engine_kwargs.update(
            time_beta_alpha_v3=getattr(args, 'time_beta_alpha_v3', 1.5),
            time_beta_beta_v3=getattr(args, 'time_beta_beta_v3', 1.0),
            flow_steps_v3=getattr(args, 'flow_steps_v3', _n_diff_steps),
            ode_solver_backend_v3=getattr(args, 'ode_solver_backend_v3', 'legacy_euler'),
            ode_solver_method_v3=getattr(args, 'ode_solver_method_v3', 'euler'),
        )

    if ENGINE_SPEC['two_time']:
        # mf / af — shared two-time knobs. if_vision drives the pre-encoded-latent
        # path in VisualMeanFlow / VisualAlphaFlow (PLAN §6.1).
        _engine_kwargs.update(
            if_vision=_if_vision,
            mf_freeze_vision_encoder=getattr(args, 'mf_freeze_vision_encoder', False),
            t_schedule=getattr(args, 't_schedule', 'logit_normal'),
            p_mean=getattr(args, 'p_mean', -0.4),
            p_std=getattr(args, 'p_std', 1.0),
        )

    if ENGINE == 'mf':
        # 🔴 MeanFlow's OWN objective constants. The alpha-Flow arm has different
        # names AND different values for the same ideas (af_ratio_fm / af_adp_eps);
        # af_diffusion.py:97 explicitly forbids harmonising eps across the two.
        # Keep these two blocks separate — do not merge them.
        _engine_kwargs.update(
            meanflow_data_proportion=getattr(args, 'meanflow_data_proportion', 0.5),
            mf_adp_p=getattr(args, 'mf_adp_p', 1.0),
            mf_adp_eps=getattr(args, 'mf_adp_eps', 0.01),
        )

    if ENGINE == 'af':
        # alpha-Flow's OWN objective constants (see the note in the mf block above).
        _engine_kwargs.update(
            af_ratio_fm=getattr(args, 'af_ratio_fm', 0.5),
            af_adp_eps=getattr(args, 'af_adp_eps', 1e-3),
            af_clamp_utgt=getattr(args, 'af_clamp_utgt', 4.0),
        )
        # 🔴 PLAN §6.2(a): the alpha anneal MUST span the ACTUAL training budget.
        # Both keys are derived from the SAME n_train_steps here so they can never
        # drift apart; af_diffusion asserts they agree.
        _n_train_steps = int(args.n_train_steps)
        _end_step = int(getattr(args, 'af_alpha_end_step', _n_train_steps))
        if _end_step != _n_train_steps:
            print(f'[ train ] Gen14: overriding af_alpha_end_step {_end_step} -> '
                  f'{_n_train_steps} (must equal n_train_steps; PLAN §6.2a)')
            _end_step = _n_train_steps
        _engine_kwargs.update(
            af_alpha_scheduler=getattr(args, 'af_alpha_scheduler', 'sigmoid'),
            af_alpha_init=getattr(args, 'af_alpha_init', 1.0),
            af_alpha_end=getattr(args, 'af_alpha_end', 0.0),
            af_alpha_init_step=getattr(args, 'af_alpha_init_step', 0),
            af_alpha_end_step=_end_step,
            af_alpha_gamma=getattr(args, 'af_alpha_gamma', 25.0),
            af_alpha_clamp=getattr(args, 'af_alpha_clamp', 0.005),
            af_n_train_steps=_n_train_steps,
        )
        print(f'[ train ] alpha anneal: {_engine_kwargs["af_alpha_init"]} -> '
              f'{_engine_kwargs["af_alpha_end"]} over {_end_step} steps '
              f'({_engine_kwargs["af_alpha_scheduler"]})')

    diffusion_config = utils.Config(
        DiffusionCls,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        device=args.device,
        **_engine_kwargs,
    )
    diffusion = diffusion_config(model)

    # ── Gen14 BLOCK 3/3 — Trainer, per arm ────────────────────────────────────
    # diffusion/fm -> utils.training.Trainer          (Gen7 verbatim)
    # mf/af   -> utils.training_twotime.Trainer  (Gen3v7 verbatim — h-stratified
    #            metrics, real gradient_clip, seeded split, set_train_step for alpha)
    # Trainer calls model.loss(*batch), which unpacks Batch(trajectories, conditions).
    TrainerCls = get_trainer_cls(ENGINE)
    _trainer_kwargs = dict(
        train_test_split=args.train_test_split,
        ema_decay=args.ema_decay,
        n_train_steps=args.n_train_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        results_folder=args.savepath,
        log_freq=cli_args.log_freq,
        save_freq=cli_args.save_every,
    )
    if ENGINE_SPEC['two_time']:
        # ⚠️ split_seed exists ONLY on the two-time trainer. This means mf/af use a
        # SEEDED train/test split while diffusion/fm use Gen7's unseeded one — a real
        # cross-arm confound on test_loss. PLAN §4: accept, document, and compare
        # arms on unguided TASK SUCCESS (split-independent), never on test_loss.
        _trainer_kwargs['split_seed']    = getattr(args, 'split_seed', 42)
        _trainer_kwargs['gradient_clip'] = getattr(args, 'gradient_clip', 0.0)
        # ── Gen14 U9 ── encoder LR scale. Only the two-time trainer (mf/af) implements it,
        # which is also the only place U9 runs. 1.0 == pre-U9: the trainer then builds the
        # optimiser exactly as before, one param group, so pre-U9 checkpoints auto-resume
        # unchanged. Passing it unconditionally is safe because the default IS the old value.
        _trainer_kwargs['vis_lr_scale']  = float(getattr(args, 'vis_lr_scale', 1.0))
    trainer_config = utils.Config(
        TrainerCls,
        savepath=(args.savepath, 'trainer_config.pkl'),
        **_trainer_kwargs,
    )
    trainer = trainer_config(diffusion, dataset)

    # ── Resume logic ──────────────────────────────────────────────────────────
    resume_step = None
    if cli_args.auto_resume:
        resume_step = find_latest_checkpoint_step(args.savepath)
        # ── Fix_10 ── FALL BACK TO state_best.pt when no numbered checkpoint survives.
        # Job 24838 is the case this exists for: ENOSPC + a 24 h kill at step 83999 left the
        # savepath holding state_best.pt and nothing else, so find_latest_checkpoint_step
        # (which parses an int out of the filename, and so cannot see 'best') returned None
        # and ~84k steps of GPU time looked unrecoverable. It was not: state_best.pt is the
        # same full payload as a periodic save. Numbered checkpoints still win when present --
        # they are the later training state; state_best is only the last val improvement.
        if resume_step is None and os.path.exists(os.path.join(args.savepath, 'state_best.pt')):
            resume_step = 'best'
            print('[ train ] auto-resume: no numbered state_<step>.pt survives; '
                  'falling back to state_best.pt')
    if should_apply_manual_resume(seed, selected_seeds, cli_args):
        resume_step = cli_args.resume_step
    if resume_step is not None:
        cp = os.path.join(args.savepath, f'state_{resume_step}.pt')
        if os.path.exists(cp):
            print(f'[ train ] Resuming seed {seed} from {os.path.basename(cp)}')
            try:
                trainer.load(resume_step)
            except Exception as _e:
                # 🔴 A checkpoint written while the disk was filling is TRUNCATED, and
                # torch.load raises rather than returning garbage. Dying here is correct:
                # silently starting from scratch would burn the whole wall producing a run
                # the operator believes is a resume.
                raise RuntimeError(
                    f'[ train ] {cp} exists but could not be loaded ({type(_e).__name__}: {_e}).\n'
                    f'  A checkpoint interrupted by ENOSPC is truncated and unrecoverable. '
                    f'Delete it and start fresh, or resume from another state_<step>.pt.') from _e
            # The TRUE step comes from the payload -- for 'best' it is whenever the last val
            # improvement landed, which is NOT the same as the last periodic save.
            print(f'[ train ] resumed at step {trainer.step} of {int(args.n_train_steps)} '
                  f'({int(args.n_train_steps) - int(trainer.step)} remaining)')
        else:
            print(f'[ train ] Resume checkpoint not found: {cp}')

    # ── Gen14 U4 ── flush losses.pkl to W&B after every epoch instead of only at the end.
    # `on_epoch_end` is a no-op for any Trainer that does not accept it (the diffusion arm's
    # Gen6V4 trainer), so the call is guarded by a signature check rather than assumed.
    losses_path = os.path.join(args.savepath, 'losses.pkl')
    wandb_cursor = {'last_step': -1}

    def _flush_wandb(epoch):
        if run is not None:
            wandb_cursor['last_step'] = log_wandb_curves_from_losses(
                losses_path, run, after_step=wandb_cursor['last_step'])

    if run is not None and 'on_epoch_end' in inspect.signature(trainer.train).parameters:
        trainer.train(on_epoch_end=_flush_wandb)
    else:
        trainer.train()

    if run is not None:
        # Final catch-up: picks up the last epoch's rows, and is a full replay if the
        # trainer had no on_epoch_end hook (cursor still at -1).
        log_wandb_curves_from_losses(losses_path, run, after_step=wandb_cursor['last_step'])
        upload_wandb_artifact(run, seed, args)
        run.summary['status'] = 'completed'
        run.summary['seed']   = seed
        run.finish()

print(f'Visual-Mix-ML (Gen14) training completed — engine={ENGINE} '
      f'({ENGINE_SPEC["label"]}), seeds={selected_seeds}.')
