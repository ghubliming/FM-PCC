#!/usr/bin/env python3
"""
Train α-Flow (Gen3v7) trajectory models on the D3IL avoiding-d3il dataset.

Engine: α-Flow (arXiv 2510.20771, snap-research) — a HOMOTOPY from flow matching to MeanFlow.
- Two-time field u(z, τ, h) with a dual u/v head (backbone identical to Gen3v4/Gen3v6)
- Target u_tgt = α·v + (1−α)·u_next, self-bootstrapped under no_grad — NO network derivative
- α annealed 1 → 0 (sigmoid): starts as plain FM, becomes MeanFlow as the field gets accurate
- No interval-CFG (α-Flow's own headline config is non-cfg), no guided target

Gen3v7 is a copy-modify sibling of Gen3v6 (`flow_matcher_v3_meanflow`); the α→0 limit IS
Gen3v6's objective, which is what makes the three-way A/B (iMF / MeanFlow / α-Flow) clean.
See logs_in_develop/Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md.

Usage:
    python FM_v3_alphaflow_test/train_flow_matching_v3_alphaflow.py --seeds 6 7 8 9 10 --use-wandb
"""

import os
import sys
import glob
import argparse
import pickle

import torch

# Standard FM-PCC imports (Note: importing from diffuser.utils is maybe not ideal/optimal?)
import diffuser.utils as utils
# The Trainer must be this package's own: it carries the U9 val-loss edits (on_epoch_end
# callback, multi-value test(), seeded split, raw_mse), the Gen3v6 additions (real gradient
# clipping, h-stratified metric plumbing) AND the Gen3v7 `set_train_step` plumbing that
# feeds the α schedule. diffuser.utils.Trainer is the shared DPCC one and has none of them —
# using it crashes on train(on_epoch_end=), and would silently freeze α at its step-0 value.
from flow_matcher_v3_alphaflow.utils.training import Trainer as AlphaFlowTrainer


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


def log_wandb_from_losses(losses_path, run, after_step=-1):
    """Replay losses.pkl into W&B (standard FM-PCC pattern).

    U9: only logs steps > after_step and returns the last step logged, so it can be
    called incrementally per epoch — curves now appear during training and survive
    SLURM timeouts, instead of only after a seed fully completes.
    """
    if not os.path.exists(losses_path):
        return after_step

    with open(losses_path, 'rb') as f:
        losses_data = pickle.load(f)

    training_losses = losses_data.get('training_losses', [])

    # U9 metric-parity pass: full DPCC/Gen0 set (a0 curves) + MeanFlow-family analogs
    # (train/raw_mse ≙ loss_u, aux_loss ≙ loss_v). val/raw_mse is the held-out raw MSE —
    # adaptive-weight-free, comparable across runs/objectives, unlike test/loss, which is
    # pinned at the adaptive ceiling by construction (COMPARE §7.1 — NEVER read train/loss
    # as convergence).
    #
    # ⭐ h-stratified residual curves train/h_mse_b0..b3 (h==0, (0,0.3), [0.3,0.6),
    # [0.6,1.0]), inherited from Gen3v6. They answer whether the field is bad ONLY at large
    # h — exactly where 1–2-NFE sampling lives. Empty buckets are dropped by the trainer, so
    # these series are sparse by design.
    #
    # ⭐⭐ Gen3v7 adds train/alpha, train/discrete_frac, train/clamp_frac. **Look at
    # train/alpha FIRST on every run**: if it is flat at 1.0 you trained plain flow matching
    # for the whole budget (PLAN §11 trap 1) and every other number is meaningless.
    # Missing pkl keys (old runs, or objectives without an aux head) just skip.
    companion_keys = {
        'test_losses': 'test/loss',
        'training_a0_losses': 'train/a0_loss',
        'test_a0_losses': 'test/a0_loss',
        'training_raw_mse_losses': 'train/raw_mse',
        'test_raw_mse_losses': 'val/raw_mse',
        'training_aux_losses': 'train/aux_loss',
        'test_aux_losses': 'test/aux_loss',
        'lr_history': 'train/lr',
        # ── Gen3v6 (inherited) ────────────────────────────────────────────────────
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
        # ── Gen3v7 — α schedule telemetry (gate G4) ────────────────────────────────
        'training_alpha_losses': 'train/alpha',
        'training_discrete_frac_losses': 'train/discrete_frac',
        'training_clamp_frac_losses': 'train/clamp_frac',
        'test_alpha_losses': 'val/alpha',
        'test_discrete_frac_losses': 'val/discrete_frac',
    }
    by_step = {
        wandb_key: dict(losses_data.get(pkl_key, []))
        for pkl_key, wandb_key in companion_keys.items()
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
    test_losses = losses_data.get('test_losses', [])
    raw_mse_losses = losses_data.get('test_raw_mse_losses', [])
    if test_losses:
        run.summary['final_test_loss'] = test_losses[-1][1]
    if raw_mse_losses:
        run.summary['final_val_raw_mse'] = raw_mse_losses[-1][1]
    # Kill-criterion inputs: the final h-stratified residuals (PLAN §8 diagnostic 1). If b3
    # (h∈[0.6,1]) is flat at its step-0 value while b0 (h=0) dropped ~10×, the field is
    # untrained exactly where 1–2 NFE lives — report and stop.
    for bucket in ('h_mse_b0', 'h_mse_b1', 'h_mse_b2', 'h_mse_b3'):
        series = losses_data.get(f'training_{bucket}_losses', [])
        if series:
            run.summary[f'final_train_{bucket}'] = series[-1][1]
            run.summary[f'first_train_{bucket}'] = series[0][1]
    # 🔴 Gen3v7 gate G4, promoted to a run SUMMARY so a dead schedule is visible without
    # opening a single chart: alpha_min == alpha_max == 1.0 ⇒ the anneal never fired.
    alpha_series = losses_data.get('training_alpha_losses', [])
    if alpha_series:
        values = [v for _s, v in alpha_series]
        run.summary['alpha_first'] = values[0]
        run.summary['alpha_last'] = values[-1]
        run.summary['alpha_min'] = min(values)
        run.summary['alpha_max'] = max(values)
        run.summary['alpha_schedule_alive'] = bool(max(values) - min(values) > 1e-6)
    return last_step


def upload_wandb_artifact(run, seed, savepath):
    """Upload checkpoint artifacts to W&B."""
    import wandb
    artifact = wandb.Artifact(
        name=f'alphaflow-seed-{seed}-model',
        type='model',
        metadata={'seed': seed, 'savepath': savepath},
    )
    for filename in ['losses.pkl', 'args.json']:
        filepath = os.path.join(savepath, filename)
        if os.path.exists(filepath):
            artifact.add_file(filepath)
    run.log_artifact(artifact)


def print_alpha_schedule(args, n_points=11):
    """Print the α curve BEFORE training starts (PLAN §11 trap 1).

    α-Flow's single most likely silent failure is an anneal that never fires — upstream
    anneals over 400 k steps and we train 100 k, so a verbatim copy of their config leaves
    α ≈ 1.0 for the whole run, i.e. plain flow matching wearing an α-Flow folder name.
    AlphaFlowODE.__init__ hard-asserts `af_alpha_end_step == n_train_steps`; this banner is
    the human-readable companion, so the log shows the schedule that was actually used.
    """
    from flow_matcher_v3_alphaflow.models import AlphaFlowODE

    n_steps = int(args.n_train_steps)
    cfg = dict(
        scheduler=getattr(args, 'af_alpha_scheduler', 'sigmoid'),
        initial_value=getattr(args, 'af_alpha_init', 1.0),
        end_value=getattr(args, 'af_alpha_end', 0.0),
        init_step=getattr(args, 'af_alpha_init_step', 0),
        end_step=getattr(args, 'af_alpha_end_step', n_steps),
        gamma=getattr(args, 'af_alpha_gamma', 25.0),
        clamp_value=getattr(args, 'af_alpha_clamp', 0.005),
    )
    steps = [int(round(i * n_steps / (n_points - 1))) for i in range(n_points)]
    alphas = [AlphaFlowODE._get_ratio(cur_step=s, **cfg) for s in steps]

    print(f"[ train ] alpha schedule: {cfg['scheduler']} "
          f"{cfg['initial_value']} -> {cfg['end_value']} over "
          f"[{cfg['init_step']}, {cfg['end_step']}] gamma={cfg['gamma']} "
          f"clamp={cfg['clamp_value']}   (n_train_steps={n_steps})")
    print("[ train ]   step : " + " ".join(f"{s:>7d}" for s in steps))
    print("[ train ]   alpha: " + " ".join(f"{a:>7.3f}" for a in alphas))
    if max(alphas) - min(alphas) < 1e-6:
        print("[ train ] *** WARNING: alpha is CONSTANT across the whole budget. "
              "alpha=1 is plain flow matching, alpha=0 is Gen3v6 MeanFlow. "
              "If that is not deliberate, fix af_alpha_end_step. ***")
    print()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train α-Flow (Gen3v7)')
    parser.add_argument('--seed', type=int, help='Single seed.')
    parser.add_argument('--seeds', type=int, nargs='+', help='List of seeds.')
    # ── fix_6: resume ────────────────────────────────────────────────────────────
    parser.add_argument('--auto-resume', action='store_true',
                        help='Resume each seed from its newest local checkpoint, and skip '
                             'seeds that already reached n_train_steps.')
    parser.add_argument('--resume-step', type=int,
                        help='Explicit checkpoint step to resume from, e.g. 60000. Use -1 '
                             'for state_best.pt. Applies to --resume-seed, or to the first '
                             'seed if that is not given.')
    parser.add_argument('--resume-seed', type=int,
                        help='Which seed --resume-step applies to.')
    parser.add_argument('--force-restart', action='store_true',
                        help='Ignore checkpoints and completion markers; train from step 0. '
                             'Overwrites losses.pkl — the run history is lost.')
    parser.add_argument('--use-wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument('--wandb-project', type=str, default='FMPCC-AlphaFlow', help='W&B project.')
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


# ──────────────────────────────────────────────────────────────────────────────────────
# fix_6 — resume. This existed in Gen1/Gen2/Gen3v1 (`FM_test/train_FM.py`,
# `FM_v3_test/train_FM_v3.py`) and was dropped when Gen3v4-iMF rewrote the launcher;
# Gen3v6 inherited the gap by copy-modify, and Gen3v7 from Gen3v6. The Trainer never
# lost the capability —
# `Trainer.train()` is written against `self.step` and `Trainer.load()` restores it — the
# only missing piece was the CLI wiring, so nothing ever called `load()`.
# Trigger: job 24069 (2026-07-31) died at seed 9 / step 80000 on `[Errno 28] No space left
# on device`, and there was no way to continue it.
# ──────────────────────────────────────────────────────────────────────────────────────

BEST_CHECKPOINT_LABEL = 'best'


def checkpoint_path(savepath, label):
    return os.path.join(savepath, f'state_{label}.pt')


def find_latest_checkpoint(savepath):
    """Newest resumable checkpoint as (label, step); (None, None) if there is none.

    Two checkpoint families live side by side, and the periodic one is NOT always newer:
      * `state_<step>.pt` — written every `n_train_steps // 5` steps (20000 for this config)
      * `state_best.pt`   — written at every log step that improves val loss, so it usually
                            sits far ahead of the last periodic file.
    Job 24069 died at step 80000 with `state_60000.pt` as its newest periodic file while
    `state_best.pt` was at ~79000; taking the periodic one alone would silently discard
    19000 steps. `state_best.pt` carries the identical full payload, so resuming from it is
    exact, not an approximation.
    """
    label, step = None, -1
    for path in glob.glob(os.path.join(savepath, 'state_*.pt')):
        token = os.path.basename(path)[len('state_'):-len('.pt')]
        try:
            candidate = int(token)
        except ValueError:
            continue                      # 'best', or anything else non-numeric
        if candidate > step:
            label, step = candidate, candidate

    best_path = checkpoint_path(savepath, BEST_CHECKPOINT_LABEL)
    if os.path.exists(best_path):
        try:
            best_step = int(torch.load(best_path, map_location='cpu')['step'])
        except Exception as e:
            print(f'[ train ] Could not read {best_path}: {e}')
            best_step = -1
        if best_step > step:
            label, step = BEST_CHECKPOINT_LABEL, best_step

    return (None, None) if step < 0 else (label, step)


def last_logged_step(savepath):
    """Highest training step recorded in losses.pkl, or None."""
    losses_path = os.path.join(savepath, 'losses.pkl')
    if not os.path.exists(losses_path):
        return None
    try:
        with open(losses_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f'[ train ] Could not read {losses_path}: {e}')
        return None
    steps = [step for step, _value in data.get('training_losses', [])]
    return max(steps) if steps else None


def training_already_complete(savepath, trainer):
    """True if this seed already ran its full step budget.

    Checkpoints alone cannot answer this. With `save_freq = n_train_steps // 5` the newest
    periodic file of a FINISHED seed is `state_80000.pt` — indistinguishable from a seed
    that died at 80000. In job 24069 both existed at once: seeds 7 and 8 had finished,
    seed 9 had died, and all three looked identical on disk. losses.pkl does answer it:
    a completed seed logs its last point at `n_train_steps - log_freq`.

    Caveat: a seed killed inside its final log period reads as complete. That is at most
    `log_freq` steps of loss, and the alternative — re-running the last 20 % with cold Adam
    moments — is strictly worse.
    """
    last = last_logged_step(savepath)
    if last is None:
        return False
    return last >= int(trainer.n_train_steps) - int(trainer.log_freq)


def resolve_resume(seed, seeds, cli_args, savepath):
    """(label, step) this seed should load, or (None, None) to start from scratch."""
    if cli_args.resume_step is not None:
        target_seed = cli_args.resume_seed if cli_args.resume_seed is not None else seeds[0]
        if seed == target_seed:
            label = (BEST_CHECKPOINT_LABEL if cli_args.resume_step < 0
                     else cli_args.resume_step)
            path = checkpoint_path(savepath, label)
            # An explicit flag that silently does nothing is how a day gets lost — fail loud.
            if not os.path.exists(path):
                raise FileNotFoundError(f'--resume-step {cli_args.resume_step} given, but '
                                        f'{path} does not exist')
            return label, cli_args.resume_step
    if cli_args.auto_resume:
        return find_latest_checkpoint(savepath)
    return None, None


if __name__ == '__main__':
    sanitize_wandb_env()
    
    cli_args, remaining = parse_args()
    seeds, seed_source = resolve_seeds(cli_args)
    
    print("=" * 80)
    print("[ train ] alpha-Flow Training (Gen3v7 — AlphaFlow-PCC)")
    print("[ train ] Engine: alpha-Flow (arXiv 2510.20771), bootstrapped no-grad target")
    print("[ train ] Role:   FM -> MeanFlow homotopy; targets COMPARE 8.2's blind direction")
    print(f"[ train ] Seeds: {seeds} ({seed_source})")
    print(f"[ train ] Resume: auto={cli_args.auto_resume} step={cli_args.resume_step} "
          f"seed={cli_args.resume_seed} force_restart={cli_args.force_restart}")
    print(f"[ train ] W&B: {cli_args.use_wandb} (project: {cli_args.wandb_project})")
    print("=" * 80)
    print()
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]] + remaining
    try:
        for seed in seeds:
            print(f"[ train ] Seed {seed}")

            # Parse config using standard FM-PCC Parser
            # Uses config/avoiding-d3il.py:flow_matching_v3_alphaflow
            parser = Parser(exe_name='train')
            args = parser.parse_args(experiment='flow_matching_v3_alphaflow', seed=seed)
            device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            predict_epsilon = getattr(args, 'predict_epsilon', True)

            print_alpha_schedule(args)

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
                # dual_head=True is required (the v head carries a full loss, Gen3v6 FIX-4).
                # interval_cfg is FALSE here — Gen3v7 has no CFG. It only changes the UNet's
                # parameter set; the DiT always carries its (inert, constant-fed) ω tokens.
                dual_head=getattr(args, 'dual_head', True),
                interval_cfg=getattr(args, 'interval_cfg', False),
                # backbone selector + DiT sizing — keep 'dit' to match Gen3v4's arm
                imf_backbone=getattr(args, 'imf_backbone', 'unet'),
                dit_depth=getattr(args, 'dit_depth', 8),
                dit_hidden_size=getattr(args, 'dit_hidden_size', 256),
                dit_num_heads=getattr(args, 'dit_num_heads', 4),
                dit_aux_head_depth=getattr(args, 'dit_aux_head_depth', 2),
                dit_patch_size=getattr(args, 'dit_patch_size', 1),
                dit_condition_on_t=getattr(args, 'dit_condition_on_t', False),
            )

            diffusion_config = utils.Config(
                args.diffusion,
                savepath=(args.savepath, 'diffusion_config.pkl'),
                horizon=args.horizon,
                observation_dim=observation_dim,
                action_dim=action_dim,
                goal_dim=goal_dim,
                loss_type=args.loss_type,
                clip_denoised=args.clip_denoised,
                predict_epsilon=predict_epsilon,
                action_weight=args.action_weight,
                loss_discount=getattr(args, 'loss_discount', args.discount),
                returns_condition=args.include_returns,
                condition_guidance_w=args.condition_guidance_w,
                u_loss_weight=args.u_loss_weight,
                v_loss_weight=args.v_loss_weight,
                loss_schedule=args.loss_schedule,
                warmup_epochs=args.warmup_epochs,
                transition_epochs=args.transition_epochs,
                time_beta_alpha_v3=getattr(args, 'time_beta_alpha_v3', 1.0),
                time_beta_beta_v3=getattr(args, 'time_beta_beta_v3', 1.0),
                t_schedule=getattr(args, 't_schedule', 'logit_normal'),
                p_mean=getattr(args, 'p_mean', -0.4),
                p_std=getattr(args, 'p_std', 1.0),
                flow_steps_v3=getattr(args, 'flow_steps_v3', getattr(args, 'ode_inference_steps_v3', 10)),
                ode_inference_steps_v3=getattr(args, 'ode_inference_steps_v3', getattr(args, 'flow_steps_v3', 10)),
                # ── Gen3v7 α-Flow objective knobs (PLAN §5.2) ──────────────────────
                af_alpha_scheduler=getattr(args, 'af_alpha_scheduler', 'sigmoid'),
                af_alpha_init=getattr(args, 'af_alpha_init', 1.0),
                af_alpha_end=getattr(args, 'af_alpha_end', 0.0),
                af_alpha_init_step=getattr(args, 'af_alpha_init_step', 0),
                af_alpha_end_step=getattr(args, 'af_alpha_end_step', args.n_train_steps),
                af_alpha_gamma=getattr(args, 'af_alpha_gamma', 25.0),
                af_alpha_clamp=getattr(args, 'af_alpha_clamp', 0.005),
                af_ratio_fm=getattr(args, 'af_ratio_fm', 0.5),
                af_clamp_utgt=getattr(args, 'af_clamp_utgt', 4.0),
                af_adp_eps=getattr(args, 'af_adp_eps', 1e-3),
                # 🔴 arms the end_step == n_train_steps assert inside AlphaFlowODE.__init__
                af_n_train_steps=args.n_train_steps,
            )

            trainer_config = utils.Config(
                AlphaFlowTrainer,   # this package's Trainer, not diffuser.utils.Trainer
                savepath=(args.savepath, 'trainer_config.pkl'),
                train_test_split=getattr(args, 'train_test_split', 0.9),
                ema_decay=args.ema_decay,
                train_batch_size=args.batch_size,
                train_lr=args.learning_rate,
                gradient_accumulate_every=getattr(args, 'gradient_accumulate_every', 1),
                # 🔴 gradient_clip is WIRED (dead key in Gen3v4/Gen13 —
                # POST_U10_III §4.1). Trainer applies it before optimizer.step().
                gradient_clip=getattr(args, 'gradient_clip', 0.0),
                n_train_steps=args.n_train_steps,
                n_steps_per_epoch=getattr(args, 'n_steps_per_epoch', 1000),
                results_folder=args.savepath,
                train_device=device,
            )

            model = model_config()
            diffusion = diffusion_config(model)
            trainer = trainer_config(diffusion, dataset)

            # ── fix_6: resume ────────────────────────────────────────────────────
            # Runs BEFORE W&B init so a skipped seed does not create an empty run.
            if cli_args.force_restart:
                print(f'[ train ] --force-restart: seed {seed} starts at step 0, '
                      f'existing losses.pkl will be overwritten')
            else:
                if cli_args.auto_resume and training_already_complete(args.savepath, trainer):
                    print(f'[ train ] Seed {seed} already reached '
                          f'{int(trainer.n_train_steps)} steps — skipping')
                    print()
                    continue
                resume_label, resume_step = resolve_resume(seed, seeds, cli_args, args.savepath)
                if resume_label is not None:
                    print(f'[ train ] Resuming seed {seed} from state_{resume_label}.pt')
                    trainer.load(resume_label)
                    remaining = int(trainer.n_train_steps) - int(trainer.step)
                    if remaining <= 0:
                        print(f'[ train ] Seed {seed} is at the step budget — skipping')
                        print()
                        continue
                    print(f'[ train ] Restored step {int(trainer.step)}; '
                          f'{remaining} of {int(trainer.n_train_steps)} steps remain')
                elif cli_args.auto_resume:
                    print(f'[ train ] Seed {seed}: no checkpoint found, starting from step 0')

            # Setup W&B
            run = None
            if cli_args.use_wandb:
                try:
                    import wandb
                    # Tag the run with the Slurm job id (empty off-cluster)
                    slurm_suffix = f"-slurm-{os.environ['SLURM_JOB_ID']}" if os.environ.get('SLURM_JOB_ID') else ''
                    run = wandb.init(
                        project=cli_args.wandb_project,
                        entity=cli_args.wandb_entity,
                        group=cli_args.wandb_group,
                        name=f'AlphaFlow-seed-{seed}{slurm_suffix}',
                        config=vars(args),
                        reinit=True,
                    )
                except Exception as e:
                    print(f"[ train ] W&B init failed: {e}")

            # Train
            print(f"[ train ] Starting training at step {int(trainer.step)} "
                  f"(budget: {int(trainer.n_train_steps)})")
            # U9: flush losses.pkl to W&B after every epoch (live curves; timeout-safe)
            losses_path = os.path.join(args.savepath, 'losses.pkl')
            wandb_cursor = {'last_step': -1}

            def _flush_wandb(epoch):
                if run is not None:
                    wandb_cursor['last_step'] = log_wandb_from_losses(
                        losses_path, run, after_step=wandb_cursor['last_step'])

            trainer.train(on_epoch_end=_flush_wandb)

            # Log to W&B (final flush catches anything after the last epoch boundary)
            if run is not None:
                log_wandb_from_losses(losses_path, run, after_step=wandb_cursor['last_step'])
                upload_wandb_artifact(run, seed, args.savepath)
                run.finish()

            print(f"[ train ] Seed {seed} complete → {args.savepath}")
            print()
    finally:
        sys.argv = original_argv
    
    print("=" * 80)
    print("[ train ] Training complete for all seeds")
    print("="* 80)

