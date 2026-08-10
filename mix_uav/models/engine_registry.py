"""Gen15 (UAV Mix-ML) — the ML-engine dispatch table.

ONE config key (`engine` ∈ {'fm', 'mf', 'af'}) selects which generative objective drives the
UAV pipeline. Everything that differs between the engines lives HERE; the train and eval
scripts contain **no** `if engine == ...` chain.

    fm  → Gen11  FlowMatchingODE   (flow matching, Euler ODE)              — the incumbent
    mf  → Gen3v6 MeanFlowODE       (MeanFlow, arXiv 2505.13447)            — two-time (u, v)
    af  → Gen3v7 AlphaFlowODE      (α-Flow,   arXiv 2510.20771)            — two-time (u, v)

The three arms differ ONLY in objective and sampler. The backbone is a config key
(`imf_backbone`) that every arm resolves to the SAME `Flow_matcher_U_Net_v2` by default, which
is what makes the comparison architecture-controlled. See
logs_in_develop/Gen15/init/PLAN_Gen15_uav_mix_ml.md §6.

── Why the model_kwargs are per-engine ────────────────────────────────────────────────────
`model_config.pkl` does NOT describe the same object on every arm:

    fm     : model_config describes the U-NET directly     (transition_dim / cond_dim / dim …)
    mf, af : model_config describes the ENGINE, which builds its backbone INTERNALLY
             (state_dim / seq_len / freq_dim / imf_backbone …)

This is inherited from Gen3v4's "Option A" wiring (Gen8 lesson L3) and it is why the eval
loader reconstructs an engine on the two-time arms and a bare U-Net on `fm`. Anything that
reads `model_config.pkl` must therefore go through this table, never assume a shape.

── A note on `device` ─────────────────────────────────────────────────────────────────────
None of the builders below returns `device`. `utils.Config` captures `device=` itself and
applies it as `.to(device)` after construction (utils/config.py:__call__). Both source
generations pass it that way; keeping it identical avoids a silent double-move.
"""

# Engine class paths are PACKAGE-RELATIVE: utils.config.import_class prefixes them with the
# top-level package name (`mix_uav`), exactly as config/uav.py's 'models.diffusion.…' does.

# ── model (backbone / engine) kwarg builders ───────────────────────────────────────────────

def _unet_kwargs(args, dataset):
    """`fm` arm — VERBATIM from FM_v3_uav_test/train_fm_uav.py's model_config block."""
    return dict(
        horizon=args.horizon,
        transition_dim=dataset.observation_dim + dataset.action_dim,
        cond_dim=dataset.observation_dim,
        dim_mults=args.dim_mults,
        returns_condition=args.returns_condition,
        dim=args.dim,
        condition_dropout=args.condition_dropout,
    )


def _twotime_kwargs(args, dataset):
    """`mf` / `af` arms — VERBATIM from the Gen3v6/Gen3v7 train scripts' model_config block.

    🔴 FIX_8_UNET_WIDTH (Gen3v6): on the 'unet' backbone `freq_dim` IS the channel width —
    it is passed straight through as the U-Net's `dim`. 32 => 3.97 M params, 256 => 253 M.
    Gen11 trains at dim=32, so config/uav_mix.py pins freq_dim=32 on every two-time arm and
    gate G3 asserts the resulting param count matches the `fm` arm. Never raise freq_dim to
    "improve the time embedding".
    """
    return dict(
        state_dim=dataset.observation_dim + dataset.action_dim,
        seq_len=args.horizon,
        freq_dim=getattr(args, 'freq_dim', 32),
        depth=getattr(args, 'depth', 8),
        num_heads=getattr(args, 'num_heads', 4),
        mlp_dim=getattr(args, 'mlp_dim', 256),
        time_dim=getattr(args, 'time_dim', 256),
        dropout_rate=getattr(args, 'dropout_rate', 0.1),
        # dual_head=True is required by Gen3v6/v7 (FIX-4: the v head is a full loss term).
        dual_head=getattr(args, 'dual_head', True),
        # interval_cfg is FALSE: neither Gen3v6 nor Gen3v7 has interval-CFG.
        interval_cfg=getattr(args, 'interval_cfg', False),
        # Backbone selector: 'unet' (locked default, architecture-controlled) | 'dit' |
        # 'mf_dit' (mf only) | 'sit' (af only). The DiT/SiT sizing keys below are inert on 'unet'.
        imf_backbone=getattr(args, 'imf_backbone', 'unet'),
        dit_depth=getattr(args, 'dit_depth', 8),
        dit_hidden_size=getattr(args, 'dit_hidden_size', 256),
        dit_num_heads=getattr(args, 'dit_num_heads', 4),
        dit_aux_head_depth=getattr(args, 'dit_aux_head_depth', 2),
        dit_patch_size=getattr(args, 'dit_patch_size', 1),
        dit_condition_on_t=getattr(args, 'dit_condition_on_t', False),
    )


# ── diffusion (objective) kwarg builders ───────────────────────────────────────────────────

def _common_diffusion_kwargs(args, dataset):
    """The kwargs all three engines share — the FMv3ODE-compatible surface."""
    return dict(
        horizon=args.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        goal_dim=getattr(dataset, 'goal_dim', 0),
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_discount=args.loss_discount,
        returns_condition=args.returns_condition,
        condition_guidance_w=args.condition_guidance_w,
        flow_steps_v3=getattr(args, 'flow_steps_v3', 10),
        ode_inference_steps_v3=getattr(args, 'ode_inference_steps_v3',
                                       getattr(args, 'flow_steps_v3', 10)),
    )


def _fm_kwargs(args, dataset):
    """`fm` arm — VERBATIM from FM_v3_uav_test/train_fm_uav.py's diffusion_config block."""
    kw = _common_diffusion_kwargs(args, dataset)
    kw.update(
        n_timesteps=getattr(args, 'n_diffusion_steps', 20),
        time_beta_alpha_v3=args.time_beta_alpha_v3,
        time_beta_beta_v3=args.time_beta_beta_v3,
    )
    return kw


def _mf_kwargs(args, dataset):
    """`mf` arm — Gen3v6 objective knobs (PLAN Gen3v6 §3.5)."""
    kw = _common_diffusion_kwargs(args, dataset)
    kw.update(
        u_loss_weight=getattr(args, 'u_loss_weight', 1.0),
        v_loss_weight=getattr(args, 'v_loss_weight', 1.0),
        loss_schedule=getattr(args, 'loss_schedule', 'balanced'),
        warmup_epochs=getattr(args, 'warmup_epochs', 0),
        transition_epochs=getattr(args, 'transition_epochs', 0),
        time_beta_alpha_v3=getattr(args, 'time_beta_alpha_v3', 1.0),
        time_beta_beta_v3=getattr(args, 'time_beta_beta_v3', 1.0),
        t_schedule=getattr(args, 't_schedule', 'logit_normal'),
        p_mean=getattr(args, 'p_mean', -0.4),
        p_std=getattr(args, 'p_std', 1.0),
        mf_objective=getattr(args, 'mf_objective', 'meanflow'),
        meanflow_data_proportion=getattr(args, 'meanflow_data_proportion', 0.5),
        mf_adp_p=getattr(args, 'mf_adp_p', 1.0),
        mf_adp_eps=getattr(args, 'mf_adp_eps', 0.01),
    )
    return kw


def _af_kwargs(args, dataset):
    """`af` arm — VERBATIM from FM_v3_alphaflow_test's diffusion_config block.

    ⚠️ NOT built on _mf_kwargs: AlphaFlowODE takes NONE of MeanFlow's objective knobs
    (`mf_objective`, `meanflow_data_proportion`, `mf_adp_p`, `mf_adp_eps`). Its FM-anchor
    fraction is `af_ratio_fm` and its adaptive-loss epsilon is `af_adp_eps` = 1e-3, which is
    DELIBERATELY ≠ MeanFlow's 0.01 — different method, different constant. Do not harmonise
    them (Gen3v7 PLAN §11 trap 7).

    🔴 `af_alpha_end_step` MUST equal `n_train_steps` — AlphaFlowODE.__init__ hard-asserts it.
    Leaving it at upstream's larger value would hold α≈1 for the whole run, i.e. train plain
    flow matching under an α-Flow folder name (Gen3v7 PLAN §11 trap 1).
    """
    kw = _common_diffusion_kwargs(args, dataset)
    kw.update(
        u_loss_weight=getattr(args, 'u_loss_weight', 1.0),
        v_loss_weight=getattr(args, 'v_loss_weight', 1.0),
        loss_schedule=getattr(args, 'loss_schedule', 'balanced'),
        warmup_epochs=getattr(args, 'warmup_epochs', 0),
        transition_epochs=getattr(args, 'transition_epochs', 0),
        time_beta_alpha_v3=getattr(args, 'time_beta_alpha_v3', 1.0),
        time_beta_beta_v3=getattr(args, 'time_beta_beta_v3', 1.0),
        t_schedule=getattr(args, 't_schedule', 'logit_normal'),
        p_mean=getattr(args, 'p_mean', -0.4),
        p_std=getattr(args, 'p_std', 1.0),
        af_alpha_scheduler=getattr(args, 'af_alpha_scheduler', 'sigmoid'),
        af_alpha_init=getattr(args, 'af_alpha_init', 1.0),
        af_alpha_end=getattr(args, 'af_alpha_end', 0.0),
        af_alpha_init_step=getattr(args, 'af_alpha_init_step', 0),
        af_alpha_end_step=getattr(args, 'af_alpha_end_step', int(args.n_train_steps)),
        af_alpha_gamma=getattr(args, 'af_alpha_gamma', 25.0),
        af_alpha_clamp=getattr(args, 'af_alpha_clamp', 0.005),
        af_ratio_fm=getattr(args, 'af_ratio_fm', 0.5),
        af_clamp_utgt=getattr(args, 'af_clamp_utgt', 4.0),
        af_adp_eps=getattr(args, 'af_adp_eps', 1e-3),
        af_n_train_steps=int(args.n_train_steps),
    )
    return kw


# ── trainer kwarg builders ─────────────────────────────────────────────────────────────────

def _common_trainer_kwargs(args):
    return dict(
        train_test_split=getattr(args, 'train_test_split', 0.9),
        ema_decay=args.ema_decay,
        n_train_steps=args.n_train_steps,
        n_steps_per_epoch=getattr(args, 'n_steps_per_epoch', 1000),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=getattr(args, 'gradient_accumulate_every', 1),
        results_folder=args.savepath,
    )


def _fm_trainer_kwargs(args):
    """`fm` arm — VERBATIM from FM_v3_uav_test/train_fm_uav.py's trainer_config block.

    Note what is ABSENT: `split_seed`. Gen11's trainer splits train/test unseeded, so a resumed
    run re-splits and leaks held-out trajectories into training. The two-time trainer fixes this
    (split_seed=42) — which means the arms train on DIFFERENT splits. Accepted and documented
    (PLAN §5 G2): compare arms on closed-loop task metrics, which are split-independent, NEVER
    on test_loss.
    """
    return _common_trainer_kwargs(args)


def _twotime_trainer_kwargs(args):
    """`mf` / `af` arms — adds the two knobs Gen11's trainer does not have."""
    kw = _common_trainer_kwargs(args)
    kw.update(
        split_seed=getattr(args, 'split_seed', 42),
        # 🔴 Gen3v6/Gen3v7: actually applied before optimizer.step(). Dead key in Gen3v4/Gen13.
        gradient_clip=getattr(args, 'gradient_clip', 0.0),
    )
    return kw


# ── the table ──────────────────────────────────────────────────────────────────────────────

ENGINES = {
    'fm': dict(
        label='Flow Matching (Gen11 FMv3ODE)',
        model='models.unet1d_temporal_cond.Flow_matcher_U_Net_v2',
        diffusion='models.diffusion.FlowMatchingODE',
        trainer='utils.training.Trainer',
        model_kwargs=_unet_kwargs,
        diffusion_kwargs=_fm_kwargs,
        trainer_kwargs=_fm_trainer_kwargs,
        wraps_backbone=False,      # model_config.pkl describes the U-Net itself
        two_time=False,
        # FlowMatchingODE.p_sample_loop has NO `num_steps=` parameter — it reads
        # self.flow_steps_v3. K is therefore applied by SETTING the attribute (see
        # apply_nfe below), not by passing a sample kwarg.
        supports_num_steps=False,
        # Backbones this arm can select. The FM lineage never grew a transformer backbone.
        backbones=('unet',),
        # Extra folder-name tokens beyond H{h}_D{diffusion} — empty keeps the `fm` path
        # Gen11-shaped (modulo prefix/logbase), so gate G1 compares like with like.
        exp_name_tokens=(),
    ),
    'mf': dict(
        label='MeanFlow (Gen3v6, arXiv 2505.13447)',
        model='models.mf_engine.MeanFlowEngine',
        diffusion='models.mf_diffusion.MeanFlowODE',
        trainer='utils.training_twotime.Trainer',
        model_kwargs=_twotime_kwargs,
        diffusion_kwargs=_mf_kwargs,
        trainer_kwargs=_twotime_trainer_kwargs,
        wraps_backbone=True,       # model_config.pkl describes the ENGINE
        two_time=True,
        supports_num_steps=True,   # MeanFlowODE.p_sample_loop(..., num_steps=K)
        backbones=('unet', 'dit', 'mf_dit'),
        # `dp` is a first-class ablation axis in Gen3v6 and MUST be in the path, or two runs
        # differing only in data-proportion overwrite each other. Same for the backbone.
        exp_name_tokens=(('meanflow_data_proportion', 'dp'), ('imf_backbone', 'bb')),
    ),
    'af': dict(
        label='alpha-Flow (Gen3v7, arXiv 2510.20771)',
        model='models.af_engine.AlphaFlowEngine',
        diffusion='models.af_diffusion.AlphaFlowODE',
        trainer='utils.training_twotime.Trainer',
        model_kwargs=_twotime_kwargs,
        diffusion_kwargs=_af_kwargs,
        trainer_kwargs=_twotime_trainer_kwargs,
        wraps_backbone=True,
        two_time=True,
        supports_num_steps=True,
        backbones=('unet', 'dit', 'sit'),
        exp_name_tokens=(('af_alpha_start', 'as'), ('af_alpha_end', 'ae'),
                         ('imf_backbone', 'bb')),
    ),
    # A 4th arm (iMF, or a DDPM/DPCC baseline) drops in as one more row — no restructuring.
    # Neither is built in Gen15: see PLAN §1.4 (iMF abandoned/refuted) and §1.5 (no UAV DDPM
    # checkpoint exists anywhere in this repo, which caps Gen15's claim at "vs naive FM").
}

ENGINE_KEYS = tuple(ENGINES.keys())
DEFAULT_ENGINE = 'fm'


def get(engine):
    """The registry row for `engine`, or a loud error naming the valid keys."""
    if engine not in ENGINES:
        raise ValueError(
            f"unknown engine '{engine}' — valid engines are {list(ENGINE_KEYS)}. "
            f"(Gen15 ships fm|mf|af; see mix_uav/models/engine_registry.py)"
        )
    return ENGINES[engine]


def experiment_name(engine, plan=False):
    """The config/uav_mix.py block name for this engine."""
    get(engine)
    return f"{'plan_' if plan else ''}mix_uav_{engine}"


def check_backbone(engine, backbone):
    """Fail loudly on an engine/backbone pair that cannot be built.

    Selecting 'mf_dit' on the `af` arm (or any backbone at all on `fm`) otherwise surfaces as
    a ValueError from deep inside the trajectory-model constructor, hours into a job.
    """
    row = get(engine)
    if backbone not in row['backbones']:
        raise ValueError(
            f"engine '{engine}' does not support backbone '{backbone}' — "
            f"valid backbones for this arm: {list(row['backbones'])}"
        )
    return backbone


def apply_nfe(diffusion, flow_steps):
    """Pin the NFE budget K onto a BUILT diffusion object, whatever the engine.

    Both attributes are set on every arm so a checkpoint's stale pickled value can never win
    at eval time: `flow_steps_v3` is what FlowMatchingODE.p_sample_loop reads, and the
    two-time engines fall back to it whenever `num_steps=` is not passed.
    Pair with `sample_kwargs_for()`, which supplies the explicit `num_steps=K` the two-time
    samplers prefer.
    """
    k = int(flow_steps)
    diffusion.flow_steps_v3 = k
    diffusion.ode_inference_steps_v3 = k
    return k


def sample_kwargs_for(engine, flow_steps):
    """`{'num_steps': K}` for the two-time arms, `{}` for `fm` (which would TypeError)."""
    return {'num_steps': int(flow_steps)} if get(engine)['supports_num_steps'] else {}
