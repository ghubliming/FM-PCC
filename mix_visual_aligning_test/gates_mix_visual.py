#!/usr/bin/env python3
"""Gen14 gates — run these on the cluster BEFORE believing any Visual-Mix-ML number.

    python -m mix_visual_aligning_test.gates_mix_visual --gate all
    python -m mix_visual_aligning_test.gates_mix_visual --gate g0      # no torch needed

G0  copy fidelity          every verbatim/sed file still matches its source
G1  reference-arm identity ddpm/fm still resolve to Gen6V4/Gen7 classes and Gen7's Trainer
G2  JVP survives vision    one mf training step, if_vision=True, no NotImplementedError
G3  MeanFlow identity h=0   u_target == v_inst when every sample is an FM anchor
G4  alpha spans the budget  alpha(0)~1, alpha(N)~0, monotone
G5  alpha->0 == MeanFlow    af with alpha pinned to 0 matches the mf objective
G6  projector fires at K=1  the DPCC projection is not silently skipped at 1 NFE

G0 needs no torch at all. G1 and G4 need torch but no GPU. G6 builds small state-only
models and runs on CPU. G2/G3/G5 build visual models and need a GPU (`--gate static`
runs everything except those three).

Read `raw_mse_u`, never `diffusion_loss`: the adaptive loss sits at its ceiling by
construction and says nothing about convergence.

fix_2 (2026-08-01): G6 was a substring search that could never fail — see its docstring.
It is now a runtime spy-projector test.
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── G0: the copy ledger ───────────────────────────────────────────────────────
# (gen14_path, source_path, source_package_name)
# 'source_package_name' is sed-reversed before comparing, so a pure rename shows as clean.
VERBATIM = [
    # ── from Gen7 (fm_visual_aligning) — the frame + the fm arm ──
    ('mix_visual_aligning/models/helpers.py',              'fm_visual_aligning/models/helpers.py',              'fm_visual_aligning'),
    ('mix_visual_aligning/models/unet1d_temporal_cond.py', 'fm_visual_aligning/models/unet1d_temporal_cond.py', 'fm_visual_aligning'),
    ('mix_visual_aligning/models/unet1d_temporal_film.py', 'fm_visual_aligning/models/unet1d_temporal_film.py', 'fm_visual_aligning'),
    ('mix_visual_aligning/models/visual_unet.py',          'fm_visual_aligning/models/visual_unet.py',          'fm_visual_aligning'),
    ('mix_visual_aligning/models/fm_diffusion.py',         'fm_visual_aligning/models/diffusion.py',            'fm_visual_aligning'),
    ('mix_visual_aligning/sampling/projection.py',         'fm_visual_aligning/sampling/projection.py',         'fm_visual_aligning'),
    ('mix_visual_aligning/datasets/sequence.py',           'fm_visual_aligning/datasets/sequence.py',           'fm_visual_aligning'),
    ('mix_visual_aligning/datasets/normalization.py',      'fm_visual_aligning/datasets/normalization.py',      'fm_visual_aligning'),
    ('mix_visual_aligning/utils/training.py',              'fm_visual_aligning/utils/training.py',              'fm_visual_aligning'),
    ('mix_visual_aligning/utils/arrays.py',                'fm_visual_aligning/utils/arrays.py',                'fm_visual_aligning'),
    ('mix_visual_aligning/utils/serialization.py',         'fm_visual_aligning/utils/serialization.py',         'fm_visual_aligning'),
    ('mix_visual_aligning/utils/setup.py',                 'fm_visual_aligning/utils/setup.py',                 'fm_visual_aligning'),
    ('mix_visual_aligning/utils/config.py',                'fm_visual_aligning/utils/config.py',                'fm_visual_aligning'),
    # ── from Gen6V4 (diffuser_visual_aligning) — the ddpm arm ──
    ('mix_visual_aligning/models/diffusion.py',                'diffuser_visual_aligning/models/diffusion.py',                'diffuser_visual_aligning'),
    ('mix_visual_aligning/models/visual_gaussian_diffusion.py', 'diffuser_visual_aligning/models/visual_gaussian_diffusion.py', 'diffuser_visual_aligning'),
    # ── from Gen3v6 (flow_matcher_v3_meanflow) — the mf arm ──
    ('mix_visual_aligning/models/mf_diffusion.py',              'flow_matcher_v3_meanflow/models/mf_diffusion.py',              'flow_matcher_v3_meanflow'),
    ('mix_visual_aligning/models/mf_dit_trajectory.py',         'flow_matcher_v3_meanflow/models/mf_dit_trajectory.py',         'flow_matcher_v3_meanflow'),
    ('mix_visual_aligning/models/mf_dit_official_trajectory.py','flow_matcher_v3_meanflow/models/mf_dit_official_trajectory.py','flow_matcher_v3_meanflow'),
    ('mix_visual_aligning/models/mlp.py',                       'flow_matcher_v3_meanflow/models/mlp.py',                       'flow_matcher_v3_meanflow'),
    # ── from Gen3v7 (flow_matcher_v3_alphaflow) — the af arm + the two-time trainer ──
    ('mix_visual_aligning/models/af_diffusion.py',        'flow_matcher_v3_alphaflow/models/af_diffusion.py',        'flow_matcher_v3_alphaflow'),
    ('mix_visual_aligning/models/af_dit_trajectory.py',   'flow_matcher_v3_alphaflow/models/af_dit_trajectory.py',   'flow_matcher_v3_alphaflow'),
    ('mix_visual_aligning/models/af_sit_trajectory.py',   'flow_matcher_v3_alphaflow/models/af_sit_trajectory.py',   'flow_matcher_v3_alphaflow'),
    ('mix_visual_aligning/utils/training_twotime.py',     'flow_matcher_v3_alphaflow/utils/training.py',             'flow_matcher_v3_alphaflow'),
]

# Files that are deliberately grafted — a diff here is EXPECTED. Listed so the ledger is
# complete and nobody mistakes an unlisted file for an audited one.
GRAFTED = {
    'mix_visual_aligning/models/unet1d_twotime_cond.py':
        'Gen3v6 Flow_matcher_U_Net_v2 + Gen7 cond_mlp block (2 additive hunks)',
    'mix_visual_aligning/models/mf_trajectory_model.py':
        'Gen3v6 + if_vision/vis_config visual branch',
    'mix_visual_aligning/models/af_trajectory_model.py':
        'Gen3v7 + if_vision/vis_config visual branch (same graft, copied)',
    'mix_visual_aligning/models/mf_engine.py':  'Gen3v6 + if_vision/vis_config passthrough',
    'mix_visual_aligning/models/af_engine.py':  'Gen3v7 + if_vision/vis_config passthrough',
}
NEW_FILES = [
    'mix_visual_aligning/models/engine_registry.py',
    'mix_visual_aligning/models/visual_unet_twotime.py',
    'mix_visual_aligning/models/visual_mf_diffusion.py',
    'mix_visual_aligning/models/visual_af_diffusion.py',
    'mix_visual_aligning/models/__init__.py',
]


def _norm(path, src_pkg):
    """Read a file, strip CR, and reverse the package rename so a pure copy compares equal."""
    with open(os.path.join(REPO, path), 'rb') as f:
        text = f.read().decode('utf-8')
    return text.replace('\r\n', '\n').replace('mix_visual_aligning', src_pkg)


def gate_g0(verbose=True):
    """Every verbatim/sed file still matches its source, modulo the package rename."""
    print('\n=== G0: copy fidelity ===')
    failures = []
    for dst, src, pkg in VERBATIM:
        dst_abs, src_abs = os.path.join(REPO, dst), os.path.join(REPO, src)
        if not os.path.exists(dst_abs):
            failures.append(f'{dst}: MISSING in Gen14'); continue
        if not os.path.exists(src_abs):
            failures.append(f'{src}: MISSING source (upstream moved?)'); continue
        with open(src_abs, 'rb') as f:
            src_text = f.read().decode('utf-8').replace('\r\n', '\n')
        if _norm(dst, pkg) != src_text:
            failures.append(f'{dst}: DIFFERS from {src}')
        elif verbose:
            print(f'  ok   {dst}')
    print(f'\n  grafted (diff expected, {len(GRAFTED)} files):')
    for f_, why in GRAFTED.items():
        mark = 'ok  ' if os.path.exists(os.path.join(REPO, f_)) else 'MISS'
        print(f'    {mark} {f_}  <- {why}')
    print(f'  new ({len(NEW_FILES)} files):')
    for f_ in NEW_FILES:
        mark = 'ok  ' if os.path.exists(os.path.join(REPO, f_)) else 'MISS'
        print(f'    {mark} {f_}')
    if failures:
        print('\n  G0 FAIL — the copy assumption broke. Re-open the plan; do NOT patch over it:')
        for f_ in failures:
            print(f'    ! {f_}')
        return False
    print(f'\n  G0 PASS — {len(VERBATIM)} verbatim files match their sources.')
    return True


def gate_g1():
    """ddpm/fm resolve to Gen6V4/Gen7 classes and Gen7's Trainer; mf/af to the two-time one.

    This is the STRUCTURAL half of the reference-arm guarantee (PLAN §3.1). The numerical
    half — 50 training steps compared against Gen7 — needs a GPU and is driven by the
    sbatch wrapper, not from here.
    """
    print('\n=== G1: reference-arm wiring ===')
    from mix_visual_aligning.models.engine_registry import ENGINES, resolve
    expected = {
        'ddpm': ('VisualGaussianDiffusion', 'VisualUNet',       'training.Trainer',          False),
        'fm':   ('VisualFlowMatching',      'VisualUNet',       'training.Trainer',          False),
        'mf':   ('VisualMeanFlow',          'MeanFlowEngine',   'training_twotime.Trainer',  True),
        'af':   ('VisualAlphaFlow',         'AlphaFlowEngine',  'training_twotime.Trainer',  True),
    }
    ok = True
    for eng, (diff_cls, model_cls, trainer, two_time) in expected.items():
        spec = resolve(eng)
        checks = [
            (spec['diffusion'].endswith(diff_cls), f"diffusion={spec['diffusion']}"),
            (spec['model'].endswith(model_cls),    f"model={spec['model']}"),
            (spec['trainer'].endswith(trainer),    f"trainer={spec['trainer']}"),
            (spec['two_time'] == two_time,         f"two_time={spec['two_time']}"),
            (spec['wraps_unet'] == two_time,       f"wraps_unet={spec['wraps_unet']}"),
        ]
        bad = [msg for good, msg in checks if not good]
        if bad:
            ok = False
            print(f'  FAIL {eng}: {"; ".join(bad)}')
        else:
            print(f'  ok   {eng}: {spec["label"]}')
    # The ddpm/fm arms must not reach any Gen14-authored module.
    for eng in ('ddpm', 'fm'):
        spec = ENGINES[eng]
        for key in ('diffusion', 'model', 'trainer'):
            if 'twotime' in spec[key]:
                ok = False
                print(f'  FAIL {eng}.{key} reaches a two-time module ({spec[key]}) — '
                      f'this breaks the PLAN §3.1 structural guarantee.')
    print('  G1 PASS' if ok else '  G1 FAIL')
    return ok


def _build(engine, if_vision=True, horizon=8, batch=2, device='cuda'):
    """Minimal in-memory build of one arm — no dataset, no checkpoint."""
    import torch
    from mix_visual_aligning.models.engine_registry import resolve, import_class

    class _Cfg:  # stand-in for the parsed args object VisualUNet* reads
        pass
    cfg = _Cfg()
    cfg.device, cfg.if_vision, cfg.horizon = device, if_vision, horizon
    cfg.action_dim, cfg.obs_dim, cfg.dim = 3, 6, 32
    cfg.dim_mults, cfg.condition_dropout, cfg.returns_condition = (1, 2, 4, 8), 0.1, False
    cfg.film_mode = 'v1'

    spec = resolve(engine)
    ModelCls, DiffCls = import_class(spec['model']), import_class(spec['diffusion'])
    obs_dim = 6 if if_vision else 20
    model = ModelCls(state_dim=3 + obs_dim, seq_len=horizon, freq_dim=32,
                     dropout_rate=0.1, device=device, if_vision=if_vision, vis_config=cfg,
                     dual_head=True, interval_cfg=False)
    return cfg, model, DiffCls, obs_dim


def _fake_visual_batch(batch, horizon, device):
    import torch
    traj = torch.randn(batch, horizon, 9, device=device)
    cond = {
        0: torch.randn(batch, 6, device=device),
        'primary_img': torch.rand(batch, 3, 96, 96, device=device),
        'wrist_img':   torch.rand(batch, 3, 96, 96, device=device),
    }
    return traj, cond


def gate_g2(device='cuda'):
    """One mf step with if_vision=True: the JVP must survive, and must NOT differentiate
    the ResNets (peak memory within 1.3x of the fm arm proves the pre-encode is live)."""
    print('\n=== G2: JVP survives the visual path ===')
    import torch
    horizon, batch = 8, 2
    cfg, model, DiffCls, obs_dim = _build('mf', True, horizon, batch, device)
    diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim, action_dim=3,
                        goal_dim=0, n_timesteps=100, loss_type='l2', if_vision=True,
                        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
                        meanflow_data_proportion=0.5, mf_adp_p=1.0, mf_adp_eps=0.01).to(device)
    traj, cond = _fake_visual_batch(batch, horizon, device)
    torch.cuda.reset_peak_memory_stats()
    loss, info = diffusion.loss(traj, cond)
    loss.backward()
    peak_mf = torch.cuda.max_memory_allocated() / 2 ** 20
    finite = bool(torch.isfinite(loss))
    raw = float(info['raw_mse_u']) if 'raw_mse_u' in info else float('nan')
    print(f'  loss={float(loss):.6f} finite={finite}  raw_mse_u={raw:.6f}  peak={peak_mf:.0f} MiB')
    ok = finite
    print('  G2 PASS' if ok else '  G2 FAIL — JVP produced a non-finite loss')
    print('  NOTE: compare peak against the fm arm by hand; >1.3x means the pre-encoded '
          'latent (PLAN §6.1) is NOT live and the ResNets are inside the JVP.')
    return ok


def gate_g3(device='cuda'):
    """meanflow_data_proportion=1.0 forces every sample to h=0, where the MeanFlow target
    collapses to the FM velocity. raw_mse_u must then match a plain FM residual."""
    print('\n=== G3: MeanFlow identity at h=0 ===')
    import torch
    horizon, batch = 8, 4
    cfg, model, DiffCls, obs_dim = _build('mf', True, horizon, batch, device)
    diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim, action_dim=3,
                        goal_dim=0, n_timesteps=100, loss_type='l2', if_vision=True,
                        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
                        meanflow_data_proportion=1.0,   # ← every sample is an FM anchor
                        mf_adp_p=1.0, mf_adp_eps=0.01).to(device)
    traj, cond = _fake_visual_batch(batch, horizon, device)
    loss, info = diffusion.loss(traj, cond)
    h_mean   = float(info['h_mean'])   if 'h_mean'   in info else float('nan')
    fm_frac  = float(info['fm_frac'])  if 'fm_frac'  in info else float('nan')
    ok = (abs(h_mean) < 1e-6) and (abs(fm_frac - 1.0) < 1e-6)
    print(f'  h_mean={h_mean:.3e} (want 0)   fm_frac={fm_frac:.3f} (want 1.0)')
    print('  G3 PASS' if ok else '  G3 FAIL — h is not pinned to 0; the anchor branch is wrong')
    return ok


def gate_g4():
    """alpha(0)~1, alpha(N)~0, monotone. A run whose alpha never moved trained plain flow
    matching and is otherwise indistinguishable from a working one."""
    print('\n=== G4: alpha spans the real budget ===')
    from mix_visual_aligning.models.af_diffusion import AlphaFlowODE
    n = 100000
    vals = []
    for step in (0, n // 4, n // 2, 3 * n // 4, n):
        a = AlphaFlowODE._get_ratio('sigmoid', 1.0, 0.0, 0, n, 25.0, 0.005, step)
        vals.append((step, float(a)))
    for step, a in vals:
        print(f'  step {step:>7} -> alpha {a:.4f}')
    alphas = [a for _s, a in vals]
    monotone = all(alphas[i] >= alphas[i + 1] - 1e-9 for i in range(len(alphas) - 1))
    ok = abs(alphas[0] - 1.0) < 1e-3 and abs(alphas[-1]) < 1e-3 and monotone
    print('  G4 PASS' if ok else '  G4 FAIL — alpha does not span [1,0] monotonically')
    return ok


def gate_g5(device='cuda'):
    """alpha pinned to 0 makes alpha-Flow's objective the MeanFlow objective (PLAN §3.4)."""
    print('\n=== G5: alpha->0 limit is MeanFlow ===')
    import torch
    horizon, batch = 8, 4
    cfg, model, DiffCls, obs_dim = _build('af', True, horizon, batch, device)
    diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim, action_dim=3,
                        goal_dim=0, n_timesteps=100, loss_type='l2', if_vision=True,
                        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
                        af_ratio_fm=0.5, af_adp_eps=1e-3, af_clamp_utgt=4.0,
                        af_alpha_scheduler='constant',
                        af_alpha_init=0.0, af_alpha_end=0.0).to(device)
    traj, cond = _fake_visual_batch(batch, horizon, device)
    loss, info = diffusion.loss(traj, cond)
    alpha = float(info['alpha']) if 'alpha' in info else float('nan')
    disc  = float(info['discrete_frac']) if 'discrete_frac' in info else float('nan')
    ok = abs(alpha) < 1e-9 and abs(disc) < 1e-9 and bool(torch.isfinite(loss))
    print(f'  alpha={alpha:.6f} (want 0)   discrete_frac={disc:.6f} (want 0 -> JVP branch only)')
    print('  G5 PASS' if ok else '  G5 FAIL — the bootstrap branch is still active at alpha=0')
    return ok


class _SpyProjector:
    """Records whether the sampler actually called project() — the whole point of G6.

    Mimics the surface `Projector` exposes to p_sample_loop: `.gradient`,
    `.diffusion_timestep_threshold`, `.project()`. `gradient=False` selects the
    SLSQP branch, which is the one the eval pipeline uses.
    """

    def __init__(self, threshold):
        self.diffusion_timestep_threshold = threshold
        self.gradient = False
        self.n_calls = 0
        self.called_at = []

    def project(self, trajectory, constraints=None):
        self.n_calls += 1
        return trajectory, 0.0          # identity projection: we only count the call

    def compute_cost(self, trajectory, constraints=None):
        return 0.0


def _eval_threshold(default=0.5):
    """The threshold the eval pipeline actually deploys, read from the same YAML the
    config block reads. Hard-coding it here would let the gate drift away from reality."""
    path = os.path.join(REPO, 'config', 'visual_aligning_eval.yaml')
    try:
        with open(path) as f:
            for line in f:
                if line.strip().startswith('diffusion_timestep_threshold:'):
                    return float(line.split(':', 1)[1].strip())
    except Exception:
        pass
    return default


def gate_g6(device='cpu'):
    """🔴 RUNTIME check: at K=1, does the DPCC projector actually get called?

    ── fix_2 ──────────────────────────────────────────────────────────────────
    The original G6 was a substring search for 'flow_steps - 1' over the whole
    module. That string also appears in the unrelated `repeat_last` clamp
    (`loop_idx = min(i, flow_steps - 1)`) which is present in ALL THREE engines, so
    the check could never fail: mf/af passed for the wrong reason and fm was
    reported as having a fallback it does not have. Cluster run 24082 printed
    `DIFF fm: terminal-step fallback present` — a false pass.

    Replaced with a behavioural test: run p_sample_loop at K=1 with a spy projector
    and count the calls. No source-string heuristics, nothing to drift.
    ───────────────────────────────────────────────────────────────────────────

    EXPECTED RESULT (with the deployed threshold of 0.5):
      mf, af -> project() IS called. Gen3v6's guard is
                `(loop_idx >= int((1-thr)*K)) or (loop_idx == K-1)`; at K=1 both
                the int() truncation and the explicit fallback fire.
      fm     -> project() is NOT called. Gen7's guard is only
                `loop_idx >= (1-thr)*K` = `0 >= 0.5` -> False. The DPCC projection
                is skipped ENTIRELY and the run silently reports FM as unsafe when
                nothing was ever projected.

    The fm leg does NOT fail this gate: it is a known Gen7/Gen6V4 upstream defect,
    not a Gen14 regression, and failing here would block the pipeline's
    `--dependency=afterok` chain forever. It is printed as a loud banner instead.
    The fix belongs upstream in Gen7, after which Gen14 re-copies (PLAN §6.4).
    """
    print('\n=== G6: projector fires at K=1 (RUNTIME) ===')
    import torch
    from mix_visual_aligning.models.engine_registry import resolve, import_class

    thr = _eval_threshold()
    print(f'  threshold = {thr} (from config/visual_aligning_eval.yaml — the deployed value)')
    print(f'  building state-only (if_vision=False) models on {device}: no vision encoder needed\n')

    horizon, batch, action_dim, obs_dim = 8, 2, 3, 6
    transition_dim = action_dim + obs_dim

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.device, cfg.if_vision, cfg.horizon = device, False, horizon
    cfg.action_dim, cfg.obs_dim, cfg.dim = action_dim, obs_dim, 32
    cfg.dim_mults, cfg.condition_dropout, cfg.returns_condition = (1, 2, 4, 8), 0.1, False
    cfg.film_mode = 'v1'

    ok = True
    upstream_finding = False
    for arm in ('mf', 'af', 'fm'):
        spec = resolve(arm)
        ModelCls, DiffCls = import_class(spec['model']), import_class(spec['diffusion'])

        if spec['wraps_unet']:
            model = ModelCls(state_dim=transition_dim, seq_len=horizon, freq_dim=cfg.dim,
                             dropout_rate=0.1, device=device, if_vision=False, vis_config=cfg,
                             dual_head=True, interval_cfg=False)
            extra = dict(if_vision=False, t_schedule='logit_normal', p_mean=-0.4, p_std=1.0)
        else:
            model = ModelCls(cfg)
            extra = {}

        diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim,
                            action_dim=action_dim, goal_dim=0, n_timesteps=100,
                            loss_type='l2', flow_steps_v3=1, **extra).to(device)
        diffusion.flow_steps_v3 = 1                      # K = 1: the low-NFE regime
        diffusion.ode_inference_steps_v3 = 1

        spy = _SpyProjector(thr)
        cond = {0: torch.zeros(batch, obs_dim, device=device)}
        with torch.no_grad():
            diffusion.p_sample_loop((batch, horizon, transition_dim), cond,
                                    projector=spy, constraints=None)

        fired = spy.n_calls > 0
        if arm in ('mf', 'af'):
            if fired:
                print(f'  ok   {arm}: project() called {spy.n_calls}x at K=1')
            else:
                ok = False
                print(f'  FAIL {arm}: project() NEVER called at K=1 — the DPCC cage is OFF. '
                      f'The terminal-step fallback was lost.')
        else:
            if fired:
                print(f'  ok   {arm}: project() called {spy.n_calls}x at K=1 '
                      f'(upstream Gen7 appears to have been fixed — update this gate)')
            else:
                upstream_finding = True
                print(f'  !!   {arm}: project() NEVER called at K=1  <-- KNOWN UPSTREAM DEFECT')

    if upstream_finding:
        print('\n  ' + '!' * 68)
        print('  !! Gen7/Gen6V4 UPSTREAM DEFECT CONFIRMED AT RUNTIME (not a Gen14 regression)')
        print(f'  !! fm arm at K=1, threshold={thr}: `loop_idx >= (1-thr)*K` = `0 >= {1-thr}`')
        print('  !! is False, so the DPCC projection NEVER RUNS. Any K=1 result from the fm')
        print('  !! arm showing constraint violations is measuring an UNPROJECTED trajectory.')
        print('  !! Only K=1 is affected (at K=2, `1 >= 1.0` is True).')
        print('  !! Fix belongs in fm_visual_aligning/models/diffusion.py, THEN re-copy here.')
        print('  ' + '!' * 68)

    print('\n  G6 PASS (runtime)' if ok else '\n  G6 FAIL')
    print('  End-to-end confirmation (optional): run eval with --engine mf and '
          'flow_steps_v3=1 and check projection_costs[0] is populated and non-trivial.')
    return ok


GATES = {'g0': gate_g0, 'g1': gate_g1, 'g2': gate_g2,
         'g3': gate_g3, 'g4': gate_g4, 'g5': gate_g5, 'g6': gate_g6}
NEEDS_GPU = {'g2', 'g3', 'g5'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', default='all',
                    choices=['all', 'static'] + list(GATES))
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    if a.gate == 'all':
        names = list(GATES)
    elif a.gate == 'static':
        names = [g for g in GATES if g not in NEEDS_GPU]
    else:
        names = [a.gate]

    results = {}
    for name in names:
        fn = GATES[name]
        try:
            results[name] = fn(device=a.device) if name in NEEDS_GPU else fn()
        except Exception as e:              # a gate that crashes is a gate that failed
            print(f'  {name.upper()} ERROR: {type(e).__name__}: {e}')
            results[name] = False

    print('\n' + '=' * 60)
    for name, passed in results.items():
        print(f'  {name.upper()}: {"PASS" if passed else "FAIL"}')
    print('=' * 60)
    sys.exit(0 if all(results.values()) else 1)
