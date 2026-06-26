"""Minimal closed-loop UAV FM eval (Gen11 Epoch 6).

Deliberately NOT forked from the 700-line D3IL/minari-coupled FMv3-ODE eval — that base
is the wrong shape for UAV and impossible to debug. Instead this mirrors the *known-good*
expert control loop (`uav_expert_data_collect/generator.run_trial`) and swaps the expert
trajectory for the trained FM policy. One scene at a time, receding-horizon (MPC) execution.

Multi-rate control (IMPORTANT):
  • physics + PID run every `dt = model.opt.timestep`
  • the FM predicts Δp_des at the DATASET rate (DATASET_HZ from dataset_writer.py)
  → the FM is queried every `decim = round(1/(dt·33))` physics steps; p_des is zero-order
    held between queries while the PID tracks it. This matches how the data was recorded.

Per FM step:
  obs = [p_des | p | v]  (9-D, raw) → policy → first Δp_des (3-D) → p_des += Δp_des →
  PID tracks p_des for `decim` physics steps → obs updated from new (p, v).

SUCCESS CRITERION (Fix2_metrics, scene-aware):
  • GOAL-PATH scenes (corridor, s_curve, pillars — fixed start + geometry route):
      `success = goal_reached AND safe` — must REACH the route endpoint (final position
      within `--goal-radius`) AND fly cleanly (contact-free + airborne, `min_z > 0.2`).
  • `empty` (RANDOM per-episode start→goal the state-only FM is never told → goal-reaching
      ill-defined): `success = safe` — just stay stable (contact-free + airborne).
  The old contact-free+airborne proxy is always reported as `safe` / `safe_rate`, and
  `goal_reached` / `goal_dist` are reported for every scene regardless. A drone that flies
  around a goal-path scene without reaching the target is NOT a success (the prior global
  definition scored that as success — a bug).

No torch/MuJoCo in the Docker dev env — this is cluster-only; here it is syntax-checked.
"""

import os
import sys
import json
import time
import argparse

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

import flow_matcher_v3_uav.utils as utils
from flow_matcher_v3_uav.sampling.policies import Policy
import FM_v3_uav_test.eval_artifacts as artifacts
from uav_expert_data_collect.dataset_writer import DATASET_HZ   # authoritative 33 Hz source

SCENES = ['empty', 'corridor', 's_curve', 'pillars']
# Scenes with a FIXED start + geometry-determined route endpoint → success REQUIRES reaching
# the goal. `empty` is excluded: it has a RANDOM per-episode start→goal that the state-only FM
# is never told (generator._build_traj_and_init), so goal-reaching is ill-defined there — its
# success is stable/safe flight only (Fix2_metrics scene-aware refinement).
GOAL_PATH_SCENES = {'corridor', 's_curve', 'pillars'}
GOAL_RADIUS = 0.30                   # m — secondary goal-reach tolerance (constrained scenes)


def parse_args():
    p = argparse.ArgumentParser(description='Closed-loop UAV FM evaluation.')
    p.add_argument('--scene', type=str, default='all', choices=['all', *SCENES],
                   help="Scene(s) to eval: 'all' runs each scene and rolls up SUMMARY.json.")
    p.add_argument('--seed', type=int, default=5, help='Trained-model seed to load.')
    p.add_argument('--n-trials', type=int, default=20, help='Closed-loop rollouts per scene.')
    p.add_argument('--goal-radius', type=float, default=GOAL_RADIUS,
                   help='Goal-reach tolerance (m). success now REQUIRES goal_dist < this (Fix2_metrics).')
    p.add_argument('--epoch', type=str, default='latest', help="Checkpoint epoch ('latest' or int).")
    p.add_argument('--projection', type=str, default='fm_only',
                   help="Projection variant for the output subfolder. 'fm_only' (state-only FM, no DPCC); "
                        "DPCC variants (dpcc-c, …) slot in here when Phase-3 lands.")
    p.add_argument('--record', type=str, default='none', choices=['none', 'gif', 'all'],
                   help="Overhead-render GIFs per rollout. 'none' (default, fast) adds ~0 overhead; "
                        "'gif'/'all' render frames and write diagnostics/rollout_<r>.gif.")
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_known_args()


def build_experiment(scene, seed, epoch, device):
    """Resolve + load the trained model & dataset (the per-variant Policy is built later)."""
    class Parser(utils.Parser):
        dataset: str = 'uav'              # overridden on the instance below
        config: str = 'config.uav'
    p = Parser()
    p.dataset = f'uav-{scene}'            # → data branch + output path segregation
    args = p.parse_args(experiment='flow_matching_v3_uav', seed=seed)

    ep = epoch if epoch == 'latest' else int(epoch)
    experiment = utils.load_diffusion(args.savepath, epoch=ep, device=device)
    return experiment.diffusion, experiment.dataset, args, int(getattr(args, 'horizon', 8))


def load_pcc_config():
    """Load the Epoch-7 PCC eval config from config/uav_eval.yaml (mirrors how the
    FMv3ODE/visual evals load config/projection_eval.yaml). Falls back to a diffuser-only
    config if the yaml is missing."""
    import yaml
    path = os.path.join(_REPO, 'config', 'uav_eval.yaml')
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f'[ eval ] {path} not found → diffuser-only fallback')
        cfg = {}
    cfg.setdefault('projection_variants', ['diffuser'])
    cfg.setdefault('constraint_types', ['dynamics'])
    cfg.setdefault('batch_size', 4)
    cfg.setdefault('dt', 1.0)
    cfg.setdefault('diffusion_timestep_threshold', 0.5)
    cfg.setdefault('enlarge_constraints', 0.0)
    cfg.setdefault('workspace_bounds', None)
    cfg.setdefault('halfspace_constraints', [])
    cfg.setdefault('obstacle_constraints', [])
    return cfg


# ── DPCC projector — copied from fm_visual_aligning_test/eval_fm_visual_aligning.py and
#    adapted to the UAV 12-D transition. Only the DYNAMICS constraint is active this epoch;
#    bounds/halfspace/obstacle blocks are kept verbatim as PLACEHOLDERS (fire only if their
#    config keys are enabled — they are not this epoch). ────────────────────────────────────

class ProjectorNormalizer:
    """Wrap obs + act LimitsNormalizers into the dict Projector('states_actions') expects
    (verbatim from the visual-aligning eval)."""
    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {'observations': obs_normalizer, 'actions': act_normalizer}


def setup_dpcc_projector(args, config, obs_normalizer, act_normalizer, variant, trajectory_dim=12):
    """Build the DPCC projector (mirrors visual-aligning `setup_dpcc_projector`).

    UAV 12-D transition: [dx(0) dy(1) dz(2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11)].
    The dynamics `deriv` binds **p_des (3,4,5)** to the action (0,1,2) — NOT the actual p —
    because p_des is the exact integrator of the action (`p_des[t+1]=p_des[t]+act`), while
    the drone's p lags. (Visual-aligning binds c_pos because its arm tracks perfectly.)

    Variant semantics (mirrors FMv3ODE/visual-aligning eval):
      gradient       → gradient-based projection (not SLSQP)
      post_processing→ threshold=0.0 (project at ALL FM steps, not just last 50%)
      model_free     → spatial constraints only; dynamics skipped (no-op until spatial designed)
      tightened      → enlarge_constraints margin applied to spatial constraints
    """
    from flow_matcher_v3_uav.sampling.projection import Projector

    _DIM = {'dx': 0, 'dy': 1, 'dz': 2, 'x': 6, 'y': 7, 'z': 8}   # x,y,z = actual position p
    pad = trajectory_dim - 9
    is_tightened = 'tightened' in variant
    tightening   = float(config.get('enlarge_constraints') or 0.0)
    enlarge      = tightening if is_tightened else 0.0
    constraint_list = []

    if 'bounds' in config.get('constraint_types', []):                 # PLACEHOLDER — not run this epoch
        ws = config['workspace_bounds']
        ws_lb = np.array(ws['lb']); ws_ub = np.array(ws['ub'])
        lb = np.concatenate([np.full(6, -np.inf), ws_lb - enlarge, np.full(pad, -np.inf)])
        ub = np.concatenate([np.full(6,  np.inf), ws_ub + enlarge, np.full(pad,  np.inf)])
        constraint_list += [['lb', lb], ['ub', ub]]

    if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
        constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]  # bind p_des

    if 'halfspace' in config.get('constraint_types', []):              # PLACEHOLDER — not run this epoch
        _hs = {'x': _DIM['x'], 'y': _DIM['y']}
        for hs in config.get('halfspace_constraints', []):
            C_row, d = utils.formulate_halfspace_constraints(hs, enlarge, trajectory_dim, _hs)
            constraint_list.append(('ineq', (C_row, d)))

    if 'obstacles' in config.get('constraint_types', []):              # PLACEHOLDER — not run this epoch
        for obs in config.get('obstacle_constraints', []):
            dims = [_DIM[d] if isinstance(d, str) else int(d) for d in obs['dimensions']]
            constraint_list.append((obs['type'], dims, obs['center'], obs['radius'] + enlarge))

    is_gradient      = 'gradient' in variant
    is_post_proc     = 'post_processing' in variant
    threshold        = 0.0 if is_post_proc else config.get('diffusion_timestep_threshold', 0.5)

    return Projector(
        horizon=int(getattr(args, 'horizon', 8)),
        transition_dim=trajectory_dim,
        action_dim=3,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=ProjectorNormalizer(obs_normalizer, act_normalizer),
        diffusion_timestep_threshold=threshold,
        variant='states_actions',
        dt=config.get('dt', 1.0),                   # action IS Δp_des → Euler dt=1.0 (NOT 1/33)
        gradient=is_gradient,
        gradient_weights=[1, 0.5, 2] if is_gradient else None,
        solver='scipy',
        device=getattr(args, 'device', 'cuda'),
    )


def _selection_for(variant):
    """FMv3ODE variant → trajectory_selection (verbatim semantics)."""
    if 'dpcc-t' in variant:
        return 'temporal_consistency'
    if 'dpcc-c' in variant:
        return 'minimum_projection_cost'
    return 'random'


def _make_overhead_renderer(mujoco, model, res=360):
    """Headless overhead renderer; None if rendering is unavailable (no hard dep).

    ONE renderer is created per scene and reused across rollouts — never one per
    rollout — so we allocate exactly one EGL/GL context instead of leaking N of them.
    """
    try:
        return mujoco.Renderer(model, height=res, width=res)
    except Exception as exc:                               # pragma: no cover - cluster-only
        print(f'[ eval ] render unavailable ({exc}); GIF skipped')
        return None


def _free_renderer(renderer):
    """Release the renderer's GL context *now*, while EGL is still initialized.

    MuJoCo's GLContext.__del__ calls eglMakeCurrent; if it runs at interpreter
    shutdown (after EGL is torn down) it raises EGL_NOT_INITIALIZED. Freeing here —
    plus reusing a single renderer per scene — prevents both that teardown error and
    the per-rollout GL-context leak. Tolerant of mujoco versions with/without close().
    """
    if renderer is None:
        return
    try:
        if hasattr(renderer, 'close'):        # mujoco >= 3.x
            renderer.close()
    except Exception:                         # pragma: no cover
        pass
    try:
        import gc
        gc.collect()                          # force GLContext.__del__ while EGL is up
    except Exception:                         # pragma: no cover
        pass


def _render_overhead(mujoco, model, data, renderer):
    """Single top-down frame. Reuses the PROVEN overhead camera from the expert GIF
    tool (uav_expert_data_collect/generate_trajectory_gifs._render_overhead); falls
    back to the same camera inline only if that import is unavailable."""
    try:
        from uav_expert_data_collect.generate_trajectory_gifs import (
            _render_overhead as _proven_overhead)
        return _proven_overhead(model, data, renderer)
    except Exception:                                      # pragma: no cover
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = data.qpos[:3]
        cam.distance = 5.0
        cam.azimuth = 0.0
        cam.elevation = -90.0
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()


def rollout_one(model, scene, homotopy, trial_seed, policy, horizon,
                renderer=None, frame_stride=2, goal_radius=GOAL_RADIUS, batch_size=1,
                variant='diffuser', log_dir=None, control_hz=DATASET_HZ, text_log=True):
    """One closed-loop MuJoCo rollout. Mirrors generator.run_trial; FM replaces traj_fn.

    `model` and `renderer` are owned by eval_scene and shared across rollouts (one
    GL context per scene, not per rollout). `batch_size` = MPC candidate-fan size: the
    policy samples a batch and (per its trajectory_selection) returns the chosen
    candidate's first action; `plans` stores the whole fan. Buffers obs/action/plan per
    FM step (U3 npz schema); if `renderer` is given, also captures overhead frames.
    Heavy arrays/frames are returned under HEAVY_KEYS and stripped from results.json.
    """
    import mujoco
    import uav_expert_data_collect.generator as gen

    rng = np.random.default_rng(trial_seed)
    data = mujoco.MjData(model)

    traj_fn, init_pos, dur = gen._build_traj_and_init(scene, homotopy, rng)
    goal = np.asarray(traj_fn(dur)[0], dtype=float)        # expert path endpoint (secondary metric)

    data.qpos[:3] = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pid = gen._make_pid(model, 'pid_default')
    dt = float(model.opt.timestep)
    dt_fm = 1.0 / DATASET_HZ
    decim = max(1, int(round(1.0 / (dt * DATASET_HZ))))    # physics steps per FM query
    n_fm = int(round(dur * DATASET_HZ))

    frames = []

    # Real-time behaviour logger (digital-twin audit). ALWAYS ON — independent of --record;
    # near-zero cost (wraps timings the loop already takes). See Real_Time_eval_loggging/PLAN.md.
    from FM_v3_uav_test.behavior_logger import BehaviorLogger
    episode_id = f'{scene}_{homotopy}_{trial_seed}'
    blog = BehaviorLogger(episode_id, variant, scene, homotopy,
                          control_hz=control_hz, batch_size=batch_size, horizon=horizon,
                          text_log=text_log)
    proj_on = (variant != 'diffuser')

    p_des = np.asarray(init_pos, dtype=float).copy()
    n_hit = 0
    n_phys = 0
    min_z = float('inf')
    track_err = []
    fm_ms = []           # PURE FM inference ms (projection time subtracted out — Real_Time logging)
    proj_ms = []         # PCC projector wall-time ms per FM step
    total_ms = []        # fm_ms + proj_ms  → the real-time budget number
    obs_traj = []        # realized [p_des|p|v] per FM step  → npz obs_all
    act_traj = []        # FM Δp_des per FM step             → npz act_all
    plans = []           # FM H-step predicted obs plan      → npz sampled_trajectories_all

    for k in range(n_fm):
        p = data.qpos[:3].copy()
        v = data.qvel[:3].copy()
        obs = np.concatenate([p_des, p, v]).astype(np.float32)   # [p_des | p | v] (9,) raw

        t0 = time.perf_counter()
        action, traj = policy({0: obs}, batch_size=batch_size, horizon=horizon)
        step_total_ms = (time.perf_counter() - t0) * 1e3         # bundled FM + projection
        step_proj_ms = float(getattr(policy, 'last_proj_ms', 0.0))
        step_fm_ms = max(step_total_ms - step_proj_ms, 0.0)      # PURE inference
        fm_ms.append(step_fm_ms)
        proj_ms.append(step_proj_ms)
        total_ms.append(step_total_ms)

        action = np.asarray(action, dtype=float).reshape(-1)[:3]  # first Δp_des
        obs_traj.append(obs)
        act_traj.append(action.astype(np.float32))
        # traj.observations = FM's unnormalized H-step plan in obs space (the foresight).
        plan = getattr(traj, 'observations', None)
        if plan is not None:
            plans.append(np.asarray(plan, dtype=np.float32))
        # FM Δp_des H-step foresight of the EXECUTED candidate (for the log's `horizon=` field).
        which = int(getattr(policy, 'last_which_trajectory', 0))
        fm_horizon = None
        if getattr(traj, 'actions', None) is not None:
            acts = np.asarray(traj.actions)
            if acts.ndim == 3 and which < acts.shape[0]:
                fm_horizon = acts[which]

        p_des = p_des + action
        v_des = action / dt_fm

        hit_before = n_hit
        for _ in range(decim):
            p = data.qpos[:3].copy()
            v = data.qvel[:3].copy()
            q = data.qpos[3:7].copy()
            om = data.qvel[3:6].copy()
            u = pid.compute(p, q, v, om, p_des, v_des)
            data.ctrl[:4] = u
            mujoco.mj_step(model, data)
            n_phys += 1
            if any(gen._is_obstacle_contact(model, data.contact[ci]) for ci in range(data.ncon)):
                n_hit += 1
            min_z = min(min_z, float(data.qpos[2]))
            track_err.append(float(np.linalg.norm(data.qpos[:3] - p_des)))

        # ── one structured log line per FM control step ──
        te_step = float(np.linalg.norm(data.qpos[:3] - p_des))
        blog.step(
            t=k / DATASET_HZ, step_idx=f'{k}/{n_fm}', obs=obs, fm_horizon=fm_horizon,
            fm_ms=step_fm_ms, proj_ms=step_proj_ms,
            proj_cost=float(getattr(policy, 'last_proj_cost', 0.0)), proj_active=proj_on,
            state_p=data.qpos[:3].copy(), state_v=data.qvel[:3].copy(),
            contact='obstacle' if n_hit > hit_before else None, track_err=te_step,
        )

        if renderer is not None and (k % frame_stride == 0):
            try:
                frames.append(_render_overhead(mujoco, model, data, renderer))
            except Exception as exc:                       # pragma: no cover
                print(f'[ eval ] frame render failed ({exc}); stopping capture')
                renderer = None     # stop capturing for THIS rollout; eval_scene still owns/frees it

    p_final = data.qpos[:3].copy()
    contact_frac = n_hit / max(n_phys, 1)
    limit = gen.SCENE_MAX_CONTACT_FRACTION.get(scene, gen.MAX_CONTACT_FRACTION)
    airborne = bool(min_z > 0.2)                           # crude floor gate
    goal_dist = float(np.linalg.norm(p_final - goal))
    goal_reached = bool(goal_dist < goal_radius)
    safe = bool(contact_frac <= limit and airborne)       # contact-free + airborne
    # Scene-aware success (Fix2_metrics): fixed-route scenes must REACH the goal AND be safe;
    # `empty` has a RANDOM goal the unconditioned FM can't be expected to hit, so there
    # success = stable/safe flight only. A goal-path drone that flies around without reaching
    # the target is NOT a success.
    if scene in GOAL_PATH_SCENES:
        success = bool(goal_reached and safe)
    else:                                                 # empty (random goal): stay stable
        success = bool(safe)

    # Constraint-aware metrics (FMv3ODE schema). Only the DYNAMICS constraint is active this
    # epoch — there are NO obstacle/halfspace/bounds (free-space) constraints to violate, so
    # these are trivially clean. They populate the schema; per-scene geometry fills them later.
    collision_free = True
    n_violations = 0
    total_violations = 0.0
    success_and_constraints = bool(success and collision_free)

    # ── persist the real-time behaviour log + capture its timing summary ──
    behaviour = {
        'result': 'SUCCESS' if success else ('FAIL(goal)' if (scene in GOAL_PATH_SCENES and safe and not goal_reached) else 'FAIL'),
        'goal_dist': f'{goal_dist:.3f}m', 'safe': safe, 'min_z': f'{min_z:.3f}',
        'contact_frac': f'{contact_frac:.3f}',
    }
    blog_summary = blog.summary_dict()
    if log_dir is not None:
        blog.save(os.path.join(log_dir, f'rollout_{episode_id}.log'), behaviour=behaviour)

    return {
        'scene': scene, 'homotopy': homotopy,
        'success': success,
        'success_and_constraints': success_and_constraints,
        'safe': safe,
        'contact_frac': contact_frac,
        'goal_dist': goal_dist,
        'goal_reached': goal_reached,
        'collision_free': collision_free,
        'n_violations': n_violations,
        'total_violations': total_violations,
        'min_z': min_z,
        'final_z': float(p_final[2]),
        'track_err_mean': float(np.mean(track_err)) if track_err else float('nan'),
        'fm_ms_mean': float(np.mean(fm_ms)) if fm_ms else float('nan'),      # PURE inference (proj subtracted)
        'fm_ms_p95': float(np.percentile(fm_ms, 95)) if fm_ms else float('nan'),
        'proj_ms_mean': float(np.mean(proj_ms)) if proj_ms else 0.0,
        'total_ms_mean': float(np.mean(total_ms)) if total_ms else float('nan'),
        'total_ms_p95': float(np.percentile(total_ms, 95)) if total_ms else float('nan'),
        'total_over_budget': int(blog_summary['total_over_budget']),
        'budget_ms': blog_summary['budget_ms'],
        'n_fm_steps': n_fm, 'decim': decim, 'dt': dt,
        # ── heavy (npz / gif only; stripped from results.json) ──
        'obs_traj': np.asarray(obs_traj),
        'act_traj': np.asarray(act_traj),
        'plans': plans,
        'frames': frames,
    }


def _run_variant(scene, variant, model_fm, dataset, parsed, horizon, config, args,
                 mj_model, mujoco, homotopies):
    """Run all trials for ONE projection variant → write its plans/<variant>/ artifacts.

    Mirrors the FMv3ODE per-variant block: `projector = None` for `diffuser`, else the DPCC
    projector; `trajectory_selection` per variant; one Policy built per variant (persists
    across trials, exactly as FMv3ODE)."""
    projector = None
    if variant != 'diffuser':
        # 12 = act(3)+obs(9), MINUS model_fm.goal_dim: diffusion.py's p_sample_loop calls
        # `projector.project(x[:, :, :-self.goal_dim], ...)` — it slices off the trailing
        # goal_dim columns BEFORE handing the trajectory to the projector, so the projector's
        # Q/A/C matrices must be built for that same (smaller) width or `project()` shape-errors
        # (seen on the cluster: "4x88 and 96x96" when goal_dim=1 was left out of this count).
        # goal_dim itself comes from SequenceDataset.get_goal_dim()'s constant-column heuristic
        # (real D3IL goal columns are appended/constant at the end of obs; for UAV it can
        # false-positive on an incidentally-constant channel) — harmless here since the
        # dynamics constraint only touches indices 0-5 (act, p_des), never the trailing columns.
        goal_dim = int(getattr(model_fm, 'goal_dim', 0))
        traj_dim = int(dataset.observation_dim + dataset.action_dim) - goal_dim
        projector = setup_dpcc_projector(
            parsed, config,
            dataset.normalizer.normalizers['observations'],
            dataset.normalizer.normalizers['actions'],
            variant, trajectory_dim=traj_dim)
    policy = Policy(model=model_fm, normalizer=dataset.normalizer,
                    preprocess_fns=getattr(parsed, 'preprocess_fns', []),
                    test_ret=getattr(parsed, 'test_ret', 0),
                    projector=projector, trajectory_selection=_selection_for(variant))

    # Outputs under the sibling plans/ tree, one subfolder per variant (FMv3ODE convention).
    scene_root = os.path.join(parsed.logbase, parsed.dataset)              # logs/UAV_FM/uav-<scene>
    sub = os.path.relpath(parsed.savepath, scene_root)                     # flow_matching_v3_uav/<exp>/<seed>
    out_dir = os.path.join(scene_root, 'plans', sub, variant)              # …/plans/…/<seed>/<variant>
    diag_dir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    record = (args.record != 'none')
    renderer = _make_overhead_renderer(mujoco, mj_model) if record else None
    batch_size = config['batch_size']

    rollouts = []
    try:
        for i in range(args.n_trials):
            homotopy = homotopies[i % len(homotopies)]
            r = rollout_one(mj_model, scene, homotopy, 10_000 + i, policy, horizon,
                            renderer=renderer, goal_radius=args.goal_radius, batch_size=batch_size,
                            variant=variant, log_dir=out_dir,
                            control_hz=config.get('control_hz', DATASET_HZ),
                            text_log=config.get('behavior_log', True))
            artifacts.save_rollout_stats(diag_dir, i, r)
            artifacts.write_mpc_foresight(diag_dir, i, r, scene)   # real candidate-fan plot (E7)
            if record:
                artifacts.save_rollout_gif(diag_dir, i, r.pop('frames', None))
            else:
                r.pop('frames', None)
            rollouts.append(r)
    finally:
        _free_renderer(renderer)
        renderer = None

    succ = np.mean([r['success'] for r in rollouts])
    summary = {
        'scene': scene, 'seed': args.seed, 'n_trials': len(rollouts), 'variant': variant,
        'success_rate': float(succ),                                       # task success: goal+safe (scene-aware)
        'success_and_constraints_rate': float(np.mean([r['success_and_constraints'] for r in rollouts])),
        'safe_rate': float(np.mean([r['safe'] for r in rollouts])),        # contact-free + airborne
        'collision_free_rate': float(np.mean([r['collision_free'] for r in rollouts])),
        'n_violations_mean': float(np.mean([r['n_violations'] for r in rollouts])),
        'total_violations_mean': float(np.mean([r['total_violations'] for r in rollouts])),
        'contact_frac_mean': float(np.mean([r['contact_frac'] for r in rollouts])),
        'goal_dist_mean': float(np.mean([r['goal_dist'] for r in rollouts])),
        'goal_reached_rate': float(np.mean([r['goal_reached'] for r in rollouts])),
        'track_err_mean': float(np.mean([r['track_err_mean'] for r in rollouts])),
        'fm_ms_mean': float(np.mean([r['fm_ms_mean'] for r in rollouts])),
        'fm_ms_p95': float(np.max([r['fm_ms_p95'] for r in rollouts])),
        'proj_ms_mean': float(np.mean([r['proj_ms_mean'] for r in rollouts])),
        'total_ms_mean': float(np.mean([r['total_ms_mean'] for r in rollouts])),
        'total_ms_p95': float(np.max([r['total_ms_p95'] for r in rollouts])),
        'total_over_budget': int(np.sum([r['total_over_budget'] for r in rollouts])),
        'budget_ms': rollouts[0]['budget_ms'] if rollouts else float('nan'),
        'projection': variant,
    }

    # ── Artifacts (legacy schema): results.json + npz + log + 2-D overview ──
    json_rollouts = artifacts.json_safe_rollouts(rollouts)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'rollouts': json_rollouts}, f, indent=2)
    npz_path = artifacts.save_npz(out_dir, variant, rollouts, vars(args))
    artifacts.write_eval_log(out_dir, variant, summary, rollouts)
    artifacts.plot_overview(out_dir, variant, scene, rollouts)

    print(f'[ eval ] {scene} variant={variant} (B={batch_size}, proj={"on" if projector else "off"}, '
          f'sel={_selection_for(variant)}): success={succ:.3f}  safe={summary["safe_rate"]:.3f}  '
          f'goal_reached={summary["goal_reached_rate"]:.3f}  track_err={summary["track_err_mean"]:.3f}  '
          f'→ {os.path.dirname(npz_path)}/')
    # Real-time timing verdict echoed to stdout (per-step detail stays in the .log files).
    _budget = summary['budget_ms']
    _rt = 'SAFE' if summary['total_over_budget'] == 0 else f'OVER×{summary["total_over_budget"]}'
    print(f'[ eval ] {scene} variant={variant} TIMING: fm_ms={summary["fm_ms_mean"]:.1f} '
          f'proj_ms={summary["proj_ms_mean"]:.1f} total_ms={summary["total_ms_mean"]:.1f} '
          f'(p95={summary["total_ms_p95"]:.1f}) budget={_budget}ms → real_time_{_rt}')
    return summary


def eval_scene(scene, args):
    """Run EVERY projection variant (diffuser, dpcc-r/-c/-t) for one scene; returns
    {variant: summary}. Model+dataset loaded once; a Policy is built per variant."""
    import mujoco
    import uav_expert_data_collect.generator as gen
    model_fm, dataset, parsed, horizon = build_experiment(scene, args.seed, args.epoch, args.device)
    config = load_pcc_config()
    homotopies = gen.HOMOTOPY_CLASSES[scene]
    mj_model = mujoco.MjModel.from_xml_path(gen.SCENE_XMLS[scene])

    print(f'[ eval ] {scene}: variants={config["projection_variants"]}  '
          f'constraints={config["constraint_types"]}  batch_size={config["batch_size"]}')
    summaries = {}
    for variant in config['projection_variants']:
        summaries[variant] = _run_variant(scene, variant, model_fm, dataset, parsed, horizon,
                                          config, args, mj_model, mujoco, homotopies)
    return summaries


def main():
    args, remaining = parse_args()
    # utils.Parser.parse_args() (called inside build_experiment) re-parses sys.argv with its
    # own argparse that only knows --config/--seed — strip our already-consumed flags
    # first or it chokes on --scene/--n-trials/--projection/--device (mirrors train_fm_uav.py).
    sys.argv = [sys.argv[0], *remaining]
    scenes = SCENES if args.scene == 'all' else [args.scene]
    summaries = {s: eval_scene(s, args) for s in scenes}

    if len(scenes) > 1:
        # experimental --scene all path; per-scene runs use aggregate_scene_summaries.py instead.
        roll = os.path.join('logs', 'UAV_FM', 'uav-all', 'plans', args.projection, 'SUMMARY.json')
        os.makedirs(os.path.dirname(roll), exist_ok=True)
        with open(roll, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f'[ eval ] cross-scene rollup → {roll}')
    print('UAV FM eval complete.')


if __name__ == '__main__':
    main()
