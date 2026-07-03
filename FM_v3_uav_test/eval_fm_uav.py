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
  obs = [p_des | p | v]  (9-D, raw) → policy → first Δp_des (3-D)
  → p_des += Δp_des  (free-running Euler in commanded space)
  → PID tracks p_des for `decim` physics steps → obs updated from new (p, v).

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

SUCCESS_RELAXED (U7): episodes never terminate early on goal-reach — they always run the
  full fixed FM-step budget. So `success` (which only checks the FINAL position) scores an
  outright FAIL for a rollout that reaches the goal and then drifts/overshoots for the rest
  of a fixed-length episode, identical to one that never got close. `success_relaxed` fixes
  this by treating the goal like a race finish line: a vertical plane (xy line, any z)
  through the goal, oriented perpendicular to the expert path's final approach heading.
  `crossed_line` latches true the first time the drone is EVER on the goal side of that
  line, regardless of what happens afterward. `success_relaxed = crossed_line AND safe`
  (goal-path scenes); `success ⇒ success_relaxed` always. See
  logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/U7_Succes_realaxed/.

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

def _uav_eval_tag(config, controller):
    """Eval-parameter folder name — mirrors args_to_watch_fm_visual_plan style.

    Format:  K{flow_steps}_mpc{B}_{controller}_T{thresh}
    Aligning analogue: K{flow_steps}_M{solver}_T{thresh}_mpc{B}_film{mode}

    Sits BETWEEN the train-identity folder (H8_D...ODE_9D) and the seed,
    so the projection variant (diffuser / dpcc-c) remains a pure leaf name.
    e.g.  flow_matching_v3_uav/H8_D...ODE_9D / K20_mpc4_pid_stopgo_T0.5 / 0 / diffuser /
    """
    k      = int(config.get('flow_steps_v3', 20))
    mpc_b  = int(config.get('mpc_batch_size', config.get('batch_size', 4)))
    thresh = config.get('diffusion_timestep_threshold', 0.5)
    parts  = [f'K{k}', f'mpc{mpc_b}', controller]
    parts.append(f'T{thresh:g}')
    return '_'.join(parts)


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
    p.add_argument('--seed', type=int, default=None,
                   help='Trained-model checkpoint seed to load. Default: seed from config/uav_projection.yaml.')
    p.add_argument('--n-trials', type=int, default=None,
                   help='Closed-loop rollouts per scene. Default: n_trials from config/uav_projection.yaml.')
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


def load_pcc_config(scene, seed):
    """Merged eval config matching the avoiding-d3il.py pattern:
      - Projection params (variants, constraints, geometry) from config/uav_projection.yaml
      - Eval control params (batch_size, thresholds, U4 knobs, logging) from the
        plan_flow_matching_v3_uav block in config/uav.py

    Both sources are merged into one dict so downstream code (_run_variant, rollout_one,
    setup_dpcc_projector) needs no structural changes."""
    import yaml

    # ── 1. Projection-only (variants, constraints, geometry) ─────────────────
    yaml_path = os.path.join(_REPO, 'config', 'uav_projection.yaml')
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f'[ eval ] {yaml_path} not found → diffuser-only fallback')
        cfg = {}
    cfg.setdefault('write_to_file', True)
    cfg.setdefault('projection_variants', ['diffuser'])
    cfg.setdefault('constraint_types', ['dynamics'])
    cfg.setdefault('dt', 1.0)
    cfg.setdefault('diffusion_timestep_threshold', 0.5)
    cfg.setdefault('enlarge_constraints', 0.0)
    cfg.setdefault('workspace_bounds', None)
    cfg.setdefault('halfspace_constraints', [])
    cfg.setdefault('obstacle_constraints', [])

    # ── 2. Eval control params from plan_flow_matching_v3_uav block ──────────
    class PlanParser(utils.Parser):
        dataset: str = 'uav'
        config: str = 'config.uav'
    pp = PlanParser()
    pp.dataset = f'uav-{scene}'
    plan_args = pp.parse_args(experiment='plan_flow_matching_v3_uav', seed=seed)
    cfg['mpc_batch_size']               = int(getattr(plan_args, 'mpc_batch_size', getattr(plan_args, 'batch_size', 4)))
    cfg['diffusion_timestep_threshold'] = float(getattr(plan_args, 'diffusion_timestep_threshold', 0.5))

    cfg['control_hz']                   = float(getattr(plan_args, 'control_hz', DATASET_HZ))
    cfg['behavior_log']                 = bool(getattr(plan_args, 'behavior_log', True))

    # E8 (Epoch8) — observation layout + tracker selection. Defaults = E7 (p_des / pid).
    cfg['cond_mode']                    = str(getattr(plan_args, 'cond_mode', 'p_des'))
    cfg['controller']                   = str(getattr(plan_args, 'controller', 'pid'))
    # U6: MJX predictive-sampling params (replaces gRPC mjpc_task_id/planner_steps).
    cfg['mjx_n_samples']                = int(getattr(plan_args, 'mjx_n_samples', 16))
    cfg['mjx_horizon']                  = float(getattr(plan_args, 'mjx_horizon', 0.3))
    cfg['mjx_n_improve']                = int(getattr(plan_args, 'mjx_n_improve', 5))
    cfg['mjx_vel_weight']               = float(getattr(plan_args, 'mjx_vel_weight', 0.1))

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
    Both position channels are real and anchored with 6 rows (DC_FIX), mirroring DPCC avoiding.
    p_des(3,4,5): commanded setpoint. p(6,7,8): actual drone position from qpos[:3].

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
        # DC_FIX: both real channels anchored — 6 rows (DPCC avoiding 4-row pattern scaled to 3D).
        # Traj layout: [act(0,1,2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11)]
        # DC_FIX: both channels always anchored. anchor_to_p/cond_on_p removed.
        constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]  # DC_FIX p_des ← act
        constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]  # DC_FIX p     ← act

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
                variant='diffuser', log_dir=None, control_hz=DATASET_HZ, text_log=True,
                controller='pid', cond_mode='p_des', mjpc_kwargs=None,
                v_des_magnitude=0.0):
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
    # U7: finish-line crossing test (success_relaxed) — a vertical plane (xy line, any z)
    # through `goal`, oriented perpendicular to the expert path's final approach heading.
    # `crossed_line` latches true the first time the drone's xy position is ever on the
    # goal side of that line, independent of where it ends up afterward.
    _p_before_goal = np.asarray(traj_fn(max(dur - 0.1, 0.0))[0], dtype=float)
    _line_dir_xy = (goal - _p_before_goal)[:2]
    _line_norm = np.linalg.norm(_line_dir_xy)
    line_dir_xy = _line_dir_xy / _line_norm if _line_norm > 1e-9 else np.array([1.0, 0.0])
    crossed_line = False

    data.qpos[:3] = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # E8: tracker selection.
    #   'pid'         (default) — E7 cascaded PID, v_des = action/dt_fm.
    #   'pid_stopgo'  (U2)      — same CascadedPID, v_des = 0 (strict stop-and-go).
    #   'pid_const_v' (U3)      — same CascadedPID, v_des = unit(action)*v_des_magnitude (constant speed).
    #   'mjpc'                  — MJPC optimal-control thrust tracker (cluster-only).
    # All four expose the same .compute(p,q,v,om,p_des,v_des) API.
    pid = gen._make_pid(model, 'pid_default')
    tracker = pid                              # pid_stopgo also uses CascadedPID (v_des differs)
    if controller == 'mjpc':
        from FM_v3_uav_test.mjpc_tracker import MJPCTracker
        mjpc_kwargs = mjpc_kwargs or {}
        tracker = MJPCTracker(model, scene=scene, **mjpc_kwargs)
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
        # E8: obs layout MUST match how the model was trained (dataset cond_mode).
        #   'pos_only' → [p_des|p] (6D, velocity dropped → 9D transition; FM→MJPC).
        #   'p_des' (default) → [p_des|p|v] (9D → 12D transition; E7 PID).
        if cond_mode == 'pos_only':
            obs = np.concatenate([p_des, p]).astype(np.float32)      # [p_des | p] (6,) raw
        else:
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
        # v_des feedforward to PID — source depends on controller:
        #   pid         (default): action / dt_fm  (E7, timing-derived).
        #   pid_stopgo  (U2):      zero → PID brakes to zero each FM step (stop-and-go).
        #   pid_const_v (U3):      unit(action)*v_des_magnitude → constant speed, timing-free.
        #   mjpc:                  v_des accepted for API parity but ignored internally.
        if controller == 'pid_stopgo':
            v_des = np.zeros(3)
        elif controller == 'pid_const_v':
            norm = float(np.linalg.norm(action))
            v_des = (action / norm) * v_des_magnitude if norm > 1e-6 else np.zeros(3)
        else:                                    # 'pid' default (and 'mjpc')
            v_des = action / dt_fm

        hit_before = n_hit
        for _ in range(decim):
            p = data.qpos[:3].copy()
            v = data.qvel[:3].copy()
            q = data.qpos[3:7].copy()
            om = data.qvel[3:6].copy()
            u = tracker.compute(p, q, v, om, p_des, v_des)   # E8: pid OR mjpc (same API)
            data.ctrl[:4] = u
            mujoco.mj_step(model, data)
            n_phys += 1
            if any(gen._is_obstacle_contact(model, data.contact[ci]) for ci in range(data.ncon)):
                n_hit += 1
            min_z = min(min_z, float(data.qpos[2]))
            track_err.append(float(np.linalg.norm(data.qpos[:3] - p_des)))
            # U7: one-way latch — true the instant the drone is ever on the goal side
            # of the finish line, regardless of what it does for the rest of the episode.
            _side = float(np.dot(data.qpos[:2] - goal[:2], line_dir_xy))
            crossed_line = crossed_line or (_side >= 0.0)

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

    # E8: release the MJPC gRPC agent server (no-op for the PID path).
    if controller == 'mjpc' and hasattr(tracker, 'close'):
        tracker.close()

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

    # U7 (success_relaxed): "crossed the finish line" instead of "ended exactly on it".
    # Episodes never terminate early on goal-reach, so a rollout that arrives and then
    # drifts/overshoots for the rest of a fixed-length episode fails `success` outright —
    # identical to one that never got close. `crossed_line` only requires the drone's xy
    # path to have ever passed the goal at some point; `success ⇒ success_relaxed` always.
    if scene in GOAL_PATH_SCENES:
        success_relaxed = bool(crossed_line and safe)
    else:                                                  # empty: no fixed goal to cross
        success_relaxed = success

    # Constraint-aware metrics (FMv3ODE schema). Only the DYNAMICS constraint is active this
    # epoch — there are NO obstacle/halfspace/bounds (free-space) constraints to violate, so
    # these are trivially clean. They populate the schema; per-scene geometry fills them later.
    collision_free = True
    n_violations = 0
    total_violations = 0.0
    success_and_constraints = bool(success and collision_free)
    success_and_constraints_relaxed = bool(success_relaxed and collision_free)

    # ── persist the real-time behaviour log + capture its timing summary ──
    behaviour = {
        'result': 'SUCCESS' if success else ('FAIL(goal)' if (scene in GOAL_PATH_SCENES and safe and not goal_reached) else 'FAIL'),
        'result_relaxed': 'SUCCESS' if success_relaxed else 'FAIL',
        'goal_dist': f'{goal_dist:.3f}m', 'safe': safe, 'min_z': f'{min_z:.3f}',
        'contact_frac': f'{contact_frac:.3f}',
    }
    blog_summary = blog.summary_dict()
    if log_dir is not None:
        blog.save(os.path.join(log_dir, f'rollout_{episode_id}.log'), behaviour=behaviour)

    return {
        'scene': scene, 'homotopy': homotopy,
        'success': success,
        'success_relaxed': success_relaxed,
        'success_and_constraints': success_and_constraints,
        'success_and_constraints_relaxed': success_and_constraints_relaxed,
        'crossed_line': crossed_line,
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
    # E8: tracker + obs-layout selection (defaults preserve E7).
    controller      = str(config.get('controller', 'pid'))
    cond_mode       = str(config.get('cond_mode', 'p_des'))
    # U3: pid_const_v speed — auto-derived from dataset so it self-calibrates to any
    # dataset/scene without a magic number.  mean(|action|) × DATASET_HZ ≡ mean(action/dt_fm)
    # i.e. the same value the default 'pid' controller produces on average.
    # Zero-padding (at-goal steps) is filtered before averaging.
    if controller == 'pid_const_v':
        _all_acts = dataset.fields.actions.reshape(-1, 3)
        _act_norms = np.linalg.norm(_all_acts, axis=-1)
        _valid = _act_norms > 1e-4
        v_des_magnitude = float(np.mean(_act_norms[_valid])) * DATASET_HZ if _valid.any() else 0.4
        print(f'[ eval ] pid_const_v: v_des_magnitude={v_des_magnitude:.3f} m/s '
              f'(mean_act={np.mean(_act_norms[_valid]):.4f} m × {DATASET_HZ} Hz)')
    else:
        v_des_magnitude = 0.0   # unused by other controllers
    # U6: MJX predictive-sampling kwargs (task_id/planner_steps removed — MJX needs neither).
    mjpc_kwargs = {
        'n_trajectories': config.get('mjx_n_samples', 16),
        'horizon':        config.get('mjx_horizon', 0.3),
        'n_improve':      config.get('mjx_n_improve', 5),
        'vel_weight':     config.get('mjx_vel_weight', 0.1),
    } if controller == 'mjpc' else None
    # Eval-parameter folder — mirrors args_to_watch_fm_visual_plan naming convention.
    # Sits BETWEEN train-identity and seed; keeps variant name pure.
    # e.g.  flow_matching_v3_uav/H8_D..._9D / mpc4_pid_stopgo_T0.5 / 0 / diffuser /
    eval_params_dir = _uav_eval_tag(config, controller)

    projector = None
    if variant != 'diffuser':
        # UAV has no semantic goal columns. SequenceDataset.get_goal_dim() can false-positive
        # on incidentally-constant channels (e.g. corridor altitude, constant p_des).
        # DC_FIX dynamics constraints touch p indices 6,7,8 — if goal_dim>0 shrinks traj_dim
        # below 9, those indices go out-of-bounds in build_matrices (IndexError: index 64).
        # Fix: always force goal_dim=0 for UAV and patch the loaded model so p_sample_loop
        # doesn't slice the trajectory before handing it to the projector.
        _detected_goal_dim = int(getattr(model_fm, 'goal_dim', 0))
        if _detected_goal_dim != 0:
            print(f'[ eval ] UAV: overriding model_fm.goal_dim {_detected_goal_dim} → 0 '
                  f'(false-positive constant channel; UAV has no goal dims)')
            model_fm.goal_dim = 0
        traj_dim = int(dataset.observation_dim + dataset.action_dim)
        projector = setup_dpcc_projector(
            parsed, config,
            dataset.normalizer.normalizers['observations'],
            dataset.normalizer.normalizers['actions'],
            variant, trajectory_dim=traj_dim)
    policy = Policy(model=model_fm, normalizer=dataset.normalizer,
                    preprocess_fns=getattr(parsed, 'preprocess_fns', []),
                    test_ret=getattr(parsed, 'test_ret', 0),
                    projector=projector, trajectory_selection=_selection_for(variant))

    # Path: scene_root / plans / <model_exp_noseed> / <eval_params> / <seed> / <variant> /
    # savepath = scene_root / flow_matching_v3_uav / H8_...9D / <seed>
    scene_root  = os.path.join(parsed.logbase, parsed.dataset)
    _model_dir  = os.path.relpath(os.path.dirname(parsed.savepath), scene_root)  # strip seed
    _seed_str   = os.path.basename(parsed.savepath)
    seed_dir    = os.path.join(scene_root, 'plans', _model_dir, eval_params_dir, _seed_str)
    out_dir     = os.path.join(seed_dir, variant)
    diag_dir    = os.path.join(out_dir, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    # Write config snapshot at the correct eval-tag-aware seed dir (once, on first variant).
    # setup.py's mkdir() no longer auto-snapshots during eval (save=False path); we do it here
    # where eval_params_dir is known, so the snapshot lands next to the variant subdirs.
    _snap_dir = os.path.join(seed_dir, f'config_snapshot_{parsed.config.split(".")[-1]}')
    if not os.path.exists(_snap_dir):
        import types as _t
        _snap_args = _t.SimpleNamespace(config=parsed.config, savepath=seed_dir)
        utils.Parser().snapshot_configs(_snap_args)

    record = (args.record != 'none')
    renderer = _make_overhead_renderer(mujoco, mj_model) if record else None
    batch_size = int(config.get('mpc_batch_size', config.get('batch_size', 4)))

    rollouts = []
    try:
        for i in range(args.n_trials):
            homotopy = homotopies[i % len(homotopies)]
            r = rollout_one(mj_model, scene, homotopy, 10_000 + i, policy, horizon,
                            renderer=renderer, goal_radius=args.goal_radius, batch_size=batch_size,
                            variant=variant, log_dir=out_dir,
                            control_hz=config.get('control_hz', DATASET_HZ),
                            text_log=config.get('behavior_log', True),
                            controller=controller, cond_mode=cond_mode, mjpc_kwargs=mjpc_kwargs,
                            v_des_magnitude=v_des_magnitude)
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
        'success_relaxed_rate': float(np.mean([r['success_relaxed'] for r in rollouts])),  # U7: crossed finish line
        'success_and_constraints_rate': float(np.mean([r['success_and_constraints'] for r in rollouts])),
        'success_and_constraints_relaxed_rate': float(np.mean([r['success_and_constraints_relaxed'] for r in rollouts])),
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
          f'sel={_selection_for(variant)}): success={succ:.3f}  success_relaxed={summary["success_relaxed_rate"]:.3f}  '
          f'safe={summary["safe_rate"]:.3f}  goal_reached={summary["goal_reached_rate"]:.3f}  '
          f'track_err={summary["track_err_mean"]:.3f}  → {os.path.dirname(npz_path)}/')
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
    config = load_pcc_config(scene, args.seed)
    homotopies = gen.HOMOTOPY_CLASSES[scene]
    mj_model = mujoco.MjModel.from_xml_path(gen.SCENE_XMLS[scene])

    # cond_mode is a MODEL property (obs layout baked into the normalizer at train time).
    # Lock it to what the checkpoint was actually trained with — ignore the plan block value,
    # which is user-editable and can silently mismatch (crash: shapes (9,) vs (6,) at normalize).
    config['cond_mode'] = str(getattr(parsed, 'cond_mode', config.get('cond_mode', 'p_des')))
    print(f'[ eval ] cond_mode={config["cond_mode"]}  (source: train checkpoint args)')

    # Tightened variants only differ from their base siblings when spatial constraints
    # (bounds/halfspace/obstacles) are active — enlarge_constraints is applied there.
    # With only 'dynamics' in constraint_types the enlarge margin is computed but never
    # used, so tightened == non-tightened == wasted compute. Skip them and say why.
    _spatial = {'bounds', 'halfspace', 'obstacles'}
    _has_spatial = bool(_spatial & set(config.get('constraint_types', [])))
    if not _has_spatial:
        _skip = [v for v in config['projection_variants'] if 'tightened' in v]
        if _skip:
            print(f'[ eval ] {scene}: skipping {len(_skip)} tightened variants '
                  f'(no spatial constraints in constraint_types — enlarge has no effect): {_skip}')
        config['projection_variants'] = [v for v in config['projection_variants'] if 'tightened' not in v]

    print(f'[ eval ] {scene}: variants={config["projection_variants"]}  '
          f'constraints={config["constraint_types"]}  batch_size={config.get("mpc_batch_size", config.get("batch_size", 4))}')
    summaries = {}
    for variant in config['projection_variants']:
        summaries[variant] = _run_variant(scene, variant, model_fm, dataset, parsed, horizon,
                                          config, args, mj_model, mujoco, homotopies)
    return summaries


def main():
    args, remaining = parse_args()

    # ── Resolve seed and n_trials: CLI wins; else read from config/uav_projection.yaml ──
    # We do this BEFORE stripping sys.argv so the yaml path is resolved cleanly.
    import yaml as _yaml
    _yaml_path = os.path.join(_REPO, 'config', 'uav_projection.yaml')
    try:
        with open(_yaml_path) as _fh:
            _proj_defaults = _yaml.safe_load(_fh) or {}
    except FileNotFoundError:
        _proj_defaults = {}

    _seed_from_cli = args.seed is not None
    args.seed = args.seed if _seed_from_cli else int(_proj_defaults.get('seed', 6))
    print(f'[ eval ] seed={args.seed}  (source: {"--seed CLI" if _seed_from_cli else _yaml_path})')

    _trials_from_cli = args.n_trials is not None
    args.n_trials = args.n_trials if _trials_from_cli else int(_proj_defaults.get('n_trials', 20))
    print(f'[ eval ] n_trials={args.n_trials}  (source: {"--n-trials CLI" if _trials_from_cli else _yaml_path})')

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
