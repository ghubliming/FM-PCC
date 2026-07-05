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


def _resolve_active_geo_matches(scene, cfg):
    """All `geo_constraint_variants` entries active for this scene (Fix_4 pattern): matched by
    `scene:` field (falls back to `name`) AND listed in `active_geo_variants` (or all, if that
    key is null). Returns a list — 0 (no active geo for this scene), 1 (the common case), or
    many (Fix_6: eval_scene runs every one of them in a single job)."""
    _all_geo = [g for g in (cfg.get('geo_constraint_variants') or []) if 'name' in g]
    _active = cfg.get('active_geo_variants')
    return [g for g in _all_geo
            if g.get('scene', g['name']) == scene and (_active is None or g['name'] in _active)]


def _apply_geo_entry(cfg, scene, entry):
    """Return a COPY of cfg with one geo_constraint_variants entry's constraint_types/geometry
    applied (or the dynamics-only global fallback if entry is None), plus its geo_tag (Fix_1).
    Shared by load_pcc_config (single-match) and eval_scene's multi-match loop (Fix_6)."""
    cfg = dict(cfg)
    if entry is not None:
        cfg['constraint_types']      = list(entry.get('constraint_types', cfg['constraint_types']))
        cfg['workspace_bounds']      = entry.get('workspace_bounds', None)
        cfg['halfspace_constraints'] = entry.get('halfspace_constraints', [])
        cfg['obstacle_constraints']  = entry.get('obstacle_constraints', [])
        print(f"[ eval ] E9 geo '{scene}' ← variant '{entry['name']}': "
              f"constraint_types={cfg['constraint_types']} "
              f"(bounds={cfg['workspace_bounds'] is not None}, "
              f"hs={len(cfg['halfspace_constraints'])}, obs={len(cfg['obstacle_constraints'])})")
    elif cfg.get('geo_constraint_variants'):
        print(f"[ eval ] E9: scene '{scene}' has no active geo variant → dynamics-only fallback")

    # E9 fix1: `geo_tag` — a second, swappable output-path axis mirroring the old avoiding-task
    # `results/halfspace_<halfspace_variant>/` folder level. Encodes WHICH geometry/constraint
    # combo produced a run (resolved geo entry name + its actually-active constraint_types),
    # so re-running the same scene under a different constraint_types subset (e.g. an ablation
    # like obstacles-only vs the full stack) lands in a DIFFERENT folder instead of overwriting
    # the previous run. `empty` (constraint_types=[]) tags as '<scene>_unconstrained'.
    _ctypes = cfg.get('constraint_types') or []
    cfg['geo_tag'] = f'{scene}_unconstrained' if not _ctypes else f"{scene}_{'+'.join(sorted(_ctypes))}"
    return cfg


def _load_base_cfg(scene, seed):
    """Build the merged config WITHOUT resolving per-scene geometry (Fix_6): yaml load +
    defaults + eval control params (batch_size, thresholds, U4 knobs, logging) from the
    plan_flow_matching_v3_uav block in config/uav.py. Geo resolution is a separate step
    (`_resolve_active_geo_matches` + `_apply_geo_entry`) so callers can run it once
    (`load_pcc_config`, single-match) or in a loop (`eval_scene`, Fix_6, possibly multi-match)."""
    import yaml

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
    cfg.setdefault('inflation', {'r_drone': 0.0, 'margin_base': 0.0})
    cfg.setdefault('action_bounds', None)

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


def load_pcc_config(scene, seed):
    """Merged eval config matching the avoiding-d3il.py pattern (single-geo-match convenience
    wrapper around `_load_base_cfg` + `_resolve_active_geo_matches` + `_apply_geo_entry`).

    Raises if the scene has MORE THAN ONE active geo_constraint_variants entry — that case is
    handled by `eval_scene`'s Fix_6 loop, which runs every active entry in one invocation
    instead of erroring. Use `eval_scene` (not this function) when a scene may have several
    active variants (e.g. testing dynamics_only vs dynamics_bounds_only vs combined_1)."""
    cfg = _load_base_cfg(scene, seed)
    _matches = _resolve_active_geo_matches(scene, cfg)
    if len(_matches) > 1:
        raise ValueError(
            f"E9: scene '{scene}' matches MULTIPLE active geo_constraint_variants "
            f"({[g['name'] for g in _matches]}) — load_pcc_config() only resolves ONE. "
            f"eval_scene() runs all of them in one job (Fix_6) — call that instead, or narrow "
            f"active_geo_variants in config/uav_projection.yaml to exactly one for this scene.")
    return _apply_geo_entry(cfg, scene, _matches[0] if _matches else None)


# ── DPCC projector — copied from fm_visual_aligning_test/eval_fm_visual_aligning.py and
#    adapted to the UAV 12-D transition. Only the DYNAMICS constraint is active this epoch;
#    bounds/halfspace/obstacle blocks are kept verbatim as PLACEHOLDERS (fire only if their
#    config keys are enabled — they are not this epoch). ────────────────────────────────────

class ProjectorNormalizer:
    """Wrap obs + act LimitsNormalizers into the dict Projector('states_actions') expects
    (verbatim from the visual-aligning eval)."""
    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {'observations': obs_normalizer, 'actions': act_normalizer}


def _exec_constraint_violations(obs_traj, config):
    """E9 exec-time violation metrics: check the FLOWN path against the RAW geometry ⊕ r_drone
    (physical collision truth — NOT the planning margin, which includes margin_base + enlarge).

    obs layout is [p_des | p | (v)] in both cond_modes → actual position p is cols 3:6.
    Returns (collision_free: bool, n_violations: int, total_violations: float).
    A step counts once toward n_violations if it violates ANY active spatial family; the
    magnitude sum accumulates every family's penetration depth (metres).
    """
    ctypes = config.get('constraint_types', []) or []
    spatial = {'geo_bounds', 'halfspace', 'obstacles'} & set(ctypes)
    if not spatial or not obs_traj:
        return True, 0, 0.0
    P = np.asarray(obs_traj, dtype=float)[:, 3:6]          # executed p, (T,3)
    r_drone = float((config.get('inflation') or {}).get('r_drone', 0.0))

    n_steps_viol = 0
    total = 0.0
    ws = config.get('workspace_bounds')
    halfspaces = config.get('halfspace_constraints', []) if 'halfspace' in spatial else []
    obstacles  = config.get('obstacle_constraints', [])  if 'obstacles'  in spatial else []
    for p in P:
        step_pen = 0.0
        if 'geo_bounds' in spatial and ws is not None:
            lb = np.array(ws['lb'], dtype=float); ub = np.array(ws['ub'], dtype=float)
            # physical box is raw lb/ub; body clears when p ∈ [lb+r, ub-r]
            below = (lb + r_drone) - p; above = p - (ub - r_drone)
            step_pen += float(np.clip(below, 0, None)[np.isfinite(lb)].sum())
            step_pen += float(np.clip(above, 0, None)[np.isfinite(ub)].sum())
        for hs in halfspaces:
            triple, x_active = _normalize_halfspace(hs)
            if x_active is not None and not (x_active[0] <= p[0] <= x_active[1]):
                continue                                    # wall not live at this x
            (x1, y1), (x2, y2), side = triple[0], triple[1], triple[2]
            dx, dy = x2 - x1, y2 - y1
            nrm = np.hypot(dx, dy)
            if nrm < 1e-9:
                continue
            nx, ny = (-dy / nrm, dx / nrm)                  # left normal of the segment
            signed = nx * (p[0] - x1) + ny * (p[1] - y1)    # + on the 'above'/left side
            feasible = signed if side == 'above' else -signed
            step_pen += max(0.0, r_drone - feasible)        # clear when feasible >= r_drone
        for ob in obstacles:
            dims = ob['dimensions']; c = ob['center']
            didx = [{'x': 0, 'y': 1, 'z': 2}[d] if isinstance(d, str) else int(d) for d in dims]
            dist = float(np.linalg.norm(p[didx] - np.asarray(c, dtype=float)))
            if ob['type'] == 'sphere_outside':
                step_pen += max(0.0, (ob['radius'] + r_drone) - dist)
            else:                                            # sphere_inside
                step_pen += max(0.0, dist - (ob['radius'] - r_drone))
        if step_pen > 1e-9:
            n_steps_viol += 1
            total += step_pen
    return (n_steps_viol == 0), int(n_steps_viol), float(total)


def plot_geo_constraints(geo_name, config, out_dir, is_tightened=False, basename='constraint_overview'):
    """E9 U2: constraint-geometry schematic — the UAV equivalent of the visual-aligning
    `constraint_overview.png` (`fm_visual_aligning_test/eval_fm_visual_aligning.py
    plot_geo_constraints`), which we were missing entirely (not a faithful port).

    3-panel: 3D wireframe | XY top-down | XZ side. Shows the workspace box (steelblue),
    halfspace boundaries + feasible-side arrow (darkorange; x_active segments drawn only
    over their live x-range, s_curve), obstacle balls (tomato). Boundaries are drawn at the
    TRUE enforced margin (r_drone + margin_base [+ enlarge if tightened]) — the same `margin`
    setup_dpcc_projector uses — not the raw scene geometry, so the figure shows what the
    projector actually believes, not just the XML.

    Saved as BOTH <basename>.png (raster, quick viewing) AND .svg (vector — the visual-aligning
    original only had .png; added here since a vector schematic is what a paper/thesis figure
    actually wants). Idempotent (skipped if both files already exist).

    `basename` distinguishes the tightened twin: unlike visual-aligning (where `-tightened`
    is baked into a whole separate NAMED geo entry with its own results/<name>/ folder), UAV's
    `-tightened` is a per-VARIANT margin modifier sharing the same geo_tag/geo_dir as its base
    sibling (matching the older DPCC-avoiding convention) — so the two margins need two
    filenames (`constraint_overview.png` vs `constraint_overview_tightened.png`) inside the
    SAME folder, not two folders.
    """
    out_png = os.path.join(out_dir, f'{basename}.png')
    out_svg = os.path.join(out_dir, f'{basename}.svg')
    if os.path.exists(out_png) and os.path.exists(out_svg):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as _mpa
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3C

    ctypes = config.get('constraint_types', [])
    _infl = config.get('inflation') or {}
    inflation_base = float(_infl.get('r_drone', 0.0)) + float(_infl.get('margin_base', 0.0))
    enlarge = float(config.get('enlarge_constraints') or 0.0) if is_tightened else 0.0
    margin = inflation_base + enlarge          # the TRUE enforced offset (matches setup_dpcc_projector)

    has_bounds = 'geo_bounds' in ctypes and config.get('workspace_bounds') is not None
    ws_lb = ws_ub = lb_d = ub_d = None
    _Z_DISP = (0.0, 2.0)        # UAV flight band default display when z is unconstrained
    if has_bounds:
        ws_lb = np.array(config['workspace_bounds']['lb'], dtype=float)
        ws_ub = np.array(config['workspace_bounds']['ub'], dtype=float)
        ws_lb = ws_lb + margin; ws_ub = ws_ub - margin
        lb_d = ws_lb.copy(); ub_d = ws_ub.copy()
        lb_d[np.isinf(lb_d)] = _Z_DISP[0]; ub_d[np.isinf(ub_d)] = _Z_DISP[1]
        # y may be ±inf too (e.g. a scene relying purely on halfspace walls for y) — clamp display
        for _i, _fallback in ((0, (-3.5, 3.5)), (1, (-2.0, 2.0))):
            if np.isinf(lb_d[_i]): lb_d[_i] = _fallback[0]
            if np.isinf(ub_d[_i]): ub_d[_i] = _fallback[1]

    halfspace_list = config.get('halfspace_constraints', []) if 'halfspace' in ctypes else []
    obstacle_list  = config.get('obstacle_constraints', [])  if 'obstacles' in ctypes else []

    def _xlim(): return (lb_d[0]-0.3, ub_d[0]+0.3) if lb_d is not None else (-3.5, 3.5)
    def _ylim(): return (lb_d[1]-0.3, ub_d[1]+0.3) if lb_d is not None else (-2.0, 2.0)
    def _zlim(): return (lb_d[2]-0.1, ub_d[2]+0.2) if lb_d is not None else (_Z_DISP[0]-0.1, _Z_DISP[1]+0.2)

    def _wall_xy(hs):
        """Resolve a halfspace to (p1, p2, side) clipped to its x_active range (if any)."""
        triple, x_active = _normalize_halfspace(hs)
        (x1, y1), (x2, y2), side = triple
        if x_active is not None:
            lo, hi = x_active
            if abs(x2 - x1) > 1e-9:
                t0 = (lo - x1) / (x2 - x1); t1 = (hi - x1) / (x2 - x1)
                t0, t1 = sorted((max(0.0, min(1.0, t0)), max(0.0, min(1.0, t1))))
                y1n = y1 + t0 * (y2 - y1); y2n = y1 + t1 * (y2 - y1)
                x1n = x1 + t0 * (x2 - x1); x2n = x1 + t1 * (x2 - x1)
                return (x1n, y1n), (x2n, y2n), side, (lo, hi)
        return (x1, y1), (x2, y2), side, None

    fig = plt.figure(figsize=(16, 5))
    _tstr = ' [tightened]' if is_tightened else ''
    fig.suptitle(f'{geo_name}{_tstr}  |  types: {ctypes}  |  margin(r_drone+base'
                 f'{"+" + str(enlarge) if enlarge else ""})={margin:.3f} m',
                 fontsize=11, fontweight='bold', y=0.98)

    # ── 3D panel ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(131, projection='3d')
    ax3.set_title('3D view', fontsize=9)
    ax3.set_xlabel('x (m)', fontsize=7); ax3.set_ylabel('y (m)', fontsize=7); ax3.set_zlabel('z (m)', fontsize=7)
    ax3.tick_params(labelsize=6)
    if lb_d is not None:
        x0, y0, z0 = lb_d; x1v, y1v, z1v = ub_d
        for xs, ys, zs in [
            ([x0,x1v],[y0,y0],[z0,z0]), ([x0,x1v],[y1v,y1v],[z0,z0]),
            ([x0,x1v],[y0,y0],[z1v,z1v]), ([x0,x1v],[y1v,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y1v],[z0,z0]), ([x1v,x1v],[y0,y1v],[z0,z0]),
            ([x0,x0],[y0,y1v],[z1v,z1v]), ([x1v,x1v],[y0,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y0],[z0,z1v]), ([x1v,x1v],[y0,y0],[z0,z1v]),
            ([x0,x0],[y1v,y1v],[z0,z1v]), ([x1v,x1v],[y1v,y1v],[z0,z1v]),
        ]:
            ax3.plot(xs, ys, zs, color='steelblue', alpha=0.7, lw=1.2)
    for obs in obstacle_list:
        dims = obs.get('dimensions', ['x', 'y'])
        cx, cy = float(obs['center'][0]), float(obs['center'][1])
        cz = float(obs['center'][2]) if ('z' in dims and len(obs['center']) > 2) else (
            (lb_d[2]+ub_d[2])/2 if lb_d is not None else 1.0)
        r = obs['radius'] + margin
        u = np.linspace(0, 2*np.pi, 20); v = np.linspace(0, np.pi, 10)
        ax3.plot_surface(cx + r*np.outer(np.cos(u), np.sin(v)), cy + r*np.outer(np.sin(u), np.sin(v)),
                          cz + r*np.outer(np.ones_like(u), np.cos(v)), color='tomato', alpha=0.25, linewidth=0)
    _hs_zlo, _hs_zhi = (lb_d[2], ub_d[2]) if lb_d is not None else _Z_DISP
    for hs in halfspace_list:
        (hx1, hy1), (hx2, hy2), side, _ = _wall_xy(hs)
        ax3.add_collection3d(_P3C([[
            [hx1, hy1, _hs_zlo], [hx2, hy2, _hs_zlo], [hx2, hy2, _hs_zhi], [hx1, hy1, _hs_zhi],
        ]], alpha=0.25, facecolor='darkorange', edgecolor='darkorange', lw=0.8))
    if not has_bounds and not obstacle_list and not halfspace_list:
        ax3.text2D(0.5, 0.5, 'no geometric\nconstraints', ha='center', va='center',
                   transform=ax3.transAxes, fontsize=9, color='gray')
    ax3.set_xlim(*_xlim()); ax3.set_ylim(*_ylim()); ax3.set_zlim(*_zlim())

    # ── XY top-down panel ────────────────────────────────────────────────────
    ax_xy = fig.add_subplot(132)
    ax_xy.set_title('XY top-down (z projected)', fontsize=9)
    ax_xy.set_xlabel('x (m)', fontsize=7); ax_xy.set_ylabel('y (m)', fontsize=7)
    ax_xy.set_aspect('equal'); ax_xy.grid(True, linestyle='--', alpha=0.4); ax_xy.tick_params(labelsize=6)
    if lb_d is not None:
        ax_xy.add_patch(_mpa.Rectangle((lb_d[0], lb_d[1]), ub_d[0]-lb_d[0], ub_d[1]-lb_d[1],
                                        lw=1.5, edgecolor='steelblue', facecolor='steelblue',
                                        alpha=0.12, label='bounds (enforced)'))
    else:
        ax_xy.text(0.5, 0.5, 'no bounds', ha='center', va='center', transform=ax_xy.transAxes,
                   fontsize=9, color='gray')
    for hs in halfspace_list:
        (hx1, hy1), (hx2, hy2), side, x_active = _wall_xy(hs)
        ax_xy.plot([hx1, hx2], [hy1, hy2], color='darkorange', lw=2.0,
                   label='halfspace wall' if hs is halfspace_list[0] else None)
        dx, dy = hx2-hx1, hy2-hy1; nrm = np.hypot(dx, dy) or 1.0
        nx, ny = (-dy/nrm, dx/nrm) if side == 'above' else (dy/nrm, -dx/nrm)
        mx, my = (hx1+hx2)/2, (hy1+hy2)/2
        ax_xy.annotate('', xy=(mx+nx*0.2, my+ny*0.2), xytext=(mx, my),
                       arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.3))
        if x_active is not None:
            ax_xy.text(mx, my, f'x∈[{x_active[0]:.1f},{x_active[1]:.1f}]', fontsize=5,
                       color='saddlebrown', ha='center', va='bottom')
    for obs in obstacle_list:
        ax_xy.add_patch(_mpa.Circle((float(obs['center'][0]), float(obs['center'][1])),
                                     obs['radius']+margin, lw=1.5, edgecolor='tomato',
                                     facecolor='tomato', alpha=0.2, label='obstacle (enforced)'))
        ax_xy.plot(float(obs['center'][0]), float(obs['center'][1]), 'r+', ms=6)
    ax_xy.set_xlim(*_xlim()); ax_xy.set_ylim(*_ylim())
    _handles, _labels = ax_xy.get_legend_handles_labels()
    if _handles:
        _seen = dict(zip(_labels, _handles))
        ax_xy.legend(_seen.values(), _seen.keys(), fontsize=6, loc='upper right')

    # ── XZ side panel ────────────────────────────────────────────────────────
    ax_xz = fig.add_subplot(133)
    ax_xz.set_title('XZ side (y projected)', fontsize=9)
    ax_xz.set_xlabel('x (m)', fontsize=7); ax_xz.set_ylabel('z (m)', fontsize=7)
    ax_xz.grid(True, linestyle='--', alpha=0.4); ax_xz.tick_params(labelsize=6)
    if lb_d is not None:
        ax_xz.add_patch(_mpa.Rectangle((lb_d[0], lb_d[2]), ub_d[0]-lb_d[0], ub_d[2]-lb_d[2],
                                        lw=1.5, edgecolor='steelblue', facecolor='steelblue', alpha=0.12))
        ax_xz.axhline(lb_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7, label=f'floor z={lb_d[2]:.2f} m')
        ax_xz.axhline(ub_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7, label=f'ceiling z={ub_d[2]:.2f} m')
        ax_xz.legend(fontsize=6, loc='upper right')
    else:
        ax_xz.text(0.5, 0.5, 'no bounds', ha='center', va='center', transform=ax_xz.transAxes,
                   fontsize=9, color='gray')
    for obs in obstacle_list:
        cz_mid = (lb_d[2]+ub_d[2])/2 if lb_d is not None else 1.0
        ax_xz.add_patch(_mpa.Circle((float(obs['center'][0]), cz_mid), obs['radius']+margin,
                                     lw=1.2, edgecolor='tomato', facecolor='tomato',
                                     alpha=0.25, linestyle='--'))
    for hs in halfspace_list:
        (hx1, hy1), (hx2, hy2), side, x_active = _wall_xy(hs)
        xb_lo, xb_hi = sorted((hx1, hx2))
        ax_xz.axvspan(xb_lo, xb_hi, color='darkorange', alpha=0.13, zorder=1)
        ax_xz.axvline(xb_lo, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
        ax_xz.axvline(xb_hi, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
    ax_xz.set_xlim(*_xlim()); ax_xz.set_ylim(*_zlim())

    if 'dynamics' in ctypes:
        fig.text(0.5, 0.01, 'Dynamics: p_des[t+1]=p_des[t]+act[t], p[t+1]=p[t]+act[t]  '
                 '(Euler link — no geometric shape)', ha='center', fontsize=7,
                 color='dimgray', style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)
    print(f'[ geo ] Constraint overview → {out_png} (+ .svg)')


def _normalize_halfspace(hs):
    """E9: accept both halfspace formats and return (constraint_triple, x_active).
      - list form  [[x1,y1], [x2,y2], 'side']                     → x_active=None (always live)
      - dict form  {line: [[..],[..]], side: '..', x_active: [lo,hi]}  → per-segment switching
    `constraint_triple` is the [p0, p1, side] that formulate_halfspace_constraints consumes.
    """
    if isinstance(hs, dict):
        line = hs['line']
        return [line[0], line[1], hs['side']], hs.get('x_active')
    return [hs[0], hs[1], hs[2]], None


def setup_dpcc_projector(args, config, obs_normalizer, act_normalizer, variant,
                         trajectory_dim=12, current_x=None):
    """Build the DPCC projector (mirrors visual-aligning `setup_dpcc_projector`).

    UAV 12-D transition: [dx(0) dy(1) dz(2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11)].
    Both position channels are real and anchored with 6 rows (DC_FIX), mirroring DPCC avoiding.
    p_des(3,4,5): commanded setpoint. p(6,7,8): actual drone position from qpos[:3].

    E9 additions:
      - GEOMETRIC constraints (halfspace, obstacles, workspace box) bind to ACTUAL p (6,7,8)
        ONLY — DPCC-faithful (see STUDY_DPCC_constraint_dim_binding.md). Velocity (9,10,11)
        is never touched.
      - `bounds` builds TWO orthogonal row-sets (§2.2): the workspace box on p AND the shared
        action-magnitude bound on the action dims (0,1,2 = Δp_des). The action bound is NOT
        inflated (it is a dataset-range cap, not a spatial surface). SETTABLE via
        `config['action_bounds']` (Fix_3): `'auto'` (default, recommended) self-derives from
        `act_normalizer.mins/maxs` (this dataset's own observed action range); an explicit
        `{lb,ub}` dict overrides it; `None` disables the action bound entirely.
      - Inflation (r_drone + margin_base) offsets every spatial surface so the BODY clears,
        always-on; the -tightened enlarge is added on top.
      - Halfspaces may carry `x_active` (s_curve): a wall is included only if `current_x` is
        inside its interval; `current_x=None` → all walls active (build-time fallback).

    Variant semantics (unchanged): gradient / post_processing / model_free / tightened.
    """
    from flow_matcher_v3_uav.sampling.projection import Projector

    _DIM = {'dx': 0, 'dy': 1, 'dz': 2, 'x': 6, 'y': 7, 'z': 8}   # x,y,z = actual position p
    pad = trajectory_dim - 9
    is_tightened = 'tightened' in variant
    tightening   = float(config.get('enlarge_constraints') or 0.0)
    enlarge      = tightening if is_tightened else 0.0
    # E9 inflation: always-on offset so the drone body (not just its center) clears geometry.
    _infl = config.get('inflation') or {}
    inflation_base = float(_infl.get('r_drone', 0.0)) + float(_infl.get('margin_base', 0.0))
    margin = inflation_base + enlarge                 # total spatial offset (surfaces only)
    ctypes = config.get('constraint_types', [])
    constraint_list = []

    if 'geo_bounds' in ctypes:
        # Workspace box on ACTUAL p (6,7,8), shrunk inward by the spatial margin.
        # Patch_Constraints_C3: renamed from 'bounds' — this flag now ONLY means the geo box;
        # 'bounds' below is a SEPARATE, independent family (the restored DPCC action limit).
        ws = config.get('workspace_bounds')
        if ws is not None:
            ws_lb = np.array(ws['lb'], dtype=float); ws_ub = np.array(ws['ub'], dtype=float)
            lb = np.concatenate([np.full(6, -np.inf), ws_lb + margin, np.full(pad, -np.inf)])
            ub = np.concatenate([np.full(6,  np.inf), ws_ub - margin, np.full(pad,  np.inf)])
            constraint_list += [['lb', lb], ['ub', ub]]

    if 'bounds' in ctypes:
        # Shared action-magnitude bound on the ACTION dims (0,1,2) — NOT inflated (§2.2).
        # Patch_Constraints_C3: independent of 'geo_bounds' above (was conflated under one
        # 'bounds' flag until this split; mirrors the same fix applied to visual-aligning's
        # eval scripts — see logs_in_develop/Gen7_FMPCC_Viusal_Aligning/Patch_Constraints_C3/).
        # SETTABLE in the yaml (`action_bounds`), default `'auto'` (Fix_3):
        #   'auto' (RECOMMENDED, default) → SELF-DERIVE from act_normalizer.mins/.maxs, the
        #     dataset's OWN observed Δp_des range (LimitsNormalizer: mins/maxs = X.min/max
        #     over the training data, fit at load time). This is what DPCC-avoiding's
        #     hardcoded ['vx','vy'] bound approximated BY HAND for ITS OWN dataset ("need to
        #     be within the limits of the dataset due to the normalization") — copying
        #     avoiding's NUMBER would be wrong here (different robot, different workspace
        #     scale, different expert speed); reusing the SAME METHOD (derive from THIS
        #     dataset's own action range) is the faithful equivalent, and it's already
        #     computed by code that runs at eval time — no placeholder, no cluster-side
        #     manual measurement needed. Mirrors the `pid_const_v` self-calibration precedent
        #     (`_run_variant`: `v_des_magnitude` derived from `dataset.fields.actions`).
        #   explicit {lb: [...], ub: [...]} → override with a hand-picked cap (e.g. to test a
        #     tighter/looser action limit than the dataset's own range) instead of 'auto'.
        ab = config.get('action_bounds', 'auto')
        if ab is None:
            a_lb = a_ub = None
        elif ab == 'auto':
            a_lb = np.asarray(act_normalizer.mins, dtype=float)
            a_ub = np.asarray(act_normalizer.maxs, dtype=float)
        else:
            a_lb = np.array(ab['lb'], dtype=float)
            a_ub = np.array(ab['ub'], dtype=float)
        if a_lb is not None:
            a_lb = np.concatenate([a_lb, np.full(trajectory_dim - 3, -np.inf)])
            a_ub = np.concatenate([a_ub, np.full(trajectory_dim - 3,  np.inf)])
            constraint_list += [['lb', a_lb], ['ub', a_ub]]

    if 'dynamics' in ctypes and 'model_free' not in variant:
        # DC_FIX: both real channels anchored — 6 rows (DPCC avoiding 4-row pattern scaled to 3D).
        # Traj layout: [act(0,1,2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11)]
        constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]  # DC_FIX p_des ← act
        constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]  # DC_FIX p     ← act

    if 'halfspace' in ctypes:
        _hs = {'x': _DIM['x'], 'y': _DIM['y']}
        for hs in config.get('halfspace_constraints', []):
            triple, x_active = _normalize_halfspace(hs)
            if x_active is not None and current_x is not None:
                if not (x_active[0] <= float(current_x) <= x_active[1]):
                    continue                              # wall not live in this x-segment
            C_row, d = utils.formulate_halfspace_constraints(triple, margin, trajectory_dim, _hs)
            constraint_list.append(('ineq', (C_row, d)))

    if 'obstacles' in ctypes:
        for obs in config.get('obstacle_constraints', []):
            dims = [_DIM[d] if isinstance(d, str) else int(d) for d in obs['dimensions']]
            constraint_list.append((obs['type'], dims, obs['center'], obs['radius'] + margin))

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
                v_des_magnitude=0.0, geo_config=None, rebuild_projector=None):
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

        # E9 s_curve: re-select the active wall set from the drone's current x before this
        # replan (only when the scene declares x_active halfspaces — else rebuild_projector
        # is None and this is skipped, preserving the build-once path exactly).
        if rebuild_projector is not None:
            policy.projector = rebuild_projector(float(p[0]))

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
                frame = _render_overhead(mujoco, model, data, renderer)
                # U2b: GIF step-count overlay ('sK', top-left) — ported from visual-aligning's
                # Aligning_Sim.capture_frame (`cv2.putText(frame, f's{self.step_counter}', ...)`),
                # which the UAV GIFs were missing entirely. Same style: yellow, top-left, FONT_HERSHEY_PLAIN.
                import cv2
                cv2.putText(frame, f's{k}', (5, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
                frames.append(frame)
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

    # Constraint-aware metrics (FMv3ODE schema). E9: computed from the FLOWN path against the
    # scene's RAW spatial geometry ⊕ r_drone (physical collision truth). Dynamics-only /
    # unconstrained (`empty`) scenes have no spatial families → trivially clean.
    collision_free, n_violations, total_violations = _exec_constraint_violations(obs_traj, geo_config or {})
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

    # E9 s_curve: per-replan active-set switching. If the scene declares any `x_active`
    # halfspaces, the active wall set depends on the drone's current x, so the projector must
    # be rebuilt each FM step. This closure does that (cheap matrix rebuild, small H×T); it is
    # passed to rollout_one only for such scenes — every other scene builds ONCE (rebuild=None,
    # behaviour byte-identical to before). Non-switching walls (no x_active) are unaffected.
    _has_x_active = (variant != 'diffuser') and any(
        isinstance(hs, dict) and hs.get('x_active') is not None
        for hs in (config.get('halfspace_constraints') or []))
    rebuild_projector = None
    if _has_x_active:
        def rebuild_projector(current_x, _tdim=int(dataset.observation_dim + dataset.action_dim)):
            return setup_dpcc_projector(
                parsed, config,
                dataset.normalizer.normalizers['observations'],
                dataset.normalizer.normalizers['actions'],
                variant, trajectory_dim=_tdim, current_x=current_x)

    # Path: scene_root / plans / <model_exp_noseed> / <eval_params> / <seed> / <geo_tag> / <variant> /
    # savepath = scene_root / flow_matching_v3_uav / H8_...9D / <seed>
    # E9 fix1: `<geo_tag>` restores the old avoiding-task `results/halfspace_<variant>/` path
    # level — a second, swappable axis (which geometry/constraint-combo produced this run)
    # alongside the projection `<variant>` folder. Without it, two runs of the SAME scene under
    # different constraint_types (e.g. an ablation subset) would collide in one output folder.
    scene_root  = os.path.join(parsed.logbase, parsed.dataset)
    _model_dir  = os.path.relpath(os.path.dirname(parsed.savepath), scene_root)  # strip seed
    _seed_str   = os.path.basename(parsed.savepath)
    seed_dir    = os.path.join(scene_root, 'plans', _model_dir, eval_params_dir, _seed_str)
    geo_dir     = os.path.join(seed_dir, config.get('geo_tag', scene))
    out_dir     = os.path.join(geo_dir, variant)
    diag_dir    = os.path.join(out_dir, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    # E9 U2: constraint-geometry schematic (constraint_overview.png + .svg), mirroring
    # visual-aligning's `plot_geo_constraints` call site — once per geo_dir, before any
    # trajectory rollouts. `plot_geo_constraints` itself is idempotent (skips once both
    # files exist), so it's safe to call once per (base, tightened) margin encountered.
    # UAV's `-tightened` is a per-variant margin modifier sharing this geo_dir with its base
    # sibling (not a separate named geo entry as in visual-aligning) — so the base and
    # tightened schematics get distinct filenames in the SAME folder (see docstring).
    _variants = config['projection_variants']
    _is_this_tightened = 'tightened' in variant
    _first_of_kind = variant == next(
        (v for v in _variants if ('tightened' in v) == _is_this_tightened), variant)
    if _first_of_kind:
        os.makedirs(geo_dir, exist_ok=True)
        _basename = 'constraint_overview_tightened' if _is_this_tightened else 'constraint_overview'
        plot_geo_constraints(config.get('geo_tag', scene), config, geo_dir,
                             is_tightened=_is_this_tightened, basename=_basename)

    # Write config snapshot at the eval-tag-aware seed dir (once, on first variant/geo_tag).
    # setup.py's mkdir() no longer auto-snapshots during eval (save=False path); we do it here
    # where eval_params_dir is known. Kept ABOVE geo_dir (model/eval-param snapshot, not
    # geometry-specific) so it's written once per seed regardless of how many geo_tags run.
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
                            v_des_magnitude=v_des_magnitude, geo_config=config,
                            rebuild_projector=rebuild_projector)
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
    """Run EVERY projection variant (diffuser, dpcc-r/-c/-t) for EVERY active geo variant of
    one scene (Fix_6). A scene may have several active `geo_constraint_variants` entries at
    once (e.g. `active_geo_variants` listing `s_curve_dynamics_only`,
    `s_curve_dynamics_bounds_only`, AND `s_curve_combined_1` together — exactly the case that
    used to raise in `load_pcc_config`) — this now runs ALL of them in one job submission,
    each writing to its own `geo_tag`-named output folder (Fix_1), instead of erroring.

    Model+dataset loaded ONCE and shared across geo variants (constraint geometry doesn't
    change them); a Policy is (re)built per (geo variant, projection variant) pair, same as
    before. Returns {variant: summary} when exactly one geo variant is active for this scene
    (the common case, byte-identical to the pre-Fix_6 return shape) — else
    {geo_variant_name: {variant: summary}}."""
    import mujoco
    import uav_expert_data_collect.generator as gen
    model_fm, dataset, parsed, horizon = build_experiment(scene, args.seed, args.epoch, args.device)
    base_cfg = _load_base_cfg(scene, args.seed)
    homotopies = gen.HOMOTOPY_CLASSES[scene]
    mj_model = mujoco.MjModel.from_xml_path(gen.SCENE_XMLS[scene])

    _matches = _resolve_active_geo_matches(scene, base_cfg)
    _entries = _matches if _matches else [None]   # None → dynamics-only global fallback (unchanged)
    if len(_entries) > 1:
        print(f"[ eval ] {scene}: {len(_entries)} active geo variants in one job (Fix_6): "
              f"{[e['name'] for e in _entries]}")

    all_summaries = {}
    for entry in _entries:
        config = _apply_geo_entry(base_cfg, scene, entry)

        # cond_mode is a MODEL property (obs layout baked into the normalizer at train time).
        # Lock it to what the checkpoint was actually trained with — ignore the plan block
        # value, which is user-editable and can silently mismatch (crash: shapes (9,) vs (6,)).
        config['cond_mode'] = str(getattr(parsed, 'cond_mode', config.get('cond_mode', 'p_des')))
        print(f'[ eval ] cond_mode={config["cond_mode"]}  (source: train checkpoint args)')

        # Tightened variants only differ from their base siblings when spatial constraints
        # (geo_bounds/halfspace/obstacles) are active — enlarge_constraints is applied there.
        # 'bounds' (the action-magnitude family, Patch_Constraints_C3) is NEVER tightened — it's
        # a dataset-range cap, not a spatial surface — so it's excluded from this set on purpose.
        # With only 'dynamics'+'bounds' in constraint_types the enlarge margin is computed but
        # never used, so tightened == non-tightened == wasted compute. Skip them and say why.
        _spatial = {'geo_bounds', 'halfspace', 'obstacles'}
        _has_spatial = bool(_spatial & set(config.get('constraint_types', [])))
        if not _has_spatial:
            _skip = [v for v in config['projection_variants'] if 'tightened' in v]
            if _skip:
                print(f'[ eval ] {scene}: skipping {len(_skip)} tightened variants '
                      f'(no spatial constraints in constraint_types — enlarge has no effect): {_skip}')
            config['projection_variants'] = [v for v in config['projection_variants'] if 'tightened' not in v]

        print(f'[ eval ] {scene} [geo_tag={config["geo_tag"]}]: variants={config["projection_variants"]}  '
              f'constraints={config["constraint_types"]}  batch_size={config.get("mpc_batch_size", config.get("batch_size", 4))}')
        summaries = {}
        for variant in config['projection_variants']:
            summaries[variant] = _run_variant(scene, variant, model_fm, dataset, parsed, horizon,
                                              config, args, mj_model, mujoco, homotopies)
        all_summaries[entry['name'] if entry is not None else config['geo_tag']] = summaries

    # Preserve the pre-Fix_6 flat {variant: summary} shape for the single-geo-variant case
    # (the overwhelming common case, and what any external caller/aggregator expects).
    if len(_entries) == 1:
        return next(iter(all_summaries.values()))
    return all_summaries


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
