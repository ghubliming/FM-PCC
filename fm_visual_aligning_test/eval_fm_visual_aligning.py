# Visual-FM (Gen7) evaluation script.
# Copy-modified from diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py.
# Logging pattern (realtime PNG/JSON/pkl, 7-metric report, expert reference,
# legacy rollout grid) is reused verbatim from Gen6V4.
#
# Key differences from Gen6V4 eval:
#   - Uses fm_visual_aligning (FM ODE engine replacing DDPM)
#   - experiment='plan_fm_visual_aligning' (new config block)
#   - Euler ODE forward (0→1) replaces DDPM reverse chain
#   - flow_steps_v3 replaces n_diffusion_steps for inference step count
#
# Output:  logs/aligning-d3il-visual/plan_fm_visual_aligning/<exp>/results/<seed>/
#          ├── expert_references/expert_rollout_<r>.{mp4,gif}   (generated once before variant loop)
#          └── <variant>/
#              ├── <variant>.npz     ← C5: SINGLE source of raw truth (UAV-style schema).
#              │                        obs_all is 6-D [des_c_pos(0:3) | c_pos(3:6)];
#              │                        sampled_trajectories_all is the FULL MPC fan;
#              │                        selected_idx_all is per-replan chosen index.
#              ├── <variant>.png     (6-panel rollout grid)
#              ├── eval_<variant>.log
#              ├── diag_first_replan.txt
#              └── diagnostics/rollout_<r>.{mp4,gif,_stats.json,_report.png,_mpc_foresight.png}
#          C5 removed the per-rollout `_data.pkl` and per-variant `results_seed_<s>.pkl`
#          (all their raw arrays now live in <variant>.npz; JSONs stay metrics-only). A
#          transient `<variant>.partial.npz` crash-safety sidecar is written every N rollouts
#          and deleted on successful variant completion (see _save_partial_npz).

import gc
import json
import time
import yaml
import os
import sys
import pickle
import argparse
import numpy as np
import torch
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection
import imageio
import cv2

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

sys.path.insert(0, os.path.abspath('d3il'))
sys.path.insert(0, os.path.abspath('d3il/environments/d3il'))
os.environ['D3IL_DIR'] = os.path.abspath('d3il/environments/d3il')

import fm_visual_aligning.utils as utils
from fm_visual_aligning.sampling.projection import Projector
# REAL_TIME_RECORDING_UPDATE — per-replan timing/digital-twin recorder (see logs_in_develop/REALTIME_RECORDING)
from realtime_recording.behavior_logger import RTRecorder
RT_CONTROL_HZ = 30   # REAL_TIME_RECORDING_UPDATE — assumed deployment loop rate (budget=1000/hz ms); tune per target hardware. total_ms = per-replan FM+projection wall-time (bundled).

import d3il
print(f'[ eval ] Using d3il from: {d3il.__file__}')
print(f'[ eval ] D3IL_DIR set to: {os.environ["D3IL_DIR"]}')

from d3il.simulation.aligning_sim import Aligning_Sim

# ── Normalizer wrapper for Projector ─────────────────────────────────────────

class ProjectorNormalizer:
    """
    Wraps obs and act LimitsNormalizers into the dict that Projector('states_actions') expects.
    LimitsNormalizer already exposes .mins and .maxs, so no adapter indirection is needed.
    ProjectionNormalizer (inside Projector) reads normalizers['observations'].mins/maxs
    and normalizers['actions'].mins/maxs to build 9D constraint bounds [act(3)|obs(6)].
    """
    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {
            'observations': obs_normalizer,   # LimitsNormalizer — .mins(6,) .maxs(6,)
            'actions':      act_normalizer,   # LimitsNormalizer — .mins(3,) .maxs(3,)
        }

# ── Projector setup ───────────────────────────────────────────────────────────

def setup_dpcc_projector(args, config, obs_normalizer, act_normalizer, variant,
                         is_tightened=False, trajectory_dim=9):
    """
    Build the DPCC SLSQP projector.

    Visual (9D): [dx(0) dy(1) dz(2) | des_x(3) des_y(4) des_z(5) | x(6) y(7) z(8)]
    Non-visual (23D): same first 9 dims + obs_20D trailing dims (9-22, unconstrained).

    trajectory_dim=9 for visual, 23 for non-visual (UF-17).
    Constraint-relevant dims are always at 0-8 regardless of trajectory_dim.

    Patch_Constraints_C3 (dangerous rename — see logs_in_develop/Gen7_FMPCC_Viusal_Aligning/
    Patch_Constraints_C3/): 'bounds' and the geo/workspace box used to be the SAME
    constraint_types flag, which silently dropped DPCC-avoiding's action-magnitude guard when
    this eval repurposed 'bounds' for a Cartesian box on position. Now split into two
    independent families:
      - 'geo_bounds' → the Cartesian workspace box on ACTUAL position (dims 6,7,8), reading
        `config['workspace_bounds']`. This is what 'bounds' used to mean here.
      - 'bounds' → RESTORED to its true DPCC meaning: an action-magnitude limit on dims
        0,1,2 (dx,dy,dz). SELF-DERIVED from `act_normalizer.mins/.maxs` by default (this
        dataset's own observed action range — mirrors UAV E9 Fix_3's 'auto' pattern), settable
        via `config['action_bounds']`: `'auto'` (default) / explicit `{lb,ub}` / `null`.
    The two are orthogonal and may both be active at once (see `combined_4`/`combined_5` in
    config/visual_aligning_eval.yaml, which now list both `'geo_bounds'` and `'bounds'`).
    """
    _DIM = {'dx': 0, 'dy': 1, 'dz': 2, 'des_x': 3, 'des_y': 4, 'des_z': 5,
            'x': 6, 'y': 7, 'z': 8}

    # FIX-18.5: slice the obs normalizer to match trajectory_dim - action_dim,
    # NOT a hardcoded 6. Visual: trajectory_dim=9 → keep obs at 6 (no-op).
    # Non-visual: trajectory_dim=23 → keep obs at 20 (was: trimmed to 6,
    # which left the projector with 9-D ranges while constraint bound vectors
    # are built at full 23-D → broadcast crash at build_matrices line 401).
    # The trailing 14 trajectory dims (box/target poses) carry zero bound
    # coefficients from formulate_halfspace_constraints anyway, so passing
    # the full normalizer doesn't change the constraint semantics for the
    # first-9 robot-kinematic dims that PCC actually constrains.
    _target_obs_dim = trajectory_dim - 3   # action_dim is hardcoded 3 throughout
    proj_obs_normalizer = obs_normalizer
    if hasattr(obs_normalizer, 'mins') and len(obs_normalizer.mins) > _target_obs_dim:
        from fm_visual_aligning.datasets.normalization import LimitsNormalizer as _LN
        _dummy = np.stack([obs_normalizer.mins[:_target_obs_dim],
                           obs_normalizer.maxs[:_target_obs_dim]])
        proj_obs_normalizer = _LN(_dummy)

    pad = trajectory_dim - 9   # extra dims to pad lb/ub with ±inf (0 for visual)
    constraint_list = []

    if 'geo_bounds' in config.get('constraint_types', []) and 'geo_free' not in variant:
        # U8: 'geo_free' (variant-level, mirrors 'model_free') skips this + halfspace +
        # obstacles TOGETHER — the "geometric/spatial" group. See
        # logs_in_develop/Gen7_FMPCC_Viusal_Aligning/Patch_Constraints_C3/Gen11E9U8_Sync/.
        tightening = config.get('enlarge_constraints') or 0.0
        ws_lb = np.array(config['workspace_bounds']['lb'])
        ws_ub = np.array(config['workspace_bounds']['ub'])
        if is_tightened and tightening > 0.0:
            ws_lb += tightening
            ws_ub -= tightening
        lb = np.concatenate([np.full(6, -np.inf), ws_lb, np.full(pad, -np.inf)])
        ub = np.concatenate([np.full(6,  np.inf), ws_ub, np.full(pad,  np.inf)])
        constraint_list.append(['lb', lb])
        constraint_list.append(['ub', ub])

    if 'bounds' in config.get('constraint_types', []) and 'bounds_free' not in variant:
        # RESTORED DPCC action-magnitude bound (dims 0,1,2 = dx,dy,dz) — see docstring.
        # NOT tightened by `enlarge_constraints`: it's a dataset-range cap, not a spatial
        # surface (mirrors UAV E9: only geo_bounds/halfspace/obstacles get the tightening margin).
        # U8: 'bounds_free' (variant-level) skips just this family — independent of and
        # composable with 'model_free' (skips dynamics) and 'geo_free' (skips the geometric
        # group): e.g. 'geo_free-bounds_free' = dynamics alone, 'geo_free-model_free' = bounds
        # alone. See Gen11E9U8_Sync changelog for the full truth table.
        ab = config.get('action_bounds', 'auto')
        if ab == 'auto':
            a_lb = np.asarray(act_normalizer.mins, dtype=float)
            a_ub = np.asarray(act_normalizer.maxs, dtype=float)
        elif ab is not None:
            a_lb = np.array(ab['lb'], dtype=float)
            a_ub = np.array(ab['ub'], dtype=float)
        else:
            a_lb = a_ub = None
        if a_lb is not None:
            lb = np.concatenate([a_lb, np.full(trajectory_dim - 3, -np.inf)])
            ub = np.concatenate([a_ub, np.full(trajectory_dim - 3,  np.inf)])
            constraint_list.append(['lb', lb])
            constraint_list.append(['ub', ub])

    if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
        # DC_FIX: both real channels anchored — 6 rows (DPCC avoiding 4-row pattern scaled to 3D).
        # Traj layout: [act(0,1,2) | des_c_pos(3,4,5) | real c_pos(6,7,8)]
        # des_c_pos(3,4,5): commanded position (real, from D3IL). c_pos(6,7,8): actual TCP (real, C4 fix).
        constraint_list.append(('deriv', [3, 0]))   # DC_FIX des_x[t+1] = des_x[t] + act_x[t]
        constraint_list.append(('deriv', [4, 1]))   # DC_FIX des_y[t+1] = des_y[t] + act_y[t]
        constraint_list.append(('deriv', [5, 2]))   # DC_FIX des_z[t+1] = des_z[t] + act_z[t]
        constraint_list.append(('deriv', [6, 0]))   # c_pos_x[t+1] = c_pos_x[t] + act_x[t]
        constraint_list.append(('deriv', [7, 1]))   # c_pos_y[t+1] = c_pos_y[t] + act_y[t]
        constraint_list.append(('deriv', [8, 2]))   # c_pos_z[t+1] = c_pos_z[t] + act_z[t]

    if 'halfspace' in config.get('constraint_types', []) and 'geo_free' not in variant:
        tightening = config.get('enlarge_constraints') or 0.0
        _hs_indices = {'x': _DIM['x'], 'y': _DIM['y']}
        for hs in config.get('halfspace_constraints', []):
            margin = tightening if is_tightened else 0.0
            C_row, d = utils.formulate_halfspace_constraints(hs, margin, trajectory_dim, _hs_indices)
            constraint_list.append(('ineq', (C_row, d)))

    if 'obstacles' in config.get('constraint_types', []) and 'geo_free' not in variant:
        tightening = config.get('enlarge_constraints') or 0.0
        for obs in config.get('obstacle_constraints', []):
            dims = [_DIM[d] if isinstance(d, str) else int(d) for d in obs['dimensions']]
            radius = obs['radius'] + (tightening if is_tightened else 0.0)
            constraint_list.append((obs['type'], dims, obs['center'], radius))

    dt = config.get('dt', 1.0)
    if   'dt0p25' in variant: dt *= 0.25
    elif 'dt0p5'  in variant: dt *= 0.50
    elif 'dt2p0'  in variant: dt *= 2.0
    elif 'dt4p0'  in variant: dt *= 4.0

    threshold = 0.0 if 'post_processing' in variant else config.get('diffusion_timestep_threshold', 0.5)
    gradient  = 'gradient' in variant

    return Projector(
        horizon=getattr(args, 'horizon', 8),
        transition_dim=trajectory_dim,
        action_dim=3,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=ProjectorNormalizer(proj_obs_normalizer, act_normalizer),
        diffusion_timestep_threshold=threshold,
        variant='states_actions',
        dt=dt,
        gradient=gradient,
        gradient_weights=[1, 0.5, 2] if gradient else None,
        solver='scipy',
        device=args.device,
    )

# ── Constraint geometry visualisation (UF-15) ────────────────────────────────

def _hs_xy_draw(ax, hs, enlarge, xlim, ylim, dashed=False):
    """Draw halfspace: shaded infeasible region + boundary line + feasible-side arrow.

    Infeasible half-plane fill uses a polygon built from the two clipped boundary
    endpoints plus the viewport corners that lie on the infeasible side.  All
    vertices stay within xlim×ylim — no BIG coordinates — so auto-scaling axes in
    the foresight SVG are not affected.  Cohen-Sutherland clipping ensures the
    boundary line segment lies strictly inside the display box.

    dashed=True: draw boundary as dashed line only — skip infeasible fill and arrow.
    Used for the inner planning boundary when showing dual-boundary on tightened plots.
    """
    pt1, pt2, side = hs
    x1, y1 = float(pt1[0]), float(pt1[1])
    x2, y2 = float(pt2[0]), float(pt2[1])
    dx, dy = x2 - x1, y2 - y1
    nx, ny = (-dy, dx) if side == 'above' else (dy, -dx)   # normal → feasible side
    nlen = np.hypot(nx, ny)
    if nlen < 1e-9:
        return
    nx /= nlen;  ny /= nlen
    x1 += enlarge * nx;  y1 += enlarge * ny    # tightening shifts boundary inward
    x2 += enlarge * nx;  y2 += enlarge * ny
    dx, dy = x2 - x1, y2 - y1
    tx = sorted([(xlim[0]-x1)/dx, (xlim[1]-x1)/dx]) if abs(dx) > 1e-9 else [-1e9, 1e9]
    ty = sorted([(ylim[0]-y1)/dy, (ylim[1]-y1)/dy]) if abs(dy) > 1e-9 else [-1e9, 1e9]
    t_lo = max(tx[0], ty[0])
    t_hi = min(tx[1], ty[1])
    if t_lo >= t_hi:
        return
    px = [x1 + t_lo*dx, x1 + t_hi*dx]
    py = [y1 + t_lo*dy, y1 + t_hi*dy]

    if not dashed:
        # ── infeasible half-plane fill ────────────────────────────────────────
        import matplotlib.patches as _mpa_hs
        _C = [(xlim[0],ylim[0]),(xlim[1],ylim[0]),(xlim[1],ylim[1]),(xlim[0],ylim[1])]
        def _inf(c): return (c[0]-x1)*nx + (c[1]-y1)*ny < -1e-9
        def _edge(p):
            if abs(p[1]-ylim[0]) < 1e-9: return 0
            if abs(p[0]-xlim[1]) < 1e-9: return 1
            if abs(p[1]-ylim[1]) < 1e-9: return 2
            return 3
        _e0, _e1 = _edge((px[0], py[0])), _edge((px[1], py[1]))
        _df = (_e1 - _e0) % 4
        if _df:
            _cf = [_C[(_e0+1+k)%4] for k in range(_df)]
            _cr = [_C[(_e0-k)%4]   for k in range(4-_df)]
            if _cf and all(_inf(c) for c in _cf):
                _poly = [(px[0],py[0])] + _cf + [(px[1],py[1])]
            elif _cr and all(_inf(c) for c in _cr):
                _poly = [(px[1],py[1])] + _cr + [(px[0],py[0])]
            else:
                _ifc = [c for c in _cf if _inf(c)] or [c for c in _cr if _inf(c)]
                _poly = ([(px[0],py[0])] + _ifc + [(px[1],py[1])]) if _ifc else None
            if _poly and len(_poly) >= 3:
                ax.add_patch(_mpa_hs.Polygon(_poly, closed=True, facecolor='darkorange',
                                              alpha=0.15, edgecolor='none', zorder=1))

    _ls = '--' if dashed else '-'
    ax.plot(px, py, color='darkorange', linewidth=1.5, linestyle=_ls, zorder=3)
    if not dashed:
        mx, my = (px[0]+px[1])/2, (py[0]+py[1])/2
        ax.annotate('', xy=(mx+nx*0.06, my+ny*0.06), xytext=(mx, my),
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
        ax.text(mx+nx*0.09, my+ny*0.09, 'feasible', fontsize=6, color='darkorange',
                ha='center', va='center')


def plot_geo_constraints(geo_name, geo_config, out_dir, is_tightened=False):
    """
    3-panel constraint geometry overview: 3D wireframe | XY top-down | XZ side.
    Saved as constraint_overview.png at out_dir (geo-level results folder).
    Called once per geo entry before any trajectory runs (UF-15).

    Panels show: workspace box (steelblue), halfspace boundary + feasible arrow
    (darkorange), obstacle sphere (tomato). Dynamics constraint has no geometric
    shape — noted in figure footer only.
    """
    out_path = os.path.join(out_dir, 'constraint_overview.png')
    if os.path.exists(out_path):
        return   # idempotent — skip if already generated this run

    import matplotlib.patches as _mpa
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3C

    constraint_types = geo_config.get('constraint_types', [])
    enlarge = (geo_config.get('enlarge_constraints') or 0.0) if is_tightened else 0.0

    # ── Workspace bounds (tighten + clamp inf for display) ───────────────────
    has_bounds = 'geo_bounds' in constraint_types
    ws_lb = ws_ub = lb_d = ub_d = None
    _Z_DISP = (0.0, 0.50)   # default z display when z is unconstrained (±inf)
    if has_bounds and 'workspace_bounds' in geo_config:
        ws_lb = np.array(geo_config['workspace_bounds']['lb'], dtype=float)
        ws_ub = np.array(geo_config['workspace_bounds']['ub'], dtype=float)
        ws_lb += enlarge;  ws_ub -= enlarge
        lb_d = ws_lb.copy();  ub_d = ws_ub.copy()
        lb_d[2] = _Z_DISP[0] if np.isinf(lb_d[2]) else lb_d[2]
        ub_d[2] = _Z_DISP[1] if np.isinf(ub_d[2]) else ub_d[2]

    def _xlim(): return (lb_d[0]-0.05, ub_d[0]+0.05) if lb_d is not None else (0.20, 0.80)
    def _ylim(): return (lb_d[1]-0.05, ub_d[1]+0.05) if lb_d is not None else (-0.45, 0.45)
    def _zlim(): return (lb_d[2]-0.02, ub_d[2]+0.05) if lb_d is not None else (_Z_DISP[0]-0.02, _Z_DISP[1]+0.05)

    halfspace_list = geo_config.get('halfspace_constraints', []) if 'halfspace' in constraint_types else []
    obstacle_list  = geo_config.get('obstacle_constraints',  []) if 'obstacles'  in constraint_types else []

    fig = plt.figure(figsize=(16, 5))
    _tstr = ' [tightened]' if is_tightened else ''
    fig.suptitle(f'{geo_name}{_tstr}  |  types: {constraint_types}',
                 fontsize=11, fontweight='bold', y=0.98)

    # ── 3D panel ─────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(131, projection='3d')
    ax3.set_title('3D view', fontsize=9)
    ax3.set_xlabel('x (m)', fontsize=7, labelpad=2)
    ax3.set_ylabel('y (m)', fontsize=7, labelpad=2)
    ax3.set_zlabel('z (m)', fontsize=7, labelpad=2)
    ax3.tick_params(labelsize=6)

    if lb_d is not None:
        x0, y0, z0 = lb_d;  x1v, y1v, z1v = ub_d
        for xs, ys, zs in [
            ([x0,x1v],[y0,y0],[z0,z0]), ([x0,x1v],[y1v,y1v],[z0,z0]),
            ([x0,x1v],[y0,y0],[z1v,z1v]), ([x0,x1v],[y1v,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y1v],[z0,z0]), ([x1v,x1v],[y0,y1v],[z0,z0]),
            ([x0,x0],[y0,y1v],[z1v,z1v]), ([x1v,x1v],[y0,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y0],[z0,z1v]), ([x1v,x1v],[y0,y0],[z0,z1v]),
            ([x0,x0],[y1v,y1v],[z0,z1v]), ([x1v,x1v],[y1v,y1v],[z0,z1v]),
        ]:
            ax3.plot(xs, ys, zs, color='steelblue', alpha=0.7, lw=1.2)
        ax3.add_collection3d(_P3C([
            [(x0,y0,z0),(x1v,y0,z0),(x1v,y1v,z0),(x0,y1v,z0)],
            [(x0,y0,z1v),(x1v,y0,z1v),(x1v,y1v,z1v),(x0,y1v,z1v)],
            [(x0,y0,z0),(x0,y0,z1v),(x0,y1v,z1v),(x0,y1v,z0)],
            [(x1v,y0,z0),(x1v,y0,z1v),(x1v,y1v,z1v),(x1v,y1v,z0)],
            [(x0,y0,z0),(x1v,y0,z0),(x1v,y0,z1v),(x0,y0,z1v)],
            [(x0,y1v,z0),(x1v,y1v,z0),(x1v,y1v,z1v),(x0,y1v,z1v)],
        ], alpha=0.06, facecolor='steelblue', edgecolor='none'))

    for obs in obstacle_list:
        _DIM_MAP = {'x': 6, 'y': 7, 'z': 8}
        cx, cy = float(obs['center'][0]), float(obs['center'][1])
        obs_dims = obs.get('dimensions', ['x', 'y'])
        cz = (float(obs['center'][2]) if len(obs['center']) > 2 and 'z' in obs_dims
              else ((lb_d[2]+ub_d[2])/2 if lb_d is not None else 0.12))
        r = obs['radius'] + enlarge
        u = np.linspace(0, 2*np.pi, 25);  v = np.linspace(0, np.pi, 15)
        ax3.plot_surface(
            cx + r*np.outer(np.cos(u), np.sin(v)),
            cy + r*np.outer(np.sin(u), np.sin(v)),
            cz + r*np.outer(np.ones_like(u), np.cos(v)),
            color='tomato', alpha=0.25, linewidth=0)

    # 3D halfspace boundary plane — vertical rect spanning workspace z, clipped to XY
    _hs_zlo = lb_d[2] if lb_d is not None else _Z_DISP[0]
    _hs_zhi = ub_d[2] if ub_d is not None else _Z_DISP[1]
    for _hs3 in halfspace_list:
        _hpt1, _hpt2, _hside = _hs3
        _hx1, _hy1 = float(_hpt1[0]), float(_hpt1[1])
        _hx2, _hy2 = float(_hpt2[0]), float(_hpt2[1])
        _hdx, _hdy = _hx2-_hx1, _hy2-_hy1
        _hn = np.array([-_hdy, _hdx]) if _hside == 'above' else np.array([_hdy, -_hdx])
        _hnl = float(np.hypot(*_hn))
        if _hnl < 1e-9: continue
        _hn /= _hnl
        _hx1 += enlarge*_hn[0]; _hy1 += enlarge*_hn[1]
        _hx2 += enlarge*_hn[0]; _hy2 += enlarge*_hn[1]
        _hdx, _hdy = _hx2-_hx1, _hy2-_hy1
        _hxl, _hyl = _xlim(), _ylim()
        _htx = sorted([(_hxl[0]-_hx1)/_hdx, (_hxl[1]-_hx1)/_hdx]) if abs(_hdx) > 1e-9 else [-1e9, 1e9]
        _hty = sorted([(_hyl[0]-_hy1)/_hdy, (_hyl[1]-_hy1)/_hdy]) if abs(_hdy) > 1e-9 else [-1e9, 1e9]
        _htlo = max(_htx[0], _hty[0]); _hthi = min(_htx[1], _hty[1])
        if _htlo >= _hthi: continue
        _hpx = [_hx1+_htlo*_hdx, _hx1+_hthi*_hdx]
        _hpy = [_hy1+_htlo*_hdy, _hy1+_hthi*_hdy]
        ax3.add_collection3d(_P3C([[
            [_hpx[0], _hpy[0], _hs_zlo], [_hpx[1], _hpy[1], _hs_zlo],
            [_hpx[1], _hpy[1], _hs_zhi], [_hpx[0], _hpy[0], _hs_zhi],
        ]], alpha=0.25, facecolor='darkorange', edgecolor='darkorange', lw=0.8))

    if not has_bounds and not obstacle_list and not halfspace_list:
        ax3.text2D(0.5, 0.5, 'no geometric\nconstraints',
                   ha='center', va='center', transform=ax3.transAxes, fontsize=9, color='gray')

    ax3.set_xlim(*_xlim());  ax3.set_ylim(*_ylim());  ax3.set_zlim(*_zlim())

    # ── XY top-down panel ────────────────────────────────────────────────────
    ax_xy = fig.add_subplot(132)
    ax_xy.set_title('XY top-down (z projected)', fontsize=9)
    ax_xy.set_xlabel('x forward (m)', fontsize=7)
    ax_xy.set_ylabel('y lateral (m)', fontsize=7)
    ax_xy.set_aspect('equal');  ax_xy.grid(True, linestyle='--', alpha=0.4)
    ax_xy.tick_params(labelsize=6)

    if lb_d is not None:
        ax_xy.add_patch(_mpa.Rectangle(
            (lb_d[0], lb_d[1]), ub_d[0]-lb_d[0], ub_d[1]-lb_d[1],
            lw=1.5, edgecolor='steelblue', facecolor='steelblue', alpha=0.12, label='geo bounds'))
    else:
        ax_xy.text(0.5, 0.5, 'no bounds', ha='center', va='center',
                   transform=ax_xy.transAxes, fontsize=9, color='gray')

    for hs in halfspace_list:
        _hs_xy_draw(ax_xy, hs, enlarge, _xlim(), _ylim())
    for obs in obstacle_list:
        ax_xy.add_patch(_mpa.Circle(
            (float(obs['center'][0]), float(obs['center'][1])), obs['radius']+enlarge,
            lw=1.5, edgecolor='tomato', facecolor='tomato', alpha=0.2, label='obstacle'))
        ax_xy.plot(float(obs['center'][0]), float(obs['center'][1]), 'r+', ms=6)

    ax_xy.set_xlim(*_xlim());  ax_xy.set_ylim(*_ylim())

    # ── XZ side panel ────────────────────────────────────────────────────────
    ax_xz = fig.add_subplot(133)
    ax_xz.set_title('XZ side (y projected)', fontsize=9)
    ax_xz.set_xlabel('x forward (m)', fontsize=7)
    ax_xz.set_ylabel('z vertical (m)', fontsize=7)
    ax_xz.grid(True, linestyle='--', alpha=0.4)
    ax_xz.tick_params(labelsize=6)

    if lb_d is not None:
        ax_xz.add_patch(_mpa.Rectangle(
            (lb_d[0], lb_d[2]), ub_d[0]-lb_d[0], ub_d[2]-lb_d[2],
            lw=1.5, edgecolor='steelblue', facecolor='steelblue', alpha=0.12))
        z_inf_lo = np.isinf(ws_lb[2])
        z_inf_hi = np.isinf(ws_ub[2])
        ax_xz.axhline(lb_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7,
                      label='z=−∞ (display clamped)' if z_inf_lo else f'floor z={lb_d[2]:.3f} m')
        ax_xz.axhline(ub_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7,
                      label='z=+∞ (display clamped)' if z_inf_hi else f'ceiling z={ub_d[2]:.3f} m')
        ax_xz.legend(fontsize=6, loc='upper right')
    else:
        ax_xz.text(0.5, 0.5, 'no bounds', ha='center', va='center',
                   transform=ax_xz.transAxes, fontsize=9, color='gray')

    for obs in obstacle_list:
        obs_dims = obs.get('dimensions', ['x', 'y'])
        cx_o = float(obs['center'][0])
        r_o  = obs['radius'] + enlarge
        if 'z' in obs_dims and len(obs['center']) > 2:
            ax_xz.add_patch(_mpa.Circle(
                (cx_o, float(obs['center'][2])), r_o,
                lw=1.5, edgecolor='tomato', facecolor='tomato', alpha=0.2))
        else:
            # 2D obstacle (xy-only): show as circle at workspace z-midpoint
            _cz_mid = (lb_d[2] + ub_d[2]) / 2 if lb_d is not None else 0.20
            ax_xz.add_patch(_mpa.Circle(
                (cx_o, _cz_mid), r_o,
                lw=1.2, edgecolor='tomato', facecolor='tomato', alpha=0.25,
                linestyle='--'))

    for _hs_xz in halfspace_list:
        _hxpt1, _hxpt2, _hxside = _hs_xz
        _hxx1, _hxy1 = float(_hxpt1[0]), float(_hxpt1[1])
        _hxx2, _hxy2 = float(_hxpt2[0]), float(_hxpt2[1])
        _hxdx, _hxdy = _hxx2 - _hxx1, _hxy2 - _hxy1
        _hxnx, _hxny = (-_hxdy, _hxdx) if _hxside == 'above' else (_hxdy, -_hxdx)
        _hxnl = float(np.hypot(_hxnx, _hxny))
        if _hxnl < 1e-9:
            continue
        _hxnx /= _hxnl;  _hxny /= _hxnl
        _hxx1 += enlarge * _hxnx;  _hxy1 += enlarge * _hxny
        _hxx2 += enlarge * _hxnx;  _hxy2 += enlarge * _hxny
        _hxdx, _hxdy = _hxx2 - _hxx1, _hxy2 - _hxy1
        _hxxl, _hxyl = _xlim(), _ylim()
        _hxtx = sorted([(_hxxl[0]-_hxx1)/_hxdx, (_hxxl[1]-_hxx1)/_hxdx]) if abs(_hxdx) > 1e-9 else [-1e9, 1e9]
        _hxty = sorted([(_hxyl[0]-_hxy1)/_hxdy, (_hxyl[1]-_hxy1)/_hxdy]) if abs(_hxdy) > 1e-9 else [-1e9, 1e9]
        _hxtlo = max(_hxtx[0], _hxty[0]);  _hxthi = min(_hxtx[1], _hxty[1])
        if _hxtlo >= _hxthi:
            continue
        _hxpx = [_hxx1 + _hxtlo*_hxdx, _hxx1 + _hxthi*_hxdx]
        _hxpy = [_hxy1 + _hxtlo*_hxdy, _hxy1 + _hxthi*_hxdy]
        _xb_lo = min(_hxpx[0], _hxpx[1])
        _xb_hi = max(_hxpx[0], _hxpx[1])
        _yb_lo = min(_hxpy[0], _hxpy[1])
        _yb_hi = max(_hxpy[0], _hxpy[1])
        ax_xz.axvspan(_xb_lo, _xb_hi, color='darkorange', alpha=0.13, zorder=1)
        ax_xz.axvline(_xb_lo, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
        ax_xz.axvline(_xb_hi, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
        _zc = (_zlim()[0] + _zlim()[1]) / 2
        ax_xz.text((_xb_lo + _xb_hi) / 2, _zc,
                   f'HS\ny∈[{_yb_lo:.2f},{_yb_hi:.2f}]',
                   ha='center', va='center', fontsize=5, color='darkorange',
                   style='italic', zorder=3)

    ax_xz.set_xlim(*_xlim());  ax_xz.set_ylim(*_zlim())

    if 'dynamics' in constraint_types:
        fig.text(0.5, 0.01,
                 'Dynamics: c_pos[t+1] = c_pos[t] + act[t]  (Euler link — no geometric shape)',
                 ha='center', fontsize=7, color='dimgray', style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[ geo ] Constraint overview → {out_path}')

# ── UF-16.3: Constraint satisfaction / violation metrics ──────────────────────

def check_trajectory_constraints(c_pos_traj, act_traj, geo_config, enlarge=0.0):
    """
    Evaluate actual EE trajectory against all active geometric constraints.

    c_pos_traj : (T, 3) actual end-effector positions in metres
    act_traj   : (T, 3) commanded actions per step (for dynamics consistency check)
    geo_config : eval geo_config dict — constraint_types, workspace_bounds, etc.
    enlarge    : tightening margin (>0 for tightened twin runs, else 0)

    Returns dict of exec_* metrics (all JSON-serialisable floats/ints/bools).
    Metrics that are inapplicable for a given constraint set are set to 0 / -1.
    """
    ct  = geo_config.get('constraint_types', [])
    pos = np.array(c_pos_traj, dtype=float)   # (T, 3)
    T   = len(pos)
    if T == 0:
        return {}
    act = np.array(act_traj, dtype=float) if (act_traj is not None and len(act_traj)) else None

    _DIM = {'x': 0, 'y': 1, 'z': 2}

    b_mag  = np.zeros(T)
    hs_mag = np.zeros(T)
    ob_pen = np.zeros(T)
    b_margin  = np.full(T, np.inf)
    hs_margin = np.full(T, np.inf)
    ob_margin = np.full(T, np.inf)

    if 'geo_bounds' in ct and 'workspace_bounds' in geo_config:
        wb = geo_config['workspace_bounds']
        lb = np.where(np.isinf(np.array(wb['lb'], dtype=float)), -1e9,
                      np.array(wb['lb'], dtype=float)) + enlarge
        ub = np.where(np.isinf(np.array(wb['ub'], dtype=float)),  1e9,
                      np.array(wb['ub'], dtype=float)) - enlarge
        viol_lo  = np.maximum(0.0, lb - pos)
        viol_hi  = np.maximum(0.0, pos - ub)
        b_mag    = np.maximum(viol_lo, viol_hi).max(axis=1)
        ok       = b_mag <= 1e-6
        margins  = np.minimum(pos - lb, ub - pos).min(axis=1)
        b_margin = np.where(ok, margins, 0.0)

    if 'halfspace' in ct:
        for hs in geo_config.get('halfspace_constraints', []):
            pt1, pt2, side = hs
            x1, y1 = float(pt1[0]), float(pt1[1])
            x2, y2 = float(pt2[0]), float(pt2[1])
            dx, dy  = x2 - x1, y2 - y1
            nx, ny  = (-dy, dx) if side == 'above' else (dy, -dx)
            nl = float(np.hypot(nx, ny))
            if nl < 1e-9:
                continue
            nx, ny = nx / nl, ny / nl
            x1 += enlarge * nx;  y1 += enlarge * ny
            sd = nx * (pos[:, 0] - x1) + ny * (pos[:, 1] - y1)
            ok = sd >= -1e-6
            hs_mag    = np.maximum(hs_mag,  np.where(~ok, -sd,  0.0))
            hs_margin = np.minimum(hs_margin, np.where(ok,  sd, np.inf))

    if 'obstacles' in ct:
        for obs in geo_config.get('obstacle_constraints', []):
            dims  = obs.get('dimensions', ['x', 'y'])
            idx   = [_DIM[d] for d in dims if d in _DIM]
            ctr   = np.array([obs['center'][k] for k in range(len(idx))], dtype=float)
            r     = float(obs['radius']) + enlarge
            dist  = np.linalg.norm(pos[:, idx] - ctr, axis=1)
            pen   = np.maximum(0.0, r - dist)
            ok    = pen <= 1e-6
            ob_pen    = np.maximum(ob_pen, pen)
            ob_margin = np.minimum(ob_margin, np.where(ok, dist - r, np.inf))

    b_viol  = b_mag  > 1e-6
    hs_viol = hs_mag > 1e-6
    ob_viol = ob_pen > 1e-6
    any_viol = b_viol | hs_viol | ob_viol

    m_all = np.full(T, np.inf)
    if 'geo_bounds' in ct: m_all = np.minimum(m_all, b_margin)
    if 'halfspace' in ct: m_all = np.minimum(m_all, hs_margin)
    if 'obstacles' in ct: m_all = np.minimum(m_all, ob_margin)
    valid_m     = m_all[(~any_viol) & np.isfinite(m_all)]
    mean_margin = float(valid_m.mean()) if len(valid_m) else 0.0

    max_streak = cur = 0
    for v in any_viol:
        cur = 0 if v else cur + 1
        if cur > max_streak:
            max_streak = cur

    viol_idx   = np.where(any_viol)[0]
    first_viol = int(viol_idx[0]) if len(viol_idx) else -1

    dyn_mean = dyn_max = 0.0
    if act is not None and T > 1:
        L    = min(T - 1, len(act))
        derr = np.linalg.norm(pos[1:L+1] - pos[:L] - act[:L], axis=1)
        dyn_mean = float(derr.mean())
        dyn_max  = float(derr.max())

    return {
        'exec_n_steps':                         int(T),
        'exec_n_violated_steps':                int(any_viol.sum()),
        'exec_constraint_sat_rate':             float(1.0 - any_viol.mean()),
        'exec_zero_violation_rollout':          bool(int(any_viol.sum()) == 0),
        'exec_bounds_viol_count':               int(b_viol.sum()),
        'exec_halfspace_viol_count':            int(hs_viol.sum()),
        'exec_obstacle_viol_count':             int(ob_viol.sum()),
        'exec_max_bounds_viol_m':               float(b_mag.max()),
        'exec_max_halfspace_viol_m':            float(hs_mag.max()),
        'exec_max_obstacle_penetration_m':      float(ob_pen.max()),
        'exec_constraint_margin_mean_m':        mean_margin,
        'exec_first_violation_step':            first_viol,
        'exec_longest_safe_streak':             int(max_streak),
        'exec_dynamics_consistency_error_mean': dyn_mean,
        'exec_dynamics_consistency_error_max':  dyn_max,
    }


def _check_planned_violations(cands_xyz, geo_config, enlarge=0.0):
    """
    Check post-projection planned c_pos candidates against all geometric constraints.
    cands_xyz : (B, H, 3) unnormalised planned EE positions
    Returns fraction of (sample, horizon_step) pairs violating any constraint.
    """
    ct = geo_config.get('constraint_types', [])
    B, H, _ = cands_xyz.shape
    flat  = cands_xyz.reshape(-1, 3)
    viol  = np.zeros(B * H, dtype=bool)
    _DIM  = {'x': 0, 'y': 1, 'z': 2}

    if 'geo_bounds' in ct and 'workspace_bounds' in geo_config:
        wb = geo_config['workspace_bounds']
        lb = np.where(np.isinf(np.array(wb['lb'], dtype=float)), -1e9,
                      np.array(wb['lb'], dtype=float)) + enlarge
        ub = np.where(np.isinf(np.array(wb['ub'], dtype=float)),  1e9,
                      np.array(wb['ub'], dtype=float)) - enlarge
        viol |= np.any((flat < lb) | (flat > ub), axis=1)

    if 'halfspace' in ct:
        for hs in geo_config.get('halfspace_constraints', []):
            pt1, pt2, side = hs
            x1, y1 = float(pt1[0]), float(pt1[1])
            x2, y2 = float(pt2[0]), float(pt2[1])
            dx, dy  = x2 - x1, y2 - y1
            nx, ny  = (-dy, dx) if side == 'above' else (dy, -dx)
            nl = float(np.hypot(nx, ny))
            if nl < 1e-9: continue
            nx, ny = nx/nl, ny/nl
            x1 += enlarge*nx;  y1 += enlarge*ny
            sd = nx*(flat[:,0]-x1) + ny*(flat[:,1]-y1)
            viol |= (sd < -1e-6)

    if 'obstacles' in ct:
        for obs in geo_config.get('obstacle_constraints', []):
            dims = obs.get('dimensions', ['x', 'y'])
            idx  = [_DIM[d] for d in dims if d in _DIM]
            ctr  = np.array([obs['center'][k] for k in range(len(idx))], dtype=float)
            r    = float(obs['radius']) + enlarge
            dist = np.linalg.norm(flat[:, idx] - ctr, axis=1)
            viol |= (dist < r - 1e-6)

    return float(viol.mean())


def _nest_constraint_metrics(cm):
    """Json_Orgnize_C4 (Findings #2/#3): reshape check_trajectory_constraints' flat exec_*/
    plan_* dict (see its own docstring) into grouped {exec: {..., by_family: {...}},
    plan: {...}} — used both for the per-rollout stats JSON and for constraint_metrics.json's
    aggregate `per_rollout` list, so both artifacts share one nesting convention. Does NOT
    change what's measured — purely a presentation reshape of the same numbers.
    `cm` may be {} (no geo_config / empty rollout) — returns empty groups in that case.
    """
    if not cm:
        return {'exec': {}, 'plan': {}}
    exec_group = {
        'n_violated_steps':       cm.get('exec_n_violated_steps', 0),
        'constraint_sat_rate':    cm.get('exec_constraint_sat_rate', 1.0),
        'zero_violation_rollout': cm.get('exec_zero_violation_rollout', True),
        'by_family': {
            'bounds':    {'viol_count': cm.get('exec_bounds_viol_count', 0),
                          'max_viol_m': cm.get('exec_max_bounds_viol_m', 0.0)},
            'halfspace': {'viol_count': cm.get('exec_halfspace_viol_count', 0),
                          'max_viol_m': cm.get('exec_max_halfspace_viol_m', 0.0)},
            'obstacles': {'viol_count': cm.get('exec_obstacle_viol_count', 0),
                          'max_viol_m': cm.get('exec_max_obstacle_penetration_m', 0.0)},
        },
        'margin_mean_m':        cm.get('exec_constraint_margin_mean_m', 0.0),
        'first_violation_step': cm.get('exec_first_violation_step', -1),
        'longest_safe_streak':  cm.get('exec_longest_safe_streak', 0),
        'dynamics_consistency_error': {
            'mean': cm.get('exec_dynamics_consistency_error_mean', 0.0),
            'max':  cm.get('exec_dynamics_consistency_error_max', 0.0),
        },
    }
    plan_group = {}
    if 'plan_post_viol_rate_mean' in cm:
        plan_group = {
            'post_viol_rate_mean': cm.get('plan_post_viol_rate_mean', 0.0),
            'post_viol_rate_max':  cm.get('plan_post_viol_rate_max', 0.0),
            'n_replan_steps':      cm.get('plan_n_replan_steps', 0),
        }
    return {'exec': exec_group, 'plan': plan_group}


# ── C5: consolidated NPZ payload (single source of raw truth) ──────────────────

def _collect_per_rollout_arrays(agent):
    """C5 consolidation: build the per-rollout raw + metric arrays for the variant NPZ from
    agent state ONLY (no dependency on Aligning_Sim.test_agent's aggregate return), so the
    identical payload can be written both at variant end AND incrementally as a crash-safety
    partial. Returns a kwargs dict for np.savez.

    Schema notes (C5 — mirrors FM_v3_uav_test/eval_artifacts.py):
      - `obs_all` is now 6-D per step: [des_c_pos(0:3) | c_pos(3:6)] — the model's actual
        6-D input and the UAV obs layout. Cols 0:3 are the commanded position (byte-identical
        to the OLD 3-D `obs_all`, so any consumer slicing [:, :3] is unaffected); cols 3:6
        are the ACTUAL executed path (was the pkl-only `c_pos_history`).
      - `sampled_trajectories_all` is now the FULL MPC candidate fan (per replan step a
        (B,H,3) array) — was the SELECTED plan only; the fan was the pkl-only `all_candidates`.
      - `selected_idx_all` is NEW (was pkl-only): which candidate index executed per replan.
    All other keys are unchanged from the pre-C5 npz (same construction, same names).
    """
    obs_all, act_all, cand_all, selidx_all = [], [], [], []
    for r in range(agent.rollout_counter + 1):
        d = agent.master_rollout_history.get(f'rollout_{r}')
        if not d:
            continue
        des  = np.asarray(d.get('real_robot_pos'))       # (T,3) commanded (des_c_pos)
        cpos = np.asarray(d.get('c_pos_history'))         # (T,3) actual executed path
        if des.ndim == 2 and cpos.ndim == 2 and des.shape == cpos.shape:
            obs_all.append(np.concatenate([des, cpos], axis=1))   # (T,6) [des | c_pos]
        else:
            obs_all.append(des)                           # fallback: des-only on shape mismatch
        act_all.append(np.asarray(d.get('desired_actions')))
        cand_all.append(list(d.get('all_candidates', [])))         # list of (B,H,3) — full fan
        selidx_all.append(np.asarray(d.get('selected_idx', []), dtype=np.int32))

    _ci = agent.history_context_info
    _max_phys = np.array([float(np.max(e)) if len(e) else 0.0
                          for e in agent.history_pos_tracking_errors], dtype=np.float32)
    _hcm = agent.history_constraint_metrics

    def _cm(key, default=0.0):
        return np.array([m.get(key, default) for m in _hcm], dtype=np.float32)

    return dict(
        obs_all=np.array(obs_all, dtype=object),
        act_all=np.array(act_all, dtype=object),
        sampled_trajectories_all=np.array(cand_all, dtype=object),
        selected_idx_all=np.array(selidx_all, dtype=object),
        success_relaxed=np.array(agent.history_success_relaxed),
        n_steps=np.array(agent.history_n_steps),
        avg_time=np.array(agent.history_avg_time),
        mean_dist_per_rollout=np.array(agent.history_rollout_mean_dist),
        physical_tracking_errors=np.array(agent.history_pos_tracking_errors, dtype=object),
        max_phys_error_per_rollout=_max_phys,
        outcome_max_physical_tracking_error=_max_phys,
        context_box_init_xy=np.array([[c.get('box_init_xy', [0.0, 0.0])[0],
                                       c.get('box_init_xy', [0.0, 0.0])[1]] for c in _ci],
                                     dtype=np.float32),
        context_target_xy=np.array([[c.get('target_xy', [0.0, 0.0])[0],
                                     c.get('target_xy', [0.0, 0.0])[1]] for c in _ci],
                                   dtype=np.float32),
        context_box_angle_deg=np.array([c.get('box_init_angle_deg', 0.0) for c in _ci],
                                       dtype=np.float32),
        context_target_angle_deg=np.array([c.get('target_angle_deg', 0.0) for c in _ci],
                                          dtype=np.float32),
        context_init_xy_dist=np.array([c.get('init_xy_dist', 0.0) for c in _ci],
                                      dtype=np.float32),
        contact_first_step=np.array(agent.history_contact_first_step, dtype=np.int32),
        contact_last_step=np.array(agent.history_contact_last_step, dtype=np.int32),
        contact_first_pos_xy=np.array(agent.history_contact_first_pos_xy, dtype=np.float32),
        contact_last_pos_xy=np.array(agent.history_contact_last_pos_xy, dtype=np.float32),
        constraint_exec_n_violated_steps=_cm('exec_n_violated_steps'),
        constraint_exec_sat_rate=_cm('exec_constraint_sat_rate', 1.0),
        constraint_exec_zero_violation=_cm('exec_zero_violation_rollout'),
        constraint_exec_bounds_viol_count=_cm('exec_bounds_viol_count'),
        constraint_exec_halfspace_viol_count=_cm('exec_halfspace_viol_count'),
        constraint_exec_obstacle_viol_count=_cm('exec_obstacle_viol_count'),
        constraint_exec_max_bounds_viol_m=_cm('exec_max_bounds_viol_m'),
        constraint_exec_max_halfspace_viol_m=_cm('exec_max_halfspace_viol_m'),
        constraint_exec_max_obstacle_penetration_m=_cm('exec_max_obstacle_penetration_m'),
        constraint_exec_margin_mean_m=_cm('exec_constraint_margin_mean_m'),
        constraint_exec_first_violation_step=_cm('exec_first_violation_step', -1),
        constraint_exec_longest_safe_streak=_cm('exec_longest_safe_streak'),
        constraint_exec_dyn_err_mean=_cm('exec_dynamics_consistency_error_mean'),
        constraint_exec_dyn_err_max=_cm('exec_dynamics_consistency_error_max'),
        constraint_plan_post_viol_rate_mean=_cm('plan_post_viol_rate_mean'),
        constraint_plan_post_viol_rate_max=_cm('plan_post_viol_rate_max'),
        constraint_plan_n_replan_steps=_cm('plan_n_replan_steps'),
        # Fix_15.3: projection-circuit-breaker flags per rollout. cb_tripped==1 → that rollout ran
        # (partly) UNPROJECTED (sustained SLSQP slowness); its constraint_exec_* metrics are INVALID.
        projection_cb_tripped=np.array([1 if s > 0 else 0 for s in agent.history_cb_skipped_steps],
                                       dtype=np.int32),
        projection_cb_skipped_steps=np.array(agent.history_cb_skipped_steps, dtype=np.int32),
    )


def _save_partial_npz(path, agent, seed, args_dict):
    """C5 crash-safety: atomically write a `<variant>.partial.npz` snapshot of every raw
    per-rollout array gathered so far.

    Deliberately a SIDECAR, not `<variant>.npz`: the DA pipeline only ever reads
    `<variant>.npz`, which is written exactly once, atomically, at variant END. So a
    completed run's analysis never sees partial data, and today's clean "missing npz =>
    variant didn't finish" semantics are preserved. This file is pure raw-data insurance
    against a mid-variant SLURM kill and is deleted on successful completion. Aggregate
    scalars that require test_agent's return (entropy, per-context mode_encoding) are not
    available mid-run, so they are best-effort (`complete=False` flags this).
    """
    done = [r for r in range(agent.rollout_counter + 1)
            if agent.master_rollout_history.get(f'rollout_{r}')]
    strict = np.array([bool(agent.master_rollout_history[f'rollout_{r}'].get('success', False))
                       for r in done], dtype=bool)
    modes  = np.array([int(agent.master_rollout_history[f'rollout_{r}'].get('mode', 0))
                       for r in done], dtype=np.int32)
    mdist  = np.array([float(agent.master_rollout_history[f'rollout_{r}'].get('mean_distance', 0.0))
                       for r in done], dtype=np.float32)
    payload = _collect_per_rollout_arrays(agent)
    payload.update(
        success_rate=float(strict.mean()) if strict.size else 0.0,
        entropy=float('nan'),
        n_success=strict, success_strict=strict,
        mode_per_rollout=modes,
        mean_distance=mdist,
        seed=seed, complete=False,
        n_rollouts_done=int(strict.size),
        args=args_dict,
    )
    tmp = f'{path}.tmp{os.getpid()}.npz'
    np.savez(tmp, **payload)
    os.replace(tmp, path)   # atomic on same filesystem


# ── Logging ───────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, *files):
        self.files = [f if hasattr(f, 'write') else open(f, 'a') for f in files]
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

# ── Expert reference generation ───────────────────────────────────────────────

def generate_expert_reference(save_path, n_rollouts=3):
    """Generate ground-truth expert videos from the dataset for reference."""
    expert_dir = os.path.join(save_path, 'expert_references')
    all_exist = all(
        os.path.exists(os.path.join(expert_dir, f'expert_rollout_{i}.mp4')) or
        os.path.exists(os.path.join(expert_dir, f'expert_rollout_{i}.gif'))
        for i in range(n_rollouts)
    )
    if all_exist:
        print(f'[ expert ] Reference videos already exist in {expert_dir}. Skipping.')
        return

    print(f'[ expert ] Generating {n_rollouts} expert reference videos...')
    os.makedirs(expert_dir, exist_ok=True)

    try:
        from agents.utils.sim_path import sim_framework_path
        from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env

        state_data_dir = sim_framework_path('environments/dataset/data/aligning/all_data/state')
        env = Robot_Push_Env(render=False, if_vision=True)
        env.start()

        for idx in range(n_rollouts):
            file_name = f'env_{idx}.pkl'
            try:
                with open(os.path.join(state_data_dir, file_name), 'rb') as f:
                    expert_data = pickle.load(f)
            except Exception:
                all_files = sorted(os.listdir(state_data_dir))
                if idx >= len(all_files):
                    continue
                with open(os.path.join(state_data_dir, all_files[idx]), 'rb') as f:
                    expert_data = pickle.load(f)

            expert_path = expert_data['robot']['des_c_pos']
            box_pos    = expert_data['push-box']['pos'][0]
            box_quat   = expert_data['push-box']['quat'][0]
            target_pos = expert_data['target-box']['pos'][0]
            target_quat = expert_data['target-box']['quat'][0]
            context = (box_pos, box_quat, target_pos, target_quat)

            env.reset(random=False, context=context)
            frames = []
            for step in range(len(expert_path)):
                sim_action = np.concatenate((expert_path[step], [0, 1, 0, 0]))
                obs, _, _, _ = env.step(sim_action)
                _, bp_img, ih_img = obs
                frames.append(np.concatenate(
                    [cv2.cvtColor(bp_img, cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(ih_img, cv2.COLOR_BGR2RGB)], axis=1))

            save_file = os.path.join(expert_dir, f'expert_rollout_{idx}.mp4')
            try:
                imageio.mimsave(save_file, frames, fps=20)
                print(f'  [ expert ] Saved {save_file}')
            except Exception:
                gif_file = save_file.replace('.mp4', '.gif')
                try:
                    imageio.mimsave(gif_file, frames, fps=10)
                    print(f'  [ expert ] Saved {gif_file}')
                except Exception as e:
                    print(f'  [ expert ] Failed to save rollout {idx}: {e}')

        env.close()
    except Exception as e:
        print(f'[ expert ] WARNING — expert reference generation failed: {e}')

# ── VisualAgentWrapper ────────────────────────────────────────────────────────

class VisualAgentWrapper:
    """
    D3IL-compatible agent wrapper for Visual-DPCC (9D trajectory).
    Logging pattern ported from ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py.

    D3IL calls:
        agent.reset()                        — start of each rollout
        agent.predict(state, if_vision=True) — every sim step
        agent.update_rollout_info(info)      — end of each rollout

    State received: (bp_image, inhand_image, des_robot_pos)
    9D obs: obs_6d = [des_robot_pos(3), des_robot_pos(3)]
            (c_pos not exposed by D3IL; des_c_pos ≈ c_pos under PD control)
    """

    def __init__(self, diffusion_model, device,
                 window_size=1, obs_seq_len=1, action_seq_size=1,
                 save_path=None, record_mode='all',
                 obs_normalizer=None, act_normalizer=None,
                 batch_size=1, projector=None,
                 trajectory_selection='random',
                 eval_on_train=False, variant='unspecified',
                 max_action_delta=None,
                 mpc_foresight_stride=6,
                 geo_config=None,
                 is_tightened=False,
                 total_rollouts=None,
                 seed=None,
                 npz_args=None,
                 partial_npz_every=5):

        self.model              = diffusion_model
        self.device             = device
        self.window_size        = window_size
        self.obs_seq_len        = obs_seq_len
        self.obs_normalizer     = obs_normalizer
        self.act_normalizer     = act_normalizer
        self.batch_size         = batch_size
        self.projector          = projector
        self.trajectory_selection = trajectory_selection
        self.eval_on_train      = eval_on_train
        self.save_path          = save_path
        self.record_mode        = record_mode
        self.variant            = variant
        # C5: crash-safety partial-npz sidecar (see _save_partial_npz).
        self._seed              = seed
        self._npz_args          = npz_args if npz_args is not None else {}
        self._partial_npz_every = partial_npz_every

        model_horizon = getattr(self.model, 'horizon', window_size)
        self.action_seq_size = min(action_seq_size, model_horizon)
        self.action_counter  = self.action_seq_size   # force replan on first step
        self.curr_action_seq = None
        self.prev_observations = None

        self.rollout_counter = -1
        self.step_counter    = 0
        self.mental_robot_pos = None
        self.last_predicted_pos = None

        self.bp_image_context    = deque(maxlen=self.window_size)
        self.inhand_image_context = deque(maxlen=self.window_size)
        self.obs_context          = deque(maxlen=self.obs_seq_len)

        self.history_real_pos            = []
        self.history_desired_actions     = []
        self.history_full_plans          = []
        self.history_all_candidates      = []   # Fix 8: (B,H,3) per replan step, all candidates
        self.history_selected_idx        = []   # Fix 8: which index was chosen per replan step
        self.history_n_steps             = []
        self.history_avg_time            = []
        self.history_rollout_mean_dist   = []   # Fix 9: mean_distance per rollout for summary
        self.history_pos_tracking_errors = []   # Fix 9: physical tracking error |c_pos - des_c_pos|
        self.curr_rollout_tracking_errors = []  # Fix 9: physical tracking error per step
        self.curr_rollout_all_candidates  = []  # Fix 8: per-rollout accumulator (Fix 9: stores c_pos dims)
        self.curr_rollout_selected_idx    = []  # Fix 8: per-rollout accumulator
        self.curr_rollout_c_pos           = []  # Fix 9: actual robot position per step
        self.curr_rollout_mode_history     = []  # Json_Orgnize_C4: per-step proximity mode (0=within 5.1cm of box)
        self.curr_context_info            = {}  # Fix 10: set by record_context_info each rollout
        self.history_context_info         = []  # Fix 10: per-rollout context records
        self.history_success_relaxed      = []  # Json_Orgnize_C4: position-only success per rollout
        self.history_contact_first_step   = []  # Json_Orgnize_C4
        self.history_contact_last_step    = []
        self.history_contact_first_pos_xy = []
        self.history_contact_last_pos_xy  = []
        self.curr_rollout_time           = 0
        self.master_rollout_history      = {}
        self.video_frames                = []
        self.max_action_delta            = max_action_delta
        self.mpc_foresight_stride        = mpc_foresight_stride
        self.geo_config                  = geo_config or {}
        self.is_tightened                = is_tightened
        # Fix_11 (Gen7/Gen6V4): progress/ETA for this (geo, variant) item's rollout loop —
        # n_contexts*n_trajectories, known at construction time; None disables the line.
        self.total_rollouts              = total_rollouts
        self._item_t0                    = time.time()
        self.curr_rollout_act_magnitudes = []
        self.curr_rollout_dist_to_target = []
        self.curr_rollout_clamp_events   = []
        self.history_act_magnitudes      = []
        self.history_dist_to_target      = []
        self.history_clamp_events        = []
        self._replan_count               = 0
        # UF-16.3: constraint metrics
        self.history_constraint_metrics  = []
        self._plan_post_viol_rates       = []
        # Fix_15.3: projection-circuit-breaker accounting (projection.py Fix_15.2). Counts FM
        # steps whose projection was SKIPPED because the sustained-slowness breaker was OPEN —
        # those steps ran UNPROJECTED, so the rollout's constraint metrics are NOT trustworthy.
        self.curr_rollout_cb_skipped_steps = 0
        self.history_cb_skipped_steps      = []
        self.history_cb_trips              = []

    def reset(self):
        self.mental_robot_pos   = None
        self.prev_observations  = None
        self.last_predicted_pos = None
        self.action_counter     = self.action_seq_size
        self.curr_action_seq    = None
        self.rollout_counter   += 1
        self.step_counter       = 0
        self.curr_rollout_time  = 0
        self.curr_rollout_tracking_errors.clear()
        self.curr_rollout_c_pos.clear()            # Fix 9
        self.curr_rollout_mode_history.clear()     # Json_Orgnize_C4
        self.curr_context_info = {}                # Fix 10
        self.history_real_pos.clear()
        self.history_desired_actions.clear()
        self.history_full_plans.clear()
        self.curr_rollout_all_candidates.clear()   # Fix 8
        self.curr_rollout_selected_idx.clear()     # Fix 8
        self.bp_image_context.clear()
        self.inhand_image_context.clear()
        self.obs_context.clear()
        self.video_frames.clear()
        self.curr_rollout_act_magnitudes.clear()
        self.curr_rollout_dist_to_target.clear()
        self.curr_rollout_clamp_events.clear()
        self._replan_count = 0
        self._plan_post_viol_rates.clear()   # UF-16.3
        self.curr_rollout_cb_skipped_steps = 0   # Fix_15.3
        # REAL_TIME_RECORDING_UPDATE — fresh per-rollout timing recorder.
        self.rt_rec = RTRecorder(
            episode_id=f'{self.variant}_rollout{self.rollout_counter}',
            variant=self.variant, scene='aligning', system='VisualAligning_FM',
            control_hz=RT_CONTROL_HZ, batch_size=self.batch_size,
            horizon=getattr(self.model, 'horizon', self.action_seq_size),
            text_log=bool(self.save_path))

    def update_rollout_info(self, info):
        """Called by Aligning_Sim at rollout end. Mirrors ddpm_encdec verbose format."""
        success   = info.get('success', False)
        mean_dist = info.get('mean_distance', 0.0)
        mode      = info.get('mode', 0)
        ridx      = int(info.get('context', self.rollout_counter))

        max_phys_err = float(np.max(self.curr_rollout_tracking_errors)
                             if self.curr_rollout_tracking_errors else 0.0)
        avg_time = float(self.curr_rollout_time / max(1, self._replan_count))  # Fix 12: per-replan avg

        # UF-16: final box position + angle from aligning_sim
        _fbp = info.get('final_box_pos')
        _fbq = info.get('final_box_quat')
        if _fbp is not None and self.curr_context_info:
            _fx, _fy = float(_fbp[0]), float(_fbp[1])
            _w, _x, _y, _z = float(_fbq[0]), float(_fbq[1]), float(_fbq[2]), float(_fbq[3])
            _final_angle_deg = float(np.degrees(
                np.arctan2(2*(_w*_z + _x*_y), 1 - 2*(_y**2 + _z**2))
            ))
            _txy = self.curr_context_info.get('target_xy', [0.0, 0.0])
            _final_xy_dist = float(np.sqrt((_fx - _txy[0])**2 + (_fy - _txy[1])**2))
            self.curr_context_info.update({
                'final_box_xy':        [_fx, _fy],
                'final_box_angle_deg': _final_angle_deg,
                'final_xy_dist':       _final_xy_dist,
            })

        # Json_Orgnize_C4 (Finding #5): success_relaxed — position-only, ignore final angle.
        # pos_min_dist is threaded through `info` from the live env (aligning_sim.py); fall
        # back to the env's own hardcoded default (aligining.py:198) only if an older
        # aligining_sim.py without the threading fix is ever used, so this never hard-fails.
        _pos_min_dist = float(info.get('pos_min_dist', 0.018))
        _final_xy_dist_for_relax = self.curr_context_info.get('final_xy_dist')
        success_relaxed = (bool(_final_xy_dist_for_relax <= _pos_min_dist)
                            if _final_xy_dist_for_relax is not None else False)

        # Json_Orgnize_C4 (Finding #6): first/last-contact step+position — a proximity proxy
        # (robot-box XY dist < aligining.py's robot_box_dist=0.051m via check_mode()'s `mode`),
        # NOT verified MuJoCo mesh contact (no such API exists for this env). `-1`/None sentinel
        # when the robot never got close during the whole rollout.
        contact_first_step = contact_last_step = -1
        contact_first_pos_xy = contact_last_pos_xy = None
        if self.curr_rollout_mode_history and self.curr_rollout_c_pos:
            _contact_idx = [i for i, m in enumerate(self.curr_rollout_mode_history) if m == 0]
            if _contact_idx:
                contact_first_step = int(_contact_idx[0])
                contact_last_step  = int(_contact_idx[-1])
                _n_pos = len(self.curr_rollout_c_pos)
                if contact_first_step < _n_pos:
                    contact_first_pos_xy = [float(self.curr_rollout_c_pos[contact_first_step][0]),
                                             float(self.curr_rollout_c_pos[contact_first_step][1])]
                if contact_last_step < _n_pos:
                    contact_last_pos_xy = [float(self.curr_rollout_c_pos[contact_last_step][0]),
                                            float(self.curr_rollout_c_pos[contact_last_step][1])]

        self.master_rollout_history[f'rollout_{ridx}'] = {
            'real_robot_pos':      np.array(self.history_real_pos),
            'c_pos_history':       np.array(self.curr_rollout_c_pos),         # Fix 9
            'desired_actions':     np.array(self.history_desired_actions),
            'full_plans':          np.array(self.history_full_plans),
            'all_candidates':      list(self.curr_rollout_all_candidates),   # Fix 8/9: list of (B,H,3) c_pos
            'selected_idx':        list(self.curr_rollout_selected_idx),     # Fix 8: list of int
            'success':   bool(success),
            'mean_distance': float(mean_dist),
            'mode':      int(mode),
            'steps':     int(self.step_counter),
            'avg_time':  avg_time,
            'max_physical_tracking_error': max_phys_err,                     # Fix 9: |c_pos - des_c_pos|
            'act_magnitudes':     list(self.curr_rollout_act_magnitudes),
            'dist_to_target':     list(self.curr_rollout_dist_to_target),
            'clamp_events':       list(self.curr_rollout_clamp_events),
            'context_info':       dict(self.curr_context_info),              # Fix 10
            'success_relaxed':        success_relaxed,                       # Json_Orgnize_C4
            'contact_first_step':     contact_first_step,
            'contact_last_step':      contact_last_step,
            'contact_first_pos_xy':   contact_first_pos_xy,
            'contact_last_pos_xy':    contact_last_pos_xy,
            # Fix_15.3: projection circuit-breaker health for this rollout.
            'projection_health': {
                'cb_tripped':       bool(self.curr_rollout_cb_skipped_steps > 0),
                'cb_skipped_steps': int(self.curr_rollout_cb_skipped_steps),
                'cb_trips':         int(getattr(self.projector, '_cb_trips', 0)),
                'backstop_hits':    int(getattr(self.projector, '_cost_exploded_count', 0)),
            },
        }

        # UF-16.3: compute constraint satisfaction metrics for this rollout.
        # Execution metrics always check against NOMINAL constraints (enlarge=0), matching
        # original DPCC paper convention: all variants measured on the same nominal boundary
        # so that tightened vs nominal comparison is meaningful (tightened should show fewer
        # violations because the projector gave it a δ buffer over the nominal boundary).
        _cmetrics = {}
        if self.geo_config and self.curr_rollout_c_pos:
            _cmetrics = check_trajectory_constraints(
                self.curr_rollout_c_pos,
                list(self.history_desired_actions),
                self.geo_config,
                0.0)
        if self._plan_post_viol_rates:
            _cmetrics['plan_post_viol_rate_mean'] = float(np.mean(self._plan_post_viol_rates))
            _cmetrics['plan_post_viol_rate_max']  = float(np.max(self._plan_post_viol_rates))
            _cmetrics['plan_n_replan_steps']       = len(self._plan_post_viol_rates)
        self.master_rollout_history[f'rollout_{ridx}']['constraint_metrics'] = _cmetrics
        self.history_constraint_metrics.append(_cmetrics)

        self.history_n_steps.append(self.step_counter)
        self.history_avg_time.append(avg_time)
        self.history_cb_skipped_steps.append(int(self.curr_rollout_cb_skipped_steps))   # Fix_15.3
        self.history_cb_trips.append(int(getattr(self.projector, '_cb_trips', 0)))       # Fix_15.3
        # REAL_TIME_RECORDING_UPDATE — write per-rollout realtime_<variant>_rollout<ridx>.log + SUMMARY.
        if getattr(self, 'rt_rec', None) is not None and self.save_path:
            self.rt_rec.save(
                os.path.join(self.save_path, f'realtime_{self.variant}_rollout{ridx}.log'),
                behaviour={'success': int(bool(success)), 'steps': int(self.step_counter),
                           'mean_distance': round(float(mean_dist), 4)})
        self.history_rollout_mean_dist.append(float(mean_dist))              # Fix 9
        self.history_pos_tracking_errors.append(
            np.array(self.curr_rollout_tracking_errors))
        self.history_all_candidates.append(list(self.curr_rollout_all_candidates))  # Fix 8
        self.history_selected_idx.append(list(self.curr_rollout_selected_idx))      # Fix 8
        self.history_act_magnitudes.append(list(self.curr_rollout_act_magnitudes))
        self.history_dist_to_target.append(list(self.curr_rollout_dist_to_target))
        self.history_clamp_events.append(list(self.curr_rollout_clamp_events))
        self.history_context_info.append(dict(self.curr_context_info))      # Fix 10
        self.history_success_relaxed.append(success_relaxed)                # Json_Orgnize_C4
        self.history_contact_first_step.append(contact_first_step)
        self.history_contact_last_step.append(contact_last_step)
        self.history_contact_first_pos_xy.append(
            contact_first_pos_xy if contact_first_pos_xy is not None else [float('nan'), float('nan')])
        self.history_contact_last_pos_xy.append(
            contact_last_pos_xy if contact_last_pos_xy is not None else [float('nan'), float('nan')])

        ctx_type = 'Seen Training Context' if self.eval_on_train else 'Unseen Test Context'
        ci = self.curr_context_info
        print(f'[ {ctx_type} {ridx} Finished ]')
        if ci:                                                               # Fix 10: scene config
            print(f'  - Context idx: {ci.get("context_idx")}')
            print(f'  - Box  init XY=({ci["box_init_xy"][0]:.3f}, {ci["box_init_xy"][1]:.3f})  '
                  f'angle={ci["box_init_angle_deg"]:.1f}°')
            print(f'  - Target   XY=({ci["target_xy"][0]:.3f}, {ci["target_xy"][1]:.3f})  '
                  f'angle={ci["target_angle_deg"]:.1f}°')
            print(f'  - Init XY dist (box→target): {ci["init_xy_dist"]:.4f} m')
            if 'final_box_xy' in ci:
                print(f'  - Box  final XY=({ci["final_box_xy"][0]:.3f}, {ci["final_box_xy"][1]:.3f})  '
                      f'angle={ci["final_box_angle_deg"]:.1f}°'
                      f'  (dist_to_target: {ci["final_xy_dist"]:.4f} m)')
        print(f'  - Total Steps: {self.step_counter}')
        print(f'  - Success status: {success}')
        print(f'  - Final Mean Distance: {mean_dist:.6f} m')
        print(f'  - Environment Mode: {mode}')
        print(f'  - Max Physical Tracking Error: {max_phys_err:.6f} m')     # Fix 9: now meaningful
        print(f'  - Avg Inference Time: {avg_time:.4f} seconds/replan')
        print(f'  - Clamp events: {len(self.curr_rollout_clamp_events)}')
        if self.curr_rollout_cb_skipped_steps > 0:   # Fix_15.3
            print(f'  - ⚠ PROJECTION CIRCUIT-BREAKER TRIPPED: {self.curr_rollout_cb_skipped_steps} '
                  f'step(s) UNPROJECTED (sustained SLSQP slowness) — constraint metrics NOT valid')
        # Fix_11: progress/ETA within this (geo, variant) item — the debug block above was
        # already good, it just never said how many rollouts are left or how long that'll
        # take. `rollout_counter` is 0-based and already incremented by this rollout's own
        # reset(), so +1 is the count completed so far.
        if self.total_rollouts:
            _done = self.rollout_counter + 1
            _elapsed = time.time() - self._item_t0
            _avg = _elapsed / _done
            _eta = _avg * (self.total_rollouts - _done)
            print(f'  - Progress: rollout {_done}/{self.total_rollouts}  '
                  f'({_elapsed:.1f}s elapsed this item, ~{_eta:.1f}s to go)')
        print('-' * 80 + '\n')

        if self.save_path is not None:
            self._export_rollout_realtime(ridx)   # Fix 9: handles PNG+JSON+pkl+video, no separate _save_diagnostics

    def record_step_info(self, info):
        """Called by Aligning_Sim after each env.step() — accumulates per-step mean_distance
        and (Json_Orgnize_C4) the proximity `mode` (0 = EE within robot_box_dist of the box),
        used at rollout-end to derive first/last-contact step+position. `mode` is a distance
        proxy (aligining.py's check_mode()), not verified MuJoCo mesh contact."""
        d = info.get('mean_distance')
        if d is not None:
            self.curr_rollout_dist_to_target.append(float(d))
        self.curr_rollout_mode_history.append(int(info.get('mode', 1)))

    def capture_frame(self, bp_np, inhand_np):
        """Non-visual GIF hook. Receives (C,H,W) float[0,1] BGR images from
        Aligning_Sim's non-visual branch — same shape/dtype/convention the
        visual predict() receives. Capture logic is copied verbatim from
        the visual predict() block (cv2.cvtColor BGR2RGB → mimsave RGB)."""
        if self.record_mode == 'none':
            return
        try:
            bp_vis     = cv2.cvtColor((bp_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            inhand_vis = cv2.cvtColor((inhand_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
            frame = np.concatenate([bp_vis, inhand_vis], axis=1)
            cv2.putText(frame, f's{self.step_counter}', (5, 18),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
            self.video_frames.append(frame)
        except Exception:
            pass

    def record_context_info(self, context, context_idx):
        """Called by Aligning_Sim after reset — stores initial scene config. Fix 10."""
        pos, quat, target_pos, target_quat = context
        init_xy_dist = float(np.linalg.norm(
            np.array([pos[0], pos[1]]) - np.array([target_pos[0], target_pos[1]])
        ))
        self.curr_context_info = {
            'context_idx':        int(context_idx),
            'box_init_xy':        [float(pos[0]), float(pos[1])],
            'box_init_angle_deg': float(pos[2]),
            'target_xy':          [float(target_pos[0]), float(target_pos[1])],
            'target_angle_deg':   float(target_pos[2]),
            'init_xy_dist':       init_xy_dist,
        }

    # Fix 9: _save_diagnostics() removed — video/gif + JSON now consolidated in _export_rollout_realtime().

    def _export_rollout_realtime(self, rollout_idx):
        """Per-rollout PNG (9-panel) + JSON + pkl + video. Fix 9: consolidated into diagnostics/."""
        try:
            diag_path = os.path.join(self.save_path, 'diagnostics')
            os.makedirs(diag_path, exist_ok=True)

            data = self.master_rollout_history[f'rollout_{rollout_idx}']

            # ── Video / GIF (Fix 9: moved from _save_diagnostics) ─────────
            if self.record_mode != 'none' and self.video_frames:
                if self.record_mode in ['video', 'all']:
                    try:
                        imageio.mimsave(os.path.join(diag_path, f'rollout_{rollout_idx}.mp4'),
                                        self.video_frames, fps=20)
                    except Exception as e:
                        print(f'[ WARNING ] MP4 failed: {e}')
                if self.record_mode in ['gif', 'all']:
                    try:
                        imageio.mimsave(os.path.join(diag_path, f'rollout_{rollout_idx}.gif'),
                                        self.video_frames, fps=10)
                    except Exception as e:
                        print(f'[ WARNING ] GIF failed: {e}')

            # C5: per-rollout `_data.pkl` removed — every raw array it held (c_pos_history,
            # all_candidates, selected_idx, …) now lives in <variant>.npz. Crash-safety is
            # provided instead by the incremental <variant>.partial.npz sidecar written at the
            # end of this method every `partial_npz_every` rollouts.

            # Json_Orgnize_C4: grouped schema replacing the old flat/chaotic layout.
            # `exec_n_steps` (Finding #1, was a literal duplicate of `steps`) is dropped —
            # `timing.steps` is the single canonical step count now.
            _cm = data.get('constraint_metrics', {})
            _nested_cm = _nest_constraint_metrics(_cm)
            _ci = data.get('context_info', {})
            _contact_note = ('proximity proxy (robot-box XY dist < 0.051m), '
                              'not physical mesh contact')
            stats = {
                'rollout_index': int(rollout_idx),
                'mode':          int(data.get('mode', 0)),
                'success': {
                    'strict':  bool(data.get('success', False)),
                    'relaxed': bool(data.get('success_relaxed', False)),
                },
                'outcome': {
                    'mean_distance':               float(data.get('mean_distance', 0.0)),
                    'max_physical_tracking_error':  float(data.get('max_physical_tracking_error', 0.0)),
                },
                'timing': {
                    'steps':                          int(data.get('steps', 0)),
                    'avg_inference_time_per_replan':   float(data.get('avg_time', 0.0)),  # Fix 12
                },
                'context': dict(_ci),
                'contact': {
                    'first_step':    (int(data['contact_first_step'])
                                       if data.get('contact_first_step', -1) >= 0 else None),
                    'first_pos_xy':  data.get('contact_first_pos_xy'),
                    'last_step':     (int(data['contact_last_step'])
                                       if data.get('contact_last_step', -1) >= 0 else None),
                    'last_pos_xy':   data.get('contact_last_pos_xy'),
                    'note':          _contact_note,
                },
                'constraint': _nested_cm,
            }
            if _cm:
                _ex = _nested_cm['exec']
                print(f'  [ constraints ] sat={_ex["constraint_sat_rate"]:.3f}  '
                      f'violated={_ex["n_violated_steps"]}steps'
                      f'  (bounds={_ex["by_family"]["bounds"]["viol_count"]} '
                      f'hs={_ex["by_family"]["halfspace"]["viol_count"]} '
                      f'obs={_ex["by_family"]["obstacles"]["viol_count"]})')
                print(f'    first_viol_step={_ex["first_violation_step"]}  '
                      f'longest_safe={_ex["longest_safe_streak"]}  '
                      f'margin={_ex["margin_mean_m"]:.4f}m  '
                      f'dyn_err={_ex["dynamics_consistency_error"]["mean"]:.4f}m')
                _pv = _nested_cm['plan'].get('post_viol_rate_mean', 0.0)
                print(f'    plan_post_viol_rate={_pv:.4f}  '
                      f'zero_viol={_ex["zero_violation_rollout"]}')
            with open(os.path.join(diag_path, f'rollout_{rollout_idx}_stats.json'), 'w') as sf:
                json.dump(stats, sf, indent=4)

            real_pos    = data['real_robot_pos']    # (T, 3) des_c_pos
            c_pos_hist  = data.get('c_pos_history') # (T, 3) actual robot pos, or None

            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'Rollout {rollout_idx} — MPC vs Real  '
                         f'(success={data.get("success")})')

            # Row 0 — Spatial
            # Fix 9: c_pos dims (actual predicted positions, unnormalized) — no cumsum, no start offset
            all_cands_list = data.get('all_candidates', [])
            sel_idx_list   = data.get('selected_idx',   [])
            for step_i, (cands, sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                if step_i % 4 != 0:
                    continue
                for b in range(cands.shape[0]):
                    if b == sel:
                        axes[0, 0].plot(cands[b, :, 0], cands[b, :, 1],
                                        color='green', linewidth=0.8, alpha=0.9, zorder=5)
                    else:
                        axes[0, 0].plot(cands[b, :, 0], cands[b, :, 1],
                                        color='gray', linewidth=0.2, alpha=0.2)
            axes[0, 0].plot(real_pos[:, 0], real_pos[:, 1], 'k-', linewidth=2,
                            label='Real Path (des_c_pos)', zorder=10)
            n_cands = all_cands_list[0].shape[0] if all_cands_list else 0
            axes[0, 0].set_title(f'XY — MPC foresight ({n_cands} cands/step: green=selected, gray=others)')
            axes[0, 0].set_xlabel('X (m)'); axes[0, 0].set_ylabel('Y (m)')
            axes[0, 0].legend(fontsize=8)

            # Fix 9: overlay des_c_pos (black) and c_pos/actual (red dashed) — analogous to DPCC x_des vs x
            axes[0, 1].plot(real_pos[:, 0], 'k-', label='X des')
            if c_pos_hist is not None and len(c_pos_hist):
                axes[0, 1].plot(np.array(c_pos_hist)[:, 0], 'r--', alpha=0.7, label='X actual')
                axes[0, 1].legend(fontsize=7)
            axes[0, 1].set_title('X — des (black) vs actual (red)')
            axes[0, 1].set_ylabel('Meters')

            axes[0, 2].plot(real_pos[:, 1], 'k-', label='Y des')
            if c_pos_hist is not None and len(c_pos_hist):
                axes[0, 2].plot(np.array(c_pos_hist)[:, 1], 'r--', alpha=0.7, label='Y actual')
                axes[0, 2].legend(fontsize=7)
            axes[0, 2].set_title('Y — des (black) vs actual (red)')

            # Row 1 — Task Progress
            dist_curve = data.get('dist_to_target', [])
            if dist_curve:
                axes[1, 0].plot(dist_curve, 'r-')
                axes[1, 0].axhline(0, color='g', linestyle='--', alpha=0.5, label='target')
                axes[1, 0].legend(fontsize=8)
            axes[1, 0].set_title('Distance to Target over Steps')
            axes[1, 0].set_ylabel('m')

            axes[1, 1].plot(real_pos[:, 2], 'k-', label='Z des')
            if c_pos_hist is not None and len(c_pos_hist):
                axes[1, 1].plot(np.array(c_pos_hist)[:, 2], 'r--', alpha=0.7, label='Z actual')
                axes[1, 1].legend(fontsize=7)
            axes[1, 1].set_title('Z — des (black) vs actual (red)')
            axes[1, 1].set_ylabel('Meters')

            # Fix 9: physical tracking error |c_pos - des_c_pos| replaces trivially-zero mental model error
            phys_errs = data.get('max_physical_tracking_error', 0.0)
            tracking_list = list(self.curr_rollout_tracking_errors) if self.curr_rollout_tracking_errors else []
            if tracking_list:
                axes[1, 2].plot(tracking_list, 'g-')
            axes[1, 2].set_title(f'Physical Tracking Error |c_pos - des| (m)\nmax={phys_errs:.4f}')
            axes[1, 2].set_ylabel('m')

            # Row 2 — Action Quality
            act_mags = data.get('act_magnitudes', [])
            if act_mags:
                axes[2, 0].plot(act_mags, 'c-')
            axes[2, 0].set_title('Action Magnitude per Step (m)')
            axes[2, 0].set_ylabel('m')

            vels = np.linalg.norm(real_pos[1:] - real_pos[:-1], axis=1)
            axes[2, 1].plot(vels, 'm-')
            axes[2, 1].set_title('End-Effector Velocity')

            clamp_events = data.get('clamp_events', [])
            if clamp_events:
                ce_steps = [e[0] for e in clamp_events]
                ce_mags  = [e[1] for e in clamp_events]
                axes[2, 2].scatter(ce_steps, ce_mags, c='r', s=20, zorder=3)
                if self.max_action_delta is not None:
                    axes[2, 2].axhline(self.max_action_delta, color='orange',
                                       linestyle='--', alpha=0.7,
                                       label=f'limit={self.max_action_delta}m')
                    axes[2, 2].legend(fontsize=8)
            axes[2, 2].set_title(f'Clamp Events ({len(clamp_events)} total)')
            axes[2, 2].set_xlabel('Step'); axes[2, 2].set_ylabel('Raw |a| (m)')

            plt.tight_layout()
            fig.savefig(os.path.join(diag_path, f'rollout_{rollout_idx}_report.png'))
            plt.close(fig)

            # ── Standalone high-res MPC decision-point plot: XY (left) + XYZ 3D (right) ──
            # U11: all candidates uniform green solid; replan decision-point dots (black);
            #      des=black solid, actual=red solid (no dashes); PNG@200DPI + SVG.
            if all_cands_list:
                from matplotlib.lines import Line2D as _Line2D
                _STRIDE   = self.mpc_foresight_stride   # U11.2: yaml-settable (mpc_foresight_stride)
                n_steps   = len(real_pos)
                n_replans = len(all_cands_list)
                spr       = max(1, n_steps // max(1, n_replans))  # env steps per replan
                c_arr = (np.array(c_pos_hist)
                         if (c_pos_hist is not None and len(c_pos_hist)) else None)

                fig_mpc = plt.figure(figsize=(26, 11))
                fig_mpc.suptitle(
                    f'Rollout {rollout_idx} — MPC Decision Points  '
                    f'(success={data.get("success")},  {n_cands} candidates/step,  '
                    f'every {_STRIDE} replans shown)',
                    fontsize=13)
                # Fix_15.3: loud banner when this rollout's projection circuit breaker tripped —
                # the candidate fan is (partly) UNPROJECTED, so it is NOT constraint-valid.
                _ph = data.get('projection_health', {}) or {}
                if _ph.get('cb_tripped'):
                    fig_mpc.text(0.5, 0.955,
                                 f'⚠ PROJECTION CIRCUIT-BREAKER TRIPPED — {int(_ph.get("cb_skipped_steps", 0))} '
                                 f'step(s) UNPROJECTED (sustained SLSQP slowness). Fan NOT constraint-valid.',
                                 color='white', backgroundcolor='crimson', fontsize=12, fontweight='bold',
                                 ha='center', va='center')
                ax_xy = fig_mpc.add_subplot(1, 2, 1)
                ax_3d = fig_mpc.add_subplot(1, 2, 2, projection='3d')

                # UF-15.2: constraint geometry variables (reused by both panels)
                _gc      = self.geo_config
                _ct      = _gc.get('constraint_types', [])
                _enlarge = (_gc.get('enlarge_constraints') or 0.0) if self.is_tightened else 0.0

                # ── XY panel ─────────────────────────────────────────────────
                for step_i, (cands, _sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                    if step_i % _STRIDE != 0:
                        continue
                    env_step = min(step_i * spr, n_steps - 1)
                    anchor   = c_arr[env_step] if c_arr is not None else real_pos[env_step]
                    for b in range(cands.shape[0]):
                        ax_xy.plot(cands[b, :, 0], cands[b, :, 1],
                                   color='green', linewidth=0.6, alpha=0.7, zorder=4)
                    ax_xy.scatter([anchor[0]], [anchor[1]],
                                  color='black', s=30, zorder=8, linewidths=0)

                if c_arr is not None:
                    ax_xy.plot(c_arr[:, 0], c_arr[:, 1],
                               color='red', linewidth=1.2, zorder=9)
                ax_xy.plot(real_pos[:, 0], real_pos[:, 1],
                           color='black', linewidth=1.2, zorder=7)
                # U11.2: start / end markers
                _ref = c_arr if c_arr is not None else real_pos
                ax_xy.scatter([_ref[0, 0]],  [_ref[0, 1]],
                              color='lime', marker='*', s=180, zorder=12, linewidths=0)
                ax_xy.scatter([_ref[-1, 0]], [_ref[-1, 1]],
                              color='red',  marker='s', s=80,  zorder=12, linewidths=0)
                _lgd = [
                    _Line2D([0],[0], color='green', lw=0.8,
                            label=f'MPC candidates ({n_cands}/step)'),
                    _Line2D([0],[0], color='black', lw=1.2, label='des (commanded)'),
                    _Line2D([0],[0], color='red',   lw=1.2, label='actual (c_pos)'),
                    _Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
                            markersize=7, label='replan decision point'),
                    _Line2D([0],[0], marker='*', color='w', markerfacecolor='lime',
                            markersize=10, label='start'),
                    _Line2D([0],[0], marker='s', color='w', markerfacecolor='red',
                            markersize=7,  label='end'),
                ]
                # Json_Orgnize_C4 (Finding #7): first/last-contact markers, XY panel only —
                # proximity proxy (see contact.note in the stats JSON), not mesh contact.
                _cfs = data.get('contact_first_step', -1)
                _cls = data.get('contact_last_step', -1)
                if _cfs is not None and _cfs >= 0 and _cfs < len(_ref):
                    ax_xy.scatter([_ref[_cfs, 0]], [_ref[_cfs, 1]], marker='*', s=140,
                                  color='blue', zorder=15, linewidths=0)
                    _lgd.append(_Line2D([0],[0], marker='*', color='w', markerfacecolor='blue',
                                         markersize=10, label='first contact'))
                if _cls is not None and _cls >= 0 and _cls < len(_ref):
                    ax_xy.scatter([_ref[_cls, 0]], [_ref[_cls, 1]], marker='X', s=140,
                                  color='purple', zorder=15, linewidths=0)
                    _lgd.append(_Line2D([0],[0], marker='X', color='w', markerfacecolor='purple',
                                         markersize=9, label='last contact'))
                if self.is_tightened and _enlarge > 0:
                    _lgd += [
                        _Line2D([0],[0], color='steelblue', lw=1.5, linestyle='-',
                                label=f'nominal constraint (eval boundary)'),
                        _Line2D([0],[0], color='steelblue', lw=1.5, linestyle='--',
                                label=f'planning constraint (δ={_enlarge:.3f}m inside)'),
                    ]
                ax_xy.legend(handles=_lgd, fontsize=9)
                ax_xy.set_title(f'XY — MPC Decision Points  (every {_STRIDE} replans)',
                                fontsize=12)
                ax_xy.set_xlabel('X (m)', fontsize=11)
                ax_xy.set_ylabel('Y (m)', fontsize=11)
                ax_xy.set_aspect('equal', adjustable='datalim')
                ax_xy.grid(True, alpha=0.3)

                # UF-15.2 / UF-16: constraint geometry overlay — drawn behind trajectories.
                # For tightened variants draw two layers:
                #   (enlarge=0,       dashed=False) → nominal boundary  — solid fill+edge
                #   (enlarge=_enlarge, dashed=True) → planning boundary — dashed edge only
                _c_layers = [(0.0, False)]
                if self.is_tightened and _enlarge > 0:
                    _c_layers.append((_enlarge, True))

                if _gc:
                    import matplotlib.patches as _mpa_uf15
                    for _cl_e, _cl_dash in _c_layers:
                        _cl_ls = '--' if _cl_dash else '-'
                        if 'geo_bounds' in _ct and 'workspace_bounds' in _gc:
                            _wb    = _gc['workspace_bounds']
                            _lb_xy = np.array(_wb['lb'][:2], dtype=float) + _cl_e
                            _ub_xy = np.array(_wb['ub'][:2], dtype=float) - _cl_e
                            ax_xy.add_patch(_mpa_uf15.Rectangle(
                                (_lb_xy[0], _lb_xy[1]), _ub_xy[0]-_lb_xy[0], _ub_xy[1]-_lb_xy[1],
                                lw=1.5, linestyle=_cl_ls, edgecolor='steelblue',
                                facecolor='steelblue' if not _cl_dash else 'none',
                                alpha=0.10 if not _cl_dash else 0.9,
                                zorder=1))
                        if 'halfspace' in _ct and _gc.get('halfspace_constraints'):
                            _wb  = _gc.get('workspace_bounds', {})
                            _xlm = (float(_wb['lb'][0])-0.05, float(_wb['ub'][0])+0.05) if _wb else (0.20, 0.80)
                            _ylm = (float(_wb['lb'][1])-0.05, float(_wb['ub'][1])+0.05) if _wb else (-0.45, 0.45)
                            for _hs in _gc['halfspace_constraints']:
                                _hs_xy_draw(ax_xy, _hs, _cl_e, _xlm, _ylm, dashed=_cl_dash)
                        if 'obstacles' in _ct:
                            for _obs in _gc.get('obstacle_constraints', []):
                                ax_xy.add_patch(_mpa_uf15.Circle(
                                    (float(_obs['center'][0]), float(_obs['center'][1])),
                                    _obs['radius'] + _cl_e,
                                    lw=1.5, linestyle=_cl_ls, edgecolor='tomato',
                                    facecolor='tomato' if not _cl_dash else 'none',
                                    alpha=0.15 if not _cl_dash else 0.9,
                                    zorder=1))
                                if not _cl_dash:
                                    ax_xy.plot(float(_obs['center'][0]), float(_obs['center'][1]),
                                               'r+', ms=6, zorder=2)

                # U_19 (Gen7/Gen6V4): draw the physical push-box footprint on the XY
                # panel so its position relative to the obstacle/constraints is visible
                # (the box may overlap the added obstacle). The aligning push-box is a
                # 0.10×0.10 m square (MuJoCo half-extent 0.05 in robot_push_box.xml);
                # box/target poses carry a yaw angle in degrees (box_space samples
                # [x, y, angle∈[-90,90]]). Drawn as rotated squares: init = solid
                # saddlebrown (filled), final = dashed saddlebrown, target = goldenrod.
                if _ci:
                    import matplotlib.patches as _mpa_box
                    _BOX_HALF = 0.05  # m — aligning push-box half-extent (robot_push_box.xml)
                    def _draw_box_xy(_cx, _cy, _ang_deg, _ec, _ls, _fill):
                        _a = np.radians(_ang_deg)
                        _R = np.array([[np.cos(_a), -np.sin(_a)],
                                       [np.sin(_a),  np.cos(_a)]])
                        _corners = np.array([[-_BOX_HALF, -_BOX_HALF], [_BOX_HALF, -_BOX_HALF],
                                             [_BOX_HALF,  _BOX_HALF], [-_BOX_HALF,  _BOX_HALF]])
                        _pts = _corners @ _R.T + np.array([_cx, _cy])
                        ax_xy.add_patch(_mpa_box.Polygon(
                            _pts, closed=True, lw=1.8, linestyle=_ls, edgecolor=_ec,
                            facecolor=_ec if _fill else 'none',
                            alpha=0.18 if _fill else 0.95, zorder=3))
                    _bi = _ci.get('box_init_xy')
                    if _bi is not None:
                        _draw_box_xy(float(_bi[0]), float(_bi[1]),
                                     float(_ci.get('box_init_angle_deg', 0.0)),
                                     'saddlebrown', '-', True)
                        _lgd.append(_Line2D([0], [0], color='saddlebrown', lw=1.8,
                                            label='push-box (init)'))
                    _bf = _ci.get('final_box_xy')
                    if _bf is not None:
                        _draw_box_xy(float(_bf[0]), float(_bf[1]),
                                     float(_ci.get('final_box_angle_deg', 0.0)),
                                     'saddlebrown', '--', False)
                        _lgd.append(_Line2D([0], [0], color='saddlebrown', lw=1.8,
                                            linestyle='--', label='push-box (final)'))
                    _tt = _ci.get('target_xy')
                    if _tt is not None:
                        _draw_box_xy(float(_tt[0]), float(_tt[1]),
                                     float(_ci.get('target_angle_deg', 0.0)),
                                     'goldenrod', '-', False)
                        _lgd.append(_Line2D([0], [0], color='goldenrod', lw=1.8,
                                            label='target box'))
                    ax_xy.legend(handles=_lgd, fontsize=9)

                # ── 3D XYZ panel ──────────────────────────────────────────────
                for step_i, (cands, _sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                    if step_i % _STRIDE != 0:
                        continue
                    env_step = min(step_i * spr, n_steps - 1)
                    anchor   = c_arr[env_step] if c_arr is not None else real_pos[env_step]
                    for b in range(cands.shape[0]):
                        ax_3d.plot(cands[b, :, 0], cands[b, :, 1], cands[b, :, 2],
                                   color='green', linewidth=0.6, alpha=0.7)
                    ax_3d.scatter([anchor[0]], [anchor[1]], [anchor[2]],
                                  color='black', s=30)

                if c_arr is not None:
                    ax_3d.plot(c_arr[:, 0], c_arr[:, 1], c_arr[:, 2],
                               color='red', linewidth=1.2, label='actual (c_pos)')
                ax_3d.plot(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2],
                           color='black', linewidth=1.2, label='des (commanded)')
                # U11.2: start / end markers
                ax_3d.scatter([_ref[0, 0]],  [_ref[0, 1]],  [_ref[0, 2]],
                              color='lime', marker='*', s=180, zorder=12)
                ax_3d.scatter([_ref[-1, 0]], [_ref[-1, 1]], [_ref[-1, 2]],
                              color='red',  marker='s', s=80,  zorder=12)
                ax_3d.set_title('XYZ — MPC Decision Points (3D)', fontsize=12)
                ax_3d.set_xlabel('X (m)', fontsize=9)
                ax_3d.set_ylabel('Y (m)', fontsize=9)
                ax_3d.set_zlabel('Z (m)', fontsize=9)
                ax_3d.legend(fontsize=9)

                # UF-15.2 / UF-16: workspace box wireframe on 3D panel
                # For tightened variants: solid nominal box + dashed inner planning box
                if _gc and 'geo_bounds' in _ct and 'workspace_bounds' in _gc:
                    for _cl_e, _cl_dash in _c_layers:
                        _wb  = _gc['workspace_bounds']
                        _x0, _y0, _z0 = np.array(_wb['lb'], dtype=float) + _cl_e
                        _x1, _y1, _z1 = np.array(_wb['ub'], dtype=float) - _cl_e
                        _z0 = 0.0  if np.isinf(_z0) else _z0
                        _z1 = 0.50 if np.isinf(_z1) else _z1
                        _3d_ls  = '--' if _cl_dash else '-'
                        _3d_alp = 0.70 if _cl_dash else 0.45
                        for _xs, _ys, _zs in [
                            ([_x0,_x1],[_y0,_y0],[_z0,_z0]),([_x0,_x1],[_y1,_y1],[_z0,_z0]),
                            ([_x0,_x1],[_y0,_y0],[_z1,_z1]),([_x0,_x1],[_y1,_y1],[_z1,_z1]),
                            ([_x0,_x0],[_y0,_y1],[_z0,_z0]),([_x1,_x1],[_y0,_y1],[_z0,_z0]),
                            ([_x0,_x0],[_y0,_y1],[_z1,_z1]),([_x1,_x1],[_y0,_y1],[_z1,_z1]),
                            ([_x0,_x0],[_y0,_y0],[_z0,_z1]),([_x1,_x1],[_y0,_y0],[_z0,_z1]),
                            ([_x0,_x0],[_y1,_y1],[_z0,_z1]),([_x1,_x1],[_y1,_y1],[_z0,_z1]),
                        ]:
                            ax_3d.plot(_xs, _ys, _zs, color='steelblue',
                                       alpha=_3d_alp, lw=1.0, linestyle=_3d_ls)

                # UF-15.2 / UF-16: halfspace boundary plane on 3D panel
                # For tightened variants: solid nominal plane + dashed inner quad edges
                if _gc and 'halfspace' in _ct and _gc.get('halfspace_constraints'):
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3C_hs
                    for _cl_e, _cl_dash in _c_layers:
                        _hs_wb = _gc.get('workspace_bounds', {})
                        if _hs_wb:
                            _hs_lb3 = np.array(_hs_wb['lb'], dtype=float) + _cl_e
                            _hs_ub3 = np.array(_hs_wb['ub'], dtype=float) - _cl_e
                        else:
                            _hs_lb3 = np.array([0.30, -0.35, 0.05])
                            _hs_ub3 = np.array([0.70,  0.35, 0.40])
                        _hs3_zlo = 0.0  if np.isinf(_hs_lb3[2]) else float(_hs_lb3[2])
                        _hs3_zhi = 0.50 if np.isinf(_hs_ub3[2]) else float(_hs_ub3[2])
                        _hs3_xlm = (float(_hs_lb3[0])-0.05, float(_hs_ub3[0])+0.05)
                        _hs3_ylm = (float(_hs_lb3[1])-0.05, float(_hs_ub3[1])+0.05)
                        for _hs3d in _gc['halfspace_constraints']:
                            _h3pt1, _h3pt2, _h3side = _hs3d
                            _h3x1, _h3y1 = float(_h3pt1[0]), float(_h3pt1[1])
                            _h3x2, _h3y2 = float(_h3pt2[0]), float(_h3pt2[1])
                            _h3dx, _h3dy = _h3x2-_h3x1, _h3y2-_h3y1
                            _h3n = np.array([-_h3dy,_h3dx]) if _h3side=='above' else np.array([_h3dy,-_h3dx])
                            _h3nl = float(np.hypot(*_h3n))
                            if _h3nl < 1e-9: continue
                            _h3n /= _h3nl
                            _h3x1 += _cl_e*_h3n[0]; _h3y1 += _cl_e*_h3n[1]
                            _h3x2 += _cl_e*_h3n[0]; _h3y2 += _cl_e*_h3n[1]
                            _h3dx, _h3dy = _h3x2-_h3x1, _h3y2-_h3y1
                            _h3tx = sorted([(_hs3_xlm[0]-_h3x1)/_h3dx, (_hs3_xlm[1]-_h3x1)/_h3dx]) if abs(_h3dx)>1e-9 else [-1e9,1e9]
                            _h3ty = sorted([(_hs3_ylm[0]-_h3y1)/_h3dy, (_hs3_ylm[1]-_h3y1)/_h3dy]) if abs(_h3dy)>1e-9 else [-1e9,1e9]
                            _h3tlo = max(_h3tx[0],_h3ty[0]); _h3thi = min(_h3tx[1],_h3ty[1])
                            if _h3tlo >= _h3thi: continue
                            _h3px = [_h3x1+_h3tlo*_h3dx, _h3x1+_h3thi*_h3dx]
                            _h3py = [_h3y1+_h3tlo*_h3dy, _h3y1+_h3thi*_h3dy]
                            if _cl_dash:
                                _h3v = [[_h3px[0],_h3py[0],_hs3_zlo],[_h3px[1],_h3py[1],_hs3_zlo],
                                        [_h3px[1],_h3py[1],_hs3_zhi],[_h3px[0],_h3py[0],_hs3_zhi]]
                                for _ei in range(4):
                                    _va, _vb = _h3v[_ei], _h3v[(_ei+1)%4]
                                    ax_3d.plot([_va[0],_vb[0]], [_va[1],_vb[1]], [_va[2],_vb[2]],
                                               color='darkorange', lw=1.2, linestyle='--', alpha=0.8)
                            else:
                                ax_3d.add_collection3d(_P3C_hs([[
                                    [_h3px[0],_h3py[0],_hs3_zlo],[_h3px[1],_h3py[1],_hs3_zlo],
                                    [_h3px[1],_h3py[1],_hs3_zhi],[_h3px[0],_h3py[0],_hs3_zhi],
                                ]], alpha=0.25, facecolor='darkorange', edgecolor='darkorange', lw=0.8))

                # UF-15.5 / UF-16: obstacle sphere on 3D panel
                # For tightened variants: solid filled sphere (nominal) + wireframe circles (planning)
                if _gc and 'obstacles' in _ct:
                    _wb_obs = _gc.get('workspace_bounds', {})
                    for _cl_e, _cl_dash in _c_layers:
                        if _wb_obs:
                            _ob_lb = np.array(_wb_obs['lb'], dtype=float) + _cl_e
                            _ob_ub = np.array(_wb_obs['ub'], dtype=float) - _cl_e
                            _ob_z0 = 0.0  if np.isinf(_ob_lb[2]) else float(_ob_lb[2])
                            _ob_z1 = 0.50 if np.isinf(_ob_ub[2]) else float(_ob_ub[2])
                        else:
                            _ob_z0, _ob_z1 = 0.02, 0.50
                    _ob_zmid = (_ob_z0 + _ob_z1) / 2
                    for _obs3d in _gc.get('obstacle_constraints', []):
                        _ocx = float(_obs3d['center'][0])
                        _ocy = float(_obs3d['center'][1])
                        _od  = _obs3d.get('dimensions', ['x', 'y'])
                        _ocz = (float(_obs3d['center'][2]) if len(_obs3d['center']) > 2 and 'z' in _od
                                else _ob_zmid)
                        for _cl_e, _cl_dash in _c_layers:
                            _or = float(_obs3d['radius']) + _cl_e
                            _ou = np.linspace(0, 2*np.pi, 20)
                            _ov = np.linspace(0, np.pi, 12)
                            if _cl_dash:
                                _th = np.linspace(0, 2*np.pi, 60)
                                ax_3d.plot(_ocx+_or*np.cos(_th), _ocy+_or*np.sin(_th),
                                           [_ocz]*60, color='tomato', lw=1.0,
                                           linestyle='--', alpha=0.8)
                                for _mz in [_ocz-_or*0.5, _ocz, _ocz+_or*0.5]:
                                    _rh = np.sqrt(max(0.0, _or**2 - (_mz-_ocz)**2))
                                    if _rh > 0:
                                        ax_3d.plot(_ocx+_rh*np.cos(_th), _ocy+_rh*np.sin(_th),
                                                   [_mz]*60, color='tomato', lw=0.6,
                                                   linestyle='--', alpha=0.5)
                            else:
                                ax_3d.plot_surface(
                                    _ocx + _or*np.outer(np.cos(_ou), np.sin(_ov)),
                                    _ocy + _or*np.outer(np.sin(_ou), np.sin(_ov)),
                                    _ocz + _or*np.outer(np.ones_like(_ou), np.cos(_ov)),
                                    color='tomato', alpha=0.30, linewidth=0)

                fig_mpc.tight_layout()
                _mpc_base = os.path.join(diag_path, f'rollout_{rollout_idx}_mpc_foresight')
                # fig_mpc.savefig(f'{_mpc_base}.png', dpi=200, bbox_inches='tight')
                fig_mpc.savefig(f'{_mpc_base}.svg', bbox_inches='tight')
                plt.close(fig_mpc)

        except Exception as e:
            print(f'[ diag ] Real-time export failed for rollout {rollout_idx}: {e}')

        # C5: incremental crash-safety snapshot every `partial_npz_every` rollouts. Kept
        # outside the plotting try/except above so a plot failure never skips the snapshot
        # (and a snapshot failure is reported distinctly rather than masked). Sidecar file;
        # see _save_partial_npz for why it is not <variant>.npz.
        if (self._partial_npz_every and self.save_path and self.geo_config
                and self.geo_config.get('write_to_file', True)
                and (self.rollout_counter + 1) % self._partial_npz_every == 0):
            try:
                _save_partial_npz(
                    os.path.join(self.save_path, f'{self.variant}.partial.npz'),
                    self, self._seed, self._npz_args)
            except Exception as e:
                print(f'[ diag ] partial NPZ snapshot failed at rollout {rollout_idx}: {e}')

    @torch.no_grad()
    def predict(self, state, goal=None, extra_args=None, if_vision=False):
        """
        D3IL agent.predict() interface.
        Visual:     state = (bp_image_np, inhand_image_np, des_robot_pos_np)
        Non-visual: state = obs_np  — D3IL concatenated obs with robot_pos at [:3]
        """
        cond = None
        if if_vision:
            bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state  # C4: unpack actual robot_pos

            # ── Video capture ──────────────────────────────────────────────
            # bp_np is (C,H,W) float32 in BGR order (fix11: no [::-1] flip, env returns BGR).
            # imageio.mimsave() expects RGB, so convert at capture time — same as expert GIF.
            if self.record_mode != 'none':
                try:
                    bp_vis     = cv2.cvtColor((bp_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    inhand_vis = cv2.cvtColor((inhand_np.copy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    frame = np.concatenate([bp_vis, inhand_vis], axis=1)
                    cv2.putText(frame, f's{self.step_counter}', (5, 18),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
                    self.video_frames.append(frame)
                except Exception:
                    pass

            if self.mental_robot_pos is None:
                self.mental_robot_pos = des_robot_pos_np.copy()

            self.history_real_pos.append(des_robot_pos_np.copy())
            self.curr_rollout_c_pos.append(robot_pos_np.copy())             # Fix 9: actual robot pos
            # Fix 9: physical tracking error = |c_pos - des_c_pos| (replaces trivially-zero mental model error)
            phys_err = float(np.linalg.norm(robot_pos_np[:2] - des_robot_pos_np[:2]))
            self.curr_rollout_tracking_errors.append(phys_err)

            # ── Build 6D obs = [des_c_pos | c_pos] ───────────────────────
            # C4 fix: use actual robot_pos from sim for the c_pos slot.
            # des_robot_pos_np = commanded position; robot_pos_np = actual sim state.
            obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # (6,) [des_c_pos | c_pos]

            if self.obs_normalizer is not None:
                obs_6d_norm = self.obs_normalizer.normalize(
                    obs_6d_np.reshape(1, -1)).astype(np.float32).squeeze(0)
            else:
                obs_6d_norm = obs_6d_np.astype(np.float32)

            bp_t     = torch.from_numpy(bp_np.astype(np.float32)).to(self.device).unsqueeze(0)
            inhand_t = torch.from_numpy(inhand_np.astype(np.float32)).to(self.device).unsqueeze(0)
            obs_t    = torch.from_numpy(obs_6d_norm).to(self.device).unsqueeze(0)  # (1, 6)

            self.bp_image_context.append(bp_t)
            self.inhand_image_context.append(inhand_t)
            self.obs_context.append(obs_t)

            while len(self.bp_image_context) < self.window_size:
                self.bp_image_context.append(bp_t)
                self.inhand_image_context.append(inhand_t)
                self.obs_context.append(obs_t)

            bp_seq     = torch.cat(list(self.bp_image_context), dim=0)      # (W, C, H, W)
            inhand_seq = torch.cat(list(self.inhand_image_context), dim=0)  # (W, C, H, W)
            obs_seq    = torch.cat(list(self.obs_context), dim=0)           # (W, 6)

            bp_batch     = bp_seq.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)
            inhand_batch = inhand_seq.unsqueeze(0).repeat(self.batch_size, 1, 1, 1, 1)
            obs_batch    = obs_seq.unsqueeze(0).repeat(self.batch_size, 1, 1)

            cond = {0: (bp_batch, inhand_batch, obs_batch)}

        else:
            # Non-visual path (UF-17): aligning_sim prepends des_c_pos to env obs →
            # state is 20D = [des_c_pos(3) | c_pos(3) | box(3) | box_q(4) | tgt(3) | tgt_q(4)]
            obs_20d_np = np.asarray(state, dtype=np.float64)   # (20,)
            des_robot_pos_np = obs_20d_np[:3].copy()
            robot_pos_np     = obs_20d_np[3:6].copy()

            if self.mental_robot_pos is None:
                self.mental_robot_pos = des_robot_pos_np.copy()

            self.history_real_pos.append(des_robot_pos_np.copy())
            self.curr_rollout_c_pos.append(robot_pos_np.copy())
            phys_err = float(np.linalg.norm(robot_pos_np[:2] - des_robot_pos_np[:2]))
            self.curr_rollout_tracking_errors.append(phys_err)

            if self.obs_normalizer is not None:
                obs_norm = self.obs_normalizer.normalize(
                    obs_20d_np.reshape(1, -1)).astype(np.float32).squeeze(0)  # (20,)
            else:
                obs_norm = obs_20d_np.astype(np.float32)
            obs_t = torch.from_numpy(obs_norm).to(self.device).unsqueeze(0)   # (1, 20)
            self.obs_context.append(obs_t)
            while len(self.obs_context) < self.obs_seq_len:
                self.obs_context.append(obs_t)
            obs_anchor = obs_t.repeat(self.batch_size, 1)   # (B, 20)
            cond = {0: obs_anchor}

        # ── Plan (or execute from cached action chunk) ─────────────────────
        if self.action_counter == self.action_seq_size:
            t_replan = time.time()   # Fix 12: time only the replan call, not cached fetches
            self.action_counter = 0
            self.model.eval()

            if self.projector is not None:
                trajectory, infos = self.model(cond, projector=self.projector)
                # Fix_15.3: the projector marks each call it SKIPS while its sustained-slowness
                # circuit breaker is OPEN (unprojected step). Count them for artifact marking.
                if getattr(self.projector, 'last_proj_skipped', False):
                    self.curr_rollout_cb_skipped_steps += 1
            else:
                trajectory, infos = self.model(cond)

            traj_np = trajectory.detach().cpu().numpy()   # (B, H, 9)
            which   = 0
            selection_method = 'default (first)'

            if self.batch_size > 1:
                if (self.trajectory_selection == 'temporal_consistency'
                        and self.prev_observations is not None):
                    diffs = traj_np - np.expand_dims(self.prev_observations, 0)
                    which = int(np.argsort(np.linalg.norm(diffs, axis=(1, 2)))[0])
                    selection_method = 'temporal_consistency'
                elif (self.trajectory_selection == 'minimum_projection_cost'
                      and self.projector is not None):
                    # Try precomputed costs from post-processing projection first
                    has_precomputed = False
                    if (infos is not None and 'projection_costs' in infos
                            and len(infos['projection_costs']) > 0):
                        costs_total = np.zeros(self.batch_size)
                        for _, cost in infos['projection_costs'].items():
                            costs_total += cost
                        if len(costs_total) == self.batch_size:
                            which = int(np.argmin(costs_total))
                            selection_method = 'minimum_projection_cost (precomputed)'
                            has_precomputed = True
                    if not has_precomputed:
                        _, projection_costs = self.projector.project(trajectory)
                        which = int(np.argmin(projection_costs))
                        selection_method = 'minimum_projection_cost (calculated)'
                else:
                    which = 0   # 'random' (DPCC default) = always index 0, deterministic
                    selection_method = 'random (index 0, DPCC semantics)'

            # c_pos at traj dims 6-8, obs dims 3-5 — true for both 9D (visual) and 23D (non-visual).
            cpos_norm = traj_np[:, :, 6:9]   # (B, H, 3) normalized predicted actual positions
            if self.obs_normalizer is not None:
                B_f, H_f = cpos_norm.shape[:2]
                obs_dim = self.obs_normalizer.mins.shape[0]   # 6 (visual) or 20 (non-visual)
                dummy = np.zeros((B_f * H_f, obs_dim), dtype=np.float32)
                dummy[:, 3:6] = cpos_norm.reshape(-1, 3).astype(np.float32)
                obs_un = self.obs_normalizer.unnormalize(dummy)
                self.curr_rollout_all_candidates.append(obs_un[:, 3:6].reshape(B_f, H_f, 3).copy())
            else:
                self.curr_rollout_all_candidates.append(cpos_norm.copy())
            self.curr_rollout_selected_idx.append(int(which))

            # UF-16.3: planning violation rate on post-projection candidates
            if self.geo_config and self.curr_rollout_all_candidates:
                _pv = _check_planned_violations(
                    self.curr_rollout_all_candidates[-1],
                    self.geo_config,
                    (self.geo_config.get('enlarge_constraints') or 0.0) if self.is_tightened else 0.0)
                self._plan_post_viol_rates.append(_pv)

            self.prev_observations = traj_np[which].copy()

            # action dims = indices 0:3
            action_traj = trajectory[[which], :, :3]   # (1, H, 3)

            if self.act_normalizer is not None:
                act_np = action_traj.detach().cpu().numpy()
                B, H, D = act_np.shape
                act_np = self.act_normalizer.unnormalize(act_np.reshape(-1, D)).reshape(B, H, D)
                action_traj = torch.from_numpy(act_np).to(self.device)

            # One-time diagnostic on first replan of first rollout.
            # Prints pre/post-denorm action magnitudes so we can immediately spot:
            #   - actions stuck near zero  → denormalization failed
            #   - actions in [-1, 1] at "denorm" stage → act_normalizer was None
            #   - horizon range outside [-1, 1] at normalized stage → model diverged
            # Also writes a dedicated diag_first_replan.txt to save_path for easy
            # grep / cross-run comparison (not buried in the full eval log).
            if self.rollout_counter == 0 and self.step_counter == 0:
                norm_a0   = trajectory[[which], 0, :3].detach().cpu().numpy().squeeze()
                denorm_a0 = action_traj[0, 0].detach().cpu().numpy()
                full_norm = trajectory[which, :, :3].detach().cpu().numpy()
                diag_lines = [
                    f'[ DIAG first-replan ] normalized   a0 = {np.round(norm_a0, 4)}'
                    f'  |mag| = {np.linalg.norm(norm_a0):.4f}',
                    f'[ DIAG first-replan ] denormalized a0 = {np.round(denorm_a0, 5)}'
                    f'  |mag| = {np.linalg.norm(denorm_a0):.6f} m',
                    f'[ DIAG first-replan ] horizon act (normalized) range: '
                    f'[{full_norm.min():.4f}, {full_norm.max():.4f}]',
                    f'[ DIAG first-replan ] per-step normalized acts (H={full_norm.shape[0]}):',
                ]
                for h_i, row in enumerate(full_norm):
                    diag_lines.append(f'  step {h_i:2d}: {np.round(row, 4)}')
                # obs health (Issue 4) — branch on if_vision because the visual
                # and non-visual paths name their obs tensors differently.
                if if_vision:
                    _diag_obs_raw  = obs_6d_np      # (6,) [des_c_pos | c_pos]
                    _diag_obs_norm = obs_6d_norm    # (6,)
                else:
                    _diag_obs_raw  = obs_20d_np     # (20,)
                    _diag_obs_norm = obs_norm       # (20,)
                diag_lines += [
                    f'[ DIAG obs ] des_c_pos={np.round(_diag_obs_raw[:3], 4)}  '
                    f'c_pos={np.round(_diag_obs_raw[3:6], 4)}',
                    f'[ DIAG obs ] obs_norm (dim={_diag_obs_norm.shape[0]})='
                    f'{np.round(_diag_obs_norm, 4)}',
                ]
                # image health (Issue 5) — visual only
                if if_vision:
                    bp_std = float(np.std(bp_np))
                    ih_std = float(np.std(inhand_np))
                    diag_lines += [
                        f'[ DIAG img ] bp_image   std={bp_std:.4f}  shape={bp_np.shape}',
                        f'[ DIAG img ] inhand_img std={ih_std:.4f}  shape={inhand_np.shape}',
                    ]
                    if bp_std < 0.01:
                        diag_lines.append('[ DIAG img ] WARNING: bp_image near-black — camera may not be rendering')
                    if ih_std < 0.01:
                        diag_lines.append('[ DIAG img ] WARNING: inhand_image near-black — camera may not be rendering')
                for line in diag_lines:
                    print(line)
                if self.save_path is not None:
                    diag_file = os.path.join(self.save_path, 'diag_first_replan.txt')
                    with open(diag_file, 'w') as _df:
                        _df.write('\n'.join(diag_lines) + '\n')
                    print(f'[ DIAG ] saved → {diag_file}')

            # Periodic DIAG every 50 replans (Issue 2)
            self._replan_count += 1
            if self._replan_count % 50 == 0:
                _pa0 = trajectory[[which], 0, :3].detach().cpu().numpy().squeeze()
                _da0 = action_traj[0, 0].detach().cpu().numpy()
                _dir = _pa0 / (np.linalg.norm(_pa0) + 1e-9)
                print(f'[ DIAG replan={self._replan_count} step={self.step_counter} ] '
                      f'norm|a0|={np.linalg.norm(_pa0):.3f}  '
                      f'denorm|a0|={np.linalg.norm(_da0):.2e} m  '
                      f'dir={np.round(_dir, 3)}')

            self.curr_action_seq = action_traj[:, :self.action_seq_size, :]
            self.history_full_plans.append(action_traj[0].detach().cpu().numpy())
            _rt_replan_ms = (time.time() - t_replan) * 1e3   # REAL_TIME_RECORDING_UPDATE — per-replan FM+projection wall-time
            self.curr_rollout_time += _rt_replan_ms / 1e3   # Fix 12: accumulate per-replan time
            # REAL_TIME_RECORDING_UPDATE — record this replan's timing (proj bundled in model call).
            if getattr(self, 'rt_rec', None) is not None:
                self.rt_rec.step(t=self.step_counter / RT_CONTROL_HZ, total_ms=_rt_replan_ms,
                                 obs=np.concatenate([des_robot_pos_np, robot_pos_np]),
                                 pos=robot_pos_np[:2], proj_active=(self.projector is not None),
                                 track_err=phys_err, step_idx=self._replan_count)

        next_action    = self.curr_action_seq[:, self.action_counter, :]
        next_action_np = next_action.detach().cpu().numpy().squeeze(0)   # (3,)
        self.curr_rollout_act_magnitudes.append(float(np.linalg.norm(next_action_np)))

        if self.max_action_delta is not None:
            raw_mag = np.linalg.norm(next_action_np)
            if raw_mag > self.max_action_delta:
                next_action_np = next_action_np * (self.max_action_delta / raw_mag)
                self.curr_rollout_clamp_events.append((self.step_counter, float(raw_mag)))
                if len(self.curr_rollout_clamp_events) <= 5 or self.step_counter % 50 == 0:
                    print(f'[ CLAMP step={self.step_counter} ] '
                          f'raw|a|={raw_mag:.4f} m → clamped to {self.max_action_delta} m')

        self.mental_robot_pos += next_action_np

        self.history_desired_actions.append(next_action_np.copy())
        self.last_predicted_pos = self.mental_robot_pos.copy()

        self.action_counter += 1
        self.step_counter   += 1
        return next_action_np.reshape(1, -1)   # (1, 3) — sim expects (1, act_dim)

# ── Model loading ─────────────────────────────────────────────────────────────

def load_diffusion_with_override(*loadpath, target_class=None, epoch='latest', device='cuda:0', override_args=None):
    lp = os.path.join(*loadpath)
    print(f'\n[ eval loading ] Loading from {lp}\n')
    dataset_config   = utils.load_config(*loadpath, 'dataset_config.pkl')
    model_config     = utils.load_config(*loadpath, 'model_config.pkl')
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config   = utils.load_config(*loadpath, 'trainer_config.pkl')
    trainer_config._dict['results_folder'] = lp

    if target_class is not None:
        diffusion_config._class = utils.config.import_class(target_class)

    # CONFIG-OVERRIDES-PKL (2026-07-13): the CURRENT config plan block is authoritative over
    # the pickled train-time diffusion config. Every pickled kwarg that also exists in the
    # parsed plan args is applied BEFORE instantiation; a console warning is printed for
    # every value that changes. Generalizes Fix 5 (flow_steps_v3) and D1 (clip_denoised),
    # which previously stamped these one attribute at a time AFTER instantiation.
    # See logs_in_develop/config_override_pkl/CHANGELOG_config_overrides_pkl.md
    if override_args is not None:
        for _k in list(diffusion_config._dict.keys()):
            if hasattr(override_args, _k):
                _new, _old = getattr(override_args, _k), diffusion_config._dict[_k]
                try:
                    _changed = bool(_new != _old)
                except Exception:
                    _changed = True
                if _changed:
                    print(f"[WARNING] config-overrides-pkl: '{_k}': {_old!r} (pkl) -> {_new!r} (config)")
                    diffusion_config._dict[_k] = _new

    dataset   = dataset_config()
    model     = model_config()
    diffusion = diffusion_config(model).to(device)
    trainer   = trainer_config(diffusion_model=diffusion, dataset=dataset)
    if epoch == 'latest':
        epoch = utils.get_latest_epoch(loadpath)
    trainer.load(epoch)
    return utils.DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, None)

# ── Parser & Main ─────────────────────────────────────────────────────────────

class Parser(utils.Parser):
    dataset: str = 'aligning-d3il-visual'
    config: str  = 'config.aligning-d3il-visual'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--record', type=str,
                        choices=['none', 'video', 'gif', 'all'], default='all')
    parser.add_argument('--eval-on-train', action='store_true')
    args_cli, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    with open('config/visual_aligning_eval.yaml', 'r') as f:
        config = yaml.safe_load(f)

    seeds               = [args_cli.seed] if args_cli.seed else config['seeds']
    projection_variants = config.get('projection_variants', ['diffuser'])
    n_contexts          = config.get('n_contexts', 30)
    n_trajectories      = config.get('n_trajectories_per_context', 1)

    for _seed_i, seed in enumerate(seeds):
        # Fix_11: seed X/N — the outermost breadcrumb level (mirrors UAV's Fix_11).
        print(f'\n=== Evaluating seed {_seed_i + 1}/{len(seeds)} (seed={seed}) ===')
        args = Parser().parse_args(experiment='plan_fm_visual_aligning', seed=seed)

        diffusion_model = None
        if not args_cli.aggregate_only:
            exp = load_diffusion_with_override(
                args.loadbase, args.dataset, args.diffusion_loadpath, str(args.seed),
                target_class=args.diffusion, epoch=args.diffusion_epoch,
                device=args.device, override_args=args,
            )
            diffusion_model = exp.diffusion
            # Original DPCC always trains/evaluates with clip_denoised=False — the cosine schedule
            # amplifies x_0 prediction by ~9.4× at early timesteps, so clipping to ±1 corrupts the
            # denoising chain. FM ODE does not clamp at all (no cosine schedule), so False is correct here too.
            # D1: config-driven (was hardcoded False). Default=False matches reference DPCC.
            # Set clip_denoised=True in plan_fm_visual_aligning config only to ablate.
            _clip_denoised = getattr(args, 'clip_denoised', False)
            diffusion_model.clip_denoised = _clip_denoised
            print(f'[ eval ] clip_denoised set → {_clip_denoised} (config-driven; original DPCC default: False)')

            # Fix 5: override flow_steps_v3 from args so Slurm --flow_steps_v3 N
            # actually changes ODE integration steps, not just the output directory name.
            # Without this, the checkpoint's baked-in value (e.g. 100) is always used.
            _args_flow = getattr(args, 'flow_steps_v3', None)
            if _args_flow is not None:
                diffusion_model.flow_steps_v3 = int(_args_flow)
                diffusion_model.ode_inference_steps_v3 = int(_args_flow)

            _model_n_ts  = getattr(diffusion_model, 'n_timesteps', '?')
            _config_n_ts = getattr(args, 'n_diffusion_steps', '?')
            _flow_steps  = getattr(diffusion_model, 'flow_steps_v3', '?')
            print(f'[ eval ] Model n_timesteps = {_model_n_ts}  '
                  f'(config n_diffusion_steps = {_config_n_ts})')
            print(f'[ eval ] FM flow_steps_v3 = {_flow_steps}  '
                  f'(Euler ODE integration steps 0→1)'
                  f'{" [overridden from args]" if _args_flow is not None else " [checkpoint default]"}')
            if isinstance(_model_n_ts, int) and isinstance(_config_n_ts, int):
                if _model_n_ts != _config_n_ts:
                    print(f'[ eval ] WARNING: n_timesteps mismatch — '
                          f'checkpoint trained with {_model_n_ts} steps, '
                          f'config says {_config_n_ts}. '
                          f'Denoising chain will use checkpoint value ({_model_n_ts}).')

        # Expert reference videos generated ONCE before the variant loop.
        # Running env.start()/env.close() inside the loop leaves residual MuJoCo
        # global state (body counter, OpenGL context) that corrupts the first
        # variant's scene init and changes bp_image pixel statistics.
        # gc.collect + cuda.empty_cache after close forces cleanup before any
        # variant env is created.  (AUDIT-FIX-1 — KEY_fix_6/BUG_REPORT.md)
        _base_results = (f'{args.savepath}/results_train_set'
                         if args_cli.eval_on_train else f'{args.savepath}/results')
        os.makedirs(_base_results, exist_ok=True)
        generate_expert_reference(_base_results, n_rollouts=3)
        gc.collect()
        torch.cuda.empty_cache()

        # FIX-7: Reset MuJoCo global robot body counter after expert gen.
        # Robot_Push_Env.__init__() creates an MjRobot which increments
        # MjRobot.GLOBAL_MJ_ROBOT_COUNTER (0→1). env.close() does NOT decrement it.
        # Without this reset, the first variant's Robot_Push_Env gets robot_id=1
        # (body prefix "rb1") instead of robot_id=0 ("rb0"), compiling a different
        # MuJoCo XML where camera bodies are named "rb1_*" instead of "rb0_*".
        # The mismatch changes what the camera renders → bp_image std 0.1978 instead
        # of clean 0.2093 → model receives wrong visual input → wrong trajectory.
        try:
            from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot as _MjRobot
            _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
            print('[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)')
        except Exception as _e:
            print(f'[ expert ] WARNING: MjRobot counter reset failed: {_e}')
        # FIX-7.2: Clear the process-global render context cache so the variant's
        # cameras create fresh RenderContextOffscreen objects bound to variant_model
        # and variant_data. Without this, __RENDER_CTX_MAP in mj_render_singleton.py
        # holds stale contexts from expert gen (bound to expert_model/expert_data),
        # causing all variant renders to show the expert gen's robot pose instead of
        # the variant's → bp_image std 0.1978 (wrong) instead of 0.2093 (correct).
        try:
            from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_render_singleton import (
                reset_singleton as _reset_render_singleton,
            )
            _reset_render_singleton()
            print('[ expert ] Render singleton cache cleared (FIX-7.2)')
        except Exception as _e:
            print(f'[ expert ] WARNING: Render singleton reset failed: {_e}')
        # Also delete stale panda_tmp_rb*.xml left by expert gen to suppress noisy
        # mju_openResource warnings on subsequent env inits.
        import glob as _glob
        _mj_dir = os.path.join(
            os.environ.get('D3IL_DIR', 'd3il/environments/d3il'), 'models/mj/robot')
        for _stale in _glob.glob(os.path.join(_mj_dir, 'panda_tmp_rb*.xml')):
            try:
                os.remove(_stale)
            except OSError:
                pass

        # ── Geo constraint outer loop (UF-14) ────────────────────────────────
        # Build flat (geo_name, geo_config, base_variant) product so the inner
        # loop body needs no indentation change.
        _geo_specs = config.get('geo_constraint_variants', [
            {'name': 'combined_2',
             'constraint_types': config.get('constraint_types', ['geo_bounds', 'dynamics']),
             'workspace_bounds': {'lb': [0.30, -0.35, 0.05], 'ub': [0.70, 0.35, 0.40]}}
        ])
        # enlarge_constraints: None when yaml sets null → no tightened twin generated
        _enlarge = config.get('enlarge_constraints')
        # active_geo_variants: name list in yaml selects which entries to run — null = all
        _active_names = config.get('active_geo_variants')
        if _active_names is not None:
            _active_set = set(_active_names)
            _geo_specs  = [gs for gs in _geo_specs if gs['name'] in _active_set]
            print(f'\n[ geo ] active_geo_variants: {[gs["name"] for gs in _geo_specs]}')
        _run_items = []
        for _gs in _geo_specs:
            _gc = dict(config)
            _gc['constraint_types'] = _gs['constraint_types']
            if 'workspace_bounds'      in _gs: _gc['workspace_bounds']      = _gs['workspace_bounds']
            if 'obstacle_constraints'  in _gs: _gc['obstacle_constraints']  = _gs['obstacle_constraints']
            if 'halfspace_constraints' in _gs: _gc['halfspace_constraints'] = _gs['halfspace_constraints']
            _has_geo = any(t in _gc['constraint_types'] for t in ('geo_bounds', 'halfspace', 'obstacles'))
            for _v in projection_variants:
                _run_items.append((_gs['name'], _gc, _v, False))
            # auto-generate tightened twin for entries with bounds/obstacles
            if _enlarge is not None and _has_geo:
                for _v in projection_variants:
                    _run_items.append((_gs['name'] + '-tightened', _gc, _v, True))

        _n_run_items = len(_run_items)
        for _item_i, (geo_name, geo_config, geo_variant, is_tightened) in enumerate(_run_items):
            # Fix_11: this is the line to grep for after a killed/timed-out job — the LAST
            # one printed is the (geo, variant) combo that was running when it got cut.
            print(f'[ eval ] >>> item {_item_i + 1}/{_n_run_items}: geo={geo_name}  '
                  f'variant={geo_variant}  tightened={is_tightened}')
            if geo_variant == projection_variants[0]:
                print(f'\n[ geo ] ── Constraint variant: {geo_name}  '
                      f'types={geo_config["constraint_types"]} ──')
                _results_root = 'results_train_set' if args_cli.eval_on_train else 'results'
                _geo_fig_dir  = os.path.join(args.savepath, _results_root, geo_name)
                os.makedirs(_geo_fig_dir, exist_ok=True)
                plot_geo_constraints(geo_name, geo_config, _geo_fig_dir, is_tightened)
            variant = geo_variant
            if args_cli.eval_on_train:
                variant   = f'{variant}_train_set'
                save_path = f'{args.savepath}/results_train_set/{geo_name}/{variant}'
            else:
                save_path = f'{args.savepath}/results/{geo_name}/{variant}'
            os.makedirs(save_path, exist_ok=True)

            if args_cli.aggregate_only:
                continue

            log_f = open(os.path.join(save_path, f'eval_{variant}.log'), 'w')
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = Tee(sys.stdout, log_f)
            sys.stderr = Tee(sys.stderr, log_f)

            try:
                # ── Load normalizers ─────────────────────────────────────────
                model_dir     = os.path.join(args.loadbase, args.dataset,
                                             args.diffusion_loadpath, str(args.seed))
                obs_norm_path = os.path.join(model_dir, 'obs_normalizer.pkl')
                act_norm_path = os.path.join(model_dir, 'act_normalizer.pkl')
                if not os.path.exists(obs_norm_path) or not os.path.exists(act_norm_path):
                    raise FileNotFoundError(
                        f'[ eval ] FATAL: normalizer pkl missing in {model_dir}\n'
                        f'  Expected: obs_normalizer.pkl + act_normalizer.pkl\n'
                        f'  Without them, sampled actions stay in [-1, 1] space and\n'
                        f'  produce wrong robot commands. Re-run training to regenerate.'
                    )
                with open(obs_norm_path, 'rb') as f: obs_normalizer = pickle.load(f)
                with open(act_norm_path, 'rb') as f: act_normalizer = pickle.load(f)
                print(f'[ eval ] Loaded normalizers from {model_dir}')
                # Sanity-check: near-zero range in any dim means zero-padded episodes
                # corrupted the scaler at training time — actions would denorm incorrectly.
                act_range = act_normalizer.maxs - act_normalizer.mins
                if np.any(act_range < 1e-4):
                    print(f'[ eval ] WARNING: act_normalizer near-zero range in dims '
                          f'{np.where(act_range < 1e-4)[0].tolist()} — '
                          f'possible zero-pad scaler corruption at train time')
                print(f'[ eval ] obs_normalizer  mins={np.round(obs_normalizer.mins, 4)}  '
                      f'maxs={np.round(obs_normalizer.maxs, 4)}')
                print(f'[ eval ] act_normalizer  mins={np.round(act_normalizer.mins, 4)}  '
                      f'maxs={np.round(act_normalizer.maxs, 4)}')

                # ── Setup DPCC projector ─────────────────────────────────────
                projector = None
                # FIX-18: derive _traj_dim from the SAVED normalizer dimensionalities
                # (action + observation) instead of args.if_vision. The flag can be
                # flipped by UF-13 record-mode auto-enable, but the checkpoint's
                # normalizers are immutable ground truth: act(3)+obs(6)=9 visual,
                # act(3)+obs(20)=23 non-visual. Decouples projector dim from CLI args.
                _act_dim_norm = act_normalizer.mins.shape[0]
                _obs_dim_norm = obs_normalizer.mins.shape[0]
                _traj_dim = _act_dim_norm + _obs_dim_norm
                if _traj_dim not in (9, 23):
                    print(f'[ eval ] WARNING: unexpected _traj_dim={_traj_dim} '
                          f'(act={_act_dim_norm}, obs={_obs_dim_norm}); '
                          f'expected 9 (visual) or 23 (non-visual)')
                if 'diffuser' not in variant and obs_normalizer is not None:
                    projector = setup_dpcc_projector(
                        args, geo_config, obs_normalizer, act_normalizer, variant, is_tightened,
                        trajectory_dim=_traj_dim)
                    print(f'[ eval ] DPCC projector active for variant {variant!r} '
                          f'(trajectory_dim={_traj_dim})')

                # Trajectory selection — exact DPCC eval.py logic (projection_eval.yaml dpcc-r/c/t).
                # 'random' = always index 0 (deterministic); same as DPCC Policy.__call__ semantics.
                trajectory_selection = 'random'
                if 'dpcc-t' in variant:
                    trajectory_selection = 'temporal_consistency'
                if 'dpcc-c' in variant:
                    trajectory_selection = 'minimum_projection_cost'

                # Fix 8: diffuser runs single sample (no projection, no candidate diversity).
                # All projected variants use args.mpc_batch_size from plan config (MPC candidate pool).
                if 'diffuser' in variant:
                    batch_size = 1
                else:
                    batch_size = getattr(args, 'mpc_batch_size', 4)

                agent = VisualAgentWrapper(
                    diffusion_model=diffusion_model,
                    device=args.device,
                    window_size=getattr(args, 'window_size', 1),
                    obs_seq_len=getattr(args, 'obs_seq_len', 1),
                    action_seq_size=getattr(args, 'action_seq_size', 1),
                    save_path=save_path,
                    record_mode=args_cli.record,
                    obs_normalizer=obs_normalizer,
                    act_normalizer=act_normalizer,
                    batch_size=batch_size,
                    projector=projector,
                    trajectory_selection=trajectory_selection,
                    eval_on_train=args_cli.eval_on_train,
                    variant=variant,
                    max_action_delta=geo_config.get('max_action_delta', None),
                    mpc_foresight_stride=geo_config.get('mpc_foresight_stride', 6),
                    geo_config=geo_config,
                    is_tightened=is_tightened,
                    total_rollouts=n_contexts * n_trajectories,   # Fix_11
                    seed=seed,                                    # C5: partial-npz sidecar
                    npz_args=vars(args),                          # C5
                    partial_npz_every=geo_config.get('partial_npz_every', 5),  # C5
                )

                _if_vision_config = getattr(args, 'if_vision', True)
                if_vision = _if_vision_config
                # FIX-18-followup: see same comment in eval_visual_aligning_dpcc.py.
                # Guard UF-13 flip on the saved normalizer dim — only auto-enable
                # visual mode when the checkpoint actually has an image encoder.
                _ckpt_is_visual = (obs_normalizer is not None
                                   and obs_normalizer.mins.shape[0] == 6)
                if not if_vision and args_cli.record != 'none':
                    if _ckpt_is_visual:
                        if_vision = True
                        print('[ eval ] WARNING: config if_vision=False but record_mode is active → '
                              'auto-enabling visual mode so GIFs/videos are captured (UF-13).')
                    else:
                        print('[ eval ] NOTE: record_mode is active but checkpoint is non-visual '
                              f'(obs_normalizer dim = {obs_normalizer.mins.shape[0]}). '
                              'Cannot auto-enable visual mode (this model has no image encoder); '
                              'proceeding with non-visual rollouts. GIFs/videos WILL be captured '
                              'via Aligning_Sim non-visual hook → agent.capture_frame().')

                sim = Aligning_Sim(
                    seed=seed, device=args.device,
                    render=False, n_cores=1,
                    n_contexts=n_contexts,
                    n_trajectories_per_context=n_trajectories,
                    if_vision=if_vision,
                    eval_on_train=args_cli.eval_on_train,
                    max_episode_length=getattr(args, 'max_episode_length', 400),
                )

                # aligning_sim.test_agent() calls wandb.log() unconditionally at the end;
                # initialize in disabled mode so it doesn't crash when no wandb run is active.
                if _wandb is not None:
                    _wandb.init(mode='disabled')

                t0 = time.time()
                success_rate, mode_encoding, successes, mean_dist = sim.test_agent(agent)
                elapsed = time.time() - t0

                # ── Metrics ──────────────────────────────────────────────────
                n_modes    = 2
                mode_probs = torch.zeros([n_contexts, n_modes])
                for c in range(n_contexts):
                    mode_probs[c] = torch.tensor([
                        (mode_encoding[c] == 0).sum().item() / n_trajectories,
                        (mode_encoding[c] == 1).sum().item() / n_trajectories,
                    ])
                m_norm  = mode_probs / (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
                entropy = -(m_norm * torch.log(m_norm + 1e-12) /
                            torch.log(torch.tensor(float(n_modes)))).sum(1).mean().item()

                # C5: DESIRED-only per-rollout path list for the legacy PNG grid below.
                # (The npz payload — incl. the 6-D obs_all and the full MPC fan — is built
                # separately by _collect_per_rollout_arrays so the grid's plotting stays 3-D.)
                obs_des_all = []
                for r in range(agent.rollout_counter + 1):
                    d = agent.master_rollout_history.get(f'rollout_{r}')
                    if d:
                        obs_des_all.append(d['real_robot_pos'])

                # ── NPZ save (C5: single source of raw truth; UAV-style schema) ──────
                # _collect_per_rollout_arrays supplies every per-rollout array (raw + metrics),
                # incl. the widened 6-D obs_all, the full fan (sampled_trajectories_all) and
                # selected_idx_all. Only the aggregate scalars that need Aligning_Sim's return
                # (success_rate/entropy/mode_encoding/n_success/mean_distance) are added here.
                if geo_config.get('write_to_file', True):
                    _npz_payload = _collect_per_rollout_arrays(agent)
                    np.savez(
                        f'{save_path}/{variant}.npz',
                        success_rate=success_rate, entropy=entropy,
                        mode_encoding=mode_encoding.numpy(),
                        elapsed_seconds=elapsed, seed=seed,
                        n_success=successes.flatten().numpy(),
                        success_strict=successes.flatten().numpy(),   # schema-consistent name
                        mean_distance=mean_dist.flatten().numpy(),
                        complete=True,                                # C5: full/authoritative run
                        args=vars(args),
                        **_npz_payload,
                    )
                    # C5: variant finished → drop the now-redundant crash-safety sidecar.
                    _partial = os.path.join(save_path, f'{variant}.partial.npz')
                    if os.path.exists(_partial):
                        try:
                            os.remove(_partial)
                        except OSError:
                            pass

                # Fix_15.3: drop a greppable sentinel + warn when the projection circuit breaker
                # tripped on any rollout of this variant — those rollouts ran (partly) UNPROJECTED
                # (sustained SLSQP slowness, projection.py Fix_15.2), so their constraint metrics are
                # NOT valid. Artifacts are still written (partial run kept), just clearly marked.
                _cb_skips = getattr(agent, 'history_cb_skipped_steps', [])
                _cb_tripped_idx = [i for i, s in enumerate(_cb_skips) if s > 0]
                if _cb_tripped_idx:
                    _tot_skipped = int(sum(_cb_skips))
                    with open(os.path.join(save_path, 'PROJECTION_CB_TRIPPED.txt'), 'w') as _f:
                        _f.write(f"PROJECTION CIRCUIT-BREAKER TRIPPED — aligning variant={variant}\n")
                        _f.write(f"tripped_rollouts={_cb_tripped_idx}  "
                                 f"({len(_cb_tripped_idx)}/{len(_cb_skips)})\n")
                        _f.write(f"total_skipped_steps={_tot_skipped}\n")
                        _f.write("Cause: sustained SLSQP slowness (projection.py Fix_15.2 breaker OPENED).\n")
                        _f.write("These rollouts ran (partly) UNPROJECTED — constraint metrics are NOT\n")
                        _f.write("valid; treat this variant as 'projection broken for this geometry'.\n")
                    print(f'[ eval ] variant={variant}: ⚠ PROJECTION CIRCUIT-BREAKER TRIPPED on '
                          f'{len(_cb_tripped_idx)}/{len(_cb_skips)} rollouts ({_tot_skipped} steps '
                          f'skipped) — results marked UNPROJECTED. See PROJECTION_CB_TRIPPED.txt.',
                          flush=True)

                # ── Legacy PNG rollout grid (mirrors ddpm_encdec) ────────────
                print(f'[ eval ] Generating PNG rollout grid for {variant}...')
                n_plot = min(len(obs_des_all), 5)
                if n_plot > 0:
                    fig, axes = plt.subplots(n_plot, 6, figsize=(30, 5 * n_plot),
                                             squeeze=False)
                    fig.suptitle(f'Visual-DPCC — {variant} (Seed {seed})')
                    if _cb_tripped_idx:   # Fix_15.3
                        fig.text(0.5, 0.965,
                                 f'⚠ PROJECTION CIRCUIT-BREAKER TRIPPED on {len(_cb_tripped_idx)}/'
                                 f'{len(_cb_skips)} rollouts — those paths are UNPROJECTED '
                                 f'(sustained SLSQP slowness)',
                                 color='white', backgroundcolor='crimson', fontsize=11,
                                 fontweight='bold', ha='center', va='center')

                    for i in range(n_plot):
                        obs_traj   = obs_des_all[i]   # (T, 3) des_robot_pos
                        rollout_data = agent.master_rollout_history.get(f'rollout_{i}', {})
                        c_pos_hist   = rollout_data.get('c_pos_history', None)  # Fix 9: actual positions

                        axes[i, 0].plot(obs_traj[:, 0], 'k-', label='des')
                        if c_pos_hist is not None and len(c_pos_hist):
                            axes[i, 0].plot(c_pos_hist[:, 0], 'r--', label='actual')
                        axes[i, 0].set_title('X — des (black) vs actual (red)')

                        axes[i, 1].plot(obs_traj[:, 1], 'k-', label='des')
                        if c_pos_hist is not None and len(c_pos_hist):
                            axes[i, 1].plot(c_pos_hist[:, 1], 'r--', label='actual')
                        axes[i, 1].set_title('Y — des (black) vs actual (red)')

                        axes[i, 2].plot(obs_traj[:, 2], 'k-', label='Z des')
                        if c_pos_hist is not None and len(c_pos_hist):
                            axes[i, 2].plot(np.array(c_pos_hist)[:, 2], 'r--',
                                            alpha=0.7, label='Z actual')
                        axes[i, 2].set_title('Z — des (black) vs actual (red)')

                        vels = np.linalg.norm(obs_traj[1:] - obs_traj[:-1], axis=1)
                        axes[i, 3].plot(vels, color='gray', alpha=0.5)
                        axes[i, 3].set_title('Step Magnitude')

                        axes[i, 4].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', linewidth=2)
                        axes[i, 4].plot(obs_traj[0, 0],  obs_traj[0, 1],  'go', markersize=10)
                        axes[i, 4].plot(obs_traj[-1, 0], obs_traj[-1, 1], 'ro', markersize=10)
                        axes[i, 4].set_title('XY Trajectory')

                        all_cands_list = rollout_data.get('all_candidates', [])
                        sel_idx_list   = rollout_data.get('selected_idx',   [])
                        axes[i, 5].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', alpha=0.4)
                        for step_i, (cands, sel) in enumerate(zip(all_cands_list, sel_idx_list)):
                            if step_i % 4 != 0:
                                continue
                            # Fix 9 I2: cands are already unnormalized c_pos (XY), no cumsum needed
                            for b in range(cands.shape[0]):
                                if b == sel:
                                    axes[i, 5].plot(cands[b, :, 0], cands[b, :, 1],
                                                    color='green', linewidth=0.8, alpha=0.9)
                                else:
                                    axes[i, 5].plot(cands[b, :, 0], cands[b, :, 1],
                                                    color='gray', linewidth=0.2, alpha=0.2)
                        n_cands = all_cands_list[0].shape[0] if all_cands_list else 1
                        axes[i, 5].set_title(f'MPC Foresight — {n_cands} candidates/step')

                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    fig.savefig(f'{save_path}/{variant}.png')
                    plt.close(fig)

                # ── Aligning eval summary ────────────────────────────────────
                n_success   = np.array(successes.flatten())
                n_steps     = np.array(agent.history_n_steps)
                dists       = np.array(agent.history_rollout_mean_dist)
                all_errs    = np.concatenate([e for e in agent.history_pos_tracking_errors
                                              if len(e)]) if agent.history_pos_tracking_errors else np.array([0.0])
                max_phys    = float(np.max(all_errs))
                mean_phys   = float(np.mean(all_errs))

                run_mode = 'seen training set' if args_cli.eval_on_train else 'default'
                print(f'--- aligning-d3il-visual [{run_mode}] {variant} seed={seed} ---')
                print(f'Success rate:              {np.mean(n_success):.4f}')
                print(f'Avg final mean distance:   {np.mean(dists):.4f} m  '
                      f'+- {np.std(dists):.4f} m')
                print(f'Min final mean distance:   {np.min(dists):.4f} m')
                print(f'Avg steps (successful):    '
                      f'{np.mean(n_steps[n_success > 0]) if n_success.sum() else 0:.2f}'
                      f' +- {np.std(n_steps[n_success > 0]) if n_success.sum() else 0:.2f}')
                print(f'Avg steps (all trials):    {np.mean(n_steps):.2f} +- {np.std(n_steps):.2f}')
                print(f'Physical tracking error:   mean={mean_phys:.4f} m  max={max_phys:.4f} m')
                print(f'Avg inference time/replan: {np.mean(agent.history_avg_time):.3f} s')

                # UF-16.3: aggregate constraint satisfaction metrics across rollouts
                _hcm = agent.history_constraint_metrics
                if _hcm:
                    def _cm_arr(key, default=0.0):
                        return np.array([m.get(key, default) for m in _hcm], dtype=float)
                    _sat_arr  = _cm_arr('exec_constraint_sat_rate', 1.0)
                    _nviol    = _cm_arr('exec_n_violated_steps')
                    _bv       = _cm_arr('exec_bounds_viol_count')
                    _hv       = _cm_arr('exec_halfspace_viol_count')
                    _ov       = _cm_arr('exec_obstacle_viol_count')
                    _bmx      = _cm_arr('exec_max_bounds_viol_m')
                    _hmx      = _cm_arr('exec_max_halfspace_viol_m')
                    _omx      = _cm_arr('exec_max_obstacle_penetration_m')
                    _mg       = _cm_arr('exec_constraint_margin_mean_m')
                    _fv       = _cm_arr('exec_first_violation_step', -1)
                    _ls       = _cm_arr('exec_longest_safe_streak')
                    _dy       = _cm_arr('exec_dynamics_consistency_error_mean')
                    _pv       = _cm_arr('plan_post_viol_rate_mean')
                    _zv_cnt   = int(sum(m.get('exec_zero_violation_rollout', False) for m in _hcm))
                    n_r = len(_hcm)
                    print(f'--- Constraint Metrics [{variant}] ---')
                    print(f'  Execution satisfaction rate:    {_sat_arr.mean():.3f} ± {_sat_arr.std():.3f}')
                    print(f'  Violated steps/rollout:         {_nviol.mean():.1f} ± {_nviol.std():.1f}  '
                          f'(bounds={_bv.mean():.1f}  hs={_hv.mean():.1f}  obs={_ov.mean():.1f})')
                    print(f'  Max bounds viol (m):            {_bmx.mean():.4f} ± {_bmx.std():.4f}')
                    print(f'  Max halfspace viol (m):         {_hmx.mean():.4f} ± {_hmx.std():.4f}')
                    print(f'  Max obstacle penetration (m):   {_omx.mean():.4f} ± {_omx.std():.4f}')
                    print(f'  Constraint margin mean (m):     {_mg.mean():.4f} ± {_mg.std():.4f}')
                    print(f'  First violation step:           {_fv[_fv>=0].mean():.1f} '
                          f'(n_rollouts_with_viol={((_fv>=0).sum())})')
                    print(f'  Longest safe streak (steps):    {_ls.mean():.1f} ± {_ls.std():.1f}')
                    print(f'  Dynamics consistency err (m):   {_dy.mean():.4f} ± {_dy.std():.4f}')
                    print(f'  Plan post-proj viol rate:       {_pv.mean():.4f} ± {_pv.std():.4f}')
                    print(f'  Zero-violation rollouts:        {_zv_cnt} / {n_r}  '
                          f'({100*_zv_cnt/max(1,n_r):.1f}%)')

                    # Json_Orgnize_C4 (Open Q3, resolved: yes): nest the aggregate the same
                    # way as the per-rollout stats JSON (exec/plan/by_family groups), so both
                    # artifacts share one convention. `per_rollout` now holds the SAME nested
                    # shape (via _nest_constraint_metrics) rather than the old flat dicts.
                    _cm_summary = {
                        'variant': variant, 'geo_name': geo_name, 'seed': int(seed),
                        'n_rollouts': n_r,
                        'exec': {
                            'constraint_sat_rate':    {'mean': float(_sat_arr.mean()), 'std': float(_sat_arr.std())},
                            'n_violated_steps':       {'mean': float(_nviol.mean()),  'std': float(_nviol.std())},
                            'by_family': {
                                'bounds':    {'viol_count':  {'mean': float(_bv.mean()),  'std': float(_bv.std())},
                                              'max_viol_m':  {'mean': float(_bmx.mean()), 'std': float(_bmx.std())}},
                                'halfspace': {'viol_count':  {'mean': float(_hv.mean()),  'std': float(_hv.std())},
                                              'max_viol_m':  {'mean': float(_hmx.mean()), 'std': float(_hmx.std())}},
                                'obstacles': {'viol_count':  {'mean': float(_ov.mean()),  'std': float(_ov.std())},
                                              'max_viol_m':  {'mean': float(_omx.mean()), 'std': float(_omx.std())}},
                            },
                            'margin_mean_m':        {'mean': float(_mg.mean()), 'std': float(_mg.std())},
                            'first_violation_step': {'mean': float(_fv[_fv>=0].mean()) if (_fv>=0).any() else -1,
                                                      'n_rollouts_with_violation': int((_fv>=0).sum())},
                            'longest_safe_streak':  {'mean': float(_ls.mean()), 'std': float(_ls.std())},
                            'dynamics_consistency_error': {'mean': float(_dy.mean()), 'std': float(_dy.std())},
                            'zero_violation_rollouts': _zv_cnt,
                        },
                        'plan': {
                            'post_viol_rate': {'mean': float(_pv.mean()), 'std': float(_pv.std())},
                        },
                        'per_rollout': [_nest_constraint_metrics(m) for m in _hcm],
                    }
                    _cm_path = os.path.join(save_path, 'constraint_metrics.json')
                    with open(_cm_path, 'w') as _cmf:
                        json.dump(_cm_summary, _cmf, indent=2, default=str)
                    print(f'  → Saved: {_cm_path}')

                print('-' * 80 + '\n')

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                log_f.close()
                # FIX-7 (per-variant): Reset MuJoCo robot body counter so next
                # variant's Robot_Push_Env gets robot_id=0 (rb0 body prefix),
                # matching the clean-process scene geometry.
                try:
                    _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
                except NameError:
                    pass
                # FIX-7.2 (per-variant): Clear render context cache so next variant
                # creates fresh RenderContextOffscreen objects.
                try:
                    _reset_render_singleton()
                except NameError:
                    pass
                for _stale in _glob.glob(os.path.join(_mj_dir, 'panda_tmp_rb*.xml')):
                    try:
                        os.remove(_stale)
                    except OSError:
                        pass

    print('Visual-FM (Gen7) evaluation completed.')
