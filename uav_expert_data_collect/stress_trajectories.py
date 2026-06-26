"""E4 U10 — Stress-test trajectory builders.

Each builder corresponds to one stress case.  Public API:

    build_stress_traj(case, scene, rng)
        -> (traj_fn, init_pos, duration, extras)

    traj_fn(t) -> (p_des, v_des, a_des, yaw_des)
    init_pos   : np.ndarray shape (3,)
    duration   : float (seconds)
    extras     : dict — case-specific metadata merged into episode metadata
                 always contains 'kp_scale' and 'kd_scale' (default 1.0)

STRESS_CASES        : list of all case names
STRESS_CASE_SCENES  : dict mapping case -> list of applicable scenes
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from uav_env_test.trajectories import traverse_line, hover_at, blended_path
from uav_expert_data_collect.trajectories import (
    _Y_L, _Y_R, BLEND_RADIUS,
    CORRIDOR_CHANNELS, PILLAR_XS, PILLAR_Y_A, PILLAR_Y_B,
    s_curve_scene_path, corridor_path,
)

# ── Case registry ──────────────────────────────────────────────────────────────

STRESS_CASES = [
    'extreme_speed',
    'tight_fillet',
    'discontinuous',
    'wall_crossing',
    'floor_dive',
    'ceiling_climb',
    'gain_extreme',
    'degenerate_hover',
]

# Scenes each case applies to (determines the collect_stress.py loop).
STRESS_CASE_SCENES = {
    'extreme_speed':    ['pillars', 's_curve', 'corridor'],
    'tight_fillet':     ['pillars'],
    'discontinuous':    ['empty', 'corridor', 'pillars'],
    'wall_crossing':    ['corridor', 'pillars'],
    'floor_dive':       ['empty', 'corridor'],
    'ceiling_climb':    ['empty'],
    'gain_extreme':     ['corridor', 'empty'],
    'degenerate_hover': ['empty'],
}

_DEFAULT_EXTRAS = {'kp_scale': 1.0, 'kd_scale': 1.0}


# ── Internal trajectory helpers ────────────────────────────────────────────────

def _step_traj(phases):
    """Piecewise hover: phases = [(t_end, p_np_array), ...], last phase unbounded."""
    phases = [(t_end, np.asarray(p, dtype=float)) for t_end, p in phases]
    p_final = phases[-1][1]

    def traj(t):
        p = p_final
        for t_end, pos in phases:
            if t <= t_end:
                p = pos
                break
        return p.copy(), np.zeros(3), np.zeros(3), 0.0

    return traj


def _pillar_path_custom_radius(homotopy_seq, altitude, duration, radius):
    """pillar_path with an overridden blend radius (for tight_fillet stress case)."""
    y_map = {'L': _Y_L, 'R': _Y_R}
    y_ch = [y_map[h] for h in homotopy_seq]
    z = float(altitude)
    T = float(duration)
    xs = [-3.2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.2]
    ys = [0.0, y_ch[0], y_ch[0], y_ch[1], y_ch[1], y_ch[2], y_ch[2], 0.0]
    wps = [(x, y, z) for x, y in zip(xs, ys)]
    return blended_path(wps, radius, T, 0.0)


# ── Case builders ──────────────────────────────────────────────────────────────

def _extreme_speed(scene, rng):
    """Validated trajectory at 3-4× the normal speed budget."""
    z = float(rng.uniform(0.90, 1.30))
    if scene == 'pillars':
        # LRL at T=3 s (validated minimum is 10 s → ~3× overspeed)
        from uav_expert_data_collect.trajectories import pillar_path
        traj_fn = pillar_path(['L', 'R', 'L'], altitude=z, duration=3.0)
        init_pos = np.array([-3.2, 0.0, z])
        dur = 3.0
    elif scene == 's_curve':
        # Z-route at T=4 s (validated minimum is 16 s → 4× overspeed)
        traj_fn = s_curve_scene_path(altitude=z, duration=4.0)
        init_pos = np.array([-3.2, -0.8, z])
        dur = 4.0
    else:  # corridor
        traj_fn = corridor_path('L', altitude=z, duration=2.0)
        init_pos = np.array([-2.8, CORRIDOR_CHANNELS['L'], z])
        dur = 2.0
    return traj_fn, init_pos, dur, dict(_DEFAULT_EXTRAS)


def _tight_fillet(scene, rng):
    """LRL pillars with r=0.05 m (22× smaller than Fix_1 fix).
    Deliberately recreates the Fix_1 failure (r=0.30 already caused 45% rejection;
    r=0.05 guarantees PID saturation and pillar contact)."""
    z = float(rng.uniform(0.90, 1.30))
    dur = float(rng.uniform(10.0, 16.0))
    traj_fn = _pillar_path_custom_radius(['L', 'R', 'L'], altitude=z, duration=dur, radius=0.05)
    return traj_fn, np.array([-3.2, 0.0, z]), dur, dict(_DEFAULT_EXTRAS)


def _discontinuous(scene, rng):
    """p_des teleports by 2–3 m mid-episode (step-input stress on PID)."""
    z = float(rng.uniform(0.90, 1.30))
    dur = 9.0
    t1, t2 = 3.0, 6.0  # jump at t=3 s, jump back at t=6 s

    if scene == 'empty':
        p0 = np.array([-1.0,  0.0, z])
        pj = np.array([ 1.5,  1.5, z])
        p1 = np.array([ 0.0, -1.0, z])
    elif scene == 'corridor':
        p0 = np.array([-2.5, CORRIDOR_CHANNELS['L'], z])
        pj = np.array([ 0.0, CORRIDOR_CHANNELS['R'], z])
        p1 = np.array([ 2.5, CORRIDOR_CHANNELS['L'], z])
    else:  # pillars
        p0 = np.array([-3.0, _Y_L, z])
        pj = np.array([ 0.0, _Y_R, z])
        p1 = np.array([ 3.0, _Y_L, z])

    traj_fn = _step_traj([(t1, p0), (t2, pj), (dur + 1.0, p1)])
    return traj_fn, p0.copy(), dur, dict(_DEFAULT_EXTRAS)


def _wall_crossing(scene, rng):
    """Trajectory that crosses directly through obstacle geometry."""
    z = float(rng.uniform(0.90, 1.30))
    if scene == 'corridor':
        # Diagonal that pierces both walls (y: −0.4 → +0.4 sweeps through ±0.5 walls)
        traj_fn = traverse_line([-2.8, -0.4, z], [2.8, 0.4, z], 8.0)
        init_pos = np.array([-2.8, -0.4, z])
        dur = 8.0
    else:  # pillars — fly straight through column A (y = −0.6)
        traj_fn = traverse_line([-3.2, PILLAR_Y_A, z], [3.2, PILLAR_Y_A, z], 10.0)
        init_pos = np.array([-3.2, PILLAR_Y_A, z])
        dur = 10.0
    return traj_fn, init_pos, dur, dict(_DEFAULT_EXTRAS)


def _floor_dive(scene, rng):
    """p_des descends to z = −0.5 (below floor at z=0); tests floor gate and crash rendering."""
    z = float(rng.uniform(0.90, 1.30))
    if scene == 'corridor':
        p_s = np.array([-2.0, 0.0, z])
        p_e = np.array([ 2.0, 0.0, -0.5])
    else:  # empty
        p_s = np.array([0.0, 0.0, z])
        p_e = np.array([0.0, 0.0, -0.5])
    traj_fn = traverse_line(p_s, p_e, 4.0)
    return traj_fn, p_s.copy(), 4.0, dict(_DEFAULT_EXTRAS)


def _ceiling_climb(scene, rng):
    """Rapid climb to z=4.5 in 2 s — saturates vertical thrust."""
    z = float(rng.uniform(0.90, 1.30))
    p_s = np.array([0.0, 0.0, z])
    p_e = np.array([0.0, 0.0, 4.5])
    traj_fn = traverse_line(p_s, p_e, 2.0)
    return traj_fn, p_s.copy(), 2.0, dict(_DEFAULT_EXTRAS)


def _gain_extreme(scene, rng):
    """Normal corridor/empty path with PID gains ×5 (high) or ×0.1 (low).
    Variant selected randomly; actual scales stored in extras → episode metadata."""
    z = float(rng.uniform(0.90, 1.30))
    high = (rng.uniform() < 0.5)
    kp_scale = 5.0 if high else 0.1
    kd_scale = 1.0 if high else 0.3
    if scene == 'corridor':
        dur = 8.0
        traj_fn = corridor_path('C', altitude=z, duration=dur)
        init_pos = np.array([-2.8, CORRIDOR_CHANNELS['C'], z])
    else:  # empty
        p_s = np.array([-1.0, -1.0, z])
        p_e = np.array([ 1.0,  1.0, z])
        dur = 6.0
        traj_fn = traverse_line(p_s, p_e, dur)
        init_pos = p_s.copy()
    extras = {'kp_scale': kp_scale, 'kd_scale': kd_scale,
              'gain_variant': 'high' if high else 'low'}
    return traj_fn, init_pos, dur, extras


def _degenerate_hover(scene, rng):
    """Static hover — tests the renderer on a drone that does not move.
    Also exercises blended_path duplicate-waypoint deduplication via a short
    episode with duplicate adjacent waypoints."""
    z = float(rng.uniform(0.90, 1.30))
    # 2 s hover — produces ~66 frames at 33 Hz (non-zero, non-trivial)
    traj_fn = hover_at([0.0, 0.0, z])
    return traj_fn, np.array([0.0, 0.0, z]), 2.0, dict(_DEFAULT_EXTRAS)


# ── Dispatcher ─────────────────────────────────────────────────────────────────

_BUILDERS = {
    'extreme_speed':    _extreme_speed,
    'tight_fillet':     _tight_fillet,
    'discontinuous':    _discontinuous,
    'wall_crossing':    _wall_crossing,
    'floor_dive':       _floor_dive,
    'ceiling_climb':    _ceiling_climb,
    'gain_extreme':     _gain_extreme,
    'degenerate_hover': _degenerate_hover,
}


def build_stress_traj(case, scene, rng):
    """Return (traj_fn, init_pos, duration, extras) for the given stress case + scene.

    extras always contains 'kp_scale' and 'kd_scale' (1.0 for all non-gain_extreme cases).
    """
    if case not in _BUILDERS:
        raise ValueError(f'Unknown stress case: {case!r}. Valid: {list(_BUILDERS)}')
    applicable = STRESS_CASE_SCENES.get(case, [])
    if scene not in applicable:
        raise ValueError(f'Stress case {case!r} not applicable to scene {scene!r}. '
                         f'Applicable: {applicable}')
    return _BUILDERS[case](scene, rng)
