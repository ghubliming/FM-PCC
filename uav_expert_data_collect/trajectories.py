"""Scene-aware trajectory generators for UAV expert data collection (Epoch 4).

Wraps Epoch-3 base factories from uav_env_test.trajectories and adds
homotopy-labelled scene-specific generators used by generator.py.

Trajectory API (same as base factories):
    traj(t) -> (p, v, a, yaw)
    p, v, a ∈ R^3   world-frame position, velocity, acceleration
    yaw     ∈ R     desired yaw (radians)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from uav_env_test.trajectories import (  # noqa: F401  (re-exported for collectors)
    hover_at, step_to, circle,
    traverse_line, s_curve_path, weave,
)

# ── Pillar scene geometry ─────────────────────────────────────────────────────
# 6 cylinders: column A at y=-0.6, column B at y=+0.6, x ∈ {-2, 0, +2}
# radius 0.12 m.  Drone enters from x≈-3.2, exits at x≈+3.2.
PILLAR_XS       = [-2.0, 0.0, 2.0]
PILLAR_Y_A      = -0.6
PILLAR_Y_B      = +0.6
PILLAR_RADIUS   = 0.12
PILLAR_MARGIN   = 0.20   # extra clearance beyond pillar edge

# Channel centres for each homotopy class:
#   L = pass to the left of column A (y < PILLAR_Y_A - r - margin ≈ -0.92)
#   C = pass between the two columns (y ≈ 0; clearance 0.48 m to each edge)
#   R = pass to the right of column B (y > PILLAR_Y_B + r + margin ≈ +0.92)
_Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_MARGIN   # ≈ -0.92
_Y_C = 0.0
_Y_R = PILLAR_Y_B + PILLAR_RADIUS + PILLAR_MARGIN   # ≈ +0.92
PILLAR_CHANNELS = {'L': _Y_L, 'C': _Y_C, 'R': _Y_R}

# ── Corridor scene geometry ───────────────────────────────────────────────────
# Walls at y=-0.5 (neg) and y=+0.5 (pos), thickness 0.05 m each.
# Inner clear-space: y ∈ (-0.45, +0.45).
CORRIDOR_CHANNELS = {
    'L':  -0.18,
    'C':   0.0,
    'R':  +0.18,
}


def pillar_path(homotopy_seq, altitude, duration,
                x_start=-3.2, x_end=3.2, yaw=0.0):
    """Explicit L/R/C homotopy path through 3 pillar pairs.

    homotopy_seq : 3-element iterable, each element 'L', 'R', or 'C'.
    altitude     : z for all waypoints (scalar) or per-waypoint sequence.
    duration     : total flight time (s).
    x_start/end  : entry and exit x positions (well outside the pillar field).

    Uses s_curve_path (piecewise cosine-blended traverse_line) through 8
    waypoints:  entry → approach-pair-1 → pair-1 → mid-1-2 → pair-2
                     → mid-2-3 → pair-3 → exit.
    """
    assert len(homotopy_seq) == 3, 'Need exactly 3 homotopy labels (one per pillar pair)'
    y_ch = [PILLAR_CHANNELS[h] for h in homotopy_seq]

    z = float(altitude) if np.isscalar(altitude) else float(np.mean(altitude))

    # Blend from y=0 at entry/exit to the channel y at each pillar station.
    # Intermediate "blend" waypoints 1 m before/after each pillar pair.
    xs = [x_start, -2.8, -2.0, -1.0, 0.0, 1.0, 2.0, x_end]
    ys = [
        0.0,
        y_ch[0] * 0.6,                        # approaching pair 1
        y_ch[0],                               # at pair 1
        y_ch[0] * 0.4 + y_ch[1] * 0.6,       # mid between pair 1 and 2
        y_ch[1],                               # at pair 2
        y_ch[1] * 0.4 + y_ch[2] * 0.6,       # mid between pair 2 and 3
        y_ch[2],                               # at pair 3
        0.0,
    ]
    wps = [(xs[i], ys[i], z) for i in range(8)]
    seg_dur = duration / 7.0   # 7 legs among 8 waypoints
    return s_curve_path(wps, seg_dur, yaw=yaw)


def corridor_path(homotopy, altitude, duration,
                  x_start=-2.8, x_end=2.8, yaw=0.0):
    """Straight traverse through the corridor with left/centre/right bias.

    homotopy : 'L', 'C', or 'R'.
    """
    y = CORRIDOR_CHANNELS[homotopy]
    return traverse_line([x_start, y, altitude], [x_end, y, altitude],
                         duration, yaw=yaw)


def s_curve_scene_path(altitude, duration, y_jitter=0.0, yaw=0.0):
    """Standard 4-waypoint S-curve path with optional y jitter.

    Scene geometry (two corridor segments):
        Seg 1: x ∈ [-3, -0.5], corridor centred at y=-0.8
        Seg 2: x ∈ [+0.5, +3], corridor centred at y=+0.8

    y_jitter : small lateral perturbation (m) applied to both corridor
               centres — shifts the entire path slightly without violating walls.
    """
    z = float(altitude)
    wps = [
        (-3.2, -0.8 + y_jitter, z),
        (-0.5, -0.8 + y_jitter, z),
        ( 0.5,  0.8 + y_jitter, z),
        ( 3.2,  0.8 + y_jitter, z),
    ]
    seg_dur = duration / 3.0
    return s_curve_path(wps, seg_dur, yaw=yaw)


def empty_path(p_start, p_end, duration, yaw=0.0):
    """Direct line from p_start to p_end (cosine-profile velocity)."""
    return traverse_line(p_start, p_end, duration, yaw=yaw)
