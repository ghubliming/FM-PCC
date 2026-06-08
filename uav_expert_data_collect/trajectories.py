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
PILLAR_XS          = [-2.0, 0.0, 2.0]
PILLAR_Y_A         = -0.6
PILLAR_Y_B         = +0.6
PILLAR_RADIUS      = 0.12
# U3: margin accounts for rotor reach (0.31 m) + 8 cm safety, not just pillar edge.
# Previous PILLAR_MARGIN=0.20 gave _Y_L=-0.92 which is inside the contact zone
# (rotor clips pillar at y ∈ (-0.55, -0.97)).  New margin gives 10.8 cm clearance.
PILLAR_ROTOR_REACH = 0.31   # max y-distance from COM to rotor ellipsoid edge
PILLAR_SAFETY      = 0.08   # 8 cm above zero-contact, sufficient for PID tracking error

# Channel centres: L = left of col A, R = right of col B.
_Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_ROTOR_REACH - PILLAR_SAFETY  # = -1.11
_Y_C = 0.0
_Y_R = PILLAR_Y_B + PILLAR_RADIUS + PILLAR_ROTOR_REACH + PILLAR_SAFETY  # = +1.11
PILLAR_CHANNELS = {'L': _Y_L, 'C': _Y_C, 'R': _Y_R}

# ── Corridor scene geometry ───────────────────────────────────────────────────
# Walls at y=-0.5 (neg) and y=+0.5 (pos), thickness 0.05 m each.
# Inner clear-space: y ∈ (-0.45, +0.45).
# U3: channels moved inward from ±0.18 to ±0.12.  At ±0.18 the rotor reached
# 9 cm into the wall at worst jitter (contact by design).  At ±0.12 with no
# L/R jitter, rotor clears wall by 2 cm: 0.12 + 0.31 = 0.43 < 0.45.
CORRIDOR_CHANNELS = {
    'L':  -0.12,   # was -0.18
    'C':   0.0,
    'R':  +0.12,   # was +0.18
}


def pillar_path(homotopy_seq, altitude, duration,
                x_start=-3.2, x_end=3.2, yaw=0.0):
    """Explicit L/R homotopy path through 3 pillar pairs.

    U3 redesign: 5-waypoint scheme with waypoints AT each pillar x position.
    Channel centres use _Y_L=-1.11 / _Y_R=+1.11, giving 10.8 cm rotor clearance.
    Zero-velocity transitions are safe at these margins (unlike the old _Y_L=-0.92
    which was inside the contact zone — the original reason for switching to weave).

    homotopy_seq : 3-element list, each 'L' or 'R' (one per pillar pair).
    """
    assert len(homotopy_seq) == 3, 'Need exactly 3 homotopy labels (one per pillar pair)'
    y_map = {'L': _Y_L, 'R': _Y_R}
    y_ch = [y_map[h] for h in homotopy_seq]
    z = float(altitude) if np.isscalar(altitude) else float(np.mean(altitude))
    T = float(duration)

    # Waypoints: entry(y=0) → pillar1(y=y_ch[0]) → pillar2 → pillar3 → exit(y=0)
    xs = [x_start, -2.0, 0.0, 2.0, x_end]
    ys = [0.0, y_ch[0], y_ch[1], y_ch[2], 0.0]

    # Time proportional to x-distance
    total_x   = x_end - x_start
    seg_durs  = [T * (xs[i+1] - xs[i]) / total_x for i in range(4)]
    t_starts  = [sum(seg_durs[:i]) for i in range(4)]

    segs = [
        traverse_line((xs[i], ys[i], z), (xs[i+1], ys[i+1], z), seg_durs[i], yaw)
        for i in range(4)
    ]

    def traj(t):
        for i in range(3, -1, -1):
            if t >= t_starts[i]:
                return segs[i](t - t_starts[i])
        return segs[0](t)

    return traj


def corridor_path(homotopy, altitude, duration,
                  x_start=-2.8, x_end=2.8, yaw=0.0):
    """Straight traverse through the corridor with left/centre/right bias.

    homotopy : 'L', 'C', or 'R'.
    """
    y = CORRIDOR_CHANNELS[homotopy]
    return traverse_line([x_start, y, altitude], [x_end, y, altitude],
                         duration, yaw=yaw)


def s_curve_scene_path(altitude, duration, y_jitter=0.0, yaw=0.0):
    """S-curve path with duration allocated proportional to segment distance.

    Scene geometry (two corridor segments):
        Seg 1: x ∈ [-3, -0.5], corridor centred at y=-0.8
        Seg 2: x ∈ [+0.5, +3], corridor centred at y=+0.8

    Fix_5: replaced tanh continuous trajectory (peak lateral speed 1.17 m/s →
    47% rejection) with 3-segment piecewise traverse_line where each segment's
    duration is proportional to its Euclidean length.  All segments run at the
    same peak speed (~0.55 m/s at T=20s), matching the corridor scene which
    achieves 87% pass rate at 0.72 m/s.

    Segment layout:
        Seg A: (-3.2, y1) → (-0.5, y1)  distance 2.7 m   (pure x, inside seg1)
        Seg B: (-0.5, y1) → (+0.5, y2)  distance 1.89 m  (diagonal gap crossing)
        Seg C: (+0.5, y2) → (+3.2, y2)  distance 2.7 m   (pure x, inside seg2)
    """
    z  = float(altitude)
    T  = float(duration)
    y1 = -0.8 + y_jitter
    y2 =  0.8 + y_jitter

    d_a = 2.7
    d_b = float(np.sqrt(1.0**2 + (y2 - y1)**2))   # ≈ 1.89 m when jitter=0
    d_c = 2.7
    d_total = d_a + d_b + d_c

    t_a = T * d_a / d_total
    t_b = T * d_b / d_total
    t_c = T * d_c / d_total

    seg_a = traverse_line((-3.2, y1, z), (-0.5, y1, z), t_a, yaw)
    seg_b = traverse_line((-0.5, y1, z), ( 0.5, y2, z), t_b, yaw)
    seg_c = traverse_line(( 0.5, y2, z), ( 3.2, y2, z), t_c, yaw)

    def traj(t):
        if t < t_a:
            return seg_a(t)
        elif t < t_a + t_b:
            return seg_b(t - t_a)
        else:
            return seg_c(t - t_a - t_b)

    return traj


def empty_path(p_start, p_end, duration, yaw=0.0):
    """Direct line from p_start to p_end (cosine-profile velocity)."""
    return traverse_line(p_start, p_end, duration, yaw=yaw)
