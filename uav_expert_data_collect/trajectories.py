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

    Fix_1 redesign: 8-waypoint scheme where channel transitions happen BETWEEN
    pillars, not at pillar x-positions.  Analytical minimum clearance: 8 cm on
    the straight stabilisation segments, 21-23 cm on the diagonals.

    Key geometry insight: the quadrotor has rotors at ±0.14 (x) and ±0.18 (y)
    from the body centre.  Any approach diagonal that reaches the target channel
    y AT a pillar x-position will cause the FRONT rotor (+0.14 in x) to contact
    the pillar on approach, and the REAR rotor (−0.14 in x) to drag through the
    pillar on departure.  The fix moves the y-transition midpoints to x=-1.5 and
    x=+0.5, leaving ≥0.5 m buffer from every pillar before/after each turn.

    Waypoint layout (7 segments):
        x: [-3.2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.2]
        y: [  0,  y0,   y0,  y1,  y1,  y2,  y2,   0 ]

    Time allocation: proportional to Euclidean segment length (not x-distance),
    so diagonal inter-channel segments get proportionally more time.

    homotopy_seq : 3-element list, each 'L' or 'R' (one per pillar pair).
    """
    assert len(homotopy_seq) == 3, 'Need exactly 3 homotopy labels (one per pillar pair)'
    y_map = {'L': _Y_L, 'R': _Y_R}
    y_ch = [y_map[h] for h in homotopy_seq]
    z = float(altitude) if np.isscalar(altitude) else float(np.mean(altitude))
    T = float(duration)

    xs = [x_start, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, x_end]
    ys = [0.0, y_ch[0], y_ch[0], y_ch[1], y_ch[1], y_ch[2], y_ch[2], 0.0]
    n  = len(xs) - 1  # 7 segments

    dists    = [np.sqrt((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2) for i in range(n)]
    total_d  = sum(dists)
    seg_durs = [T * d / total_d for d in dists]
    t_starts = [sum(seg_durs[:i]) for i in range(n)]

    segs = [
        traverse_line((xs[i], ys[i], z), (xs[i+1], ys[i+1], z), seg_durs[i], yaw)
        for i in range(n)
    ]

    def traj(t):
        for i in range(n - 1, -1, -1):
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
    """S-curve path with hover pauses at each segment junction.

    Scene geometry (two corridor segments):
        Seg 1: x ∈ [-3, -0.5], corridor centred at y=-0.8
        Seg 2: x ∈ [+0.5, +3], corridor centred at y=+0.8

    Fix_5: replaced tanh continuous trajectory with 3-segment piecewise
    traverse_line, proportional duration allocation.

    U4 Fix A: added 1.0 s hover pauses at junctions. Shown in F4 to be at
    wrong location (junction v was already ~0 per cosine profile) and at
    x=±0.5 which is on the wall end-face plane (only 0.14 m lateral margin).

    U5 Step 3: Seg B time budget doubled (2× geometric weight) to halve peak
    lateral accel/velocity — quadratically reduces attitude-loop overshoot that
    causes the motor saturation → altitude collapse chain.

    U5 Step 5: hover waypoints relocated from x=±0.5 (wall end-face) to
    x=∓0.7 (0.2 m inside corridor) to remove the wall-proximity risk.

    Segment layout (5 phases):
        Seg A:  (-3.2, y1, z) → (-0.7, y1, z)  2.5 m   pure-x
        Hov 1:  hover at (-0.7, y1, z)          1.0 s   stabilise
        Seg B:  (-0.7, y1, z) → (+0.7, y2, z)  2.13 m  diagonal gap crossing
        Hov 2:  hover at (+0.7, y2, z)          1.0 s   stabilise
        Seg C:  (+0.7, y2, z) → (+3.2, y2, z)  2.5 m   pure-x
    """
    z  = float(altitude)
    T  = float(duration)
    y1 = -0.8 + y_jitter
    y2 =  0.8 + y_jitter

    T_HOVER = 1.0                    # seconds per junction pause
    T_move  = T - 2.0 * T_HOVER     # time budget for the three traverse segments

    d_a = 2.5   # x: -3.2 → -0.7
    d_b = float(np.sqrt(1.4**2 + (y2 - y1)**2))   # ≈ 2.13 m when jitter=0
    d_c = 2.5   # x: +0.7 → +3.2

    # U5 Step 3: give Seg B 2× weight so peak lateral velocity is ~halved
    d_total = d_a + 2.0 * d_b + d_c
    t_a = T_move * d_a / d_total
    t_b = T_move * 2.0 * d_b / d_total
    t_c = T_move * d_c / d_total

    seg_a = traverse_line((-3.2, y1, z), (-0.7, y1, z), t_a, yaw)
    hov_1 = hover_at((-0.7, y1, z), yaw)
    seg_b = traverse_line((-0.7, y1, z), ( 0.7, y2, z), t_b, yaw)
    hov_2 = hover_at(( 0.7, y2, z), yaw)
    seg_c = traverse_line(( 0.7, y2, z), ( 3.2, y2, z), t_c, yaw)

    # Cumulative phase-end times
    t1 = t_a
    t2 = t_a + T_HOVER
    t3 = t2 + t_b
    t4 = t3 + T_HOVER

    def traj(t):
        if t < t1:
            return seg_a(t)
        elif t < t2:
            return hov_1(t)
        elif t < t3:
            return seg_b(t - t2)
        elif t < t4:
            return hov_2(t)
        else:
            return seg_c(t - t4)

    return traj


def empty_path(p_start, p_end, duration, yaw=0.0):
    """Direct line from p_start to p_end (cosine-profile velocity)."""
    return traverse_line(p_start, p_end, duration, yaw=yaw)
