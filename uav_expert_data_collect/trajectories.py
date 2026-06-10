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
    """S-curve path: corridor 1 → gap crossing (Z-route) → corridor 2.

    Scene geometry:
        Seg 1: x ∈ [-3, -0.5], corridor centred at y=-0.8
        Seg 2: x ∈ [+0.5, +3], corridor centred at y=+0.8
        Gap:   x ∈ [-0.5, +0.5], open (no walls)

    U7 C1 — replaced the Seg B diagonal with a 3-leg Z-route through x=0:

    WHY the diagonal was infeasible (all prior fixes were misdiagnosed):
        Gap-side wall corners: A=(−0.5,−0.25) on seg1_wall_pos,
                               B=(+0.5,+0.25) on seg2_wall_neg.
        Diagonal (−0.5,y1)→(+0.5,y2) passes 0.291 m from both corners —
        INSIDE the 0.31 m rotor reach on the nominal path alone (0.019 m
        penetration before any tracking error). No speed or gain change
        can resolve a geometric infeasibility.

    Z-route clearances (verified):
        Leg B1 and B3 (pure-x): ≥ 0.55 m from both corners.
        Leg B2 (pure-y at x=0): 0.50 m from both corners.
        All legs parallel to the nearest wall at every pinch point →
        tracking lag is along-path and cannot reduce wall clearance.

    Segment layout (7 phases):
        Seg A:   (-3.2, y1, z) → (-0.5, y1, z)   2.7 m  pure-x
        Hov 1:   hover at (-0.5, y1, z)           1.0 s  stabilise
        Leg B1:  (-0.5, y1, z) → ( 0.0, y1, z)   0.5 m  pure-x, exit corridor 1
        Leg B2:  ( 0.0, y1, z) → ( 0.0, y2, z)   1.6 m  pure-y, cross on centerline
        Leg B3:  ( 0.0, y2, z) → (+0.5, y2, z)   0.5 m  pure-x, enter corridor 2
        Hov 2:   hover at (+0.5, y2, z)           1.0 s  stabilise
        Seg C:   (+0.5, y2, z) → (+3.2, y2, z)   2.7 m  pure-x

    Time allocation: proportional to Euclidean segment distance (no weighting).
    """
    z  = float(altitude)
    T  = float(duration)
    y1 = -0.8 + y_jitter
    y2 =  0.8 + y_jitter

    T_HOVER = 1.0
    T_move  = T - 2.0 * T_HOVER

    d_a  = 2.7
    d_b1 = 0.5
    d_b2 = float(abs(y2 - y1))   # ≈ 1.6 m when jitter=0
    d_b3 = 0.5
    d_c  = 2.7
    d_total = d_a + d_b1 + d_b2 + d_b3 + d_c

    t_a  = T_move * d_a  / d_total
    t_b1 = T_move * d_b1 / d_total
    t_b2 = T_move * d_b2 / d_total
    t_b3 = T_move * d_b3 / d_total
    t_c  = T_move * d_c  / d_total

    seg_a  = traverse_line((-3.2, y1, z), (-0.5, y1, z), t_a,  yaw)
    hov_1  = hover_at((-0.5, y1, z), yaw)
    leg_b1 = traverse_line((-0.5, y1, z), ( 0.0, y1, z), t_b1, yaw)
    leg_b2 = traverse_line(( 0.0, y1, z), ( 0.0, y2, z), t_b2, yaw)
    leg_b3 = traverse_line(( 0.0, y2, z), ( 0.5, y2, z), t_b3, yaw)
    hov_2  = hover_at(( 0.5, y2, z), yaw)
    seg_c  = traverse_line(( 0.5, y2, z), ( 3.2, y2, z), t_c,  yaw)

    # Cumulative phase-end times
    p1 = t_a
    p2 = p1 + T_HOVER
    p3 = p2 + t_b1
    p4 = p3 + t_b2
    p5 = p4 + t_b3
    p6 = p5 + T_HOVER

    def traj(t):
        if t < p1:
            return seg_a(t)
        elif t < p2:
            return hov_1(t)
        elif t < p3:
            return leg_b1(t - p2)
        elif t < p4:
            return leg_b2(t - p3)
        elif t < p5:
            return leg_b3(t - p4)
        elif t < p6:
            return hov_2(t)
        else:
            return seg_c(t - p6)

    return traj


def empty_path(p_start, p_end, duration, yaw=0.0):
    """Direct line from p_start to p_end (cosine-profile velocity)."""
    return traverse_line(p_start, p_end, duration, yaw=yaw)
