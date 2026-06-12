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
    traverse_line, s_curve_path, weave, blended_path,
)

# ── U9 smooth trajectories ────────────────────────────────────────────────────
# Corner-blend radius for pillar_path / s_curve_scene_path (U8 Stop_and_Go
# analysis, Option A).  Max corner deviation r·(sec(β/2)−1) ≤ 0.13 m at 90°;
# verified clearances at every blend region (see U9 CHANGELOG):
#   s_curve Z-corners: ≥ 0.55 m to nearest wall corner (rotor reach 0.31 m)
#   pillar corners:    blends stay ≥ 0.33 m in x from pillar axes (reach 0.26 m)
BLEND_RADIUS = 0.30

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

    U9: the per-segment traverse_line chain (v=0 at all 6 interior waypoints —
    the stop-and-go behaviour, see U8 analysis) is replaced by blended_path:
    same 8-waypoint skeleton, circular fillets of radius ≤ BLEND_RADIUS at each
    non-collinear corner, one global cosine speed profile (v=0 only at episode
    start/end).  Straight portions near the pillars — where the 8 cm minimum
    clearance lives — are untouched; fillets only cut corners in open space
    ≥ 0.5 m in x from every pillar.  Peak speed π·L/(2T) is unchanged from the
    length-proportional chain.

    homotopy_seq : 3-element list, each 'L' or 'R' (one per pillar pair).
    """
    assert len(homotopy_seq) == 3, 'Need exactly 3 homotopy labels (one per pillar pair)'
    y_map = {'L': _Y_L, 'R': _Y_R}
    y_ch = [y_map[h] for h in homotopy_seq]
    z = float(altitude) if np.isscalar(altitude) else float(np.mean(altitude))
    T = float(duration)

    xs = [x_start, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, x_end]
    ys = [0.0, y_ch[0], y_ch[0], y_ch[1], y_ch[1], y_ch[2], y_ch[2], 0.0]
    wps = [(x, y, z) for x, y in zip(xs, ys)]

    return blended_path(wps, BLEND_RADIUS, T, yaw)


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

    U9 — same Z-route skeleton, smooth (U8 Stop_and_Go, Option A):
        The 7-phase chain (v=0 at every joint + two 1.0 s hovers) is replaced
        by blended_path over the SAME waypoints.  The (∓0.5, y, z) breakpoints
        are collinear with their neighbours → no fillet there; the two 90°
        Z-corners at (0, y1) and (0, y2) get 0.3 m fillets.

        Fillet clearance (the corner the hovers used to protect is removed
        rather than paused at): each fillet's closest point to the nearest
        gap-side wall corner (A=(−0.5,−0.25), B=(+0.5,+0.25)) is ≥ 0.55 m —
        well above the 0.31 m rotor reach.  The hovers are dropped entirely;
        v > 0 throughout the episode.

    Waypoint skeleton (unchanged from U7):
        (-3.2, y1, z) → (-0.5, y1, z) → (0, y1, z) → (0, y2, z)
                      → (+0.5, y2, z) → (+3.2, y2, z)
    """
    z  = float(altitude)
    T  = float(duration)
    y1 = -0.8 + y_jitter
    y2 =  0.8 + y_jitter

    wps = [(-3.2, y1, z), (-0.5, y1, z), (0.0, y1, z),
           ( 0.0, y2, z), ( 0.5, y2, z), (3.2, y2, z)]

    return blended_path(wps, BLEND_RADIUS, T, yaw)


def empty_path(p_start, p_end, duration, yaw=0.0):
    """Direct line from p_start to p_end (cosine-profile velocity)."""
    return traverse_line(p_start, p_end, duration, yaw=yaw)
