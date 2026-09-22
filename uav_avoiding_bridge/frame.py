"""Frames and constants of the avoiding -> quadrotor similarity map (Gen15 U18).  Pure python on purpose:
this module is imported by the scene generator, which must run anywhere (no numpy, no mujoco).

Avoiding frame  a = (x_a, y_a): the D3IL table. Progress along +y_a from the start (0.525, -0.28) to the
                finish line y_a = 0.35. Field x_a in [0.2, 0.8], y_a in [-0.3, 0.4] (config/projection_eval.yaml
                ax_limits). Every planner, projector, constraint and score lives here.
World frame     w = (X, Y, Z): the MuJoCo quadrotor scene. Progress along +X like every UAV scene in the repo.

    X =  SCALE * (y_a - ORIGIN_A[1])
    Y = -SCALE * (x_a - ORIGIN_A[0])          # proper rotation by -90 deg: left/right are preserved
    Z =  ALTITUDE                             # the plan has no z; the PID's z-loop holds it

At SCALE = 10 (see the plan, section 2): arena 6.0 x 7.0 m, start (-3.15, -0.25), finish line X = 3.15,
pillars r = 0.25 / 0.30 m, the planner's one keep-out disk per geometry maps to r = 0.80 m.
"""
import os

# ── the knobs (env overrides so a scale / altitude variant needs no code edit) ─────────────────────
SCALE    = float(os.environ.get('FMPCC_AVOID_UAV_SCALE', '10'))
ALTITUDE = float(os.environ.get('FMPCC_AVOID_UAV_ALT',   '1.0'))
ORIGIN_A = (0.5, 0.035)          # avoiding point mapped to world (0, 0): field centre line, start/finish midpoint

# ── the avoiding table, copied from d3il/.../gym_avoiding/envs/objects/avoiding_objects.py ────────
START_A       = (0.525, -0.28)   # init_end_eff_pos[:2]
GOAL_Y_A      = -0.1 + 2.5 * 0.18   # = 0.35, ObstacleAvoidanceEnv.goal_ypos (success: y_a > GOAL_Y_A)
FIELD_X_A     = (0.2, 0.8)       # ax_limits
FIELD_Y_A     = (-0.3, 0.4)
_mid, _off, _y1, _lvl = 0.5, 0.075, -0.1, 0.18
# (name, x_a, y_a, physical radius) — Cylinder size[0] in avoiding_objects.py (l1: 0.03, others 0.025)
OBSTACLES_A = (
    ('l1_obs',        _mid,            _y1,            0.03),
    ('l2_top_obs',    _mid - _off,     _y1 + _lvl,     0.025),
    ('l2_bottom_obs', _mid + _off,     _y1 + _lvl,     0.025),
    ('l3_top_obs',    _mid - 2 * _off, _y1 + 2 * _lvl, 0.025),
    ('l3_mid_obs',    _mid,            _y1 + 2 * _lvl, 0.025),
    ('l3_bottom_obs', _mid + 2 * _off, _y1 + 2 * _lvl, 0.025),
)
PILLAR_HEIGHT = 2.0              # m, the "infinite" extrusion (drone flies at ALTITUDE, well below the top)


def to_world_xy(x_a, y_a, scale=None):
    s = SCALE if scale is None else scale
    return s * (y_a - ORIGIN_A[1]), -s * (x_a - ORIGIN_A[0])


def to_avoiding_xy(X, Y, scale=None):
    s = SCALE if scale is None else scale
    return ORIGIN_A[0] - Y / s, ORIGIN_A[1] + X / s


def world_field(scale=None):
    """((Xmin, Xmax), (Ymin, Ymax)) of the avoiding field in the world."""
    s = SCALE if scale is None else scale
    xs = [to_world_xy(x, y, s)[0] for x in FIELD_X_A for y in FIELD_Y_A]
    ys = [to_world_xy(x, y, s)[1] for x in FIELD_X_A for y in FIELD_Y_A]
    return (min(xs), max(xs)), (min(ys), max(ys))


def scene_tag(scale=None):
    s = SCALE if scale is None else scale
    return f's{int(round(s))}' if abs(s - round(s)) < 1e-9 else f's{s:g}'


def describe(scale=None):
    s = SCALE if scale is None else scale
    (x0, x1), (y0, y1) = world_field(s)
    sx, sy = to_world_xy(*START_A, s)
    lines = [f'scale {s:g}  altitude {ALTITUDE:g} m  origin_a {ORIGIN_A}',
             f'field X [{x0:.2f}, {x1:.2f}]  Y [{y0:.2f}, {y1:.2f}]  start ({sx:.2f}, {sy:.2f})  '
             f'finish X = {to_world_xy(0.5, GOAL_Y_A, s)[0]:.2f}']
    for n, xa, ya, r in OBSTACLES_A:
        X, Y = to_world_xy(xa, ya, s)
        lines.append(f'  {n:14s} a=({xa:.3f},{ya:.3f}) r={r:.3f}  ->  w=({X:+.2f},{Y:+.2f}) r={s * r:.3f}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe())
