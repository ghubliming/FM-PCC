"""U9 blend verification: clearance + smoothness checks for blended paths.

Numerically verifies the Option A redesign (U8 Stop_and_Go analysis) without
mujoco — pure numpy on the NOMINAL trajectories:

  1. No stop-and-go: interior speed never drops to ~0 (the whole point of U9).
  2. Consistency: analytic v matches finite-difference of p (no kinks/jumps
     at element joints — the PID feedforward is trustworthy).
  3. Clearance: nominal path keeps ≥ rotor-reach distance from every obstacle,
     with the same margins the U3/U7 straight-line proofs established.

Run:  python uav_expert_data_collect/verify_blends.py
Exit code 0 = all PASS.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import uav_expert_data_collect.trajectories as trajs

DT = 0.002          # physics timestep used by the generator
ROTOR_REACH = 0.31  # max COM→rotor-ellipsoid-edge distance (conservative, all dirs)


def _sample(traj_fn, T):
    ts = np.arange(0.0, T, DT)
    out = [traj_fn(t) for t in ts]
    p = np.array([o[0] for o in out])
    v = np.array([o[1] for o in out])
    a = np.array([o[2] for o in out])
    return ts, p, v, a


def _check_smooth(name, ts, p, v, a, T):
    """Interior speed > 0 and analytic v consistent with finite-diff of p."""
    speed = np.linalg.norm(v, axis=1)
    # Interior = exclude 5% ramp-in/out where v→0 by design (episode ends).
    lo, hi = int(0.05 * len(ts)), int(0.95 * len(ts))
    v_min_int = speed[lo:hi].min()
    v_peak = speed.max()
    a_peak = np.linalg.norm(a, axis=1).max()

    fd_v = np.diff(p, axis=0) / DT
    v_mid = 0.5 * (v[1:] + v[:-1])
    fd_err = np.linalg.norm(fd_v - v_mid, axis=1).max()

    ok = v_min_int > 0.05 and fd_err < 0.05
    print(f'  [{name}] interior v_min={v_min_int:.3f}  v_peak={v_peak:.3f} m/s  '
          f'a_peak={a_peak:.2f} m/s²  fd_err={fd_err:.4f}  '
          f'{"PASS" if ok else "FAIL"}')
    return ok


def _dist_to_box_xy(p_xy, center, half):
    """Min distance from points to an axis-aligned box (xy plane)."""
    d = np.maximum(np.abs(p_xy - np.asarray(center)) - np.asarray(half), 0.0)
    return np.linalg.norm(d, axis=1)


def check_pillars():
    print('pillars — clearance gate: ≥ 0.43 m to pillar axes '
          '(0.12 radius + 0.31 reach):')
    pillar_xy = [(x, y) for x in (-2.0, 0.0, 2.0) for y in (-0.6, 0.6)]
    ok = True
    for homotopy in (['L', 'L', 'L'], ['L', 'R', 'L'], ['R', 'L', 'R'], ['R', 'R', 'R']):
        for T in (10.0, 16.0):
            fn = trajs.pillar_path(homotopy, altitude=1.0, duration=T)
            ts, p, v, a = _sample(fn, T)
            d_min = min(np.linalg.norm(p[:, :2] - np.array(c), axis=1).min()
                        for c in pillar_xy)
            tag = f'{"".join(homotopy)} T={T:g}'
            clear_ok = d_min >= 0.43
            print(f'  [{tag}] min pillar-axis dist={d_min:.3f} m '
                  f'{"PASS" if clear_ok else "FAIL"}')
            ok &= clear_ok
            ok &= _check_smooth(tag, ts, p, v, a, T)
    return ok


def check_s_curve():
    print('s_curve — clearance gate: ≥ 0.31 m (rotor reach) to all wall boxes:')
    walls = [  # (center_xy, half_xy) from generator.SCENE_OBSTACLES
        ((-1.75, -1.3), (1.25, 0.05)),
        ((-1.75, -0.3), (1.25, 0.05)),
        (( 1.75,  0.3), (1.25, 0.05)),
        (( 1.75,  1.3), (1.25, 0.05)),
    ]
    corners = [(-0.5, -0.25), (0.5, 0.25)]  # gap-side critical corners A, B
    ok = True
    for y_jitter in (-0.04, 0.0, 0.04):
        for T in (16.0, 22.0):
            fn = trajs.s_curve_scene_path(altitude=1.0, duration=T,
                                          y_jitter=y_jitter)
            ts, p, v, a = _sample(fn, T)
            d_wall = min(_dist_to_box_xy(p[:, :2], c, h).min() for c, h in walls)
            d_corner = min(np.linalg.norm(p[:, :2] - np.array(c), axis=1).min()
                           for c in corners)
            tag = f'jit={y_jitter:+.2f} T={T:g}'
            clear_ok = d_wall >= ROTOR_REACH
            print(f'  [{tag}] min wall dist={d_wall:.3f} m  '
                  f'min A/B-corner dist={d_corner:.3f} m '
                  f'{"PASS" if clear_ok else "FAIL"}')
            ok &= clear_ok
            ok &= _check_smooth(tag, ts, p, v, a, T)
    return ok


if __name__ == '__main__':
    ok = check_pillars()
    print()
    ok &= check_s_curve()
    print()
    print('ALL PASS' if ok else 'FAILURES — do not collect')
    sys.exit(0 if ok else 1)
