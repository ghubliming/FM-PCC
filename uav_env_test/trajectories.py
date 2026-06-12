"""Hand-coded reference trajectories for Epoch-2 naive fly tests.

Each factory returns a callable `traj(t) -> (p, v, a, yaw)` where:
  p, v, a ∈ R^3  (world-frame position, velocity, acceleration)
  yaw    ∈ R     (desired yaw, radians)

The driver decides whether to pass `a` to the controller (9-D format) or
ignore it (6-D format). Trajectories themselves always return full info.
"""

import numpy as np


def hover_at(point, yaw=0.0):
    """Constant position target."""
    p_const = np.asarray(point, dtype=float)
    yaw_const = float(yaw)

    def traj(t):
        return p_const, np.zeros(3), np.zeros(3), yaw_const

    return traj


def step_to(p_from, p_to, t_step, yaw=0.0):
    """Hold p_from until t_step, then hold p_to."""
    p_from = np.asarray(p_from, dtype=float)
    p_to = np.asarray(p_to, dtype=float)
    yaw_const = float(yaw)

    def traj(t):
        if t < t_step:
            return p_from, np.zeros(3), np.zeros(3), yaw_const
        return p_to, np.zeros(3), np.zeros(3), yaw_const

    return traj


def circle(center_xy, radius, period, altitude, yaw=0.0):
    """Constant-altitude circle in the XY plane.

    p(t) = (cx + r·cosθ, cy + r·sinθ, z)   with θ = (2π/T)·t
    v(t) = r·θ̇ · (−sinθ, cosθ, 0)
    a(t) = −r·θ̇² · (cosθ, sinθ, 0)
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    z = float(altitude)
    omega = 2.0 * np.pi / float(period)
    yaw_const = float(yaw)

    def traj(t):
        theta = omega * t
        c, s = np.cos(theta), np.sin(theta)
        p = np.array([cx + radius * c, cy + radius * s, z])
        v = np.array([-radius * omega * s, radius * omega * c, 0.0])
        a = np.array([-radius * omega**2 * c, -radius * omega**2 * s, 0.0])
        return p, v, a, yaw_const

    return traj


# ─── Env-specific trajectories (Epoch 3) ──────────────────────────────────────


def traverse_line(p_start, p_end, duration, yaw=0.0):
    """Smooth point-to-point with cosine-profile velocity.

    Uses s(t) = ½(1 − cos(πt/T)) blend so v(0)=v(T)=0 and a(0)=a(T)=0.
    Peak speed ≈ π·||Δp||/(2T); peak accel ≈ π²·||Δp||/(2T²).
    """
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    delta = p_end - p_start
    T = float(duration)
    yaw_const = float(yaw)

    def traj(t):
        if t >= T:
            return p_end.copy(), np.zeros(3), np.zeros(3), yaw_const
        tau = t / T
        s = 0.5 * (1.0 - np.cos(np.pi * tau))
        s_dot = 0.5 * np.pi / T * np.sin(np.pi * tau)
        s_ddot = 0.5 * (np.pi / T) ** 2 * np.cos(np.pi * tau)
        return (p_start + s * delta,
                s_dot * delta,
                s_ddot * delta,
                yaw_const)

    return traj


def s_curve_path(waypoints, segment_duration, yaw=0.0):
    """Piecewise traverse_line through a list of waypoints, each leg same duration.

    Smooth within each leg (cosine profile), continuous in position at joints
    but with zero velocity at every waypoint — fine for low-speed env demos.
    """
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    T_leg = float(segment_duration)
    n_legs = len(wps) - 1
    yaw_const = float(yaw)

    def traj(t):
        if t >= n_legs * T_leg:
            return wps[-1].copy(), np.zeros(3), np.zeros(3), yaw_const
        leg = int(t // T_leg)
        t_local = t - leg * T_leg
        leg_traj = traverse_line(wps[leg], wps[leg + 1], T_leg, yaw_const)
        return leg_traj(t_local)

    return traj


def blended_path(waypoints, radius, duration, yaw=0.0):
    """Waypoint chain with circular corner blends and ONE global speed profile.

    E4 U9 smooth-trajectory primitive (replaces per-segment traverse_line chains
    whose cosine profile forces v=0 at every waypoint joint — see Gen11 E4
    U8_Stop_and_Go analysis).

    Geometry: straight segments between waypoints; each interior corner is cut
    by a circular fillet of radius ≤ `radius` tangent to both adjacent
    segments.  Collinear corners get no fillet.  The fillet radius is clamped
    per-corner so the tangent offset d = r·tan(β/2) never consumes more than
    half of either adjacent segment (β = turn angle).

    Speed: one global cosine profile over total arc length L,
        s(t) = L·½(1 − cos(πt/T)),
    so v > 0 for all t ∈ (0, T) and v = 0 only at episode start/end.
    Peak speed = π·L/(2T) — identical to the per-segment chain with
    length-proportional time allocation, so the speed regime validated by the
    E4 rejection gates is preserved.

    On fillets the returned acceleration includes the centripetal term
    ṡ²/r (feedforward for the cascaded PID); on straights it is purely
    tangential.

    Max deviation of the fillet from the sharp corner is r·(sec(β/2) − 1)
    (= 0.414·r for a 90° corner) — clearance must be re-verified per scene
    at every blend region.
    """
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    # Drop consecutive duplicates (zero-length segments break tangent math).
    wps = [wps[0]] + [w for i, w in enumerate(wps[1:], 1)
                      if np.linalg.norm(w - wps[i - 1]) > 1e-9]
    if len(wps) < 2:
        raise ValueError('blended_path needs ≥ 2 distinct waypoints')
    T = float(duration)
    r_max = float(radius)
    yaw_const = float(yaw)

    seg_vecs = [wps[i + 1] - wps[i] for i in range(len(wps) - 1)]
    seg_lens = [float(np.linalg.norm(v)) for v in seg_vecs]
    seg_dirs = [v / l for v, l in zip(seg_vecs, seg_lens)]

    # Per-interior-corner fillet parameters (None → collinear, no fillet).
    fillets = []
    for i in range(1, len(wps) - 1):
        u_in, u_out = seg_dirs[i - 1], seg_dirs[i]
        cos_b = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
        beta = float(np.arccos(cos_b))
        if beta < 1e-6:
            fillets.append(None)
            continue
        if beta > np.pi - 0.1:
            raise ValueError(f'blended_path: near-reversal corner at waypoint '
                             f'{i} (turn {np.degrees(beta):.1f}°) cannot be blended')
        d = min(r_max * np.tan(beta / 2.0),
                0.5 * seg_lens[i - 1], 0.5 * seg_lens[i])
        r = d / np.tan(beta / 2.0)
        n = (u_out - cos_b * u_in)
        n = n / np.linalg.norm(n)          # unit normal toward turn side
        p1 = wps[i] - d * u_in             # fillet entry tangent point
        center = p1 + r * n
        fillets.append({'p1': p1, 'p2': wps[i] + d * u_out, 'center': center,
                        'r': r, 'beta': beta, 'u0': u_in,
                        'e_r0': (p1 - center) / r})

    # Build ordered path elements: (length, eval(s_local) -> (p, tangent, curv)).
    def _line(a, u):
        return lambda s: (a + s * u, u, np.zeros(3))

    def _arc(c, r, e_r0, u0):
        def ev(s):
            phi = s / r
            cp, sp = np.cos(phi), np.sin(phi)
            radial = cp * e_r0 + sp * u0
            return c + r * radial, -sp * e_r0 + cp * u0, -radial / r
        return ev

    elements = []           # list of (length, eval_fn)
    cursor = wps[0]
    for i in range(len(seg_dirs)):
        f = fillets[i] if i < len(fillets) else None
        seg_end = f['p1'] if f is not None else wps[i + 1]
        l = float(np.linalg.norm(seg_end - cursor))
        if l > 1e-9:
            elements.append((l, _line(cursor.copy(), seg_dirs[i])))
        if f is not None:
            elements.append((f['r'] * f['beta'],
                             _arc(f['center'], f['r'], f['e_r0'], f['u0'])))
            cursor = f['p2']
        else:
            cursor = wps[i + 1]

    lengths = np.array([e[0] for e in elements])
    cum = np.concatenate(([0.0], np.cumsum(lengths)))
    L = float(cum[-1])
    p_end = wps[-1]

    def traj(t):
        if t >= T:
            return p_end.copy(), np.zeros(3), np.zeros(3), yaw_const
        tau = t / T
        s = L * 0.5 * (1.0 - np.cos(np.pi * tau))
        s_dot = L * 0.5 * np.pi / T * np.sin(np.pi * tau)
        s_ddot = L * 0.5 * (np.pi / T) ** 2 * np.cos(np.pi * tau)
        idx = min(int(np.searchsorted(cum, s, side='right')) - 1,
                  len(elements) - 1)
        idx = max(idx, 0)
        p, tang, curv = elements[idx][1](s - cum[idx])
        v = s_dot * tang
        a = s_ddot * tang + s_dot ** 2 * curv
        return p, v, a, yaw_const

    return traj


def weave(x_range, y_amplitude, period, altitude, duration=None, yaw=0.0):
    """Sinusoidal weave in y while progressing linearly in x.

    p(t) = (x_start + v_x·t,  y_amp · sin(2π t / period),  altitude)

    Drone enters at left of x_range, exits at right; oscillates in y to
    weave between staggered obstacles.
    """
    x_start, x_end = float(x_range[0]), float(x_range[1])
    if duration is None:
        duration = period  # default: one full y-period over the x sweep
    T = float(duration)
    v_x = (x_end - x_start) / T
    omega = 2.0 * np.pi / float(period)
    A = float(y_amplitude)
    z = float(altitude)
    yaw_const = float(yaw)

    def traj(t):
        t_eff = min(t, T)
        ang = omega * t_eff
        p = np.array([x_start + v_x * t_eff, A * np.sin(ang), z])
        v = np.array([v_x, A * omega * np.cos(ang), 0.0])
        a = np.array([0.0, -A * omega**2 * np.sin(ang), 0.0])
        if t >= T:
            v[:] = 0.0
            a[:] = 0.0
        return p, v, a, yaw_const

    return traj
