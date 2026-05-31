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
