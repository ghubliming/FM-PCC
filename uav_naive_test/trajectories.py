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
