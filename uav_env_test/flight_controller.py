"""Cascaded PID flight controller for the Skydio X2 (Lee/Mellinger structure).

Layers (outer → inner):
  1. Position PD + feed-forward acceleration → desired world-frame force
  2. Force direction → desired body-z axis; combined with desired yaw → R_des
  3. Attitude PD on SO(3) error + gyroscopic compensation → body torques
  4. Allocation: (T, τ_x, τ_y, τ_z) → 4 motor thrusts via X2 geometry

Allocation matrix is built at __init__ from the loaded MuJoCo model:
  per-motor column = [1, r_y_i, -r_x_i, κ_i]
where (r_x_i, r_y_i) is the motor site position in body frame and κ_i is
the actuator's yaw torque coefficient (gear[5]).

omega_body is body-frame (free-joint qvel[3:6] in MuJoCo convention).
"""

import numpy as np


GRAVITY_MAG = 9.81


def quat_to_rot(q):
    """MuJoCo quaternion (w, x, y, z) → 3x3 rotation matrix (body → world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


class CascadedPID:
    def __init__(self, model,
                 motor_actuator_names=('thrust1', 'thrust2', 'thrust3', 'thrust4'),
                 motor_site_names=('thrust1', 'thrust2', 'thrust3', 'thrust4'),
                 body_name='x2'):
        body_id = model.body(body_name).id
        self.mass = float(model.body_subtreemass[body_id])
        # Inertia: diagonal in body principal frame
        self.inertia = np.asarray(model.body_inertia[body_id], dtype=float).copy()

        cols = []
        for site_name, act_name in zip(motor_site_names, motor_actuator_names):
            site_id = model.site(site_name).id
            r = np.asarray(model.site_pos[site_id], dtype=float)
            gear = np.asarray(model.actuator_gear[model.actuator(act_name).id], dtype=float)
            kappa_z = gear[5]
            cols.append([1.0, r[1], -r[0], kappa_z])
        self.M = np.column_stack(cols)
        try:
            self.M_inv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                f'CascadedPID: allocation matrix is singular. '
                f'Motor positions or yaw gear values may be degenerate. M=\n{self.M}'
            ) from e

        # Hover thrust per motor (for diagnostics + safety floor)
        self.u_hover = self.mass * GRAVITY_MAG / 4.0
        self.u_max = max(2.0 * self.u_hover, 6.0)
        self.u_min = 0.0

        # Position-loop gains (world frame)
        self.Kp_pos = np.array([4.0, 4.0, 8.0])
        self.Kd_pos = np.array([3.0, 3.0, 4.0])

        # Attitude-loop gains (body frame)
        self.Kp_att = np.array([70.0, 70.0, 4.0])
        self.Kp_omega = np.array([2.5, 2.5, 1.0])

        # Safety floor on total thrust (avoid free fall during recovery)
        self.thrust_floor = 0.1 * self.mass * GRAVITY_MAG

    def compute(self, p, q, v, omega_body,
                p_des, v_des=None, a_des=None, yaw_des=0.0):
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        omega_body = np.asarray(omega_body, dtype=float)
        p_des = np.asarray(p_des, dtype=float)
        v_des = np.zeros(3) if v_des is None else np.asarray(v_des, dtype=float)
        a_des = np.zeros(3) if a_des is None else np.asarray(a_des, dtype=float)

        # ── outer loop: position PD + feed-forward ──────────────────────────
        e_p = p - p_des
        e_v = v - v_des
        a_cmd = -self.Kp_pos * e_p - self.Kd_pos * e_v + a_des

        # Required world-frame thrust vector (force) to produce a_cmd against gravity
        F_world = self.mass * (a_cmd + np.array([0.0, 0.0, GRAVITY_MAG]))

        F_norm = np.linalg.norm(F_world)
        if F_norm < 1e-6:
            return np.full(4, self.u_hover)

        # Desired body-z axis (rotor disk normal points along required thrust)
        b3_des = F_world / F_norm

        # Desired body-x axis from yaw (project into the plane orthogonal to b3_des)
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        b2_des = np.cross(b3_des, x_c)
        b2_norm = np.linalg.norm(b2_des)
        if b2_norm < 1e-6:
            # Singular: thrust direction is parallel to x_c. Fall back to y-axis.
            b2_des = np.array([0.0, 1.0, 0.0])
        else:
            b2_des = b2_des / b2_norm
        b1_des = np.cross(b2_des, b3_des)

        R_des = np.column_stack([b1_des, b2_des, b3_des])

        # ── current rotation from quaternion ────────────────────────────────
        R = quat_to_rot(q)

        # ── inner loop: SO(3) attitude error (Lee 2010) ─────────────────────
        E = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([E[2, 1], E[0, 2], E[1, 0]])
        e_omega = omega_body  # ω_des_body = 0 here (acceptable for slow tracking)

        gyro = np.cross(omega_body, self.inertia * omega_body)
        tau = -self.Kp_att * e_R - self.Kp_omega * e_omega + gyro

        # ── total thrust scalar: project world force onto current body-z ────
        b3 = R[:, 2]
        T = float(F_world @ b3)
        if T < self.thrust_floor:
            T = self.thrust_floor

        # ── allocation: solve M u = [T; τ_x; τ_y; τ_z] ─────────────────────
        wrench = np.array([T, tau[0], tau[1], tau[2]])
        u = self.M_inv @ wrench
        u = np.clip(u, self.u_min, self.u_max)
        return u


def diagnostic_string(ctrl: CascadedPID) -> str:
    return (
        f'mass={ctrl.mass:.4f} kg  u_hover={ctrl.u_hover:.4f} N  u_max={ctrl.u_max:.2f}\n'
        f'inertia diag = {ctrl.inertia}\n'
        f'allocation M =\n{ctrl.M}\n'
        f'Kp_pos={ctrl.Kp_pos}  Kd_pos={ctrl.Kd_pos}\n'
        f'Kp_att={ctrl.Kp_att}  Kp_omega={ctrl.Kp_omega}'
    )
