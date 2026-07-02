"""MJX predictive-sampling tracker for the UAV (Gen11 E8 U6).

Replaces the gRPC agent_server approach (C++ binary — aborted, see
logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/Fix_5_MUJOCO_MPC_pkg&ode_step/).

Uses DeepMind's predictive_sampling.py (mujoco_mpc.mjx) verbatim — zero translation.
JAX + MJX run nsample parallel physics rollouts on GPU per improve_policy() call.
No gRPC. No subprocess. No binary.

Dependencies (cluster one-time install):
    pip install "jax[cuda12]" mujoco-mjx
    (brax NOT required — mpc_rollout stripped from our predictive_sampling copy)

API is identical to the old gRPC MJPCTracker: same .compute(p, q, v, om, p_des, ...)
signature, so rollout_one in eval_fm_uav.py is unchanged.

Design ref: logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/
            Fix_5_MUJOCO_MPC_pkg&ode_step/MEMO_python_mirror_plan.md
"""

import time
import numpy as np


class MJPCTracker:
    """MJX predictive-sampling position tracker. Drop-in for CascadedPID."""

    def __init__(self, model, scene='', task_id=None,
                 n_trajectories=16, horizon=0.3, **_):
        """
        Args:
          model:          mujoco.MjModel of the UAV scene (same one the eval steps).
          scene:          scene name (diagnostics only).
          n_trajectories: parallel sampled action sequences per improve_policy() call.
          horizon:        planning horizon in seconds → converted to MuJoCo timesteps.
          task_id:        ignored (was gRPC task name, no longer used).
          **_:            absorbs deprecated kwargs (planner_steps, etc.).
        """
        try:
            import jax
            import jax.numpy as jnp

            # mujoco-mjx calls jax.extend.backend.backends() to detect CUDA devices.
            # JAX 0.5+ removed this attribute; shim it with jax.devices() which is stable.
            import jax.extend.backend as _jax_eb
            if not hasattr(_jax_eb, 'backends'):
                _jax_local = jax
                _jax_eb.backends = lambda: {d.platform for d in _jax_local.devices()}

            from mujoco import mjx
            from mujoco_mpc.mjx import predictive_sampling as ps
        except ImportError as e:
            raise RuntimeError(
                'MJPCTracker (MJX): missing dependency. '
                'Run: pip install "jax[cuda12]" mujoco-mjx\n'
                f'Original error: {e}'
            ) from e

        self._jax = jax
        self._jnp = jnp
        self._mjx = mjx
        self._ps  = ps
        self.scene = scene
        self.last_plan_ms = 0.0

        dt = float(model.opt.timestep)
        horizon_steps = max(4, int(round(horizon / dt)))
        nu = int(model.nu)

        # ── UAV position-tracking cost ────────────────────────────────────────
        # instruction = p_des (jnp array [3]) passed directly to improve_policy.
        # Cost = squared Euclidean distance to goal at every rollout step.
        def uav_pos_cost(mx_model, mx_data, p_des):
            pos = mx_data.qpos[:3]   # free-joint: qpos = [x, y, z, qw, qx, qy, qz]
            return jnp.sum((pos - p_des) ** 2), ()

        # instruction_fn only called by mpc_rollout (offline eval) — not our path.
        dummy_instruction_fn = lambda m, d: (jnp.zeros(3), jnp.zeros(0))

        mx_model = mjx.put_model(model)
        self._mx_model = mx_model
        self._planner = ps.Planner(
            model=mx_model,
            cost=uav_pos_cost,
            noise_scale=0.3,
            horizon=horizon_steps,
            nspline=horizon_steps,   # one spline knot per step (zero-order hold)
            nsample=n_trajectories,
            interp='zero',
            instruction_fn=dummy_instruction_fn,
        )
        self._policy = jnp.zeros((horizon_steps, nu))
        self._rng    = jax.random.PRNGKey(0)
        # JIT once — first compute() call takes ~5–10 s compile; afterwards ~1–5 ms on GPU.
        self._improve_jit = jax.jit(ps.improve_policy)
        print(f'[ MJPCTracker ] init: scene={scene} horizon={horizon_steps} steps '
              f'({horizon:.2f}s @ dt={dt}s) n_samples={n_trajectories} nu={nu}')
        print('[ MJPCTracker ] JIT warmup will happen on first compute() call (~5–10 s)')

    def compute(self, p, q, v, om, p_des, v_des=None, a_des=None, yaw_des=0.0):
        """Return 4 motor thrusts toward p_des.

        v_des / a_des / yaw_des accepted for API parity with CascadedPID but unused —
        MJX recovers the velocity profile from MuJoCo state internally.
        """
        jnp  = self._jnp
        mjx  = self._mjx
        ps   = self._ps

        # Build MJX data from current drone state
        qpos    = jnp.array([*p, *q], dtype=jnp.float64)   # 7D free joint
        qvel    = jnp.array([*v, *om], dtype=jnp.float64)  # 6D
        p_des_j = jnp.array(p_des, dtype=jnp.float64)

        mx_data = mjx.make_data(self._mx_model)
        mx_data = mx_data.replace(qpos=qpos, qvel=qvel)

        self._rng, rng = self._jax.random.split(self._rng)
        t0 = time.perf_counter()
        self._policy, _ = self._improve_jit(
            self._planner, mx_data, p_des_j, self._policy, rng
        )
        self.last_plan_ms = (time.perf_counter() - t0) * 1e3

        action = np.array(self._policy[0])
        self._policy = ps.resample(self._planner, self._policy, 1)
        return action[:4]

    def close(self):
        pass  # no subprocess to kill (contrast with old gRPC tracker)
