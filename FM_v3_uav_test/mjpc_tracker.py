"""MJPC optimal-control thrust tracker for the UAV (Epoch 8).

Drop-in alternative to `uav_env_test.flight_controller.CascadedPID`: same `.compute(...)`
signature, so `rollout_one`'s inner physics loop calls either polymorphically. Where the
PID converts (p_des, v_des) → 4 motor thrusts via cascaded position/attitude PD, this
hands a POSITION GOAL to MuJoCo MPC, which optimizes the 4 motor thrusts over a short
physics-rollout horizon — inferring the required velocity profile internally from the
MuJoCo state (NOT from the FM tensor). This is the FM→MJPC "IK-style UAV" path.

  Design basis : logs_in_develop/Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/U4_cond/
                 DESIGN_control_chain_arm_vs_UAV.md §6 (IK-style UAV) + §4 (MJPC API)
  Plan         : logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/PLAN_MJPC_Thrust_Control.md §4

This mirrors the PROVEN driving loop from the repo's own example
`mujoco_mpc/python/mujoco_mpc/demos/agent/cartpole.py` (set_state → N× planner_step →
get_action), rather than inventing one — only the goal-setting (mocap_pos = p_des) and the
PID-compatible signature are adapted.

MJPC API used (mujoco_mpc/python/mujoco_mpc/agent.py + cartpole.py L84-101):
  Agent(task_id, model)               — spawn the gRPC planner server for `task_id`
  set_state(qpos, qvel, mocap_pos=…)  — set drone state + the position GOAL (mocap_pos)
  planner_step()  ×N                  — run the sampling optimizer N times (cartpole: N=10)
  get_action()                        — latest 4 motor thrusts

OPEN QUESTION (see PLAN §4.3): the stock 'Quadrotor' task carries its own gate waypoints
+ TransitionLocked auto-advance. For a PURE tracker we override `mocap_pos = p_des` every
step; if the stock task re-advances its own goal, point `mjpc_task_id` at a minimal
position-tracking task instead. This module does NOT build that task — it consumes whatever
`task_id` is configured. Cluster-only (no mujoco_mpc in the Docker dev env); syntax-checked here.
"""

import time
import numpy as np


class MJPCTracker:
    """Wrap a MuJoCo MPC agent as a position→thrust tracker with a PID-compatible API."""

    def __init__(self, model, scene='', task_id='Quadrotor',
                 n_trajectories=16, horizon=0.3, planner_steps=10):
        """
        Args:
          model         : mujoco.MjModel of the UAV scene (same one the eval steps).
          scene         : scene name (diagnostics only).
          task_id       : MJPC task identifier (must exist in the compiled agent_server).
          n_trajectories: sampling fan size (tune for the 33 Hz budget — PLAN §4.4).
          horizon       : MJPC prediction horizon in seconds (tune for budget).
          planner_steps : planner_step() iterations per compute() — the sampling optimizer
                          needs several passes to converge (cartpole.py uses 10). Lower for
                          the real-time budget (PLAN §4.4).
        """
        self.scene = scene
        self.task_id = task_id
        self.n_trajectories = int(n_trajectories)
        self.horizon = float(horizon)
        self.planner_steps = int(planner_steps)
        self.last_plan_ms = 0.0          # wall-time of the last compute()'s planning (real-time logging)

        # Lazy import — mujoco_mpc is a cluster-only dependency (gRPC + compiled server).
        try:
            from mujoco_mpc import agent as agent_lib
        except ImportError as e:                              # pragma: no cover - cluster-only
            raise RuntimeError(
                'MJPCTracker: mujoco_mpc Python package not found. '
                'Ensure PYTHONPATH includes FM-PCC/third_party/mujoco_mpc. '
                'Also place the compiled agent_server binary at '
                'third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server '
                '(see third_party/mujoco_mpc/mujoco_mpc/mjpc/PLACE_BINARY_HERE.txt).'
            ) from e

        import subprocess, sys
        self.agent = agent_lib.Agent(
            task_id=task_id, model=model,
            subprocess_kwargs={'stdout': sys.stdout, 'stderr': sys.stderr},
        )

        # Push sampling knobs into the planner if the task exposes them (best-effort —
        # these are the same names as quadrotor/task.xml's <custom> numerics).
        try:
            self.agent.set_task_parameters({
                'sampling_trajectories': float(self.n_trajectories),
                'agent_horizon': self.horizon,
            })
        except Exception as exc:                              # pragma: no cover - cluster-only
            print(f'[ MJPCTracker ] set_task_parameters skipped ({exc}); using task defaults')

    def compute(self, p, q, v, om, p_des, v_des=None, a_des=None, yaw_des=0.0):
        """Return 4 motor thrusts that drive the drone toward p_des.

        Signature mirrors CascadedPID.compute so the eval inner loop is controller-agnostic.
        v_des / a_des / yaw_des are accepted for API parity but UNUSED — MJPC recovers the
        velocity profile internally from the MuJoCo state (the whole point of E8). The drone's
        real velocity enters MJPC via set_state(qvel=…), never through the FM tensor.
        """
        qpos = np.concatenate([np.asarray(p, float), np.asarray(q, float)])     # 7D free joint
        qvel = np.concatenate([np.asarray(v, float), np.asarray(om, float)])    # 6D
        # ── mirror cartpole.py L85-101: set_state → N× planner_step → get_action ──
        # Goal is set via mocap_pos = p_des (the quadrotor task's position residual targets it).
        self.agent.set_state(qpos=qpos, qvel=qvel, mocap_pos=np.asarray(p_des, float))
        t0 = time.perf_counter()
        for _ in range(self.planner_steps):                   # run planner for num_steps (proven)
            self.agent.planner_step()
        self.last_plan_ms = (time.perf_counter() - t0) * 1e3
        u = np.asarray(self.agent.get_action(), dtype=float)  # latest 4 motor thrusts
        return u[:4]

    def close(self):
        """Release the gRPC agent server (kills the subprocess)."""
        try:
            self.agent.close()
        except Exception:                                      # pragma: no cover - cluster-only
            pass
