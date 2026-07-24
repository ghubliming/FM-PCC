# Archived — mujoco_mpc gRPC/C++ client (Fix_5, abandoned 2026-07-01)

The original UAV MJPC tracker used a gRPC client (`agent.py`, `mjpc_parameters.py`,
`proto/`) talking to a compiled C++ `agent_server` binary (`mjpc/PLACE_BINARY_HERE.txt`
was its placeholder slot).

**Why abandoned:** three-layer deployment failure on the Slurm cluster —
`ModuleNotFoundError`, `agent_server not found`, then a SIGSEGV inside the C++ gRPC
server that was undebuggable without node shell access. See
`logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/U6_rebuild_mujoco_MPC/CHANGELOG_U6_mjx_tracker.md`.

**Replaced by:** `third_party/mujoco_mpc/mujoco_mpc/mjx/predictive_sampling.py` — a pure
Python/JAX implementation (DeepMind's own `mjx.predictive_sampling`, used verbatim) that
runs directly in-process via `mujoco.mjx`, no gRPC/subprocess/binary. Used by
`FM_v3_uav_test/mjpc_tracker.py`.

Kept here (not deleted) in case the gRPC path is ever revisited on a cluster where the
binary can be debugged properly.
