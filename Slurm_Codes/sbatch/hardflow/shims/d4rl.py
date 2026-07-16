"""Lightweight `d4rl` shim for the HardFlow avoiding (d3il) replication.

WHY THIS EXISTS
---------------
HardFlow's `hardflow/datasets/d4rl.py` runs a top-level `import d4rl`, inherited
from the diffuser codebase it is built on. On the **avoiding** task d4rl is never
actually used:
  * environments come from d3il's gym registration (`gym.make("avoiding-v0")`),
  * datasets come from d3il `.pkl` files via `env.get_dataset()`,
  * grep across HardFlow shows NO `d4rl.<symbol>` call anywhere.

The real d4rl pulls in mujoco_py / dm_control / old-gym build pain and is absent
from the FMPCC clone (`hardflow_clone` is cloned from FMPCC, which has no d4rl).
Rather than install it (heavy, unnecessary) or edit HardFlow's source (upstream
stays pristine), this empty module satisfies the import. The bridge
(`_hardflow_common.sh`) puts this shims/ dir on PYTHONPATH *after* the HardFlow
repo, so it only supplies the otherwise-missing top-level `d4rl` and shadows
nothing real.

If a run ever accesses a real d4rl attribute, Python raises AttributeError from
here — that is the signal this task genuinely needs the real package.
"""
