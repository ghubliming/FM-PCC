"""Gen15 U18 — `pillars_v2`: the D3IL-avoiding planner executed by the quadrotor.

The avoiding models, normaliser, DPCC/HardFlow projectors, geometries and scorer are reused unchanged
(everything stays in the avoiding frame). Only the *plant* changes: a quadrotor with the existing
CascadedPID tracks the 2-D setpoint in a MuJoCo scene that is the avoiding obstacle field scaled by
`frame.SCALE` and extruded into pillars. Two modes:

  Mode T (turbo, `turbo.py`)   replay the STORED avoiding executions (`obs_all` of any results npz) through the
                               plant and rescore on the drone's path — no network, no NLP, CPU only.
  Mode L (live, `factory.py`)  the five avoiding eval scripts get the plant instead of the Panda env when
                               FMPCC_AVOIDING_PLANT=uav (default path byte-identical to before).

Plan: logs_in_develop/Gen15/U18/PLAN_20260922_U18_pillars_v2_avoiding_bridge.md
"""
