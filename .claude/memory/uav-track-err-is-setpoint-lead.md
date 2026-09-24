---
name: uav-track-err-is-setpoint-lead
description: UAV eval's track_err (|p_des − p|, ~0.3 m) is mostly the setpoint's LEAD along the path, not sideways error — attribute violations with cross-track to the commanded path and corner/wall clearance of commanded vs flown paths (npz obs = [p_des | p])
metadata:
  type: project
---

Found 2026-09-24 (R44 s-curve, `DA_in_Paper/analysis/DA_20260924_scurve_R44bc_projection_controller.md` §4): the
evaluation's `track_err_mean` = mean |p_des − p| ≈ 0.30 m on every flow model, but the flown path stays **~1 cm**
(mean cross-track, ≤ 11 cm at the corner) from the polyline of commanded setpoints under the cascaded geometric
controller. The s-curve violations came from the **commanded path** cutting the second inside corner (8–19 cm inside
the keep-out on every flight). My first reading ("tracking error 0.30 m > route clearance 0.121 m ⇒ violations") was
wrong and had to be corrected in v3.

**Why:** a same-step setpoint-vs-position check also misleads. The setpoint leads by ~0.3 m, so the command violates on
different steps than the vehicle (lag), and "setpoint clean at the violating step" looks like tracking error.

**How to apply:** for any UAV violation attribution use `scurve_r44_raw.py::paths()`: cross-track to the commanded
polyline, and closest approach of the commanded path vs the flown path to each constraint. Default projection binds
the plan's measured p (DPCC); only `-pdes` variants bind the commanded p_des (`eval_mix_uav.py:1400–1418`). Related:
[[uav-eval-deterministic]], [[da-requires-csv-never-from-logs]].
