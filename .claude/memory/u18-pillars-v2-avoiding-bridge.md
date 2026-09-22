---
name: u18-pillars-v2-avoiding-bridge
description: "Gen15 U18 pillars_v2 (Sep 2026): D3IL-avoiding planners executed by the quadrotor in a 10x similarity-scaled scene via uav_avoiding_bridge/; Mode T replays stored obs_all (no NN/NLP, CPU), Mode L = env switch in the five avoiding evals; user gave full authority; not run yet"
metadata:
  type: project
---

**U18 `pillars_v2` (2026-09-22, replaces the abandoned U17 idea).** Reuse ALL trained avoiding models + their
DPCC/HardFlow projectors + the three test-time geometries + scorer unchanged; swap only the plant: quadrotor +
CascadedPID in `scene_avoiding_pillars_s10.xml` (avoiding field × 10, six 2 m pillars; X = s(y_a−0.035),
Y = −s(x_a−0.5), z = 1.0). Code: `uav_avoiding_bridge/{frame,scene,plant,scoring,turbo,factory}.py`;
driver `Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh` (dry-run unless GO=1). Plan + changelog in
`logs_in_develop/Gen15/U18/`. The user said "you fully have the authority on this U18".

**Why 10×, not the user's 2–3×:** the X2 spans 0.62 m; demonstrated gap clearance is 0.05 avoiding-units →
s·0.05 − 0.31 m of tracking slack (negative below 8×). Each geometry = 1–2 halfspaces + ONE keep-out disk (0.08),
NOT a disk per pillar; physical pillars are avoided by the learned behaviour.

**How to apply:**
- Mode T (turbo) = the paper grid candidate: every avoiding npz has `obs_all`; replay its setpoints, rescore on
  the drone path, write `<eval>_msguavpv2s10turbo/…` (same keys, `src_*` kept, `avg_time` copied). Open loop —
  say so; Mode L subset (FM K20 + MeanFM K2, seed 6) is the closed-loop check. Gates G1–G3 in the plan §4.
- Mode L: `FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=<tag containing uav>`; factory refuses otherwise (results share
  the Panda layout).
- Nothing in `mix_uav*` / `uav_projection.yaml` changed; old `pillars_hg` stays the section of record until G2
  passes; do not notify v3 before that.

Related: [[u17-pillars-xl-floor-result]], [[slurm-sbatch-is-real-entrypoint]], [[da-requires-csv-never-from-logs]]
