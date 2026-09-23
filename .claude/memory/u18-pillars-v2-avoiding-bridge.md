---
name: u18-pillars-v2-avoiding-bridge
description: "Gen15 U18 pillars_v2 (Sep 2026): D3IL-avoiding planners executed by the quadrotor in a 36x similarity-scaled scene (rod radius 0.01 -> drone reach 0.36) via uav_avoiding_bridge/; Mode T replays stored obs_all (no NN/NLP, CPU), Mode L = env switch in the five avoiding evals; user gave full authority; not run yet"
metadata:
  type: project
---

**U18 `pillars_v2` (2026-09-22, replaces the abandoned U17 idea).** Reuse ALL trained avoiding models + their
DPCC/HardFlow projectors + the three test-time geometries + scorer unchanged; swap only the plant: quadrotor +
CascadedPID in `scene_avoiding_pillars_s10.xml` (avoiding field × 10, six 2 m pillars; X = s(y_a−0.035),
Y = −s(x_a−0.5), z = 1.0). Code: `uav_avoiding_bridge/{frame,scene,plant,scoring,turbo,factory,overview_plot}.py`;
driver `Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh` (dry-run unless GO=1). Plan + changelog in
`logs_in_develop/Gen15/U18/`. The user said "you fully have the authority on this U18".

**Why 36× (fix3, pilot 26073):** the Panda paths hug obstacles at the rod radius (0.009–0.02 units); the X2's
radial reach is 0.36 m → faithful scale 0.36/0.01 = 36. At 10× the drone hit pillars in 15–18/20 episodes with
perfect tracking (settle mode). Each geometry = 1–2 halfspaces + ONE keep-out disk (0.08), NOT a disk per pillar.
**Clock mode:** velocity feed-forward flips the CascadedPID above ~0.5 m/s of v_des (motor saturation via the
attitude loop) → ff OFF, rate-limited reference 1 m/s, 1 Hz; settle mode is the geometric bound.

**How to apply:**
- Mode T (turbo) = the paper grid candidate: every avoiding npz has `obs_all`; replay its setpoints, rescore on
  the drone path, write `<eval>_msguavpv2s10turbo/…` (same keys, `src_*` kept, `avg_time` copied). Open loop —
  say so; Mode L subset (FM K20 + MeanFM K2, seed 6) is the closed-loop check. Gates G1–G3 in the plan §4.
- Mode L: `FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=<tag containing uav>`; factory refuses otherwise (results share
  the Panda layout).
- Nothing in `mix_uav*` / `uav_projection.yaml` changed; old `pillars_hg` stays the section of record until G2
  passes; do not notify v3 before that.

**Paper runs (PENDING_20260923 §2, R39) — author 23-09 FINAL: real eval only, no turbo.** The thesis reads the four
Ch 6 cells (MeanFM, CI-MeanFM α0.2 × K1, K2; `diffuser` + `dpcc-t-tightened`) evaluated LIVE, seeds 6–10 × 3 geos × 2 ep,
via `Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh` → `sbatch/uav_avoiding_bridge/live_p23_pillars.sh`, tag
`p23uavpv2live`. **The run tag must contain `uav`** (factory refuses) — `live_p2_*.sh` default `p23pv2live` is broken/superseded.
Turbo (`turbo.sh MODE=paper`) is not read by the thesis. L1 check (26077) showed live = Panda within 0.05.

Related: [[u17-pillars-xl-floor-result]], [[slurm-sbatch-is-real-entrypoint]], [[da-requires-csv-never-from-logs]]
