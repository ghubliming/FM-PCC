# DA — 2026-09-23 · UAV-pillars (`pillars_v2`, Gen15 U18): the avoiding planner flown live by the quadrotor

**Analysis of record for `sec:res:uav:pillars` (R39).** Corpus:
`Data_Analysis/analysis_results_checkpoint/23-09-UAV-Pillars-v2/live_p23uavpv2live` — jobs 26151–26154, tag
`p23uavpv2live`, the four Chapter-6 avoiding cells (MeanFM and CI-MeanFM α_end 0.2, U-Net, nfe 1 and 2), variants
`diffuser` and `dpcc-t-tightened`, DPCC protocol (seeds 6–10 × 3 geometries × 2 episodes = 30 flights per cell, 240
in all), evaluated by the avoiding eval scripts themselves with the quadrotor as the environment
(`uav_avoiding_bridge/`, scale 36, clock 1 Hz, no feed-forward). Table reference: the same cells on the table,
`19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703` (`tab:avoiding-dpcc-protocol`).
Script: [`pillars_v2_live.py`](pillars_v2_live.py). Runbook: `data_status/SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md` §3.

> **Verdict.** The runs are complete and clean (240 flights, no divergence, no step-cap end). **Before projection the air
> reproduces the table** (S&C 0.03–0.10 vs 0.03; the same violating steps; the same steps to the line). **After projection
> the declared constraint set is satisfied in every flight that flew to the end — 0 violating steps in 78 flights — yet
> S&C falls from 0.97–1.00 on the table to 0.60–0.70 in the air.** Every one of the 51 failed flights is a **physical
> pillar contact**: the drone's last position lies within its rotor reach of a pillar surface, and in 50 of 51 the
> planner's *commanded* path itself passes within that reach, so perfect tracking would not have saved it. The pillars
> are not in the constraint set the projection enforces (a geometry declares one or two halfspaces and one keep-out
> disk), and the demonstrations skim them at the rod's radius, so on the table nothing ever penalised that proximity;
> in the air the same proximity is a collision. **The finding: satisfying the declared constraints transferred exactly;
> physical safety did not follow from it.** This is a result of the scene as designed (rod radius mapped onto the rotor
> reach), not a bug: see §4.

## 1 · The table cells in the air (the thesis table)

Aggregation as on the table: mean per geometry over the five seeds, then mean over the three geometries. *steps* =
steps to the line on successful flights; *ms* = planner time per control step (network + projector; the plant is not
timed). *contact* = share of flights ended by a pillar contact.

| model | nfe | variant | S&C table | **S&C air** | success air | contact | viol. steps table / air | steps table / air | ms table / air |
| :-- | --: | :-- | --: | --: | --: | --: | :-- | :-- | :-- |
| MeanFM | 1 | `diffuser` | 0.03 | 0.10 | 1.00 | 0.00 | 15.5 / 19.9 | 63.4 / 62.7 | 10.3 / 10.3 |
| MeanFM | 1 | `dpcc-t-tightened` | 0.97 | **0.60** | 0.60 | 0.40 | 0.0 / 0.0 | 58.6 / 61.2 | 18.3 / 17.8 |
| MeanFM | 2 | `diffuser` | 0.03 | 0.10 | 0.80 | 0.20 | 15.5 / 19.1 | 62.7 / 61.5 | 19.3 / 19.3 |
| MeanFM | 2 | `dpcc-t-tightened` | 0.97 | **0.67** | 0.67 | 0.33 | 0.0 / 0.0 | 59.4 / 60.0 | 27.7 / 26.8 |
| CI-MeanFM | 1 | `diffuser` | 0.03 | 0.10 | 1.00 | 0.00 | 15.6 / 18.1 | 60.5 / 59.7 | 9.8 / 9.9 |
| CI-MeanFM | 1 | `dpcc-t-tightened` | 1.00 | **0.63** | 0.63 | 0.37 | 0.0 / 0.0 | 59.2 / 60.9 | 18.1 / 18.0 |
| CI-MeanFM | 2 | `diffuser` | 0.03 | 0.03 | 0.90 | 0.10 | 15.7 / 19.8 | 62.9 / 62.8 | 18.7 / 18.9 |
| CI-MeanFM | 2 | `dpcc-t-tightened` | 1.00 | **0.70** | 0.70 | 0.30 | 0.0 / 0.0 | 60.1 / 60.6 | 27.0 / 27.0 |

Per geometry, projected (S&C table / air, contacts of 10):

| model | nfe | top-right-hard | top-left-hard | both-hard |
| :-- | --: | :-- | :-- | :-- |
| MeanFM | 1 | 0.90 / 0.90 (1) | 1.00 / 0.70 (3) | 1.00 / **0.20** (8) |
| MeanFM | 2 | 0.90 / 0.80 (2) | 1.00 / 0.80 (2) | 1.00 / **0.40** (6) |
| CI-MeanFM | 1 | 1.00 / 0.80 (2) | 1.00 / 0.90 (1) | 1.00 / **0.20** (8) |
| CI-MeanFM | 2 | 1.00 / 0.90 (1) | 1.00 / 0.90 (1) | 1.00 / **0.30** (7) |

## 2 · What the failures are

- 51 of 240 flights failed; **51 are pillar contacts**, 0 hit the 200-step cap, 0 diverged (plant sidecars agree:
  `ended = contact` 51, `success` 189).
- Projected flights: **0 violating steps against the declared set in all 78 that flew to the end**; the 42 that failed
  did so on a pillar, with 0 declared violations up to the contact.
- Where: on *both-hard* the contacts are on the third row (`l3_top` 16, `l3_bottom` 13 of 29): the two halfspaces
  funnel the plan toward the finish line's centre and it passes the outer third-row pillars at the rod's radius. On
  the single-halfspace geometries the contacts are on the second row (`l2_top` / `l2_bottom`, 6 each), where the
  keep-out disk sits and the plan rounds the neighbouring pillar.
- The commanded path of a failed projected flight passes the nearest pillar at 0.0076 units (median; max 0.0101);
  a surviving one at 0.028 (median; 5th percentile 0.0114). The rotor reach is 0.0100 units — the rod's radius on the
  table (0.010) by construction of the scale.
- Unprojected: 9 contacts of 120, all on `l3_bottom`, the same skimming.
- The plant: tracking error 0.18 m mean (0.27 m at the 95th percentile of per-flight p95), setpoint gap 0.0066 units
  (the table's own gap was 0.02–0.04), |v| ≤ 0.61 m/s. The lag is not what decides a contact (50 of 51 commanded paths
  are already inside the reach), but it is the residue that turns the one borderline case.

## 3 · Reading for the thesis

1. **Before projection the transfer is exact.** Same success, same violating steps, same steps to the line, same
   planner time. The plant adds nothing and removes nothing.
2. **After projection the declared constraints are satisfied exactly in the air** — the projector's guarantee
   survives the vehicle — **and the scene still fails a third of the flights on physics the constraint set does not
   contain.** On the table those pillars are the visible obstacles of the avoiding task; the demonstrations pass them
   at the rod's own radius and the benchmark's failure signal never fired for that. Put the same paths in the air and
   the vehicle's reach makes them collisions.
3. The scene is therefore **the physical-safety case beside corridor's repair case**: it shows what a declared
   constraint set does and does not buy. It does not re-order the models (MeanFM and CI-MeanFM lose the same
   0.3–0.4, on the same geometry, at both budgets).
4. What would close the gap is a scene-design statement, not a run: the pillars would have to be in the constraint
   set (six keep-out disks at pillar radius plus reach), or the scale larger than the rod-to-reach map. Neither is
   claimed here; both are Chapter 7 material.

## 4 · Why this is not our fault (checked)

| check | result |
| :-- | :-- |
| every failed flight ends within the reach of a pillar (0.0096–0.0100 units for the projected ones) | 51 / 51 |
| commanded path of the failed flight itself within the reach | 50 / 51 |
| step-cap or divergence ends | 0 |
| declared-constraint scoring in the air = the table's scorer (the eval's own code, unchanged) | yes: 0 violating steps on every surviving projected flight, 15–20 on the unprojected ones as on the table |
| the plant's contact rule | any drone–pillar MuJoCo contact ends the flight, the rule of every UAV scene in the thesis |
| the map | rod radius 0.01 → rotor reach 0.36 m (scale 36); the constraint set is the avoiding one, unmapped |
| the same plant on FM nfe 20 (job 26077, U18 L1) | S&C within 0.05 of the table on every cell, 0 contacts — the difference here is the plans, not the vehicle |

## 5 · Figures

`fig_uav_pillars_paths` (MeanFM nfe 1) and `fig_uav_pillars_paths_cimf` (CI-MeanFM nfe 1): rows before / after
projection, columns the three geometries, ten flights per panel over that geometry's constraint set and the six
pillars; a cross marks a pillar contact. Built from `data/uav_paths.json` (`extract/pillars_v2_paths.py`) by
`builders/paths.py`, store group `da`.
