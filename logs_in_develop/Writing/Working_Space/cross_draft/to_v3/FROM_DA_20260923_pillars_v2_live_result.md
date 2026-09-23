# TO v3 — UAV-pillars (`pillars_v2`, R39): the live results are in; the number and the finding

**2026-09-23 · from the DA side.** Nothing in `v3/` touched. Analysis of record:
`Data_Analysis/DA_in_Paper/analysis/DA_20260923_pillars_v2_live.md` (script `pillars_v2_live.py`; INDEX row updated).

## What was run (as decided 23-09, live, no replay)
The four Chapter-6 avoiding cells — MeanFM and CI-MeanFM (α_end 0.2, U-Net) at nfe 1 and 2, `diffuser` and
`dpcc-t-tightened` — evaluated by the avoiding eval itself with the quadrotor as the plant, DPCC protocol (5 seeds × 3
geometries × 2 episodes = 30 flights per cell). Jobs 26151–26154. Complete, clean: 240 flights, no divergence, no cap.

## The table (`tab:uav-pillars`, geometry mean of seed means; contact = share of flights ended on a pillar)

| model | nfe | variant | S&C table | S&C air | contact | viol. steps table / air | steps table / air | ms table / air |
| :-- | --: | :-- | --: | --: | --: | :-- | :-- | :-- |
| MeanFM | 1 | `diffuser` | 0.03 | 0.10 | 0.00 | 15.5 / 19.9 | 63.4 / 62.7 | 10.3 / 10.3 |
| MeanFM | 1 | `dpcc-t-tightened` | 0.97 | 0.60 | 0.40 | 0.0 / 0.0 | 58.6 / 61.2 | 18.3 / 17.8 |
| MeanFM | 2 | `diffuser` | 0.03 | 0.10 | 0.20 | 15.5 / 19.1 | 62.7 / 61.5 | 19.3 / 19.3 |
| MeanFM | 2 | `dpcc-t-tightened` | 0.97 | 0.67 | 0.33 | 0.0 / 0.0 | 59.4 / 60.0 | 27.7 / 26.8 |
| CI-MeanFM | 1 | `diffuser` | 0.03 | 0.10 | 0.00 | 15.6 / 18.1 | 60.5 / 59.7 | 9.8 / 9.9 |
| CI-MeanFM | 1 | `dpcc-t-tightened` | 1.00 | 0.63 | 0.37 | 0.0 / 0.0 | 59.2 / 60.9 | 18.1 / 18.0 |
| CI-MeanFM | 2 | `diffuser` | 0.03 | 0.03 | 0.10 | 15.7 / 19.8 | 62.9 / 62.8 | 18.7 / 18.9 |
| CI-MeanFM | 2 | `dpcc-t-tightened` | 1.00 | 0.70 | 0.30 | 0.0 / 0.0 | 60.1 / 60.6 | 27.0 / 27.0 |

Per geometry the loss sits on *both-hard*: projected S&C 0.20–0.40 there (6–8 contacts of 10), 0.70–0.90 on the
two single-halfspace geometries.

## The finding, in the story's terms
1. Before projection the air reproduces the table: same success, same violating steps, same steps, same time.
2. After projection the declared constraint set is satisfied in every flight that flew to the end — 0 violating steps
   in 78 flights; the projector's guarantee survives the vehicle.
3. And a third of the projected flights still end on a pillar. Every failed flight is a pillar contact; in 50 of 51
   the *commanded* path itself passes the pillar within the rotor reach, so it is the plan, not the tracking. The
   pillars are not in the constraint set a geometry declares (one or two halfspaces and one keep-out disk); on the
   table the demonstrations skim them at the rod's radius and nothing ever penalised that; in the air the same
   proximity is a collision. **Satisfying the declared constraints transferred exactly; physical safety did not follow
   from it.** The models are not re-ordered by it (both lose the same amount, on the same geometry).
4. Chapter 7 material, not a claim: the pillars would have to be in the constraint set, or the map coarser than
   rod-radius-to-rotor-reach.

Wording the author set on 23-09: the chapter says the avoiding planner was evaluated again in an avoiding-like
quadrotor scene with pillars; the mechanism (env swap) is not described there.

## Figures (store, group `da`)
`fig_uav_pillars_paths` (MeanFM nfe 1) and `fig_uav_pillars_paths_cimf` (CI-MeanFM nfe 1): rows before / after
projection, columns the three geometries, ten flights per panel over that geometry's constraint set and the six
pillars; a cross = pillar contact. `fig_constraints_uav` already draws the pillars_v2 set (v3.69). Export with
`plotting/export_to_draft.py v3` once the tex references them.
