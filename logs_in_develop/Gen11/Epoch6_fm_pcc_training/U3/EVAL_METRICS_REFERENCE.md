# UAV-FM eval metrics reference (per-rollout `stats.json` / `results.json` fields)

Each closed-loop rollout writes one `diagnostics/rollout_<i>_stats.json`; the same
fields appear per-trial inside `results.json` and are aggregated into the npz / the
analyzer CSVs. Source: `FM_v3_uav_test/eval_fm_uav.py:rollout_one`.

## Field-by-field

| field | units | meaning |
|---|---|---|
| `scene` | — | which scene (`empty`/`corridor`/`s_curve`/`pillars`). |
| `homotopy` | — | the expert path-mode for this trial (e.g. `default`, `L`, `(L,R,L)`). Multi-mode scenes are where the FM fails — see the homotopy finding. |
| `success` | bool | the headline gate: **contact-free AND airborne** (`contact_frac ≤ scene limit` AND `min_z > 0.2`). NOTE: this is "flew safely", **not** "reached the goal". |
| `contact_frac` | fraction | share of physics steps in obstacle contact. `0.0` = never touched a wall/pillar. |
| `goal_dist` | m | distance from the drone's FINAL position to the expert path's endpoint (the nominal goal). |
| `goal_reached` | bool | `goal_dist < GOAL_RADIUS (0.30 m)`. A *stricter*, secondary metric — usually `false` even when `success=true`. |
| `min_z` | m | lowest altitude over the whole rollout. The airborne gate is `> 0.2`; `min_z ≈ 0.08` means it sat on the floor. |
| `final_z` | m | altitude at the last step. If `final_z ≈ min_z` and both are healthy, it held a steady height. |
| `track_err_mean` | m | mean `‖actual p − commanded p_des‖` over the rollout. **The health signal**: tiny (mm) = the drone tracked its command (on the trained "command≈actual" diagonal); huge (tens of m) = the command ran away from the drone (the explosion). |
| `fm_ms_mean` / `fm_ms_p95` | ms | FM inference latency per control step (mean / 95th pct). Compute cost only — not quality. |
| `n_fm_steps` | count | number of 33 Hz FM control steps in the rollout (episode length × 33). |
| `decim` | count | physics steps per FM query = `round(1 / (dt · 33))` ≈ 3 (100 Hz physics ÷ 33 Hz FM). |
| `dt` | s | physics timestep = `0.01` (100 Hz). |

## Reading the example (s_curve, a *healthy* trial)

```json
"success": true,  "contact_frac": 0.0,
"min_z": 0.835,   "final_z": 0.835,
"track_err_mean": 0.0046,
"goal_dist": 6.001, "goal_reached": false,
"n_fm_steps": 642
```

- `track_err_mean = 0.0046 m` (4.6 mm) → the drone tracked its commanded waypoint
  **almost perfectly**. Contrast the exploded pillars trials at `track_err_mean ≈ 92 m`.
  This single number is the cleanest healthy-vs-exploded discriminator.
- `min_z = final_z = 0.835 m` → held a steady altitude, well above the 0.2 m gate. No
  runaway, no floor crash.
- `contact_frac = 0.0` → never hit a wall.
- **So `success = true`** (safe, airborne, contact-free).

But the subtlety:
- `goal_reached = false`, `goal_dist = 6.0 m` → it did **not** end near the expert's
  endpoint. It flew a safe, well-tracked path but **did not complete the route**.

> Key takeaway: `success` here means **"flew safely"**, not **"reached the goal"**. A
> single-mode scene like s_curve produces coherent, well-tracked flight (`success=true`,
> tiny `track_err`) yet can still miss the goal (`goal_reached=false`). The state-only FM
> has no goal signal, so "complete the specific route" is not something it is being asked
> (or able) to do — `success` is the contact-free/airborne proxy. To make goal-reaching
> the headline metric, the FM must be goal-conditioned (Epoch 7).
