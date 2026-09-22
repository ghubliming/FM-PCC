# CHANGELOG — Gen15 U18 · fix5 (2026-09-22) · Mode L check (L1) wired: the planner in the loop with the drone

**Question it answers.** Turbo replays plans made on the table; a live eval makes new plans while watching the drone.
They agree only if the drone stays where the arm was. Decision rule (author, 2026-09-22): run L1; if it matches turbo
within the seed spread, turbo's grid is the table; if not, the live full eval (groups A–E) replaces turbo.

**L1 = the turbo pilot cell, closed loop.** FM K20, seed 6, 3 geometries, `diffuser` + `dpcc-r-tightened`, 20 trials
with the same torch trial seeds as the Panda's `msg20trials` cell; plant = quadrotor at 36×, clock 1 Hz, no
feed-forward, v_max 1 m/s (the fix3/4 defaults). GPU (the network runs every step), ≈ 1.5 h.

| file | change |
| :-- | :-- |
| `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | `FMPCC_PROJ_CFG` selects the projection yaml (default `config/projection_eval.yaml`, unchanged behaviour) — the override `config/avoiding-d3il.py` already documents |
| `config/projection_eval_u18_live.yaml` | `projection_eval.yaml` with seeds [6], n_trials 20, variants [diffuser, dpcc-r-tightened]; constraints identical |
| `uav_avoiding_bridge/factory.py` | Mode L defaults = fix3/4 (1 Hz, ff off, v_max 1.0); `FMPCC_AVOID_UAV_SIDECAR_DIR` dumps the plant's per-episode records on `env.close()` |
| `Slurm_Codes/sbatch/uav_avoiding_bridge/live_l1.sh` | the GPU driver (dry-run unless `GO=1`); results in the avoiding tree under `…_msguavpv2s36live20/6/results/`, sidecar under `UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/` |
| `uav_avoiding_bridge/compare_live_turbo.py` | per-cell Panda / turbo-clock / turbo-settle / live means + paired per-episode agreement; numpy only, runs locally |

```bash
GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_l1.sh
# afterwards, on the downloaded folders:
python uav_avoiding_bridge/compare_live_turbo.py --live <…_msguavpv2s36live20/6/results> \
       --turbo <…-uavpv2s36turbo/6/results> --turboset <…-uavpv2s36turboset/6/results>
```

## Where live and turbo can diverge (what to look at in the comparison)

1. **Feedback on a different position.** The planner conditions on `[x_des, y_des, x, y]`; live, `x, y` is the
   drone (lag ≈ 0.2 m world = 0.006 units in clock mode) instead of the arm (its own gap 0.03–0.13). The drone is
   *closer* to the setpoint than the arm was, so this moves the observation toward the setpoint, inside the data.
   Expect small, systematic differences in the plans, not a regime change.
2. **Random candidate selection.** `dpcc-r` picks a random candidate each step; once positions differ by anything
   the two runs draw different candidates and the episodes decorrelate. Episode-level agreement will therefore be
   partial even when the statistics agree — read the cell means, use the paired view only for gross effects.
3. **Different episode ends.** Turbo ends where the stored sequence ends (+ momentum grace); live ends where the
   drone crosses the line, hits a pillar, or the 200-step cap — steps-to-goal can shift by a few steps.
4. **Contact.** Same rule both modes (drone reach 0.36 m against physical pillars); a live planner might route a
   hugging path slightly differently and avoid — or find — a contact the replay had.
5. **Timing.** `avg_time` live includes the same network + projector as on the table; the plant is not counted.
   Should match the table's column.

A strong divergence would be: live S&C differing from turbo by more than the Panda's seed spread on the same cell,
or a systematic shift in violating steps. Then groups A–E run live and turbo is dropped.
