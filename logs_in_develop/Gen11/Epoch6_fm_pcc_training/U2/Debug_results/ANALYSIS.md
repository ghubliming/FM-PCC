# U2 close-out — pillars train+eval analysis

Artifact: `pillars_seed6_results.json` (eval of `pillars` seed 6, 20 trials, `fm_only`).

## Status
- **Code/pipeline: WORKING.** Train→eval→results run end-to-end, no NaN (the
  `SafeLimitsNormalizer` fix holds — eval logged `Constant data in dimension 0`
  and `2` instead of dividing by zero), outputs land under `logs/UAV_FM/...`.
- **Policy: FAILING.** `success_rate = 0.0` across all 20 trials. This is real,
  not a borderline-threshold artifact.

## Evidence (consistent across all 20 rollouts)
| Field | Value | Reading |
|-------|-------|---------|
| `min_z` | ~0.08 m | Airborne gate is `>0.2`; it's <½ that → **never takes off** |
| `final_z` | 0.08715727426298712 (identical to 15 digits every trial) | Not flying — physics settles to the same resting height regardless of seed/homotopy |
| `track_err_mean` | ~92 m | Commanded `p_des` diverges ~90 m from the drone → FM emits runaway Δp_des |
| `goal_dist` | ~6.5 m, unchanging | Never progresses toward goal (`goal_reached_rate=0`) |
| `contact_frac` | 0.0 | Only "avoids" pillars because it never moves into them |
| `fm_ms` | ~82 mean / ~89 p95 | Inference latency fine |

**Mechanism:** FM outputs oversized/wrong Δp_des → `p_des` runs away to tens of
meters → PID saturates → drone tips and drops on takeoff → grounded at z≈0.087 →
0% airborne → 0% success.

## Leads (next unit, not U2)
1. **Constant action dims.** Eval flagged action dims 0 & 2 constant for pillars.
   If 2 of 3 Δp_des channels are constant-zero in the pillars *training data*,
   the FM only learned 1 DOF → stable 3-D flight is impossible. Check the pillars
   expert dataset first.
2. **Checkpoint was step 80000, not 100000** — verify train fully completed;
   could be undertrained.
3. **A/B isolation:** eval `empty` (trained clean, no constant dims). If `empty`
   also = 0% → systemic policy/eval issue; if it flies → pillars-data-specific.

## U2 verdict — CLOSED
U2's scope (consolidated submit logic, wandb/GPU-leak parity, `logs/UAV_FM/` path,
NaN normalizer fix, working train→eval→aggregate on per-scene models) is **done
and verified**: the pipeline runs clean. The remaining issue is **policy quality**,
not pipeline plumbing — handed to a follow-up unit per the leads above.
