# CHANGELOG — Gen15 U18 · fix4 (2026-09-22) · pilot 26076 (36×) read: gate G1 passed, three replay details fixed

Pilot 26076: scale 36, FM K20 `msg20trials`, seed 6, 3 geometries × {`diffuser`, `dpcc-r-tightened`}, 20 episodes,
`clock` (1 Hz, v_max 1.0, no feed-forward) and `settle`. Log + run JSONs in [`pilot_s36/`](pilot_s36/). Outputs on the
cluster under `logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/` (fix2 layout), full suite per cell.

## Gate G1 — passed

| cell (settle) | Panda succ / S&C | drone succ / S&C | agree | ended | track_err | gap_a p95 (Panda) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| both-hard `diffuser` | 1.00 / 0.05 | 1.00 / 0.00 | 19/20 | 20 success | 0.05 m | 0.0014 (0.027) |
| both-hard `dpcc-r-t` | 1.00 / 1.00 | 0.85 / 0.85 | 17/20 | 17 success, 3 exhausted | 0.05 m | 0.0014 (0.041) |
| top-left `diffuser` | 1.00 / 0.05 | 1.00 / 0.00 | 19/20 | 20 success | 0.05 m | 0.0014 (0.027) |
| top-left `dpcc-r-t` | 1.00 / 0.95 | 0.70 / 0.70 | 14/20 | 14 success, 1 contact, 1 exhausted, 4 diverged | 0.05 m | 0.0014 (0.135) |
| top-right `diffuser` | 1.00 / 0.00 | 1.00 / 0.00 | 20/20 | 20 success | 0.05 m | 0.0014 (0.027) |
| top-right `dpcc-r-t` | 0.95 / 0.95 | 0.90 / 0.90 | 19/20 | 18 success, 1 contact, 1 exhausted | 0.05 m | 0.0014 (0.034) |

Clock mode: identical outcomes, track_err mean 0.16–0.19 m, gap_a p95 0.007–0.009, |v| ≤ 0.63 m/s, no PID blow-up.
The scale is right (no contact except the two obstacle-hugging paths), the drone's setpoint gap is far below the
Panda's own, and the unprojected rows reproduce the Panda's violations. **The plant is validated; Mode L (planner
in the loop) is not needed to validate the environment — it stays the optional closed-loop check of §5.**

## What the residual disagreements were (all replay details, fixed here)

| symptom | cause | fix |
| :-- | :-- | :-- |
| 3 + 1 + 1 `exhausted`: drone parked at y_a = 0.345–0.349, finish 0.35 | the Panda arm crosses the line with the momentum of its last commanded step (its gap p95 is 0.03–0.04); the grace **hold** parks the drone exactly on the last setpoint | `--grace-mode extend` (default): the grace period continues the last commanded increment; `hold` kept as option |
| 4 `diverged` (`left_arena`) in top-left `dpcc-r-t` | those Panda paths go around the keep-out on the far left, outside the plotted field (x_a < 0.2); the arena guard (field + 2 m) aborted them although the Panda finished | `DIV_ENVELOPE_SLACK_M` 2 → 8 m (the speed guard is what catches PID blow-ups) |
| foresight SVGs after the first cell drew a flat altitude path | the plant's episode counter ran across cells, so the sidecar record lookup missed | `plant.episode` reset per cell |

`diffuser` drone violations 16.4 vs Panda 12.2: the drone tracks the **setpoints** (which are what violate) while the
Panda's lagging end-effector stayed a little further inside — the open-loop caveat, recorded, not a bug.

## Consequence for the running `MODE=all` job

It started on pre-fix4 code: its cells carry the at-the-line misses and the far-left aborts. Let it finish (its
numbers are a valid first look), then re-run with the fixes:
```bash
GO=1 MODE=all EXTRA="--force" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh
```
(idempotency skips existing cells; `--force` redoes them in place — under an hour, CPU.)
