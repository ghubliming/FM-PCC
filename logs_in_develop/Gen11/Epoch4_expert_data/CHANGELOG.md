# Gen11 Epoch 4 — Expert Data Collection: Changelog

**Date**: 2026-06-04
**Status**: Code complete (uncommitted) — pending cluster collection run

---

## Files touched

### Decision 2 — PID stability fix

| File | Change |
|---|---|
| `uav_naive_test/flight_controller.py` | `Kp_omega` `[10,10,2]` → `[2.5,2.5,1.0]` (one line) |
| `uav_env_test/flight_controller.py` | Same fix (both files are identical in structure) |

**Why**: Epoch 2 closure identified that `Kp_omega = [10,10,2]` causes limit-cycle instability on the s_curve scene (41% obstacle contact rate). The fix was specified in Decision 2 of `EPOCH4_EXECUTION_PLAN.md` and must be applied before any Phase 4-β trial runs to avoid corrupting ≥30% of generated trajectories.

---

### Phase 4-α — UAV-Flow statistics

| File | Change |
|---|---|
| `logs_in_develop/Gen11/Epoch4_expert_data/phase4_alpha_uavflow_stats.json` | **New** — kinematic statistics mined from all 273 UAV-Flow eval trajectories |

Key findings: median episode 38 waypoints, path length mean 9.5 m (UAV-Flow scenes are larger than our MuJoCo scenes). Velocity cannot be derived (no timestamps in JSON). Our generator targets 0.3–0.5 m/s, 0.7–1.1 m altitude.

---

### Phase 4-β — Generator pipeline

| File | Change |
|---|---|
| `uav_expert_data_collect/__init__.py` | **New** — empty package marker |
| `uav_expert_data_collect/trajectories.py` | **New** — scene-aware trajectory generators |
| `uav_expert_data_collect/generator.py` | **New** — PID trial runner + scene/homotopy/gain configuration |
| `uav_expert_data_collect/dataset_writer.py` | **New** — rollout → schema-locked episode pickle |
| `uav_expert_data_collect/collect.py` | **New** — CLI driver (scene, n_trials, seed, gain, homotopy) |
| `uav_expert_data_collect/stats_validator.py` | **New** — dataset stats comparison vs Phase 4-α targets |

#### `trajectories.py`
Imports `traverse_line`, `s_curve_path`, `weave` from `uav_env_test.trajectories` (Epoch 3). Adds three scene-specific wrappers:
- `corridor_path(homotopy, altitude, duration)` — L/C/R lateral bias through corridor
- `s_curve_scene_path(altitude, duration, y_jitter)` — parameterised 4-waypoint S-path
- `pillar_path(homotopy_seq, altitude, duration)` — explicit L/R/C homotopy through 3 pillar pairs via 8-waypoint `s_curve_path`

#### `generator.py`
- `HOMOTOPY_CLASSES`: corridor `{L,C,R}`, pillars `{(L,L,L),(L,R,L),(R,L,R),(R,R,R)}`, s_curve `{default}`, empty `{N/A}`
- `GAIN_VARIANTS`: `pid_default` (×1.0), `pid_high_gain` (×1.2 Kp), `pid_low_gain` (×0.8 Kp / ×0.9 Kd)
- `run_trial(scene, homotopy, gain_variant, seed)` — runs MuJoCo rollout; returns `None` if obstacle contact > 2%
- `sample_trial_specs(scene, n_trials, base_seed)` — generates spec dicts cycling homotopies and gains

#### `dataset_writer.py`
- Downsamples 100 Hz physics to ~33 Hz dataset (every 3rd step)
- Applies Gaussian noise `N(0, 0.02²)` to targets to thicken the data manifold (AUDIT R5)
- Action convention: `actions[t] = targets[t+1] - targets[t]` (position-delta, Decision 1)
- Saves pickles with the Decision-3 schema: `{episode_id, scene, homotopy, controller, dt, obs(T,6), actions(T-1,3), targets(T,3), obstacles, metadata}`

#### `collect.py`
- `--homotopy all` cycles all classes for the scene (default)
- Logs rejection rate; aborts if rate exceeds `--reject-limit` (default 30%)
- Saves `run_summary.json` alongside episodes

#### `stats_validator.py`
- Computes speed mean/percentiles, episode length, action Δp norm
- Prints comparison table vs Phase 4-α targets
- Saves `dataset_stats.json` to the data root

---

### Phase 4-γ — SLURM

| File | Change |
|---|---|
| `Slurm_Codes/sbatch/uav_expert_data/collect.sh` | **New** — SLURM wrapper (single job or `--array=0-3` for all 4 scenes in parallel) |

---

### Planning docs

| File | Change |
|---|---|
| `logs_in_develop/Gen11/Epoch4_expert_data/EPOCH4_EXECUTION_PLAN.md` | §11 Gap 1 corrected (trajectory factories exist in `uav_env_test/`, not `uav_naive_test/`) |

---

## Pending

- [ ] Push to cluster, smoke-test 10 trials per scene: `sbatch Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 10`
- [ ] Full Phase 4-γ collection: `sbatch --array=0-3 collect.sh all_scenes 500`
- [ ] Run `stats_validator.py` on output; confirm speed in 0.3–0.5 m/s range
- [ ] Mini-FM sanity gate (§3 of plan): train on 100 empty-scene episodes, verify RMS < 0.1 m
