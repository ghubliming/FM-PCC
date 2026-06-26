# Gen11 E5 U4 — Stress-Test Rendering: Changelog

**Date:** 2026-06-12
**Branch:** update_into_FM
**Status:** COMPLETE — code implemented, pending E4 U10 stress data + cluster run

---

## Files changed

| File | Change |
|------|--------|
| `uav_expert_data_collect/generate_physics_gifs.py` | Stress dispatch in `physics_replay_frames`; `$6 data_dir` |
| `uav_expert_data_collect/generate_trajectory_gifs.py` | `gifs_physics` exclusion hardening |
| `Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh` | `$6 = data_dir` |
| `Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh` | `$6 = data_dir` |

---

## C1 — `generate_physics_gifs.py`: stress dispatch

**Location:** `physics_replay_frames()`, trajectory reconstruction block.

Before this change, the function called `_build_traj_and_init(scene, homotopy, rng)`.
For a stress episode, `homotopy` is a stress-case name (e.g. `wall_crossing`) → the
call would either raise `ValueError` or silently rebuild the wrong (normal) trajectory.

**Change:**

```python
if episode.get('stress', False):
    from uav_expert_data_collect.stress_trajectories import build_stress_traj
    from uav_env_test.flight_controller import CascadedPID
    stress_case = episode.get('stress_case', homotopy)
    traj_fn, init_pos, dur, _ = build_stress_traj(stress_case, scene, rng)
    # Rebuild PID with stored gain scales (non-1.0 only for gain_extreme)
    kp_scale = episode.get('metadata', {}).get('kp_scale', 1.0)
    kd_scale = episode.get('metadata', {}).get('kd_scale', 1.0)
    if kp_scale != 1.0 or kd_scale != 1.0:
        pid = CascadedPID(model)
        pid.Kp_pos = pid.Kp_pos * kp_scale
        pid.Kd_pos = pid.Kd_pos * kd_scale
    else:
        pid = _make_pid(model, 'pid_default')
else:
    traj_fn, init_pos, dur = _build_traj_and_init(scene, homotopy, rng)
    pid = _make_pid(model, controller)
```

Seed recovery is unchanged (`int(ep_id.split('_')[-1])`). The stress episode stores
the seed in the last `_`-separated token (`stress_{case}_{scene}_{seed:07d}`).

For `gain_extreme` episodes, `kp_scale` and `kd_scale` are stored in
`episode['metadata']` by `collect_stress.py` → the physics replay uses the identical
PID gains as the original collection run.

## C2 — `generate_trajectory_gifs.py`: `gifs_physics` exclusion

```python
# BEFORE:
if scene in ('images', 'gifs'):
    continue

# AFTER:
if scene in ('images', 'gifs', 'gifs_physics'):
    continue
```

When `--data-dir` points at the stress root, the output subfolders (`gifs/`,
`gifs_physics/`) are created inside the stress root. Without this guard, a second run
would attempt to discover and render pickles from inside the `gifs_physics/` folder
(which contains no `.pkl` files, but the walk would produce noisy WARN output and
slightly wrong directory-level scene detection).

The 0-frame guard (`if not frames: tqdm.write(WARN); continue`) was already present —
no change needed for `degenerate_hover` or other short episodes.

## C3 — Both sbatch scripts: optional `$6 = data_dir`

Both `generate_gifs.sh` and `generate_physics_gifs.sh` now accept:

```bash
$5 = per_homotopy   (unchanged)
$6 = data_dir       (new — default: production root)
```

If `$6` is non-empty: `--data-dir $DATA_DIR` is appended to the Python call.
If `$6` is empty (omitted): behaviour is **byte-identical to before this change**.

---

## Stress-rendering commands

```bash
# After E4 U10 collect_stress.sh run:

# Trajectory GIFs — 1 per (scene, case), stride 5, stress root
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh \
    "" "" "" 5 1 logs/uav_expert_data_stress

# Physics GIFs — 1 per (scene, case), stride 5, stress root
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh \
    "" "" "" 5 1 logs/uav_expert_data_stress
```

Output lands in:
```
logs/uav_expert_data_stress/gifs/<scene>/<stress_case>/
logs/uav_expert_data_stress/gifs_physics/<scene>/<stress_case>/
```

---

## Invariants preserved

- Default `generate_gifs.sh` / `generate_physics_gifs.sh` with `$6` omitted: unchanged.
- Production GIF trees (`logs/uav_expert_data/gifs*`): untouched.
- Non-stress episodes: `physics_replay_frames` path unchanged (else-branch).
