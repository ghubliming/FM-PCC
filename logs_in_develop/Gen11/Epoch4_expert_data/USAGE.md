# Gen11 Epoch 4 — Usage Guide: UAV Expert Data Collection

**Audience**: Someone running the pipeline on the cluster for the first time.  
**Prerequisite**: Repo synced, `conda activate FMPCC`, `Kp_omega` fix already applied (done in this session).

---

## Quick-start

### Option A — Local Python (no SLURM)

```bash
# 1. Smoke-test: 10 trials, empty scene
python uav_expert_data_collect/collect.py --scene empty --n-trials 10 --seed 0

# 2. Validate the 10 episodes
python uav_expert_data_collect/stats_validator.py --data-dir logs/uav_expert_data/empty

# 3. Full collection — run once per scene (or wrap in a loop)
python uav_expert_data_collect/collect.py --scene corridor --n-trials 500 --seed 0
python uav_expert_data_collect/collect.py --scene s_curve  --n-trials 500 --seed 0
python uav_expert_data_collect/collect.py --scene pillars  --n-trials 500 --seed 0

# 4. Validate after collection
python uav_expert_data_collect/stats_validator.py --data-dir logs/uav_expert_data/corridor
```

### Option B — SLURM (cluster, recommended for full collection)

The `collect.sh` script runs **collect → validate** as a single job.
Submit from the repo root after `git push` / sync to cluster.

```bash
# 1+2. Smoke-test: 10 trials, empty scene — validator output appears in job log
sbatch Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 10

# 3+4. Full collection — all 4 scenes in parallel (500 trials each)
sbatch --array=0-3 Slurm_Codes/sbatch/uav_expert_data/collect.sh all_scenes 500
```

Each job writes to the cluster filesystem (gitignored):
- `logs/uav_expert_data/<scene>/` — episode pickles
- `logs/uav_expert_data/<scene>/run_summary.json` — rejection rate, timing
- `logs/uav_expert_data/<scene>/dataset_stats.json` — speed/length stats vs targets
- validator comparison table printed to the job's stdout log

---

## Command reference

### `collect.py`

```
python uav_expert_data_collect/collect.py
    --scene        {empty|corridor|s_curve|pillars}   REQUIRED
    --n-trials     INT      episodes to attempt        [200]
    --seed         INT      base RNG seed              [0]
    --gain-variant {pid_default|pid_high_gain|pid_low_gain}  [pid_default]
    --homotopy     {all|<label>}  'all' cycles all classes  [all]
    --noise-sigma  FLOAT    noise std on targets (m)   [0.02]
    --out-dir      PATH     override default output root
    --reject-limit FLOAT    abort threshold on rejection rate  [0.30]
```

**Output location**: `logs/uav_expert_data/<scene>/<homotopy_safe>/<episode_id>.pkl`  
**Summary**: `logs/uav_expert_data/<scene>/run_summary.json`

### `stats_validator.py`

```
python uav_expert_data_collect/stats_validator.py
    --data-dir  PATH   root of uav_expert_data/<scene> (or parent for all scenes)
    --stats-json PATH  Phase 4-α reference [auto-located]
```

---

## What the pipeline produces

### Episode pickle schema (`<episode_id>.pkl`)

```python
{
  'episode_id': 'corridor_L_pid_default_0000042',
  'scene':      'corridor',
  'homotopy':   'L',
  'controller': 'pid_default',
  'dt':         0.03,            # dataset timestep (~33 Hz)

  'obs':     np.ndarray (T, 6),  # [p_x, p_y, p_z, v_x, v_y, v_z]  float32
  'actions': np.ndarray (T-1,3), # [Δp_des_x, Δp_des_y, Δp_des_z]   float32
  'targets': np.ndarray (T, 3),  # absolute p_des — for debugging only

  'obstacles': [...],            # obstacle geometry list (baked in)
  'metadata':  {...},            # start_pos, duration, contact_fraction, …
}
```

**FM-PCC dataloader expects**: chunks of shape `(H=8, D=9)` where `D = [actions(3) ‖ obs(6)]`.  
The episode gives you all timesteps; the dataloader slices them into H-step chunks.

### Homotopy classes per scene

| Scene | Classes | What they mean |
|---|---|---|
| `empty` | `N/A` | Random start → end, no obstacles |
| `corridor` | `L`, `C`, `R` | Fly left-biased / centred / right-biased through the 1 m wide corridor |
| `s_curve` | `default` | Standard 4-waypoint S-path (only one topological route) |
| `pillars` | `(L,L,L)` `(L,R,L)` `(R,L,R)` `(R,R,R)` | Pass left or right of each of the 3 pillar pairs |

Each homotopy class maps to a distinct directory under `logs/uav_expert_data/<scene>/`.

---

## Interpreting the output

### `run_summary.json`

```json
{
  "saved": 478,
  "rejected": 22,
  "rejection_rate": 0.044,
  "sec_per_episode": 1.3
}
```

| Field | Healthy value | Action if not |
|---|---|---|
| `rejection_rate` | < 0.10 for corridor/empty; < 0.20 for s_curve/pillars | Check PID stability; try `--gain-variant pid_low_gain` |
| `sec_per_episode` | 1–5 s | If > 10 s, reduce `--n-trials` per job or check CPU allocation |

### `stats_validator.py` output

```
  Speed (m/s):
    Generated   mean=0.387  median=0.341  p95=0.812
    Our target  0.30–0.50 m/s  (RESEARCH §3.3)
    STATUS: ✅ OK

  Episode length (steps at ~33 Hz):
    Generated   mean=198  median=182  [44, 528]

  Action Δp_des norm (m per step):
    Generated   mean=0.0118  p95=0.0241
    Expected    0.009–0.015 m/step (0.3–0.5 m/s at 33 Hz)
    STATUS: ✅ OK
```

| What to look for | ✅ Good | ⚠️ Problem |
|---|---|---|
| Mean speed | 0.3–0.5 m/s | < 0.1 → PID diverged; > 1.0 → bad init conditions |
| Action Δp norm mean | 0.008–0.020 m/step | Outside range → wrong downsample stride or duration |
| Episode length | > 100 steps | < 50 → trajectories too short; FM can't learn horizon-8 chunks |
| Homotopy counts | Roughly equal across classes | Heavily skewed → a class is nearly always rejected |

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `rejection_rate > 0.30`, script aborts | PID instability (s_curve, pillars) | Confirm `Kp_omega = [2.5, 2.5, 1.0]`; try `--gain-variant pid_low_gain` |
| All episodes have `contact_fraction = 0` but speed is ~0 | Drone hovers at init, never tracks reference | Check `traj_fn` returned wrong `p_des`; verify scene XML path |
| `KeyError: 'obs'` in validator | Stale pickle from an old schema | Delete `logs/uav_expert_data/` and re-collect |
| Stats show mean speed > 1 m/s | Duration too short → drone tries to cover too much ground quickly | Increase `--n-trials` with longer `duration` range in `generator.py` |

---

## Mini-FM sanity gate (§3 of plan — run before full Phase 4-γ)

After collecting ~100 empty-scene episodes, run a tiny FM training pass:

```bash
# Adapt your existing FM train script to use the UAV dataset loader.
# Target: RMS position error < 0.10 m on empty-scene held-out episodes.
# If this fails, the data format is wrong — fix before scaling to 4k episodes.
```

If FM reproduces PID trajectories at < 0.1 m RMS, the schema, action convention, and horizon configuration are all correct. Only then proceed to full collection.

---

## Scaling up (Phase 4-γ)

```bash
# Each array task handles one scene, 500 trials, two gain variants cycling.
# 4 jobs run in parallel. Each completes in ~1–2 h on a 4-CPU node.
sbatch --array=0-3 Slurm_Codes/sbatch/uav_expert_data/collect.sh all_scenes 500

# After all jobs finish, validate the combined dataset:
for scene in empty corridor s_curve pillars; do
    python uav_expert_data_collect/stats_validator.py \
        --data-dir logs/uav_expert_data/$scene
done
```

Expected output: **~2000 episodes** total (500 × 4 scenes), stored under `logs/uav_expert_data/` (gitignored). At ~1.3 s/episode on 4 CPUs, each scene takes ~11 min. Plan for 2 cluster-days including debugging (AUDIT R9).

---

## SLURM script

**File**: `Slurm_Codes/sbatch/uav_expert_data/collect.sh`

```bash
# Single scene, single job
sbatch Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 200

# All 4 scenes in parallel (one array task per scene)
sbatch --array=0-3 Slurm_Codes/sbatch/uav_expert_data/collect.sh all_scenes 500

# With explicit gain variant and seed offset
sbatch Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500 pid_high_gain 1000
```

**Arguments** (positional):

| Position | Name | Default | Values |
|---|---|---|---|
| `$1` | scene | `empty` | `empty` `corridor` `s_curve` `pillars` `all_scenes` |
| `$2` | n_trials | `200` | any int |
| `$3` | gain | `pid_default` | `pid_default` `pid_high_gain` `pid_low_gain` |
| `$4` | seed_offset | `0` | any int |

When `scene=all_scenes`, `SLURM_ARRAY_TASK_ID` (0–3) selects the scene automatically: 0=empty, 1=corridor, 2=s_curve, 3=pillars.

Seed for each array task = `seed_offset + ARRAY_ID × 10000`, so parallel tasks never produce overlapping episodes.

**Resource spec** (in script header): 1 node, 4 CPUs, 8 GB RAM, 4 h walltime, `gpu-1-student` partition with `gres=gpu:1` — same as all other jobs in this project. Collection is headless MuJoCo (`MUJOCO_GL=egl`); the GPU allocation is unused but required by the partition.

---

## Cross-references

| Document | Content |
|---|---|
| `EPOCH4_EXECUTION_PLAN.md` | Full phase plan, risk register, blocking decisions |
| `Materials/AUDIT.md` | R1–R9 recommendations; R4 (12D fallback), R5 (covariate shift), R7 (velocity stat) |
| `phase4_alpha_uavflow_stats.json` | UAV-Flow reference statistics (273 episodes) |
| `CHANGELOG.md` | All files touched in this session |
| `Slurm_Codes/sbatch/uav_expert_data/collect.sh` | SLURM wrapper for Phase 4-γ collection |
