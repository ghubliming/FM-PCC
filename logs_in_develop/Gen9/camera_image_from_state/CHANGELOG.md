# Gen9 — Camera Image Collection from State Demos: Changelog

**Date**: 2026-05-29  
**Branch**: `update_into_FM`  
**Scope**: Standalone pipeline to collect bp-cam camera frames for the D3IL avoiding
task by replaying existing state expert demonstrations in MuJoCo (EGL offscreen).  
**Status**: Code complete, not yet executed on cluster.  
**Plan**: [`../VISUAL_AVOIDING_DATA_COLLECTION.md`](../VISUAL_AVOIDING_DATA_COLLECTION.md)

> **Boundary**: This work ends with camera data on disk. No FM-PCC visual avoiding
> model, no training, no eval. Future-reference sections (§8–§13 of the plan MD) are
> preserved but not implemented.

---

## Files Created

| File | Role |
|---|---|
| `collect_visual_avoiding_data/__init__.py` | Empty — marks the directory as a Python package |
| `collect_visual_avoiding_data/collect_visual_avoiding_data.py` | Standalone Python script: replays each demo, captures `env.bp_cam.get_image()` at every timestep, saves 96×96 BGR PNGs |
| `Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh` | SLURM job wrapper — sets `MUJOCO_GL=egl`, activates `FMPCC` env, runs collection script |
| `logs_in_develop/Gen9/camera_image_from_state/CHANGELOG.md` | This file |

## Files Modified

None. The collection pipeline is fully additive — no existing code is touched.

## Files Expected at Runtime (Created by Script)

When `collect_visual_avoiding.sh` runs, it creates:

```
d3il/environments/dataset/data/avoiding/all_data/
  state/                              ← symlink → ../data
  images/bp-cam/
    env_<id>/0.png, 1.png, ...        ← 96×96 BGR PNG, one per timestep
  images/inhand-cam/
    env_<id>/0.png, 1.png, ...        ← duplicate of bp-cam (Option B placeholder)
  train_files.pkl                     ← list of state filenames for training
  eval_files.pkl                      ← list for eval
```

---

## Key Implementation Details

| Aspect | Decision |
|---|---|
| Replay strategy | `env.reset()` then `env.step([des_c_pos[t+1][:2], fixed_z, [0,1,0,0]])` per step |
| `fixed_z` | Read from `env.robot_state()[2:]` after reset (≈ 0.12 m, not hardcoded) |
| Camera API | `env.bp_cam.get_image(width=res, height=res)` — already returns BGR uint8, vertically corrected |
| Image format | 96×96 BGR PNG (matches visual aligning convention) |
| `inhand-cam` | Duplicate of bp-cam (Option B) — fastest path; Option A (robot-POV cam) deferred |
| Skip-existing | On by default — re-runs are resumable |
| Train/eval split | Default 0.9 / 0.1, configurable via `--train-ratio` |
| `sys.path` setup | Inlined at top of script (`_REPO_ROOT`, `_D3IL_ROOT`) — no shell prerequisite |
| `D3IL_DIR` env | Defaulted in script if unset |
| `MUJOCO_GL=egl` | Set in SLURM wrapper before Python starts |

---

## How to Run

### Smoke (10 episodes, local)

```bash
cd /workspaces/FM-PCC
export MUJOCO_GL=egl
python collect_visual_avoiding_data/collect_visual_avoiding_data.py --max-episodes 10
```

### Full collection (SLURM)

```bash
sbatch Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh
#                                                                  ^ all default args
# Or with overrides:
sbatch Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh 96 "" 0.9
#                                                                   ^res ^max ^train_ratio
```

---

## Prerequisites (Outside This Work)

1. **D3IL avoiding state demos** must be present at:
   `d3il/environments/dataset/data/avoiding/data/env_*.pkl`  
   (Currently not present in this repo — must be downloaded from the D3IL release
   or recorded via the gamepad pipeline before this script can run.)

2. **GPU node with EGL support** — required for offscreen MuJoCo rendering.
   The SLURM script requests `--gres=gpu:1` and sets `MUJOCO_GL=egl`.

3. **Conda env `FMPCC`** with `mujoco`, `cv2`, `tqdm`, `numpy` installed.

---

## Verification Plan (After Run)

```bash
# 1. Image count matches state timesteps for a sample episode
python -c "
import os, pickle
ep = sorted(os.listdir('d3il/environments/dataset/data/avoiding/all_data/state'))[0]
n_img = len(os.listdir(f'd3il/environments/dataset/data/avoiding/all_data/images/bp-cam/{ep[:-4]}'))
T = len(pickle.load(open(f'd3il/environments/dataset/data/avoiding/all_data/state/{ep}','rb'))['robot']['des_c_pos']) - 1
print(f'{ep}: images={n_img}, T={T}, match={n_img==T}')
"

# 2. Image content sanity (non-black, non-uniform)
python -c "
import cv2
img = cv2.imread('d3il/environments/dataset/data/avoiding/all_data/images/bp-cam/env_0/0.png')
print(f'shape={img.shape}, mean={img.mean():.1f}, std={img.std():.1f}')
"
```

---

## What Is NOT Done (Out of Scope)

- `VisualAvoidingDataset` class — future ref in plan §8
- Training script wiring — future ref in plan §9
- Eval harness adaptation — future ref in plan §10
- `RobotCam` (Option A robot-POV camera) — future enhancement
- Image conversion to compressed npz / lmdb — kept as PNGs for now (matches aligning)
