# Gen9 — Camera Image Collection from State Demos: Changelog

**Date**: 2026-05-29  
**Branch**: `update_into_FM`  
**Scope**: Standalone pipeline to collect **both** `bp-cam` (cage / third-person)
and `inhand-cam` (wrist / first-person) frames for the D3IL avoiding task by
replaying existing state expert demonstrations in MuJoCo (EGL offscreen).  
**Status**: bp-cam stream verified end-to-end on the cluster; inhand-cam now
sourced from `env.robot.inhand_cam` instead of duplicating bp-cam.  
**Plan**: [`../VISUAL_AVOIDING_DATA_COLLECTION.md`](../VISUAL_AVOIDING_DATA_COLLECTION.md)  
**Background**: [`CAMERAS_IN_D3IL_AND_VISUAL_ALIGNING.md`](CAMERAS_IN_D3IL_AND_VISUAL_ALIGNING.md) — what bp-cam vs inhand-cam are, where they live in MuJoCo, and how the visual aligning pipeline consumes them.

> **Boundary**: This work ends with camera data on disk. No FM-PCC visual avoiding
> model, no training, no eval. Future-reference sections (§8–§13 of the plan MD) are
> preserved but not implemented.

---

## Files Created

| File | Role |
|---|---|
| `collect_visual_avoiding_data/__init__.py` | Empty — marks the directory as a Python package |
| `collect_visual_avoiding_data/collect_visual_avoiding_data.py` | Standalone Python script: replays each demo, captures **both** `env.bp_cam.get_image()` and `env.robot.inhand_cam.get_image()` at every timestep, saves 96×96 BGR PNGs |
| `Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh` | SLURM job wrapper — sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, activates `FMPCC` env, runs collection script |
| `logs_in_develop/Gen9/camera_image_from_state/CHANGELOG.md` | This file |
| `logs_in_develop/Gen9/camera_image_from_state/CAMERAS_IN_D3IL_AND_VISUAL_ALIGNING.md` | Explainer MD — both camera streams, MuJoCo-side definitions, per-task wiring asymmetry (aligning exposes both, avoiding only exposes bp), visual-aligning DPCC/FM consumption (MultiImageObsEncoder → 128-D → FiLM), and D3IL provenance of the two-stream design |

## Files Modified

None. The collection pipeline is fully additive — no D3IL files are touched.
The wrist view is read from `env.robot.inhand_cam`, which already exists on
every D3IL robot (`MjRobot.py:62`) and is auto-registered with the scene by
`MjScene.py:65` — no env-level surgery required.

## Files Expected at Runtime (Created by Script)

When `collect_visual_avoiding.sh` runs, it creates:

```
d3il/environments/dataset/data/avoiding/all_data/
  state/                              ← symlink → ../data
  images/bp-cam/
    env_<id>/0.png, 1.png, ...        ← env.bp_cam        (cage / third-person), 96×96 BGR PNG
  images/inhand-cam/
    env_<id>/0.png, 1.png, ...        ← env.robot.inhand_cam (wrist / first-person), 96×96 BGR PNG
  train_files.pkl                     ← list of state filenames for training
  eval_files.pkl                      ← list for eval
```

---

## Key Implementation Details

| Aspect | Decision |
|---|---|
| Replay strategy | `env.reset()` then `env.step([des_c_pos[t+1][:2], fixed_z, [0,1,0,0]])` per step |
| `fixed_z` | Read from `env.robot_state()[2:]` after reset (≈ 0.12 m, not hardcoded) |
| `bp-cam` source | `env.bp_cam.get_image(width=res, height=res, depth=False)` |
| `inhand-cam` source | `env.robot.inhand_cam.get_image(width=res, height=res, depth=False)` — the same MjInhandCamera the aligning env re-exports; avoiding env never named it but the robot still owns it |
| `depth=False` | REQUIRED — default `depth=True` returns a `(rgb, depth)` tuple, not an ndarray |
| Color order | `get_image` returns RGB; we `cv2.cvtColor(..., RGB2BGR)` before `cv2.imwrite` so the aligning loader's BGR→RGB read produces correct colors |
| Image format | 96×96 BGR PNG (matches visual aligning convention) |
| Skip-existing | Skip only if **both** `bp-cam/<ep>/` AND `inhand-cam/<ep>/` are populated — guards against partial Option-B runs |
| Preflight | Renders one frame from each cam after `env.start()`; aborts on bad shape or near-uniform pixels rather than silently producing garbage |
| D3IL files touched | **None** — wrist cam comes from `env.robot.inhand_cam` (already in MjRobot+MjScene); no env edits |
| Train/eval split | Default 0.9 / 0.1, configurable via `--train-ratio` |
| `sys.path` setup | Inlined at top of script (`_REPO_ROOT`, `_D3IL_ROOT`) — no shell prerequisite |
| `D3IL_DIR` env | Defaulted in script if unset |
| `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` | Set in SLURM wrapper before Python starts (some cluster shells preset `PYOPENGL_PLATFORM` to a non-egl value, which mujoco refuses) |

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
# 1. Image counts match state timesteps AND match across streams
python -c "
import os, pickle
root='d3il/environments/dataset/data/avoiding/all_data'
ep = sorted(os.listdir(f'{root}/state'))[0]
T  = len(pickle.load(open(f'{root}/state/{ep}','rb'))['robot']['des_c_pos']) - 1
nb = len(os.listdir(f'{root}/images/bp-cam/{ep[:-4]}'))
ni = len(os.listdir(f'{root}/images/inhand-cam/{ep[:-4]}'))
print(f'{ep}: T={T}, bp={nb}, inhand={ni}, all_match={nb==T==ni}')
"

# 2. Both streams non-uniform AND visually different from each other
python -c "
import cv2, numpy as np
bp = cv2.imread('d3il/environments/dataset/data/avoiding/all_data/images/bp-cam/env_000_00/0.png')
ih = cv2.imread('d3il/environments/dataset/data/avoiding/all_data/images/inhand-cam/env_000_00/0.png')
print(f'bp     shape={bp.shape}, mean={bp.mean():.1f}, std={bp.std():.1f}')
print(f'inhand shape={ih.shape}, mean={ih.mean():.1f}, std={ih.std():.1f}')
print(f'are_identical (Option-B regression check): {np.array_equal(bp, ih)}')   # must be False
"
```

---

## What Is NOT Done (Out of Scope)

- `VisualAvoidingDataset` class — future ref in plan §8
- Training script wiring — future ref in plan §9
- Eval harness adaptation — future ref in plan §10
- Image conversion to compressed npz / lmdb — kept as PNGs for now (matches aligning)

---

## Revision Log

| Rev | What changed |
|---|---|
| Initial | Option B placeholder: bp-cam only, inhand-cam duplicated bp-cam |
| Bugfix 1 | SLURM `${BASH_SOURCE[0]}` resolved to `/var/lib` (script staged into slurmd spool). Switched to `$SLURM_SUBMIT_DIR` + upward marker-dir search. |
| Bugfix 2 | mujoco refused `PYOPENGL_PLATFORM` preset by cluster shell. Pinned `PYOPENGL_PLATFORM=egl` in the SLURM wrapper alongside `MUJOCO_GL=egl`. |
| Bugfix 3 | `get_image()` default `depth=True` returned a `(rgb, depth)` tuple → `.astype` failed on every episode. Fixed: pass `depth=False`. |
| Bugfix 4 | `cv2.imwrite` was receiving RGB while loader reads as BGR — channels would swap on decode. Fixed: `cv2.cvtColor(..., RGB2BGR)` before write. |
| Option A | Switched `inhand-cam` from a bp-cam duplicate to a real wrist view via `env.robot.inhand_cam.get_image(depth=False)`. Added preflight render check + both-folders skip-existing guard. No D3IL files touched. |
