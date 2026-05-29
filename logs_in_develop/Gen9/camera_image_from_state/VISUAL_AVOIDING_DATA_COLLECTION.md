# Visual Avoiding — Camera Data Collection Guide

**Date**: 2026-05-29  
**Status**: Learning material / practice run — **camera data collection only**  
**Reference**: `old_idea_collect_camera_data_in_d3il_avoiding.md`  
**Goal**: Collect camera image data alongside existing D3IL avoiding state demos.

> **Scope boundary**: This plan covers §1–§7 only — camera data collection pipeline
> (SLURM → collection script → images saved to disk). We stop here.  
> §8–§13 are retained as future reference for when a real FM-PCC visual avoiding
> model is built, but they are **not current work**.  
> Do NOT delete the previously-built visual avoiding plans in `logs_in_develop/Gen9/` —
> they remain valid future architecture blueprints.

---

## 1. Current State (What Exists)

### State data (exists)
```
d3il/environments/dataset/data/avoiding/data/
  env_0.pkl
  env_1.pkl
  ...                       ← expert state demos, recorded via gamepad
```

State pickle layout per episode (same structure as aligning):
```python
env_state['robot']['des_c_pos']   # (T+1, 3) — desired TCP position (x, y, z)
env_state['robot']['c_pos']       # (T+1, 3) — actual TCP position (x, y, z)
# Note: z ≈ 0.12 m throughout (robot slides in XY plane)
```

### Camera (exists but images never saved)
`ObstacleAvoidanceEnv` already instantiates a `BPCageCam` (bird's-eye, 1024×1024)
and registers it as `cam_dict = {"bp-cam": CamLogger(scene, self.bp_cam)}`.

The image data was **not saved** during state recording because the data pipeline used
the state-only `Avoiding_Dataset`, which never calls the camera logger.

### What is missing
```
d3il/environments/dataset/data/avoiding/all_data/
  images/bp-cam/      ← MISSING
  images/inhand-cam/  ← MISSING (camera not in env; see §3 for design choice)
  train_files.pkl     ← MISSING
  eval_files.pkl      ← MISSING
```

---

## 2. API Target — Visual Aligning Parity

The Gen7 visual pipeline (`VisualUNet`, `ParityAligningDataset`, eval wrapper) expects:

| Component | Visual Aligning | Visual Avoiding (target) |
|---|---|---|
| Camera 1 | `bp-cam` (96×96 RGB) | `bp-cam` (96×96 RGB) |
| Camera 2 | `inhand-cam` (96×96 RGB) | `robot-cam` or `bp-cam` copy — see §3 |
| Trajectory dim | 9D `[act(3)\|des_c_pos(3)\|c_pos(3)]` | **9D** `[act(3)\|des_pos(3)\|c_pos(3)]` |
| Action | 3D velocity `[dx, dy, dz]` | 3D velocity `[dx, dy, dz]` (dz≈0) |
| Obs in trajectory | 6D `[des_c_pos(3)\|c_pos(3)]` | 6D `[des_pos(3)\|c_pos(3)]` |
| Dataset class | `ParityAligningDataset` | `VisualAvoidingDataset` (new) |
| Image keys in conditions | `primary_img`, `wrist_img` | `primary_img`, `wrist_img` |
| `VisualUNet` | `TRANSITION_DIM=9` ✓ | `TRANSITION_DIM=9` ✓ (same) |

**Using 3D (not 2D) positions**: Even though the avoiding robot slides in the XY plane,
we keep z in the trajectory so `TRANSITION_DIM = 9` matches the visual aligning UNet
exactly — no code change needed in `VisualUNet`. The z dimension is near-constant
(≈0.12 m) and the model trivially learns to ignore it.

---

## 3. Camera Design Decision — `inhand-cam`

The avoiding env has **one camera** (`bp-cam`). Visual aligning expects two (`primary_img`
and `wrist_img`). Three options:

### Option A — Add a second camera to the avoiding env (recommended)
Add a `RobotCam` attached to the robot end-effector looking forward (robot POV).
Provides genuinely different visual information: shows obstacle proximity from robot
perspective. Requires adding a camera body to the MuJoCo XML.

```python
class RobotCam(MjCamera):
    """Forward-facing camera attached to robot EE — robot's POV."""
    def __init__(self, width=96, height=96):
        super().__init__(
            "robot_cam",
            width, height,
            init_pos=[0.5, 0.0, 0.12],   # robot EE start
            init_quat=[0.707, 0.0, 0.707, 0.0],  # facing forward (+Y)
        )
```

Attach to the scene and add `"inhand-cam": CamLogger(scene, self.robot_cam)` to
`cam_dict`. Camera position must be updated to track the robot EE at each step:
```python
robot_pos = env.robot.current_c_pos
env.robot_cam.set_position(robot_pos)
```

### Option B — Duplicate bp-cam as inhand-cam (minimal effort)
Use the same bp-cam frame for both `primary_img` and `wrist_img`. The model sees
identical views for both channels. Works for getting the pipeline running but wastes
capacity — the ResNet's two independent heads learn redundant features.

### Option C — Single camera mode
Modify `VisualUNet` to accept one camera instead of two. Requires code change.

**Recommended**: Start with Option B for smoke testing the full pipeline, implement
Option A for real training runs.

---

## 4. Data Directory Structure (Target)

Mirrors visual aligning exactly:

```
d3il/environments/dataset/data/avoiding/
  data/                          ← existing (state demo files, D3IL convention)
    env_0.pkl
    env_1.pkl
    ...

  all_data/                      ← NEW (matches aligning structure)
    state/                       ← symlink or copy of data/
      env_0.pkl
      env_1.pkl
      ...
    images/
      bp-cam/
        env_0/
          0.png                  ← frame at timestep 0 (96×96 RGB, BGR order)
          1.png
          ...
        env_1/
          0.png
          ...
      inhand-cam/                ← robot-cam or bp-cam duplicate
        env_0/
          0.png
          ...
    train_files.pkl              ← list of state filenames for training (same format as aligning)
    eval_files.pkl               ← list for evaluation
```

Image storage format: **BGR order PNG, 96×96**, matching the aligning dataset convention.
(The env returns BGR from cv2; the model was trained on BGR-order images.)

---

## 5. Replay Strategy

The avoiding task has **fixed obstacle positions** — obstacles do not move between
episodes. Only the robot's trajectory varies. This makes replay simple:

```python
# For each demo:
env.reset()                          # robot at init_end_eff_pos, obstacles fixed
for t in range(T):
    action = vel_state[t]            # des_c_pos[t+1] - des_c_pos[t]  (3D)
    obs, reward, done, info = env.step(
        np.concatenate([des_c_pos[t+1], [0, 1, 0, 0]])  # 7D pose command
    )
    frame = env.bp_cam.get_image()   # (H, W, 3) BGR numpy array
    frame_96 = cv2.resize(frame, (96, 96))
    cv2.imwrite(f"{bp_dir}/{t}.png", frame_96)
```

The robot follows the recorded trajectory exactly (pose command replays the expert).
Camera frames are captured at the commanded position, not the lagging actual position.

---

## 6. Standalone Collection Script

**File**: `collect_visual_avoiding_data/collect_visual_avoiding_data.py`

```python
"""
Collect camera data for D3IL avoiding task by replaying state expert demonstrations.

Creates:
    d3il/environments/dataset/data/avoiding/all_data/
        state/           ← symlinked to ../data/
        images/bp-cam/   ← collected here
        images/inhand-cam/ ← either robot-cam or bp-cam duplicate
        train_files.pkl
        eval_files.pkl

Usage:
    python collect_visual_avoiding_data/collect_visual_avoiding_data.py \
        --resolution 96 \
        --duplicate-bp-as-inhand \
        --train-ratio 0.9
"""

import os
import sys
import cv2
import glob
import pickle
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('d3il'))
os.environ['D3IL_DIR'] = os.path.abspath('d3il/environments/d3il')

from agents.utils.sim_path import sim_framework_path
from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import \
    ObstacleAvoidanceEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resolution',           type=int,   default=96)
    p.add_argument('--train-ratio',          type=float, default=0.9)
    p.add_argument('--duplicate-bp-as-inhand', action='store_true',
                   help='Use bp-cam for inhand-cam too (Option B)')
    p.add_argument('--max-episodes',         type=int,   default=None,
                   help='Limit episodes for testing')
    return p.parse_args()


def build_output_dirs(all_data_dir, res):
    os.makedirs(f"{all_data_dir}/state",                 exist_ok=True)
    os.makedirs(f"{all_data_dir}/images/bp-cam",         exist_ok=True)
    os.makedirs(f"{all_data_dir}/images/inhand-cam",     exist_ok=True)


def load_state_files(data_dir):
    """Return sorted list of state demo filenames."""
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    return files


def replay_episode(env, des_c_pos, c_pos, resolution):
    """
    Replay one expert episode, capturing bp-cam at each step.
    Returns list of (bp_frame, inhand_frame) pairs — (H,W,3) BGR uint8.
    """
    T = len(des_c_pos) - 1

    env.reset()
    frames = []

    for t in range(T):
        # Pose command: desired position + fixed horizontal gripper quaternion
        cmd = np.concatenate([des_c_pos[t + 1], [0, 1, 0, 0]])   # (7,)
        env.step(cmd)

        # Capture bp-cam at this robot position
        bp_raw = env.bp_cam.get_image()   # (H, W, 3) BGR, vertically flipped internally
        bp_frame = cv2.resize(bp_raw, (resolution, resolution),
                              interpolation=cv2.INTER_AREA)
        frames.append(bp_frame)

    return frames


def save_episode_images(frames, bp_dir, inhand_dir, duplicate_bp):
    """Save frames to bp-cam and inhand-cam directories."""
    os.makedirs(bp_dir,     exist_ok=True)
    os.makedirs(inhand_dir, exist_ok=True)

    for t, frame in enumerate(frames):
        cv2.imwrite(f"{bp_dir}/{t}.png",     frame)
        ih_frame = frame if duplicate_bp else frame   # replace with robot_cam if Option A
        cv2.imwrite(f"{inhand_dir}/{t}.png", ih_frame)


def make_train_eval_split(filenames, train_ratio):
    n = len(filenames)
    n_train = int(n * train_ratio)
    train = filenames[:n_train]
    eval_ = filenames[n_train:]
    return train, eval_


def main():
    args = parse_args()

    # ── paths ──────────────────────────────────────────────────────────────────
    state_dir  = sim_framework_path("environments/dataset/data/avoiding/data")
    all_data   = sim_framework_path("environments/dataset/data/avoiding/all_data")
    build_output_dirs(all_data, args.resolution)

    # Symlink state/ → ../data/ so ParityAligningDataset-style loaders find it
    state_link = os.path.join(all_data, "state")
    if not os.path.exists(state_link):
        os.symlink(state_dir, state_link)
        print(f"[ collect ] symlinked state/ → {state_dir}")

    # ── env (EGL offscreen, no GUI) ────────────────────────────────────────────
    print("[ collect ] Initialising ObstacleAvoidanceEnv (EGL offscreen)...")
    env = ObstacleAvoidanceEnv(render=False)
    env.start()
    print("[ collect ] Env ready.")

    # ── collect ────────────────────────────────────────────────────────────────
    state_files = load_state_files(state_dir)
    if args.max_episodes:
        state_files = state_files[:args.max_episodes]

    print(f"[ collect ] {len(state_files)} episodes to collect.")

    for fname in tqdm(state_files, desc="Episodes"):
        ep_name = os.path.splitext(fname)[0]   # e.g. "env_0"

        bp_dir     = os.path.join(all_data, "images", "bp-cam",     ep_name)
        inhand_dir = os.path.join(all_data, "images", "inhand-cam", ep_name)

        # Skip if already collected
        if os.path.isdir(bp_dir) and len(os.listdir(bp_dir)) > 0:
            continue

        with open(os.path.join(state_dir, fname), 'rb') as f:
            env_state = pickle.load(f)

        des_c_pos = env_state['robot']['des_c_pos']   # (T+1, 3)
        c_pos     = env_state['robot']['c_pos']       # (T+1, 3)

        frames = replay_episode(env, des_c_pos, c_pos, args.resolution)
        save_episode_images(frames, bp_dir, inhand_dir, args.duplicate_bp_as_inhand)

    # ── train / eval split ─────────────────────────────────────────────────────
    train_files, eval_files = make_train_eval_split(state_files, args.train_ratio)

    with open(os.path.join(all_data, "train_files.pkl"), 'wb') as f:
        pickle.dump(train_files, f)
    with open(os.path.join(all_data, "eval_files.pkl"), 'wb') as f:
        pickle.dump(eval_files, f)

    print(f"[ collect ] Done. {len(train_files)} train / {len(eval_files)} eval episodes.")
    print(f"[ collect ] Output: {all_data}")


if __name__ == "__main__":
    main()
```

---

## 7. SLURM Job

**File**: `Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh`

```bash
#!/bin/bash
#SBATCH --job-name=collect_visual_avoiding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-1-student

# EGL offscreen rendering — requires a GPU node even though no training is done
#SBATCH --gres=gpu:1

set -e
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
cd "$REPO"

echo "========================================"
echo "COLLECT VISUAL AVOIDING DATA"
echo "DATE: $(date)"
echo "NODE: $(hostname)"
echo "========================================"

# ── Args ──
# $1 = resolution (default: 96)
# $2 = duplicate-bp-as-inhand: 1=yes (default), 0=add robot-cam
RESOLUTION="${1:-96}"
DUPLICATE="${2:-1}"

CONDA_ENV="FMPCC"
source activate $CONDA_ENV || conda activate $CONDA_ENV

DUP_FLAG=""
if [ "$DUPLICATE" = "1" ]; then
    DUP_FLAG="--duplicate-bp-as-inhand"
fi

python collect_visual_avoiding_data/collect_visual_avoiding_data.py \
    --resolution "$RESOLUTION" \
    $DUP_FLAG

echo "========================================"
echo "COLLECTION DONE: $(date)"
echo "========================================"
```

Submit:
```bash
# Full collection at 96×96, bp-cam duplicated as inhand-cam
./Slurm_Codes/submit.sh \
    Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh \
    96 1

# Smoke test — only 10 episodes
sbatch --wrap="cd $REPO && python collect_visual_avoiding_data/collect_visual_avoiding_data.py \
    --resolution 96 --duplicate-bp-as-inhand --max-episodes 10"
```

---

---

## ✅ CURRENT SCOPE ENDS HERE

**What is done when §1–§7 are executed:**
- Camera images collected for all avoiding episodes
- Images saved to `all_data/images/bp-cam/` and `all_data/images/inhand-cam/`
- `train_files.pkl` and `eval_files.pkl` written
- Data is on disk and ready — pipeline is complete for data collection purposes

**What is NOT done (and not planned now):**
- No FM-PCC / DPCC visual avoiding model is built
- No training script is run
- No eval harness is set up
- The sections below (§8–§13) are kept as **future reference only**

---

## 8. [FUTURE REF] New Dataset Class — `VisualAvoidingDataset`

**File**: `flow_matcher_v3_avoiding_visual/datasets/sequence.py`  
(or add to `fm_visual_aligning/datasets/sequence.py` with an `if task == 'avoiding'` branch)

```python
class VisualAvoidingDataset(torch.utils.data.Dataset):
    """
    9D trajectory dataset for Visual Avoiding (Gen9).

    API-identical to ParityAligningDataset. Trajectory:
        x[t] = [ dx   dy   dz | des_x des_y des_z | x    y    z  ]
                  act(3)         des_c_pos(3)           c_pos(3)
                 0-2             3-5                    6-8

    dz ≈ 0 throughout (robot moves in XY plane at fixed height).
    Using full 3D keeps TRANSITION_DIM=9, matching VisualUNet without changes.

    Conditions returned:
        {0: obs_6d_norm[0],            ← (6,) anchor for apply_conditioning
         'primary_img': bp_frame,      ← (3, 96, 96) tensor
         'wrist_img':   inhand_frame}  ← (3, 96, 96) tensor
    """

    ACTION_DIM = 3
    OBS_DIM    = 6    # [des_c_pos(3), c_pos(3)]
    TRAJ_DIM   = 9

    def __init__(self, dataset_path, horizon=8, max_n_episodes=1000):
        super().__init__()
        self.horizon = horizon

        from agents.utils.sim_path import sim_framework_path

        state_files = np.load(sim_framework_path(dataset_path), allow_pickle=True)
        rp_data_dir = sim_framework_path(
            "environments/dataset/data/avoiding/all_data/state")
        data_dir    = sim_framework_path(
            "environments/dataset/data/avoiding/all_data")

        n_eps = min(len(state_files), max_n_episodes)
        all_obs_6d  = []
        all_actions = []

        for file in tqdm(state_files[:n_eps], desc='Loading states'):
            with open(os.path.join(rp_data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            des_c_pos = env_state['robot']['des_c_pos']   # (T+1, 3) full 3D
            c_pos     = env_state['robot']['c_pos']       # (T+1, 3)
            T = len(des_c_pos) - 1

            obs_6d  = np.concatenate([des_c_pos[:T], c_pos[:T]], axis=-1)  # (T, 6)
            actions = (des_c_pos[1:] - des_c_pos[:-1]).astype(np.float32)  # (T, 3)

            all_obs_6d.append(obs_6d.astype(np.float32))
            all_actions.append(actions)

        self.obs_normalizer = LimitsNormalizer(np.concatenate(all_obs_6d))
        self.act_normalizer = LimitsNormalizer(np.concatenate(all_actions))
        self._obs_6d  = all_obs_6d
        self._actions = all_actions

        # Load images
        self.bp_cam_imgs     = []
        self.inhand_cam_imgs = []
        for file in tqdm(state_files[:n_eps], desc='Loading images'):
            ep = os.path.basename(file).split('.')[0]
            self.bp_cam_imgs.append(    self._load_images(data_dir, 'bp-cam',     ep))
            self.inhand_cam_imgs.append(self._load_images(data_dir, 'inhand-cam', ep))

        self.n_episodes = n_eps
        self.indices    = self._make_indices()
        print(f'[ VisualAvoidingDataset ] {n_eps} eps, {len(self.indices)} windows '
              f'(horizon={horizon}, traj_dim={self.TRAJ_DIM})')

    # __len__, __getitem__, _make_indices, _load_images
    # → identical to ParityAligningDataset (copy verbatim)
```

> The `__getitem__`, `_make_indices`, and `_load_images` methods are **verbatim copies**
> of `ParityAligningDataset` — the interface is identical. Only the state key names and
> the 3D (not 2D) position extraction differ.

---

## 9. [FUTURE REF] Gen7 Training Entry Point

The existing `fm_visual_aligning_test/train_fm_visual_aligning.py` works with minimal
changes:

```python
# In train_fm_visual_aligning.py — replace dataset loading:
if task == 'avoiding':
    from flow_matcher_v3_avoiding_visual.datasets.sequence import VisualAvoidingDataset
    dataset = VisualAvoidingDataset(
        dataset_path='environments/dataset/data/avoiding/all_data/train_files.pkl',
        horizon=args.horizon,
    )
else:
    from fm_visual_aligning.datasets.sequence import ParityAligningDataset
    dataset = ParityAligningDataset(...)
```

Config block to add to `config/avoiding-d3il.py` (or a new `avoiding-d3il-visual.py`):

```python
'fm_visual_avoiding': {
    **base['fm_visual_aligning'],           # inherit all visual aligning settings
    'prefix': 'fm_visual_avoiding/',
    'action_dim': 3,                        # [dx, dy, dz]
    'obs_dim': 6,                           # [des_c_pos(3) | c_pos(3)]
    'if_vision': True,
},
```

Since `action_dim=3`, `obs_dim=6`, `if_vision=True`, `VisualUNet.TRANSITION_DIM=9` —
the visual aligning UNet is used **without any modification**.

---

## 10. [FUTURE REF] What Needs to Change for Eval

The visual aligning eval (`eval_fm_visual_aligning.py`) needs one addition: the
`Aligning_Sim` must be replaced with `Avoiding_Sim`, and the env's observation format
adapted:

```python
# Visual avoiding prediction interface — matches visual aligning exactly
state = (bp_image, inhand_image, des_robot_pos, robot_pos)
agent.predict(state, if_vision=True)
```

The avoiding sim needs to expose the same 4-tuple interface as the aligning sim. The
`ObstacleAvoidanceEnv` already has `bp_cam` — `inhand_cam` needs to be added (Option A)
or aliased (Option B). `des_robot_pos` = last commanded position (from `des_c_pos`).
`robot_pos` = `env.robot.current_c_pos`.

---

## 11. [FUTURE REF] Full Pipeline Summary (End-to-End)

```
SLURM: collect_visual_avoiding.sh
    │
    ▼
collect_visual_avoiding_data.py
    │   ObstacleAvoidanceEnv (EGL, no GUI)
    │   replay each demo → capture bp-cam at each step → save 96×96 PNG
    │
    ▼
d3il/environments/dataset/data/avoiding/all_data/
    state/          ← symlink to existing state demos
    images/bp-cam/  ← freshly collected
    images/inhand-cam/ ← duplicate of bp-cam or new robot-cam
    train_files.pkl
    eval_files.pkl
    │
    ▼
VisualAvoidingDataset (9D trajectory + 2× 96×96 images)
    │   API-identical to ParityAligningDataset
    │
    ▼
fm_visual_aligning_test/train_fm_visual_aligning.py
    │   VisualUNet (TRANSITION_DIM=9, unchanged)
    │   VisualFlowMatching (unchanged)
    │
    ▼
Checkpoint: logs/avoiding-d3il/fm_visual_avoiding/.../state_best.pt
    │
    ▼
eval (future) → same VisualAgentWrapper, Avoiding_Sim instead of Aligning_Sim
```

---

## 12. [FUTURE REF] Key Decisions Captured

| Decision | Choice | Reason |
|---|---|---|
| Trajectory dim | **9D** (3D act + 6D obs) | API parity with visual aligning — no UNet change |
| z dimension | Included (dz≈0) | Keeps `TRANSITION_DIM=9`; trivially learned as constant |
| inhand-cam (initial) | **bp-cam duplicate** (Option B) | Fastest path to a runnable pipeline |
| inhand-cam (production) | **Robot-POV cam** (Option A) | Distinct visual information improves model |
| Image resolution | **96×96** | Matches visual aligning, ResNet input |
| Image channel order | **BGR** (cv2 native) | Matches what aligning model was trained on |
| State keys | `robot['des_c_pos']`, `robot['c_pos']` full 3D | Same pickle keys as aligning |
| Data root | `all_data/` parallel to existing `data/` | Does not touch existing state-only pipeline |

---

## 13. [FUTURE REF] Verification Checklist (Post-Training)

Before training:

```bash
# 1. Check image count matches state timestep count
python -c "
import os, pickle
ep = 'env_0'
n_imgs = len(os.listdir(f'd3il/environments/dataset/data/avoiding/all_data/images/bp-cam/{ep}'))
state = pickle.load(open(f'd3il/environments/dataset/data/avoiding/all_data/state/{ep}.pkl','rb'))
T = len(state['robot']['des_c_pos']) - 1
print(f'images={n_imgs}, state_T={T}, match={n_imgs==T}')
"

# 2. Check image content (should show robot + obstacles from above)
python -c "
import cv2
img = cv2.imread('d3il/environments/dataset/data/avoiding/all_data/images/bp-cam/env_0/0.png')
print(f'shape={img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.1f}')
# mean > 50 = not black; mean < 220 = not white; confirms real scene
"

# 3. Smoke-train: 5 steps, verify no shape errors
python fm_visual_aligning_test/train_fm_visual_aligning.py \
    "experiment=fm_visual_avoiding" \
    "n_train_steps=5" \
    "seed=42"
```
