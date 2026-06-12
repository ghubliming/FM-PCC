# Gen11 Epoch 5 — Usage Guide

**Prerequisite**: Epoch 4 state pickles present at `logs/uav_expert_data/{scene}/{homotopy}/{ep}.pkl` (1769 episodes from Epoch 4 CLOSURE).

---

## WS-C — Mini-FM Sanity Gate (run this first)

Trains a tiny numpy-only flow-matching model on ≤100 empty-scene episodes and checks whether the data pipeline is correct before touching cameras.

### Run (local, no GPU needed)

```bash
python uav_expert_data_collect/mini_fm_sanity.py
```

Custom config:
```bash
python uav_expert_data_collect/mini_fm_sanity.py --n-episodes 50 --n-steps 500 --seed 42
```

### Run (SLURM)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 10
```

### Expected output

```
[ mini-fm ] Loaded 80 train / 20 eval chunks  shape=(N, 8, 9)
[ mini-fm ] step 0   loss=2.341
[ mini-fm ] step 100 loss=0.187
...
[ mini-fm ] step 500 loss=0.031

=== Mini-FM Sanity Results ===
  Shape gate  : (8, 9)  ✅
  RMS pos err : 0.042 m  ✅  (< 0.1 m)
  Action ratio: 1.12      ✅  (0.5–2.0)
  VERDICT: GO — Epoch 4 data pipeline is correct.
```

### Interpretation

| Result | Meaning |
|---|---|
| Shape `(8, 9)` ✅ | Dataloader produces correct `(H=8, D=9)` chunks matching FM-PCC format |
| RMS < 0.1 m ✅ | Model learned to reproduce PID trajectories; action convention (delta not absolute) is correct |
| Action ratio 0.5–2.0 ✅ | Predicted action magnitudes match ground truth; no scale inversion |
| **VERDICT: GO** | Safe to start Epoch 6 FM-PCC training |
| RMS diverging / NaN | Action convention wrong — re-examine Epoch 4 delta vs absolute. Do NOT proceed to Epoch 6. |
| RMS > 0.1 m but converging | Try `--n-steps 2000`; if still failing, check for normalisation bug |

---

## WS-A — Camera Image Collection

Replays each Epoch 4 episode in MuJoCo (EGL offscreen), captures overhead `bp-cam` and body-mounted `track-cam` at 96×96 px, saves as numbered PNGs.

### Run (local — requires EGL GPU)

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Smoke: 1 episode
python uav_expert_data_collect/collect_camera_images.py --max-episodes 1

# Full collection (all scenes)
python uav_expert_data_collect/collect_camera_images.py

# Single scene
python uav_expert_data_collect/collect_camera_images.py --scene corridor
```

### Run (SLURM)

```bash
# All scenes
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh

# Single scene
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh "" corridor
```

### Expected output

```
[ cam-collect ] 1769 episodes  resolution=96×96  skip_existing=True

[ cam-collect ] Scene: corridor (436 episodes)
  corridor: 100%|█████████| 436/436

[ cam-collect ] Scene: empty (500 episodes)
...

============================================================
[ cam-collect ] Done.
  Collected:   1769 episodes
  Skipped:     0 episodes (already existed)
  Errors:      0
  Output dir:  logs/uav_expert_data/images/
============================================================
```

### Output layout

```
logs/uav_expert_data/images/
  bp-cam/{scene}/{homotopy}/{ep_id}/
    0.png, 1.png, …, T-1.png
  track-cam/{scene}/{homotopy}/{ep_id}/
    0.png, 1.png, …, T-1.png
```

### Verification

```bash
# Frame count matches state timesteps
python -c "
import os, pickle
pkl = 'logs/uav_expert_data/empty/N_A/ep_000000.pkl'
ep = pickle.load(open(pkl,'rb'))
T = len(ep['obs'])
bp = len(os.listdir('logs/uav_expert_data/images/bp-cam/empty/N_A/ep_000000'))
print(f'T={T}, bp_frames={bp}, match={T==bp}')
"

# Non-trivial image content (std > 10 = real content, not black)
python -c "
import cv2
img = cv2.imread('logs/uav_expert_data/images/bp-cam/empty/N_A/ep_000000/0.png')
print(f'shape={img.shape}, mean={img.mean():.1f}, std={img.std():.1f}')
"
```

### Interpretation

| Result | Meaning |
|---|---|
| Frame count = T ✅ | State injection aligned correctly with episode length |
| `std > 10` ✅ | Image has real scene content (drone + floor/obstacles visible) |
| `std < 2` or all-black | EGL not initialised, or camera spec wrong |
| Errors: 0 ✅ | All episodes replayed without MuJoCo exceptions |

---

## WS-B — GIF / Video Generation

Produces a per-episode side-by-side GIF (`[bp-cam | track-cam]`, 192×96 px) for human inspection. Does NOT feed into training.

### Option B1 — render from state (standalone, GPU required)

```bash
export MUJOCO_GL=egl

# Smoke: 1 episode
python uav_expert_data_collect/generate_trajectory_gifs.py --max-episodes 1

# Full (SLURM, all scenes)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh

# With MP4 output
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" "" mp4

# Subsampled (every 3rd frame — smaller files)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" "" "" 3
```

### Option B2 — assemble from saved PNGs (CPU, after WS-A)

```bash
# No GPU needed; requires WS-A images already on disk
python uav_expert_data_collect/assemble_gifs_from_pngs.py
```

### Output layout

```
logs/uav_expert_data/gifs/
  {scene}/{homotopy}/
    {ep_id}.gif          ← 10 fps side-by-side [bp | track]
    {ep_id}.mp4          ← optional
```

### Interpretation

Spot-check 5 GIFs per scene:

| What to look for | Correct | Wrong |
|---|---|---|
| Drone visible and moving | ✅ smooth flight | Black frames → EGL issue |
| Obstacles present | ✅ corridor walls / pillars visible | Empty scene when should have obstacles → wrong XML loaded |
| Episode duration | Matches T in pickle | Truncated → frame-stride too large |
| Text overlay | Scene/homotopy/timestep readable | Optional — missing overlay is not a bug |

---

## Full run order

```
1. WS-C (mini-FM gate)   → confirms data is correct — do first, ~10 min
2. WS-B (GIFs)           → visual spot-check of trajectories
3. WS-A (camera images)  → full image dataset for Epoch 6 training
```

WS-C failure blocks Epoch 6. WS-A and WS-B are independent and can run in parallel once WS-C passes.
