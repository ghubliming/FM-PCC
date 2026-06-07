# Gen11 Epoch 5 — U2 Changelog & Usage

**Date**: 2026-06-07  
**Status**: Code changes applied — Phase 2 (re-collect + re-render) pending  
**Parent plan**: [`PLAN.md`](PLAN.md)

---

## Files touched

| File | Change | Type |
|---|---|---|
| `uav_expert_data_collect/collect_camera_images.py` | Obs column fix (A1) + quat injection (D-prep) | Code bug + D-prep |
| `uav_expert_data_collect/generate_trajectory_gifs.py` | Obs column fix (A2) + quat injection (D-prep) | Code bug + D-prep |
| `uav_expert_data_collect/mini_fm_sanity.py` | OBS_DIM 6→9, DATA_DIM 9→12 (B) | Config |
| `d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml` | Remove `mode="trackcom"` (C) | Design fix |
| `uav_expert_data_collect/generator.py` | Add `'q'` to step dict (D-prep) | D-prep |
| `uav_expert_data_collect/dataset_writer.py` | Write `q=(T,4)` field to pickle (D-prep) | D-prep |

---

## Change A1 — WS-A obs column fix

**File**: `uav_expert_data_collect/collect_camera_images.py`

After E4 U2, obs is `(T, 9) = [p_des(3), p(3), v(3)]`. The old code read
`obs[t, :3]` as position (now p_des) and `obs[t, 3:6]` as velocity (now p).
The drone was being rendered at the commanded position, not the actual position.

```python
# BEFORE:
obs = episode['obs']        # (T, 6) = [p(3), v(3)]
...
p = obs[t, :3]
v = obs[t, 3:6]
data.qpos[:3] = p
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qvel[:3] = v

# AFTER:
obs      = episode['obs']          # (T, 9)  U2: [p_des(3) | p(3) | v(3)]
q_stored = episode.get('q', None)  # (T, 4)  D-prep: actual quaternion (E4 U3+); None until re-collect
...
p = obs[t, 3:6]   # U2: p at columns 3:6
v = obs[t, 6:9]   # U2: v at columns 6:9
data.qpos[:3] = p
data.qpos[3:7] = q_stored[t] if q_stored is not None else [1.0, 0.0, 0.0, 0.0]
data.qvel[:3] = v
```

---

## Change A2 — WS-B obs column fix

**File**: `uav_expert_data_collect/generate_trajectory_gifs.py`

Identical pattern to A1.

```python
# BEFORE:
obs = episode['obs']  # (T, 6)
...
data.qpos[:3] = obs[t, :3]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qvel[:3] = obs[t, 3:6]

# AFTER:
obs      = episode['obs']          # (T, 9)  U2: [p_des(3) | p(3) | v(3)]
q_stored = episode.get('q', None)  # (T, 4)  D-prep: actual quaternion (E4 U3+); None until re-collect
...
data.qpos[:3] = obs[t, 3:6]   # U2: p at columns 3:6
data.qpos[3:7] = q_stored[t] if q_stored is not None else [1.0, 0.0, 0.0, 0.0]
data.qvel[:3] = obs[t, 6:9]   # U2: v at columns 6:9
```

---

## Change B — WS-C mini-FM tensor dim

**File**: `uav_expert_data_collect/mini_fm_sanity.py`

```python
# BEFORE:
OBS_DIM = 6       # [p(3), v(3)]
DATA_DIM = 9      # ACTION_DIM + OBS_DIM = 3 + 6
# chunk shape: (H, 9)
# pass criterion: (B, H=8, D=9)

# AFTER:
OBS_DIM = 9       # [p_des(3), p(3), v(3)]  U2
DATA_DIM = 12     # ACTION_DIM + OBS_DIM = 3 + 9  U2
# chunk shape: (H, 12)
# pass criterion: (B, H=8, D=12)
```

---

## Change C — XML camera body-frame fix

**File**: `d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml` line 35

```xml
<!-- BEFORE — chase cam: orientation world-fixed -->
<camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>

<!-- AFTER — body-frame FPV: rotates with drone -->
<camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2"/>
```

No visual difference until Change D removes forced-level attitude. The fix ensures
correct camera semantics in place before E4 U3 re-renders.

---

## Change D-prep — Quaternion field in E4 pickles

These changes wire up the quaternion throughout the data pipeline.
**No SLURM re-collection yet** — these take effect when E4 U3 is submitted.

### `generator.py` — step dict

`q` is already read from `data.qpos[3:7]` at line 222.  Added to the step dict:

```python
# BEFORE:
steps.append({'p': p, 'v': v, 'p_des': np.asarray(p_des, dtype=float)})

# AFTER:
steps.append({'p': p, 'v': v, 'p_des': np.asarray(p_des, dtype=float),
              'q': q.astype(np.float32)})  # D-prep: actual quaternion for attitude rendering
```

### `dataset_writer.py` — pickle schema

```python
# BEFORE (returned dict):
'obs':     obs,       # (T, 9)
'actions': actions,   # (T-1, 3)
'targets': targets,   # (T, 3)

# AFTER:
'obs':     obs,       # (T, 9)
'actions': actions,   # (T-1, 3)
'targets': targets,   # (T, 3)
'q':       q,         # (T, 4)  [w,x,y,z]  D-prep: attitude for rendering
```

Schema comment updated to document the new `q` field.

---

## State of rendering after each phase

| Phase | `q_stored` in pickle? | Quaternion injected? | Visual result |
|---|---|---|---|
| Now (pre-E4 U3) | `None` | Falls back to identity | Drone level, same as before |
| After E4 U3 re-collect | `(T, 4)` float32 | `q_stored[t]` used | Drone shows actual tilt |

---

## Smoke tests — verify all code changes cheaply before full Phase 2

The sbatch wrappers support `max_episodes` as `$1` and `scene` as `$2`, so a 3-episode
single-scene job is valid SLURM.  WS-C is numpy-only — no GPU or sbatch needed; run
locally.  Run these **before** submitting the full Phase 2 jobs.

| Smoke job | Command | Cost | What it verifies |
|---|---|---|---|
| E4 U3 mini | `collect.sh empty 5` | ~30 s | D-prep: `q` field written to pickle by generator + dataset_writer |
| WS-A mini | `collect_camera_images.sh 3 corridor` | ~5 min GPU | A1: obs columns correct; C: XML loads; D-prep: graceful `q` fallback (identity) |
| WS-B mini | `generate_gifs.sh 3 corridor` | ~2 min GPU | A2: obs columns correct in GIF frames |
| WS-C local | `python mini_fm_sanity.py ...` | ~2 min CPU | B: D=12 tensor flows, RMS gate |

```bash
# ── Smoke 1 — E4 U3 mini (verify q field in pickle) ──────────────────────────
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 5

# Check pickle has 'q' after job completes:
python -c "
import pickle, glob
ep = pickle.load(open(sorted(glob.glob('logs/uav_expert_data/empty/**/*.pkl', recursive=True))[0], 'rb'))
print('obs:', ep['obs'].shape)   # expect (T, 9)
print('q  :', ep['q'].shape)     # expect (T, 4)
print('q[0]:', ep['q'][0])       # expect near [1, 0, 0, 0] at hover start
"

# ── Smoke 2 — WS-A mini (verify A1 obs columns + XML + quat fallback) ─────────
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh 3 corridor

# Check: 3 episode dirs exist under both bp-cam and track-cam; images non-black
# (visual spot-check: drone should be centred in overhead frame, not shifted to wall)

# ── Smoke 3 — WS-B mini (verify A2 obs columns in GIF frames) ─────────────────
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh 3 corridor

# Check: 3 GIFs exist under logs/uav_expert_data/gifs/corridor/
# Open one and confirm drone is not systematically at the wall

# ── Smoke 4 — WS-C local (verify B: D=12 tensor, numpy-only, no GPU) ──────────
python uav_expert_data_collect/mini_fm_sanity.py --n-episodes 50 --n-steps 200
# Pass: prints "Tensor shape: (B, 8, 12) ✅" and "RMS < 0.1 m ✅"
# Note: no sbatch script for WS-C — it is CPU-only and fast enough to run locally
```

**After all 4 smokes pass** → submit Phase 2 full runs below.

---

## Phase 2 — Re-collection and re-render commands

After auditing this changelog, submit in this order:

**E4 U3 — re-collect all 4 scenes** (adds `q` field to pickles):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty    500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  500
```

**WS-A re-render** (attitude-aware images, full re-collect):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh
```

Add `--no-skip` argument if existing images need to be replaced.

**WS-B re-render** (attitude-aware GIFs, full re-collect):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh
```

**WS-C — first run** (no re-collect needed, just run with updated D=12 config):

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/mini_fm_sanity.sh
```

---

## Phase 3 — Verification

```python
import pickle, glob, numpy as np

# Check E4 U3 pickles have 'q' field and correct obs shape
for scene in ['empty', 'corridor', 's_curve', 'pillars']:
    eps = [pickle.load(open(p, 'rb'))
           for p in glob.glob(f'logs/uav_expert_data/{scene}/**/*.pkl', recursive=True)]
    ep = eps[0]
    print(f"\n{scene}: {len(eps)} episodes")
    print(f"  obs shape : {ep['obs'].shape}   (expect (T, 9))")
    print(f"  q shape   : {ep['q'].shape}     (expect (T, 4))")
    print(f"  q[0]      : {ep['q'][0]}        (expect near [1,0,0,0] at hover start)")
```

WS-B spot-check: open a corridor L or R GIF — the FPV panel should show slight
nose-down pitch at mid-trajectory, not a level drone at all speeds.

WS-C pass criterion: tensor shape `(B, H=8, D=12)` printed; RMS < 0.1 m on held-out
empty-scene episodes; no NaN.

---

## Cross-references

| Document | Content |
|---|---|
| [`PLAN.md`](PLAN.md) | Full rationale for each change |
| [`../../Epoch4_expert_data/U2/CLOSURE.md`](../../Epoch4_expert_data/U2/CLOSURE.md) | E4 U2 final state (obs 9D, no q field) |
| [`../../Epoch4_expert_data/U2/Fix_1/CHANGELOG.md`](../../Epoch4_expert_data/U2/Fix_1/CHANGELOG.md) | stats_validator Fix_3 — same column-shift pattern as A1/A2 |
| `quadrotor_modified.xml` line 35 | The one line Change C edits |
