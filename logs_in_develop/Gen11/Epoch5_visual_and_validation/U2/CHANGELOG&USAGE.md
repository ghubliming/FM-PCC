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

### How WS-A and WS-B relate

**WS-B (GIFs) is fully independent of WS-A (PNGs).** Both scripts do the same
state-injection loop and render from the same pickles; they just write different outputs:
- WS-A → numbered PNGs (training data for Epoch 6 visual FM)
- WS-B → GIFs (human inspection only — not consumed by training)

You can run WS-B alone at any time.  **For a quick visual smoke test, WS-B on 3 episodes
is the cheapest way to confirm the obs column fix (A2) and XML load (C) in one job.**
WS-A does not need to run first.

### Do we need WS-C?

**No, not before the smoke test.** WS-C is a mini-FM training gate — it tests that
the *Epoch 6 training pipeline* can load the data correctly.  The data correctness itself
was already verified and closed in E4 U2 (stats_validator, Fix_1 CLOSURE, 1769 episodes ✅).
WS-C is optional: run it only when you are about to start Epoch 6 FM training and want a
final sanity check on the D=12 tensor dimension.  Skip it for now.

### Recommended smoke sequence (2 jobs only)

```bash
# ── Smoke 1 — E4 U3 mini: verify 'q' field written to pickle ─────────────────
# Collects 5 empty episodes with the new generator/dataset_writer (D-prep).
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty 5

# After job completes — check q field exists:
python -c "
import pickle, glob
ep = pickle.load(open(sorted(glob.glob('logs/uav_expert_data/empty/**/*.pkl', recursive=True))[0], 'rb'))
print('obs shape:', ep['obs'].shape)   # expect (T, 9)
print('q   shape:', ep['q'].shape)     # expect (T, 4)  ← new field
print('q[0]     :', ep['q'][0])        # expect near [1, 0, 0, 0] at hover start
"

# ── Smoke 2 — WS-B mini: verify obs column fix visually ──────────────────────
# Generates 5 corridor GIFs. WS-A does NOT need to run first — WS-B is standalone.
# Covers: A2 obs columns, Change C XML load, D-prep quat fallback (identity, no crash).
#
# sbatch args: $1=max_episodes  $2=scene  $3="mp4"|""  $4=frame_stride
#
# WHY SLOW: default stride=1 renders EVERY frame (~274 frames/ep × GPU overhead = ~16s/ep).
# USE stride=5: renders every 5th frame → ~3s/ep → 5 eps done in ~15s total.
# GIFs are still readable at 1/5 frames — just slightly choppier.
#
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh 5 corridor "" 5
#                                                                              ^ep ^scene ^mp4 ^stride

# After job completes — open a GIF and confirm:
#   bp-cam panel: drone is centred in corridor, NOT pushed against the wall
#   track-cam panel: renders without crash (XML Change C loaded OK)
#   (Drone will still be level — quat tilt only kicks in after full E4 U3 re-collect)
```

**After both smokes pass → submit Phase 2 full runs below.**

### Frame stride reference

| Stride | Frames rendered | Speed vs stride=1 | Use case |
|---|---|---|---|
| `1` | Every frame | 1× (slowest) | Full-quality archive GIFs |
| `3` | Every 3rd frame | ~3× | Full-run default (still smooth enough) |
| `5` | Every 5th frame | ~5× | Smoke test / quick inspection |
| `10` | Every 10th frame | ~10× | Ultra-fast sanity check only |

Quick run — all scenes, fast (stride 5):
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" "" "" 5
#                                                                              ^all ^all ^no-mp4 ^stride
```

Full WS-B run with stride 3 (recommended for Phase 2 archive):
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" "" "" 3
```

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
