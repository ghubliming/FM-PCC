# Gen11 Epoch 5 — Visual Collection, Trajectory Visualisation & FM Sanity Gate: CHANGELOG

**Author**: Antigravity (AI pair-programmer)  
**Date**: 2026-06-05  
**Branch**: `update_into_FM`  
**Status**: Files created — awaiting cluster execution.  
**Plan**: [`EPOCH5_PLAN.md`](EPOCH5_PLAN.md)

> **Scope**: Three workstreams (WS-A camera images, WS-B GIF generation,
> WS-C mini-FM sanity gate) that consume Epoch 4's 1769 state-only
> episodes. All files are additive — **zero existing files were modified**.

---

## Files Created

### WS-A — Camera Image Collection

| File | Role |
|---|---|
| `uav_expert_data_collect/collect_camera_images.py` | Standalone replay-and-capture script. Loads Epoch 4 pickles, injects `qpos`/`qvel` state per timestep via `mj_forward()` (no action replay), renders two camera streams (overhead `bp-cam` + body-mounted `track-cam`), saves as 96×96 BGR PNGs. |
| `Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh` | SLURM wrapper: sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `EGL_DEVICE_ID=0`, activates `FMPCC` conda env, runs collection script. Args: `$1`=max_episodes, `$2`=scene, `$3`=resolution. |

### WS-B — GIF / Video Generation

| File | Role |
|---|---|
| `uav_expert_data_collect/generate_trajectory_gifs.py` | Option B1 (standalone): replays state pickles in MuJoCo offscreen, stitches bp-cam + track-cam side-by-side (192×96 px), saves per-episode GIF via `imageio.mimsave()`. Supports `--mp4`, `--frame-stride`, text overlay with scene/homotopy/timestep info. |
| `uav_expert_data_collect/assemble_gifs_from_pngs.py` | Option B2 (CPU-only post-process): reads WS-A's saved PNGs, stitches bp + track side-by-side, assembles into GIF. No GPU or MuJoCo rendering needed — lightweight alternative when images are already on disk. |
| `Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh` | SLURM wrapper for B1. Args: `$1`=max_episodes, `$2`=scene, `$3`=`"mp4"` flag, `$4`=frame_stride. |

### WS-C — Mini-FM Sanity Gate

| File | Role |
|---|---|
| `uav_expert_data_collect/mini_fm_sanity.py` | Standalone sanity gate: loads ≤100 empty-scene episodes, converts to `(N, H=8, D=9)` chunks matching FM-PCC dataloader format, trains a tiny MLP flow-matching model (numpy-only, no PyTorch), evaluates RMS position error and action norm ratio, prints pass/fail verdict. |

### Documentation

| File | Role |
|---|---|
| `logs_in_develop/Gen11/Epoch5_visual_and_validation/EPOCH5_PLAN.md` | Full blueprint (created before this changelog) |
| `logs_in_develop/Gen11/Epoch5_visual_and_validation/CHANGELOG.md` | This file |

---

## Files Modified

**None.** All work is additive. Epoch 4 code (`uav_expert_data_collect/collect.py`, `generator.py`, `dataset_writer.py`, `stats_validator.py`) is untouched. Scene XMLs are untouched. D3IL is untouched.

---

## Key Implementation Details

### WS-A: State injection over action replay

| Aspect | Decision | Rationale |
|---|---|---|
| Replay method | `qpos/qvel` injection + `mj_forward()` | Avoids PID replay drift; guarantees pixel-perfect correspondence between stored obs and rendered frame |
| Overhead camera | `MjvCamera` free-camera spec (elevation=-90°, distance=5.0) | No XML modification needed; camera follows drone via `lookat = data.qpos[:3]` |
| Body camera | `"track"` camera from `quadrotor_modified.xml` (line 35) | Already exists: `mode="trackcom"`, body-mounted on X2 |
| Color convention | RGB→BGR via `cv2.cvtColor` before `cv2.imwrite` | Matches Gen9 and D3IL visual aligning convention (BGR on disk, loaders do BGR→RGB) |
| Skip-existing | Both `bp-cam/` and `track-cam/` dirs must be populated to skip | Guards against partial runs |
| Scene grouping | Episodes grouped by scene; model/renderer reused per scene | Avoids reloading MuJoCo model per episode |

### WS-B: GIF generation

| Aspect | Decision | Rationale |
|---|---|---|
| Frame layout | `[bp-cam | track-cam]` side-by-side → 192×96 px | Matches Gen7 visual aligning GIF convention (bp + inhand side-by-side) |
| Text overlay | Scene, homotopy, timestep burned into frame via `cv2.putText` | Quick human identification without opening the pickle |
| Frame subsampling | `--frame-stride N` keeps every Nth frame | Controls GIF file size (640 frames at 96×96 → ~5 MB per GIF) |
| Two options | B1=render-from-state (standalone), B2=assemble-from-PNGs (depends on WS-A) | B1 for independent execution; B2 for lightweight post-processing |

### WS-C: Mini-FM sanity gate

| Aspect | Decision | Rationale |
|---|---|---|
| Framework | Pure numpy (no PyTorch) | Zero-dependency verification of data flow; avoids framework installation issues |
| Model | 2-layer MLP, hidden=128 | Tiny — just needs to learn trivially simple empty-scene PID trajectories |
| Training | Numerical gradient descent | Very slow (~minutes) but correct and dependency-free |
| Chunk format | `(N, H=8, D=9)` where `D = [Δp_des(3) ‖ p(3), v(3)]` | Matches FM-PCC expected dataloader format |
| Pass criteria | Shape=(H,9) ✅, RMS<0.1m ✅, action_norm_ratio 0.5–2.0 ✅ | From EPOCH5_PLAN.md §4.3 |
| Normalisation | Per-feature z-score on training set | Standard practice; prevents magnitude imbalance across features |

---

## How to Run

### WS-A: Camera collection

```bash
# Smoke (1 episode, local with EGL)
export MUJOCO_GL=egl
python uav_expert_data_collect/collect_camera_images.py --max-episodes 1

# Full collection (SLURM)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh

# Single scene
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_camera_images.sh "" corridor
```

### WS-B: GIF generation

```bash
# Smoke (1 episode, local)
export MUJOCO_GL=egl
python uav_expert_data_collect/generate_trajectory_gifs.py --max-episodes 1

# Full generation (SLURM, with MP4)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" "" mp4

# Subsampled (every 3rd frame — smaller files)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh "" "" "" 3

# Option B2: assemble from existing PNGs (CPU, after WS-A)
python uav_expert_data_collect/assemble_gifs_from_pngs.py
```

### WS-C: Mini-FM sanity gate

```bash
# Run locally (no GPU needed — pure numpy)
python uav_expert_data_collect/mini_fm_sanity.py

# Custom config
python uav_expert_data_collect/mini_fm_sanity.py --n-episodes 50 --n-steps 500
```

---

## Output Layout (After All Workstreams Run)

```
logs/uav_expert_data/
  {scene}/{homotopy_safe}/{ep_id}.pkl          ← Epoch 4 (existing)
  images/
    bp-cam/{scene}/{homotopy_safe}/{ep_id}/
      0.png, 1.png, …                          ← WS-A
    track-cam/{scene}/{homotopy_safe}/{ep_id}/
      0.png, 1.png, …                          ← WS-A
  gifs/
    {scene}/{homotopy_safe}/{ep_id}.gif         ← WS-B
    {scene}/{homotopy_safe}/{ep_id}.mp4         ← WS-B (optional)
  empty/mini_fm_results.json                    ← WS-C
```

---

## What Is NOT Done (Out of Scope)

- FM-PCC training on the generated images *(Epoch 6)*
- Visual encoder / ResNet fine-tuning *(Epoch 6)*
- Domain randomisation (texture/lighting variation) *(Epoch 6+)*
- Per-scene montage GIF (4-up grid) — *(nice-to-have, not implemented)*
- DAgger / on-policy correction *(Epoch 7+)*

---

## Prerequisites

1. **Epoch 4 state pickles** must be present at `logs/uav_expert_data/{scene}/{homotopy}/{ep_id}.pkl`
2. **GPU node with EGL** for WS-A and WS-B (B1 option)
3. **Conda env `FMPCC`** with `mujoco`, `cv2`, `imageio`, `numpy`, `tqdm`
4. **No GPU needed** for WS-C (mini-FM) or WS-B Option B2

---

## Cross-References

| Document | Content |
|---|---|
| [`EPOCH5_PLAN.md`](EPOCH5_PLAN.md) | Full blueprint with risk register and execution order |
| [`../Epoch4_expert_data/CLOSURE.md`](../Epoch4_expert_data/CLOSURE.md) | Dataset stats, schema, fix history |
| [`../../Gen9/camera_image_from_state/CHANGELOG.md`](../../Gen9/camera_image_from_state/CHANGELOG.md) | Gen9 camera collection template |
| [`../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_13/UF13_nonvisual_gif_investigation.md`](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_13/UF13_nonvisual_gif_investigation.md) | Gen7 GIF generation template |

---

## Audit Fix (2026-06-05) — Mini-FM normalisation mismatch in `evaluate()`

**Status**: ✅ Fixed

### Bug (WS-C `mini_fm_sanity.py`)

`evaluate()` sampled from the model in **normalised** (z-scored) space but compared the result against **unnormalised** `eval_chunks` (raw metres). `sample_ode()` returns samples in the same space the model was trained on — and training used `train_norm = (train_chunks - data_mean) / data_std`. The RMS at line 235 therefore subtracted z-scores from metres, producing a meaningless large error that would fail the `< 0.1 m` gate regardless of whether the data pipeline is correct.

`data_mean` / `data_std` were computed in `main()` but never passed into `evaluate()`.

This is a true correctness bug in the gate itself: a healthy dataset would still report NO-GO. Because the gate's purpose is to greenlight Epoch 6, a false NO-GO is blocking.

### Fix

- `evaluate(model, eval_chunks, data_mean, data_std, ...)` — added the two normaliser args.
- De-normalise model output before comparison: `pred_flat = pred_norm * data_std + data_mean`.
- Updated the call site in `main()` to pass `data_mean, data_std`.

Ground truth stays in metres; predictions are now mapped back to metres, so the RMS and action-norm comparisons are on a common scale.

### Files touched

| File | Change |
|---|---|
| `uav_expert_data_collect/mini_fm_sanity.py` | `evaluate()` de-normalises predictions before RMS/action-norm comparison |

### Audited and confirmed correct (no changes)

| File | Verdict |
|---|---|
| `collect_camera_images.py` | ✅ state injection, overhead free-camera spec, RGB→BGR convention, per-scene model reuse all correct |
| `generate_trajectory_gifs.py` / `assemble_gifs_from_pngs.py` | ✅ not modified — out of this audit pass's reported issue |
| `mini_fm_sanity.py` chunking / shape gate | ✅ `episodes_to_chunks` correctly builds `(N, H=8, D=9)` with `[actions(3) ‖ obs(6)]` |
