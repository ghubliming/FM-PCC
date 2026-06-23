# Fix 3 — UAV-FM SLURM scripts: restore W&B + GPU-leak guard parity

## What was wrong

The Gen11 E6 UAV-FM job scripts (`Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh`,
`eval_fm_uav.sh`) were written from scratch instead of from the proven FMPCC
job template, so they silently dropped two pieces of behaviour that **every
other** train/eval `.sh` in this repo has:

1. **No W&B logging.** Training ran but registered nothing on Weights & Biases.
   - `train_fm_uav.sh` never sourced `$HOME/FMPCC/.wandb_api_key` and never
     passed `--use-wandb` to `train_fm_uav.py`. Since `--use-wandb` is
     `action='store_true'` (default off), the entire `wandb.init` block in
     `train_fm_uav.py` was skipped on every run.

2. **No GPU-leak guard.** `train_fm_uav.sh` had none of the EGL-device pinning;
   `eval_fm_uav.sh` set `MUJOCO_EGL_DEVICE_ID` but was missing the
   `[ GPU-CHECK ]` echo and the abort-on-mismatch check. This is the guard we
   added across the repo to stop MuJoCo/EGL from rendering on a GPU other than
   the Slurm-allocated one (cross-GPU "leakage").

## Reference scripts used as the source of truth

Instead of inventing, the canonical blocks were copied verbatim from the
already-working scripts:

- `Slurm_Codes/sbatch/fm_visual_avoiding/train_fm_visual_avoiding.sh`
- `Slurm_Codes/sbatch/fm_visual_avoiding/eval_fm_visual_avoiding.sh`
- `Slurm_Codes/sbatch/train_fmv3_ode_job.sh`
- `Slurm_Codes/sbatch/eval_fmv3_ode_job.sh`

All four share the identical GPU-leak guard and W&B login block reproduced below.

## Changes

### `Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh`

- Added the GPU-leak guard after the EGL exports:
  ```bash
  ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
  echo "[ GPU-CHECK ] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID"
  if [ "$MUJOCO_EGL_DEVICE_ID" != "${CUDA_VISIBLE_DEVICES%%,*}" ]; then
      echo "[ GPU-LEAK ] EGL device ($MUJOCO_EGL_DEVICE_ID) != CUDA (${CUDA_VISIBLE_DEVICES%%,*}) -- aborting"
      exit 1
  fi
  ```
- Added the W&B login block:
  ```bash
  if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
      export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
      export WANDB_MODE="online"
  fi
  ```
- Passed W&B flags to the trainer (inside the per-seed loop):
  ```bash
  python FM_v3_uav_test/train_fm_uav.py --scene "$SCENE" --seed "$seed" \
      --use-wandb --wandb-project FM-PCC-uav-fm --wandb-group "uav-$SCENE"
  ```
  Project name `FM-PCC-uav-fm` follows the existing `FM-PCC-visual-avoiding-FM`
  convention; group `uav-<scene>` clusters that scene's seeds in one W&B group.

### `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`

- Completed the GPU-leak guard (the `[ GPU-CHECK ]` echo + abort check were
  missing; the `MUJOCO_EGL_DEVICE_ID` export was already present).
- Added the W&B login block for parity. NOTE: `eval_fm_uav.py` does **not** log
  to W&B today, so this is inert — it's included only so the script matches the
  template and is ready if eval logging is added later.

## Not changed / why

- `aggregate_summaries.sh` — CPU-only stdlib roll-up, no GPU/EGL/MuJoCo, so no
  GPU-leak guard or W&B needed.
- `mini_fm_gate.sh` — runs the pure-numpy `mini_fm_sanity.py` (no torch, no
  MuJoCo rendering), so no EGL leak surface; left as-is.
- The seed-loop consolidation from F1 (one job per scene, seeds looped inside)
  is untouched — these flags ride along inside that same internal loop.

## Verify after sync to cluster

- `[ GPU-CHECK ]` line appears in both train and eval logs; job aborts with
  `[ GPU-LEAK ]` if EGL and CUDA devices disagree.
- A W&B run appears per (scene, seed) under project `FM-PCC-uav-fm`, grouped by
  `uav-<scene>`. Requires `$HOME/FMPCC/.wandb_api_key` to exist on the node
  (same precondition as all other FMPCC W&B jobs).
