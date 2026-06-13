# Gen9 Epoch 2 — Architecture Comparison

**Date**: 2026-06-03

Side-by-side: the new **Visual Avoiding** pipeline vs the predecessor **Visual Aligning** (Gen6V4 / Gen7) vs **D3IL non-visual avoiding** (state-only baseline). Read the rightmost column first — it's what Gen9 Epoch 2 actually ships.

| Aspect | D3IL non-visual avoiding *(baseline)* | Visual aligning *(Gen6V4/7 predecessor)* | **Visual avoiding (Gen9 Ep 2 — NEW)** |
|---|---|---|---|
| **Task** | Dodge 6 fixed obstacles on a 2-D plane | Push-box → target slot in 3-D workspace | Dodge 6 fixed obstacles on a 2-D plane, with vision |
| **Trajectory dim** | 6 `[act(2) + obs(4)]` (state-only) | **9** `[act(3) + des_c_pos(3) + c_pos(3)]` | **6** `[act(2) + des_xy(2) + c_xy(2)]` |
| **Action dim** | 2 (Δx, Δy) | 3 (Δx, Δy, Δz) | **2** (Δx, Δy) |
| **Obs vector** | `[des_x, des_y, c_x, c_y]` (4-D) | `[des_c_pos(3), c_pos(3)]` (6-D) | **`[des_xy(2), c_xy(2)]`** (4-D, same as D3IL) |
| **Obstacle representation** | Implicit (model learns from data) | N/A — aligning has no obstacles | **Explicit DPCC `sphere_outside`** constraints in plan config (6 spheres, r=0.04 m) |
| **Horizon** | 1 (single-step BC) | 8 | **8** |
| **Cameras** | None | bp-cam + inhand-cam (dual) | **bp-cam only (single)** |
| **Image latent** | — | 128-D (2× ResNet-64, concatenated) | **64-D (1× ResNet-64)** |
| **Vision encoder** | — | `MultiImageObsEncoder` with both keys | `MultiImageObsEncoder` with `agentview_image` only |
| **Backbone (DPCC)** | small MLP / BeT / IBC etc. (D3IL choice) | `Flow_matcher_U_Net_v2` (1-D temporal U-Net) | **`Flow_matcher_U_Net_v2`** (unchanged class, `TRANSITION_DIM=6`) |
| **Backbone (FM)** | n/a | Same U-Net, `VisualFlowMatching` ODE wrapper | **Same U-Net**, `VisualFlowMatching` |
| **U-Net `dim_mults`** | n/a | (1, 2, 4, 8) | (1, 2, 4, 8) |
| **U-Net `dim`** | n/a | 32 | **32** |
| **FiLM conditioning** | n/a | `time_mlp(t) + latent(128)` per ResidualTemporalBlock | `time_mlp(t) + latent(64)` |
| **Diffusion engine (DPCC)** | varies | `VisualGaussianDiffusion` (K=100) | **`VisualGaussianDiffusion`** (K=100) |
| **Diffusion engine (FM)** | n/a | `VisualFlowMatching` (Beta α=1.5, β=1.0, 100 Euler steps) | **`VisualFlowMatching`** (same hyperparams) |
| **Apply-conditioning anchor** | — | `cond[0]` is 6-D obs at t=0 | `cond[0]` is **4-D** obs at t=0 |
| **DPCC projector** | n/a | `SafetyConstraints` (workspace bounds on `c_pos`) + `DynamicConstraints` | **`ObstacleConstraints` (6× `sphere_outside`)** + dynamics ← *the killer feature, only active in this variant* |
| **Episode length** | ~100 | ~400 (aligning tasks are long) | 200 (avoiding episodes max ~106) |
| **Train data class** | `Avoiding_Dataset` (D3IL) | `ParityAligningDataset` (FM-PCC) | **`ParityAvoidingDataset`** (FM-PCC, new) |
| **Train data source** | `state/*.pkl` | `state/*.pkl` + `images/{bp,inhand}-cam/*` | **`state/*.pkl` + `images/bp-cam/*`** (inhand dropped) |
| **State pickle keys used** | `robot.des_c_pos`, `robot.c_pos` | + `push-box.{pos,quat}`, `target-box.{pos,quat}` | `robot.des_c_pos[:,:2]`, `robot.c_pos[:,:2]` only |
| **Config file** | `config/avoiding-d3il.py` | `config/aligning-d3il-visual.py` | **`config/avoiding-d3il-visual.py`** (new, mirrors aligning's split) |
| **Eval YAML** | n/a | `config/visual_aligning_eval.yaml` | **`config/visual_avoiding_eval.yaml`** (new stub) |
| **D3IL Hydra config** | `avoiding_config.yaml` | `aligning_vision_config.yaml` | **`avoiding_vision_config.yaml`** (new) |
| **Slurm sbatch** | `Slurm_Codes/sbatch/templates/` | `Slurm_Codes/sbatch/{diffuser,fm}_visual_aligning/` | **`Slurm_Codes/sbatch/{diffuser,fm}_visual_avoiding/`** (new) |
| **Sim class** | `simulation.avoiding_sim.Avoiding_Sim` | `simulation.aligning_sim.Aligning_Sim` | `simulation.avoiding_sim.Avoiding_Sim` |
| **Gym env class** | `ObstacleAvoidanceEnv` | `Robot_Push_Env` | `ObstacleAvoidanceEnv` |

## Single-line architectural summary

**Gen9 Epoch 2 = Visual aligning's architecture (U-Net + FiLM + DPCC) downscaled from 9-D to 6-D and from dual-camera to single-camera, applied to avoiding's 2-D plane task — with DPCC's `sphere_outside` projector constraints finally doing the thing they were designed to do.**

## What's new vs aligning (the bits actually invented in Epoch 2)

1. **`ParityAvoidingDataset`** — 4-D obs from 2-D-sliced `des_c_pos`/`c_pos`; single-cam image stream.
2. **Single-camera `VisualUNet`** — `LATENT_DIM=64` not 128, `shape_meta` drops `in_hand_image`, `encode_visual()` takes one tensor.
3. **6 `sphere_outside` constraints in plan configs** — first time DPCC's `ObstacleConstraints` is engaged on real obstacles in this repo. Position list hardcoded from `avoiding_objects.py:get_obj_xy_list()`, radius 0.04 m.
4. **`config/avoiding-d3il-visual.py`** — new file mirroring the aligning visual split.
5. **`Avoiding_Img_Dataset` in upstream D3IL** — for D3IL-native training loops; FM-PCC's own pipeline uses the new `ParityAvoidingDataset` instead.

## What's NOT new (deliberately reused, byte-identical)

- U-Net backbone (`Flow_matcher_U_Net_v2`), FiLM mechanism, `MultiImageObsEncoder`, ResNet-64 image encoder
- `GaussianDiffusion` / `FlowMatchingODE` base engines
- `Projector` / `ObstacleConstraints` / `SafetyConstraints` / `DynamicConstraints` (all already supported `sphere_outside`)
- Trainer loop, config Parser machinery, `args_to_watch` checkpoint naming
- Slurm template structure
