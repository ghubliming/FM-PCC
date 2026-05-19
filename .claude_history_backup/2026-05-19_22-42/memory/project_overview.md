---
name: project-overview
description: "High-level summary of the FM-PCC project — what it is, its generation lineage, current active work, and key folder map"
metadata: 
  node_type: memory
  type: project
  originSessionId: c9ec8b03-7774-4a8a-a163-60517cad8227
---

# FM-PCC Project Overview

A robotics thesis project combining **Flow Matching** (generative trajectory models) with **DPCC** (Differentiable Predictive Constraint Control / SLSQP projector) for safe, vision-conditioned robot control in the D3IL simulation environment (aligning and avoiding tasks).

## Research Context
- **Why:** Build a safe visual robot policy that guarantees constraint satisfaction during trajectory denoising using an inline QP projector (SLSQP).
- **Comparison baseline:** DPCC (diffusion-based) vs FM (flow matching) for trajectory quality and safety.
- **Tasks:** D3IL `aligning-d3il` (push blocks with cameras) and `avoiding-d3il` (narrow-gap navigation).
- **SLURM cluster:** `vmknoll`, Conda env `FMPCC` (Python 3.10), W&B for logging.

## Generation Lineage (Gen1 → Gen7)

| Gen | Folder | What | Status |
|-----|--------|------|--------|
| Gen1-2 | `flow_matcher/` | State-only FM, uniform time, reversed ODE bug found+fixed | Done/archived |
| Gen3 | `flow_matcher_v3/` | SafeFlow-style continuous-time FM (FMv3) | Done |
| Gen3v2 | `flow_matcher_v3_ode_selectable/` | ODE solver add-on (Euler/RK4/Dopri5), deterministic benchmark suite | Done |
| Gen3v3 | `flow_matcher_v3_drifting/` | Drifting Engine (FM-D hybrid loss) | Done |
| Gen3v4 | `flow_matcher_v3_imeanflow/` | iMeanFlow (iMF) dual-velocity decomposition | Done |
| Gen4 | (Abandoned) | Visual avoiding — abandoned due to code coupling | Abandoned |
| **Gen5** | `ddpm_encdec_vision/` | Visual aligning DDPM pipeline, D3IL bridge (ResNet encoder + 1D U-Net) | **STABLE** |
| **Gen6** | `ddpm_encdec_vision/` + eval | DPCC projector added on top of Gen5 visual DDPM | **STABLE** (fix_5 resolved regressions) |
| **Gen6v3** | same codebase | Non-visual aligning trajectory addon | **Done** |
| **Gen6V4** | `diffuser_visual_aligning/` | New DPCC-native visual model with `AligningImgSequenceDataset` + `LimitsNormalizer` | **In progress** |
| **Gen7** | `fm_encdec_vision/` | Visual Flow Matching (FMv3ODE) decoupled sibling of Gen5/6 | **Just created (May 18)** |

## Key Fix: Gen6 fix_5 (May 18, 2026)
Gen6 had architectural regressions vs. the legacy working run:
1. `VisualUNet` was using `config.obs_dim=128` (dummy) → now hardcoded to 3 (real proprioception)
2. `Scaler` changed from `1e-2` std floor to `1e-12` → reverted to `1e-2`
Result: Gen6v3 is now stable, declared done after fix_5.

## Key Folders
- `ddpm_encdec_vision/` + `ddpm_encdec_vision_test/` — Gen5/Gen6 visual DDPM pipeline
- `diffuser_visual_aligning/` — Gen6V4 sandboxed DPCC-native visual model
- `fm_encdec_vision/` + `fm_encdec_vision_test/` — Gen7 visual FM pipeline
- `flow_matcher_v3_ode_selectable/` — Gen3v2 state-only FM (ODE selectable)
- `diffuser/` — Base DPCC code (state-only avoiding), source of `Projector`
- `d3il/` — Vendored D3IL simulation environment
- `config/aligning-d3il-visual.py` — All visual experiment config blocks
- `config/avoiding-d3il.py` — State-only DPCC/FM config blocks
- `Slurm_Codes/sbatch/Visual_Aligning/` — SLURM submit scripts for Gen5/6/7
- `Data_Analysis/` — DA tool (v1/v2/v3) for aggregating 834+ .npz eval results
- `logs_in_develop/` — All development logs, plans, and audits

## Trajectory Architecture
The visual DPCC model operates on 6D joint trajectories (H=8):
- `x_t = [action (3D dx/dy/dz), proprio (3D x/y/z)]` for visual aligning
- Visual conditioning via dual ResNet encoders (agentview + wrist camera), FiLM modulation into 1D U-Net
- SLSQP projector enforces workspace bounds + Euler dynamics constraints during denoising

**How to apply:** Always check which generation/folder is being discussed. The "active" production visual model is `ddpm_encdec_vision` (Gen6). Gen6V4 (`diffuser_visual_aligning`) is the next architectural step. Gen7 (`fm_encdec_vision`) is FMv3ODE visual.
