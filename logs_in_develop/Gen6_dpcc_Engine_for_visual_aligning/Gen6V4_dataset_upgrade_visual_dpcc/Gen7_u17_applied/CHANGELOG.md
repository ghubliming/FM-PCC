# UF-17 Applied to Gen6V4 (DPCC) — Non-Visual Fix

**Date**: 2026-05-29  
**Branch**: `update_into_FM`  
**Source (Gen7 canonical)**: [`u_f_17_fix_non_visual/CHANGELOG_UF17_GEN7.md`](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_17_fix_non_visual/CHANGELOG_UF17_GEN7.md)  
**Plan**: [`u_f_17_fix_non_visual/PLAN_NON_VISUAL_FIX.md`](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_17_fix_non_visual/PLAN_NON_VISUAL_FIX.md)  
**Scope**: `config/aligning-d3il-visual.py`,
           `diffuser_visual_aligning/datasets/sequence.py`,
           `diffuser_visual_aligning/datasets/__init__.py`,
           `diffuser_visual_aligning/models/visual_gaussian_diffusion.py`,
           `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`,
           `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py`

> No DPCC-specific divergence from the Gen7 FM version. All changes are shared
> infrastructure applied identically to both eval stacks.

---

## Changes Applied

### `config/aligning-d3il-visual.py`

`action_dim: 2 → 3` in `ddpm_encdec_vision_nonvisual` block.
Trajectory: 22D → 23D `[act(3)|obs(20)]`.

### `diffuser_visual_aligning/datasets/sequence.py`

New `StateOnlyAligningDataset` class — 23D trajectory, no image loading.
Identical logic to `fm_visual_aligning.datasets.sequence.StateOnlyAligningDataset`.

### `diffuser_visual_aligning/datasets/__init__.py`

Added `StateOnlyAligningDataset` to exports.

### `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` — G1

`VisualGaussianDiffusion.loss()` — `if not self.model.if_vision:` guard routes
non-visual batches (no image keys) to base `p_losses()` directly.

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — G2 + G3

**G2**: `setup_dpcc_projector()` — `trajectory_dim` param, padded lb/ub,
`proj_obs_normalizer` sliced to 6D for projector normalisation.

**G3**: `predict()` non-visual branch — full 20D obs used for `apply_conditioning`;
`curr_rollout_c_pos` from actual c_pos (obs dims 3-5); obs-dim-agnostic c_pos
extraction using `obs_normalizer.mins.shape[0]`.

### `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py`

`if_vision` detection: routes to `StateOnlyAligningDataset` and sets
`observation_dim=20` when `args.if_vision=False`.
