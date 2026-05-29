# UF-17 — Non-Visual Aligning Fix: Gen7 FM-PCC Changelog

**Date**: 2026-05-29  
**Branch**: `update_into_FM`  
**Scope**: Non-visual (state-only) aligning pipeline — Gen7 FM path  
**Plan**: [`PLAN_NON_VISUAL_FIX.md`](PLAN_NON_VISUAL_FIX.md)

---

## Summary

The non-visual aligning pipeline was structurally broken: `action_dim=2` (D3IL
convention) placed `c_pos` at trajectory dims 5-7, breaking the projector whose
dynamics wiring targets dims 6-8. The eval wrapper collapsed the 20D obs to a fake
6D vector, discarding all box and target information. This fix restores the pure
original DPCC principle — everything in the trajectory, `apply_conditioning` pins
full obs at step 0, no FiLM — with 3D actions matching the visual path.

---

## Changed Files

### `config/aligning-d3il-visual.py`

`ddpm_encdec_vision_nonvisual` block:

```python
# Before
'action_dim': 2,

# After
'action_dim': 3,   # UF-17: 3D velocity [dx,dy,dz] — matches visual path and DPCC principle
```

Trajectory: 22D → **23D** `[act(3) | des_c_pos(3) | c_pos(3) | box_pos(3) | box_quat(4) | tgt_pos(3) | tgt_quat(4)]`

---

### `fm_visual_aligning/datasets/sequence.py` — new class `StateOnlyAligningDataset`

New 23D trajectory dataset for non-visual training. No image loading.

- **Pickle keys**: `robot['des_c_pos']`, `robot['c_pos']`, `push-box['pos']`,
  `push-box['quat']`, `target-box['pos']`, `target-box['quat']`
- **Action**: `des_c_pos[t+1] - des_c_pos[t]` → 3D velocity
- **Obs**: 20D full state `[des_c_pos(3)|c_pos(3)|box_pos(3)|box_quat(4)|tgt_pos(3)|tgt_quat(4)]`
- **Trajectory**: `[act(3)|obs(20)]` = 23D
- **Conditions**: `{0: obs_norm[0]}` — no image keys
- **Normalizers**: `obs_normalizer` (20D), `act_normalizer` (3D) — saved to checkpoint

---

### `fm_visual_aligning/datasets/__init__.py`

Added `StateOnlyAligningDataset` to exports.

---

### `fm_visual_aligning/models/visual_gaussian_diffusion.py` — G1 fix

`VisualFlowMatching.loss()` — added `if_vision` guard:

```python
if not self.model.if_vision:
    # Non-visual: no image keys — route directly to base p_losses
    x = trajectories
    t = 1.0 - Beta(α, β).sample((B,))
    return self.p_losses(x, conditions, t)
# visual path unchanged below
```

Prevents crash on `conditions['primary_img']` access when dataset has no image keys.

---

### `fm_visual_aligning_test/eval_fm_visual_aligning.py` — G2 + G3 fixes

**G2 — `setup_dpcc_projector()` parameterised for 23D:**

- Added `trajectory_dim=9` parameter (backward-compatible default)
- `lb`/`ub` arrays padded with `±inf` for trailing dims: `np.full(pad, ±inf)`
- Halfspace `C_row` built with `trajectory_dim` instead of hardcoded `9`
- `Projector(transition_dim=trajectory_dim, ...)`
- `proj_obs_normalizer`: for 20D obs_normalizer, slices to 6D (des_c_pos + c_pos)
  for the projector's normalisation of constraint-relevant dims only

Call site detects `if_vision` and passes `trajectory_dim=23` for non-visual.

**G3 — `predict()` non-visual branch rewritten:**

Before (broken):
```python
obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])  # fake 6D
cond = {0: obs_6d_norm_6d}   # discards box/target
```

After (fixed):
```python
obs_20d_np = np.asarray(state, dtype=np.float64)   # full 20D from sim
obs_norm = obs_normalizer.normalize(obs_20d_np)    # 20D normalised
cond = {0: obs_anchor_20d}   # pins full obs at step 0 via apply_conditioning
```

Also fixed `curr_rollout_c_pos` to use actual `c_pos` from obs dims 3-5.

**c_pos extraction — obs-dim-agnostic:**

```python
obs_dim = self.obs_normalizer.mins.shape[0]   # 6 or 20
dummy[:, 3:6] = cpos_norm   # c_pos at obs dims 3-5 in both layouts
```

Works for both visual (6D obs) and non-visual (20D obs) without branching.

---

### `fm_visual_aligning_test/train_fm_visual_aligning.py`

- Detects `args.if_vision` to select dataset: `ParityAligningDataset` (visual) vs
  `StateOnlyAligningDataset` (non-visual)
- Sets `observation_dim = 6 if if_vision else 20` for `VisualFlowMatching`

---

## What This Achieves

| Property | Before UF-17 | After UF-17 |
|---|---|---|
| Trajectory dim | 22D (broken) | **23D** |
| Projector c_pos dims | 5-7 ❌ | **6-8** ✓ |
| Eval obs quality | Fake 6D (no box/target) | **Full 20D** ✓ |
| Architecture principle | Broken DPCC | **Pure DPCC** ✓ |
| FM training | Crashes on no-image batch | **Routes to base p_losses** ✓ |
