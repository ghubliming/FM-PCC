# Hotfix 1 — projector shape crash on `dpcc-*` variants (`4x88 and 96x96`)

**Date:** 2026-06-25. Found via `temp/Gen11E7/output1` (corridor, seed 6, 2 trials, `--record gif`).

## What happened
`diffuser` ran fine (success=0, as expected — see U3 finding doc). The very next variant,
`dpcc-r`, crashed inside `Projector.project`:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x88 and 96x96)
```
(plus a benign `EGL_NOT_INITIALIZED` `__del__` warning on the way down — not the cause,
just noise from the abrupt process exit.)

## Root cause
`flow_matcher_v3_uav/models/diffusion.py:p_sample_loop` only calls the projector when
`self.goal_dim > 0`, and when it does it slices the trajectory first:
`x[:, :, :-self.goal_dim]`. `model_fm.goal_dim` comes from
`SequenceDataset.get_goal_dim()` (`flow_matcher_v3_uav/datasets/sequence.py`) — a
heuristic forked verbatim from FMv3ODE/D3IL that counts observation columns with zero
std across episode 0 and assumes they're trailing goal-conditioning columns. For the
corridor scene this heuristic found **1** such column (some incidentally-constant
channel, not a real "goal"), so `model_fm.goal_dim == 1` for this checkpoint.

`setup_dpcc_projector` was called with `trajectory_dim=12` (act 3 + obs 9) regardless,
so `Projector.Q` was built `96×96` (`12×horizon8`). But the trajectory the projector
actually receives is already sliced to 11 columns by the line above
(`88 = 11×8`) → shape mismatch.

## Fix
`_run_variant` now subtracts `model_fm.goal_dim` from the transition width passed to
`setup_dpcc_projector`:
```python
goal_dim = int(getattr(model_fm, 'goal_dim', 0))
traj_dim = int(dataset.observation_dim + dataset.action_dim) - goal_dim
```
Safe regardless of *which* column the heuristic flagged: the dynamics constraint only
binds indices 0–5 (`act`, `p_des`), never the trailing column(s) removed by the slice.

## Not changed (scope)
- `get_goal_dim()`'s heuristic itself — it's shared engine code (identical in
  `flow_matcher_v3_ode_selectable`); not touched, since the existing trained
  checkpoint's weights and conditioning behaviour are tied to whatever `goal_dim` it
  produced at train time (`train_fm_uav.py:253` reads the same `dataset.goal_dim`).
  Changing the heuristic would require retraining to stay consistent; out of scope for
  unblocking this eval run.
- No retrain needed for *this* fix: `goal_dim` only changes which trailing columns
  `apply_conditioning`/the projector treat as fixed — it doesn't change `transition_dim`
  (still `obs_dim+action_dim`), so model weights are unaffected.

## Next
Resubmit the same corridor smoke test; `dpcc-r`/`dpcc-c`/`dpcc-t` should now run the
SLSQP projector end-to-end. Watch for: (1) wall-clock — SLSQP × batch=4 × ~horizon8 ×
hundreds of FM steps may be slow; (2) whether projected variants show lower
`plan_cand_spread` / cleaner GIFs than `diffuser` on this multi-homotopy scene.
