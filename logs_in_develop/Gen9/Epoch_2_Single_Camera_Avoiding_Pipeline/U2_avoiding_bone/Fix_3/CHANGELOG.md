# Gen9 E2 U2 Fix_3 — Root fix: `VisualAgent` B=1 vs state-based B=20 fan

**Date**: 2026-06-06  
**Status**: ✅ Fixed  
**Symptom reported**: Col-5 MPC-foresight lines showing but "incorrect" (bad) compared to
FM K=20 state-based avoiding eval  
**Parent**: [`../Fix_2/CHANGELOG.md`](../Fix_2/CHANGELOG.md)

---

## Why Fix_2 was incomplete

Fix_2 correctly wired up the pipeline (VisualAgent returns `planned_xy`, col-5 plots it).
The extraction logic was correct: `traj[:, :, 4:6]` → unnorm → c_xy. **But** it used
`batch_size = 1` (single trajectory sample per step).

The state-based FM K=20 eval calls `policy(conditions={0: obs}, batch_size=20)` which
draws **20 independent ODE trajectories** from different random seeds.  The col-5 plot
shows up to 4 of these, forming a **fan** that looks coherent and visually matches the
actual rollout distribution.

With B=1, col-5 shows a **single** trajectory that may be noisy or atypical — the user
sees "lines but feels incorrect."

---

## Root cause

`VisualAgent.predict()` hardcoded `bp_b = bp_t.unsqueeze(0).unsqueeze(0)` → shape
`(1, 1, C, H, W)`.  The ODE loop (`p_sample_loop`) initialises noise from
`x = 0.5 * torch.randn(shape)` — with B=1, only one noise seed is drawn.

State-based `Policy.__call__` uses `batch_size=args.batch_size` (typically 20), so the
ODE draws 20 independent noise seeds → 20 diverse trajectory samples → fan.

---

## Fix

`VisualAgent.__init__` gains a `plan_batch_size=4` parameter (4 ≈ `min(20, 4)` from
state-based col-5 loop `range(min(args.batch_size, 4))`).

In `predict()`:
```python
B = self.plan_batch_size
bp_b  = bp_t.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B, 1, C, H, W)
obs_b = obs_t.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)        # (B, 1, 4)
...
# traj shape: (B, H, 6)
obs_norm_traj = traj[:, :, 2:].detach().cpu().numpy()           # (B, H, 4)
obs_raw_traj  = self.obs_normalizer.unnormalize(obs_norm_traj)  # (B, H, 4)
planned_xy    = obs_raw_traj[:, :, 2:4]                         # (B, H, 2)
```

The condition is repeated (same image/obs for all B), but each ODE sample starts from a
different random noise → B diverse trajectory samples, matching the state-based fan.

The col-5 plot loop `for k in range(traj_np.shape[0])` already handles arbitrary B.

---

## Files touched

| File | Change |
|---|---|
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | `VisualAgent`: add `plan_batch_size=4`; repeat batch; return `(B,H,2)` |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Same change |

---

## Before / after

| Metric | Fix_2 (B=1) | Fix_3 (B=4) |
|--------|-------------|-------------|
| Trajectories per save step | 1 (single line) | 4 (fan — like state-based) |
| `traj shape returned` | `(1, H, 2)` | `(4, H, 2)` |
| Col-5 visual | one thin line, may look noisy | 4-line fan consistent with K=20 state-based |

---

## Note

B=4 keeps inference cost 4× per replan step. For eval (not real-time), this is
acceptable. If compute is tight, `plan_batch_size=1` can be passed explicitly.
