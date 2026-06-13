# Gen9 Epoch 2 — Fix-2: `_obs_dim` hardcoded to 6 (aligning) in both train scripts

**Date**: 2026-06-03
**Status**: ✅ Fixed (uncommitted)
**Triggered by**: `temp/debug_Gen9E2/outputs` — Slurm job 21144 (commit `0e47d73`, after Fix-1 land).
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md), [`../Fix_1/CHANGELOG.md`](../Fix_1/CHANGELOG.md)

---

## 1. Symptom

After Fix-1 cleared the package-import crash, the trainer got past dataset/model construction and crashed at the first training step:

```
File "fm_visual_avoiding/utils/training.py", line 124, in train_epoch
    loss, infos = self.model.loss(*batch)
File "fm_visual_avoiding/models/visual_gaussian_diffusion.py", line 57, in loss
    return self.p_losses(x, cond, t)
File "fm_visual_avoiding/models/diffusion.py", line 283, in p_losses
    loss, info = self.loss_fn(v_pred, v_target)
File "fm_visual_avoiding/models/helpers.py", line 187, in forward
    weighted_loss = (loss * self.weights).mean()
RuntimeError: The size of tensor a (6) must match the size of tensor b (8) at non-singleton dimension 2
```

The trajectory tensor is `(B, H, 6)` (correct: action_dim=2 + obs_dim=4). The loss-weights tensor is `(H, 8)` (wrong — built with `transition_dim=8`). Broadcast fails at dim 2: 6 vs 8.

Also visible in the same log (`outputs:85`):
```
observation_dim: 6              ← WRONG (should be 4 for avoiding visual)
```

## 2. Root cause

Both train scripts inherited an aligning-era hardcode on line 228:

```python
_obs_dim = 6 if _if_vision else 20   # UF-17: non-visual uses 20D obs in trajectory
```

The `6` was aligning's visual obs dimension (`[des_c_pos(3), c_pos(3)]`). For visual avoiding the correct value is **4** (`[des_xy(2), c_xy(2)]`) — a fact already correctly captured in `config/avoiding-d3il-visual.py` (`obs_dim: 4`) and in `ParityAvoidingDataset.OBS_DIM = 4`. But neither was wired into the diffusion engine's `observation_dim` argument — the train scripts overrode `args.obs_dim` with the stale hardcoded `6`.

`FlowMatchingODE.__init__` then computed `self.transition_dim = observation_dim + action_dim = 6 + 2 = 8`, and `get_loss_weights()` built `dim_weights = torch.ones(8)`, producing `loss_weights` of shape `(H, 8)` (`diffusion.py:91-101`). At the first training step, this `(H, 8)` weight tensor failed to broadcast against the actual trajectory loss `(B, H, 6)`.

Why the audit didn't catch it earlier: the `_obs_dim = 6 if _if_vision else 20` line was a *self-consistent* aligning-era hardcode — perfectly correct for aligning, perfectly wrong for avoiding, syntactically clean either way. No Docker AST or grep step flagged it because "6" is a valid integer literal and the variable name `_obs_dim` matched the visual config's expected value. It only surfaces at runtime when the dataset's actual 6-D trajectories meet the model's 8-D loss weights.

## 3. Fix

Both train scripts updated (file:line):

**`diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py:228-235`** and
**`fm_visual_avoiding_test/train_fm_visual_avoiding.py:228-235`**:

```python
# Before (broken):
_obs_dim = 6 if _if_vision else 20   # UF-17: non-visual uses 20D obs in trajectory
diffusion_config = utils.Config(
    VisualGaussianDiffusion,   # or VisualFlowMatching for the FM script
    ...
    observation_dim=_obs_dim,   # 6D visual / 20D non-visual
    action_dim=args.action_dim, # 3D act: [dx, dy, dz]
    ...
)

# After (fix):
# Gen9 Ep 2 Fix-2: visual avoiding obs is 4-D [des_xy(2), c_xy(2)] (NOT 6).
# Aligning hardcoded _obs_dim=6 because its visual obs was [des_c_pos(3), c_pos(3)].
# transition_dim = action(2) + obs(4) = 6 for visual avoiding.
_obs_dim = 4 if _if_vision else 20   # visual=4 (avoiding 2-D), non-visual=20 (out of scope)
diffusion_config = utils.Config(
    VisualGaussianDiffusion,
    ...
    observation_dim=_obs_dim,   # 4D visual avoiding / 20D non-visual (not currently used)
    action_dim=args.action_dim, # 2D act: [dx, dy]
    ...
)
```

Also fixed the stale comment two lines up:
```python
# Before:  # ── 2. Model — VisualUNet with hardcoded transition_dim=9 ────
# After:   # ── 2. Model — VisualUNet with hardcoded transition_dim=6 (Gen9 Ep 2) ────
```

## 4. Files Touched

```
M  diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py   (1 logic line + 2 comments)
M  fm_visual_avoiding_test/train_fm_visual_avoiding.py           (1 logic line + 2 comments)
```

Total: **2 files, ~3 lines** of substantive change each.

## 5. Why the value isn't read from the config

For symmetry with the aligning visual path, `_obs_dim` is *hardcoded* in the train script (with a `4 if _if_vision else 20` guard) rather than read from `args.obs_dim`. This is deliberate — Gen6V4's Fix-5 lesson was that `args.obs_dim` can be a stale placeholder (set wrong in a legacy config copy and never noticed). The hardcode is a defensive guardrail.

The config (`config/avoiding-d3il-visual.py:'visual_avoiding_dpcc'.obs_dim: 4` and same for `fm_visual_avoiding`) is *also* set correctly so that downstream code paths consuming `args.obs_dim` get the right value. The two should agree; this fix aligns the train-script hardcode with the config.

If you ever change the visual avoiding state representation (e.g. add velocity → 8-D obs), update **both** the config AND this hardcode together.

## 6. Verification

| Check | Result |
|---|---|
| `_obs_dim = 4 if _if_vision else 20` in both train scripts | ✅ Confirmed by grep |
| AST parses on both train scripts | ✅ |
| `args.obs_dim` in `config/avoiding-d3il-visual.py` is `4` (matches hardcode) | ✅ |
| `args.action_dim` is `2`; `_obs_dim + action_dim = 6` matches `transition_dim` | ✅ |
| `ParityAvoidingDataset.OBS_DIM = 4`, `ACTION_DIM = 2`, `TRAJ_DIM = 6` | ✅ |
| `VisualUNet.TRANSITION_DIM = 6`, `LATENT_DIM = 64` | ✅ |

**Cluster-side rerun expectation**: training should now pass the first `p_losses` call. `loss_weights` will be shape `(H, 6)`, broadcasting cleanly against `(B, H, 6)` loss tensors.

## 7. Known pre-existing eval-side hardcodes (NOT fixed in this commit — separate Fix-N)

While auditing the train scripts I noticed the eval scripts still carry aligning-era hardcodes that will fail when eval runs. These are out of scope for Fix-2 (which targets the trainer crash) but flagged here so they don't surprise the next debug round:

| File:Line | Hardcode | Avoiding-correct value |
|---|---|---|
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:95` | `_target_obs_dim = trajectory_dim - 3` (assumes `action_dim=3`) | `trajectory_dim - 2` |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py:99` | Same | Same |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:104` | `pad = trajectory_dim - 9` (assumes aligning's 9-D traj) | `trajectory_dim - 6` (avoiding's 6-D) |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py:108` | Same | Same |
| Various | `'x': 6, 'y': 7, 'z': 8` (c_pos indices in aligning's 9-D layout) | `'x': 4, 'y': 5` (c_xy indices in avoiding's 6-D layout; z dropped) |

These will surface in Fix-3 if/when eval is run. **Do not preemptively fix without seeing actual eval failure output** — the eval pipeline has many task-specific helpers (workspace bounds, halfspace variants, MPC foresight diagnostics) that may need coordinated changes, and patching blindly risks introducing more bugs than it fixes.

## 8. Lesson for next time

When porting a train/eval pipeline from one task to another in this repo, **`grep` for hardcoded integers that match the source task's dimensions** (3, 6, 9, 20, 23 for aligning; 2, 4, 6, 20 for avoiding) — they're typically guardrails inherited from defensive hardcodes and need flipping. The grep target for this fix would have been:

```bash
grep -nE "= 6 if|= 9 if|trajectory_dim - 3|trajectory_dim - 9" \
    diffuser_visual_avoiding* fm_visual_avoiding*
```

— which would have caught this in Phase 1 had we run it. Adding to the parent CHANGELOG's smoke recipe as step 0.5.

## 9. Cross-reference

- Parent: [`../CHANGELOG.md`](../CHANGELOG.md) §2.2 (the train scripts were copied from aligning carrying these hardcodes).
- Sibling: [`../Fix_1/CHANGELOG.md`](../Fix_1/CHANGELOG.md) (stale `__init__.py` re-exports — Fix-1 cleared the import error so this Fix-2 could surface).
- Failure log: `temp/debug_Gen9E2/outputs` (Slurm job 21144, commit `0e47d73`).
- Diffusion init: `fm_visual_avoiding/models/diffusion.py:11-105` (`FlowMatchingODE.__init__` + `get_loss_weights`).
- Loss tensor shape: `fm_visual_avoiding/models/helpers.py:174-189` (`WeightedLoss.forward`).
