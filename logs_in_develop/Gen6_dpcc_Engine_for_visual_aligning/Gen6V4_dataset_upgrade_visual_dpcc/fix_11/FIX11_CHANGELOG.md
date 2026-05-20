# Fix 11 Changelog — Revert BGR→RGB Flip in aligning_sim.py

**Date:** 2026-05-20  
**Branch:** update_into_FM  
**Scope:** Eval-only — one file, two lines. No model weights, no training code changed.

---

## Root Cause (from FIX11_INVESTIGATION.md)

Fix8 (commit `7ba1f07`) introduced a `[::-1]` channel flip in `aligning_sim.py`
intending to correct a BGR/RGB mismatch. The reasoning was:

> D3IL env explicitly converts `cvtColor(RGB2BGR)` → returns BGR.  
> Training `_load_images()` does `cvtColor(BGR2RGB)` → model trained on RGB.  
> Therefore flip BGR→RGB at eval time to match training.

This was **logically sound but empirically wrong.**

The decisive test (job 20560, 2026-05-20):

| Job | Eval code | Checkpoint | Norm. act range |
|---|---|---|---|
| 20551 (fix7) | no `[::-1]` | step 50000 | **[-0.78, +0.99]** ✓ |
| 20560 (debug) | with `[::-1]` | step 42000 | **[-85.93, +77.91]** ✗ |

Same training run (~same checkpoint age), same seed 6, same context 0.
**Only the eval code differed.** The flip is the bug.

**Why the flip is wrong:** The dataset images on disk are stored in RGB byte order
(saved by the D3IL demo pipeline via imageio/PIL). `cv2.imread` reads them as "BGR"
(OpenCV convention), then `cvtColor(BGR2RGB)` swaps back — net result is the model
trains on BGR. The D3IL env also returns BGR. The no-flip path is correct: BGR → BGR.

---

## Change

### `d3il/simulation/aligning_sim.py`

Reverted both image transpose calls (initial obs from `env.reset()` and per-step obs
from `env.step()`) to remove the `[::-1]` channel reversal.

```python
# Before (fix8, BROKEN — flips BGR→RGB, mismatches BGR-trained model):
bp_image = bp_image.transpose((2, 0, 1))[::-1].copy() / 255.       # BGR→RGB (A1)
inhand_image = inhand_image.transpose((2, 0, 1))[::-1].copy() / 255.  # BGR→RGB (A1)

# After (fix11, CORRECT — no flip, BGR matches training):
# Fix 11: no channel flip. Dataset images are stored RGB-on-disk;
# cv2.imread+cvtColor(BGR2RGB) in _load_images() accidentally produces BGR.
# The model is trained on BGR. The env also returns BGR (aligning.py:212).
# [::-1] introduced in fix8 incorrectly flipped to RGB → mismatch → divergence.
bp_image = bp_image.transpose((2, 0, 1)).copy() / 255.
inhand_image = inhand_image.transpose((2, 0, 1)).copy() / 255.
```

Applied at two locations:
- Line ~86-87: initial observation from `env.reset()`
- Line ~110-111: per-step observation inside the `while not done` loop

---

## What Is NOT Changed

| Item | Decision |
|---|---|
| `ParityAligningDataset._load_images()` `cvtColor(BGR2RGB)` | **Leave as-is.** The model trained on the BGR output of this pipeline. Changing it would break future retraining on the same dataset. |
| `eval_visual_aligning_dpcc.py` — removed `cvtColor` in `predict()` | **Leave removed.** That was visualization-only (GIF capture), not model input. |
| Projector A4+B1 fixes from fix8 | **Keep.** Logically correct, unrelated to the image format issue. |
| `clip_denoised=False` | **Keep.** Matches original DPCC design, enforced in eval script. |

---

## Expected Result After Fix 11

The next eval run should restore the fix7 `diffuser_train_set` action range:
`[-0.78, +0.99]` (or similar healthy range from whichever checkpoint is current).

With all three corrections now in place simultaneously for the first time:
1. **Fix 10** — `max_episode_length` wired to env (400 steps, proven baseline)
2. **Fix 11** — correct image channel order (BGR, matching training)
3. **Fix 8 projector** — A4+B1 projector bugs corrected

The next eval is the first run with a coherent, fully-consistent pipeline.

---

## Files Changed

| File | Change |
|---|---|
| `d3il/simulation/aligning_sim.py` | Removed `[::-1]` from 2 transpose calls |
