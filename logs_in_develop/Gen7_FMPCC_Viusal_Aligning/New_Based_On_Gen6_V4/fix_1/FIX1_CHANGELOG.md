# Gen7 Fix 1 — Revert Incorrect Phase 0 BGR/RGB Changes

**Date:** 2026-05-20
**Branch:** update_into_FM
**Scope:** Two files, eval + comment only. No training code changed.

---

## TLDR

Reverted two incorrect Phase 0 changes that violated the empirically certified Fix 11
BGR pipeline. Phase 0.1 removed `cvtColor(RGB2BGR)` from `aligning.py`, creating an
inference-RGB vs training-BGR mismatch. Phase 0.5 rewrote the `aligning_sim.py` comment
to falsely claim "model trains on RGB." Both have been corrected.

---

## Root Cause

During Gen7 Phase 0 implementation, the author incorrectly concluded that:
- `cv2.imread + cvtColor(BGR2RGB)` on imageio/PIL-saved files produces RGB
- Therefore `aligning.py` should remove its `cvtColor(RGB2BGR)` to make
  inference RGB, matching a claimed RGB training pipeline

This reasoning contradicts the empirically validated FIX11_CHANGELOG.md conclusion.

---

## Authoritative Pipeline (from FIX11_CHANGELOG.md, lines 37–40)

> "The dataset images on disk are stored in RGB byte order (saved by the D3IL demo
> pipeline via imageio/PIL). cv2.imread reads them as 'BGR' (OpenCV convention), then
> cvtColor(BGR2RGB) swaps back — **net result is the model trains on BGR**. The D3IL
> env also returns BGR. The no-flip path is correct: BGR → BGR."

**Training:** `cv2.imread(RGB-on-disk) + cvtColor(BGR2RGB)` → **BGR**
**Inference (fix11 certified):** `MuJoCo(RGB) → aligning.py cvtColor(RGB2BGR)` → **BGR**
**Result:** BGR ↔ BGR → MATCH ✓

---

## What Was Wrong

### Phase 0.1 — `aligning.py` cvtColor removal (INCORRECT)

**Was (broken Phase 0 state):**
```python
bp_image = self.bp_cam.get_image(depth=False)
# Phase 0 Gen7 fix: MuJoCo get_image() returns RGB natively.
# Training pipeline (_load_images) also produces RGB (cv2.imread→cvtColor(BGR2RGB)).
# Return RGB directly — no cvtColor needed, eliminates train/inference channel mismatch.
inhand_image = self.inhand_cam.get_image(depth=False)
return robot_pos, bp_image, inhand_image
```

**Effect:** Inference returned RGB. Training produces BGR. MISMATCH → divergence risk.

### Phase 0.5 — `aligning_sim.py` comment rewrite (INCORRECT)

**Was (broken Phase 0 state):**
```python
# Fix 11: no channel flip.
# Training: cv2.imread(BGR) → cvtColor(BGR2RGB) → RGB. Model trains on RGB.
# Inference: MuJoCo get_image() → RGB → cvtColor(RGB2BGR) → BGR (aligning.py:212).
# There IS a channel mismatch (training=RGB, inference=BGR), but fix7 empirically
# proved the ResNet encoder is robust to channel swap for this geometric task.
# The [::-1] flip (fix8) was reverted: divergence is checkpoint-driven, not channel-driven.
```

**Problems:**
1. "Model trains on RGB" — factually wrong per FIX11 (trains on BGR)
2. References "aligning.py:212 cvtColor" — which Phase 0.1 had already removed (stale reference)
3. Reframes the mismatch as "empirically OK" — masks that there should be no mismatch at all

---

## Changes Applied (Fix 1)

### `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py`

**Restored fix11 certified state:**
```python
bp_image = self.bp_cam.get_image(depth=False)
bp_image = cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)
inhand_image = self.inhand_cam.get_image(depth=False)
inhand_image = cv2.cvtColor(inhand_image, cv2.COLOR_RGB2BGR)

return robot_pos, bp_image, inhand_image
```

### `d3il/simulation/aligning_sim.py`

**Restored fix11 certified comment:**
```python
# Fix 11: no channel flip. Dataset images are stored RGB-on-disk;
# cv2.imread+cvtColor(BGR2RGB) in _load_images() accidentally produces BGR.
# The model is trained on BGR. The env also returns BGR (aligning.py:212).
# [::-1] introduced in fix8 incorrectly flipped to RGB → mismatch → divergence.
```

---

## Retraining Requirement

| Model | Retraining needed? | Reason |
|---|---|---|
| Gen6V4 (existing Slurm checkpoints) | **NO** | Trained with fix11 BGR pipeline (correct). After this revert, eval again sees BGR → matches training. |
| Gen7 (new FM model) | **NO** | Gen7 has not been trained yet. The reverted pipeline is the correct baseline for Gen7 training. When Gen7 training begins, training and inference will both use BGR — consistent. |

**Important:** If Gen7 had been trained with the broken Phase 0.1 state (inference RGB at eval time but training BGR from dataset), those weights would need to be discarded and retrained. Since no Gen7 training has occurred, this is not an issue.

---

## Files Changed

| File | Change |
|---|---|
| `d3il/environments/.../aligning.py` | Restored `cvtColor(RGB2BGR)` for bp_image and inhand_image |
| `d3il/simulation/aligning_sim.py` | Restored correct fix11 BGR pipeline comment |

---

## Certified Pipeline State (after Fix 1)

```
Dataset on disk (imageio/PIL, RGB byte order)
    ↓ cv2.imread + cvtColor(BGR2RGB)
Training frames: BGR  ←── model learns from BGR
                                    ↑
MuJoCo get_image() → RGB           │  MATCH
    ↓ aligning.py cvtColor(RGB2BGR) │
Inference frames: BGR ──────────────┘
    ↓ aligning_sim.py transpose + /255 (no flip)
Model input: BGR  ✓
```

This matches the fix7 empirically verified working state (action range `[-0.78, +0.99]`).
