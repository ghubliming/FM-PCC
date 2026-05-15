# Visual Diagnostic Fidelity Fix Log

**Date:** 2026-05-15
**Path:** `logs_in_develop/gen5_rewire_existing_visual_models_plan/New_gen5/fix10_downsampling/`

## 1. Issue Description
During the implementation of the U-Net Downsampling fix ($H=2$ support), the diagnostic GIF generation in `eval_ddpm_encdec_vision.py` exhibited two critical failures:
1.  **Frozen View:** The `BPCageCam` (left camera) appeared static/frozen even as the robot moved.
2.  **Color Distortion:** The output images had "catastrophic" colors (inverted/thermal appearance) compared to the original D3IL benchmarks.

## 2. Root Cause Analysis

### A. Color Distortion (Overflow)
The simulation often returns pixel intensities with slight floating-point noise above $1.0$ (due to lighting calculations). When multiplying by $255$ and casting directly to `np.uint8`, these values "wrapped around" (e.g., $1.01 \times 255 = 257 \to 1$). This created the inverted color artifacts.

### B. Frozen View (Memory Pointers)
The simulator environment sometimes reuses the same memory buffer for observations. Without a deep copy, the diagnostic `video_frames` buffer was accidentally capturing pointers that did not update correctly during the planning step, or were being shadowed by static goal images.

## 3. Implemented Fix

The diagnostic capture block in `VisualAgentWrapper.predict` was hardened with the following logic:

```python
# 1. Force a deep copy of the observation to decouple from simulator memory
bp_frame = bp_image_np.copy()
ih_frame = inhand_image_np.copy()

# 2. Add safety clipping (0-255) to prevent floating-point wrap-around
bp_vis = (bp_frame.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
inhand_vis = (ih_frame.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

# 3. Restore BGR-to-RGB parity for D3IL/OpenCV consistency
bp_vis = cv2.cvtColor(bp_vis, cv2.COLOR_BGR2RGB)
inhand_vis = cv2.cvtColor(inhand_vis, cv2.COLOR_BGR2RGB)
```

## 4. Verification
- **Colors:** Restored to original D3IL benchmark quality.
- **Motion:** Both `BPCageCam` and `Inhand` views now update at 100% frequency (every simulation step).
- **Stability:** No impact on the $H=2$ padding or the 7-metric reporting logic.

---
**Status:** Resolved. Visual parity with legacy benchmarks restored.
