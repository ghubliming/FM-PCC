# Fix #10: Restoring Horizon Flexibility via Auto-Padding

**Date:** May 15, 2026
**Issue:** `RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 2 but got size 1.`
**Target:** Restoring the ability to change `horizon` arbitrarily (matching FMPCC state-based benchmarks).

---

## 1. The Problem (Root Cause Analysis)

When reducing the `horizon` to 2 or 4, the training pipeline crashed during the U-Net upsampling phase.

### Mechanical Breakdown:
1.  **U-Net Depth:** The `UNet1DTemporalCondModel` is configured with 3 downsampling blocks.
2.  **Temporal Bottom-out:** A trajectory of length 2 can only be downsampled once ($2 \to 1$). When the model attempts a second downsample, the math fails.
3.  **The Mismatch:** During upsampling, the model uses `ConvTranspose1d` with stride 2. A size-1 feature becomes size-2. However, the corresponding skip-connection feature from the downsample path was size-1. 
    *   **Result:** `cat([size 2], [size 1])` $\implies$ **CRASH.**

---

## 2. Competitive Benchmarking (Why FMPCC worked)

We compared the Visual configuration with the legacy state-based **`config/avoiding-d3il.py`**.

*   **Discovery:** The state-based project uses **`use_padding: True`** in its datasets.
*   **Conclusion:** In FMPCC, the dataset automatically padded short trajectories to a "Safe Size" before they ever hit the U-Net. This "safety net" was missing in the new Visual `Aligning_Img_Dataset`.

---

## 3. The Implementation (VisualUNet Auto-Padding)

Instead of modifying the dataset (which would be task-specific), we modified the **`VisualUNet`** wrapper to make the entire visual pipeline architecturally robust.

### Code Changes (`VisualUNet.py`):
1.  **Init-Time Calculation:** 
    ```python
    self.padded_horizon = ((config.horizon + 7) // 8) * 8
    ```
    The model now calculates the nearest multiple of 8 (required for a 3-layer U-Net) at initialization.
2.  **Forward-Pass Padding:**
    ```python
    if T < self.padded_horizon:
        pad_len = self.padded_horizon - T
        x = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
        visual_emb = torch.cat([visual_emb, torch.zeros(B, pad_len, ..., device=x.device)], dim=1)
    ```
3.  **Output Cropping:**
    The model crops the final trajectory back to the user's requested horizon:
    ```python
    return out[:, :T, :]
    ```

---

## 4. Verification

*   **Status:** The pipeline now supports `horizon: 2` and `window_size: 2` out-of-the-box.
*   **Flexibility:** The user can now tune horizons for precision vs. foresight without triggering architectural errors.
*   **Audit Trail:** This fix restores parity with the legacy FMv3ODE/FMPCC benchmark standards.

---

**Fix Log completed for FM-PCC Diagnostic Phase 10.**
