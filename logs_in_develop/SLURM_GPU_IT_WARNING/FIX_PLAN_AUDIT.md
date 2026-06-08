# FIX_PLAN.md — Audit Report

**Auditor:** Antigravity  
**Date:** 2026-06-08  
**Verdict: 5 issues found (2 significant, 3 minor)**

---

## Issue 1 — Heading Count Typo (MINOR)

**Location:** Section 3, Group A heading (line 59)

The heading reads:

> **Group A — GPU-allocated, EGL set, no pinning (20 scripts — add 4-line block)**

But the table that follows contains **25 rows**, and the subtotal on line 89 correctly says **25**. The heading should say **25**, not 20.

---

## Issue 2 — `EGL_VISIBLE_DEVICES` Is Not a Real Variable (SIGNIFICANT)

**Location:** Section 2, Fix Block (line 48)

The proposed 4-line fix block includes:

```bash
export EGL_VISIBLE_DEVICES="$ALLOCATED_GPU"
```

**`EGL_VISIBLE_DEVICES` does not exist.** It is not recognized by NVIDIA drivers, the EGL specification, or MuJoCo. This line is a complete no-op — it sets an environment variable that nothing reads. The plan presents it as if it provides EGL-level device isolation alongside `MUJOCO_EGL_DEVICE_ID`, which is misleading.

**Risk:** If the plan is executed as-is, this fake variable creates a false sense of security. Anyone reading the script will assume EGL is properly fenced, but the fencing is entirely dependent on `MUJOCO_EGL_DEVICE_ID` alone. Any non-MuJoCo EGL code in the same process (e.g., a custom OpenGL renderer) will **not** be isolated.

**Fix:** Remove this line from the block entirely, or replace it with a comment explaining that no general EGL device-pinning variable exists.

---

## Issue 3 — CUDA ↔ EGL Device Index Mapping Is Not Guaranteed (SIGNIFICANT)

**Location:** Section 2, Fix Block (lines 45–48)

The fix block does:

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```

This assumes that SLURM's `CUDA_VISIBLE_DEVICES` value (a physical GPU index like `2`) maps 1:1 to the EGL device index that `eglQueryDevicesEXT()` returns. **This is not guaranteed.**

| Index Source | Ordering Basis |
|---|---|
| `CUDA_VISIBLE_DEVICES` (SLURM) | Physical GPU index / PCI bus order |
| `MUJOCO_EGL_DEVICE_ID` | EGL device enumeration order (`eglQueryDevicesEXT()`) |

With `CUDA_DEVICE_ORDER=PCI_BUS_ID` these orderings **usually** align on standard NVIDIA setups, but the EGL driver is not contractually bound to enumerate in PCI bus order. On the i6-gpu-1 node (or any node with non-standard topology, passthrough GPUs, or mixed GPU models), the mapping can diverge.

**Recommendation:** The fix block should be accompanied by a **one-time verification script** that runs on each compute node to confirm EGL↔CUDA index alignment. Example:

```python
#!/usr/bin/env python3
"""Run on each node to verify EGL device order matches CUDA/nvidia-smi order."""
import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco

# Query EGL devices
try:
    from mujoco.egl import egl_ext as EGL
    devices = EGL.eglQueryDevicesEXT()
    print(f"EGL sees {len(devices)} device(s)")
except ImportError:
    print("Cannot query EGL devices directly — verify with nvidia-smi during rendering")

# For each EGL device ID, render a frame and check nvidia-smi to see which GPU activates
print("Set MUJOCO_EGL_DEVICE_ID=0,1,2,... and monitor nvidia-smi to build translation table")
```

If the mapping does **not** align, the fix block needs a translation table or a runtime probe.

---

## Issue 4 — Template Update Instruction Is Underspecified (MINOR)

**Location:** Section 6 (lines 141–143)

The plan says:

> The two template files (…) should also be updated so all future scripts start with the correct block. The job template already has `--gres=gpu:1`.

But the job template (`2026_04_30_job_template.sh`) **does not contain** `MUJOCO_GL` or `PYOPENGL_PLATFORM` exports. The fix block instruction in Section 2 says "add immediately after the existing MUJOCO_GL / PYOPENGL_PLATFORM exports," but there are no such exports to add after.

**Fix:** Section 6 should explicitly state:

> Add **both** the EGL exports (`MUJOCO_GL`, `PYOPENGL_PLATFORM`) **and** the GPU isolation block to the job template's Environment Setup section (after the conda activation). Comment them out or gate them behind a flag for scripts that don't need rendering.

---

## Issue 5 — `EGL_DEVICE_ID` Legacy Claim Is Inaccurate (MINOR)

**Location:** Section 2, after the fix block (line 53)

The plan states:

> `EGL_DEVICE_ID` is not a MuJoCo variable — it had no effect anyway.

This is **partially incorrect**. `EGL_DEVICE_ID` was a recognized variable in older `mujoco-py` and `dm_control` versions. Whether it "had no effect" depends on which MuJoCo binding the project uses. If the project has ever used `mujoco-py` or older `dm_control`, that variable may have been actively consumed.

The dismissal "it had no effect anyway" should be softened to: "It is a legacy variable from `mujoco-py` / older `dm_control`. Verify it is not consumed by the current MuJoCo version before removing."

---

## Corrected Fix Block

Based on the audit, the fix block should be revised to:

```bash
# ─── EGL GPU Isolation ──────────────────────────────────────────────────────
# Forces MuJoCo's EGL renderer onto the same physical GPU as CUDA compute.
# Without this, EGL defaults to /dev/dri/renderD128 (GPU 0) regardless of
# CUDA_VISIBLE_DEVICES.
#
# IMPORTANT: This assumes EGL device enumeration order matches PCI bus order.
# Run the node verification script to confirm (see FIX_PLAN_AUDIT.md Issue 3).
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
export MUJOCO_EGL_DEVICE_ID="$ALLOCATED_GPU"
```

**Changes vs. original:**
- Removed the fabricated `EGL_VISIBLE_DEVICES` line
- Added a comment documenting the EGL index assumption and pointing to verification

---

## Script Inventory Verification

The plan's file inventory was **verified against the codebase** and is **correct** for Groups A, B, and C — all files exist and have the exports described. The only error is the heading count (20 → 25).

| Check | Result |
|---|---|
| Group B files have `EGL_DEVICE_ID=0` | ✅ Confirmed (5/5) |
| Group C `collect.sh` has no `--gres=gpu` | ✅ Confirmed |
| Group A files have `MUJOCO_GL` + `PYOPENGL_PLATFORM` but no pinning | ✅ Confirmed (25/25) |
| No scripts missing from inventory | ✅ Confirmed |
| Pipeline/DA/load_results/extract scripts correctly excluded | ✅ Confirmed |
