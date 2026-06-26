# collect.sh — MuJoCo GL Backend Fix

**Date:** 2026-06-09  
**Job that exposed this:** 21354 (Gen11 E4 U3 F1 pillar rerun)  
**File fixed:** `Slurm_Codes/sbatch/uav_expert_data/collect.sh`

---

## The Crash

After the Group C EGL cleanup removed `export MUJOCO_GL=egl`, the pillar job crashed immediately:

```
File ".../mujoco/gl_context.py", line 38, in <module>
    from mujoco.osmesa import GLContext as _GLContext
...
AttributeError: 'NoneType' object has no attribute 'glGetError'
```

The job ran 0 episodes and exited within 1 second.

---

## Root Cause: MuJoCo 2.3.7 gl_context.py behaviour

**Source:** `mujoco-2.3.7/mujoco/gl_context.py` lines 24–48 (verified from PyPI source).

```python
_MUJOCO_GL = os.environ.get('MUJOCO_GL', '').lower().strip()
if _MUJOCO_GL not in ('disable', 'disabled', 'off', 'false', '0'):
    if _SYSTEM == 'Linux' and _MUJOCO_GL == 'osmesa':
        from mujoco.osmesa import GLContext ...
    elif _SYSTEM == 'Linux' and _MUJOCO_GL == 'egl':
        from mujoco.egl import GLContext ...
    else:                                    # ← empty MUJOCO_GL falls here
        from mujoco.glfw import GLContext ... # ← or osmesa on headless nodes
```

When `MUJOCO_GL` is unset, MuJoCo's auto-detection path tries **osmesa** as a fallback.  
osmesa requires the OSMesa shared library, which is **not installed on i6-gpu-1** → crash.

---

## Why `MUJOCO_GL=egl` is also wrong for collect.sh

**Source:** `mujoco-2.3.7/mujoco/egl/__init__.py` line 65.

```python
EGL_DISPLAY = create_initialized_egl_device_display()   # ← MODULE-LEVEL
```

`create_initialized_egl_device_display()` calls:
1. `eglQueryDevicesEXT()` — queries all GPU devices
2. `eglGetPlatformDisplayEXT()` — opens a device handle
3. `eglInitialize()` — fully initialises the EGL display

This runs **at `import mujoco` time**, not when a `Renderer` is created.  
`collect.sh` has no `--gres=gpu` allocation, so SLURM does not assign a GPU.  
EGL would open whatever device it finds first (GPU 0) — **same violation as the original IT warning**.

---

## Correct Fix: `MUJOCO_GL=disabled`

**Source:** `gl_context.py` line 25 — the entire GL block is inside:

```python
if _MUJOCO_GL not in ('disable', 'disabled', 'off', 'false', '0'):
    # ALL backend imports are here
```

Setting `MUJOCO_GL=disabled` skips this block entirely:
- No osmesa imported → no crash
- No EGL imported → no device opened → **zero GPU footprint**
- Physics APIs (`MjModel`, `MjData`, `mj_step`, `mj_forward`) work without any GL backend

`collect.py` + `generator.py` use only physics APIs — confirmed by grep:
```
generator.py: mujoco.MjModel, mujoco.MjData, mujoco.mj_forward, mujoco.mj_step
collect.py:   (imports from generator, no direct mujoco.Renderer calls)
```

No `mujoco.Renderer` is ever created → `GLContext.__init__()` is never called → the disabled value is safe.

---

## Final State of collect.sh

```bash
export MUJOCO_GL="disabled"
# [DEBUG] GPU-leak check — uncomment once to verify MUJOCO_GL=disabled opens no DRM device.
# Expected output: "DRI fds: NONE — clean". Remove after verification.
# python3 -c "import os,mujoco; import subprocess; r=subprocess.run(['lsof',f'/proc/{os.getpid()}/fd'],capture_output=True,text=True); dri=[l for l in r.stdout.splitlines() if 'dri' in l.lower() or 'renderD' in l]; print('[ GPU-LEAK CHECK ] DRI fds:', dri or 'NONE — clean')"
```

No `PYOPENGL_PLATFORM` needed (PyOpenGL EGL is irrelevant when MuJoCo GL is disabled).  
No 3-line GPU-pinning block needed (no GPU allocated, and `CUDA_VISIBLE_DEVICES` is unset).

The debug check line runs inside the compute node process immediately after `import mujoco`
and prints to the SLURM output. Uncomment, submit once, verify, then comment back out.

---

## CHANGELOG Update

This corrects the Group C entry in `logs_in_develop/SLURM_GPU_IT_WARNING/CHANGELOG.md`.  
Net change from original script: `MUJOCO_GL=egl` → `MUJOCO_GL=disabled`, `PYOPENGL_PLATFORM=egl` removed, comment updated, debug check line added (commented out).
