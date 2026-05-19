# Fix 7 — Code Logic Effects Analysis (2026-05-19)

## Overview

Fix 7 is a series of three manual reverts to restore D3IL parity. This document traces
the **runtime logic path** each change affects and whether it resolves active failures.

---

## FIX_7.1 — Remove max_episode_length Plumbing

### What changed
- `Aligning_Sim.__init__`: removed `max_episode_length` parameter and stored field
- `Robot_Push_Env(...)`: no longer receives `max_steps_per_episode=...`
- `eval_ddpm_encdec_vision.py`: no longer passes that arg to `Aligning_Sim`

### Logic path effect
```
Before: eval script → Aligning_Sim(max_episode_length=N) → Robot_Push_Env(max_steps_per_episode=N)
After:  eval script → Aligning_Sim()                     → Robot_Push_Env()  [internal default]
```

`Robot_Push_Env` now uses its own internal `max_steps` default. If that default equals
what was being passed (e.g. 400), **behavior is identical**. If different, rollouts run
longer or shorter.

### Does it fix a problem?
**Yes — fixes server crash.** The `max_steps_per_episode` kwarg apparently caused a
failure at env construction time (reported in FIX_7.1 log). Removing it unblocks the
eval run entirely.

**No impact on trajectory quality.** This does not change what the model predicts,
only the episode termination budget.

---

## FIX_7.2 — Revert BGR→RGB Conversion in Dataset

### What changed
`d3il/environments/dataset/aligning_dataset.py` — image loading loop:

```python
# Before (broken):
image = cv2.imread(img).astype(np.float32)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # ← was added, now removed
image = image.transpose((2, 0, 1)) / 255.

# After (restored original D3IL):
image = cv2.imread(img).astype(np.float32)
image = image.transpose((2, 0, 1)) / 255.
```

### Logic path effect
`cv2.imread` returns BGR. Without conversion, channel 0 = Blue, channel 1 = Green, channel 2 = Red.

```
Training tensor:   [B G R] / 255.  (BGR, original D3IL order)
Eval image tensor: [B G R] / 255.  (BGR, same after revert)
```

With the removed conversion active:
```
Training: [B G R]      ← trained on BGR
Eval:     [R G B]      ← evaluated on RGB (cv2.cvtColor added in drift commit)
```
That is a **systematic distribution shift on channel 0 and channel 2** for every
visual observation. The ResNet encoder would extract features from the wrong channels —
blue objects look red, red objects look blue. The visual embedding fed into the
diffusion model is corrupted every inference step.

### Does it fix a problem?
**Critical fix if legacy checkpoints were trained on BGR.** Any DDPM-EncDec or
Visual-DPCC checkpoint trained before the drift commit (6f42a73) was trained on BGR.
Evaluating with the RGB conversion active gives wrong visual features → completely
wrong conditioning → model never recovers context → 0% success even if the action
head is correct.

**For new K=100 training:** ensures train and eval use the same BGR pipeline →
no channel mismatch possible.

---

## FIX_7.3 — Revert D3IL Simulation / Camera / Physics Drift

### 7.3-A: aligning_sim.py behavior parity

Three sub-changes:

#### a) eval_on_train removed
```python
# Drift version allowed:
obs = env.reset(random=False, context=train_contexts[context])  # eval on TRAIN contexts

# Restored original:
obs = env.reset(random=False, context=test_contexts[context])   # eval on TEST contexts
```
**Logic effect:** The drift version would silently evaluate on training contexts (in-distribution),
inflating apparent success rate and masking true generalization. Restored original evaluates
on held-out test contexts. **This is the correct behavior for measuring generalization.**

If success was 0% even on train contexts → the model is definitively broken (not just overfitting).
If success was nonzero on train contexts but 0% on test contexts → model overfits.
Either way, restoring test-context eval gives the honest number.

#### b) CPU pinning restored
```python
assign_process_to_cpu(os.getpid(), cpu_set)   # restored for all modes
```
**Logic effect:** Pins each worker to a specific CPU core, preventing OS scheduling
interference during MuJoCo stepping. Without pinning, context switches can desynchronize
the physics clock from the control loop → nondeterministic stepping → subtle trajectory
corruption under load.

#### c) Return signature restored
```python
return success_rate, mode_encoding   # restored (removed mean_distance tuple)
```
**Logic effect:** Any caller unpacking exactly 2 values from the return would crash if 3
were returned. Restored parity prevents silent unpack errors.

---

### 7.3-B: BPCageCam constructor parity

```python
# Drift version (named key):
self.bp_cam = BPCageCam("bp_cam_key")   # non-original arg

# Restored original:
self.bp_cam = BPCageCam()               # no positional key
```
**Logic effect:** The named key was forwarded into the camera pipeline. If the MuJoCo
scene does not define a named camera with that exact key, `render()` returns a blank
frame or crashes. Restoring the original constructor ensures the hardcoded camera
position (`init_pos=[1.05, 0, 1.2]`, `init_quat=[...]`) is used directly without
a name lookup.

**Impact:** Visual observations during eval were either blank or causing a runtime
crash when the key was active. Restoring original camera wiring means the bp_cam
frame is correctly rendered and delivered to the ResNet encoder.

---

### 7.3-C: rod:tip collision flags restored

```xml
<!-- Drift (colliding): -->
<geom ... contype="1" conaffinity="1" name="rod:tip"/>

<!-- Restored (non-colliding): -->
<geom ... contype="0" conaffinity="0" name="rod:tip"/>
```
**Logic effect:** With `contype=1/conaffinity=1`, the invisible rod tip sphere
participates in the MuJoCo contact solver. Any time the tip is near an object or table,
the solver generates contact forces — phantom forces not present during the original
D3IL data collection. This biases the robot's real dynamics away from those in the
training dataset:

```
Training data:   collected without rod:tip contacts
Drift eval:      generates spurious contact forces from rod:tip
→ real c_pos trajectory diverges from des_c_pos even with perfect actions
```

Restoring `contype=0/conaffinity=0` removes phantom contacts → physics matches
training distribution → SLSQP projector constraints are consistent with actual
env dynamics.

---

## Combined Effect Matrix

| Fix | Layer | Failure mode addressed | Fixes 0% issue? |
|-----|-------|------------------------|-----------------|
| 7.1 | Env construction | Server crash on `max_steps_per_episode` kwarg | Unblocks eval from crashing |
| 7.2 | Visual preprocessing | BGR/RGB channel swap → corrupted ResNet features | Yes — critical for visual correctness |
| 7.3-A | Sim rollout | eval_on_train (in-distribution leak); CPU desync | Eval result is now honest |
| 7.3-B | Camera rendering | Named key crash → blank frames → bad visual obs | Yes — camera was broken |
| 7.3-C | Physics | Phantom rod:tip contacts → wrong dynamics | Reduces trajectory drift |

---

## Will Fix 7 Alone Achieve Nonzero Success?

**Probably not with K=16 checkpoints.**

Fix 7 corrects the **evaluation infrastructure** — camera, physics, visual pipeline, and
episode termination. These are necessary conditions for correct eval, but the K=16
checkpoint's weights are still trained with only 16 denoising steps. At t=15,
`sqrt_recip_alphas_cumprod ≈ 11×` amplification causes `x_recon` std ≈ 10.5,
well outside training distribution. The chain diverges regardless of eval infrastructure.

**Fix 7 is essential groundwork for the K=100 run.** When the K=100 checkpoint
is trained with the corrected BGR pipeline (FIX_7.2) and evaluated with the corrected
camera/physics (FIX_7.3), the evaluation will be a fair and correct measurement.
Without Fix 7, even a perfect K=100 model could silently fail due to channel swap or
camera crash.

### Priority order for reaching nonzero success:
1. Fix 7 (infrastructure) — **done**
2. K=100 retrain with `clip_denoised=False` — **pending**
3. Eval K=100 checkpoint with corrected infrastructure — **pending**
