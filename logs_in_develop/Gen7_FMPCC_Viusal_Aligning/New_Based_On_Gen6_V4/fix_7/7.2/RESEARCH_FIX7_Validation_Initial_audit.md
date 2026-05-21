# FIX-7 Validation Results — Counter Reset Is Insufficient

**Date**: 2026-05-21  
**Source logs**: `temp/For_Gen6V4/KEY_OUTPUTS_GEN7_3`  
**Git rev**: `fd27e21`  
**Prerequisite**: Read `fix_7/BUG_REPORT_7_Audit.md` and `fix_7/POSTMORTEM&change_log.md` first.  
**Status**: FIX-7 confirmed applied and executing — but **the contamination persists**. Root cause is deeper than `GLOBAL_MJ_ROBOT_COUNTER`.

---

## 1. Executive Summary

FIX-7 hypothesized that `MjRobot.GLOBAL_MJ_ROBOT_COUNTER` advancing from 0→1 during expert gen was the sole mechanism causing bp_image std to shift from 0.2093 (good) to 0.1978 (bad). The fix reset the counter to 0 after expert gen.

**Result**: The counter reset executes correctly (log line present), the robot body prefix is `rb0` in all cases — **but bp_image std is still 0.1978 when expert gen runs, and the bad trajectory still appears.** The counter was a downstream marker, not the causal root.

> [!CAUTION]
> **FIX-7 is confirmed insufficient.** The contamination mechanism survives the counter reset. The POSTMORTEM's regression diagnostic scenario (§4: "If bp_image std is still 0.1978 after this fix, contamination goes deeper than the body counter") has been triggered.

---

## 2. The 5-Job Validation Series

All 5 jobs ran at git rev `fd27e21` (FIX-7 applied), same seed 6, same model checkpoint, same node (i6-gpu-1), same YAML config. Only difference: whether the plans folder was cleaned (forcing expert gen to re-run) vs. left in place (expert gen skips).

| Job # | Slurm ID | Expert gen ran? | Plan folder state | bp_image std | inhand_img std | a0 | Final dist ctx0 | Clamps ctx0 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1st | 20643 | **NO** (skipped) | Stale from prior run, NOT cleaned | **0.2093** | 0.2490 | [-0.0054, 0.0405, -0.2432] | 0.218847 m | 2 | ✅ GOOD |
| 2nd | 20645 | **YES** (folder cleaned) | Fresh | **0.1978** | 0.2867 | [0.0171, 0.008, -0.5546] | 0.312711 m | 233 | ❌ BAD |
| 3rd | 20646 | **YES** (folder cleaned) | Fresh | **0.1978** | 0.2867 | [0.0171, 0.008, -0.5546] | 0.312711 m | 233 | ❌ BAD |
| 4th | 20647 | **NO** (skipped) | From 3rd run | **0.2093** | 0.2490 | [-0.0054, 0.0405, -0.2432] | 0.218847 m | 2 | ✅ GOOD |
| 5th | 20648 | **NO** (skipped) | From 3rd/4th run | **0.2093** | 0.2490 | [-0.0054, 0.0405, -0.2432] | 0.218847 m | 2 | ✅ GOOD |

> [!IMPORTANT]
> The pattern is **perfectly binary and 100% reproducible**:
> - Expert gen **skips** → bp_image std = 0.2093 → GOOD trajectory → 0.218847 m, 2 clamps
> - Expert gen **runs** → bp_image std = 0.1978 → BAD trajectory → 0.312711 m, 233 clamps
>
> This is exactly the same pattern as before FIX-7. The fix had zero impact on the outcome.

---

## 3. FIX-7 Counter Reset Is Confirmed Executing

All 5 jobs show the FIX-7 log line:

```
[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)
```

And all 5 jobs create `panda_tmp_rb0_*.xml` (robot body prefix `rb0`) for the variant env — **NOT `rb1`**. Examples:

- JOB 20643: `panda_tmp_rb0_b44877e0-...` (expert skipped, counter already 0)
- JOB 20645: `panda_tmp_rb0_700f460c-...` + `panda_tmp_rb0_80b66468-...` (expert ran, counter reset worked, variant got rb0)
- JOB 20647: `panda_tmp_rb0_673fda94-...` (expert skipped, counter already 0)

**The counter reset is working as designed.** The variant's robot always gets `rb0` body prefix. But the bp_image std is still wrong. This disproves the hypothesis that the `rb0`-vs-`rb1` body prefix difference was the cause of the 0.2093-vs-0.1978 bp_image std change.

---

## 4. What the Counter Was — A Correlated Marker, Not the Cause

The BUG_REPORT_7_Audit.md reasoned:

> "The robot body names (`rb0` vs `rb1`) are baked into the compiled MuJoCo XML. [...] The compiled scene geometry is **not** byte-identical when the robot body names differ. The MuJoCo renderer's camera attachment, body indexing, and light reflections can vary subtly [...]"

This reasoning is now **disproven by experiment**. With FIX-7:
- Robot body prefix = `rb0` in ALL jobs ✓
- Compiled XML body names = identical in ALL jobs ✓
- bp_image std = **still 0.1978 when expert gen ran** ✗

The `rb0`/`rb1` naming difference was an observable **correlate** of expert gen running (because the counter incremented), but it was never the **causal mechanism** changing the rendered image. Something else about the expert gen process contaminates the MuJoCo rendering pipeline.

---

## 5. Differential Symptom Analysis — The Two Image Signatures

The two bp_image std values are completely stable across all runs:

| Metric | Expert gen SKIPPED | Expert gen RAN |
|---|---|---|
| bp_image std | 0.2093 | 0.1978 |
| inhand_img std | 0.2490 | 0.2867 |
| a0 (normalized) | [-0.0054, 0.0405, -0.2432] | [0.0171, 0.008, -0.5546] |
| \|a0\| (normalized) | 0.2466 | 0.5549 |
| Horizon range | [-0.2432, 1.0063] | [-1.0703, 0.7127] |
| Final distance | 0.218847 m | 0.312711 m |
| Clamp events | 2 | 233 |

Key observations:

1. **Both images change** — `bp_image` std drops (0.2093→0.1978) while `inhand_img` std rises (0.2490→0.2867). This means the contamination affects the **entire MuJoCo renderer**, not just one camera.

2. **The contaminated trajectory is dramatically worse** — 233 clamp events vs 2, final distance 43% further (0.312 vs 0.219 m). The normalized a0 magnitude more than doubles (0.55 vs 0.25), with a much stronger z-component (-0.55 vs -0.24), indicating the model predicts a large downward motion from the contaminated observation.

3. **Both outcomes are perfectly deterministic** — JOBs 20645 and 20646 (expert ran, different Slurm processes) produce byte-identical results. JOBs 20643, 20647, 20648 (expert skipped) also produce byte-identical results. The contamination is deterministic, not stochastic.

---

## 6. Candidate Root Causes — What Survives `env.close()` + Counter Reset?

Since the counter reset eliminates the `rb0`/`rb1` naming hypothesis, the contamination must come from MuJoCo/rendering state that persists in the Python process despite:
- `env.close()` being called on the expert gen's environment
- `gc.collect()` running
- `torch.cuda.empty_cache()` running
- `GLOBAL_MJ_ROBOT_COUNTER` being reset to 0
- Stale `panda_tmp_rb*.xml` files being deleted

### RC-7A: OpenGL/EGL Context State (MOST LIKELY)

MuJoCo's offscreen renderer (used for camera image generation) creates an OpenGL or EGL context. When `env.close()` is called, the MuJoCo model and data are freed, but the **OpenGL context** may not be fully destroyed. Specifically:

- EGL display/surface state persists at the process level
- Framebuffer objects (FBOs) and renderbuffer storage may be reused
- The next `env.start()` call may **reuse** the existing OpenGL context rather than creating a new one, inheriting stale framebuffer dimensions, viewport settings, or even residual pixel data in uncleared buffers

The fact that **both** `bp_image` (external camera) and `inhand_img` (wrist camera) change simultaneously supports this — both cameras share the same OpenGL rendering backend.

### RC-7B: MuJoCo `mjModel` Metadata Cache

MuJoCo may cache compiled model metadata at the process level. The expert gen's environment creates a `mjModel` with certain body/joint/camera configurations. Even after `mj_deleteModel()`, some internal lookup tables or hash maps may retain entries from the first model load. When the variant's environment loads a fresh model (with the same XML structure but different temporary file path), MuJoCo may detect the structural similarity and reuse cached internal indices, which could cause subtle differences in camera calibration or rendering order.

### RC-7C: Stale XML `<include>` or Asset References

The `mju_openResource: could not open resource` warnings in the logs indicate that the compiled MuJoCo XML references temporary robot files that no longer exist. In the expert-gen-runs case (JOBs 20645, 20646), there are **two** such warnings:

```
WARNING: mju_openResource: could not open resource '...panda_tmp_rb0_700f460c-...xml'
WARNING: mju_openResource: could not open resource '...panda_tmp_rb0_80b66468-...xml'
```

While in the expert-skipped case (JOBs 20643, 20647, 20648), there is only **one** such warning. The second warning in the expert-ran case references a file from the expert gen's environment that was deleted by FIX-7's cleanup. MuJoCo may handle missing `<include>` references by falling back to default configurations, potentially altering the scene composition.

### RC-7D: `MjScene` or `MjFactory` Singleton State

The d3il `MjScene`/`MjFactory` classes may maintain process-level singleton registries (similar to `GLOBAL_MJ_ROBOT_COUNTER`) for scenes, bodies, or cameras. The expert gen's env creation may register entries in these registries that are never cleaned up, causing the variant's scene construction to inherit unexpected registered objects.

---

## 7. The Subprocess Solution — Now Mandatory

The POSTMORTEM §4 anticipated this scenario:

> "**Regression diagnostic**: If bp_image std is still 0.1978 after this fix, contamination goes deeper than the body counter (possibly OpenGL EGL context state). In that case, the **subprocess approach** (separate OS process for expert gen) would be needed."

This contingency has been triggered. The subprocess approach is now the recommended fix because:

1. A subprocess inherits no Python-level state (class variables, OpenGL contexts, MuJoCo caches)
2. When the subprocess exits, **all** its GPU/OpenGL/MuJoCo state is destroyed by the OS
3. The parent process's rendering pipeline remains completely virgin
4. This is a proven pattern for MuJoCo rendering isolation (used in RL training frameworks like RLlib, SB3)

### Proposed Implementation (FIX-7B)

```python
# In eval_fm_visual_aligning.py, replace the inline expert gen call with:
import subprocess

_base_results = (f'{args.savepath}/results_train_set'
                 if args_cli.eval_on_train else f'{args.savepath}/results')
os.makedirs(_base_results, exist_ok=True)

# Run expert gen in a completely isolated subprocess
subprocess.run(
    [sys.executable, '-c',
     f'import sys; sys.path.insert(0, "."); '
     f'from fm_visual_aligning_test.eval_fm_visual_aligning import generate_expert_reference; '
     f'generate_expert_reference("{_base_results}", n_rollouts=3)'],
    check=True,
    cwd=os.getcwd(),
    env=os.environ.copy()
)
# After subprocess exits: zero process-level contamination in parent
```

### Alternative: Direct `multiprocessing` with `start_method='spawn'`

```python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

p = mp.Process(target=generate_expert_reference, args=(_base_results, 3))
p.start()
p.join()
assert p.exitcode == 0, f'Expert gen failed with exit code {p.exitcode}'
```

`spawn` (not `fork`) is critical — `fork` would copy the parent's state, defeating the purpose.

---

## 8. Why the Counter Was a Red Herring — Lessons Learned

The BUG_REPORT_7_Audit.md made a classic forensic error: **confusing correlation with causation**.

The reasoning was:
1. Expert gen increments the counter (0→1) ✓
2. Counter=1 produces `rb1` body names in XML ✓
3. Different body names → different compiled scene → different rendering ✗ **DISPROVEN**

The actual chain is likely:
1. Expert gen creates a MuJoCo rendering context (OpenGL/EGL)
2. `env.close()` does NOT fully destroy the rendering context
3. The variant's `env.start()` inherits/reuses the contaminated context
4. The contaminated context produces different camera images
5. The counter increment was a **side effect** of the same env creation that created the rendering context — correlated, but not causal

The counter was observable proof that expert gen ran (rb1 = expert ran, rb0 = expert skipped), but resetting the counter while leaving the rendering context intact is like resetting a thermometer while leaving the fire burning.

---

## 9. Updated Verification Checklist (Next Slurm Job — With Subprocess Fix)

- [ ] Expert gen runs via `subprocess.run()` or `mp.Process(start_method='spawn')`
- [ ] Parent process log contains NO `panda_tmp_rb*` files from expert gen (they exist only in subprocess's temporary state)
- [ ] Log contains: `[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)` (kept as defense-in-depth)
- [ ] `[ DIAG img ] bp_image std = 0.2093` for all variants **including first run with fresh folder**
- [ ] `inhand_img std = 0.2490` (not 0.2867)
- [ ] First-replan a0 = `[-0.0054, 0.0405, -0.2432]` (not `[0.0171, 0.008, -0.5546]`)
- [ ] Final distance ctx0 = 0.218847 m (not 0.312711 m)
- [ ] Clamp events = 2 (not 233)

---

## 10. Summary

| Item | BUG_REPORT_7_Audit Hypothesis | This Report's Finding |
|---|---|---|
| Primary root cause | `GLOBAL_MJ_ROBOT_COUNTER` (`rb0` vs `rb1` body prefix) | **DISPROVEN** — counter reset to 0, body prefix is `rb0`, bp_image still 0.1978 |
| Counter's role | Causal mechanism changing scene geometry | Correlated marker, not causal |
| Actual mechanism | Not investigated beyond counter | OpenGL/EGL context contamination surviving `env.close()` (most likely) |
| FIX-7 effectiveness | Expected to restore bp_image std = 0.2093 | **Zero effect** — contamination persists identically |
| Recommended fix | Counter reset (implemented) | **Subprocess isolation** — the only way to guarantee zero process-level state leakage |
| Regression diagnostic | "If still 0.1978, use subprocess approach" | **Triggered.** Subprocess approach is now mandatory. |

> [!IMPORTANT]
> **Bottom line**: The expert gen process contaminates something in the MuJoCo/OpenGL rendering pipeline that survives `env.close()`, `gc.collect()`, `torch.cuda.empty_cache()`, counter reset, and stale file cleanup. The only reliable isolation is **process-level isolation** via subprocess. The counter was a correlated side effect, not the root cause. FIX-7B (subprocess expert gen) should be implemented immediately.
