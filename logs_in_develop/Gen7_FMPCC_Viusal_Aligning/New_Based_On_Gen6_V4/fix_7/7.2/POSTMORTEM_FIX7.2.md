# Post-Mortem: Visual Alignment Contamination & Rendering Singleton Freeze (FIX-7.2)

**Author:** Antigravity (Auditor)  
**Date:** 2026-05-21  
**Status:** Cleanly Resolved & Verified  

---

## 1. Executive Summary

During the evaluations of the Gen6V4 (Visual-DPCC) and Gen7 (Visual Flow Matching) pipelines, a catastrophic "frozen" trajectory behavior was intermittently observed. Evaluation variants (like `post_processing` or `model_free`) planned static trajectories, logged high action clamping counts, and yielded wrong visual input statistics (`bp_image std = 0.1978` instead of the baseline `0.2093`).

This post-mortem details the forensic investigation that traced this bug back to its injection point—**the automation of expert reference video generation in the same process**—and details how the subsequent fixes (specifically FIX-7 and the final FIX-7.2) initially exacerbated, and then completely cured, the state contamination.

---

## 2. Timeline of Commits & System Impact

Below is the chronological timeline of commits leading to the injection, discovery, false mitigation, and final resolution of the rendering contamination.

```mermaid
gantt
    title Bug Timeline & Remediation (May 2026)
    dateFormat  YYYY-MM-DD HH:mm
    axisFormat %m-%d
    
    section Inception
    Expert Gen Integrated (f07b463) :crit, active, 2026-05-15 13:41, 1m
    
    section Vulnerable Window
    Gen6 Upgrade Affected (eefec19) :active, 2026-05-17 15:16, 2d
    Gen7 Upgrade Affected (bc24734) :active, 2026-05-20 14:55, 1d
    
    section Remediation
    Suspect Identified (Fix-7: fd27e21) :2026-05-21 12:38, 1m
    Root Cause Resolved (Fix-7.2: 765cd17) :success, 2026-05-21 13:35, 1m
```

### Commit Timeline

| Commit Hash | Author | Timestamp | Message & System Impact |
| :--- | :--- | :--- | :--- |
| **`f07b4631e2c9...`** | `ghubliming` | **2026-05-15 13:41:44** | **THE INJECTION POINT (feat: Integrate automated expert reference generation...)**<br>Expert reference generation was integrated to run *in-process* immediately before the variant loops. This acted as a silent time-bomb, leaking both `MjRobot.GLOBAL_MJ_ROBOT_COUNTER` (advancing it `0 -> 1`) and populating the global `__RENDER_CTX_MAP` cache with contexts bound to `expert_model`/`expert_data`. |
| **`eefec19603cd...`** | `ghubliming` | **2026-05-17 15:16:15** | **MAJOR UPDATE AFFECTED (feat: Implement Gen6 Visual Differentiable MPC)**<br>The new Visual-DPCC pipeline was evaluated under visual contamination. Because the first variant environment compiled `"rb1"` body names instead of `"rb0"`, the camera viewport shifted (`std = 0.1978`), causing the MPC to receive mutated observations. |
| **`bc247343d0e3...`** | `ghubliming` | **2026-05-20 14:55:21** | **MAJOR UPDATE AFFECTED (feat: implement Gen7 visual Flow Matching...)**<br>The native RGB visual Flow Matching pipeline was introduced. Due to the visual state leakage, the FM models received stale visual inputs and planned trajectories that immediately crashed or clamped, hiding the true capabilities of the Gen7 models. |
| **`fd27e2105ec0...`** | `ghubliming` | **2026-05-21 12:38:25** | **THE FALSE DAWN (fix: reset MuJoCo global robot body counter... - FIX-7)**<br>We identified the robot counter leakage and reset it to `0` after expert gen. This established robot body name parity (`"rb0"`). **Crucially, this triggered identical camera names (`"rgbd_cage"`) across both expert and variant envs.** When the variant rendered, it hit the global cache and received the expert's stale render context. Trajectories became 100% frozen. |
| **`765cd17aa726...`** | `ghubliming` | **2026-05-21 13:35:50** | **THE FINAL RESOLUTION (Implement FIX-7.2: Clear Render Context Cache...)**<br>We injected calls to `reset_singleton()` immediately after expert gen and in the per-variant `finally:` blocks. This cleanly clears the global cache, ensuring variants compile isolated and accurate render contexts. |

---

## 3. Technical Breakdown of the Contamination

The rendering bug operated as a two-stage process-level memory leak:

### Stage 1: Robot Body Counter Leakage (FIX-7)
MuJoCo robot compilation dynamically appends a prefix to body and camera names based on a class-level variable:
```python
class MjRobot:
    GLOBAL_MJ_ROBOT_COUNTER = 0 # Increments on every __init__
```
* **Expert Gen Phase:** Initializes a `Robot_Push_Env` (Counter: `0 -> 1`, body prefix `"rb0"`).
* **Variant Phase:** Initializes a new env. Because the counter was not reset, it compiled the XML with a body prefix of `"rb1"`.
* **Result:** Camera and robot joints were named `"rb1_*"`. The camera compiled with a slightly different orientation/offset due to the prefix mismatch, altering visual data statistics (`std = 0.1978` instead of `0.2093`).

### Stage 2: Cache Contamination (FIX-7.2)
When we reset the counter to `0` in FIX-7, the variant env successfully compiled under the `"rb0"` prefix. This meant its cage camera was named `"rgbd_cage"`, identical to the expert's camera name.

MuJoCo offscreen rendering is governed by a global cache in `mj_render_singleton.py`:
```python
__RENDER_CTX_MAP = {}

def get_renderer(name, width, height, model, data):
    global __RENDER_CTX_MAP
    if name not in __RENDER_CTX_MAP:
        ctx = RenderContextOffscreen(width, height, model, data)
        __RENDER_CTX_MAP[name] = ctx
    ctx = __RENDER_CTX_MAP[name]
    return ctx
```
1. **The Cache Hit:** The variant environment requested a camera render for `"rgbd_cage"`.
2. **The Stale Context:** Because `"rgbd_cage"` was already in the map, `get_renderer()` returned the cached `RenderContextOffscreen` bound to the **expert's model and data**.
3. **The Freeze:** Every `ctx.render()` call internally executed `mjv_updateScene(self.model, self.data, ...)` using the expert's static model. The variant's actual environment steps were ignored. The agent planned actions against a static, frozen expert scene, leading to failed trials, high tracking errors, and massive clamp events.

---

## 4. The Resolution

The issue was resolved by importing and executing the `reset_singleton()` utility to clear the cached renderer contexts at boundaries where the simulation model or data changes.

### Code Fix Injection Locations

1. **Post-Expert Gen (Pre-Loop Boundary):**
   ```python
   # FIX-7.2: Clear the process-global render context cache so the variant's
   # cameras create fresh RenderContextOffscreen objects bound to variant_model
   # and variant_data.
   try:
       from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_render_singleton import (
           reset_singleton as _reset_render_singleton,
       )
       _reset_render_singleton()
       print('[ expert ] Render singleton cache cleared (FIX-7.2)')
   except Exception as _e:
       print(f'[ expert ] WARNING: Render singleton reset failed: {_e}')
   ```

2. **Per-Variant Teardown (Variant Loop Boundary):**
   ```python
   finally:
       # FIX-7.2 (per-variant): Clear render context cache so next variant
       # creates fresh RenderContextOffscreen objects.
       try:
           _reset_render_singleton()
       except NameError:
           pass
   ```

---

## 5. Why the Naive Legacy Parity Retrieval (FIX-7) Initially Failed & Why the Cache Bug Was Missed

To resolve earlier evaluation issues, a systematic "legacy parity retrieval" was performed in [Manual_Legacy_retrieval_FIX_7](file:///workspaces/FM-PCC/logs_in_develop/Gen6_dpcc_Engine_for_visual_aligning/Gen6V4_dataset_upgrade_visual_dpcc/Manual_Legacy_retrieval_FIX_7/) to audit and revert drifts between the vendored `FM-PCC/d3il` folder and the original `/workspaces/d3il` repository. 

While the audit successfully identified and reverted multiple physical and visual drifts (such as dataset BGR→RGB conversion, `BPCageCam` named constructor positional keys, and phantom `rod:tip` colliding geom flags), the initial implementation of the FIX-7 retrieval failed to restore healthy visual evaluation. This failure occurred across two operational levels, and the core cache bug was missed due to a critical systemic blind spot:

### 5.1 Naive Parity Induced CPU Starvation Deadlock (FIX-7.3)
In the original D3IL codebase, process CPU affinity pinning was used to prevent scheduling latency during simulation stepping:
```python
assign_process_to_cpu(os.getpid(), cpu_set)
```
However, the original D3IL codebase only supports lightweight, state-only rollouts and does not define a visual mode.
* **The Failure:** Blindly applying D3IL parity in `FIX_7.3` unconditionally enabled this pinning. Running the heavy FM-PCC visual evaluation pipeline—which coordinates OpenGL rendering, ResNet feature extraction, PyTorch inference, and SciPy SLSQP optimization—on a **single pinned core** (`cpu_set={0}`) caused severe thread starvation.
* **The Symptom:** An immediate, permanent thread deadlock / hang at Context 0 Rollout 0 (spinning indefinitely for 15+ minutes).
* **The Cure (FIX-7.5):** We explicitly gated process CPU pinning so that it bypasses pinning when visual mode is active:
  ```python
  if not self.if_vision:
      assign_process_to_cpu(os.getpid(), cpu_set)
  ```

### 5.2 Parity Alone Cannot Overcome K=16 Denoising Chain Instability
Even after visual channel, camera constructor, physics contacts, and CPU deadlock fixes were applied, evaluating the `K=16` baseline model yielded a **0% success rate**. 
* **The Failure:** The `K=16` DDPM reverse chain mathematically reaches step `t=15` with an alpha noise amplification factor `sqrt_recip_alphas_cumprod[15] ≈ 11x`. This causes the predicted clean trajectory `x_recon` to explode to a standard deviation of `≈ 10.5` in normalized space—far outside the `[-1, 1]` trainable range of `LimitsNormalizer`.
* **The Symptom:** The normalizer clipped the exploded trajectories to boundary constants, delivering garbage input to the SLSQP projector and causing all rollouts to mathematically diverge, regardless of correct simulator physics.
* **The Lesson:** While correct simulation infrastructure parity is a *necessary* condition for success, it was not *sufficient* without increasing the denoising steps to `K=100`. At `K=100`, the gradual reverse chain remains stable and stays within trainable normalizer bounds, enabling the model to actually succeed.

### 5.3 Forensic Deep-Dive: Why We Missed the Expert Gen & Singleton Cache Contaminations
Despite our highly detailed, file-by-file audit of the `d3il/mujoco` integration in the legacy retrieval effort, both the expert video generation contamination (`GLOBAL_MJ_ROBOT_COUNTER`) and the global rendering singleton cache contamination (`__RENDER_CTX_MAP`) went completely unnoticed. This exposes critical cognitive blind spots:
1. **The Scope Boundary Trap (Why We Missed the Expert Gen Bug):** Our deep legacy audit focused exclusively on diffing the `FM-PCC/d3il` folder against the original `/workspaces/d3il` repository. However, the automated expert video generation was injected into the high-level evaluation scripts (`eval_fm_visual_aligning.py`), completely outside the `d3il` folder. We audited the engine but ignored how our own top-level scripts were misusing the engine's lifecycle.
2. **The Parity Cognitive Trap ("Parity Equals Correctness"):** Because `mj_render_singleton.py` was identical in both the vendored and reference repositories (no custom drifts had ever been made in that file), it was skipped. We operated under the dangerous assumption that *if the vendored code matches original D3IL, it must be correct*, failing to realize that the original legacy code itself possessed structural assumptions incompatible with our new runtime.
3. **Architectural Shift (Single-Run vs. In-Process Multi-Variant Loops):** The original D3IL codebase was designed for single-task, state-only rollouts. In a single rollout process, caching the OpenGL offscreen rendering context under a global dictionary is a harmless performance optimization. In FM-PCC, however, we introduced in-process multi-variant loops (`[diffuser, post_processing, model_free]`) and automated in-process expert reference video generation sequentially. The legacy singleton had no concept of context lifetimes or boundary invalidation. Sequential loops inside a single process turned a harmless optimization into a catastrophic memory contamination bug.
4. **The Masking Mask (Naming Discrepancies Hid Cache Hits):** Prior to `FIX-7`, the lack of robot counter resets caused variant environments to request renderer keys like `"rb1_rgbd_cage"`, which missed the dirty cache (which held `"rgbd_cage"` from the expert gen). Because the cache missed, a new context compiled and variants rendered dynamic poses, masking the latent caching bug. We were lulled into a false sense of security, assuming that resetting the counter (`FIX-7`) was the only required fix.
5. **The Toughest Lesson:** **Parity is not correctness, and system boundaries matter.** When adapting or auditing legacy code libraries, we cannot simply verify that our copies match the original source, nor can we limit our audit to the vendored folder. We must systematically audit how our high-level scripts misuse the global variables, singletons, and cache Lifecycles of the legacy codebase under **new architectural assumptions** (such as sequential in-process loops). If we fail to do this, our own fixes (like establishing naming parity in FIX-7) will serve only to detonate latent time-bombs left by legacy code.

---

## 6. Key Lessons Learned

1. **Process-Global Singletons are Dangerous in Multi-Task Pipelines:** When writing test/evaluation suites that sequentially load different configurations, any process-global state (counters, OpenGL caches) must have explicit lifecycle teardown methods.
2. **Symptom Parity Can Mask Underlying Bugs:** Solving the body prefix discrepancy (FIX-7) was a correct step, but because of the global render cache, achieving prefix parity transformed a subtle visual shift into a complete observation freeze. Always audit caches when establishing naming parities in singletons.

---

## 7. Behavioral Divergences From First Principles (Audit Record)

> **Purpose:** A factual record of every change (bug-fix or upgrade) that makes the current `diffuser_visual_aligning` / `fm_visual_aligning` pipelines behave differently from their respective first-principle baselines: the original **DPCC-diffuser `diffuser` variant** and the reference **`flow_matcher_v3_ode_selectable`**. Each entry is tagged **`REVIEW`** (actively changes trajectory / success outcome — candidate for explicit on/off decision) or **`LEAVE`** (no-op under current config, crash-prevention only, or restores an original correctness property — no reason to touch).

---

### 7.1 Shared Changes (Both DPCC and FM Pipelines)

> **Note on S4 / S5:** `diffusion_timestep_threshold: 0.5` and `enlarge_constraints: 0.01` were part of the Gen6 Visual-DPCC first-principle design from its very first commit (`eefec19`). Both values are present in the DPCC reference repo (`projection_eval.yaml`) and were carried over correctly. They are **not divergences** — current = first principle. They are excluded from this table and from the §7.4 REVIEW list. See `upgrade_8/RESEARCH_BATCH_SIZE_TRAJECTORY_SELECTION.md` §2.2 for cross-repo confirmation.

| # | Parameter / Behaviour | Before (First Principle) | After (Current) | Introduced By | Commit |
|:--|:---|:---|:---|:---|:---|
| S1 | `max_action_delta` | No per-step clamp (`null`) | `0.01 m` per step (≈45× the healthy step size) | Fix 3+4 — ODE Explosion Protection | `2c87cb7` |
| S2 | `constraint_types` | `['bounds', 'dynamics']` (original in Gen6) → `[]` (disabled Fix 8) | `['bounds', 'dynamics']` (re-enabled, AUDIT-FIX-2) | Gen6 `eefec19` → disabled `7ba1f07` → re-enabled `0b8acfe` | `0b8acfe` |
| S3 | `trajectory_selection` for `post_processing`, `model_free` variants | `'random'` (= always index 0 in DPCC/FMv3ODE) | `'minimum_projection_cost'` (selects lowest-cost traj from batch) | Fix 9.4 | `0dedc33` |

**S1 — `max_action_delta: 0.01` detail:**
```python
# VisualAgentWrapper.get_action() — both eval_visual_aligning_dpcc.py and eval_fm_visual_aligning.py
if self.max_action_delta is not None:
    raw_mag = np.linalg.norm(next_action_np)
    if raw_mag > self.max_action_delta:
        next_action_np = next_action_np * (self.max_action_delta / raw_mag)  # rescale to unit ball
        self.curr_rollout_clamp_events.append((self.step_counter, float(raw_mag)))
```
Fires whenever the raw action vector exceeds 0.01 m; scales it down to exactly 0.01 m rather than clipping axes independently. `null` disables entirely.

---

### 7.2 DPCC-Only Changes (`diffuser_visual_aligning`)

| # | Parameter / Behaviour | Before (First Principle) | After (Current) | Introduced By | Commit |
|:--|:---|:---|:---|:---|:---|
| D1 | `clip_denoised` | `True` (wrong — caused ±5 clamp at first denoising step under cosine schedule ~9.4× amplification) | `False` — forced at eval regardless of checkpoint; also fixed in training | Gen6V4F6 — denoising chain corruption fix | `c0f0caa` |
| D2 | `LimitsNormalizer.normalize()` — constant dims | `x = (x - mins) / (maxs - mins)` → division by zero for constant dims | `range_[range_ < 1e-8] = 1.0` → constant dims map to 0 in normalised space | Fix 8 sub-fix A3 | `7ba1f07` |
| D3 | `LimitsNormalizer.unnormalize()` — constant dims | `x * (maxs - mins) + mins` → multiplies by 0, loses signal | `range_[range_ < 1e-8] = 0.0` → constant dims recover original `mins` | Fix 8 sub-fix A3 | `7ba1f07` |
| D4 | `DynamicConstraints` initial-state row coefficient | `mat_fix_initial[0, x_idx] = 1` (unscaled — weaker anchor than dynamics rows by factor `x_diff ≈ 0.4`) | `mat_fix_initial[0, x_idx] = x_diff`; `b[...] = x_diff * s_0[x_idx]` — row scaled to match dynamics matrix | Fix 8 sub-fix B1 | `7ba1f07` |
| D5 | SLSQP call when no constraints active | SLSQP runs unconditionally, QP cost corrupts healthy trajectories | Skip SLSQP entirely when `A.shape[0] == 0` and `C.shape[0] == 0` and no obstacles; return trajectory unchanged | Fix 9.1 | `0dedc33` |
| D6 | Gradient computation when no constraints active | Computes gradient unconditionally | Return `torch.zeros_like(trajectory)` when no constraints active | Fix 9.2 | `0dedc33` |
| D7 | Initial-state anchor applied per-sample | `s_0 = trajectory_reshaped[0, ...]` — batch[0] anchor used for all samples | `s_0 = trajectory_reshaped[i, ...]` — per-sample anchor | Fix 8 sub-fix A4 | `7ba1f07` |

**D1 — `clip_denoised` detail:**
```python
# eval_visual_aligning_dpcc.py — forced override regardless of what the checkpoint saved:
diffusion_model.clip_denoised = False

# train_visual_aligning_dpcc.py — fixed for all future training runs:
# was: clip_denoised=True
# now: clip_denoised=False
```

---

### 7.3 FM-Only Changes (`fm_visual_aligning`)

| # | Parameter / Behaviour | Before (First Principle) | After (Current) | Introduced By | Commit |
|:--|:---|:---|:---|:---|:---|
| F1 | `flow_steps_v3` Slurm override | Checkpoint's baked-in value always used; `--flow_steps_v3 N` in sbatch only changed the output directory name | `diffusion_model.flow_steps_v3 = int(args.flow_steps_v3)` — Slurm arg propagated to model at eval start | Fix 5 + F6 | `f4b9120`, `0b8acfe` |
| F2 | ODE solver params consumed by `VisualGaussianDiffusion.__init__` | `ode_solver_rtol_v3`, `ode_solver_atol_v3`, `ode_solver_step_size_v3` caused `TypeError` if present in config (not in base `GaussianDiffusion.__init__`) | Intercepted and silently discarded in `VisualGaussianDiffusion.__init__` | Gen7fix2 | `9fff089` |

**F1 — `flow_steps_v3` override detail:**
```python
# eval_fm_visual_aligning.py — at model load time, before the variant loop:
_args_flow = getattr(args, 'flow_steps_v3', None)
if _args_flow is not None:
    diffusion_model.flow_steps_v3         = int(_args_flow)
    diffusion_model.ode_inference_steps_v3 = int(_args_flow)
```
Before Fix 5, `--flow_steps_v3 10` would create a results directory named `flow_steps_v3_10/` but the model would still integrate at (e.g.) 100 steps from the checkpoint.

---

### 7.4 Summary Table — Action Classification

`REVIEW` = genuinely diverges from first principle AND actively changes trajectory / success outcome → decide whether to keep or revert.  
`LEAVE` = no divergence from first principle, no-op under current config, crash-prevention only, or restores a correctness property → ignore.

| ID | DPCC / FMv3ODE First Principle | Current Value | Tag | How to Revert |
|:--|:--|:--|:--|:--|
| S1 `max_action_delta` | `null` — no clamp existed in either baseline | `0.01 m/step` (rescales action vector to unit ball) | **`REVIEW`** | `max_action_delta: null` in YAML |
| S2 `constraint_types` | `['bounds','dynamics']` in Gen6 orig → `[]` user-disabled (Fix8) | `['bounds','dynamics']` (re-enabled AUDIT-FIX-2) | **`REVIEW`** | `constraint_types: []` in YAML |
| S3 `trajectory_selection` for `post_processing`/`model_free` | `'random'` in both baselines | `'minimum_projection_cost'` (best-of-6 by cost) | **`REVIEW`** | Change assignment to `'random'` in eval script |
| D1 `clip_denoised` | `False` (correct original DPCC spec) — wrongly `True` in old checkpoints | `False` forced at eval; `False` in new training | **`REVIEW`** | Change `clip_denoised` override back to `True` in eval/train scripts |
| D2/D3 `LimitsNormalizer` constant dims | Unprotected (div-by-zero possible) | Protected (`range_<1e-8 → 1.0/0.0`) | **`LEAVE`** | — no-op for 6D aligning obs (all dims have non-zero range) |
| D4 B1 initial-state row coefficient | `mat_fix_initial[0,x_idx] = 1` | `= x_diff ≈ 0.4` (scaled to match dynamics rows) | **`REVIEW`** | Search `[DANGEROUS_FLAG_B1_SCALING]` to revert scaling back to `1` and `s_0` |
| D5/D6 skip SLSQP / grad when no constraints | SLSQP runs unconditionally | Early-return when `A`, `C` empty (currently dead code — constraints enabled) | **`LEAVE`** | — inactive; guards against QP corruption if constraints ever disabled |
| D7 A4 per-sample initial-state anchor | `trajectory_reshaped[0,...]` — batch[0] anchor for all | `trajectory_reshaped[i,...]` — per-sample anchor | **`REVIEW`** | Search `[DANGEROUS_FLAG_A4_PER_SAMPLE_ANCHOR]` to revert `[i]` back to `[0]` |
| F1 `flow_steps_v3` Slurm propagation | Checkpoint value used; `--flow_steps_v3` arg silently ignored | Slurm arg overrides checkpoint at eval start | **`LEAVE`** | — removing breaks step-count ablations with no benefit |
| F2 ODE params intercepted in `VisualGaussianDiffusion` | `TypeError` crash if config has `ode_solver_rtol_v3` etc. | Silently consumed in `__init__` | **`LEAVE`** | — crash-prevention only; zero behavioral change |

> **Removed from table:** S4 (`diffusion_timestep_threshold: 0.5`) and S5 (`enlarge_constraints: 0.01`) — both values were present in the Gen6 DPCC first-principle baseline from the first commit (`eefec19`) and have not changed. No divergence exists.

**Only 6 `REVIEW` items remain: S1, S2, S3, D1, D4, D7. Everything else is `LEAVE`.**

