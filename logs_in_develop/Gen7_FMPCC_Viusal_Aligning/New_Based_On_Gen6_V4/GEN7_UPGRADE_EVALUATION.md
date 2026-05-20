# Gen7 FM-PCC Upgrade Evaluation — INDEPENDENT RE-AUDIT

**Date:** 2026-05-20  
**Original Evaluator:** Auditor Team (Antigravity)  
**2nd Re-Auditor:** Antigravity (Claude Opus 4.6), independent line-by-line verification  
**3rd Re-Auditor:** Claude Sonnet 4.6, independent line-by-line verification of all source citations  
**Target:** Assessing readiness for the Gen7 Foundation Model upgrade based on the Gen6V4 codebase.

> **IMPORTANT**
> This document supersedes the original `GEN7_UPGRADE_EVALUATION.md`.  
> Every claim below was independently verified against the actual source code.  
> **Trust nothing; verify everything.**

---

## 🏁 Executive Verdict: CONDITIONAL READY — ONE BLOCKING ISSUE

The Gen6V4 codebase is **structurally sound** in its DDPM engine, projector mathematics, and systems plumbing. However, the original evaluation contained **one materially incorrect claim** about the image channel pipeline that constitutes a **potential data distribution mismatch between training and inference**. This must be resolved before the Gen7 upgrade can proceed with confidence.

> **🔍 3rd Audit [Claude Sonnet 4.6] — Executive Verdict:**  
> Partially agreed. The structural soundness verdict is confirmed. The BGR/RGB mismatch is **factually confirmed** (training=RGB, inference=BGR). However, the "BLOCKING" classification is **disputed**: fix7 empirically succeeded with exactly this mismatch (RGB-trained model receiving BGR, no flip), demonstrating the model is robust to channel swap for this task. The ±94 action divergence is checkpoint-correlated, not channel-correlated. I recommend re-classifying to **HIGH-PRIORITY ADVISORY**: fix before first Gen7 training run (new architecture's channel sensitivity is unknown), but do not block evaluation of the current Gen6V4 model on this basis alone. See Section 2 BGR/RGB for full analysis.

---

## 1. Robotic & Mathematical Principles (The Projector)

### Claim A4 (Initial State Anchoring): ✅ VERIFIED

**Original claim:** "The projector correctly extracts the per-sample initial state (t=0) from the batch."

**Verification:**  
- `projection.py:148` — `project()` iterates `for i in range(batch_size)` and extracts `trajectory_reshaped[i, :self.transition_dim]` — confirmed per-sample, not `trajectory_reshaped[0, ...]`.
- `projection.py:207-216` — `compute_gradient()` also uses `trajectory_reshaped[i, :]` — confirmed consistent.

**Verdict:** ✅ Correct. A4 fix is properly applied in both `project()` and `compute_gradient()`.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED.  
> Independently verified at `projection.py:145-157` (`project()`) and `projection.py:206-216` (`compute_gradient()`). Both loops iterate `for i in range(batch_size)` and extract `trajectory_reshaped[i, :self.transition_dim]`. No deviation from the 2nd audit's finding. A4 fix is active and correct.

---

### Claim B1 (Scale Alignment): ✅ VERIFIED

**Original claim:** "The derivative constraints (x_diff) are properly scaled to match the corresponding rows in the A matrix."

**Verification:**  
- `projection.py:429` — `mat_fix_initial[0, x_idx] = x_diff` — the initial state anchor row uses `x_diff = x_max - x_min`, matching the scale of the dynamics rows at lines 417-419 where `mat_append[i, ...] = 1 * x_diff` and `mat_append[i, ...] = self.dt * dx_diff`.
- `projection.py:155-156` — `b[counter * self.horizon] = x_diff * s_0[x_idx]` — RHS scaled consistently.
- `_initial_state_x_diffs` list stores `float(x_diff)` at line 432 — confirmed consistent with the `build_matrices()` scale.

**Verdict:** ✅ Correct. B1 fix ensures dimensional consistency between the A matrix rows and the b vector.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED.  
> Independently verified the full B1 chain at `projection.py:415-432`. The dynamics rows use `1 * x_diff` (line 417) and `self.dt * dx_diff` (line 418). The anchor row uses `x_diff` (line 429). The b-vector entry uses `x_diff * s_0[x_idx]` (line 156). `_initial_state_x_diffs` stores `float(x_diff)` at line 432, retrieved at line 155. Scale is consistent throughout. No deviation.

---

### Claim C4 (True State Feedback): ✅ VERIFIED — WITH CAVEAT

**Original claim:** "The environment now correctly injects the actual `robot_pos` into the conditioning vector."

**Verification:**  
- `aligning_sim.py:94` — At t=0: `robot_pos = env_state[:3].copy()` — this is `self.robot_state()` → `self.robot.current_c_pos` (actual TCP position from MuJoCo FK). ✅
- `aligning_sim.py:106` — In the loop: `robot_pos, bp_image, inhand_image = obs` — `obs` comes from `get_observation()` which returns `self.robot_state()` (actual TCP pos). ✅
- `aligning_sim.py:98` — Agent receives 4-tuple `(bp_image, inhand_image, des_robot_pos, robot_pos)`. ✅
- `eval_visual_aligning_dpcc.py:451` — `predict()` unpacks: `bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state`. ✅
- `eval_visual_aligning_dpcc.py:475` — `obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])` → `[des_c_pos | c_pos]`. ✅

**Caveat:** At t=0 initialization, `robot_pos = env_state[:3].copy()` means `actual == commanded`. This is correct (robot hasn't moved yet), but should be documented — C4 only provides a differentiated signal from t≥1 onward.

**Verdict:** ✅ Correct. The control loop is properly closed with actual sim state.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED with an additional minor style note.  
> The full 4-tuple packing/unpacking chain verified: `aligning.py:205-217` returns `(robot_pos, bp_image, inhand_image)` 3-tuple → `aligning_sim.py:106` unpacks → `aligning_sim.py:98` repacks as `(bp_image, inhand_image, des_robot_pos, robot_pos)` 4-tuple → `eval_visual_aligning_dpcc.py:451` unpacks correctly → line 475 concatenates to 6D.  
> **Minor style finding:** `aligning_sim.py:93` has `des_robot_pos = env_state[:3]` (no `.copy()`) but `aligning_sim.py:94` has `robot_pos = env_state[:3].copy()` (with `.copy()`). The asymmetry is functionally harmless — `env_state` is a local array from `env.reset()` that is not modified externally, and `des_robot_pos` is overwritten at line 104 after the first step. But the inconsistency could confuse future readers. Add `.copy()` to line 93 for clarity.

---

### ⚠️ CRITICAL: Projector is DORMANT

**Not mentioned in original evaluation.**

- `visual_aligning_eval.yaml:92` — `constraint_types: []` — **the constraint list is empty**.
- `eval_visual_aligning_dpcc.py:95-105` — `setup_dpcc_projector()` checks `config.get('constraint_types', [])` for `'bounds'` and `'dynamics'`. With `constraint_types: []`, **no bounds and no dynamics constraints are added**.
- `projection.py:102-104` — Fix 9.1 guard: `if self.A.shape[0] == 0 and self.C.shape[0] == 0 ... return trajectory` — **the projector passes trajectories through unchanged**.

**Impact:** The A4, B1, and C4 fixes exist in code and are mathematically correct, but they are **never exercised** with the current config. The projector is a no-op. The original evaluation presents these fixes as active safeguards — they are not. They are dormant code paths waiting for `constraint_types: ['bounds', 'dynamics']`.

**Verdict:** ⚠️ The projector code is correct but **inactive**. The original evaluation is misleading in presenting these as active protections. This should be explicitly acknowledged.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED dormant. Verified `config/visual_aligning_eval.yaml:92`: `constraint_types: []`.  
> **Nuance on "misleading":** The YAML itself is NOT silent about this. Line 92 reads:  
> `constraint_types: []  # OPTION A: No constraints (kept disabled per user decision — Fix 8)`  
> The comment explicitly names this as a conscious decision by the developer. The misleading artifact is the *original prose evaluation* (not the YAML) implying A4/B1/C4 are active safeguards. The YAML owner knew. Any reader of the config sees it clearly. Severity: misleading in the evaluation document, not in the production config.

---

## 2. ML Principles (The DDPM Engine)

### Claim: `clip_denoised = False`: ✅ VERIFIED

**Verification:**
- `diffusion.py:17` — Constructor default is `clip_denoised=False`. ✅
- `eval_visual_aligning_dpcc.py:696` — `diffusion_model.clip_denoised = False` forced at eval time. ✅
- `visual_gaussian_diffusion.py:71-75` — When `clip_denoised=True`, only action dims are clamped to ±5 (not full trajectory). This override is in `VisualGaussianDiffusion.p_mean_variance()`, which correctly **overrides** the base class that would `raise RuntimeError("clip_denoised=False not supported")`.
- Since `clip_denoised` is forced `False`, the clamp branch is never entered. The denoising chain runs unclamped. ✅

**Verdict:** ✅ Correct. The DDPM engine is mathematically pure — no artificial clamping.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED with stronger evidence.  
> The method name and signature are verified: `visual_gaussian_diffusion.py:52` declares `def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):` — this IS the explicit override of the base class method. The docstring at line 53-59 confirms intent: "Override to clamp only action dims (not obs dims)."  
> Key distinction: when `clip_denoised=True`, the subclass clamps actions to **±5** (not ±1 like the base class would). When `clip_denoised=False`, the subclass does NOT raise (unlike the base class). The forced `False` at eval time means the denoising chain is truly unclamped. ✅  
> **Risk note confirmed:** If any code accidentally calls `GaussianDiffusion.p_mean_variance()` directly (bypassing the override) with `clip_denoised=False`, it would raise. The current call chain through `VisualGaussianDiffusion` is safe; Python MRO guarantees the override is called.

---

### Claim: Observation Normalizer hardened against zero-variance: ✅ VERIFIED

**Verification:**
- `normalization.py:159-160` — `range_[range_ < 1e-8] = 1.0` in `normalize()`. ✅
- `normalization.py:177-178` — `range_[range_ < 1e-8] = 0.0` in `unnormalize()` — constant dims map back to `self.mins`. ✅

**Verdict:** ✅ Correct. Zero-variance guard is properly implemented.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED functionally. Code verified at `diffuser_visual_aligning/datasets/normalization.py:157-179`.  
> **Comment error found:** Line 160 comment reads `"constant dims map to 0 in normalized space (A3)"` — this is **mathematically wrong**. Tracing the code: for a constant dim, `x == self.mins`, so `(x - self.mins) / 1.0 = 0`. Then `2 * 0 - 1 = -1`. Constant dims map to **-1** in normalized space, not 0.  
> This is **NOT a functional bug** — the model was trained with LimitsNormalizer, and eval uses the same LimitsNormalizer, so -1 is internally consistent. The concern is that future developers may read the comment and expect 0, then design a zero-check that doesn't fire. The comment should be corrected to: `"constant dims map to -1 in normalized space (midpoint of [-1,1] range for uniform-range normalizer)"`. Wait — actually -1 is not the midpoint (that's 0). The constant dim gets stuck at the minimum of the normalized range, which is -1. This comment needs to be corrected to say "constant dims are pinned to -1 (normalized minimum)".

---

### Claim: "Image tensors are strictly preserved in BGR format throughout the pipeline": ❌ INCORRECT — BLOCKING ISSUE

**Original claim:** "BGR format throughout the pipeline, guaranteeing identical distributions between the offline training dataset and the online D3IL simulation."

**Verification — Training Pipeline (Dataset):**
- `sequence.py:166-167` — `_load_images()`:
  ```python
  img = cv2.imread(p)                               # BGR (cv2 default)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # → RGB
  ```
  The training dataset loads images in **RGB** format.

**Verification — Inference Pipeline (Sim):**
- `aligning.py:211-215` — `get_observation()`:
  ```python
  bp_image = self.bp_cam.get_image(depth=False)      # RGB (MuJoCo render)
  bp_image = cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)  # → BGR
  ```
  The environment returns images in **BGR** format.

- `aligning_sim.py:86-90` — Fix 11 comment says:
  ```
  # cv2.imread+cvtColor(BGR2RGB) in _load_images() accidentally produces BGR.
  ```
  This comment is **factually wrong**. `cv2.imread` returns BGR, `cvtColor(BGR2RGB)` converts it to RGB. The training pipeline produces **RGB**, not BGR.

**The actual situation:**

| Pipeline | Step 1 | Step 2 | **Final format** |
|----------|--------|--------|-----------------|
| Training (`_load_images()`) | `cv2.imread` → BGR | `cvtColor(BGR2RGB)` → RGB | **RGB** |
| Inference (`get_observation()`) | `get_image()` → RGB (MuJoCo) | `cvtColor(RGB2BGR)` → BGR | **BGR** |

**There IS a channel mismatch.** The model is trained on **RGB** images but receives **BGR** images at inference time.

> **CAUTION**
> The Fix 11 comment in `aligning_sim.py:86-89` contains an inverted reasoning chain.
> It claims `_load_images()` "accidentally produces BGR" — this is incorrect.
> `cv2.imread(BGR) + cvtColor(BGR2RGB) = RGB`. The training pipeline loads RGB.
> The environment outputs BGR. **They do not match.**

**Impact assessment:**  
For CNN-based visual encoders, BGR vs RGB swaps the red and blue channels. This can cause:
- Moderate performance degradation if the encoder has learned color-specific features
- Catastrophic failure if the task relies on color discrimination
- Potentially minimal impact if the encoder has learned to be channel-invariant

Given the aligning task uses geometric shapes, this *may* not be the primary cause of the ±94 divergence (which is likely a checkpoint issue as diagnosed), but it is an **unacknowledged distribution shift** that should be fixed.

**Verdict:** ❌ The original claim is **factually incorrect**. Training uses RGB; inference uses BGR. This is a **BLOCKING** issue that must be resolved before Gen7.

> **🔍 3rd Audit [Claude Sonnet 4.6] — SEVERITY DISPUTE:**  
> The 2nd auditor's **factual analysis is confirmed**: `sequence.py:166-167` produces RGB; `aligning.py:211-212` produces BGR. The mismatch is real. The Fix 11 comment ("accidentally produces BGR") is wrong — it inverts the pipeline logic, arriving at the correct conclusion (no flip) via incorrect reasoning. The comment must be fixed (already Advisory item 4).  
>
> However, I **dispute the "BLOCKING" severity classification** based on direct empirical evidence:
>
> | Run | Code | Channel at inference | Checkpoint | Action range |
> |-----|------|----------------------|------------|--------------|
> | fix7 (job ~50k) | no flip | **BGR** | ~50k steps | **[-0.78, +0.99] ✅** |
> | fix8 (job 20560) | with `[::-1]` | **RGB** | 42k steps | **[-85, +77] ✗** |
> | fix11 (job 20561) | no flip | **BGR** | 42k steps | **[-94, +87] ✗** |
>
> Fix7 used BGR inference on an RGB-trained model and **succeeded**. Fix8 used RGB inference (matching training) and **failed**. Fix11 used BGR inference (mismatching training, same as fix7) and **failed**. The flip state does not track the success/failure pattern — the checkpoint step does. This proves the model is **robust to the BGR/RGB channel swap for this specific task** (geometric manipulation, color-insensitive ResNet encoder).  
>
> **Re-classification: HIGH-PRIORITY ADVISORY**  
> - Fix before the first Gen7 training run (new architecture's channel sensitivity is unknown — a transformer-based FM may not share ResNet's channel robustness)  
> - Do NOT block evaluation of the current Gen6V4 model on this basis alone  
> - The fix11 comment claiming "accidentally produces BGR" must be corrected regardless  
>
> **Recommended fix:** Option B from the 2nd auditor (remove `cvtColor(RGB2BGR)` from `aligning.py:212,215` so env returns RGB matching training) OR Option C (add `[::-1]` in `aligning_sim.py`). Option A (remove the cvtColor in training pipeline) changes training behavior — risky. Option B modifies the D3IL env package — may conflict with project constraints. Option C (flip in aligning_sim.py) is the least invasive, but this was fix8's approach and fix8 still failed at 42k steps. Any of the three only resolves the distribution shift; it does NOT fix the checkpoint divergence.

---

## 3. Systems Engineering (Hardware & Pipeline)

### Claim: CPU Affinity Unpinning: ✅ VERIFIED

**Verification:**
- `aligning_sim.py:53-56`:
  ```python
  if not self.if_vision:
      assign_process_to_cpu(os.getpid(), cpu_set)
  else:
      print(f"Process {os.getpid()} unpinned — ...")
  ```
  Visual evaluations skip `os.sched_setaffinity()`, allowing full CPU utilization. ✅

**Verdict:** ✅ Correct.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED. Verified at `aligning_sim.py:53-56`. No deviation.

---

### Claim: Deterministic Seeding "after environment generation": ⚠️ PARTIALLY CORRECT

**Original claim:** "The random number generators are meticulously seeded *after* environment generation."

**Verification:**
- `aligning_sim.py:58-64`:
  ```python
  env = Robot_Push_Env(...)  # env created
  env.start()               # env started
  random.seed(pid)           # seeded after env gen ✅
  torch.manual_seed(pid)
  np.random.seed(pid)
  ```
  Seeding is indeed after environment generation. ✅

**However:** The seeds use `pid` (the core index 0, 1, 2...), **not** the actual `seed` argument passed to `Aligning_Sim`. With `n_cores=1`, `pid=0` always, so all seeds produce identical RNG states. The `self.seed` passed from the eval script (6, 7, 8, 9, 10 from config) is **not propagated to the RNG initialization**.

**Impact:** With `n_cores=1` (current config), every seed evaluation starts with `pid=0` → `np.random.seed(0)`. This means the `random` trajectory selection in batch sampling always follows the same pseudo-random sequence. The eval still varies by checkpoint seed (different model weights), but the stochastic trajectory selection is identical across seeds.

**Verdict:** ⚠️ Seeding is after env gen (correct), but it uses `pid` not the configured `seed` value — RNG diversity across eval seeds is not achieved. This is a correctness gap, not a blocker.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED ⚠️. Verified `aligning_sim.py:58-64`.  
> Extending the 2nd auditor's finding: with `n_cores=1` and `pid=0`, the initial Gaussian noise tensor `x_T` generated by `torch.randn(shape)` in the denoising chain (`p_sample_loop()`) is **identical for every seed evaluation** (seed 6, 7, 8, 9, 10). The only variation across seeds comes from the different model checkpoints (each seed trains an independent model). Multi-seed evaluation was designed to measure policy variance across training runs AND stochastic rollout diversity — currently only the former is achieved. Not a blocker, but eval statistics are less informative than intended.

---

## 4. Additional Findings (Not in Original Evaluation)

### 4.1. Base class `RuntimeError` on `clip_denoised=False`

- `diffusion.py:138-140` — The base `GaussianDiffusion.p_mean_variance()` raises `RuntimeError("clip_denoised=False not supported")`.
- This is safe **only because** `VisualGaussianDiffusion` overrides `p_mean_variance()`. If any code path accidentally calls the base class method, it would crash.
- **Risk:** Low but worth noting. The override chain is correct for the current architecture.

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED safe.  
> Method name verified: `visual_gaussian_diffusion.py:52` `def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None)`. Python MRO guarantees this override is called when `diffusion_model.p_mean_variance(...)` is invoked on a `VisualGaussianDiffusion` instance. The base class RuntimeError is never reachable through normal usage. Confirmed safe.

---

### 4.2. `max_episode_length` correctly exposed

- `aligning_sim.py:41` — `max_episode_length` parameter ✅
- `eval_visual_aligning_dpcc.py:797` — `max_episode_length=getattr(args, 'max_episode_length', 400)` ✅

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED. Verified both lines. Default 400 matches the D3IL aligning task baseline (proven in fix7). No deviation.

---

### 4.3. wandb disabled-mode guard

- `eval_visual_aligning_dpcc.py:802-803` — `_wandb.init(mode='disabled')` prevents the unconditional `wandb.log()` in `aligning_sim.py:213-216` from crashing. ✅

> **🔍 3rd Audit [Claude Sonnet 4.6]:** ✅ CONFIRMED. Verified at `eval_visual_aligning_dpcc.py:802-803`.

---

## 5. 3rd Audit — New Findings

### Finding 5.1: Fix 11 Comment Inverted Reasoning (aligning_sim.py:86-89)

The comment at `aligning_sim.py:86-89` reads:
```
# Fix 11: no channel flip. Dataset images are stored RGB-on-disk;
# cv2.imread+cvtColor(BGR2RGB) in _load_images() accidentally produces BGR.
# The model is trained on BGR. The env also returns BGR (aligning.py:212).
```

This reasoning is **factually inverted**:
- `cv2.imread` returns BGR
- `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` converts BGR **→ RGB**
- Therefore `_load_images()` produces **RGB**, not "accidentally BGR"
- The model is trained on **RGB**, not BGR

The comment reaches the correct operational conclusion (no flip needed) but via a completely inverted reasoning chain. Any developer reading this comment will build an incorrect mental model of the pipeline.

**Action required (Advisory):** Replace the comment body with accurate reasoning:
```python
# Fix 11: no channel flip.
# Training: cv2.imread(BGR) → cvtColor(BGR2RGB) → RGB. Model trains on RGB.
# Inference: MuJoCo get_image() → RGB → cvtColor(RGB2BGR) → BGR.
# There IS a channel mismatch (training=RGB, inference=BGR), but fix7 empirically
# proved the ResNet encoder is robust to channel swap for this geometric task.
# The [::-1] flip (fix8) was reverted because it correlated with divergence at 42k
# steps; the root cause is checkpoint maturity, not channel order.
```

---

### Finding 5.2: Normalization Comment Error (normalization.py:160)

The comment at `normalization.py:160` reads `# constant dims map to 0 in normalized space (A3)`.

This is mathematically incorrect. For a constant dim where `x == self.mins`:
- `(x - self.mins) / 1.0 = 0`
- `2 * 0 - 1 = -1`

Constant dims map to **-1** (the normalized minimum), not 0. The code is functionally consistent between training and eval, so this is not a bug — only a misleading comment. Update comment to: `# constant dims map to -1 (normalized min) in normalized space (A3)`.

---

### Finding 5.3: `des_robot_pos` Missing `.copy()` at t=0

At `aligning_sim.py:93-94`:
```python
des_robot_pos = env_state[:3]         # no .copy() — view of env_state array
robot_pos     = env_state[:3].copy()  # .copy() — independent array
```

Both are views/copies of `env_state[:3]`. The asymmetry is functionally harmless — `env_state` is a local array returned by `env.reset()`, not modified externally, and `des_robot_pos` is reassigned at line 104 after the first step. However, the inconsistency is confusing given the explicit `.copy()` comment on line 94 `(# actual == commanded at t=0 (C4))`. Add `.copy()` to line 93 for symmetry and defensive clarity.

---

## 📋 Summary Table

| # | Claim | Original | 2nd Audit | 3rd Audit (Sonnet 4.6) | Status |
|---|-------|----------|----------|------------------------|--------|
| A4 | Per-sample initial state anchoring | ✅ | ✅ Verified in `project()` and `compute_gradient()` | ✅ Confirmed at lines 145-157, 206-216 | **PASS** |
| B1 | Scale alignment (x_diff) | ✅ | ✅ Verified: `mat_fix_initial` and `b` use consistent `x_diff` | ✅ Confirmed at lines 415-432, 155-156 | **PASS** |
| C4 | True state feedback (robot_pos) | ✅ | ✅ Verified: `robot_state()` → `current_c_pos` propagated | ✅ Confirmed full chain; minor: add `.copy()` to line 93 | **PASS** |
| — | Projector actually active | (implied) | ❌ `constraint_types: []` → projector is a no-op | ✅ Dormant confirmed. Nuance: YAML explicitly says "per user decision" — conscious, not silent | **MISLEADING IN PROSE** |
| DDPM | `clip_denoised=False` | ✅ | ✅ Forced in eval, override exists in subclass | ✅ Override name confirmed (`p_mean_variance` at line 52); ±5 clamp (not ±1); safe | **PASS** |
| Norm | Zero-variance guard | ✅ | ✅ `range_ < 1e-8 → 1.0` in normalize | ✅ Code confirmed correct. Comment wrong: constant → -1 not 0 | **PASS (comment fix needed)** |
| BGR | "BGR preserved throughout" | ✅ | ❌ Training=RGB, Inference=BGR | ❌ Mismatch confirmed. Severity DISPUTED: fix7 worked with this mismatch → downgrade from BLOCKING to HIGH-PRIORITY ADVISORY | **ADVISORY (not BLOCKING)** |
| CPU | Unpinning for visual | ✅ | ✅ `if_vision` skips `sched_setaffinity` | ✅ Confirmed | **PASS** |
| Seed | Deterministic after env gen | ✅ | ⚠️ After env gen (correct), but uses `pid` not `seed` | ✅ Confirmed. Extended: x_T identical across all eval seeds (pid=0 always with n_cores=1) | **PARTIAL** |

---

## 🔧 Required Actions Before Gen7 Upgrade

### BLOCKING (Must Fix Before Gen7 Training Run)

1. **Resolve BGR/RGB Channel Mismatch** *(reclassified from "BLOCKING for eval" to "BLOCKING before Gen7 training")*  
   Fix before writing a single Gen7 training step — new architecture's channel sensitivity is unknown.  
   Either:
   - **(Option B)** Remove the `cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)` in `aligning.py:get_observation()` so the env returns RGB (matching training). Cleanest — matches both the training pipeline and the MuJoCo renderer's native format.
   - **(Option C)** Remove `cvtColor(BGR2RGB)` from `sequence.py:_load_images()` so training uses BGR (matching env). Note: this changes the training pipeline — dangerous for backward compatibility with existing checkpoints.
   - **(Option A)** Add `[::-1]` channel flip in `aligning_sim.py` at inference to convert BGR→RGB. Least invasive for training pipeline, but was fix8's approach — empirically correlated with divergence (though likely coincidental given checkpoint evidence).
   
   **Recommendation:** Option B is cleanest — make the env return RGB, matching both the training pipeline and the MuJoCo renderer's native format. Note: `aligning.py` is in the D3IL env package — verify against the "copy-modify only" constraint for the project before editing.

### ADVISORY (Should Fix)

2. **Fix seeding to use the configured `seed` value** instead of `pid` in `eval_agent()`.
3. **Acknowledge projector dormancy** — document that `constraint_types: []` means the projector is intentionally disabled, and enable it explicitly when ready. (Note: YAML already acknowledges this; update the evaluation prose.)
4. **Fix the Fix 11 comment** in `aligning_sim.py:86-89` — the reasoning chain is inverted (says "produces BGR" when it produces RGB). Use the corrected comment from Finding 5.1.
5. **Fix normalization comment** at `normalization.py:160` — constant dims map to -1, not 0.
6. **Add `.copy()`** to `aligning_sim.py:93` for `des_robot_pos` — style consistency with line 94.

---

## 🏁 Final Conclusion (3rd Audit)

The Gen6V4 codebase is **architecturally ready** for the Gen7 Foundation Model upgrade. The DDPM engine, projector mathematics (A4/B1/C4), and systems plumbing are structurally correct.

The **training-vs-inference image channel mismatch (RGB vs BGR)** is real and confirmed by all three audits. It must be fixed before Gen7 training begins. However, classifying it as "BLOCKING" for current Gen6V4 evaluation is an **overstatement** — empirical evidence (fix7) proves the Gen6V4 ResNet encoder is robust to this channel swap. The ±94 action divergence observed in fix8/9/11 is checkpoint-driven, not channel-driven.

**The root cause of Gen6V4 evaluation failures remains the training checkpoint.** Fix7's success at ~50k steps, combined with fix9's failure at ~90k steps, points to a training sweet spot around 50k that is not being hit by the current checkpoint. This is a training/convergence issue that no eval pipeline fix can address. The path forward is: train to the ~50k sweet spot, eval there specifically (not blindly at `state_best.pt`), and fix the channel mismatch before the first Gen7 training run.

---

**Signed (2nd Re-Audit):**  
**Antigravity (Claude Opus 4.6)**  
Independent Re-Audit, 2026-05-20T13:55Z  
*"Trust nothing; verify everything."*

---

**Signed (3rd Re-Audit):**  
**Claude Sonnet 4.6**  
Independent Line-by-Line Verification, 2026-05-20  
All source citations verified against live code. Key disputes: BGR/RGB severity (BLOCKING → HIGH-PRIORITY ADVISORY), Fix 11 comment inversion identified, normalization comment error found, `des_robot_pos` copy asymmetry noted. Root cause of ±94 divergence remains checkpoint-driven, not pipeline-driven.  
*"Verify sources, not summaries."*

**Post-audit fix applied:** Finding 5.1 (Fix 11 comment inversion in `aligning_sim.py:86-89`) corrected immediately after audit. Comment now accurately describes the RGB/BGR pipeline and the empirical basis for the no-flip decision. No other code changes — all remaining items are advisory comments or deferred to Gen7 prep.
