# Gen6V4 Visual-DPCC — Code & Math Audit Report

**Auditor:** Antigravity  
**Date:** 2026-05-19  
**Revised:** 2026-05-19 (post developer response — see `GEN6V4_AUDIT_RESPONSE.md`)  
**Scope:** Full module audit of the Gen6V4 Visual-DPCC codebase post-Fix 7  
**Focus:** ML/Robot Math correctness, real pitfalls, material bugs  

---

## Executive Summary

The codebase is well-structured and the Gen6V4 invariants (9D trajectory, Euler integration, SLSQP projector) are consistently upheld across the visual diffusion pipeline. Fix 7 resolved several critical parity issues.

~~However, the audit uncovered **4 critical ML/math bugs**, **3 high-severity issues**, and **several medium observations**. The two most dangerous findings are a **train↔eval image channel mismatch** and a **dead assertion in the base diffusion class** that silently passes the `clip_denoised=False` path during training when it shouldn't.~~

**[REV 1]** After developer review: **1 actively critical bug** (A1 — BGR/RGB mismatch, confirmed by developer), **3 confirmed-but-dormant/latent bugs** (A2, A3, A4 — valid code defects not triggered in current config), and **several design observations**. A4/B1 are fully moot while `constraint_types: []` (C1) keeps the projector disabled. The single highest-impact fix is A1.

---

## Table of Findings

| ID | Original Severity | **Post-Review Status** | Category | File | Short Title |
|----|----------|----------|----------|------|-------------|
| A1 | 🔴 CRITICAL | ✅ **CONFIRMED ACTIVE** | ML/Image | `sequence.py` vs `aligning.py` | Train BGR→RGB but eval feeds raw BGR |
| A2 | ~~🔴 CRITICAL~~ | 🟡 Confirmed dormant | ML/Diffusion | `diffusion.py:140` | Dead assertion — `assert RuntimeError()` always passes |
| A3 | ~~🔴 CRITICAL~~ | 🟡 Valid latent, not active | Math/Normalizer | `normalization.py:159` | LimitsNormalizer div-by-zero on constant dimensions |
| A4 | ~~🔴 CRITICAL~~ | 🟠 Confirmed, moot (C1) | Math/Projector | `projection.py:120,127` | Projector `skip_initial_state` uses only batch[0] |
| B1 | ~~🟠 HIGH~~ | 🟡 Plausible, moot (C1) | Robot/Dynamics | `projection.py:405-407` | Euler constraint normalization scale inconsistency |
| B2 | ~~🟠 HIGH~~ | ⚪ Not a bug (D3IL design) | Robot/Eval | `aligning_sim.py:91,96` | `des_robot_pos` drifts via dead-reckoned `pred_action` |
| B3 | 🟠 HIGH | 🟡 Confirmed dormant | ML/Conditioning | `eval_visual_aligning_dpcc.py:489-491` | `appendleft` reverses temporal order in deque window |
| C1 | 🟡 MEDIUM | ✅ Confirmed intentional | Config | `visual_aligning_eval.yaml:86` | Constraints entirely disabled — projector is a no-op |
| C2 | 🟡 MEDIUM | ✅ Confirmed | ML/Eval | `eval_visual_aligning_dpcc.py:760` | Projector skipped for `'diffuser'` variant |
| C3 | 🟡 MEDIUM | ✅ Confirmed (deliberate) | Math/Noise | `diffusion.py:158,168` | Halved noise scale `0.5 * randn` in sampling |
| C4 | ~~🟡 MEDIUM~~ | 🟠 **Escalated (REV 2)** — fix before DPCC-eval | Robot/Obs | `eval_visual_aligning_dpcc.py:477` | `obs_6d = [des_pos, des_pos]` — c_pos always == des_pos |

---

## 🔴 Critical Findings

### A1 — Train↔Eval Image Channel Mismatch (BGR vs RGB)

**CAUTION: The model is trained on RGB images but receives BGR images at eval time.** This is the single most impactful bug — it directly corrupts the visual conditioning signal.

**Root Cause:**

- **Training** — `sequence.py` line 167:
  ```python
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
  ```
  `ParityAligningDataset._load_images()` explicitly converts BGR→RGB before feeding to the model.

- **Eval** — `aligning.py` lines 211-215:
  ```python
  bp_image = self.bp_cam.get_image(depth=False)
  bp_image = cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)  # MuJoCo gives RGB, this makes BGR
  ```
  The env's `get_observation()` converts MuJoCo's native RGB **to BGR**. The eval sim loop (`aligning_sim.py` lines 83-84) then just normalizes:
  ```python
  bp_image = bp_image.transpose((2, 0, 1)) / 255.  # Feeds BGR to model
  ```
  **No BGR→RGB conversion happens before the model receives the image.**

- **The GEN6V4_CODE_STRUCT.md invariant** (line 131-133) claims "No `cv2.cvtColor` anywhere in the training or eval image loading path (FIX_7.2)" — but this is **factually wrong**: `sequence.py` line 167 does exactly that in training.

**Impact:** Red and Blue channels are swapped at inference. The ResNet visual encoder was trained on RGB pixel statistics (and uses `imagenet_norm=True` which assumes RGB channel ordering). At eval, it receives BGR — the ImageNet normalization subtracts wrong means from wrong channels, producing a distribution shift in the FiLM conditioning. This silently degrades policy quality without crashing.

**Fix:** Either:
1. Remove `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` from `sequence.py:167` so training also uses BGR (matching the `d3il` convention), **or**
2. Add `bp_image = bp_image[::-1]` (channel flip) in `aligning_sim.py` after the `transpose` so eval feeds RGB.

Option 2 is recommended — train on RGB is correct when `imagenet_norm=True`.

> **[REV 1] Developer Response:** ✅ **CONFIRMED REAL. FIX_7.2 was incomplete.** Developer verified against live code and agrees. FIX_7.2 fixed `d3il/environments/dataset/aligning_dataset.py` (the D3IL agent dataset), not `sequence.py` (the DPCC training dataset actually used by `train_visual_aligning_dpcc.py`). The GEN6V4_CODE_STRUCT.md invariant claim has been retracted by the developer. **Auditor's Option 2 accepted as recommended fix.**

### A2 — Dead Assertion in Base `GaussianDiffusion.p_mean_variance`

**CAUTION: `assert RuntimeError()` is always True** — the `clip_denoised=False` else-branch never actually raises.

`diffusion.py` lines 137-140:
```python
if self.clip_denoised:
    x_recon.clamp_(-1., 1.)
else:
    assert RuntimeError()
```

`RuntimeError()` constructs an exception **object**, which is truthy. `assert <truthy>` always passes. This means:
- The base class's `p_mean_variance` silently succeeds with `clip_denoised=False` — no error is raised.
- The code *intended* to force `clip_denoised=True` in the base class, but the assertion is a no-op.

**Impact for Gen6V4:** The child class `VisualGaussianDiffusion` overrides `p_mean_variance` entirely, so this dead branch is never reached during Visual-DPCC inference. However:
- During **training**, `p_losses` calls `self.model(x_noisy, cond, t)` on the **backbone** (not the diffusion `p_mean_variance`), so this doesn't fire during training either.
- If anyone calls the base class `p_mean_variance` directly (e.g., for non-visual paths or debugging), it will silently succeed instead of raising.

**Fix:**
```python
else:
    raise RuntimeError("clip_denoised=False not supported in base GaussianDiffusion")
```

> **[REV 1] Developer Response:** ✅ Confirmed. However, **dormant for Gen6V4** — `VisualGaussianDiffusion` overrides `p_mean_variance` entirely and `clip_denoised` is forced False at eval. This branch is never reached. ~~🔴 CRITICAL~~ → 🟡 Low priority. Fix it as a one-liner to prevent future misuse of the base class.

---

### A3 — LimitsNormalizer Division-by-Zero on Constant Dimensions

**CAUTION: If any dimension of the training data is constant, `LimitsNormalizer.normalize()` produces `NaN` or `±inf`, silently poisoning the entire trajectory.**

`normalization.py` lines 157-162:
```python
def normalize(self, x):
    x = (x - self.mins) / (self.maxs - self.mins)  # division by zero if maxs == mins
    x = 2 * x - 1
    return x
```

`ParityAligningDataset` uses `LimitsNormalizer` directly (not `SafeLimitsNormalizer` which has the eps-fallback). If any action or obs dimension has constant value across all training episodes (e.g., z-position on a flat table), `maxs - mins = 0` → `NaN` propagates through the entire diffusion training.

**Why this is realistic:** The Aligning task operates on a flat table. The z-axis action (`dz`) and z-position (`c_pos_z`) may have near-zero variance. Even if not exactly constant, floating-point coincidences can make `maxs == mins` for a single dimension.

**Impact:** Silent `NaN` in normalized trajectories → diffusion loss becomes `NaN` → model parameters destroyed within a few gradient steps. The training script does log normalizer stats, but doesn't check for zero-range and will not stop.

**Fix:** Use `SafeLimitsNormalizer` instead of `LimitsNormalizer` in `sequence.py` lines 94-95, or add a guard:
```python
range_ = self.maxs - self.mins
range_[range_ < 1e-8] = 1.0  # constant dims map to 0 in normalized space
x = (x - self.mins) / range_
```

> **[REV 1] Developer Response:** Valid latent bug, **not currently active.** Live checkpoint normalizer ranges are all non-zero (`act_normalizer mins=[-0.0083, -0.0083, -0.0083] maxs=[0.0083, 0.0083, 0.0134]`; `obs_normalizer` similarly healthy). No div-by-zero in production today. ~~🔴 CRITICAL~~ → 🟡 Latent risk for edge-case datasets or retraining. Worth adding eps-guard, not urgent.

---

### A4 — Projector `skip_initial_state` Uses Only `batch[0]` for All Samples

**CAUTION: The SLSQP projector anchors **all** batch trajectories to the initial state of the **first** sample, causing wrong Euler dynamics constraints for `batch_size > 1`.**

`projection.py` lines 119-128:
```python
if self.skip_initial_state:
    s_0 = trajectory_reshaped[0, :self.transition_dim]  # only batch[0]!
    ...
    for constraint in self.dynamic_constraints.constraint_list:
        if constraint[0] == 'deriv':
            x_idx = int(constraint[1][0])
            b[counter * self.horizon] = s_0[x_idx]  # same s_0 for all batches
```

The `b` vector (RHS of equality constraints) is set once from `trajectory_reshaped[0]` and reused for **all** batch elements in the SLSQP loop at line 151. This means every trajectory in the batch is constrained to start from sample 0's initial state.

The same bug exists in `compute_gradient` at `projection.py` lines 192-199.

**Impact:** At eval with `batch_size=6` (DPCC variants), the 5 non-first trajectories are projected onto an incorrect constraint manifold — their Euler dynamics are enforced starting from the wrong position. This biases trajectory selection (`minimum_projection_cost`) and corrupts the projected trajectory shapes.

**Fix:** Move `s_0` extraction inside the per-sample loop:
```python
for i in range(batch_size):
    if self.skip_initial_state:
        s_0 = trajectory_reshaped[i, :self.transition_dim]
        # update b for this sample
        ...
    res = minimize(...)
```

> **[REV 1] Developer Response:** ✅ Confirmed correct reading. **However, currently moot** — finding C1 (`constraint_types: []`) means the projector's constraint list is empty and `project()` is a no-op for all variants. A4 has zero runtime impact while C1 is active. ~~🔴 CRITICAL~~ → 🟠 Fix when constraints are re-enabled.

---

## 🟠 High-Severity Findings

### B1 — Euler Constraint Normalization Scale Inconsistency on Initial State

`projection.py` lines 402-418:

The dynamic constraint encodes: `x[t+1] = x[t] + dt * dx[t]`  
In normalized coordinates, the matrix encodes:
```
x_diff * x_n[t] + dt * dx_diff * dx_n[t] - x_diff * x_n[t+1] = -dt * dx_sum
```

When `skip_initial_state=True`, the initial-state row is:
```python
mat_fix_initial[0, x_idx] = 1
vec_append = torch.cat((torch.tensor([0], ...), vec_append), dim=0)
```

But at projection time (line 127):
```python
b[counter * self.horizon] = s_0[x_idx]
```

This sets `b[0] = s_0[x_idx]`, which is the **normalized** initial state value. However, the constraint row multiplies by 1 (not by `x_diff`), while all other rows multiply by `x_diff`. This scale mismatch means the initial-state constraint is `1 * x_n[0] = s_0_normalized` while dynamics rows enforce `x_diff * x_n[t] + ... = ...`.

**Impact:** For typical Franka workspace values (`x_diff ~ 0.4`), the initial-state constraint is ~2.5x "weaker" relative to the dynamics rows. The SLSQP solver may allow small drift from the true initial state.

**Fix:** The initial-state row should also multiply by `x_diff`:
```python
mat_fix_initial[0, x_idx] = x_diff
```
and `b[0]` should be set to `x_diff * s_0[x_idx]` at projection time.

> **[REV 1] Developer Response:** Plausible argument, not independently verified against the matrix math. **Currently moot** for the same reason as A4 — `constraint_types: []` (C1) means constraints are empty and the projector is a no-op. ~~🟠 HIGH~~ → 🟡 Flag for when constraints are re-enabled.

---

### B2 — Eval `des_robot_pos` Drifts via Dead-Reckoned Action Integration

`aligning_sim.py` lines 91-96:
```python
pred_action = agent.predict((bp_image, inhand_image, des_robot_pos), if_vision=True)
pred_action = pred_action[0] + des_robot_pos      # absolute position from delta
...
des_robot_pos = pred_action[:3]                     # next "des_robot_pos" = old + delta
```

After step, the new observation is:
```python
robot_pos, bp_image, inhand_image = obs            # robot_pos from env
```

But `robot_pos` is **NOT** used as the next `des_robot_pos`. Instead, the dead-reckoned `pred_action[:3]` becomes the next `des_robot_pos`. This means the actual robot position (from the sim) diverges from the commanded position (what the agent sees as `des_robot_pos`).

**Note:** This is the D3IL baseline behavior (the original code does the same). It's a design decision, not strictly a bug. But it's worth understanding that the agent never sees the actual sim state — only its own commands. Under high-velocity SLSQP-projected trajectories with K=100 denoising steps, the PD tracking error accumulates over the rollout.

> **[REV 1] Developer Response:** Correct observation, **not a bug.** This is confirmed original D3IL behavior. The agent was trained on the same dead-reckoned `des_robot_pos` loop, so training and eval are consistent. ~~🟠 HIGH~~ → ⚪ Design decision, reclassified as observation.

---

### B3 — `appendleft` Reverses Temporal Ordering in Deque Window

`eval_visual_aligning_dpcc.py` lines 489-491:
```python
self.bp_image_context.appendleft(bp_t)
self.inhand_image_context.appendleft(inhand_t)
self.obs_context.appendleft(obs_t)
```

`appendleft` inserts the **newest** frame at index 0. When `window_size > 1`, the deque contains `[newest, ..., oldest]`. Then at lines 498-500:
```python
bp_seq = torch.cat(list(self.bp_image_context), dim=0)      # (W, C, H, W)
```

This produces a tensor where `bp_seq[0]` = newest frame, `bp_seq[-1]` = oldest frame.

The VisualUNet's `encode_visual` (line 103) does `features.view(B, T, -1).mean(dim=1)`, which mean-pools over the window — making ordering irrelevant **for the current architecture**.

**Impact:** Currently dormant (`window_size=1` in config). But the code intention is clearly temporal ordering — using `appendleft` with `torch.cat(list(...))` produces reversed-time tensors, which would be wrong for any order-sensitive encoder.

> **[REV 1] Developer Response:** ✅ Confirmed dormant. Safe to leave for now, worth fixing before experimenting with temporal context windows (`window_size > 1`).

---

## 🟡 Medium Observations

### C1 — Constraints Entirely Disabled in Config

`visual_aligning_eval.yaml` line 86:
```yaml
constraint_types: []  # OPTION A: No constraints (Turned Off)
```

This means `constraint_list` in `setup_dpcc_projector` is always empty → the Projector has no bounds, no dynamics → `project()` solves an unconstrained QP (identity) → DPCC is a no-op.

**Impact:** All projection variants (`gradient`, `post_processing`, `model_free`, etc.) produce identical results to `diffuser` (no projection). The DPCC constraint engine is effectively bypassed.

> **[REV 1] Developer Response:** ✅ **Confirmed intentional.** This is a deliberate "baseline first" evaluation choice — run the diffusion backbone without projection constraints to isolate model quality from projection quality. Not an oversight. **Consequence:** A4 and B1 have zero impact right now. DPCC projection is not yet active in production eval.

---

### C2 — Projector Skipped for `'diffuser'` Variant

`eval_visual_aligning_dpcc.py` line 760:
```python
if 'diffuser' not in variant and obs_normalizer is not None:
    projector = setup_dpcc_projector(...)
```

The `'diffuser'` variant never gets a projector. Combined with C1 (empty constraints), all variants are effectively projector-less.

---

### C3 — Halved Noise Scale in Sampling

`diffusion.py` lines 158, 168:
```python
noise = 0.5 * torch.randn_like(x)
x = 0.5 * torch.randn(shape, device=device)
```

Standard DDPM sampling uses unit-variance noise. The `0.5x` scaling reduces sample diversity. This is a deliberate DPCC modification (also in the original codebase), not a bug — but it effectively uses `sigma_t = 0.5 * sqrt(beta_tilde_t)` instead of the standard `sigma_t = sqrt(beta_tilde_t)`.

---

### C4 — `obs_6d = [des_pos, des_pos]` — c_pos Slot Duplicates des_pos

`eval_visual_aligning_dpcc.py` line 477:
```python
obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])  # (6,)
```

The 6D observation is `[des_c_pos(3) | c_pos(3)]`, but at eval time, both halves are identical. The model was **trained** on real `(des_c_pos, c_pos)` pairs where `des_c_pos != c_pos` (PD tracking lag). At eval, the model always sees `des == actual`, which is a distribution shift on the observation.

> **[REV 1] Developer Response:** ✅ Plausible distribution shift confirmed. Training data has non-equal `(des_c_pos, c_pos)` pairs (confirmed by obs_normalizer range split). Low impact with constraints off since the 6D obs only feeds the projector's initial state anchor.

> **[REV 2] Developer Escalation:** ~~🟡 MEDIUM~~ → 🟠 **Promoted to fix-before-DPCC-eval.** Once C1 re-enables constraints, the projector's `b[0]` vector is seeded from the 6D obs. Feeding `[des_pos, des_pos]` (zero PD lag) when the model trained on real PD lag may suppress constraint-aware trajectory generation. **Fix C4 alongside A4+B1, before re-enabling C1.**

---

## Non-Issues Verified Clean

| Item | Status | Notes |
|------|--------|-------|
| 9D trajectory format consistency | ✅ | `[act(3) | des_c_pos(3) | c_pos(3)]` upheld everywhere |
| `clip_denoised=False` forced at eval | ✅ | Line 698 of eval script forces it |
| Projector `transition_dim=9` | ✅ | Correctly set in `setup_dpcc_projector` |
| Euler indices `[6←0, 7←1, 8←2]` | ✅ | Config constraint_list matches trajectory layout |
| U-Net horizon padding | ✅ | `padded_horizon = ceil(H/8)*8` correct |
| `apply_conditioning` obs anchor | ✅ | Correctly snaps `x[:,0,3:]` to initial obs |
| Training loss weights | ✅ | `action_weight=10` on `a0` dims, rest balanced |
| EMA decay | ✅ | 0.995 standard for visual DDPM |
| Window-size = 1 config | ✅ | Matches single-frame training data |

---

## Recommended Priority Order

~~1. **A1** — Fix the BGR/RGB mismatch. This is likely the #1 factor hurting eval performance right now.~~
~~2. **A4** — Fix the batch-0 initial state broadcast in the Projector (matters for `batch_size > 1`).~~
~~3. **A3** — Switch to `SafeLimitsNormalizer` or add zero-range guards.~~
~~4. **A2** — Fix the dead assertion (low effort, prevents future misuse).~~
~~5. **C1** — Re-enable constraints if DPCC projection is desired.~~
~~6. **B1** — Fix the normalizer scale mismatch in the initial-state constraint row.~~
~~7. **B2/B3** — Document or fix depending on intended behavior.~~

~~**[REV 1] Revised Priority Order** (agreed between auditor and developer):~~

~~1. **A1** — Fix BGR→RGB at eval. The one active visual pipeline bug confirmed to affect every inference step. Fix immediately.~~
~~2. **C1** — Re-enable constraints (`constraint_types: ['bounds', 'dynamics']`) before claiming DPCC results. Running with empty projector = measuring plain diffusion baseline, not Visual-DPCC.~~
~~3. **A4 + B1** — Fix together when constraints (C1) are re-enabled. A4's batch-0 broadcast and B1's scale mismatch are both moot until then.~~
~~4. **A2** — One-line fix (`assert` → `raise`), do it in passing.~~
~~5. **A3** — Add eps-guard to `LimitsNormalizer`. Valid latent bug, not urgent for current data.~~
~~6. **B3** — Fix `appendleft` ordering before experimenting with `window_size > 1`.~~
~~7. **C4** — Low impact observation, revisit if constraint-projection performance is unexpectedly poor.~~
~~8. ~~**B2**~~ — Not a bug. D3IL design decision. No action needed.~~

**[REV 2] Final Priority Order** (auditor + developer Round 2 agreement):

> **Key amendment:** A4+B1+C4 must be fixed *before* C1 is re-enabled, not after. Enabling constraints with the batch-0 broadcast active means 5/6 SLSQP trajectories are anchored to the wrong initial state.

| # | Item(s) | Action | When |
|---|---------|--------|------|
| 1 | **A1** | Add `cv2.cvtColor(BGR2RGB)` in `aligning_sim.py` after transpose | Immediately, before next eval run |
| 2 | **A4 + B1 + C4** | Fix batch-0 broadcast, scale row, and c_pos duplication | Before re-enabling constraints |
| 3 | **C1** | Re-enable `constraint_types: ['bounds', 'dynamics']` | After #2 is fixed |
| 4 | **A2** | `assert RuntimeError()` → `raise RuntimeError(...)` | Any passing commit |
| 5 | **A3** | Add eps-guard to `LimitsNormalizer` | Before next retraining run |
| 6 | **B3** | Fix `appendleft` → `append` | Before `window_size > 1` experiments |
| — | ~~**B2**~~ | Not a bug. D3IL design decision. | No action needed |

---

## Recommended Fixes

Concrete code changes for each actionable finding, ordered by the REV 2 priority.

---

### Fix A1 — BGR→RGB at eval (`aligning_sim.py`)

**File:** `d3il/simulation/aligning_sim.py`  
**Lines:** 82-84, 106-107

The env returns BGR images. Add a channel flip after `transpose` so the model receives RGB (matching training).

```diff
 # After initial reset (lines 82-84):
                     env_state, bp_image, inhand_image = obs
-                    bp_image = bp_image.transpose((2, 0, 1)) / 255.
-                    inhand_image = inhand_image.transpose((2, 0, 1)) / 255.
+                    bp_image = bp_image.transpose((2, 0, 1))[::-1].copy() / 255.       # BGR→RGB
+                    inhand_image = inhand_image.transpose((2, 0, 1))[::-1].copy() / 255.  # BGR→RGB

 # After each step (lines 106-107):
-                        bp_image = bp_image.transpose((2, 0, 1)) / 255.
-                        inhand_image = inhand_image.transpose((2, 0, 1)) / 255.
+                        bp_image = bp_image.transpose((2, 0, 1))[::-1].copy() / 255.       # BGR→RGB
+                        inhand_image = inhand_image.transpose((2, 0, 1))[::-1].copy() / 255.  # BGR→RGB
```

**Why `.copy()`:** `[::-1]` produces a view with negative strides. PyTorch's `torch.from_numpy()` doesn't support negative strides — `.copy()` makes it contiguous.

---

### Fix A4 — Projector batch-0 initial state broadcast (`projection.py`)

**File:** `diffuser_visual_aligning/sampling/projection.py`

Two locations must be patched: `project()` (line 119-128) and `compute_gradient()` (line 192-199). Move `s_0` extraction inside the per-sample loop.

**`project()` — lines 119-128 and 151:**

```diff
-        if self.skip_initial_state:
-            s_0 = trajectory_reshaped[0, :self.transition_dim]
-            if self.solver == 'proxsuite' or self.solver == 'gurobi':
-                s_0 = s_0.cpu().numpy()
-            counter = 0
-            for constraint in self.dynamic_constraints.constraint_list:
-                if constraint[0] == 'deriv':
-                    x_idx = int(constraint[1][0])
-                    b[counter * self.horizon] = s_0[x_idx]
-                    counter += 1
 
         ...
 
         for i in range(batch_size):
+            # Per-sample initial state anchor
+            if self.skip_initial_state:
+                s_0 = trajectory_reshaped[i, :self.transition_dim]
+                if self.solver == 'proxsuite' or self.solver == 'gurobi':
+                    s_0 = s_0.cpu().numpy()
+                counter = 0
+                for constraint in self.dynamic_constraints.constraint_list:
+                    if constraint[0] == 'deriv':
+                        x_idx = int(constraint[1][0])
+                        b[counter * self.horizon] = s_0[x_idx]
+                        counter += 1
+
             # Cost
             cost_fun = lambda x: 0.5 * x @ Q @ x + r_np_double[i] @ x
             ...
```

**`compute_gradient()` — lines 192-199 and 204:**

```diff
-        if self.skip_initial_state:
-            s_0 = trajectory_reshaped[0, :self.transition_dim]
-            counter = 0
-            for constraint in self.dynamic_constraints.constraint_list:
-                if constraint[0] == 'deriv':
-                    x_idx = int(constraint[1][0])
-                    b[counter * self.horizon] = s_0[x_idx]
-                    counter += 1
 
         ...
 
         for i in range(trajectory.shape[0]):
+            if self.skip_initial_state:
+                s_0 = trajectory_reshaped[i, :self.transition_dim]
+                counter = 0
+                for constraint in self.dynamic_constraints.constraint_list:
+                    if constraint[0] == 'deriv':
+                        x_idx = int(constraint[1][0])
+                        b[counter * self.horizon] = s_0[x_idx]
+                        counter += 1
             grad1[i] = - A.T @ (A @ trajectory_reshaped[i] - b)
             ...
```

---

### Fix B1 — Euler constraint initial-state scale row (`projection.py`)

**File:** `diffuser_visual_aligning/sampling/projection.py`  
**Lines:** 414-418

The initial-state constraint row uses coefficient `1` while dynamics rows use `x_diff`. Match the scale:

```diff
                 if self.skip_initial_state:
                     mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)
-                    mat_fix_initial[0, x_idx] = 1
+                    mat_fix_initial[0, x_idx] = x_diff  # Scale must match dynamics rows
                     mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                     vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)
```

And in the `project()` / `compute_gradient()` methods where `b[0]` is set (after applying Fix A4), the assigned value must also scale:

```diff
-                        b[counter * self.horizon] = s_0[x_idx]
+                        b[counter * self.horizon] = x_diff * s_0[x_idx]  # match scaled row
```

Where `x_diff = self.normalizer.maxs[x_idx] - self.normalizer.mins[x_idx]`. This requires storing `x_diff` per constraint at build time (e.g., `self._x_diffs`).

> **[REV 3] Developer Response:** Mathematically self-consistent argument accepted, but **not independently verified** against the full constraint matrix construction. Developer recommends: **apply with a unit test** — run a trivial 1D trajectory through the projector before and after the change, and confirm the initial-state constraint is tighter, not looser. Do not deploy without this validation.

---

### Fix C4 — `obs_6d` c_pos duplication (`eval_visual_aligning_dpcc.py`)

**File:** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`  
**Line:** 477

The `c_pos` slot should use the actual robot position from the sim, not the commanded position:

```diff
-            obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])  # (6,)
+            # c_pos = actual robot state from env (robot_pos_np),
+            # des_c_pos = commanded position (des_robot_pos_np)
+            obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # (6,)
```

This requires `robot_pos_np` to be extracted from the env observation. In the `predict()` method, the sensor tuple is `(bp_image, inhand_image, des_robot_pos)`. To get `c_pos`, either:

1. **Expand the sensor tuple** to include `robot_pos` from `aligning_sim.py` (line 98: `robot_pos, bp_image, inhand_image = obs`), or
2. **Pass `robot_pos` as a 4th element**: `agent.predict((bp_image, inhand_image, des_robot_pos, robot_pos), ...)` and unpack in the wrapper.

Option 2 minimal change — **full 3-file diff** (updated per REV 3 developer feedback):

**Step 1 — `aligning_sim.py`: Initialize `robot_pos` before the loop and pass it as 4th element:**
```diff
 # aligning_sim.py lines 86-90:
                     des_robot_pos = env_state[:3]
+                    robot_pos = env_state[:3].copy()  # actual == commanded at t=0
                     done = False

                     while not done:
-                        pred_action = agent.predict((bp_image, inhand_image, des_robot_pos), if_vision=self.if_vision)
+                        pred_action = agent.predict((bp_image, inhand_image, des_robot_pos, robot_pos), if_vision=self.if_vision)
```
`robot_pos` is automatically updated each step via line 98: `robot_pos, bp_image, inhand_image = obs`.

**Step 2 — `eval_visual_aligning_dpcc.py`: Unpack the 4th element in `predict()`:**
```diff
 # eval_visual_aligning_dpcc.py predict() unpack (line ~449):
-            bp_np, inhand_np, des_robot_pos_np = state
+            bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state
```

**Step 3 — `eval_visual_aligning_dpcc.py`: Use `robot_pos_np` in obs construction (line 477):**
```diff
-            obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])  # (6,)
+            # c_pos = actual robot state from env (robot_pos_np),
+            # des_c_pos = commanded position (des_robot_pos_np)
+            obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # (6,)
```

> **[REV 3] Developer Response:** Original diff was incomplete — missing `robot_pos` initialization before the while loop and `predict()` unpack update. Both additions above are required; without them, the fix raises an unpack error at runtime. The 3-step diff above is the complete fix.

---

### Fix A2 — Dead assertion (`diffusion.py`)

**File:** `diffuser_visual_aligning/models/diffusion.py`  
**Line:** 140

One-line fix:

```diff
         if self.clip_denoised:
             x_recon.clamp_(-1., 1.)
         else:
-            assert RuntimeError()
+            raise RuntimeError("clip_denoised=False not supported in base GaussianDiffusion")
```

---

### Fix A3 — LimitsNormalizer eps-guard (`normalization.py`)

**File:** `diffuser_visual_aligning/datasets/normalization.py`  
**Lines:** 157-162

Add a zero-range guard:

```diff
     def normalize(self, x):
         ## [ 0, 1 ]
-        x = (x - self.mins) / (self.maxs - self.mins)
+        range_ = self.maxs - self.mins
+        range_[range_ < 1e-8] = 1.0  # constant dims → 0 in normalized space
+        x = (x - self.mins) / range_
         ## [ -1, 1 ]
         x = 2 * x - 1
         return x
```

Also guard `unnormalize` symmetrically:

```diff
     def unnormalize(self, x, eps=1e-4):
         ...
         x = (x + 1) / 2.
-        return x * (self.maxs - self.mins) + self.mins
+        range_ = self.maxs - self.mins
+        range_[range_ < 1e-8] = 0.0  # constant dims → original min value
+        return x * range_ + self.mins
```

---

### Fix B3 — Deque temporal ordering (`eval_visual_aligning_dpcc.py`)

**File:** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`  
**Lines:** 489-491

Use `append` (right) instead of `appendleft` so the deque stores `[oldest, ..., newest]`:

```diff
-            self.bp_image_context.appendleft(bp_t)
-            self.inhand_image_context.appendleft(inhand_t)
-            self.obs_context.appendleft(obs_t)
+            self.bp_image_context.append(bp_t)
+            self.inhand_image_context.append(inhand_t)
+            self.obs_context.append(obs_t)
```

And the fill loop (lines 493-496) similarly:

```diff
-                self.bp_image_context.appendleft(bp_t)
-                self.inhand_image_context.appendleft(inhand_t)
-                self.obs_context.appendleft(obs_t)
+                self.bp_image_context.append(bp_t)
+                self.inhand_image_context.append(inhand_t)
+                self.obs_context.append(obs_t)
```

This ensures `torch.cat(list(...))` produces `[oldest_frame, ..., newest_frame]` — correct temporal ordering for any future order-sensitive encoder.

---

### Fix C1 — Re-enable constraints (`visual_aligning_eval.yaml`)

**File:** `config/visual_aligning_eval.yaml`  
**Line:** 86

After A4+B1+C4 are fixed, flip the switch:

```diff
-constraint_types: []               # OPTION A: No constraints (Turned Off)
+constraint_types: ['bounds', 'dynamics']  # Original (with constraints)
```

---

*End of audit. Revised 2026-05-19 (REV 3 — final). Audit closed by mutual agreement between auditor and developer. All findings accepted; only open item is unit test validation for B1 before deployment.*
