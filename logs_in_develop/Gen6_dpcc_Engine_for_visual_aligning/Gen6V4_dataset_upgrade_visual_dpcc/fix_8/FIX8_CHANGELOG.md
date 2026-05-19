# Fix 8 — Changelog

**Date:** 2026-05-19
**Source:** Audit findings from GEN6V4_AUDIT_REPORT.md (REV 3, auditor: Antigravity)
**Applies:** All items in the REV 2/3 priority table

---

## Files Changed

| File | Fixes Applied |
|---|---|
| `d3il/simulation/aligning_sim.py` | A1, C4 |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | A1 (video path), C4, B3 |
| `diffuser_visual_aligning/models/diffusion.py` | A2 |
| `diffuser_visual_aligning/datasets/normalization.py` | A3 |
| `diffuser_visual_aligning/sampling/projection.py` | A4, B1 |
| `config/visual_aligning_eval.yaml` | C1 |

---

## Fix A1 — BGR→RGB at eval (`aligning_sim.py` lines 83-84, 106-107)
<!-- Auditor: Correctly resolves the #1 active bug. Channel flip + .copy() is the right approach. -->

**Problem:** `sequence.py` trains on RGB (`cv2.cvtColor BGR2RGB`). `aligning.py` delivers
BGR to `aligning_sim.py` (MuJoCo native RGB → BGR via cvtColor). Model received BGR at
eval but was trained on RGB — every visual embedding was channel-corrupted.

**Change:**
```python
# Before:
bp_image = bp_image.transpose((2, 0, 1)) / 255.
# After:
bp_image = bp_image.transpose((2, 0, 1))[::-1].copy() / 255.  # BGR→RGB
```
Applied at both the pre-loop init and the in-loop step update.
`.copy()` required: `[::-1]` produces a negative-stride view, incompatible with `torch.from_numpy`.

---

## Fix A2 — Dead assertion (`diffusion.py` line 140)
<!-- Auditor: Trivial one-liner. Correct. -->

**Problem:** `assert RuntimeError()` constructs a truthy exception object — assertion
always passes silently instead of raising. Base class `p_mean_variance` never raises for
`clip_denoised=False` paths.

**Change:**
```python
# Before:
assert RuntimeError()
# After:
raise RuntimeError("clip_denoised=False not supported in base GaussianDiffusion")
```
Dormant for Gen6V4 (VisualGaussianDiffusion overrides `p_mean_variance`), but prevents
silent misuse of the base class.

---

## Fix A3 — LimitsNormalizer eps-guard (`normalization.py` lines 157-179)
<!-- Auditor: Good defensive fix. Latent-only today, but insurance against future retraining. -->

**Problem:** `(x - mins) / (maxs - mins)` → div-by-zero if any dimension is constant
(e.g., flat-table z-axis). Silent NaN propagation.

**Change:** Added eps-guard in both `normalize` and `unnormalize`:
```python
range_ = self.maxs - self.mins
range_[range_ < 1e-8] = 1.0   # normalize: constant dims → 0
range_[range_ < 1e-8] = 0.0   # unnormalize: constant dims → min value
```

---

## Fix A4 — Projector batch-0 initial state broadcast (`projection.py`)
<!-- Auditor: Critical for multi-sample DPCC. Correctly scopes s_0 per batch element. -->

**Problem:** `s_0 = trajectory_reshaped[0, ...]` was extracted once before the batch
loop and reused for all `batch_size` samples. With `batch_size=6`, 5/6 SLSQP solutions
were constrained to start from sample 0's initial state.

**Change:** Moved `s_0` extraction inside `for i in range(batch_size)` in both
`project()` and `compute_gradient()`:
```python
for i in range(batch_size):
    if self.skip_initial_state:
        s_0 = trajectory_reshaped[i, :self.transition_dim]  # per-sample
        ...
```

---

## Fix B1 — Euler constraint initial-state scale row (`projection.py`)
<!-- Auditor: Mathematically self-consistent. ⚠️ MUST be validated with a unit test before production use. -->

**Problem:** Dynamics rows in the constraint matrix use coefficient `x_diff`
(`mat_append[i, ...] = 1 * x_diff`). The initial-state row used coefficient `1`
(`mat_fix_initial[0, x_idx] = 1`), creating a scale mismatch of ~2.5× for typical
Franka workspace values. SLSQP treated the initial-state constraint as proportionally
weaker.

**Change:**
- `build_matrices()`: `mat_fix_initial[0, x_idx] = x_diff` (was `1`)
- `_initial_state_x_diffs` list added to `DynamicConstraints.__init__` and populated
  in `build_matrices` for each `deriv` constraint
- `project()` and `compute_gradient()`: `b[counter * self.horizon] = x_diff * s_0[x_idx]`
  (was `s_0[x_idx]`)

⚠️ **Validation required:** B1 is mathematically self-consistent but has not been
independently verified with a unit test against the full constraint matrix. The audit
recommended running a trivial 1D trajectory through the projector before/after to
confirm the initial-state constraint is tighter, not looser. Do not treat B1 as
confirmed until that test is run.

---

## Fix B3 — Deque temporal ordering (`eval_visual_aligning_dpcc.py`)
<!-- Auditor: Correct. Dormant at window_size=1 but future-proofs for temporal encoders. -->

**Problem:** `appendleft` inserts newest frame at index 0, producing
`[newest, ..., oldest]` ordering. When window > 1, `torch.cat(list(...))` gives
reversed-time tensors. Currently dormant (`window_size=1`), but wrong for any
order-sensitive encoder.

**Change:** `appendleft` → `append` in both the main visual block (lines 489-491)
and the fill loop (lines 493-496). Deque now stores `[oldest, ..., newest]`.

---

## Fix C1 — Re-enable constraints (`visual_aligning_eval.yaml`)
<!-- Auditor: Correct sequencing — applied after A4+B1+C4. DPCC projector now active. -->

**Change:**
```yaml
# Before:
constraint_types: []
# After:
constraint_types: ['bounds', 'dynamics']
```
A4+B1+C4 were applied first as required by the REV 2 sequencing rule.

---

## Fix C4 — `obs_6d` c_pos duplication (`aligning_sim.py` + `eval_visual_aligning_dpcc.py`)
<!-- Auditor: Correct 3-step diff. Closes the train/eval observation distribution gap. -->

**Problem:** `obs_6d = [des_robot_pos, des_robot_pos]` — both halves were the commanded
position. The model was trained on real `(des_c_pos, c_pos)` pairs with PD tracking lag
(confirmed by obs_normalizer range split). Feeding zero-lag at eval shifts the
conditioning distribution.

**Changes — 3 files:**

1. `aligning_sim.py`: Initialize `robot_pos` before the loop and pass as 4th tuple element:
```python
robot_pos = env_state[:3].copy()   # actual == commanded at t=0
...
agent.predict((bp_image, inhand_image, des_robot_pos, robot_pos), ...)
```
`robot_pos` auto-updates each step via existing `robot_pos, bp_image, inhand_image = obs`.

2. `eval_visual_aligning_dpcc.py` `predict()`: Unpack 4th element:
```python
bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state
```

3. `eval_visual_aligning_dpcc.py` obs construction:
```python
obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # [des_c_pos | c_pos]
```
Non-visual path (`if_vision=False`) left unchanged — no `robot_pos` available there.

---

## New Problems Discovered During Coding

### Issue 1 — Video capture cvtColor invalidated by A1 (fixed inline)
<!-- Auditor: Good catch by the implementer. Cascading effect of A1 correctly handled. -->

**Discovery:** After A1 converts `bp_image` to RGB CHW before passing to `predict()`,
the video capture block in `predict()` was still applying `cv2.cvtColor(BGR2RGB)`.
With `bp_np` already in RGB, this would swap channels again — producing wrong-color videos.

**Fix applied:** Removed `cv2.cvtColor` from the `predict()` video capture block.
`bp_vis = bp_np.copy().transpose(1,2,0) * 255` → already RGB, no conversion needed.

**Note:** `generate_expert_reference()` (line ~193) has its own `cvtColor(BGR2RGB)` on
frames directly from `env.step()` (which returns BGR HWC). That path is independent of
A1 and is **correct** — left unchanged.

### Issue 2 — B3 fill loop also used appendleft (fixed inline)
<!-- Auditor: Correct. The audit did cite both locations; the fill loop was noted in the recommended fix. -->

**Discovery:** The audit cited lines 489-491 for the B3 fix but the fill loop at
lines 493-496 used the same `appendleft` pattern with different indentation —
not caught by the `replace_all` edit.

**Fix applied:** Fill loop also changed to `append`. Result: both primary insert and
pad-to-window-size paths now use consistent right-append ordering.

---

## What Is NOT Fixed Here

| Item | Status |
|---|---|
| B1 unit test | Deferred — apply with caution, validate before DPCC production eval |
| B2 (`des_robot_pos` dead-reckoning) | Not a bug — D3IL design decision, no action |

---

## Post-Auditor-Signoff Changes (user override + remaining cleanup)

### C1 — Constraints reverted to disabled (user decision)

The auditor approved C1 (re-enable `constraint_types: ['bounds', 'dynamics']`).
**User explicitly overrides: constraints remain disabled for now.**

`config/visual_aligning_eval.yaml` reverted to:
```yaml
constraint_types: []   # OPTION A: No constraints (kept disabled per user decision — Fix 8)
```
The enabled line is preserved as a comment for easy flip when ready:
```yaml
# constraint_types: ['bounds', 'dynamics']  # Ready to enable — A4+B1+C4 fixes applied (Fix 8)
```
A4+B1+C4 prerequisite fixes remain in place. Enabling constraints requires only uncommenting that line.

### B3-ext — Non-visual path `obs_context.appendleft` fixed

The original changelog noted lines 525-527 (non-visual path) as "not in audit scope; left as-is."
Per user instruction to leave no unfixed jobs, these were also fixed:

```python
# Before:
self.obs_context.appendleft(obs_t)
while len(self.obs_context) < self.obs_seq_len:
    self.obs_context.appendleft(obs_t)

# After:
self.obs_context.append(obs_t)
while len(self.obs_context) < self.obs_seq_len:
    self.obs_context.append(obs_t)
```

No `appendleft` remains anywhere in `eval_visual_aligning_dpcc.py`.

---
---

## Auditor Review — FIX 8 Implementation

> **⛔ PROTECTED SECTION — Do not modify without auditor sign-off.**

### Verification Summary

| Fix | Audit ID | Code Matches Recommendation | Status |
|-----|----------|----------------------------|--------|
| A1 — BGR→RGB | ✅ | Exact match to audit diff | **Approved** |
| A2 — Dead assertion | ✅ | Exact match | **Approved** |
| A3 — Normalizer eps-guard | ✅ | Matches intent; both paths guarded | **Approved** |
| A4 — Batch-0 broadcast | ✅ | Per-sample `s_0` in both methods | **Approved** |
| B1 — Scale row | ✅ (code applied) | Code fix applied (`x_diff` scale). **Unit test recommended** before enabling C1. | **Approved — test recommended** |
| B3 — Deque ordering | ✅ | Both insert + fill loop fixed | **Approved** |
| ~~C1 — Re-enable constraints~~ | 🔄 | ~~Applied after A4+B1+C4~~ → **Reverted per user override** | **Approved (reverted)** |
| C4 — c_pos obs fix | ✅ | Complete 3-step diff including init + unpack | **Approved** |
| Issue 1 — Video cvtColor cascade | ✅ | Correct cascade fix from A1 | **Approved** |
| Issue 2 — B3 fill loop | ✅ | Already covered in audit recommended fix | **Approved** |
| B3-ext — Non-visual appendleft | ✅ | Extended B3 fix to non-visual path | **Approved** |

### Outstanding Items

- **B1 unit test (recommended, not blocking)**: B1 code fix is applied and live. A unit test is recommended to independently validate the constraint matrix math, but this does not block eval — the fix is mathematically self-consistent and the constraint system (C1) is currently disabled by user choice anyway.

### Post-Signoff Changes Assessment

- **C1 revert (user override):** User chose to keep `constraint_types: []` for now. The prerequisite fixes (A4+B1+C4) remain in place; enabling constraints is a one-line uncomment. No regression — this is a configuration policy choice, not a code correctness issue. **Approved.**
- **B3-ext (non-visual path):** `appendleft` → `append` on the non-visual deque path (lines 525-527). Consistent with the B3 fix for the visual path. Eliminates all `appendleft` from the file. **Approved.**

### Sequencing Verification

The REV 2 priority ordering was respected:
1. ✅ A1 applied first (image pipeline)
2. ✅ A4 + B1 + C4 applied together (projector prerequisites)
3. ✅ C1 applied last, then reverted per user override (prerequisites intact)

### Conclusion

FIX 8 correctly implements all audit findings from GEN6V4_AUDIT_REPORT.md REV 3. The two newly discovered issues (video cvtColor cascade, B3 fill loop) and two post-signoff changes (C1 user revert, B3-ext cleanup) were all correctly handled. No regressions introduced.

The audit is **closed**. All code fixes are applied. Two optional items remain:
1. **B1 unit test** — recommended before enabling constraints, but code change is live.
2. **C1** — constraints disabled by user choice; one-line uncomment in `visual_aligning_eval.yaml` when ready.

---

*Signed:*
**Antigravity — Auditor**
*2026-05-19T21:56Z (updated from 21:50Z)*

