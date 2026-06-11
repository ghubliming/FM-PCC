# Gen8 Epoch_1 U2 — Changelog

**Date:** 2026-06-11  
**Branch:** update_into_FM  
**Implements:** `Fix_2/Gen8E1F2_Problem&Solution_Fable.md` Steps 1 + 4 + B3  
**Scope:** `imf_visual_aligning/` — frozen-t B1 fix, redundant sampler B3 cleanup

---

## C1 — Replace frozen `t=0.5` with true `t_i = i/N` in `p_sample_loop` (B1, primary fix)

**File:** `imf_visual_aligning/models/imf_diffusion.py`

### Root cause

`T_CONST_INFERENCE = 0.5` (Deviation B, introduced in Gen3v4u2 fix_3) passed an identical
constant t to the model at every Euler step. This was justified by the reference iMF DiT
ignoring t — but that model has no `t_embedder` by design. Our UNet has both `time_mlp(t)`
and `h_mlp(h)` additively combined (unet1d_temporal_cond.py:118-123, :249), and was trained
with the true t at every step. Freezing t converts the learned `u(x, t, h)` into a biased
`u(x, constant_bias + h_embed)` — every step is out-of-distribution → chaotic rollouts.

### Change

```diff
-        T_CONST_INFERENCE = 0.5
-        t_const = torch.full((batch_size,), T_CONST_INFERENCE, device=device, dtype=torch.float32)
-
         for i in range(total_steps):
             loop_idx = min(i, flow_steps - 1)
-            velocity = self._predict_velocity(x, cond, t_const, h=h_batch, returns=returns)
+            t_i = torch.full(
+                (batch_size,), loop_idx / max(flow_steps, 1),
+                device=device, dtype=torch.float32,
+            )
+            velocity = self._predict_velocity(x, cond, t_i, h=h_batch, returns=returns)
```

`t_i = loop_idx / flow_steps` is the position the sampler is currently AT — the only
value consistent with how the model was trained. For `repeat_last` steps, `loop_idx`
is clamped to `flow_steps - 1`, so `t_i ≈ 1.0` (near the data manifold) throughout
the extra steps, which is physically correct.

The old frozen-t comment block is replaced with the guardrail comment explaining why
this architecture must never use a constant t.

### Expected outcome

With the existing checkpoint, `p_sample_loop` now behaves as vanilla FM multi-step
Euler — not one-step iMF (that requires a retrain, Step 3 in the P&S), but the
rollouts should produce structured trajectories instead of chaotic lines. Validate
via A/B against the frozen-t checkpoint (same seed, same `flow_steps`).

---

## C2 — Delete `iMFEngine.sample()` (B3, dead inconsistent sampler)

**File:** `imf_visual_aligning/models/imf_engine.py`

### Root cause

`iMFEngine.sample()` (lines 96-147 before this patch) was an independent sampling loop
that disagreed with `p_sample_loop` on two axes simultaneously:
- Passed true `t_cur` per step (no B1 bug) — inconsistent with `p_sample_loop`'s frozen t
  (now fixed), but would be inconsistent in the opposite direction post-fix.
- Mixed aux velocity: `u_weight·u + 0.1·v_weight·aux` — `p_sample_loop` discards aux
  (Deviation A: auxiliary branch improves training stability but is not added at inference).

No callers found in the live eval path (verified by grep). Dead but dangerous: any debug
harness calling `model.model.sample()` would produce results that differ from eval on
both t and aux axes simultaneously, corrupting any A/B comparison.

### Change

Deleted the entire `sample()` method from `iMFEngine`. The `forward_train()` method
that immediately followed is unaffected.

---

## C3 — Delete `iMFTrajectoryModel.sample_trajectory()` and `.sample()` (B3, same issue)

**File:** `imf_visual_aligning/models/imf_trajectory_model.py`

### Root cause

`iMFTrajectoryModel.sample_trajectory()` (lines 92-124) and `iMFTrajectoryModel.sample()`
(lines 126-155) were a second independent sampling stack with the same defects as
`iMFEngine.sample()`: aux mixing in the velocity, no connection to `p_sample_loop`.
`sample()` was a thin wrapper over `sample_trajectory()` — both deleted together.

No callers in the live eval path. Dead code removed entirely.

---

## Files changed

| File | Change |
|------|--------|
| `imf_visual_aligning/models/imf_diffusion.py` | C1: frozen t → true `t_i = loop_idx / N`; guardrail comment |
| `imf_visual_aligning/models/imf_engine.py` | C2: deleted `sample()` (dead, inconsistent) |
| `imf_visual_aligning/models/imf_trajectory_model.py` | C3: deleted `sample_trajectory()` + `sample()` (dead, inconsistent) |

---

## Not done (requires retrain)

**Step 3 — endpoint swap in `p_losses`:** condition on `(x_r, r, h)` instead of `(x_t, t, h)`.
This is the correct data-at-1 transcription of reference iMF and is required before
few-step / one-step sampling is legitimate. Deferred until Step 1+2 A/B confirms
B1 was the primary root cause. Implemented in a future U3 training run.

## Validation plan

```bash
# A/B: frozen t vs true t, same seed, same flow_steps
# Expect (b) structured, (a) chaotic
python imf_visual_aligning/eval.py --num_steps 10 --seed 42   # fixed-t (rollback to check)
python imf_visual_aligning/eval.py --num_steps 10 --seed 42   # true-t (this patch)
```
