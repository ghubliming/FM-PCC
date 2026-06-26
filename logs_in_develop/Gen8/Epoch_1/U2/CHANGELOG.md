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

## C4 — Swap conditioning endpoint in `p_losses` from `x_t` to `x_r` (B2, final fix)

**File:** `imf_visual_aligning/models/imf_diffusion.py`

### Root cause

Training conditioned the model on `x_t` (data-biased interpolant, near t=1) at time `t`
(data-biased). At inference, the sampler presents `x` at `t_i = i/N` — the **noise-side**
current state. This train/inference mismatch means the model was never taught to predict
velocity from the noise-side inputs it receives at sampling time. B1 fix alone (true t)
corrected the time label but not the conditioning point, so rollouts remained chaotic.

### Change

```diff
- x_t = apply_conditioning(x_t, ...)    # old: condition model input on data-side point
+ x_r = apply_conditioning(x_r, ...)    # new: condition model input on noise-side point

- velocity_pred, aux_pred = self._predict_uv(x_t, cond, t, h=h, ...)
+ velocity_pred, aux_pred = self._predict_uv(x_r, cond, r, h=h, ...)
```

The mean-flow **target** `(x_t - x_r) / h = data - noise` is unchanged — it is
endpoint-independent for the linear interpolant. Only the model input and its
paired time shift from the data-side `(x_t, t)` to the noise-side `(x_r, r)`.

This requires a **fresh training run** — the existing checkpoint was trained with
the old `(x_t, t)` conditioning and cannot be patched without retraining.

### After this retrain

The sampler presents `(x@t_i, t_i)` and the model was trained on `(x_r, r)` pairs —
fully consistent. One-step and few-step iMF sampling become legitimate.

---

## Files changed (full U2 summary)

| File | Change |
|------|--------|
| `imf_visual_aligning/models/imf_diffusion.py` | C1: frozen t → true `t_i`; C4: `p_losses` endpoint swap `x_t,t` → `x_r,r` |
| `imf_visual_aligning/models/imf_engine.py` | C2: deleted dead `sample()` |
| `imf_visual_aligning/models/imf_trajectory_model.py` | C3: deleted dead `sample_trajectory()` + `sample()` |

---

## Validation plan

Retrain from scratch with these fixes, then eval:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/imf_visual_aligning/train_imf_visual_aligning.sh
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/imf_visual_aligning/eval_imf_visual_aligning.sh
```

If still chaotic after retrain → conclusion is architectural/capacity, not a code bug.
Gen8 closes; fall back to vanilla FM (Gen7 baseline).
