# Gen3v4 U3 — Changelog

**Date:** 2026-06-11  
**Branch:** update_into_FM  
**Implements:** `Gen8/Epoch_1/Fix_2/Gen8E1F2_Problem&Solution_Fable.md` Step 1 (Gen3v4 arm)  
**Scope:** `flow_matcher_v3_imeanflow/` — frozen-t B1 fix (same bug, same fix as Gen8 U2 C1)

---

## C1 — Replace frozen `t=0.5` with true `t_i = i/N` in `p_sample_loop` (B1)

**File:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py`

### Context

The Gen8 P&S (Fix_2) explicitly flags this as the same bug in both codebases:

> "Apply the same change to the Gen3v4 source (`flow_matcher_v3_imeanflow/models/imf_diffusion.py:187`) — same bug."

The Gen3v4 `p_sample_loop` was a literal copy of the Gen8 version, including Deviation B
(`T_CONST_INFERENCE = 0.5`). Both were derived from Gen3v4u2 fix_3's frozen-t patch.

### Root cause (identical to Gen8 U2 C1)

The model backbone (`unet1d_temporal_cond.py`) has `time_mlp(t)` and `h_mlp(h)` additively
combined and was trained with true t. Freezing t at inference produces out-of-distribution
(x, t) pairs at every step — chaotic rollouts regardless of the checkpoint quality.

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

Comment block updated with U3-B1 guardrail (identical rationale to Gen8 U2).

---

## Files changed

| File | Change |
|------|--------|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | C1: frozen t → true `t_i = loop_idx / N`; guardrail comment |

---

## C2 — Swap conditioning endpoint in `p_losses` from `x_t` to `x_r` (B2, final fix)

**File:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py`

Identical change to Gen8 U2 C4. Same root cause, same fix:

```diff
- x_t = apply_conditioning(x_t, ...)
+ x_r = apply_conditioning(x_r, ...)

- velocity_pred, aux_pred = self._predict_uv(x_t, cond, t, h=h, ...)
+ velocity_pred, aux_pred = self._predict_uv(x_r, cond, r, h=h, ...)
```

Requires fresh training run. Target `(x_t - x_r) / h` unchanged.

---

## Files changed (full U3 summary)

| File | Change |
|------|--------|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | C1: frozen t → true `t_i`; C2: `p_losses` endpoint swap `x_t,t` → `x_r,r` |

---

## Scope note

Gen3v4 does not have the `imf_engine.py` / `imf_trajectory_model.py` redundant-sampler
issue (B3) — those files are Gen8-specific. Only the two `imf_diffusion.py` fixes apply here.

## Validation plan

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/iMF/train_imf.sh
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/iMF/eval_imf.sh
```

If still chaotic after retrain → architectural/capacity conclusion, not a code bug.
