# Gen9 E2 U2 Fix_4 — Full Investigation Record

**Status: ⚠️ REVERTED — open question remains**  
**Date range:** 2026-06-08 to 2026-06-09  
**Symptom:** DDPM (Visual-DPCC) trajectory lines explode to extreme coordinates; FM lines are well-behaved  
**Prior analysis doc:** `../Fix_3/Diffu_FM_Comparison.md`  
**Original (now-superseded) changelog:** `CHANGELOG.md`

---

## Timeline

### Step 1 — Fix_3 diagnosis pointed to `clip_denoised=False`

`../Fix_3/Diffu_FM_Comparison.md` concluded:

> "With `clip_denoised=False`, `x_recon` is never clamped. DDPM's `p_sample` injects stochastic
> Gaussian noise at every one of the 100 denoising steps → extreme `x_recon` → extreme
> `q_posterior` mean → noise amplification → divergence."

Verdict was **code bug (config error)** — change `clip_denoised=False` → `True`.

---

### Step 2 — Fix_4 applied: changed config file

`config/avoiding-d3il-visual.py`, entry `plan_visual_avoiding_dpcc`:
```diff
-'clip_denoised': False,
+'clip_denoised': True,
```

---

### Step 3 — Re-eval showed identical output to Fix_3 (no change)

Eval results were byte-for-byte the same as before the config change. The fix had no effect.

---

### Step 4 — Root cause of no-effect: PKL is the source of truth, not the config file

Traced the eval loading chain:

```
eval_visual_avoiding_dpcc.py
  └─ load_diffusion_with_override(args.loadbase, ...)
       └─ utils.load_config(..., 'diffusion_config.pkl')   ← reads PKL, not .py
            └─ diffusion_config(model)
                 └─ Config.__call__ → self._class(model, **self._dict)
                                                  ↑ clip_denoised from PKL._dict
```

The `.py` config file is only used to build `args` (loadpath, epoch, etc.). It is **never
re-read at eval time**. `clip_denoised` is frozen in the PKL at training time.

**Relevant files:**
- `diffuser_visual_avoiding/utils/config.py:96` — `Config.__call__` passes `**self._dict`
- `fm_visual_avoiding/utils/serialization.py:34` — `load_config` is plain `pickle.load`
- `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:117` — `load_diffusion_with_override`

---

### Step 5 — Further root cause: training script hardcodes `clip_denoised=False`

`diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py:241`:
```python
clip_denoised=False,   # hardcoded — never reads args.clip_denoised
```

So even if the `.py` config is changed, retraining regenerates the PKL with `False`.
This was **intentional**, not a bug — see Step 7.

---

### Step 6 — Attempted fixes (all later reverted)

Three code changes were made during the investigation:

| File | Change | Status |
|------|--------|--------|
| `config/avoiding-d3il-visual.py` | `clip_denoised: False → True` | **REVERTED** |
| `train_visual_avoiding_dpcc.py:241` | `False → getattr(args, 'clip_denoised', False)` | **REVERTED** |
| `eval_visual_avoiding_dpcc.py` | Added `config_overrides` param to `load_diffusion_with_override` | **REVERTED** |

A patch script was also created: `diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py`  
(can be deleted — it was for the wrong fix).

---

### Step 7 — Found contradicting historical decision in MASTER_TEST_HISTORY

`logs_in_develop/MASTER_TEST_HISTORY.md` lines 997–999:

> **Problem**: Setting `clip_denoised=True` in training scripts caused the ±5 action clamping
> to trigger at every early denoising step. Combined with the cosine noise schedule, this
> amplified bounds mathematically and permanently corrupted the actions by pinning them to
> thresholds, leading to **100% rollout failures**.
>
> **Resolution**: Disabled denoising clipping by setting `clip_denoised=False` by default in
> training and **forced it to `False` in evaluation routines**.

Also `MASTER_TEST_HISTORY.md` line 1146–1152:
> Made `clip_denoised` config-driven in `aligning-d3il-visual.py`, preventing unwanted hard
> clamps inside early denoising chains unless explicitly required.

**Conclusion:** the hardcode `False` in `train_visual_avoiding_dpcc.py:241` was a deliberate
protection after observing 100% failures. Fix_4 reversed a known-good decision.

---

### Step 8 — All Fix_4 changes reverted

`config/avoiding-d3il-visual.py` and `train_visual_avoiding_dpcc.py` restored to `False`.
Eval script restored to original. Codebase matches pre-Fix_4 state.

---

## Contradiction Summary

| Source | clip_denoised | Outcome |
|--------|---------------|---------|
| MASTER_TEST_HISTORY (prior work) | `False` ✓ | smooth trajectories |
| MASTER_TEST_HISTORY (prior work) | `True` ✗ | 100% rollout failures |
| Fix_3 analysis (this session) | `False` ✗ | exploded DDPM lines |
| Fix_3 analysis (this session) | `True` ✓ | would bound trajectories |

These are directly contradictory. Two hypotheses for why:

**H1 — Context differs**: The prior `True`-caused-failure may have been during a different
architecture version where the action normalizer produced a different scale, making ±5 too
tight and pinning actions. Gen9 E2 U2 uses `VisualGaussianDiffusion` with `action_dim=2`
(2-D avoiding), which may have different normalization characteristics.

**H2 — Fix_3 diagnosis was wrong**: The exploded lines have a different root cause unrelated
to `clip_denoised`. The coincidence that FM (deterministic) is stable and DDPM (stochastic)
explodes is real, but the mechanism may not be the clamping guard.

---

## Open Question for Next Research

**Why do DDPM trajectory lines explode while FM is stable?**

Known facts:
- Both use identical `VisualUNet` and `VisualAgent.predict()` pipeline
- Both are loaded from checkpoints with `clip_denoised=False`
- FM deterministic ODE step: `return model_mean` (no noise)
- DDPM stochastic step: `return model_mean + noise * scale` (noise injected 100×)
- The explosion appears only with `plan_batch_size=4` (Fix_3 introduced this); with B=1 harder to see

**Candidate alternative root causes to investigate:**

1. **Model not converged / undertrained**: DDPM checkpoint may not have trained long enough.
   Check `losses.pkl` loss curve for DDPM vs FM.

2. **Wrong checkpoint loaded**: `diffusion_epoch='best'` — verify what "best" resolves to.
   `utils.get_latest_epoch` scans `state_*.pt` files; check which epoch is actually loaded.

3. **Action normalizer mismatch**: If `act_normalizer` was saved from a different data version,
   unnormalization produces wrong scale → exploded coordinates in plot space even if internal
   values are bounded.
   Check: `ckpt_dir/obs_normalizer.pkl` and `act_normalizer.pkl` — what are their min/max?

4. **`apply_conditioning` wipe bug**: `cond['visual']` is cleared and re-applied each
   iteration in `p_sample_loop`. If the visual conditioning is not re-applied correctly in
   the DDPM branch, the model runs unconditioned → random walk.

5. **`VisualGaussianDiffusion` vs `VisualFlowMatching` class mismatch**: Eval uses
   `target_class=args.diffusion` which resolves to `VisualGaussianDiffusion`. Verify the
   PKL's `_class` matches and the `Fix_1` class-swap logic in `load_diffusion_with_override`
   didn't silently drop a required parameter.

---

## Files Referenced

| File | Role |
|------|------|
| `logs_in_develop/MASTER_TEST_HISTORY.md:997-999` | Prior decision: `True` → 100% failure |
| `logs_in_develop/MASTER_TEST_HISTORY.md:1146-1152` | Config-driven approach rationale |
| `../Fix_3/Diffu_FM_Comparison.md` | Original (now-questioned) diagnosis |
| `config/avoiding-d3il-visual.py:192` | `plan_visual_avoiding_dpcc.clip_denoised` (now `False`) |
| `diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py:241` | Hardcoded `False` — intentional |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:117` | `load_diffusion_with_override` |
| `diffuser_visual_avoiding/utils/config.py:96` | `Config.__call__` — PKL `_dict` flows into model |
| `diffuser_visual_avoiding/models/visual_gaussian_diffusion.py:65` | `p_mean_variance` clamp guard |
| `diffuser_visual_avoiding/models/diffusion.py:155` | `p_sample` — noise injection point |
| `fm_visual_avoiding/models/diffusion.py:157` | FM `p_sample` — deterministic, no noise |
| `diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py` | Patch script (created but not needed) |
