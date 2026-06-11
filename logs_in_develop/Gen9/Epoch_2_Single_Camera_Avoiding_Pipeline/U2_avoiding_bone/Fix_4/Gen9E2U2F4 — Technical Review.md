# Gen9E2U2F4 — Technical Review

**Date:** 2026-06-11 (rev.2 @ 10:28Z — corrected clip_denoised analysis)  
**Reviewer:** Antigravity (Claude Opus 4.6 Thinking)  
**Scope:** Full source-code audit of all claims and suspects in `Gen9E2U2F4_Problem&Solution_Fable.md`  
**Verdict:** The Fable's diagnostic framework is **correct**. No code fixes to block; proceed to execute.

---

## 1. Overall Assessment

The Fable correctly identifies the four likeliest root causes (S1–S4) for "perfect train loss, zero eval structure" and ranks them in the right order. The proposed solution (Step 0 → instrument one-batch probe → fix divergence) is the correct methodology.

The current workspace code is **clean** against every claim the Fable makes. The remaining unknowns are all gated behind **Step 0** (recover the remote code state).

---

## 2. clip_denoised — Correction from Rev.1

### User's Question

> "I thought the clip_denoised is finally overridden by the current .py setting, which means it is always False if we set, no matter what is loaded in the pkl?"

### Answer: You Are Wrong About the Mechanism, But Right That It Doesn't Matter

**Two separate facts:**

#### Fact 1 — The .py config does NOT override the pkl

The eval loading path is:

```
eval_visual_avoiding_dpcc.py L193:
    args = Parser().parse_args(experiment='plan_visual_avoiding_dpcc', ...)
    → args.clip_denoised = False  (from avoiding-d3il-visual.py L192)

eval_visual_avoiding_dpcc.py L201-203:
    load_diffusion_with_override(... target_class=args.diffusion ...)

load_diffusion_with_override L123:
    diffusion_config = utils.load_config(*loadpath, 'diffusion_config.pkl')
    → diffusion_config._dict['clip_denoised'] = whatever_was_saved_at_training_time

L143:
    diffusion = diffusion_config(model)
    → GaussianDiffusion.__init__(..., clip_denoised=<pkl value>, ...)
```

`args.clip_denoised` from the `.py` config sits in `args` and is **never injected** into the diffusion constructor. The pkl's value wins. There is no override mechanism.

#### Fact 2 — It doesn't matter, because `VisualGaussianDiffusion` overrides `p_mean_variance`

The base class `GaussianDiffusion.p_mean_variance` ([diffusion.py L137-L140](file:///workspaces/FM-PCC/diffuser_visual_avoiding/models/diffusion.py#L137-L140)) has:

```python
if self.clip_denoised:
    x_recon.clamp_(-1., 1.)
else:
    raise RuntimeError("clip_denoised=False not supported in base GaussianDiffusion")
```

But `VisualGaussianDiffusion` **completely overrides** this method ([visual_gaussian_diffusion.py L54-L78](file:///workspaces/FM-PCC/diffuser_visual_avoiding/models/visual_gaussian_diffusion.py#L54-L78)):

```python
def p_mean_variance(self, x, cond, t, ...):
    ...
    if self.clip_denoised:
        x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)
    # ← no else branch, no crash
    ...
```

The base class's `raise RuntimeError` is **dead code** for any `VisualGaussianDiffusion` instance. It is never reached, regardless of whether `clip_denoised` is `True` or `False`.

| pkl value | What happens | Result |
|-----------|-------------|--------|
| `False` | Override's `if` block is skipped | No clamping, no crash. ✅ |
| `True` | Override clamps actions only to [-5, 5] | Selective clamp, fine. ✅ |

#### Conclusion

My Rev.1 Issue A ("CRITICAL booby trap") was **wrong**. The `fix_pkl_clip_denoised.py` tool is harmless for `VisualGaussianDiffusion` — it just toggles between "no clamping" and "selective action clamping". Neither path crashes.

> [!NOTE]
> **Status: Issue A withdrawn.** `clip_denoised` is not a concern for this investigation. Keep the status quo — don't waste time on it. Focus on Step 0 → S1–S4 as the Fable directs.

---

## 3. Remaining Issues (downgraded from Rev.1)

### Issue B — LOW: Eval script imports from `fm_visual_avoiding` package

**File:** [eval_visual_avoiding_dpcc.py L22-L24](file:///workspaces/FM-PCC/diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py#L22-L24)

```python
import fm_visual_avoiding.utils as utils                                    # L22
from fm_visual_avoiding.sampling.projection import Projector                # L23
from fm_visual_avoiding.models.visual_gaussian_diffusion import VisualFlowMatching  # L24
```

The DPCC eval uses utilities from the **FM** package (`fm_visual_avoiding`), not its own package (`diffuser_visual_avoiding`). I verified that both packages' `utils/config.py` have **identical logic** — the only difference is the `repo_name` derived from `__name__`, but since `load_diffusion_with_override` bypasses `import_class` entirely (it uses raw importlib at L131-133), this is harmless.

The `VisualFlowMatching` import at L24 is used only for a string-based class name check at L218, so it's cosmetic.

> [!NOTE]
> **Not a bug.** The FM and DPCC utils are functionally identical. The cross-import creates a spurious dependency but won't cause wrong behavior. Low priority — cosmetic cleanup only.

### Issue C — TIP: `to_device` and the one-batch probe

**File:** [arrays.py L27-L33](file:///workspaces/FM-PCC/diffuser_visual_avoiding/utils/arrays.py#L27-L33)

`to_device` raises on anything that isn't a tensor or dict. PyTorch's DataLoader auto-converts numpy arrays to tensors, so training is safe. But if the Fable's Step 2 one-batch probe fetches data directly from `ParityAvoidingDataset.__getitem__` (skipping DataLoader), the numpy `conditions[0]` will crash `to_device`.

> [!TIP]
> **When building the one-batch probe (Step 2):** Use a DataLoader to fetch the batch, not raw `dataset[idx]`. Or manually wrap numpy arrays in `torch.from_numpy()` first.

### Issue D — Verified correct, no action needed

Shape analysis of `VisualAgent.predict` → `VisualGaussianDiffusion.forward` → `apply_conditioning` chain: all shapes match. No issue.

### Issue E — LOW: No guard against cross-engine pkl loading

`load_diffusion_with_override` will silently prune constructor args if you accidentally point a DPCC eval at an FM checkpoint (or vice versa). Not an issue for the intended workflow, but a defensive assertion would prevent future confusion.

> [!NOTE]
> **Not blocking.** Optional hardening — add if convenient during Step 4.

---

## 4. Fable Claims Verified as Correct

| Claim | Status | Evidence |
|-------|--------|----------|
| S1: Image must be BGR→RGB, CHW, /255, float32 at eval | ✅ Correct | `eval_visual_avoiding_dpcc.py:396` matches `sequence.py:155` |
| S1: `apply_conditioning` has `isinstance(t, str)` guard | ✅ Correct | `helpers.py:160` |
| S1: cond unpacking `{0: (bp_imgs, obs_seq)}` → `{visual: ..., 0: snap_obs}` | ✅ Correct | `visual_gaussian_diffusion.py:94-103` |
| S2: Normalizer pkl loaded from same ckpt dir | ✅ Correct | `eval_visual_avoiding_dpcc.py:206-211` |
| S2: Delta-decode `next_pos_des = action + obs[:2]` | ✅ Correct | `eval_visual_avoiding_dpcc.py:405` matches `sequence.py:88` |
| S3: `load_state_dict` runs strict (default=True) | ✅ Correct | `training.py:320` — no `strict=False` arg |
| S3: importlib workaround for cross-package classes | ✅ Correct | `eval_visual_avoiding_dpcc.py:131-133` |
| S4: `n_timesteps` from frozen pkl, not from .py config | ✅ Correct | `serialization.py:56` loads pkl; .py config never read at eval |
| S4: Config sets `clip_denoised=False` | ✅ Correct | `avoiding-d3il-visual.py:241` |
| S4: `.py` config does NOT override pkl at eval | ✅ Verified | `load_diffusion_with_override` reads pkl only; `args.clip_denoised` unused |

---

## 5. Verdict on Executability

> [!IMPORTANT]
> **The Fable is correct. Execute it as written.** No code fixes required before Step 0.

The only practical addition for Step 2 (one-batch probe): fetch via DataLoader, not raw `dataset[idx]`.

Everything else in the Fable — the ranking S1 > S2 > S3 > S4, the Step 0 prerequisite, the probe methodology, the "fix the first divergence" strategy — is validated against the current codebase.

---

*Signed: Antigravity — 2026-06-11T10:28Z (Rev.2 — corrected clip_denoised analysis per user challenge)*
