# Gen9E2U2F4 — Technical Review

**Date:** 2026-06-11  
**Reviewer:** Antigravity (Claude Opus 4.6 Thinking)  
**Scope:** Full source-code audit of all claims and suspects in `Gen9E2U2F4_Problem&Solution_Fable.md`  
**Verdict:** The document's *diagnostic framework* is sound and well-prioritized. However, **five concrete issues** require reconsideration before execution.

---

## 1. Overall Assessment

The Fable correctly identifies the four likeliest root causes (S1–S4) for "perfect train loss, zero eval structure" and ranks them in the right order. The proposed solution (Step 0 → instrument one-batch probe → fix divergence) is the correct methodology.

**However**, the code that exists in the *current workspace* is **not just clean** — it contains **active bugs and architectural hazards** that the Fable does not mention, because the Fable was written before this code state existed. These must be addressed regardless of whether Step 0 recovers a different remote version.

---

## 2. Issues Requiring Reconsideration

### Issue A — CRITICAL: Base `GaussianDiffusion.p_mean_variance` crashes when `clip_denoised=False`

**File:** [diffusion.py](file:///workspaces/FM-PCC/diffuser_visual_avoiding/models/diffusion.py#L137-L140)

```python
# diffusion.py lines 137–140
if self.clip_denoised:
    x_recon.clamp_(-1., 1.)
else:
    raise RuntimeError("clip_denoised=False not supported in base GaussianDiffusion")
```

The training config sets `clip_denoised=False` ([avoiding-d3il-visual.py L241](file:///workspaces/FM-PCC/config/avoiding-d3il-visual.py#L241)), and `VisualGaussianDiffusion` overrides `p_mean_variance` (correctly — it handles `clip_denoised=False` with selective action-only clamping). So training works fine.

**But:** The `fix_pkl_clip_denoised.py` utility patches the pkl to set `clip_denoised=True`. If someone runs this patch tool (which the repo ships for exactly this purpose), the **base class** path executes and clamps the *entire* trajectory to `[-1, 1]` — including the observation channels. The `VisualGaussianDiffusion` override only clamps `[..., :action_dim]`. These two behaviors are mutually exclusive.

> [!WARNING]
> **Reconsideration:** The Fable's S4 says "Verify `clip_denoised` ended up False at runtime." This is correct, but the Fable does not flag the **booby trap** in the codebase: if `fix_pkl_clip_denoised.py` is ever run, it routes eval through the base class's raise-on-False path or the base class's full-tensor clamp, depending on the pkl value. Either way, it corrupts the output. The fix tool should either:
> - Set `clip_denoised=False` (matching `VisualGaussianDiffusion`'s design), or
> - Not exist, to avoid confusion.

### Issue B — MODERATE: Eval script imports `VisualFlowMatching` instead of `VisualGaussianDiffusion`

**File:** [eval_visual_avoiding_dpcc.py L24](file:///workspaces/FM-PCC/diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py#L24)

```python
from fm_visual_avoiding.models.visual_gaussian_diffusion import VisualFlowMatching
```

The eval script for **DPCC** (diffusion-based) imports `VisualFlowMatching` from the **FM** package. It uses this class only in the `isinstance` check at line 218:

```python
if diffusion.__class__.__name__ in ('GaussianDiffusion', 'VisualGaussianDiffusion', 'VisualFlowMatching'):
```

This import is **harmless in practice** (the class name check uses strings, and the actual class loaded from pkl is `VisualGaussianDiffusion`), but it creates a **hard dependency** on the `fm_visual_avoiding` package being importable. If that package is missing or has import errors, the DPCC eval crashes at import time, not at usage time.

> [!NOTE]
> **Reconsideration:** The Fable's Swap A comment (line 22) says `fm_visual_avoiding.utils instead of diffuser.utils` — but the actual import on line 22 is `import fm_visual_avoiding.utils as utils`. **This means the DPCC eval script loads utilities from the FM package, not from the DPCC package.** The Fable says "Swap A — package" but doesn't flag whether `fm_visual_avoiding.utils` and `diffuser_visual_avoiding.utils` are identical or divergent. If they diverge (e.g. `Config.import_class` prepends different repo names), this is a live S3 vector.

### Issue C — MODERATE: `to_device` cannot handle tuple values in conditions dict

**File:** [arrays.py L77-L82](file:///workspaces/FM-PCC/diffuser_visual_avoiding/utils/arrays.py#L77-L82)

```python
def batch_to_device(batch, device='cuda:0'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)
```

And `to_device`:

```python
def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        raise RuntimeError(f'Unrecognized type in `to_device`: {type(x)}')
```

During training, `conditions` is `{0: obs_norm[0], 'primary_img': tensor}`. The `to_device` call recurses into the dict. `obs_norm[0]` is a **numpy array** (from `ParityAvoidingDataset.__getitem__`), not a tensor. `to_device` will hit the `raise RuntimeError` branch.

**However**, PyTorch's `DataLoader` automatically converts numpy arrays to tensors via the default collate function, so by the time `batch_to_device` sees the data, numpy arrays have already been converted. This is **safe in practice**, but only because of an implicit reliance on DataLoader's collate behavior.

> [!NOTE]
> This is not a bug, but the Fable's S1 analysis should note that the **DataLoader collate path** is the only reason training doesn't crash at `to_device`. If anyone calls `batch_to_device` on a raw `ParityAvoidingDataset` item (e.g., in a one-batch probe), it will crash.

### Issue D — MINOR: `VisualAgent.predict` uses `obs_seq[:, -1]` for snap anchor, but `obs_b` shape is `(B, 1, 4)`

**File:** [eval_visual_avoiding_dpcc.py L61-L87](file:///workspaces/FM-PCC/diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py#L61-L87) and [visual_gaussian_diffusion.py L94-L103](file:///workspaces/FM-PCC/diffuser_visual_avoiding/models/visual_gaussian_diffusion.py#L94-L103)

In `VisualAgent.predict`:
```python
obs_b = obs_t.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)  # (B, 1, 4)
cond = {0: (bp_b, obs_b)}
```

In `VisualGaussianDiffusion.forward`:
```python
obs_seq = payload[-1]      # last element is always obs → (B, 1, 4)
snap_obs = obs_seq[:, -1]  # (B, 4) — correct, since dim-1 has size 1
```

This is **correct** — `obs_seq[:, -1]` extracts the last (and only) timestep, yielding `(B, 4)`. No issue here.

**But:** `snap_obs` is then stored as `new_cond[0] = snap_obs` of shape `(B, 4)`. In `apply_conditioning`:

```python
x[:, t, action_dim:] = val.clone()  # t=0, val shape (B, 4), x[:, 0, 2:] shape (B, 4)
```

This writes `(B, 4)` into `x[:, 0, 2:]` — trajectory dim is 6, action_dim is 2, so `x[:, 0, 2:]` is indeed `(B, 4)`. **Correct.**

> [!TIP]
> The Fable's S1 claims are **validated** for the current code. The image transform (BGR→RGB, CHW, /255) in eval matches the dataset's `_load_images` transform. The cond unpacking is correct. The `isinstance(t, str)` guard exists in `apply_conditioning`.

### Issue E — IMPORTANT: `load_state_dict` strictness is correct, but the Fable's S3 concern about pkl class-swap pruning is real

**File:** [eval_visual_avoiding_dpcc.py L117-L151](file:///workspaces/FM-PCC/diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py#L117-L151)

The `load_diffusion_with_override` function:
1. Loads `diffusion_config.pkl` (which carries the class used at training time)
2. If `target_class` differs, it swaps the class AND **prunes** `_dict` keys that don't match the new class's `__init__` signature
3. `trainer.load()` calls `load_state_dict` without `strict=False` — **default strict=True** ✓

The Fable correctly identifies this as a risk. In the current code, the pruning is done via `inspect.signature`, which is correct. But the risk is: if the pkl was saved with `VisualFlowMatching` (FM class) and the eval loads with `VisualGaussianDiffusion` (DDPM class), the `__init__` signatures differ (FM has no `n_timesteps`, `predict_epsilon`, etc.), causing key deletions that break instantiation.

> [!IMPORTANT]
> **This is not an issue for the intended use case** (DPCC pkl → DPCC eval, FM pkl → FM eval), but the code doesn't guard against cross-engine mistakes. The Fable should recommend adding a class-name sanity check at the top of `load_diffusion_with_override`.

---

## 3. Fable Claims Verified as Correct

| Claim | Status | Evidence |
|-------|--------|----------|
| S1: Image must be BGR→RGB, CHW, /255, float32 at eval | ✅ Correct | `eval_visual_avoiding_dpcc.py:396` matches `sequence.py:155` |
| S1: `apply_conditioning` has `isinstance(t, str)` guard | ✅ Correct | `helpers.py:160` |
| S1: cond unpacking `{0: (bp_imgs, obs_seq)}` → `{visual: ..., 0: snap_obs}` | ✅ Correct | `visual_gaussian_diffusion.py:94-103` |
| S2: Normalizer pkl loaded from same ckpt dir | ✅ Correct | `eval_visual_avoiding_dpcc.py:206-211` |
| S2: Delta-decode `next_pos_des = action + obs[:2]` | ✅ Correct | `eval_visual_avoiding_dpcc.py:405` matches `sequence.py:88` |
| S3: `load_state_dict` runs strict | ✅ Correct | `training.py:320` — no `strict=False` arg |
| S3: importlib workaround for cross-package classes | ✅ Correct | `eval_visual_avoiding_dpcc.py:131-133` |
| S4: `n_timesteps` from frozen pkl, not from .py config | ✅ Correct | `serialization.py:56` loads pkl; config .py is never read at eval |
| S4: Config sets `clip_denoised=False` | ✅ Correct | `avoiding-d3il-visual.py:241` |

---

## 4. Verdict on Executability

> [!CAUTION]
> **Do NOT skip Step 0.** The Fable's entire value depends on comparing against the remote code state. The current workspace is clean for the claims it makes, but if the remote version diverged in *any* of S1–S4, only the remote diff reveals the actual bug.

The Fable's proposed solution (Step 0 → probe → fix) is **correct and should be executed as written**, with these additions:

1. **Before Step 2**: Run `fix_pkl_clip_denoised.py` with `--dry-run` to verify whether the pkl already has `clip_denoised=False`. If it has `True`, **do not run the fix** — instead set it back to `False` (matching `VisualGaussianDiffusion`'s override design).

2. **During Step 2**: When building the one-batch probe, use `batch_to_device` through a DataLoader (not raw dataset `__getitem__`), or manually convert numpy arrays to tensors first.

3. **Step 4 addition**: Add a class-name match assertion at the top of `load_diffusion_with_override` to prevent silent cross-engine loading.

4. **Regarding Issue B**: Resolve whether `eval_visual_avoiding_dpcc.py` should import from `diffuser_visual_avoiding.utils` (its own package) or `fm_visual_avoiding.utils`. If they are meant to be identical, document this. If not, switch the import.

---

*Signed: Antigravity — 2026-06-11T09:38Z*
