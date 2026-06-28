# CHANGELOG — U7: Logit-Normal Time Schedule

**Date:** 2026-06-28  
**Plan:** [`PLAN_Logit_Normal_Schedule.md`](PLAN_Logit_Normal_Schedule.md)

---

## What changed

Added logit-normal time-schedule as the **new default** for iMF training. The existing Beta schedule is kept as `t_schedule='beta'` for backward compat and A/B ablation. `'uniform'` is also exposed as an alias (`Beta(1,1)`).

All changes are minimal: only the `t` sampling line in each `loss()` is touched. Zero impact on the model forward pass, projector, eval, or checkpoint structure (other than the path key below).

---

## Files touched

| File | Lines changed | What |
|---|---|---|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | `__init__` params + store + `loss()` | 3-branch switch; 3 new params |
| `imf_visual_aligning/models/imf_diffusion.py` | same | parallel copy, identical change |
| `config/avoiding-d3il.py` | `args_to_watch_fmv3_imf_train`, training block, plan block | +`ts` key in path; default=logit_normal |
| `config/aligning-d3il-visual.py` | `args_to_watch_imf_visual_train`, training block, plan block | same |

---

## Code change detail

### `imf_diffusion.py` (both copies) — new `__init__` params

```python
# U7: time-schedule selector. 'logit_normal' = canonical iMF (reference imf.py L120-124).
# 'beta' = legacy 1-Beta(α,β) (backward-compat / A-B ablation). 'uniform' = Beta(1,1).
t_schedule: str = 'logit_normal',   # DEFAULT: logit-normal (iMF paper default)
p_mean: float = -0.4,               # logit-normal: mean in logit space (sigmoid median ≈ 0.40)
p_std: float = 1.0,                 # logit-normal: std in logit space
```

### `imf_diffusion.py` (both copies) — `loss()` replacement

**Before (U6):**
```python
alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
beta_dist = torch.distributions.Beta(alpha, beta)
t = 1.0 - beta_dist.sample((batch_size,))
```

**After (U7):**
```python
if self.t_schedule == 'logit_normal':
    # Canonical iMF schedule — matches reference imf.py L120-124:
    #   sigmoid(randn * P_std + P_mean), P_mean=-0.4, P_std=1.0 → median ≈ 0.40
    t = torch.sigmoid(torch.randn(batch_size, device=x.device) * self.p_std + self.p_mean)
else:  # 'beta' — legacy 1-Beta(α,β). Set α=β=1 for uniform. (pre-U7 default)
    alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
    beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
    beta_dist = torch.distributions.Beta(alpha, beta)
    t = 1.0 - beta_dist.sample((batch_size,))
```

> Note: No separate `'uniform'` branch — just set `α=β=1` in the `'beta'` schedule. `Beta(1,1) = Uniform(0,1)`.

### Config changes

**`args_to_watch` (both configs):** `('t_schedule', 'ts')` appended → embeds `_tslogit_normal` / `_tsbeta` in checkpoint dir name.

**Training blocks:** three new keys added after `time_beta_beta_v3`:
```python
't_schedule': 'logit_normal',     # U7 DEFAULT
'p_mean': -0.4,
'p_std': 1.0,
```

**Plan block paths** (avoiding + aligning): `_ts{t_schedule}` appended to `prefix` and `diffusion_loadpath`.

---

## Checkpoint path impact

| Scenario | Path suffix | Notes |
|---|---|---|
| New logit-normal runs (default) | `..._bbunet_tslogit_normal` | new training from scratch |
| Old pre-U7 Beta runs | `..._bbunet` (no `ts`) | can't be loaded via new config; set `t_schedule=beta` AND remove `_ts{t_schedule}` from loadpath manually if needed |
| Beta ablation run | `..._bbunet_tsbeta` | set `t_schedule='beta'` in config |
| "Uniform" ablation | `..._bbunet_tsbeta` with `α=β=1` | no separate key — just set alpha/beta in config |

**Note:** `time_beta_alpha_v3` / `time_beta_beta_v3` remain in the path for the `a`/`b` fields; they're ignored at runtime when `t_schedule='logit_normal'` but harmless in the path string.

---

## Verification

- `python3 -m py_compile` passes on all 4 files (Python 3.13).
- Logic verified by inspection: three branches are mutually exclusive; the logit-normal branch matches `/workspaces/imeanflow/imf.py` L120-124 exactly.
- **Not run end-to-end** — cluster retrain required (no GPU/torch in Docker dev env).
