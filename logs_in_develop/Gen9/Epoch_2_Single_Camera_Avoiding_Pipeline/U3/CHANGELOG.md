# Gen9 E2 U3 — Changelog

**Date:** 2026-06-11
**Branch:** update_into_FM
**Implements:** `Gen9E2U2F4_Problem&Solution_Fable.md` S5 + Solution Step 5
**Scope:** Visual-DPCC avoiding pipeline — fix `clip_denoised` live trap; add pkl/config mismatch warning

---

## C1 — Restore `clip_denoised=True` in `plan_visual_avoiding_dpcc` config block (S5 + Solution Step 1)

**File:** `config/avoiding-d3il-visual.py`

### Root cause

`clip_denoised=False` in the `plan_visual_avoiding_dpcc` block was reverted to `False` by commit 9797209
("misc trivial NOT important reverts") after Fix_3 had set it to `True` (ca7b232).  The revert
re-froze the trap for all future retrains.

### Why this matters

`VisualGaussianDiffusion.p_mean_variance` has no `else` branch — when `clip_denoised=False` it
silently skips clamping.  Unclamped `x_recon` errors compound across K denoising steps → exploded
chaotic trajectories at eval, while training loss stays healthy (single-step epsilon MSE, never runs
the chain).

### The `.py` config caveat (pkl wins)

This `.py` change does **NOT** fix in-flight checkpoints.  The eval script reads `clip_denoised`
exclusively from the frozen `diffusion_config.pkl` saved at training time — `args.clip_denoised`
from the `.py` config is never injected into the diffusion constructor.

**For existing failing checkpoints: run on cluster before re-eval:**
```bash
python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/
```

This `.py` change only takes effect for **future retrains** — the new training run will freeze
`True` into the new pkl.

### Change

```diff
-        'clip_denoised':    False,
+        # U3-C1: must be True — False lets VisualGaussianDiffusion run unclamped, compounding
+        # x_recon errors across K denoising steps → exploded trajectories at eval.
+        # NOTE: this .py value is NEVER read at eval — the frozen diffusion_config.pkl wins.
+        # For in-flight checkpoints run: python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/
+        'clip_denoised':    True,
```

---

## C2 — Add `_warn_pkl_config_mismatch()` to surface pkl/config divergence at every eval run (Solution Step 5)

**File:** `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py`

### Problem

The eval script reads `clip_denoised`, `n_timesteps` / `n_diffusion_steps`, and `horizon`
exclusively from the frozen `diffusion_config.pkl`.  Changing those fields in the `.py` plan config
has zero effect at eval — the values are never injected.  This "silent precedence" caused Fix_3's
`clip_denoised=True` fix to be a no-op at eval, undetected until U3.

### Change

Added `_warn_pkl_config_mismatch(diffusion, args)` helper (before `load_diffusion_with_override`):

```python
def _warn_pkl_config_mismatch(diffusion, args):
    checks = [
        ('clip_denoised',     diffusion.clip_denoised,  args.clip_denoised),
        ('n_diffusion_steps', diffusion.n_timesteps,    args.n_diffusion_steps),
        ('horizon',           diffusion.horizon,        args.horizon),
    ]
    # Always print pkl values — silent precedence is the entire problem
    print('\n[ eval pkl values ] (these win over the .py plan config)')
    for key, pkl_v, cfg_v in checks:
        mismatch = (pkl_v is not None and cfg_v is not None and pkl_v != cfg_v)
        tag = '  *** MISMATCH — patch pkl or retrain ***' if mismatch else ''
        print(f'    {key}: {pkl_v!r}  (config: {cfg_v!r}){tag}')
    # Also emit warnings.warn for any mismatch so it appears in log files
    for key, pkl_v, cfg_v in checks:
        if pkl_v is not None and cfg_v is not None and pkl_v != cfg_v:
            warnings.warn(f'[ pkl/config mismatch ] {key}: pkl={pkl_v!r}, config={cfg_v!r}. ...')
```

Called immediately after `diff_experiment = load_diffusion_with_override(...)`:
```python
diffusion = diff_experiment.diffusion
_warn_pkl_config_mismatch(diffusion, args)   # ← new
```

### Effect at runtime

Every eval run now prints a table like:
```
[ eval pkl values ] (these win over the .py plan config)
    clip_denoised: False  (config: True)  *** MISMATCH — patch pkl or retrain ***
    n_diffusion_steps: 20  (config: 100)
    horizon: 8  (config: 8)
```
A `warnings.warn` is also emitted for any mismatched field so the mismatch appears in Slurm log
files even when stdout is suppressed.

---

## Files changed

| File | Change |
|------|--------|
| `config/avoiding-d3il-visual.py` | C1: `clip_denoised` False → True in `plan_visual_avoiding_dpcc` block |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | C2: `_warn_pkl_config_mismatch()` helper + call after checkpoint load |

---

## What is NOT done (cluster-side tasks, pending)

The following require running on the Slurm cluster — they are NOT code changes:

1. **Read + patch existing checkpoint pkl** (S5 cheapest test):
   ```bash
   # On cluster, from repo root — inspect first:
   python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/ --dry-run
   # Then apply:
   python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/
   ```
   If the pkl had `clip_denoised=False`, re-run eval only (no retrain). If the explosion disappears,
   S5 is confirmed as root cause.

2. **Step 0** — pin down exact remote code version of failing runs (`git log -1` on cluster, match
   Slurm submit time against reflog), then diff against current local code before running S1–S4
   probes.

3. **S1–S4 one-batch probe** (if S5 doesn't fully resolve): instrument eval to compare image tensor
   stats, normalised obs cond, raw model output, and unnormalised action against training-batch
   equivalents. First divergence = the bug.

---

## Scope note

C1 + C2 together close the "silent precedence" trap permanently for future retrains and evals.
The actual diagnosis of whether `clip_denoised=False` IS the root cause requires the cluster-side
steps above (Step 1 pkl patch + re-eval).
