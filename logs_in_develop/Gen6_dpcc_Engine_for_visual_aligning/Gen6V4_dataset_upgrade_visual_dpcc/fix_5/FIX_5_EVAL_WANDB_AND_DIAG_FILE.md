# Fix 5 — Eval: wandb crash + DIAG output to file (2026-05-19)

## Context

First real Slurm eval run on a trained visual model (git rev 6fc83a7, seed 6,
H8/K16/steps256 checkpoint at step 4000).

All 3 rollout contexts completed successfully. Two bugs caused the run to crash at
the end and made post-run analysis harder:

---

## Bug 1 — wandb crash after all rollouts complete (CRASH)

`d3il/simulation/aligning_sim.py:212` calls `wandb.log(...)` unconditionally
after all rollout contexts finish:

```python
wandb.log({'score': 0.5 * (success_rate + entropy)})
```

With no `wandb.init()` called first, wandb raises:

```
wandb.errors.Error: You must call wandb.init() before wandb.log()
```

This crashes the eval process **after** rollouts but **before** the final 7-metric
report, NPZ save, and PNG rollout grid are written. The `p(m|c)` matrix is printed
(so we know the rollout results), but all saved artifacts are lost.

We cannot modify `d3il/` (copy-modify constraint), so the fix is in our eval wrapper.

**Fix:** Import wandb at the top of `eval_visual_aligning_dpcc.py` (graceful import
so missing wandb doesn't break headless runs) and call `wandb.init(mode='disabled')`
immediately before `sim.test_agent(agent)`:

```python
# top of file
try:
    import wandb as _wandb
except ImportError:
    _wandb = None

# before sim.test_agent(agent)
if _wandb is not None:
    _wandb.init(mode='disabled')
```

`mode='disabled'` routes all wandb calls to no-ops — no network traffic, no run
created, no side effects.

---

## Bug 2 — DIAG lines only in console / log, no dedicated output file

The `[ DIAG first-replan ]` lines from Fix 3 are captured by the `Tee` redirector
into `eval_{variant}.log`. However, for cross-run analysis (comparing DIAG values
across seeds or training checkpoints) it is awkward to parse them out of the full
log. A dedicated file is easier to `cat` / `grep`.

Additionally, the original DIAG only printed:
- `normalized a0` + magnitude
- `denormalized a0` + magnitude
- `range` of the full H-step normalized action trajectory

This hides the per-step pattern — whether actions oscillate (+5, -5, +5, ...) or
are stuck at one extreme — which is diagnostic for DDPM convergence issues.

**Fix:** Expand the DIAG block to:
1. Build a list of strings instead of printing inline
2. Add per-step breakdown: `step 0: [+2.7 -3.6 +1.2]`, etc.
3. Print all lines to stdout (same as before, captured by Tee into eval log)
4. Also write to `{save_path}/diag_first_replan.txt`

```
diag_first_replan.txt (example):
[ DIAG first-replan ] normalized   a0 = [-2.6467  2.7854 -3.6031]  |mag| = 5.2674
[ DIAG first-replan ] denormalized a0 = [-0.00833  0.00833 -0.00833]  |mag| = 0.014434 m
[ DIAG first-replan ] horizon act (normalized) range: [-5.0000, 5.0000]
[ DIAG first-replan ] per-step normalized acts (H=8):
  step  0: [-2.6467  2.7854 -3.6031]
  step  1: [ 5.      -5.      5.    ]
  ...
```

---

## What the DIAG output revealed about the Slurm run

The first real eval DIAG output was:

```
[ DIAG first-replan ] normalized   a0 = [-2.6467  2.7854 -3.6031]  |mag| = 5.2674
[ DIAG first-replan ] denormalized a0 = [-0.00833  0.00833 -0.00833]  |mag| = 0.014434 m
[ DIAG first-replan ] horizon act (normalized) range: [-5.0000, 5.0000]
```

Interpretation:
- Training data normalized actions live in `[-1, 1]`. Values of `[-2.6, +2.8, -3.6]`
  are 2-4× outside this range — the DDPM has not converged.
- `horizon range: [-5.0000, 5.0000]` — the ±5 action clamp in `p_mean_variance`
  (`clip_denoised=True`) fires on every step of the 8-step horizon simultaneously.
  This is characteristic of a model predicting pure noise that is then clamped.
- `LimitsNormalizer.unnormalize()` clips out-of-range values to `[min, max]`,
  so every step produces the exact same action = normalizer boundary
  `[-0.00833, +0.00833, -0.00833]`. The robot oscillates at max velocity → all fail.

**Root cause of failure**: this checkpoint was from a 5000-step smoke test run
(~6 passes over the 44 k-window dataset). The model needs the full `n_train_steps=5e5`
to converge. The DIAG is working correctly and correctly identified the issue.
The normalizers are correct (Fix 1 confirmed); no structural code bugs were found.

---

## Files Changed

| File | Change |
|------|--------|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | Bug 1: graceful wandb import + `wandb.init(mode='disabled')` before `sim.test_agent()`; Bug 2: DIAG per-step breakdown + write `diag_first_replan.txt` to save_path |
