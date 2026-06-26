# Gen9 E2 U3 — Run, Debug & Results Guide

**Scope:** Visual-DPCC avoiding pipeline post-U3 changes  
**Goal:** Confirm whether `clip_denoised=False` frozen in the checkpoint pkl is the root cause of the exploded-lines eval failure, then interpret the outcome.

---

## Quick decision: which path to take first

```
Do you have an existing failing DPCC checkpoint on the cluster?
│
├─ YES ──► S5 Cheap Test (Section 1) — patch pkl, re-eval, zero retrain
│           Cost: ~5 min on cluster. Highest prior fix.
│
└─ NO  ──► Fresh Retrain (Section 2) — U3-C1 bakes True into the new pkl automatically.
            Cost: full train (~24 h) + eval (~4 h).
```

---

## Section 1 — S5: Patch existing checkpoint pkl and re-eval

**Do this before a full retrain.** The pkl patch is reversible (`.bak` backup created automatically).

### Step 1.1 — Inspect current pkl value (dry run, no writes)

```bash
# On cluster, from repo root
python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/ --dry-run
```

Expected output for a broken checkpoint:
```
  PATCH  logs/avoiding-d3il-visual/visual_avoiding_dpcc/H8_K20_.../5/diffusion_config.pkl
         clip_denoised: False → True
```
Expected output for an already-correct checkpoint:
```
  OK    logs/.../diffusion_config.pkl — already True, nothing to do
```

If **all** pkls already say `True` → skip to Section 2 (S5 not applicable, root cause is elsewhere).

### Step 1.2 — Apply the patch

```bash
python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/
```

A `.bak` file is created next to every patched pkl.  To roll back:
```bash
cp logs/.../diffusion_config.pkl.bak logs/.../diffusion_config.pkl
```

### Step 1.3 — Re-run eval only (no retrain)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh
```

Or for a single seed:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh 5
```

---

## Section 2 — Fresh retrain (or when no existing checkpoint)

U3-C1 already set `clip_denoised=True` in the `.py` config.  A new training run will freeze `True`
into the new `diffusion_config.pkl` — no manual pkl patching needed.

```bash
# Train (≈24 h)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh

# Eval (scheduled automatically after train via pipeline script, OR submit manually)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh
```

Or chain both at once:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/diffuser_visual_avoiding/visual_avoiding_pipeline_dpcc.sh
```

---

## Section 3 — Reading the new pkl/config banner at eval start

U3-C2 added `_warn_pkl_config_mismatch()`. Every eval run now prints a block like this before
any trajectory is generated:

```
[ eval pkl values ] (these win over the .py plan config)
    clip_denoised: False  (config: True)  *** MISMATCH — patch pkl or retrain ***
    n_diffusion_steps: 20  (config: 100)
    horizon: 8  (config: 8)
```

### Interpreting each line

| What you see | Meaning | Action |
|---|---|---|
| `clip_denoised: True  (config: True)` | Pkl correctly patched / new retrain. Clamping active. | None — proceed. |
| `clip_denoised: False  (config: True)  *** MISMATCH ***` | Old unpatched pkl. Fix_3 + U3-C1 config change was never applied to this checkpoint. | Run `fix_pkl_clip_denoised.py` (Section 1) or retrain. |
| `clip_denoised: False  (config: False)` | Both agree on False. No retrain has happened since U3-C1. | Run patch or retrain. |
| `n_diffusion_steps: 20  (config: 100)  *** MISMATCH ***` | Checkpoint was trained with K=20 (the `H8_K20_...` run dir). Eval uses 20 steps regardless of config. | This is a separate issue (secondary factor in P&S). Note it, don't block on it. |
| `horizon: 8  (config: 8)` | Matches — fine. | None. |

A `warnings.warn` is also emitted for any `*** MISMATCH ***` line, so it appears in the Slurm
`.log` file even if stdout is piped elsewhere.

### Finding the banner in a Slurm log

```bash
grep -A 6 "\[ eval pkl values \]" Slurm_Codes/logs/$(ls -t Slurm_Codes/logs/ | head -1)/latest.log
# or
grep "MISMATCH\|pkl values\|clip_denoised" Slurm_Codes/logs/*/latest.log
```

---

## Section 4 — Interpreting eval results

### 4.1 What success looks like (clip_denoised was root cause)

| Metric | Before (broken) | After (fixed) |
|---|---|---|
| Trajectory plot | Dozens of lines radiating to screen edges, no scene structure | Bounded curves that bend away from obstacles |
| Obstacle contact rate | ~100% or undefined | Should drop toward FM baseline |
| Action magnitudes | Unbounded (`>>5.0` after unnormalize) | Bounded near training-data range |
| Eval console | May print large `|action|` values | Action norms in reasonable range |

The fix eliminates the divergence mechanism — trajectories may still be noisy (DDPM adds stochastic
noise at each denoising step, inherent) but they will be **bounded and scene-aware**.

### 4.2 What partial improvement looks like

Trajectories are no longer exploded but still worse than FM:
- Bounded lines, but most collide with obstacles
- Low success rate (~10–30%) vs FM baseline

This is expected: clamping stops divergence but the model was trained with K=20 steps while the
current config expects K=100 (see `n_diffusion_steps` mismatch above). The checkpoint may also be
at step 99k (overfit) rather than step 11k (best val). Both are secondary factors from the P&S.

**Action:** re-eval the best-val checkpoint (`--epoch 11000` or the step with minimum `losses.pkl`
test loss), and/or retrain with K=100 baked in.

### 4.3 What failure looks like (clip_denoised was NOT the root cause)

Trajectories still show zero structure even after patching `clip_denoised`:
- Lines still radiate to edges, or all trajectories are near-zero deltas
- No obvious correlation with obstacle positions

This confirms the problem is one of S1–S4.  Proceed to Section 5.

---

## Section 5 — Debugging if still exploded after S5 (Step 0 → S1–S4)

### Step 0 — Pin the exact remote code version

```bash
# On cluster, from repo root
git log -1                    # commit that ran the failing jobs
git stash list                # any uncommitted edits stashed at submit time
git diff HEAD                 # any current uncommitted edits
```

Cross-reference the commit timestamp against the Slurm job submit time in the `.log` file header.
Diff `diffuser_visual_avoiding/`, `diffuser_visual_avoiding_test/`, and
`config/avoiding-d3il-visual.py` between that commit and current HEAD.

**Any divergence in those three trees is a prime suspect** — the local audit only proves the
*current* code is clean.

### S1 — Visual conditioning probe

Add a one-batch probe in `VisualAgent.predict()` (or before it in the eval loop) to print:

```python
print(f"[probe] bp_image dtype={bp_image.dtype} min={bp_image.min():.3f} max={bp_image.max():.3f}")
print(f"[probe] bp_image shape={bp_image.shape}")  # expect (H, W, C) or (C, H, W)
print(f"[probe] obs_4d={obs_4d}")
```

Expected training-time values (from `ParityAvoidingDataset`):
- dtype: `float32`, range `[0.0, 1.0]`, shape `(96, 96, 3)` before CHW conversion
- If you see `uint8` or range `[0, 255]` → image pre-processing bug (S1 confirmed)

**Note:** fetch the training-side batch using a `DataLoader`, not raw `dataset[idx]` — `to_device`
raises on numpy arrays that DataLoader would auto-convert.

### S2 — Normalizer probe

```python
# Before unnormalize:
print(f"[probe] act_norm min={act_norm.min():.3f} max={act_norm.max():.3f}")
# After unnormalize:
print(f"[probe] action={action}")  # should be small delta in metres, not >1.0
```

Also verify the normalizer files match the checkpoint run dir:
```bash
ls -la logs/avoiding-d3il-visual/visual_avoiding_dpcc/<run_dir>/<seed>/
# should contain: obs_normalizer.pkl  act_normalizer.pkl  diffusion_config.pkl
```

### S3 — Class-swap probe

In `load_diffusion_with_override`, add after line 143:
```python
print(f"[probe] loaded class: {type(diffusion).__module__}.{type(diffusion).__name__}")
print(f"[probe] vision encoder loaded: {hasattr(diffusion.model, 'visual_encoder')}")
```

Expected: `diffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion`  
If you see `VisualFlowMatching` or the wrong module → class-swap bug (S3 confirmed).

### S4 — Sampling-chain probe

Print `diffusion.n_timesteps` after load.  The `H8_K20_...` checkpoint directory name tells you
it was trained with K=20.  If `diffusion.n_timesteps=20` but you expected 100, that's a mismatch
(already surfaced by the new banner — Section 3).

---

## Section 6 — Decision tree summary

```
Run S5 pkl patch → re-eval
│
├─ Trajectories bounded and scene-aware?
│   └─ YES → clip_denoised confirmed as root cause. ✅
│             Optional: retrain with K=100 for full performance.
│
├─ Trajectories bounded but still high collision?
│   └─ Check banner: n_diffusion_steps mismatch? K=20 vs K=100?
│       └─ YES → re-eval best-val checkpoint, or retrain with matching K.
│       └─ NO  → secondary factor (overfit checkpoint). Try step-11k ckpt.
│
└─ Trajectories still exploded / zero structure?
    └─ clip_denoised was NOT the root cause.
        → Step 0: pin remote code version
        → S1 probe: check image pre-processing
        → S2 probe: check normalizer
        → S3 probe: check loaded class
        → S4 probe: already surfaced by banner
        → Fix first divergence found, re-eval only (no retrain needed).
```

---

## Reference — relevant file locations

| File | Purpose |
|---|---|
| `config/avoiding-d3il-visual.py` | Plan config (`.py` values NEVER read at eval — pkl wins) |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Eval entry point; contains `_warn_pkl_config_mismatch` |
| `diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py` | Pkl patcher tool |
| `diffuser_visual_avoiding/models/visual_gaussian_diffusion.py:65-66` | `p_mean_variance` override — clamping guard |
| `logs/.../diffusion_config.pkl` | Frozen checkpoint config (source of truth at eval) |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh` | Eval sbatch |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/train_visual_avoiding_dpcc.sh` | Train sbatch |
| `Slurm_Codes/sbatch/diffuser_visual_avoiding/visual_avoiding_pipeline_dpcc.sh` | Full train→eval chain |
