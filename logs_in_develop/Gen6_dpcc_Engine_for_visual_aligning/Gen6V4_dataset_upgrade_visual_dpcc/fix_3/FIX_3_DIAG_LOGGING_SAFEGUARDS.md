# Fix 3 — Diagnostic Logging & Safeguards for Train-Eval Gap (2026-05-19)

## Motivation

Training converged (wandb `final_test_loss=0.048`, `test/a0_loss=0.0083`) but eval GIFs showed
ridiculous behavior. Five failure modes can produce this pattern:

| # | Failure | Symptom in GIF |
|---|---------|----------------|
| 1 | Denormalization missing / silent RAW mode | Actions stuck near zero or wildly scaled |
| 2 | Normalizer scaler corrupted (zero-pad frames) | Wrong action scale in all directions |
| 3 | `n_diffusion_steps` mismatch train vs eval | Plausible start, chaotic divergence mid-rollout |
| 4 | Visual encoder producing random embeddings | No directional coherence across rollouts |
| 5 | Obs anchor scale mismatch (raw vs normalized) | Trajectory starts from wrong state |

None of these are detectable from training loss alone — the loss is computed in normalized space.

---

## Changes

### 1. `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

#### A. Crash on missing normalizers (was: silent RAW mode)

**Before** (`~line 659`):
```python
obs_normalizer, act_normalizer = None, None
if os.path.exists(obs_norm_path) and os.path.exists(act_norm_path):
    ...load...
else:
    print(f'[ eval ] WARNING: normalizer pkl not found — RAW mode')
```

**After**:
```python
if not os.path.exists(obs_norm_path) or not os.path.exists(act_norm_path):
    raise FileNotFoundError(
        '[ eval ] FATAL: normalizer pkl missing ...'
        'Without them, sampled actions stay in [-1,1] and produce wrong robot commands.'
    )
...load normalizers...
```

**Why**: Silent RAW mode means actions are sent to the robot in normalized `[-1, 1]` space.
The robot receives millimeter-scale deltas (or huge ones) and produces exactly the "ridiculous"
GIF pattern. Crashing loudly prevents this from ever being silently ignored.

#### B. Normalizer statistics log + zero-range warning

After loading, prints mins/maxs for both obs and act normalizers so the eval log can be
cross-checked against training logs:

```
[ eval ] obs_normalizer  mins=[...]  maxs=[...]
[ eval ] act_normalizer  mins=[...]  maxs=[...]
```

Also warns if any action dim has `maxs - mins < 1e-4`:
```
[ eval ] WARNING: act_normalizer near-zero range in dims [0] — possible zero-pad scaler corruption
```

Zero-range means the scaler's `normalize()` divides by ≈0, producing NaN or ±inf actions
(Fix 1 lesson: zero-padded episodes inflate `max_len_data` and bias the scaler).

#### C. `n_diffusion_steps` mismatch warning after model load

After `diffusion_model = exp.diffusion`:
```
[ eval ] Model n_timesteps = 100  (config n_diffusion_steps = 100)
```

If the checkpoint's baked-in `n_timesteps` differs from the current config:
```
[ eval ] WARNING: n_timesteps mismatch — checkpoint trained with 32 steps, config says 100.
```

**Why**: If a checkpoint trained with K=32 is run with K=100 denoising steps, `p_sample_loop`
iterates over timesteps the model has never seen. The denoising chain produces garbage starting
from about step 33 onward, while the first steps look plausible — matching the observed GIF pattern.

#### D. First-replan action magnitude diagnostic in `predict()`

On the very first replan of the first rollout, logs:
```
[ DIAG first-replan ] normalized   a0 = [0.0312  0.0451 -0.0221]  |mag| = 0.0590
[ DIAG first-replan ] denormalized a0 = [0.00031 0.00045 -0.00022]  |mag| = 0.000590 m
[ DIAG first-replan ] horizon act (normalized) range: [-0.3142, 0.4121]
```

**How to read**:
- If denormalized `|mag|` ≈ normalized `|mag|` → denormalization did nothing (normalizer was None
  or identity) — check that `act_normalizer.pkl` covers the real action range.
- If both magnitudes are ≈ 0.0 → model predicts near-zero actions (trivial solution or collapsed).
- If normalized range is `[-1, 1]` exactly → model is predicting at the clamp boundary (over-clamped).
- Healthy expected values: normalized `|mag| ∈ [0.02, 0.5]`, denormalized `|mag| ∈ [0.001, 0.05] m`.

---

### 2. `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py`

#### E. Log normalizer statistics at train time

After saving `obs_normalizer.pkl` / `act_normalizer.pkl`:
```
[ train ] obs_normalizer [ Normalizer ] dim: 6
    -: [0.21  0.18  0.82  ...]
    +: [0.54  0.49  0.92  ...]
[ train ] act_normalizer [ Normalizer ] dim: 3
    -: [-0.042 -0.038 -0.001]
    +: [ 0.041  0.039  0.001]
```

These should match the eval log's normalizer stats. Any divergence means the checkpoint loaded
by eval was saved from a different training run (path mismatch).

#### F. Log `n_diffusion_steps` before creating diffusion config

```
[ train ] n_diffusion_steps = 100  (must match eval config to avoid denoising-chain mismatch)
```

Extracts into `_n_diff_steps` so the value is visible in the log before the `utils.Config` call
potentially masks it.

---

### 3. `diffuser_visual_aligning/models/visual_unet.py`

#### G. Log visual encoder initialization

After `hydra.utils.instantiate(obs_encoder_cfg)`:
```
[ VisualUNet ] MultiImageObsEncoder initialized — LATENT_DIM=128, imagenet_norm=True, share_rgb_model=False
```

**Why**: If the visual encoder silently fails to instantiate (e.g., hydra config resolution error
that is swallowed), the model falls back to `visual_cond = None` in `forward()`, making the UNet
effectively unconditional. The GIF would show random-walk behavior with no visual grounding.
This log line confirms the encoder is live.

---

## Files Changed

| File | Change |
|------|--------|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | A: crash on missing normalizers; B: log stats + zero-range warn; C: n_steps mismatch warn; D: first-replan diagnostic |
| `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` | E: log normalizer stats; F: log n_diffusion_steps |
| `diffuser_visual_aligning/models/visual_unet.py` | G: log encoder init confirmation |

---

## Diagnosis Protocol (next eval run)

1. Check `[ eval ] act_normalizer  mins=...  maxs=...` — does it match training log?
2. Check `[ eval ] Model n_timesteps = X` — does X match the training `n_diffusion_steps`?
3. Check `[ VisualUNet ] MultiImageObsEncoder initialized` — is it present?
4. Check `[ DIAG first-replan ] denormalized a0` — is `|mag|` in `[0.001, 0.05] m` range?
5. If denormalized `|mag|` ≈ normalized `|mag|` → denormalization failed (check pkl path)
6. If normalized range is outside `[-1, 1]` → model diverged during denoising (n_steps issue)
