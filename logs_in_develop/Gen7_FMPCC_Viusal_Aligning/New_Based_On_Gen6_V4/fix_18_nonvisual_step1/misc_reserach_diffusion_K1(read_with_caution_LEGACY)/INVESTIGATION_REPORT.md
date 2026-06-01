# One-Shot Run Investigation — DPCC steps=1 train + FM ODE=1 eval

**Date:** 2026-05-30
**Inputs:** `temp/one_shot_run/visual_dpcc`, `temp/one_shot_run/visual_fm`
**Branch:** `update_into_FM` (git rev `4723e20`)

> **Status banner (added 2026-05-30 post-verification):**
> The user separately re-ran the **visual** versions of the same one-shot
> experiment (DPCC `n_diffusion_steps=1` train + FM `flow_steps_v3=1` eval) on
> the canonical `visual_aligning_dpcc` / `fm_visual_aligning` variants with
> `if_vision=True` (default). **Both ran without crashing.** This confirms
> that the two crashes diagnosed below (§1 and §2b) are **non-visual-path-only
> bugs**, triggered exclusively when one CLI-overrides `if_vision=False` on a
> variant whose other fields (`obs_dim`, model spec, checkpoint normalizers)
> were defined for the visual path. The visual path itself is internally
> consistent — `obs_dim=6`, `if_vision=True`, dataset `ParityAligningDataset`
> (9-D), `_traj_dim=9` all line up by construction; there is no plumbing to
> mismatch.

**TL;DR — what actually happened:**

| Run | What you wanted | What actually ran | Result |
|---|---|---|---|
| `visual_dpcc` | Train Visual-DPCC with `n_diffusion_steps=1` | **Train crashed on iteration 0** before any optimizer step. No model was trained. | ❌ shape mismatch (9 vs 23 channels) at the first conv |
| `visual_fm`   | Eval Visual-FM with ODE steps=1 | Loaded an existing 100-step-trained FM checkpoint, **forced eval to 1 Euler step**, ran 5 contexts → **0/5 success, 0.28 m mean dist**, then crashed setting up the projector for the next variant | ⚠️ partial success then crash; FM was NOT trained for one-shot, only eval-overridden |

Both failures share a single root cause: **UF-17 non-visual mode is half-wired into the visual config/eval path. Toggling `if_vision=False` from the CLI silently mixes 9-D and 23-D plumbing.**

---

## 1. DPCC training (`visual_dpcc`) — channel mismatch at first conv

### Evidence

CLI config dump (log line 43):
```
config='config.aligning-d3il-visual', if_vision=False,
action_dim=3, obs_dim=6, n_diffusion_steps=1,
prefix='visual_aligning_dpcc/', model='...VisualUNet', diffusion='...VisualGaussianDiffusion'
```

Dataset path actually taken (log line 14, 28):
```
[ train ] dataset=StateOnlyAligningDataset (non-visual, 23D trajectory)
[ StateOnlyAligningDataset ] 900 episodes, 168274 windows (horizon=8, traj_dim=23)
```

Normalizer dims (log lines 31, 37): `obs_normalizer dim=20`, `act_normalizer dim=3` → **23D trajectory feed**.

Crash (log line 133):
```
RuntimeError: Given groups=1, weight of size [32, 9, 5],
expected input[64, 23, 8] to have 9 channels, but got 23 channels instead
```

### Root cause

You launched **variant `visual_aligning_dpcc`** (the *visual* variant, `config/aligning-d3il-visual.py:321`) and tried to flip it non-visual by overriding `if_vision=False` on the CLI. The visual variant hardcodes:

```python
# config/aligning-d3il-visual.py:330
'obs_dim': 6,               # 6D obs: [des_c_pos(3), c_pos(3)] — MUST be 6, never 3 or 128
'action_dim': 3,
'if_vision': True,          # ← you overrode this to False
```

In `VisualUNet.__init__` (`diffuser_visual_aligning/models/visual_unet.py:70-74`):

```python
if self.if_vision:
    transition_dim = self.TRANSITION_DIM        # 9
else:
    obs_dim = getattr(config, 'obs_dim', 20)
    transition_dim = config.action_dim + obs_dim   # 3 + 6 = 9   ← STILL 9, not 23
```

So with `if_vision=False` but `obs_dim` still 6 (visual leftover), the first conv was built with **9 input channels** — but the **dataset code** correctly read `if_vision=False` and picked `StateOnlyAligningDataset` which feeds **23D** trajectories (`[action(3) | obs(20)]`).

`n_diffusion_steps=1` had **zero** to do with the crash — the model would have failed at any step count.

### The variant you should have used

`config/aligning-d3il-visual.py:688` already defines the correct non-visual variant:

```python
base['ddpm_encdec_vision_nonvisual'] = {
    **base['ddpm_encdec_vision'],
    'action_dim': 3,
    'obs_dim': 20,                # ← correct full state
    'if_vision': False,
    'prefix': 'ddpm_encdec_vision_nonvisual/',
    ...
}
```

CLI fix:
```bash
# wrong (what you ran):
... --config config.aligning-d3il-visual --prefix visual_aligning_dpcc/ \
    --if_vision False --n_diffusion_steps 1

# right:
... --config config.aligning-d3il-visual --prefix ddpm_encdec_vision_nonvisual/ \
    --n_diffusion_steps 1
# (no need to override if_vision — the variant already sets it to False with obs_dim=20)
```

---

## 2. FM evaluation (`visual_fm`) — two issues stacked

### 2a. The FM model was **not** trained one-shot — only eval was overridden

Log line 26 — checkpoint path loaded:
```
logs/aligning-d3il-visual/fm_visual_aligning/
  H8_Dfm_visual_aligning.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1_VFalse_steps900_bs64/6
```

This is a pre-existing checkpoint (line 45: `Restored loss history from checkpoint at step 87000`). Log line 47:
```
[ eval ] Model n_timesteps = 100   (config n_diffusion_steps = ?)
[ eval ] FM flow_steps_v3 = 1  (Euler ODE integration steps 0→1) [overridden from args]
```

So the loaded checkpoint was trained with **100 ODE steps**, and eval forced a **1-step Euler integration** on top of that velocity field. That is *not* the same as a model trained to be one-shot — it's a brutal under-integration of a model that learned a long-curvature path. Expect quality to collapse, which it did (see §2c).

### 2b. Projector setup crashed — same UF-17 contamination

Crash (log line 274-283):
```
File ".../eval_fm_visual_aligning.py", line 1851, in <module>
    projector = setup_dpcc_projector(...)
File ".../fm_visual_aligning/sampling/projection.py", line 401, in build_matrices
    a = bound[0] * (x_max - x_min) / 2
ValueError: operands could not be broadcast together with shapes (23,) (9,)
```

Smoking-gun branch at `eval_fm_visual_aligning.py:1848-1853`:

```python
_if_vis   = getattr(args, 'if_vision', True)
_traj_dim = 9 if _if_vis else 23      # UF-17: non-visual uses 23D trajectory
if 'diffuser' not in variant and obs_normalizer is not None:
    projector = setup_dpcc_projector(
        args, geo_config, obs_normalizer, act_normalizer, variant, is_tightened,
        trajectory_dim=_traj_dim)
```

Your run was launched with `if_vision=False` (the `_VFalse_` in the checkpoint path, and log line 65: `WARNING: config if_vision=False but record_mode is active → auto-enabling visual mode...`). So:

- `args.if_vision = False`  →  `_traj_dim = 23` (projector built for 23D)
- **But** the actual model+dataset are visual: `ParityAligningDataset traj_dim=9` (line 44), normalizer is 6-D obs / 3-D act (lines 63-64). The model emits 9-D trajectories.

Inside `Projector.build_matrices` the bounds vector is 23-element (built from `_traj_dim`) while `x_max - x_min` is 9-element (from the visual normalizer). They can't broadcast → crash.

UF-13's "auto-enable visual mode for recording" only patches the rendering path; it does **not** flip `args.if_vision` back to True for the projector branch. So the projector keeps thinking it's non-visual.

### 2c. The one variant that DID complete shows the under-integration damage

The first variant (`combined_5` / diffuser) finished before the crash. Across 5 contexts:

| Metric | Value |
|---|---|
| Success rate | **0/5 = 0.0%** |
| Avg final mean distance | 0.2804 m  (±0.0837 m) |
| Min final mean distance | 0.1853 m |
| Avg steps | 300 / 300 (always max-out) |
| Constraint satisfaction (execution) | 0.691 ± 0.240 |
| Avg violated steps / rollout | 92.6 (mostly bounds: 85.0; obstacles: 10.4) |
| Max bounds violation | 0.36 m  (±0.43) |
| Zero-violation rollouts | **1 / 5 (20%)** |
| Avg inference time / replan | 0.012 s |

Diagnostic action magnitudes (lines 95-100, 124-129, …): `norm|a0|` grew from ~1.0 at replan 50 to **~2.9 at replan 300** — far outside the unit-normalized space. The denormalized step is still small (~1.2 cm) because the action normalizer range is tiny (±0.0083), but **the model is hallucinating actions far outside the training distribution** and getting clamped by the env. Classic symptom of single-Euler-step integration of a non-straight velocity field: the predicted velocity is correct at t=0, but extrapolating it for the full 0→1 interval lands way outside the data manifold.

That's exactly what mean-flow / one-shot training (iMeanFlow, Drifting) tries to *fix*: train the model so that a single Euler step is a valid integral, not just a tangent.

---

## 3. The common thread

UF-17 (non-visual aligning) added a parallel 23-D path through three components:

1. **Dataset selection** — `if_vision` switch picks `StateOnlyAligningDataset` (23-D) vs `ParityAligningDataset` (9-D). ✅ works.
2. **Model construction** — `VisualUNet` branches on `if_vision` to pick `transition_dim`. ⚠️ relies on `config.obs_dim` being correctly set (20 for non-visual, 6 for visual). The CLI does **not** override `obs_dim` when you flip `if_vision`.
3. **Projector construction** — eval script derives `_traj_dim` from `args.if_vision` alone. ⚠️ doesn't cross-check against the actual normalizer or loaded checkpoint.

Toggling **only** `if_vision` from the CLI silently mixes the two paths because the other two components read different sources of truth:
- Model reads `config.obs_dim` (frozen at variant-definition time).
- Projector reads `args.if_vision` (CLI-mutable).
- Dataset reads `config.if_vision` (CLI-mutable).

When you set `if_vision=False` on a visual variant, dataset and projector flip but model construction keeps the visual `obs_dim=6`. Result: 9-D model vs 23-D data (training crash) or 23-D projector vs 9-D model (eval crash).

---

## 4. What's broken, what's fine, what's untested

### Broken / misleading

- **Variant selection ergonomics**: There's no safety check that `if_vision=False + variant=visual_aligning_dpcc` is incoherent. Should refuse to start, or auto-pick the matching `_nonvisual` variant.
- **Eval `_traj_dim` derivation**: `eval_fm_visual_aligning.py:1849` should derive from `obs_normalizer.mins.shape[0] + act_normalizer.mins.shape[0]` (or from a saved config), not from `args.if_vision`. That would have prevented §2b regardless of CLI flags.
- **Train script**: should assert `config.obs_dim` matches the dataset's `obs_dim` at the start of training, with a fail-fast error.

### Fine

- Config blocks themselves (`visual_aligning_dpcc`, `ddpm_encdec_vision_nonvisual`) are individually correct.
- `StateOnlyAligningDataset` itself is correctly producing 23-D trajectories with `obs_dim=20`.
- The non-visual UF-17 path (with the correct variant) was previously verified end-to-end — neither of these crashes invalidates that.

### Untested / unknown after this run

- **Whether DPCC actually works with `n_diffusion_steps=1`** — training never started. To test, retry against the right variant.
- **Whether FM trained for one-shot would do better** than the under-integrated 100-step model — needs a new training run with `n_diffusion_steps=1` (or `flow_steps_v3=1` baked into training, not just eval).

---

## 5. Recommended fixes (in priority order)

| # | Fix | File / Line | Effort |
|---|---|---|---|
| 1 | **Re-run DPCC training using the right variant** (`ddpm_encdec_vision_nonvisual`) with `n_diffusion_steps=1` | CLI invocation in your sbatch | trivial |
| 2 | **For the FM under-integration: re-eval the existing checkpoint with more ODE steps** (e.g. `flow_steps_v3=20`). Vanilla FM's train and eval ARE decoupled — the model learned a continuous velocity field `v_θ(x, t)` for all `t ∈ [0,1]` regardless of eval step count. The 0% success was caused by 1-step Euler being too coarse for a *curved* probability flow (multi-modal aligning), not by training. No retraining needed. If you specifically want a model where 1-step is **valid by construction**, switch to mean-flow / **iMeanFlow** — that is a different *loss formula* (regresses interval-averaged velocity `(x_data - x_r)/h` and conditions on `h`), and that case does require new training. | eval re-run *or* iMF retrain | trivial / small |
| 3 | **Make `_traj_dim` derive from saved normalizers**, not `args.if_vision` — would have prevented §2b | `fm_visual_aligning_test/eval_fm_visual_aligning.py:1848-1849` | ~5 lines |
| 4 | **Assert config.obs_dim ↔ dataset.obs_dim at train start** | `diffuser_visual_aligning/utils/training.py` (top of `train()`) | ~5 lines |
| 5 | **Reject incoherent variant+if_vision combos at CLI parse time** — refuse to start if `prefix='visual_aligning_dpcc/'` and `if_vision=False` (and vice versa for `_nonvisual` + `if_vision=True`) | wherever args are validated post-parse | ~10 lines |

Fixes 3–5 are belt-and-suspenders; #1 alone unblocks the experiment you actually wanted to run.

---

## 6. Open question for you — what actually tests "one-shot DGM = in-loop projection"?

The original goal was to test "does one-shot DGM behave like in-loop projection
if it's powerful enough" (the FM-D philosophical thread from earlier). The
DDPM and FM sides of that question behave **differently** w.r.t. train/eval
coupling:

- **(a) Train DPCC with `n_diffusion_steps=1`** on the visual variant
  (`visual_aligning_dpcc`, 9-D), eval with the matching plan config.
  DDPM's noise schedule is *discrete*: a model trained with T=100 sees a
  different forward process than one trained with T=1, so the schedule must
  match between train and eval. Apples-to-apples within the diffusion family
  requires **retraining at T=1**.

- **(b) For FM, train and eval are decoupled.** The model learns a continuous
  velocity field `v_θ(x, t)` for all `t ∈ [0,1]`; eval picks any ODE solver +
  step count. So testing "does the existing FM model integrate cleanly in one
  step" is just an **eval re-run with `flow_steps_v3=1`** (which we did — and
  it failed at 0% success because the underlying flow is curved, not because
  the model was trained wrong). To test "does one-shot FM emerge naturally,"
  the right next step is to **eval the same checkpoint with progressively
  fewer ODE steps** (20 → 10 → 5 → 2 → 1) and see at what step count quality
  degrades. That curve answers the philosophical question directly: if
  quality holds at low step counts, the learned field is near-straight and
  one-shot-ish for free. If it collapses, the field is curved and a single
  Euler step is fundamentally unsuitable — at which point the only remedy is
  a **different objective** (mean-flow / iMeanFlow), not more vanilla FM
  training.

So the corrected experiment plan:
- **DDPM side**: retrain at `n_diffusion_steps=1` (and use the correct
  `ddpm_encdec_vision_nonvisual` variant if you want non-visual). Mandatory.
- **FM side**: do *not* retrain. Sweep eval `flow_steps_v3 ∈ {1, 2, 5, 10, 20}`
  on the existing checkpoint to characterize the curvature of the learned
  flow. Only retrain (with mean-flow / iMF) if you have specific evidence
  that one-step Euler is structurally hopeless.
