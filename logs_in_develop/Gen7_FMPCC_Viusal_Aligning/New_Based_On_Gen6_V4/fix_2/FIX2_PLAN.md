# Gen7 Fix 2 — Align `fm_visual_aligning` Config with FMv3ODE Pattern

**Date:** 2026-05-20
**Branch:** update_into_FM
**Reference model:** `FM_v3_ode_selectable_test/` + `config/avoiding-d3il.py` (`flow_matching_v3_ode_selectable` / `plan_fm_v3_ode_selectable`)

---

## Problem Statement

The `fm_visual_aligning` + `plan_fm_visual_aligning` blocks in `config/aligning-d3il-visual.py`
were modelled after the DPCC (`visual_aligning_dpcc`) pattern rather than the FMv3ODE pattern.
Six concrete issues were identified by comparing against the `avoiding-d3il.py` reference.

---

## Issue 1 — `args_to_watch_fm_visual_train`: DDPM naming instead of FM naming

**Location:** `config/aligning-d3il-visual.py` lines 56-64

**Current (wrong):**
```python
args_to_watch_fm_visual_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'K'),   # ← DDPM concept; FM has no discrete denoising chain
    ('diffusion', 'D'),
    ('action_weight', 'aw'),
    ('if_vision', 'V'),
    ('max_path_length', 'steps'),
]
```

**Reference (`args_to_watch_fmv3_ode_train` in `avoiding-d3il.py`):**
```python
args_to_watch_fmv3_ode_train = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('diffusion', 'D'),
    ('time_beta_alpha_v3', 'a'),   # Beta distribution α identifies the FM training dist
    ('time_beta_beta_v3', 'b'),    # Beta distribution β
    ('action_weight', 'aw'),
]
```

**Fix:** Replace `('n_diffusion_steps', 'K')` with `('time_beta_alpha_v3', 'a')` and
`('time_beta_beta_v3', 'b')`. Keep `('if_vision', 'V')` and `('max_path_length', 'steps')`
as visual-specific additions (not in avoiding-d3il but meaningful here).

**⚠ Checkpoint naming impact:** This changes the directory name of every trained Gen7
checkpoint. Since Gen7 has **not been trained yet**, no existing checkpoints are affected.
`diffusion_loadpath` in `plan_fm_visual_aligning` must be updated in sync (Issue 5).

---

## Issue 2 — `args_to_watch_fm_visual_plan`: missing `ode_solver_method_v3`, has wrong `max_episode_length`

**Location:** `config/aligning-d3il-visual.py` lines 66-74

**Current (wrong):**
```python
args_to_watch_fm_visual_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
    ('if_vision', 'V'),
    ('max_episode_length', 'steps'),   # ← eval-only env param; irrelevant in checkpoint dir names
]
# Missing: ('ode_solver_method_v3', 'M')
```

**Reference (`args_to_watch_fmv3_ode_plan`):**
```python
args_to_watch_fmv3_ode_plan = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('flow_steps_v3', 'K'),
    ('ode_solver_method_v3', 'M'),           # ← distinguishes euler/rk4/dopri5 runs
    ('diffusion_timestep_threshold', 'T'),
    ('diffusion', 'D'),
]
```

**Fix:**
- Add `('ode_solver_method_v3', 'M')`.
- Remove `('max_episode_length', 'steps')` — this is a simulator step budget, not a model
  hyperparameter; it does not belong in a checkpoint results directory name.
- Keep `('if_vision', 'V')` (visual-specific, valid addition).

---

## Issue 3 — `fm_visual_aligning` training block: inference-only params polluting training config

**Location:** `config/aligning-d3il-visual.py` `base['fm_visual_aligning']`

**Current (wrong — has inference-only params in training block):**
```python
'n_diffusion_steps': 100,            # DEAD for FM — continuous time, no discrete chain
'flow_steps_v3': 100,                # DEAD in training — inference-only ODE step count
'ode_solver_backend_v3': 'legacy_euler',  # DEAD in training — inference-only
'ode_solver_method_v3': 'euler',          # DEAD in training — inference-only
```

**Reference (`flow_matching_v3_ode_selectable` in `avoiding-d3il.py`):**
```python
# 'n_diffusion_steps': 20, # DEAD code (mathematically irrelevant for FM flow)
# 'flow_steps_v3': 10, # DEAD code (inference-only parameter)
# ODE backend/method NOT present in training block at all
'time_beta_alpha_v3': 1.5,           # Only FM training params are present
'time_beta_beta_v3': 1.0,
```

**Fix:** In the `fm_visual_aligning` training block:
- Mark `n_diffusion_steps` as DEAD code (comment it out or add explicit comment).
- Remove `flow_steps_v3`, `ode_solver_backend_v3`, `ode_solver_method_v3` from the
  training block. They belong only in `plan_fm_visual_aligning`.
- `time_beta_alpha_v3` and `time_beta_beta_v3` stay — they are training parameters.

**Note:** The train script (`train_fm_visual_aligning.py`) uses
`getattr(args, 'flow_steps_v3', _n_diff_steps)` as a fallback — this is fine because
`getattr` with default won't crash if the config key is absent.

---

## Issue 4 — `plan_fm_visual_aligning` plan block: wrong prefix format, missing ODE tolerance params

**Location:** `config/aligning-d3il-visual.py` `base['plan_fm_visual_aligning']`

### 4a — Prefix uses DDPM `K{n_diffusion_steps}` instead of FM Beta params

**Current prefix:**
```python
'prefix': (
    'f:plans/fm_visual_aligning/'
    'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
    '_aw{action_weight}_V{if_vision}_steps{max_path_length}/'
),
```

**Reference (`plan_fm_v3_ode_selectable`):**
```python
'prefix': 'f:plans/flow_matching_v3_ode_selectable/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/',
```

**Fix:**
```python
'prefix': (
    'f:plans/fm_visual_aligning/'
    'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
    '_aw{action_weight}_V{if_vision}_steps{max_path_length}/'
),
```

### 4b — Missing ODE tolerance/step-size params

**Reference has:**
```python
'ode_solver_rtol_v3': None,
'ode_solver_atol_v3': None,
'ode_solver_step_size_v3': None,
```

These must be present in the plan block because the Config system passes all dict keys to the
constructor (no filtering). If they are absent and the eval script loads the config, they won't
be passed — but if any code later tries to read them from `args`, it will fail. Adding them as
`None` ensures forward compatibility.

**Fix:** Add the three tolerance params to `plan_fm_visual_aligning`.

---

## Issue 5 — `diffusion_loadpath` must match new training `exp_name`

**Location:** `config/aligning-d3il-visual.py` `base['plan_fm_visual_aligning']['diffusion_loadpath']`

After fixing Issue 1 (training `args_to_watch` changes from K-style to a/b-style), the
`exp_name` of the training block will generate directory names like:
`fm_visual_aligning/H8_DVisualGaussianDiffusion_a1.5_b1.0_aw10_VTrue_steps1000/<seed>`

The `diffusion_loadpath` must mirror this exactly.

**Current (wrong):**
```python
'diffusion_loadpath': (
    'f:fm_visual_aligning/'
    'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
    '_aw{action_weight}_V{if_vision}_steps{max_path_length}'
),
```

**Fix:**
```python
'diffusion_loadpath': (
    'f:fm_visual_aligning/'
    'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
    '_aw{action_weight}_V{if_vision}_steps{max_path_length}'
),
```

Also remove `'n_diffusion_steps': 100` from the plan block (it's no longer in the naming
and is DEAD for FM). The `time_beta_alpha_v3` and `time_beta_beta_v3` must be present in the
plan block (they already are) so the f-string can resolve.

---

## Issue 6 — Core model: `VisualGaussianDiffusion` missing ODE tolerance param intercepts

**Location:** `fm_visual_aligning/models/visual_gaussian_diffusion.py` line 22

**Current:**
```python
def __init__(self, *args, ode_solver_backend_v3='legacy_euler', ode_solver_method_v3='euler', **kwargs):
```

**Problem:** Adding `ode_solver_rtol_v3`, `ode_solver_atol_v3`, `ode_solver_step_size_v3` to
the plan config block (Issue 4b) means the Config system will pass them to the constructor.
They are NOT in the base `GaussianDiffusion.__init__` signature, and there is no `**kwargs`
there — so they would land in `VisualGaussianDiffusion.__init__`'s `**kwargs` and get passed
to `super().__init__(**kwargs)`, which WILL raise `TypeError: unexpected keyword argument`.

**Fix:** Intercept all three in `VisualGaussianDiffusion.__init__`:
```python
def __init__(self, *args,
             ode_solver_backend_v3='legacy_euler',
             ode_solver_method_v3='euler',
             ode_solver_rtol_v3=None,
             ode_solver_atol_v3=None,
             ode_solver_step_size_v3=None,
             **kwargs):
    super().__init__(*args, **kwargs)
```

These are stored as instance attributes only if the ODE engine needs them at inference time
(the current `legacy_euler` backend doesn't use them). Adding the signature intercepts is
zero-cost and prevents crashes when the full plan config is loaded.

---

## Issue 7 — Train script misleading DDPM print comment

**Location:** `fm_visual_aligning_test/train_fm_visual_aligning.py` lines 201-203

**Current:**
```python
_n_diff_steps = getattr(args, 'n_diffusion_steps', 100)
print(f'[ train ] n_diffusion_steps = {_n_diff_steps}  '
      f'(must match eval config to avoid denoising-chain mismatch)')
```

**Problem:** "denoising-chain mismatch" is pure DDPM thinking. FM has no discrete denoising chain.
`n_diffusion_steps` is only used as fallback for `n_timesteps` in the constructor (buffer sizing),
which is functionally irrelevant for FM's continuous-time training.

**Fix:** Update the print message to reflect FM semantics:
```python
_n_diff_steps = getattr(args, 'n_diffusion_steps', 100)
print(f'[ train ] n_timesteps (legacy buffer size) = {_n_diff_steps}  '
      f'(FM uses continuous time; this value does not affect training dynamics)')
```

---

## Retraining Required?

| Scenario | Answer |
|---|---|
| Gen7 not yet trained | **No retraining.** No checkpoints exist with the old K-naming. |
| Gen6V4 (diffuser_visual_aligning) | **Not affected.** This fix only touches `fm_visual_aligning` config blocks. |
| Re-evaluation of Gen7 after training with new naming | Eval `diffusion_loadpath` will resolve correctly after fix. |

---

## Files to Change

| File | Issues addressed |
|---|---|
| `config/aligning-d3il-visual.py` | Issues 1, 2, 3, 4a, 4b, 5 |
| `fm_visual_aligning/models/visual_gaussian_diffusion.py` | Issue 6 |
| `fm_visual_aligning_test/train_fm_visual_aligning.py` | Issue 7 |

---

## Change Summary Table

| Item | Before (DDPM-style) | After (FM-style) |
|---|---|---|
| Train `args_to_watch` key | `('n_diffusion_steps', 'K')` | `('time_beta_alpha_v3', 'a'), ('time_beta_beta_v3', 'b')` |
| Plan `args_to_watch` key | missing `M` | `('ode_solver_method_v3', 'M')` added |
| Plan `args_to_watch` key | `('max_episode_length', 'steps')` | removed |
| Train block params | `flow_steps_v3`, `ode_solver_backend/method_v3` present | removed (inference-only) |
| Plan block prefix | `K{n_diffusion_steps}` | `a{time_beta_alpha_v3}_b{time_beta_beta_v3}` |
| Plan block ODE tolerances | absent | `rtol/atol/step_size_v3=None` added |
| `diffusion_loadpath` | `K{n_diffusion_steps}` | `a{...}_b{...}` |
| `VisualGaussianDiffusion.__init__` | 2 ODE params intercepted | 5 ODE params intercepted |
| Train script print | "denoising-chain mismatch" | FM-accurate message |
