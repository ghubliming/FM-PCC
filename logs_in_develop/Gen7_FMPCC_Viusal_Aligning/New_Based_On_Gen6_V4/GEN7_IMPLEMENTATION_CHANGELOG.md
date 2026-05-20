# Gen7 Visual-FM (fm_visual_aligning) — Implementation Changelog

**Date:** 2026-05-20  
**Author:** Claude Sonnet 4.6  
**Branch:** update_into_FM  
**Plan reference:** GEN7_IMPLEMENTATION_PLAN.md  

---

## Phase 0 — Pre-Requisite Fixes (applied to shared codebase)

### 0.1  BGR/RGB pipeline fix
**File:** `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py` (lines ~205-217)  
**Change:** Removed both `cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)` and
`cv2.cvtColor(inhand_image, cv2.COLOR_RGB2BGR)` from `get_observation()`.  
**Why:** MuJoCo renderer returns RGB natively. Training pipeline (`_load_images()`) also uses
`cv2.COLOR_BGR2RGB` to produce RGB. The env's conversion inverted colors unnecessarily.
Removing it makes train/eval colour spaces consistent.  
**Benefit:** Applies to both Gen6V4 and Gen7.

### 0.2  Eval seeding diversity fix
**File:** `d3il/simulation/aligning_sim.py` (lines 62-64)  
**Change:** `random.seed(pid)` / `torch.manual_seed(pid)` / `np.random.seed(pid)` →  
`random.seed(self.seed + pid)` / `torch.manual_seed(self.seed + pid)` / `np.random.seed(self.seed + pid)`  
**Why:** Fixed-pid seeding made all eval runs share the same random state regardless of which
seed (6,7,8,9,10) was requested, eliminating statistical diversity across seeds.

### 0.3  des_robot_pos `.copy()` fix
**File:** `d3il/simulation/aligning_sim.py` (line 93)  
**Change:** `des_robot_pos = env_state[:3]` → `des_robot_pos = env_state[:3].copy()`  
**Why:** Symmetry with line 94 (`robot_pos = env_state[3:6].copy()`); prevents possible
aliasing if env_state array is mutated downstream.

### 0.4  Normalization comment fix (A3)
**File:** `diffuser_visual_aligning/datasets/normalization.py` (line 160)  
**Change:** Comment corrected from `"constant dims map to 0"` → `"constant dims map to -1 (normalized minimum)"`.  
**Why:** LimitsNormalizer maps constant dims to the `-1` endpoint of `[-1,1]`, not to `0`.
The wrong comment would mislead anyone reading the normalization code during debugging.

### 0.5  Fix11 comment inversion fix
**File:** `d3il/simulation/aligning_sim.py` (lines 86-89)  
**Change:** Rewrote the comment block that incorrectly stated `_load_images()` "accidentally
produces BGR" — corrected to accurately describe the RGB training pipeline and the empirical
evidence from Fix7 that the model was robustly handling the colour space.  
**Why:** Inverted comment would mislead future debugging of image-space issues.

---

## Phase 1 — New `fm_visual_aligning/` Package

All files created under `fm_visual_aligning/` as a sibling package to `diffuser_visual_aligning/`.
Copy-modify pattern: shared modules copied verbatim; only the diffusion engine and import paths changed.

### 1.1  Package scaffolding
- `fm_visual_aligning/__init__.py` — `from . import *`
- `fm_visual_aligning/setup.py` — `name = 'fm_visual_aligning'`
- `fm_visual_aligning/models/__init__.py` — imports GaussianDiffusion, VisualGaussianDiffusion, VisualUNet, UNet1DTemporalCondModel
- `fm_visual_aligning/datasets/__init__.py` — verbatim
- `fm_visual_aligning/sampling/__init__.py` — verbatim
- `fm_visual_aligning/utils/__init__.py` — verbatim (all relative imports)

### 1.2  `fm_visual_aligning/models/diffusion.py` — NEW FM ODE engine
**Key FM changes vs Gen6V4 DDPM engine:**

| Property | DDPM (Gen6V4) | FM (Gen7) |
|---|---|---|
| Time domain | `randint(0, T)` discrete | `Beta(α,β)` continuous `t ∈ [0,1]` |
| Noise schedule | cosine β schedule | `betas = linspace(1,0)` (placeholder) |
| Noising | `sqrt(ᾱ)*x + sqrt(1-ᾱ)*ε` | `(1-t)*x_noise + t*x_data` |
| Prediction target | ε (noise) | v = x_data - x_noise (velocity) |
| Reverse step | stochastic DDPM posterior | deterministic Euler: `x + v*dt` |
| Inference direction | T→0 reverse chain | 0→1 forward ODE |
| Inference steps | `n_timesteps=100` | `flow_steps_v3=100` |

**Constructor signature** accepts both `ode_solver_backend_v3` AND `ode_solver_method_v3`
(critical: `utils/config.py:92` passes ALL dict keys to constructor with no filtering).

**Key methods:**
- `_time_from_timestep(t)`: normalises int/float t to `[0,1]`
- `q_sample(x_start, t, noise)`: FM linear interpolation
- `p_losses(x_start, cond, t)`: velocity loss `||v_pred - (x_data - x_noise)||`
- `p_mean_variance(x, cond, t)`: Euler step `x + velocity * (1/flow_steps_v3)`
- `p_sample(x, ...)`: deterministic (no Gaussian noise added)
- `p_sample_loop(shape, ...)`: forward ODE 0→1, projector applied at `t ≥ (1-threshold)*K`

### 1.3  `fm_visual_aligning/models/visual_gaussian_diffusion.py` — FM subclass
**Changes vs Gen6V4:**
- `loss()` method: replaces `torch.randint` DDPM time with `Beta(1.5, 1.0)` continuous time + inversion `t = 1 - t`
- No `p_mean_variance` override — base FM class handles Euler step correctly
- `forward()` cond-dict reshaping: **verbatim from Gen6V4** (D3IL inference API unchanged)

### 1.4  `fm_visual_aligning/models/visual_unet.py` — import path update
**Change:** `from diffuser_visual_aligning.models.unet1d_temporal_cond import ...`
→ `from fm_visual_aligning.models.unet1d_temporal_cond import ...`  
All other logic identical.

### 1.5  `fm_visual_aligning/models/unet1d_temporal_cond.py` — verbatim copy
No package-specific imports; all `.helpers` imports are relative.

### 1.6  `fm_visual_aligning/models/helpers.py` — one import change
**Change:** `import diffuser_visual_aligning.utils as utils`
→ `import fm_visual_aligning.utils as utils`  
(Used by `ValueLoss.forward()` for `utils.to_np`.)

### 1.7  `fm_visual_aligning/datasets/` — verbatim copies
- `datasets/sequence.py` — `ParityAligningDataset` (9D, LimitsNormalizer, per-episode pkl loading)
- `datasets/normalization.py` — `LimitsNormalizer`, `CDFNormalizer`, etc.

### 1.8  `fm_visual_aligning/sampling/projection.py` — verbatim copy
`Projector` with SLSQP solver, Fix 9.1/9.2/9.3 fixes for empty-constraint early exit.

### 1.9  `fm_visual_aligning/utils/` — verbatim copies
`config.py`, `training.py`, `serialization.py`, `arrays.py`, `setup.py`,
`progress.py`, `logger.py`, `plot.py`, `constraints_helpers.py`  
All use relative imports only; `config.py` uses `__name__.split('.')[0]` which auto-resolves
to `fm_visual_aligning` — no string replacement needed.

---

## Phase 2 — Entry Scripts

### 2.1  `fm_visual_aligning_test/train_fm_visual_aligning.py`
**Changes from Gen6V4 train script:**
- Header comment updated to "Visual-FM (Gen7)"
- `import diffuser_visual_aligning.utils` → `import fm_visual_aligning.utils`
- `ParityAligningDataset` import from `fm_visual_aligning.datasets.sequence`
- `VisualUNet` import from `fm_visual_aligning.models.visual_unet`
- `VisualGaussianDiffusion` import from `fm_visual_aligning.models.visual_gaussian_diffusion`
- `experiment='visual_aligning_dpcc'` → `experiment='fm_visual_aligning'`
- `wandb_project` default: `'FMPCC-visual-aligning-dpcc'` → `'FM-PCC-visual-aligning-gen7'`
- `diffusion_config` gains FM params:
  `time_beta_alpha_v3`, `time_beta_beta_v3`, `flow_steps_v3`, `ode_solver_backend_v3`, `ode_solver_method_v3`
- Final print: `'Visual-FM (Gen7) training completed.'`

### 2.2  `fm_visual_aligning_test/eval_fm_visual_aligning.py`
**Changes from Gen6V4 eval script:**
- Header comment updated to "Visual-FM (Gen7)" with FM-specific notes
- `import diffuser_visual_aligning.utils` → `import fm_visual_aligning.utils`
- `from diffuser_visual_aligning.sampling.projection import Projector`
  → `from fm_visual_aligning.sampling.projection import Projector`
- `experiment='plan_visual_aligning_dpcc'` → `experiment='plan_fm_visual_aligning'`
- Added `flow_steps_v3` diagnostic print after model load:  
  `[ eval ] FM flow_steps_v3 = {_flow_steps}  (Euler ODE integration steps 0→1)`
- Final print: `'Visual-FM (Gen7) evaluation completed.'`

---

## Phase 3 — Config (`config/aligning-d3il-visual.py`)

### 3.1  New `args_to_watch` functions
Added `args_to_watch_fm_visual_train` and `args_to_watch_fm_visual_plan`.
- Train watches: `prefix, horizon, n_diffusion_steps, diffusion, action_weight, if_vision, max_path_length`
- Plan watches: `prefix, horizon, flow_steps_v3, diffusion_timestep_threshold, diffusion, if_vision, max_episode_length`
- Both include `if_vision` (absent from dpcc variants), ensuring visual/non-visual checkpoints
  land in distinct directories.

### 3.2  New `base['fm_visual_aligning']` training block
- `model/diffusion` class paths point to `fm_visual_aligning.*`
- FM params: `time_beta_alpha_v3=1.5`, `time_beta_beta_v3=1.0`, `flow_steps_v3=100`,
  `ode_solver_backend_v3='legacy_euler'`, `ode_solver_method_v3='euler'`
- `n_diffusion_steps=100` retained as checkpoint dir key (matches `plan_fm_visual_aligning`)
- `prefix='fm_visual_aligning/'`
- All other hyperparams match Gen6V4 dpcc block

### 3.3  New `base['plan_fm_visual_aligning']` eval block
- FM inference params added
- `diffusion_loadpath` mirrors training prefix with matching key fragments
- `exp_name` uses `args_to_watch_fm_visual_plan`
- `max_episode_length=400` (proven D3IL default)

---

## Phase 4 — YAML (`config/visual_aligning_eval.yaml`)

Added `'fm_visual_aligning'` to `exps` list.  
`eval_fm_visual_aligning.py` is entry-point agnostic w.r.t. experiment; the YAML's `exps` list
is used only by aggregation scripts. The eval script uses `experiment='plan_fm_visual_aligning'`
which directs the `Parser` to the correct config block.

---

## Invariants preserved

| Invariant | Status |
|---|---|
| 9D trajectory `[act(3)\|des_c_pos(3)\|c_pos(3)]` | ✅ TRANSITION_DIM=9 hardcoded in VisualUNet |
| DPCC SLSQP projector | ✅ verbatim copy, unchanged |
| Euler dynamics constraint `[6←0, 7←1, 8←2]` | ✅ eval script unchanged |
| Visual/non-visual both supported | ✅ `if_vision` flag preserved in VisualUNet |
| Trainer `loss(*batch)` API | ✅ `VisualGaussianDiffusion.loss(trajectories, conditions)` |
| LimitsNormalizer + pkl normalizer files | ✅ dataset/normalizer code verbatim |
| `clip_denoised=False` | ✅ forced in both train and eval |

---

## Files changed summary

| File | Type | Change |
|---|---|---|
| `d3il/environments/.../aligning.py` | Fix | Removed BGR↔RGB env conversion (Phase 0.1) |
| `d3il/simulation/aligning_sim.py` | Fix | Seeding, .copy(), comment (Phase 0.2-0.5) |
| `diffuser_visual_aligning/datasets/normalization.py` | Fix | Comment A3 (Phase 0.4) |
| `fm_visual_aligning/**` (25 files) | New | Full FM package (Phase 1) |
| `fm_visual_aligning_test/train_fm_visual_aligning.py` | Modified | FM imports + params (Phase 2.1) |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Modified | FM imports + experiment (Phase 2.2) |
| `config/aligning-d3il-visual.py` | Modified | 2 args_to_watch + 2 config blocks (Phase 3) |
| `config/visual_aligning_eval.yaml` | Modified | Added `fm_visual_aligning` to exps (Phase 4) |
