# Gen9 Epoch 2 — Single Camera Visual Avoiding Pipeline

**Date**: 2026-06-02
**Status**: 📋 PLAN (pending implementation)
**Parent**: Gen9 Visual Avoiding
**Predecessor**: Gen7 Visual Aligning (Gen6V4 DPCC + FM dual-camera)

---

## 1. Executive Summary

Upgrade the existing **dpcc visual aligning** (`diffuser_visual_aligning/`) and **fm visual aligning** (`fm_visual_aligning/`) pipelines to a new **visual avoiding** task. Key architectural change: **single camera (bp-cam only)** — the wrist/inhand camera is irrelevant for the avoiding task because the robot never grasps objects; it only dodges obstacles.

**Strategy**: Copy → Rename → Modify. Minimal new code creation — just targeted changes in the copies.

---

## 2. Dimensional Analysis: Aligning → Avoiding

> **IMPORTANT**: The avoiding task has fundamentally different dimensions from aligning. All hardcoded dimension constants must be updated.

| Aspect | **Visual Aligning** (current) | **Visual Avoiding** (target) | Delta |
|---|---|---|---|
| **D3IL `obs_dim`** | 3 `[robot_ee_pos]` | 4 `[des_x, des_y, c_x, c_y]` *(see §12 audit — NOT obstacle positions)* | +1 (2D world) |
| **D3IL `action_dim`** | 3 `[dx, dy, dz]` | 2 `[dx, dy]` | -1 (2D plane) |
| **D3IL `max_len_data`** | 512 | 200 | shorter episodes |
| **Our obs_dim** (DPCC) | 6 `[des_c_pos(3), c_pos(3)]` | **TBD** — see §2.1 | — |
| **Our action_dim** | 3 | 2 | -1 |
| **Trajectory dim** | 9 `[act(3)+obs(6)]` | **TBD** — see §2.1 | — |
| **Cameras** | bp-cam + inhand-cam (dual) | bp-cam only (single) | -1 cam |
| **Image latent dim** | 128 (dual ResNet-64 cat) | **64** (single ResNet-64) | halved |

### 2.1 Avoiding Obs Dim Design Decision — *corrected by §12 audit (2026-06-02)*

> **CORRECTION**: The original version of this section claimed D3IL avoiding obs includes obstacle positions. **It does not.** D3IL's `avoiding_dataset.py:60-64` builds obs as `[des_c_pos(:,:2), c_pos(:,:2)]` = 4-D — identical pattern to aligning, just 2-D-sliced. The 6 obstacle positions are *fixed environment constants* (`get_obj_xy_list()` returns the same 6 positions on every reset), not per-episode state. See §12 audit for the diff between original and corrected obs decomposition.

**D3IL avoiding** state-only uses `obs_dim=4`: **`[des_x, des_y, c_x, c_y]`** — robot desired+current 2-D position. **No obstacle positions in obs.** Obstacles are environmental constants the policy must learn to dodge from training-set patterns (or, in the visual case, see in the bp-cam image).

**Our DPCC trajectory options (corrected):**

| Option | obs_dim | traj_dim | Notes |
|---|---|---|---|
| **A: D3IL-parity** | 4 `[des_xy(2), c_xy(2)]` | 6 `[act(2)+obs(4)]` | Matches D3IL exactly. Mirrors aligning's `[des_c_pos, c_pos]` structure scaled to 2-D. Horizon=8 ÷ traj=6: dim_mults work; final conv handles non-power-of-2 channel size. |
| **B: With explicit obstacle positions in obs** | 16 `[des_xy(2), c_xy(2), obs_xy(2)×6]` | 18 `[act(2)+obs(16)]` | Hardcodes the 6 known obstacle positions into the state vector. Only useful if positions varied per-episode (they don't — `get_obj_xy_list()` is static). Adds redundant capacity; not recommended. |
| **C: A + DPCC sphere_outside obstacle constraints** | 4 (same as A) | 6 (same as A) | Same trajectory as A, but the *planning* config adds `('sphere_outside', center=obs_xy, radius=R)` for each of the 6 obstacles. This is what `ObstacleConstraints` in `projection.py:80-81` already supports. **DPCC's actual selling point.** |

> **Recommendation (corrected)**: **Option C** — `obs_dim=4`, `traj_dim=6`, *plus* add `sphere_outside` constraints to the planning config for each of the 6 fixed obstacle positions. Option B (embed obstacle positions in obs) is redundant because positions are constants; Option A (no projector obstacle constraints) leaves DPCC's killer feature unused for the very task it was designed for.
>
> The dynamics constraint `c_pos[t+1] = c_pos[t] + Δt·des_c_pos[t]` still holds via Option A's obs layout — same as aligning, just 2-D.

---

## 3. What Gets Copied (4 Folders)

### Source → Destination mapping:

```
FM-PCC/
├── diffuser_visual_aligning/          ──COPY──►  diffuser_visual_avoiding/
├── diffuser_visual_aligning_test/     ──COPY──►  diffuser_visual_avoiding_test/
├── fm_visual_aligning/                ──COPY──►  fm_visual_avoiding/
└── fm_visual_aligning_test/           ──COPY──►  fm_visual_avoiding_test/
```

After copy, all internal `diffuser_visual_aligning` / `fm_visual_aligning` references within each folder must be updated to `diffuser_visual_avoiding` / `fm_visual_avoiding`.

---

## 4. Code Changes Per Folder (File-Level Checklist)

### 4.1 `diffuser_visual_avoiding/` (from `diffuser_visual_aligning/`)

#### 4.1.1 `datasets/sequence.py` — Dataset Loader

- [ ] **Rename** `ParityAligningDataset` → `ParityAvoidingDataset`
- [ ] **Change data paths**: `aligning/` → `avoiding/`
  - `train_files.pkl` / `eval_files.pkl` path → `avoiding/all_data/`
  - `state/` → `avoiding/all_data/state/` (via the symlink created by data collector)
- [ ] **Update obs extraction**: Replace `robot_des_pos`, `robot_c_pos` extraction with avoiding state keys
  - Avoiding state dict keys: `robot['des_c_pos'][:, :2]`, `robot['c_pos'][:, :2]`, obstacle positions
- [ ] **ACTION_DIM**: 3 → **2** (2D plane velocities)
- [ ] **OBS_DIM**: 6 → **TBD** (per §2.1 decision, likely 4 or 6)
- [ ] **TRAJ_DIM**: 9 → **ACTION_DIM + OBS_DIM**
- [ ] **Drop `inhand_cam` loading entirely** — single camera mode
  - Remove `self.inhand_cam_imgs` list
  - Remove `'wrist_img'` from conditions dict
  - Keep only `'primary_img'` (bp-cam)
- [ ] **Remove** `StateOnlyAligningDataset` class (not needed for visual avoiding in Epoch 2)

#### 4.1.2 `models/visual_unet.py` — VisualUNet

- [ ] **Update `TRANSITION_DIM`**: 9 → new traj_dim
- [ ] **Update `LATENT_DIM`**: 128 → **64** (single ResNet-64, no concatenation)
- [ ] **Single camera encoder**: Modify `shape_meta` to remove `in_hand_image` entry
  - Change MultiImageObsEncoder to single-image mode (only `agentview_image`)
  - Or use a simpler single-image encoder if MultiImageObsEncoder doesn't support single-image
- [ ] **Update `encode_visual()`**: Drop `inhand_imgs` parameter — only bp_imgs
- [ ] **Update `forward()`**: Unpack only `(bp_imgs, obs_seq)` from `cond['visual']` (remove inhand_imgs)
- [ ] **Fix import**: `from diffuser_visual_aligning.models...` → `from diffuser_visual_avoiding.models...`
- [ ] **Update action_dim default**: 3 → 2

#### 4.1.3 `models/visual_gaussian_diffusion.py`

- [ ] **Fix import**: Update any internal `diffuser_visual_aligning` → `diffuser_visual_avoiding`
- [ ] **Verify `apply_conditioning`**: Ensure obs anchor dimension matches new OBS_DIM

#### 4.1.4 `models/diffusion.py`, `helpers.py`, `unet1d_temporal_cond.py`

- [ ] **Import path updates only** — these are architecture-generic, no dim hardcodes expected

#### 4.1.5 `sampling/projection.py` — DPCC Projector

- [ ] **Update projector constraints**: Avoiding workspace bounds differ from aligning
  - Aligning: 3D workspace box constraints on c_pos (indices 6-8)
  - Avoiding: 2D workspace bounds on robot_xy (avoiding has different arena limits)
- [ ] **Update dynamics constraint indices**: The dynamics link `c_pos[t+1] = c_pos[t] + Δt·act[t]` must reference the correct indices for the new traj layout
- [ ] **Obstacle avoidance constraint** *(promoted by §12 audit from "optional" to **HIGH PRIORITY**)*: Add `('sphere_outside', center=obs_xy, radius=R)` for each of the **6 fixed obstacles** (positions from `d3il/.../avoiding_objects.py:get_obj_xy_list()`). This is what `ObstacleConstraints` in `sampling/projection.py:80-81` already implements — the existing class accepts `'sphere_outside'` constraint specs natively. Without these, DPCC's signature feature (constraint-aware projection around obstacles) is **inactive on the very task it was designed for**.

#### 4.1.6 `utils/` — Utility Files

- [ ] **Import path sweep**: `diffuser_visual_aligning` → `diffuser_visual_avoiding` in all `__init__.py`, `setup.py`, `training.py`
- [ ] **`constraints_helpers.py`**: Update workspace bounds if hardcoded for aligning

#### 4.1.7 `setup.py`

- [ ] Package name: `diffuser_visual_aligning` → `diffuser_visual_avoiding`

---

### 4.2 `fm_visual_avoiding/` (from `fm_visual_aligning/`)

**Identical change pattern** as §4.1, with these additional FM-specific items:

#### 4.2.1 `datasets/sequence.py`

- [ ] Same as §4.1.1 (rename, paths, dims, single camera)

#### 4.2.2 `models/visual_unet.py`

- [ ] Same as §4.1.2, but import from `fm_visual_avoiding.models...`

#### 4.2.3 `models/visual_gaussian_diffusion.py`

- [ ] This is `VisualFlowMatching` (not `VisualGaussianDiffusion`) — FM ODE engine
- [ ] Fix imports: `fm_visual_aligning` → `fm_visual_avoiding`
- [ ] Verify `apply_conditioning` obs anchor dim

#### 4.2.4 All other files

- [ ] Same import path sweep as §4.1

#### 4.2.5 `setup.py`

- [ ] Package name: `fm_visual_aligning` → `fm_visual_avoiding`

---

### 4.3 Test Folders (`*_test/`)

- [ ] Copy `diffuser_visual_aligning_test/` → `diffuser_visual_avoiding_test/`
- [ ] Copy `fm_visual_aligning_test/` → `fm_visual_avoiding_test/`
- [ ] Update all internal references
- [ ] Update test dataset paths to avoiding data

---

## 5. New Config Entries

### 5.1 `config/avoiding-d3il.py` — Add Visual Avoiding Blocks

> **IMPORTANT**: Two new train + plan pairs must be added to the existing `config/avoiding-d3il.py` (which already has non-visual avoiding configs).

#### 5.1.1 Training Config: `visual_avoiding_dpcc`

```python
'visual_avoiding_dpcc': {
    'model': 'diffuser_visual_avoiding.models.visual_unet.VisualUNet',
    'diffusion': 'diffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion',
    'action_dim': 2,            # 2D velocity: [dx, dy]
    'obs_dim': 4,               # [des_xy(2), c_xy(2)] — corrected per §12 audit (Option C)
    'if_vision': True,
    'horizon': 8,
    'n_diffusion_steps': 100,
    'action_weight': 10,
    'loss_type': 'l2',
    'dim': 32,
    'dim_mults': (1, 2, 4, 8),
    'condition_dropout': 0.1,
    'returns_condition': False,
    'max_path_length': 200,     # Avoiding episodes are shorter (~106 steps max)
    # ... standard training hyperparams ...
    'batch_size': 64,
    'learning_rate': 2e-4,
    'ema_decay': 0.995,
    'n_steps_per_epoch': 1000,
    'n_train_steps': 1e5,
    'gradient_accumulate_every': 2,
    'prefix': 'visual_avoiding_dpcc/',
    'exp_name': watch(args_to_watch_dpcc_train),
}
```

#### 5.1.2 Training Config: `fm_visual_avoiding`

```python
'fm_visual_avoiding': {
    'model': 'fm_visual_avoiding.models.visual_unet.VisualUNet',
    'diffusion': 'fm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching',
    'action_dim': 2,
    'obs_dim': 4,               # [des_xy(2), c_xy(2)] — corrected per §12 audit
    'if_vision': True,
    'horizon': 8,
    'time_beta_alpha_v3': 1.5,
    'time_beta_beta_v3': 1.0,
    'action_weight': 1,
    'loss_type': 'l2',
    'dim': 32,
    'dim_mults': (1, 2, 4, 8),
    'condition_dropout': 0.1,
    'returns_condition': False,
    'max_path_length': 200,
    'batch_size': 64,
    'learning_rate': 2e-4,
    'prefix': 'fm_visual_avoiding/',
    'exp_name': watch(args_to_watch_fm_visual_train),
}
```

#### 5.1.3 Planning Configs: `plan_visual_avoiding_dpcc` + `plan_fm_visual_avoiding`

Mirror of the aligning planning configs with:
- Updated model paths (`diffuser_visual_avoiding` / `fm_visual_avoiding`)
- `max_episode_length`: 200 (avoiding default)
- Updated `diffusion_loadpath` prefixes
- `if_vision: True`, `mpc_batch_size: 4`

### 5.2 `d3il/configs/avoiding_vision_config.yaml` — New D3IL Config

> **NOTE**: D3IL currently has `avoiding_config.yaml` (state-only) but **no** `avoiding_vision_config.yaml`. We need to create one, modeled on `aligning_vision_config.yaml`.

```yaml
defaults:
  - agents: ddpm_encdec_vision

agent_name: ddpm_encdec_vision
log_dir: logs/avoiding/

train_data_path: environments/dataset/data/avoiding/all_data/train_files.pkl
eval_data_path: environments/dataset/data/avoiding/all_data/eval_files.pkl

# Environment
obs_dim: 4        # [robot_x, robot_y, obs_x, obs_y] — single-camera visual
action_dim: 2     # [dx, dy] — 2D plane
max_len_data: 200
window_size: 8

# Dataset — uses visual dataset class
trainset:
  _target_: environments.dataset.avoiding_dataset.Avoiding_Img_Dataset
  ...

simulation:
  _target_: simulation.avoiding_sim.Avoiding_Sim
  if_vision: True
  ...
```

---

## 6. Single Camera Architecture Change

### 6.1 Why Single Camera

The visual aligning pipeline uses **dual cameras** (bp-cam + inhand-cam) because:
- bp-cam sees the workspace from above (box + target positions)
- inhand-cam sees the gripper contact point (grasp quality)

The avoiding task has **no grasping** — the robot dodges obstacles in a 2D plane. The inhand/wrist camera adds no information; only bp-cam captures obstacle positions.

### 6.2 Implementation: MultiImageObsEncoder → Single Image

**Current** (dual, LATENT_DIM=128):
```
bp_cam (3,96,96) ─► ResNet-64 ─┐
                                ├─cat──► 128D latent ──► FiLM
inhand_cam (3,96,96)─► ResNet-64─┘
```

**Target** (single, LATENT_DIM=64):
```
bp_cam (3,96,96) ─► ResNet-64 ──► 64D latent ──► FiLM
```

**Code change in `visual_unet.py`**:
```python
# Remove 'in_hand_image' from shape_meta:
shape_meta = {
    'obs': {
        'agentview_image': {'shape': [3, 96, 96], 'type': 'rgb'},
        # 'in_hand_image' removed — single camera for avoiding
    }
}

# LATENT_DIM = 64 (not 128)
```

The `MultiImageObsEncoder` should handle single-image input natively (it iterates over keys in `shape_meta['obs']`). Verify this during implementation.

---

## 7. Implementation Order

> **TIP**: Follow this exact order to minimize debugging time. Each step can be smoke-tested independently.

### Phase 1: Foundation (folders + imports)

| Step | Task | Est. Time |
|---|---|---|
| 1.1 | `cp -r diffuser_visual_aligning diffuser_visual_avoiding` | 1 min |
| 1.2 | `cp -r fm_visual_aligning fm_visual_avoiding` | 1 min |
| 1.3 | `cp -r diffuser_visual_aligning_test diffuser_visual_avoiding_test` | 1 min |
| 1.4 | `cp -r fm_visual_aligning_test fm_visual_avoiding_test` | 1 min |
| 1.5 | Global find-replace: `diffuser_visual_aligning` → `diffuser_visual_avoiding` in copied folders | 5 min |
| 1.6 | Global find-replace: `fm_visual_aligning` → `fm_visual_avoiding` in copied folders | 5 min |
| 1.7 | Update `setup.py` package names | 2 min |

### Phase 2: Dataset Loaders

| Step | Task | Est. Time |
|---|---|---|
| 2.1 | Audit avoiding env state dict keys (run a quick script to pickle-inspect) | 10 min |
| 2.2 | Decide obs_dim (§2.1) based on audit | 5 min |
| 2.3 | Rewrite `diffuser_visual_avoiding/datasets/sequence.py` → `ParityAvoidingDataset` | 20 min |
| 2.4 | Mirror changes in `fm_visual_avoiding/datasets/sequence.py` | 10 min |

### Phase 3: Model Architecture (Single Camera)

| Step | Task | Est. Time |
|---|---|---|
| 3.1 | Update `visual_unet.py` in both folders (single cam, new dims) | 15 min |
| 3.2 | Update `visual_gaussian_diffusion.py` in both folders | 10 min |
| 3.3 | Update `sampling/projection.py` for avoiding constraints | 20 min |

### Phase 4: Config Entries

| Step | Task | Est. Time |
|---|---|---|
| 4.1 | Add `visual_avoiding_dpcc` + `plan_visual_avoiding_dpcc` to `config/avoiding-d3il.py` | 15 min |
| 4.2 | Add `fm_visual_avoiding` + `plan_fm_visual_avoiding` to `config/avoiding-d3il.py` | 15 min |
| 4.3 | Create `d3il/configs/avoiding_vision_config.yaml` | 10 min |

### Phase 5: Smoke Test

| Step | Task | Est. Time |
|---|---|---|
| 5.1 | Import smoke test: `python -c "import diffuser_visual_avoiding"` | 2 min |
| 5.2 | Dataset smoke test: Load 5 episodes, verify tensor shapes | 10 min |
| 5.3 | Model forward pass smoke: Random input through VisualUNet, verify output shape | 10 min |
| 5.4 | Config parse smoke: `Parser().parse_args(experiment='visual_avoiding_dpcc')` | 5 min |

**Total estimated time: ~3 hours** (including audit and debugging)

---

## 8. Data Dependency

> **IMPORTANT**: The visual avoiding dataset must be already collected at:
> `d3il/environments/dataset/data/avoiding/all_data/`
> with `images/bp-cam/`, `images/inhand-cam/`, `train_files.pkl`, `eval_files.pkl`

The data collection is done via `collect_visual_avoiding_data/collect_visual_avoiding_data.py` (Gen9 Epoch 1). Confirm this is complete before starting Phase 2.

> **HARD-BLOCKER VERIFICATION (§12 audit)**: At time of audit (2026-06-02) the path `d3il/environments/dataset/data/avoiding/` **does not exist in the local repo** — only `aligning/`, `pushing/`, `sorting/`, `stacking/`. The avoiding data is either (a) cluster-only and gitignored (likely — `images/` are typically excluded from git), or (b) genuinely not collected yet. Before starting Phase 2, **run an SSH+ls on the cluster** at the target path to distinguish. If case (b), Gen9 Epoch 1's collection must be executed first (treat as a prerequisite epoch, not a "should be done" assumption).

Even though we collected **both** cameras during data collection, the visual avoiding pipeline will only load bp-cam images. The inhand-cam images remain on disk but are unused.

---

## 9. Risk Assessment & Known Gotchas

| Risk | Mitigation |
|---|---|
| `MultiImageObsEncoder` may not support single-image input | Test with single key in `shape_meta`; fallback: use a bare `get_resnet` directly |
| DPCC projector constraints wrong for avoiding workspace | Audit avoiding env workspace bounds from D3IL source before coding |
| `max_len_data=200` shorter episodes may cause dataset issues | Verify sliding window math with `horizon=8` on short episodes (200-8+1=193 windows max) |
| Latent dim 64 vs 128 may break FiLM layer size assumptions | Check `UNet1DTemporalCondModel` cond_dim parameter — it's configurable |
| Avoiding env doesn't export `des_c_pos` in the same format | Audit pickle keys from actual collected data |

---

## 10. Files NOT Modified (No Touch)

These existing files remain untouched:
- `diffuser_visual_aligning/` — original aligning code preserved
- `fm_visual_aligning/` — original aligning code preserved
- `config/aligning-d3il-visual.py` — aligning configs unchanged
- `scripts/train.py` — already generic (config-driven)
- `scripts/eval.py` — already generic (config-driven)
- `collect_visual_avoiding_data/` — data collection complete

---

## 11. Success Criteria

- [ ] `python scripts/train.py --dataset avoiding-d3il --config config.avoiding-d3il visual_avoiding_dpcc --seed 5 --num-seeds 1` trains without error
- [ ] `python scripts/train.py --dataset avoiding-d3il --config config.avoiding-d3il fm_visual_avoiding --seed 5 --num-seeds 1` trains without error
- [ ] Training loss decreases over 1000 steps
- [ ] Eval rollout in avoiding env produces non-trivial trajectories
- [ ] Single bp-cam image correctly fed through ResNet → 64D latent → FiLM → UNet

---

## 12. Audit (added 2026-06-02)

**Audit scope**: Cross-check every load-bearing claim in §§1–11 against the actual codebase. Sources read: `d3il/environments/d3il/envs/gym_avoiding_env/.../avoiding.py`, `.../avoiding_objects.py`, `d3il/configs/avoiding_config.yaml`, `d3il/environments/dataset/avoiding_dataset.py`, `diffuser_visual_aligning/models/visual_unet.py`, `diffuser_visual_aligning/sampling/projection.py`, `config/avoiding-d3il.py`, `collect_visual_avoiding_data/collect_visual_avoiding_data.py`, and the local filesystem inventory of `d3il/environments/dataset/data/`.

**Verdict**: Skeleton sound; **four substantive technical errors** in the original plan have been corrected inline. Two additional risks added.

### 12.1 Claims verified ✅ (unchanged)

| § | Claim | Evidence |
|---|---|---|
| §2 | D3IL avoiding `obs_dim=4`, `action_dim=2`, `max_len_data=200` | `d3il/configs/avoiding_config.yaml:45-47` |
| §2 | Aligning `LATENT_DIM=128` from dual ResNet-64 cat | `diffuser_visual_aligning/models/visual_unet.py:23` (literal comment `# dual ResNet-64 concatenated`) |
| §2 | Aligning trajectory_dim=9 = `[act(3)+obs(6)]` | `visual_unet.py:22 TRANSITION_DIM=9` |
| §3 | 4-folder copy-rename structure | Source paths exist; pattern is the same as Gen6/7 sync-into-Gen6V4 mirror commits |
| §4.1.5 | DPCC projector has `ObstacleConstraints`, `SafetyConstraints`, `DynamicConstraints` | `projection.py:63-90` |
| §6 | Avoiding has no grasping → wrist cam is uninformative | True by env construction; the wrist cam fixed-mounted geometry never sees the obstacle field |
| §10 | Generic scripts (`train.py`, `eval.py`) untouched | They've been config-driven since Gen5 |

### 12.2 Errors found and corrected ❌→✏️

#### Error 1 — Obs decomposition: D3IL avoiding obs does NOT include obstacle positions

**Original claim** (§2 table + §2.1 prose): D3IL `obs_dim=4` = `[robot_x, robot_y, obs_x, obs_y]` (obstacle in obs).

**Reality** (`d3il/environments/dataset/avoiding_dataset.py:60-64`):
```python
robot_des_pos = env_state['robot']['des_c_pos'][:, :2]   # [T, 2]
robot_c_pos   = env_state['robot']['c_pos'][:, :2]       # [T, 2]
input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)  # [T, 4]
```

D3IL avoiding obs is **`[des_xy(2), c_xy(2)]`** — robot-state only, the exact 2-D analogue of aligning's `[des_c_pos(3), c_pos(3)]`. **No obstacle positions appear in the obs vector.**

**Fix applied**: §2 table cell rewritten; §2.1 prose and option table rewritten; recommendation switched from "Option B (with obstacle in obs)" to **"Option C (D3IL-parity 4-D obs + projector sphere_outside constraints)"**.

#### Error 2 — Number of obstacles: 6, not 1

**Original claim** (§2.1 Option A/B implicit assumption): single `obs_xy(2)` for one obstacle.

**Reality** (`d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py:68-82`):
```python
def get_obj_xy_list():
    return [
        [0.500, -0.10],   # level 1: 1 obstacle
        [0.425,  0.08], [0.575,  0.08],   # level 2: 2 obstacles
        [0.350,  0.26], [0.500,  0.26], [0.650,  0.26],   # level 3: 3 obstacles
    ]
```

**Six obstacles** in a Christmas-tree layout. They are **fixed across all episodes** (no per-reset randomization) — so they're an environment constant, not state. Embedding them into the obs vector would (a) waste capacity on training-time constants and (b) require 12 extra dims (2-D × 6) not the 2 the plan implied.

**Fix applied**: §2.1 Option B reframed as "16-D with all 6 obstacles embedded (not recommended — positions are constants)". Option C added as the correct path: keep obs at 4-D and let the *planning config* declare the 6 obstacle positions as `sphere_outside` constraints.

#### Error 3 — Projector obstacle constraint marked "optional" but is DPCC's signature feature

**Original claim** (§4.1.5 last bullet): "Consider whether the projector should enforce minimum distance from obstacle positions (new feature, optional for Epoch 2)."

**Reality**:
- `diffuser_visual_aligning/sampling/projection.py:80-81` shows the projector already routes `'sphere_outside'`-typed `constraint_spec` entries into `ObstacleConstraints`. No new code is needed — just config entries.
- The avoiding task is *literally the task DPCC was designed to demonstrate*: dynamic-feasible projection around fixed obstacles. Skipping `sphere_outside` constraints reduces DPCC to "FM with a no-op projector," ablating its main contribution.

**Fix applied**: §4.1.5 bullet rewritten from "optional" to "**HIGH PRIORITY**", with the constraint spec format (`('sphere_outside', center, radius)`) and the obstacle-position source file cited.

#### Error 4 — Data prerequisite stated as "should be done" but unverified

**Original claim** (§8): "The visual avoiding dataset must be already collected... Confirm this is complete before starting Phase 2."

**Reality** (filesystem check at audit time):
```
$ ls d3il/environments/dataset/data/
aligning  pushing  sorting  stacking
```

**No `avoiding/` folder exists locally.** The `collect_visual_avoiding_data.py` script exists, but the collected output may be cluster-only (and gitignored, since image directories typically are). Or it may genuinely not have been run yet. The plan didn't distinguish these.

**Fix applied**: §8 promoted from "confirm" to a **HARD-BLOCKER VERIFICATION** note with an explicit cluster-side check instruction. Until the cluster path is verified, Phase 2 cannot start.

### 12.3 Risks added (not in original §9)

| Risk | Mitigation |
|---|---|
| Trajectory dim = 6 is not a power of 2; some U-Net `dim_mults=(1,2,4,8)` paths may produce odd channel sizes if the implementation assumes power-of-2 transition_dim | Aligning's traj_dim was 9 (also non-power-of-2) and worked — `Conv1dBlock` operates on the channel axis with no power-of-2 requirement. **Low risk, but smoke-test channel shapes in Phase 5.3.** |
| The "6 obstacles" assumption is brittle if D3IL ever randomizes `get_obj_xy_list()` between resets | Pin obstacle positions in the planning config to the values *at planning-config-creation time*; document the source-file reference so a regression review will catch the divergence. |
| `Avoiding_Dataset.__init__` declares `obs_dim=20` as the *buffer size* default but only fills indices 0-3 — this is a confusing upstream convention | Don't inherit the 20-default; our `ParityAvoidingDataset` should declare `obs_dim=4` as the actual data dim. **Don't confuse buffer size with actual obs dim** when reading any upstream config. |
| `dim=32` in §5.1.1/§5.1.2 may be too small for visual inputs (aligning's visual config uses `dim=64` or higher in some variants) | Verify against `config/aligning-d3il-visual.py` before launching first train; if aligning visual uses a larger `dim`, mirror that value. Low-risk smoke item. |

### 12.4 Items NOT changed (deemed acceptable)

- §3 copy-rename strategy — sound.
- §4 file-level checklist — sound (except §4.1.5 fixed above).
- §4.1.2 LATENT_DIM=64 with single ResNet-64 — sound; matches the dual=128 architecture exactly halved.
- §6 single-camera reasoning — sound.
- §7 phased implementation order — sound; time estimates are optimistic but order is right.
- §11 success criteria — sound (though "training loss decreases over 1000 steps" is a weak criterion; consider adding "eval success rate > random baseline" once data is available).

### 12.5 Recommended next actions before starting Phase 1

1. **SSH the cluster**, check `d3il/environments/dataset/data/avoiding/all_data/` actually exists with `train_files.pkl`, `eval_files.pkl`, and `images/bp-cam/`. If not, Gen9 Epoch 1 (data collection) must run first — treat as a real prerequisite epoch, not an assumption.
2. **Once data is confirmed**, run Phase 1 (copy + global find-replace + setup.py rename). Phase 1 is risk-free and can land before the obstacle-constraint design is finalized.
3. **Before Phase 4 (configs)**, write the 6-obstacle list into the planning config as a Python literal — pin the values from `avoiding_objects.py:68-82` and add a sphere radius (estimate ≈ obstacle physical radius + safety margin; verify by reading the avoiding XML for the actual MJ geom size).
4. **Smoke-test the projector with `sphere_outside` constraints active *before* training begins** — run a dummy trajectory through `Projector.project()` with the 6 obstacle constraints and a random input traj; verify the projection moves the trajectory away from obstacle centers as expected. This catches projector-side issues before they manifest as training divergence.

### 12.6 Audit verdict

**Plan is APPROVED to proceed with the four inline corrections above.** The skeleton (4 folders, copy-rename, config-driven, single-camera latent halving) is correct. The errors were all in the *task-specific dimensional and constraint details* — common when porting from one D3IL task to another — and have been corrected without restructuring the plan. **Phase 1 can start as soon as the §8 hard-blocker (data on cluster) is verified.**
