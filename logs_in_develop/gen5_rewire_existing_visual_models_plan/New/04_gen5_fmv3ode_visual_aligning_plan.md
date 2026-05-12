# 04 Gen5: FMv3ODE Visual Aligning Plan (12 May)

Date: 2026-05-12
Status: Draft
Depends on: 03_gen5_part1_rollback_execution_record.md (completed)

---

## 0) Background & Situation Summary

### What FM-PCC (FMv3ODE) is today
- A **pure trajectory/position model**.
- Data contract: loads pkl state files via `flow_matcher_v3_ode_selectable/datasets/d4rl.py` → `sequence_dataset()` → `avoiding-d3il` branch.
- Input: `observations = [robot_des_pos(2D), robot_c_pos(2D)]`, `actions = velocity`.
- UNet backbone: `Flow_matcher_U_Net_v2` in `flow_matcher_v3_ode_selectable/models/unet1d_temporal_cond.py`.
- Flow core: `GaussianDiffusion` in `flow_matcher_v3_ode_selectable/models/diffusion.py` (ODE-based flow matching, NOT DDPM).
- **No image data** is loaded, processed, or conditioned on.

### What D3IL visual aligning is today
- A **visual + state → action diffusion model** (DDPM, NOT flow matching).
- Agent: `ddpm_encdec_vision_agent.py` → `DiffusionPolicy` wraps:
  - `obs_encoder`: `MultiImageObsEncoder` (ResNet18 per camera → 64-dim feature each).
  - `model`: `DiffusionEncDec` (Transformer encoder-decoder with DDPM diffusion).
- Data contract: `Aligning_Img_Dataset` in `d3il/environments/dataset/aligning_dataset.py`.
  - Returns: `(bp_imgs, inhand_imgs, obs, act, mask)`.
  - Images: `bp-cam` and `inhand-cam` per trajectory step, shape `[C=3, H=96, W=96]`.
  - obs: `robot_des_pos` (3D).
  - action: `vel_state = des_pos[1:] - des_pos[:-1]` (3D).
- Simulation: `aligning_sim.py` with `if_vision=True` returns `(env_state, bp_image, inhand_image)`.
- Config: `configs/aligning_vision_config.yaml` sets `agent_name: ddpm_encdec_vision`, `if_vision: True`.
- Entrypoint: `run_vision.py` → calls `agent.train_vision_agent()`.

### The goal
Run **aligning** (a visual task) inside FM-PCC using FMv3ODE instead of DDPM.
This means:
1. **Phase 1 (Rewire)**: Take D3IL's working visual pipeline (dataset, encoder, sim) and wire it into FM-PCC so training/eval runs end-to-end — initially still with a DDPM-like core, as a control baseline.
2. **Phase 2 (Replace)**: Swap the DDPM diffusion core for FMv3ODE flow matching core, keeping the visual encoder and data pipeline unchanged.
3. **Phase 3 (Validate)**: Run aligning benchmarks and compare FMv3ODE-visual vs DDPM-visual to prove the model actually uses image conditioning.

---

## 1) Architecture Gap Analysis (Code-Grounded)

### 1.1 Data Pipeline Gap

| Aspect | D3IL Visual (Working) | FM-PCC FMv3ODE (Current) |
|---|---|---|
| Dataset class | `Aligning_Img_Dataset` in `d3il/environments/dataset/aligning_dataset.py` | `SequenceDataset` in `flow_matcher_v3_ode_selectable/datasets/sequence.py` via `d4rl.py` |
| Data source | pkl state + image folders `all_data/images/bp-cam/`, `all_data/images/inhand-cam/` | pkl state files only `data/avoiding/data/` |
| Returns | `(bp_imgs, inhand_imgs, obs, act, mask)` — 5-tuple | `(observations, actions, rewards, terminals)` dict → sequence windows |
| Image dims | `[B, T, 3, 96, 96]` per camera | None |
| Obs dims | 3 (robot_des_pos) | 4 (robot_des_pos + robot_c_pos, 2D each) |
| Action dims | 3 (velocity 3D) | 2 (velocity 2D) |
| Task | aligning (push box to target) | avoiding (obstacle avoidance) |

### 1.2 Model Architecture Gap

| Aspect | D3IL Visual (Working) | FM-PCC FMv3ODE (Current) |
|---|---|---|
| Obs encoder | `MultiImageObsEncoder` (2× ResNet18 → 64-dim each → concat → 128-dim) | None (raw state concatenated into trajectory) |
| Generative core | DDPM `Diffusion` with `DiffusionEncDec` (Transformer Enc-Dec) | Flow Matching `GaussianDiffusion` with `Flow_matcher_U_Net_v2` (1D UNet) |
| Conditioning | obs_encoder output (128-dim) injected as conditioning into Transformer | State vector concatenated into trajectory tensor, conditioning via `apply_conditioning()` |
| Training loss | DDPM ε-prediction loss | Flow Matching velocity field loss |
| Sampling | DDPM reverse process (16 steps cosine schedule) | ODE forward integration t=0→1 (configurable solver) |

### 1.3 Runtime / Eval Gap

| Aspect | D3IL Visual | FM-PCC FMv3ODE |
|---|---|---|
| Entrypoint | `run_vision.py` (Hydra) | `train_flow_matching_v3_ode_selectable.py` / `eval_flow_matching_v3_ode_selectable.py` (argparse + custom Parser) |
| Sim class | `Aligning_Sim` with `if_vision=True` | `ObstacleAvoidanceEnv` (state-only) |
| Predict API | `agent.predict((bp_img, inhand_img, des_robot_pos), if_vision=True)` | `policy(conditions={0: obs}, batch_size=B, horizon=H)` |

---

## 2) Execution Strategy: Three Phases

### Phase 1: Rewire D3IL Visual Pipeline into FM-PCC (Control Baseline)

**Goal**: Get D3IL's DDPM visual aligning running inside the FM-PCC repo so we have a working control that proves the visual pipeline is healthy.

**Scope**: **Copy-Modify, NOT create from scratch.** All new folders are full copies of their proven originals, then surgically modified. No D3IL originals are touched.

**Safety principle**: Copy first → verify copy works identically → then modify. This guarantees we always have a working rollback point.

---

#### Step 1.1: Copy engine folder — `flow_matcher_v3_ode_selectable/` → `ddpm_encdec_vision/`

- **Action**: `cp -r flow_matcher_v3_ode_selectable/ ddpm_encdec_vision/`
- **Naming convention**: Follows D3IL's original naming — `ddpm_encdec_vision` matches `ddpm_encdec_vision_agent.py` and `ddpm_encdec_vision_agent.yaml` in D3IL.
- **Result**: `ddpm_encdec_vision/` is a complete, working copy of the FMv3ODE engine.
- **Verification**: Before any modification, ensure the copy is byte-identical to the original (`diff -r`).

#### Step 1.2: Create bridging module — `ddpm_encdec_vision/models/d3il_visual_bridge.py`

- **New file** (inside the copied folder): `ddpm_encdec_vision/models/d3il_visual_bridge.py`
- **Purpose**: A thin bridging layer that imports and directly uses D3IL's visual diffusion components without duplicating them. This is the **single point of integration** between FM-PCC's engine and D3IL's visual backbone.
- **What it imports from D3IL** (read-only, no modifications to D3IL):
  - `d3il/agents/ddpm_encdec_vision_agent.py` → `DiffusionPolicy` (obs encoder + diffusion model wrapper)
  - `d3il/agents/models/vision/multi_image_obs_encoder.py` → `MultiImageObsEncoder` (ResNet18 × 2 → 128-dim)
  - `d3il/agents/models/diffusion/diffusion_policy.py` → `Diffusion` (DDPM core with cosine schedule)
  - `d3il/agents/models/diffusion/diffusion_models.py` → `DiffusionEncDec` (Transformer Encoder-Decoder backbone)
- **Key API exposed**:
  ```python
  class VisualDiffusionBridge:
      """Bridges D3IL's visual DDPM into FM-PCC's engine structure."""
      def __init__(self, config):  # config from aligning-d3il-visual.py
          self.obs_encoder = MultiImageObsEncoder(...)  # 2× ResNet18 → 128-dim
          self.diffusion_model = Diffusion(
              model=DiffusionEncDec(...),  # Transformer Enc-Dec
              state_dim=128, action_dim=3, ...
          )
      def encode_visual(self, bp_imgs, inhand_imgs, state):
          """[B,T,3,96,96] × 2 + [B,T,3] → [B,T,128]"""
      def loss(self, visual_emb, action):
          """Training loss using D3IL's DDPM."""
      def predict(self, visual_emb):
          """Inference using D3IL's DDPM."""
  ```
- **Rationale**: Best approach per user advice — directly use D3IL's visual diffusion rather than rebuilding it. Minimal code surface, maximum reuse.

#### Step 1.3: Modify copied engine internals

- **Files to modify** (inside `ddpm_encdec_vision/`, NOT the original `flow_matcher_v3_ode_selectable/`):
  - `ddpm_encdec_vision/models/__init__.py` — Add import for `d3il_visual_bridge`
  - `ddpm_encdec_vision/models/diffusion.py` — Replace FMv3ODE `GaussianDiffusion` backbone with call-through to `VisualDiffusionBridge`
  - `ddpm_encdec_vision/datasets/d4rl.py` — Modify `sequence_dataset()` to load `Aligning_Img_Dataset` from `d3il/environments/dataset/aligning_dataset.py` instead of the avoiding pkl data.
  - `ddpm_encdec_vision/__init__.py` — Update package name references
  - `ddpm_encdec_vision/utils/setup.py` — Update package name if needed
- **Files NOT modified** (kept from copy, used as-is for infrastructure):
  - `ddpm_encdec_vision/sampling/` — Policy sampling infrastructure
  - `ddpm_encdec_vision/utils/` — Config, Parser, Trainer utilities (mostly unchanged)
  - `ddpm_encdec_vision/datasets/normalization.py` — Data normalization
  - `ddpm_encdec_vision/datasets/sequence.py` — Sequence windowing logic (adapted for visual)

#### Step 1.4: Create config — `config/aligning-d3il-visual.py`

- **Action**: Copy `config/avoiding-d3il.py` → `config/aligning-d3il-visual.py`, then modify.
- **Key changes from `avoiding-d3il.py`**:
  - **Task**: `aligning` (not `avoiding`)
  - **Engine reference**: Points to `ddpm_encdec_vision` package instead of `flow_matcher_v3_ode_selectable`
  - **Model**: `ddpm_encdec_vision.models.d3il_visual_bridge.VisualDiffusionBridge`
  - **Dataset**: `ddpm_encdec_vision.datasets.Aligning_Img_Dataset` wrapper
  - **Dims**: `obs_dim=128` (encoded), `action_dim=3`, `horizon=8`, `window_size=8`
  - **Visual-specific params**: `visual_input=True`, `obs_seq_len=5`, `action_seq_size=4` (matching D3IL's `ddpm_encdec_vision_agent.yaml`)
  - **Training config blocks**: `ddpm_encdec_vision` (train) + `plan_ddpm_encdec_vision` (eval)
  - **Serialization prefix**: `ddpm_encdec_vision/` and `plans/ddpm_encdec_vision/`
- **args_to_watch**: New tuple for visual aligning experiment naming

#### Step 1.5: Copy test entry folder — `FM_v3_ode_selectable_test/` → `ddpm_encdec_vision_test/`

- **Action**: Create new folder `ddpm_encdec_vision_test/` by copying the 3 entry scripts from `FM_v3_ode_selectable_test/`:
  ```
  cp FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py  → ddpm_encdec_vision_test/train_ddpm_encdec_vision.py
  cp FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py   → ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py
  cp FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py → ddpm_encdec_vision_test/load_results_ddpm_encdec_vision.py
  ```
- **Note**: `Benchmark_ode_solver_Tests/` subfolder is NOT copied (not relevant for visual DDPM).

#### Step 1.6: Modify copied entry scripts

##### `ddpm_encdec_vision_test/train_ddpm_encdec_vision.py`
- **Source**: Copy of `train_flow_matching_v3_ode_selectable.py`
- **Changes**:
  - `import ddpm_encdec_vision.utils as utils` (was `flow_matcher_v3_ode_selectable.utils`)
  - `exp = 'aligning-d3il-visual'` (was `'avoiding-d3il'`)
  - `Parser.config = 'config.aligning-d3il-visual'`
  - `experiment='ddpm_encdec_vision'` in `Parser().parse_args()`
  - Dataset loading: Uses `Aligning_Img_Dataset` wrapper via the new engine
  - Model config: Instantiates `VisualDiffusionBridge` via config
  - Training loop: Handles 5-tuple `(bp_imgs, inhand_imgs, obs, act, mask)`, calls `bridge.encode_visual()` → `bridge.loss()`
  - W&B naming: Updated to reflect `ddpm_encdec_vision` identity

##### `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`
- **Source**: Copy of `eval_flow_matching_v3_ode_selectable.py`
- **Changes**:
  - `import ddpm_encdec_vision.utils as utils`
  - `exp = 'aligning-d3il-visual'`
  - Simulation: Uses `Aligning_Sim` from `d3il/simulation/aligning_sim.py` with `if_vision=True`
  - Predict loop:
    1. Reset env → get `(env_state, bp_image, inhand_image)`
    2. Call `bridge.encode_visual(bp_image, inhand_image, des_robot_pos)` → 128-dim embedding
    3. Call `bridge.predict(embedding)` → action prediction
    4. Apply action: `pred_action = pred[0] + des_robot_pos`, append `[0, 1, 0, 0]` quaternion
    5. Step env
  - Metrics: success rate, entropy, mean distance (same as D3IL `aligning_sim.py`)

##### `ddpm_encdec_vision_test/load_results_ddpm_encdec_vision.py`
- **Source**: Copy of `load_results_flow_matching_v3_ode_selectable.py`
- **Changes**:
  - Updated paths/prefixes for `ddpm_encdec_vision` logs
  - Plot output directory: `logs/aligning-d3il-visual/load_results_output_all_seeds/`

---

### Phase 2: Replace DDPM Core with FMv3ODE

**Goal**: Swap the generative core from DDPM to FMv3ODE while keeping the visual pipeline from Phase 1.

**What changes**: Only the `VisualDiffusionBridge` internals. The obs_encoder, dataset, and sim remain identical.

**Method**: Copy-modify again — copy `ddpm_encdec_vision/` → `fm_v3_ode_vision/`, then swap the DDPM core for FMv3ODE.

#### Step 2.1: Verify Phase 1 control baseline produces sensible results

- Run at least one aligning visual training + eval using `ddpm_encdec_vision_test/`.
- Record success rate and entropy as baseline.
- Confirm the bridge is correctly forwarding to D3IL's DDPM.

#### Step 2.2: Copy-modify for FMv3ODE core swap

- Copy `ddpm_encdec_vision/` → `fm_v3_ode_vision/`
- In `fm_v3_ode_vision/models/d3il_visual_bridge.py`:
  - Keep `obs_encoder` (MultiImageObsEncoder) — unchanged
  - Replace `Diffusion` (DDPM) with `GaussianDiffusion` (FMv3ODE flow matching) from the original `flow_matcher_v3_ode_selectable/models/diffusion.py`
  - Replace `DiffusionEncDec` (Transformer) with `Flow_matcher_U_Net_v2` (1D UNet) as backbone
  - Training loss: flow matching velocity field loss (not DDPM ε-prediction)
  - Sampling: ODE forward integration t=0→1 (not DDPM reverse)

#### Step 2.3: Copy-modify entry scripts

- Copy `ddpm_encdec_vision_test/` → `fm_v3_ode_vision_test/`
- Update imports and config references to point to `fm_v3_ode_vision`
- Config: `config/aligning-d3il-visual-fm.py` (copy-modify from `aligning-d3il-visual.py`)

### Phase 3: Validation

#### Step 3.1: Image sensitivity test

- Perturb images (e.g., zero out one camera, add noise).
- Compare predicted actions with vs without perturbation.
- **Pass criteria**: Action predictions change meaningfully under image perturbation.
- **Fail criteria**: Actions are identical regardless of image content → model ignores images.

#### Step 3.2: Benchmark comparison

- Compare FMv3ODE-visual vs D3IL DDPM-visual on aligning task.
- Metrics: success rate, entropy, mean distance.

---

## 3) Concrete File Plan

### 3.1 New folders and files (copy-modify approach, no D3IL edits)

#### Engine folder: `ddpm_encdec_vision/` (copied from `flow_matcher_v3_ode_selectable/`)

| # | File | Action | Purpose |
|---|---|---|---|
| 1 | `ddpm_encdec_vision/` (entire folder) | COPY from `flow_matcher_v3_ode_selectable/` | Full engine copy as safety baseline |
| 2 | `ddpm_encdec_vision/models/d3il_visual_bridge.py` | NEW | Bridging module to directly use D3IL's visual DDPM |
| 3 | `ddpm_encdec_vision/models/__init__.py` | MODIFY | Add `d3il_visual_bridge` import |
| 4 | `ddpm_encdec_vision/models/diffusion.py` | MODIFY | Wire through to `VisualDiffusionBridge` |
| 5 | `ddpm_encdec_vision/datasets/d4rl.py` | MODIFY | Load `Aligning_Img_Dataset` for visual aligning data |
| 6 | `ddpm_encdec_vision/__init__.py` | MODIFY | Update package references |

#### Config file

| # | File | Action | Purpose |
|---|---|---|---|
| 7 | `config/aligning-d3il-visual.py` | NEW (copy-modify from `avoiding-d3il.py`) | Config for visual aligning linking to `ddpm_encdec_vision` engine |

#### Test entry folder: `ddpm_encdec_vision_test/` (copied from `FM_v3_ode_selectable_test/`)

| # | File | Action | Purpose |
|---|---|---|---|
| 8 | `ddpm_encdec_vision_test/train_ddpm_encdec_vision.py` | COPY-MODIFY from `train_flow_matching_v3_ode_selectable.py` | Training entry point for visual aligning |
| 9 | `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | COPY-MODIFY from `eval_flow_matching_v3_ode_selectable.py` | Eval entry point with `Aligning_Sim(if_vision=True)` |
| 10 | `ddpm_encdec_vision_test/load_results_ddpm_encdec_vision.py` | COPY-MODIFY from `load_results_flow_matching_v3_ode_selectable.py` | Results loading and plotting |

### 3.2 Files read (imported) but NOT modified

| File | Reason |
|---|---|
| `d3il/agents/ddpm_encdec_vision_agent.py` | Source of `DiffusionPolicy` class, imported via bridge |
| `d3il/agents/models/diffusion/diffusion_policy.py` | Source of `Diffusion` (DDPM core), imported via bridge |
| `d3il/agents/models/diffusion/diffusion_models.py` | Source of `DiffusionEncDec` (Transformer Enc-Dec), imported via bridge |
| `d3il/agents/models/vision/multi_image_obs_encoder.py` | Source of `MultiImageObsEncoder`, imported via bridge |
| `d3il/agents/models/vision/model_getter.py` | Source of `get_resnet`, imported via encoder |
| `d3il/environments/dataset/aligning_dataset.py` | Source of `Aligning_Img_Dataset` class, imported for data loading |
| `d3il/simulation/aligning_sim.py` | Source of `Aligning_Sim`, used for eval |
| `d3il/configs/agents/ddpm_encdec_vision_agent.yaml` | Reference for hyperparameters (state_dim=128, n_timesteps=16, etc.) |
| `d3il/configs/aligning_vision_config.yaml` | Reference for task config (obs_dim=3, action_dim=3, window_size=8) |

### 3.3 Original files preserved (NOT modified, serve as rollback reference)

| File | Role |
|---|---|
| `flow_matcher_v3_ode_selectable/` (entire folder) | Original FMv3ODE engine — untouched, used as copy source |
| `FM_v3_ode_selectable_test/` (entire folder) | Original test entry scripts — untouched, used as copy source |
| `config/avoiding-d3il.py` | Original config — untouched, used as copy-modify source |

### 3.4 D3IL data assets required

| Path | Content |
|---|---|
| `d3il/environments/dataset/data/aligning/train_files.pkl` | Training file list |
| `d3il/environments/dataset/data/aligning/eval_files.pkl` | Eval file list |
| `d3il/environments/dataset/data/aligning/all_data/state/` | State pkl files |
| `d3il/environments/dataset/data/aligning/all_data/images/bp-cam/` | Bird's-eye camera images |
| `d3il/environments/dataset/data/aligning/all_data/images/inhand-cam/` | In-hand camera images |
| `d3il/environments/dataset/data/aligning/test_contexts.pkl` | Test contexts for sim |
| `d3il/environments/dataset/data/aligning/train_contexts.pkl` | Train contexts for sim |

---

## 4) Key Technical Decisions

### 4.1 Conditioning strategy: Bridging over rebuilding

**Decision (Phase 1)**: Use D3IL's native DDPM conditioning via `VisualDiffusionBridge`. The bridge imports and directly calls D3IL's `DiffusionPolicy.forward()` which internally handles `obs_encoder → DiffusionEncDec`. No custom conditioning logic needed.

**Decision (Phase 2)**: When swapping to FMv3ODE, inject visual embeddings via `apply_conditioning()`, replacing raw state in the trajectory tensor. The UNet processes `[B, transition_dim, H]` tensors with `transition_dim = action_dim + visual_embed_dim`.

**Rationale**: Phase 1 copy-modify + bridge approach is safest — directly reuse proven D3IL code, minimize new code surface. Phase 2 swaps only the generative core, keeping the visual encoder unchanged.

**Alternative considered**: Build new conditioning from scratch in Phase 1. Rejected — too much debug surface, violates copy-modify safety principle.

### 4.2 Task scope: Aligning only, NOT Avoiding

**Decision**: Build and validate on Aligning task first.

**Rationale**:
1. Aligning has working visual data + pipeline in D3IL.
2. Avoiding does NOT have visual data/pipeline (rollback completed in Part 1).
3. Only after Aligning visual proof succeeds, extend to Avoiding visual (Phase 3+).

### 4.3 Obs encoder: Reuse D3IL's, don't rebuild

**Decision**: Import `MultiImageObsEncoder` + `get_resnet` from vendored D3IL.

**Rationale**: Known working, reduces debug surface. The encoder produces a fixed 128-dim embedding per timestep, which is a clean interface to swap what's downstream.

---

## 5) Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| D3IL visual data assets not present in FM-PCC/d3il | Blocks dataset loading | Check data existence before starting Phase 1. If missing, copy from /workspaces/d3il |
| UNet transition_dim mismatch with visual embedding size | Training crash | Explicitly configure in `aligning-d3il-visual.py`, verify dims before training |
| `apply_conditioning()` incompatible with 128-dim embedding | Silent conditioning failure | Unit test: verify conditioning is actually injected into trajectory tensor |
| Visual encoder weights not training (frozen accidentally) | Model ignores images | Check `requires_grad=True` on encoder params, monitor encoder gradient norm |
| Aligning sim env not importable from FM-PCC | Eval crash | Verify import path resolution for `d3il/simulation/aligning_sim.py` |

---

## 6) Prerequisite Checks (Before Phase 1 Starts)

1. [ ] Verify `d3il/environments/dataset/data/aligning/` data exists with images.
2. [ ] Verify `d3il/agents/models/vision/multi_image_obs_encoder.py` is importable from FM-PCC.
3. [ ] Verify `d3il/simulation/aligning_sim.py` is importable from FM-PCC.
4. [ ] Verify D3IL's `run_vision.py` aligning visual runs end-to-end (control smoke test).
5. [ ] Record D3IL aligning visual baseline metrics.

---

## 7) Definition of Done

Gen5-04 is complete when:
1. Phase 1 control baseline (DDPM-visual wired into FM-PCC) runs and produces aligning results.
2. Phase 2 FMv3ODE-visual runs on aligning task.
3. Phase 3 image sensitivity test proves visual conditioning is real.
4. Benchmark comparison table is recorded.
5. All new code is in isolated files (no D3IL originals modified).
6. Execution record is written in `logs_in_develop/gen5_rewire_existing_visual_models_plan/12_May/`.
