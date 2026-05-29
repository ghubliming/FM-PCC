# Visual Aligning: Architecture, Math, and D3IL Comparison

**Scope**: Gen7 FM-PCC and Gen6V4 DPCC — visual (image + state) aligning pipeline  
**Date**: 2026-05-28  
**Companion doc**: [`NON_VISUAL_EXPL/NON_VISUAL_ALIGNING_ARCHITECTURE.md`](../NON_VISUAL_EXPL/NON_VISUAL_ALIGNING_ARCHITECTURE.md)

---

## 1. What "Visual Aligning" Means in This Codebase

The visual aligning pipeline (`if_vision=True`) is the **primary operating mode** for both
Gen7 FM-PCC and Gen6V4 DPCC. The robot receives both camera images and a compact state
vector and must learn to push a box to a target pose.

| Mode | `if_vision` | Input to model | Cameras |
|---|---|---|---|
| **Visual** (primary) | `True` | bp-cam image + inhand-cam image + 6D state | EGL offscreen |
| Non-visual (ablation) | `False` | 20D flat state only | BLIND render |

The visual mode is primary because:
1. It matches how physical robots perceive the world — position sensors are noisy; cameras provide direct task-relevant context (box location, target marker)
2. D3IL paper visual agents are the canonical academic baseline for fair comparison
3. The FM-PCC/DPCC constraint projector operates in Cartesian robot space regardless of modality; visual conditioning does not add projection complexity

---

## 2. Observation Space

### 2.1 Camera Observations

Two cameras are active during visual rollouts:

| Camera | Key | Resolution | Content |
|---|---|---|---|
| Bird's-eye (`bp-cam`) | `agentview_image` | 96 × 96, RGB | Top-down view of workspace, box, target |
| In-hand (`inhand-cam`) | `in_hand_image` | 96 × 96, RGB | Wrist-mounted view of gripper and box |

Images are returned by `Robot_Push_Env.get_visual_obs()` as `(C, H, W)` float32 arrays
in BGR channel order (OpenCV convention). The eval wrapper converts to RGB at capture time
for GIF/MP4 writing but passes BGR directly to the model — the ResNet encoder was trained
with ImageNet normalization which is channel-agnostic in its learned features.

### 2.2 State Observation — 6D

At evaluation the visual path uses a **6D state vector**:

```
obs_6d[0:3] = des_c_pos   (commanded TCP position, x/y/z)
obs_6d[3:6] = c_pos        (actual TCP position from sim, x/y/z)
────────────────────────────────────
Total: 6D
```

This is a strict subset of the 20D state used in the non-visual path. The visual path
drops box position, box quaternion, target position, and target quaternion from the
explicit state — the ResNet encoder extracts that information from the images directly.

Why `des_c_pos` AND `c_pos` instead of just one?

- `des_c_pos` is the **commanded** position sent to the low-level controller (what the
  policy chose to do at the last step)
- `c_pos` is the **actual** position achieved (what the robot reached given controller lag)
- The gap between them encodes **accumulated tracking error**, which the model learns to
  reason about when planning the next action

### 2.3 The 6D at Training vs Evaluation

**Training** (`ParityAligningDataset`):

```python
robot_des_pos = env_state['robot']['des_c_pos']   # (T+1, 3) — commanded positions
robot_c_pos   = env_state['robot']['c_pos']       # (T+1, 3) — actual positions

obs_6d  = np.concatenate([robot_des_pos[:T], robot_c_pos[:T]], axis=-1)   # (T, 6)
actions = (robot_des_pos[1:] - robot_des_pos[:-1]).astype(np.float32)     # (T, 3)
```

Both `des_c_pos` and `c_pos` are read directly from the demonstration pickle.

**Evaluation** (`VisualAgentWrapper.predict()`):

```python
bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state   # sim provides both

obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # [des_c_pos | c_pos]
```

`Robot_Push_Env.get_visual_obs()` returns four items: the two images plus two distinct
position vectors. The eval wrapper assembles the same 6D structure the model was trained on.

There is **no `mental_robot_pos` bridge** in the visual path — unlike the non-visual path,
the sim directly provides a separate `des_robot_pos` (= last commanded position, maintained
by the env's internal desired-position tracker) and `robot_pos` (= actual MuJoCo state).

---

## 3. Trajectory Representation — 9D

The model does not operate on raw 3D actions alone. It generates full **9D trajectories**
that concatenate actions and observations:

```
x[h, :] = [ dx   dy   dz  |  des_x  des_y  des_z  |  x    y    z  ]
             action (3D)      des_c_pos (3D)           c_pos (3D)
             indices 0-2      indices 3-5              indices 6-8
```

**Why include obs in the trajectory?**

The DDPM/FM generative model learns the **joint distribution** of (actions, observations)
over the horizon. Including obs in the trajectory lets the model:

1. Check self-consistency: the predicted `c_pos[t+1]` should equal `c_pos[t] + action[t]`
   (Euler dynamics). The projector enforces this via a `deriv` constraint.
2. Generate feasible trajectories: the model can reason about whether a planned action
   sequence keeps the robot inside workspace bounds not just at step `t` but across the
   entire `H`-step horizon.
3. Apply geometric constraints during denoising/ODE integration via the SLSQP projector.

**Actions only (3D) at execution:** The 9D trajectory is generated; only indices `0:3`
are executed. The obs dimensions `3:9` are used by the projector and for diagnostics but
are discarded after the action chunk is extracted.

---

## 4. Action Space

### 4.1 Policy output: 3D velocity

```
action[0] = dx = des_c_pos_x[t+1] - des_c_pos_x[t]
action[1] = dy = des_c_pos_y[t+1] - des_c_pos_y[t]
action[2] = dz = des_c_pos_z[t+1] - des_c_pos_z[t]
────────────────────────────────────────────────────
Policy action_dim: 3
```

All three spatial dimensions are predicted. In practice `dz ≈ 0` throughout the push
task (robot slides horizontally), but the model is not constrained to zero the z component.
This is the key difference from the non-visual path which explicitly uses 2D actions.

### 4.2 Execution: 7D pose command

The 3D velocity output is promoted to a full 7D pose command for the MuJoCo controller:

```python
mental_robot_pos += next_action_np          # integrate velocity into position
gripper_quat = [0, 1, 0, 0]               # fixed horizontal orientation
action_7d = np.concatenate([mental_robot_pos, gripper_quat])
env.step(action_7d)
```

The `mental_robot_pos` here tracks the *desired* position (integrated commands), distinct
from the actual sim position `robot_pos_np`. Both are fed back into `obs_6d` at the next
step.

---

## 5. Model Architecture

### 5.1 Vision Encoder — MultiImageObsEncoder

```
bp_img_seq     (B, T_win, 3, 96, 96)     ─► ResNet-64 ─► (B*T_win, 64)
inhand_img_seq (B, T_win, 3, 96, 96)     ─► ResNet-64 ─► (B*T_win, 64)

Concat per frame: (B*T_win, 128)
Reshape + mean-pool over window: (B, 128)   ← FiLM conditioning vector
```

Implementation details:
- Two **separate ResNet backbones** (no weight sharing — `share_rgb_model=False`)
- Each ResNet outputs a **64D latent** per image; concatenation yields **128D**
- Mean-pooling over the temporal window `T_win` (= `window_size` = 1 in this config)
  collapses the window before FiLM injection — zero-padded frames never dilute the signal
- ImageNet normalization applied (`imagenet_norm=True`)
- Group normalization (not BatchNorm) for stability at small batch sizes
- Instantiated via Hydra `MultiImageObsEncoder` from the D3IL agent library

> Note on `window_size=1`: `ParityAligningDataset` provides single-frame images per
> sample, so the model trains on `T_win=1`. Using `window_size>1` at eval would
> mean-pool over multiple frames and shift the FiLM conditioning distribution.
> Both train and eval scripts enforce `obs_seq_len=1`.

### 5.2 Temporal U-Net Backbone — UNet1DTemporalCondModel

```
Input:  noisy trajectory x ∈ ℝ^{B × H × 9}    (H=8, padded to 8 internally)
Cond:   visual_cond ∈ ℝ^{B × 128}              (from ResNet encoder above)

Architecture: UNet1D with temporal downsampling
  dim=128, dim_mults=(1,2,4,8) → channel dims: 128, 256, 512, 1024
  3 stride-2 downsampling levels → requires H divisible by 8
  Conditioning: FiLM gates on visual_cond at each resolution level
  condition_dropout=0.1 (classifier-free guidance compatible)
  use_cond_projection=True (visual mode)

Output: denoised/velocity field ∈ ℝ^{B × H × 9}
```

The backbone processes the 9D trajectory as a **1D temporal signal** of length `H=8`,
with the 9 features as the "channel" dimension. FiLM (Feature-wise Linear Modulation)
injects the 128D visual conditioning vector at each spatial resolution.

### 5.3 Full Forward Pass (VisualUNet)

```python
def forward(self, x, cond, t, ...):
    # 1. Encode visual context (BEFORE padding — avoids dilution by zero frames)
    bp_imgs, inhand_imgs, _ = cond['visual']
    visual_cond = self.encode_visual(bp_imgs, inhand_imgs)   # (B, 128)

    # 2. Pad trajectory to multiple of 8
    B, T, D = x.shape      # x: (B, 8, 9) — already valid; pad if H < 8
    if T < self.padded_horizon:
        x = torch.cat([x, x.new_zeros(B, pad, D)], dim=1)

    # 3. Run U-Net with FiLM conditioning
    out = self.backbone(x, visual_cond, t, ...)   # (B, padded_H, 9)

    return out[:, :T, :]    # trim padding
```

Note: `config.obs_dim` is intentionally **ignored** by `VisualUNet`. It is often set to
a stale placeholder (`128`) in legacy configs. The visual path hardcodes
`TRANSITION_DIM = 9` = act(3) + obs(6). This was a key lesson from fix_5.

---

## 6. MPC Planning

Both Gen7 (FM) and Gen6V4 (DPCC) operate as **MPC controllers**: they plan a full
`H=8` horizon and execute a chunk before replanning.

### 6.1 Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `horizon` | 8 | Full planning horizon (timesteps) |
| `obs_seq_len` | 1 | Observation window fed to model (single frame) |
| `action_seq_size` | 4 | Steps executed before replanning (chunk size) |
| `mpc_batch_size` | 4 (FM) / 1 (DPCC) | Candidate trajectories per replan |
| `max_episode_length` | 400 | Hard rollout cutoff |

Window equation: `window_size = obs_seq_len + action_seq_size - 1 = 4`
(but `obs_seq_len=1` means the model only ever sees the current frame).

### 6.2 Replan Loop

```
env.reset(if_vision=True)  →  (obs_6d, bp_img, inhand_img)
action_counter = action_seq_size    # force replan on step 0

loop (up to 400 steps):
    obs_6d_norm ← normalize(des_robot_pos, robot_pos)
    append (bp_img, inhand_img, obs_6d_norm) to sliding-window deques

    if action_counter == action_seq_size:        # ← replan trigger
        state = (bp_img_seq, inhand_seq, obs_seq)
        cond  = {0: (bp_batch, inhand_batch, obs_batch)}  # B copies of window

        trajectory, infos ← model(cond)   # (B, H, 9) — see Gen model below
        trajectory ← projector.project(trajectory)  # SLSQP if constraints active

        which ← trajectory_selection(trajectory, batch_size)  # pick best candidate
        action_seq ← unnormalize(trajectory[which, :action_seq_size, :3])  # (4, 3)
        action_counter = 0

    action ← action_seq[action_counter]
    mental_robot_pos += action                   # integrate velocity
    env.step([mental_robot_pos | gripper_quat])  # 7D pose command
    action_counter += 1

    (obs_6d, bp_img, inhand_img) ← env output
```

### 6.3 Trajectory Selection (FM-PCC, `mpc_batch_size=4`)

With 4 candidates generated per replan, one is selected by:

| Method | How | When used |
|---|---|---|
| `random` | always index 0 (DPCC semantics) | `no_constraint` and `dynamics_only` variants |
| `temporal_consistency` | pick candidate with minimum L2 distance to previous trajectory | tightened variants without obstacle constraints |
| `minimum_projection_cost` | pick candidate with minimum total SLSQP projection cost | variants with obstacle or halfspace constraints |

DPCC (`mpc_batch_size=1`) has no selection — single candidate, takes index 0.

---

## 7. Geometric Constraint Projection (SLSQP)

This is the defining feature of FM-PCC/DPCC over all D3IL baselines.

### 7.1 The 9D Constraint Space

The SLSQP projector operates in the full **9D trajectory space**:

```
Trajectory column layout:
  dim 0: dx   (action x)
  dim 1: dy   (action y)
  dim 2: dz   (action z)
  dim 3: des_x (commanded EE x)
  dim 4: des_y (commanded EE y)
  dim 5: des_z (commanded EE z)
  dim 6: x    (actual EE x — workspace constraint target)
  dim 7: y    (actual EE y — workspace constraint target)
  dim 8: z    (actual EE z — workspace constraint target)
```

Workspace bounds and halfspace/obstacle constraints are enforced on `dims 6-8` (actual
EE position `c_pos`). Dynamics constraints link actions to position:
`c_pos[t+1] = c_pos[t] + action[t]` (enforced as `deriv` constraints on `[6←0, 7←1, 8←2]`).

### 7.2 Constraint Types (from `geo_constraint_variants` in YAML)

| Type | Dims constrained | Effect |
|---|---|---|
| `bounds` | 6, 7, 8 (c_pos) | Workspace box: `lb ≤ c_pos ≤ ub` |
| `dynamics` | 6←0, 7←1, 8←2 | Euler consistency: `c_pos[t+1] = c_pos[t] + Δ` |
| `halfspace` | 6, 7 (c_pos XY) | Linear half-plane in XY: keep robot on named side of line |
| `obstacle` | 6, 7, 8 (c_pos) | Sphere exclusion: `‖c_pos − center‖ ≥ radius` |

### 7.3 Tightened vs Nominal Variants

Tightened variants (`dpcc-t`) apply an `enlarge_constraints` margin `δ` inward:

```
ws_lb_planning = ws_lb_nominal + δ    (lower bound rises)
ws_ub_planning = ws_ub_nominal − δ    (upper bound drops)
halfspace_planning = halfspace shifted δ meters inward
```

The projector enforces the tightened (planning) boundary.
Execution metrics always check against the nominal boundary (`enlarge=0`).

This creates a safety buffer: if the projector succeeds at the tightened boundary,
the actual trajectory has at least `δ` meters of margin over the nominal boundary.

Dual-boundary visualization (UF-16.6): tightened runs draw both layers in MPC plots —
solid for nominal (eval boundary), dashed for planning boundary.

### 7.4 Projection Normalizer

The projector receives normalized trajectories. A `ProjectorNormalizer` wrapper exposes
the fitted `LimitsNormalizer` objects under the key names the projector expects:

```python
class ProjectorNormalizer:
    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {
            'observations': obs_normalizer,   # .mins (6,)  .maxs (6,)
            'actions':      act_normalizer,   # .mins (3,)  .maxs (3,)
        }
```

`ProjectionNormalizer` (inside `Projector`) reads `normalizers['observations'].mins/maxs`
and `normalizers['actions'].mins/maxs` to map the 9D constraint bounds into normalised
space before calling SLSQP.

---

## 8. The Generative Model Math

Both Gen7 and Gen6V4 use the same `VisualUNet` backbone; the difference is in the
**generative engine** that wraps it.

### 8.1 Gen7 FM-PCC: Flow Matching (`VisualFlowMatching`)

The policy learns a **vector field** `v_θ(x_t, t | visual_cond)` that transports noise
to demonstrations via an ODE:

```
x_0 ~ p_data(τ)          # sample a 9D demonstration trajectory τ ∈ ℝ^{H×9}
x_1 ~ N(0, I)            # sample pure Gaussian noise
x_t = (1-t)·x_0 + t·x_1  # linear OT interpolation  (t ∈ [0,1])

target: u_t = x_1 - x_0   (constant along the interpolated path)

Loss: L = E_{t,x_0,x_1} [ ‖ v_θ(x_t, t | cond) - u_t ‖² ]
```

**Time sampling during training**: `t ~ Beta(α=1.5, β=1.0)` — biases toward `t ≈ 0`
(near the data distribution), which concentrates gradient signal where the vector field
is hardest to learn. Controlled by `time_beta_alpha_v3` / `time_beta_beta_v3` in config.

**Inference**: Euler ODE integration from `x_1` (noise) → `x_0` (trajectory):

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t | cond),   Δt = 1 / flow_steps_v3

Default: flow_steps_v3 = 100 Euler steps, t: 1.0 → 0.0
Backend: 'legacy_euler' (custom; no torchdiffeq overhead)
```

**Constraint projection** happens inline during the ODE forward pass: after each Euler
step, the SLSQP projector maps `x_{t-Δt}` back onto the feasible manifold. This projects
at every denoising step, not just once at the end.

**Action weight**: `action_weight = 1` (equal weighting of action and obs loss). The FM
vector field is defined for the full 9D trajectory; no additional action emphasis needed
because the straight OT path already biases the model toward high-probability data.

### 8.2 Gen6V4 DPCC: DDPM (`VisualDiffusion` / `VisualGaussianDiffusion`)

The policy learns a **denoising network** `ε_θ(x_t, t | visual_cond)`:

```
x_0 ~ p_data(τ)          # sample trajectory τ ∈ ℝ^{H×9}
t   ~ Uniform{1, ..., T}  # discrete timestep (T=100)
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)   # forward noise

Loss: L = E_{t,x_0,ε} [ action_weight · ‖ ε_action ‖² + ‖ ε_obs ‖² ]
     action_weight = 10   (upweights 3D action dims over 6D obs dims)
```

**Inference**: `n_diffusion_steps = 100` denoising steps, `t: T → 0`.
DPCC uses `mpc_batch_size=1` — one candidate trajectory per replan, no selection.

### 8.3 Training Data Flow (shared)

```
ParityAligningDataset.__getitem__(idx):
  ep, start, end = self.indices[idx]        # sliding window

  obs_raw = obs_6d_episode[ep][start:end]   # (H, 6)
  act_raw = actions_episode[ep][start:end]  # (H, 3)

  obs_norm = obs_normalizer.normalize(obs_raw)    # (H, 6)  LimitsNormalizer
  act_norm = act_normalizer.normalize(act_raw)    # (H, 3)  LimitsNormalizer

  trajectory = concat([act_norm, obs_norm], axis=-1)  # (H, 9)

  conditions = {
      0:             obs_norm[0],                  # (6,) anchor for apply_conditioning
      'primary_img': bp_cam_imgs[ep][start],       # (3, 96, 96) tensor
      'wrist_img':   inhand_cam_imgs[ep][start],   # (3, 96, 96) tensor
  }
  return Batch(trajectory, conditions)

↓  train_vision_agent() batch:

  bp_imgs     (B, T_win, 3, 96, 96)   # T_win = obs_seq_len = 1 (sliced by agent)
  inhand_imgs (B, T_win, 3, 96, 96)
  obs_seq     (B, T_win, 6)
  action      (B, H - obs_seq_len + 1, 3)   # or (B, H, 3) if no slice offset
  mask        (B, H)
```

### 8.4 Evaluation Data Flow

```
eval_fm_visual_aligning.py — per step:

  env → (bp_img: (3,96,96), inhand_img: (3,96,96), des_robot_pos: (3,), robot_pos: (3,))

  obs_6d = [des_robot_pos | robot_pos]              # (6,)
  obs_6d_norm = obs_normalizer.normalize(obs_6d)    # (6,)

  # Build B=mpc_batch_size copies of the window
  bp_batch     = (B, W, 3, 96, 96)   W = window_size = 1
  inhand_batch = (B, W, 3, 96, 96)
  obs_batch    = (B, W, 6)

  cond = {0: (bp_batch, inhand_batch, obs_batch)}

  # Replan every action_seq_size steps:
  trajectory, infos ← VisualFlowMatching(cond, projector)   # (B, H, 9)

  # Select candidate:
  which ← trajectory_selection(...)

  # Extract and denormalize actions:
  action_seq ← act_normalizer.unnormalize(trajectory[which, :action_seq_size, :3])

  # Execute action chunk (4 steps before next replan):
  for step in [0, 1, 2, 3]:
      action = action_seq[step]                   # (3,)
      mental_robot_pos += action
      env.step([mental_robot_pos, gripper_quat])  # 7D
```

---

## 9. Output Artifacts

Each variant produces a full suite of diagnostic files:

```
logs/aligning-d3il-visual/plan_fm_visual_aligning/<exp>/results/<seed>/
  expert_references/
    expert_rollout_<r>.gif        ← recorded before variant loop (reference)
    expert_rollout_<r>.mp4
  <variant>/
    <variant>.npz                 ← raw array results
    <variant>.png                 ← 6-panel rollout grid (legacy)
    results_seed_<s>.pkl          ← Python dict, all rollouts
    eval_<variant>.log            ← full console output
    diag_first_replan.txt         ← first-replan action magnitudes, obs health
    constraint_metrics.json       ← UF-16.3: cross-rollout aggregate (15 metrics)
    diagnostics/
      rollout_<r>.gif
      rollout_<r>.mp4
      rollout_<r>_data.pkl        ← per-rollout full data
      rollout_<r>_stats.json      ← per-rollout metrics (success, dist, context)
      rollout_<r>_report.png      ← 2-panel summary (EE path + dist-over-time)
      rollout_<r>_mpc_foresight.png  ← MPC decision-point visualization
```

### MPC Foresight Plot (`_mpc_foresight.png`)

Three-panel visualization per rollout:
- **XY top-down**: EE trajectory (black line), replan decision points (black dots), all
  candidate c_pos trajectories (uniform green), selected candidate (blue), constraint
  overlays (bounds box, halfspace line, obstacle circles). For tightened variants: solid
  = nominal boundary, dashed = planning boundary.
- **3D view**: same trajectory in 3D with constraint shapes rendered as wireframes.
- **Per-step tracking error**: `‖c_pos − des_c_pos‖` over episode timesteps.

---

## 10. Comparison to D3IL Visual Agents

D3IL provides 11 imitation learning methods with visual variants. These are the direct
academic baselines for this pipeline.

### 10.1 Observation Space Comparison

| | FM-PCC / DPCC Visual | D3IL Visual Agents |
|---|---|---|
| Images | 2 cameras, 96×96 (bp + inhand) | 2 cameras, 96×96 (same) |
| State | 6D (des_c_pos + c_pos) | 6D (or varies; most use des_c_pos + c_pos) |
| Trajectory conditioned on | 6D state + 128D ResNet | 6D state + 128D ResNet |
| Obs window | `T_win=1` (single frame) | `T_win=5` (D3IL default `obs_seq_len=5`) |

A key divergence: D3IL baseline agents (`ddpm_encdec_vision`, `beso_vision`, etc.) use
`obs_seq_len=5` — a 5-frame sliding window of images and state fed as a temporal context.
Our FM-PCC/DPCC uses `obs_seq_len=1` because `ParityAligningDataset` provides
single-frame samples. This means:

- D3IL agents have **temporal context** (last 5 frames)
- FM-PCC/DPCC compensates via **MPC horizon** (plans 8 future steps)

The information available to each agent at decision time is different in structure but
comparable in total information content.

### 10.2 Architecture Comparison

| | Gen7 FM-PCC | Gen6V4 DPCC | D3IL `ddpm_encdec_vision` | D3IL `bc_vision` |
|---|---|---|---|---|
| Generative model | Flow Matching (ODE) | DDPM (100 steps) | DDPM (16 steps, EncDec Transformer) | Deterministic MLP |
| Backbone | UNet1D + FiLM | UNet1D + FiLM | Transformer encoder-decoder | MLP |
| Vision encoder | 2× ResNet-64 → 128D | 2× ResNet-64 → 128D | 2× ResNet-64 → 128D | 2× ResNet-64 → 128D |
| Obs window | 1 frame | 1 frame | 5 frames | 5 frames |
| Trajectory horizon | H=8 (9D traj) | H=8 (9D traj) | H=8 (internal) | Single step |
| MPC replanning | Yes (every 4 steps) | Yes (every 4 steps) | No | No |
| Constraint projection | Yes (SLSQP, YAML) | Yes (SLSQP, YAML) | No | No |
| Action chunk size | 4 | 4 | 1 (single-step) | 1 |
| Candidates | 4 (FM) | 1 (DPCC) | 1 | 1 |

### 10.3 The MPC Advantage in the Visual Context

Both FM-PCC and DPCC plan H=8 steps and execute 4 before replanning (closed-loop MPC).
D3IL agents are typically **open-loop single-step**: generate one action from current
observation, execute, repeat.

```
D3IL visual (single-step):
  (bp_img, inhand_img, obs_6d)_t → ResNet → DDPM/MLP → action_t → execute → t+1 → repeat

FM-PCC MPC (closed-loop horizon):
  (bp_img, inhand_img, obs_6d)_t → ResNet → FM ODE → τ_{t:t+8} (8-step trajectory)
  execute τ_{t:t+4} (chunk of 4)
  replan at t+4 using fresh observation → τ_{t+4:t+12}
  ...
```

Replanning with fresh observations allows FM-PCC to correct for:
- Policy drift (model's predicted `des_c_pos` diverges from actual sim state)
- Box disturbances
- Constraint violations detected at execution (post-projection residuals)

### 10.4 Constraint Projection: Absent in All D3IL Agents

No D3IL baseline has a geometric projector. Their rollouts are unconstrained:
- The robot may reach table edges or obstacle regions if the policy predicts such actions
- There is no YAML-configured safety boundary
- Comparison against constrained FM-PCC/DPCC variants requires acknowledging this structural difference

The `no_constraint` and `dynamics_only` variants of FM-PCC/DPCC are the closest to
D3IL's operating conditions (still with MPC replanning).

### 10.5 Shared Success Metric

All variants use the same D3IL success criterion:

```
mean_distance = 0.5 × (3D_position_error_metres + rotation_error / π)
success       = mean_distance < 0.033
```

This allows direct `success_rate` and `mean_distance` comparison between:
- FM-PCC (`no_constraint`, `dynamics_only`, `combined_4`, `combined_5`)
- DPCC (`no_constraint`, `combined_4`)
- D3IL visual agents (`ddpm_encdec_vision`, `beso_vision`, `act_vision`, `bc_vision`, …)

---

## 11. Config Flags and Key Parameters

### Core visual aligning switches

```python
# fm_visual_aligning (training)
'if_vision':     True          # visual path
'obs_dim':       6             # MUST be 6 — never 20 or 128 (config comment)
'action_dim':    3             # 3D velocity
'horizon':       8             # U-Net horizon
'obs_seq_len':   5             # training window
'action_seq_size': 4           # chunk size
'time_beta_alpha_v3': 1.5      # FM Beta prior α
'time_beta_beta_v3':  1.0      # FM Beta prior β
'action_weight':      1        # FM: equal weights

# plan_fm_visual_aligning (eval)
'flow_steps_v3':  100          # Euler ODE steps
'mpc_batch_size': 4            # candidate pool
'obs_seq_len':    1            # MUST match ParityAligningDataset
'window_size':    1            # single-frame conditioning
```

### Critical constraint: `obs_dim = 6`

The `obs_dim` config key MUST be set to `6` for the visual path. Legacy configs from
before fix_5 sometimes set it to `128` (a stale placeholder for the ResNet latent size).
`VisualUNet` ignores this key entirely and hardcodes `TRANSITION_DIM = 9` (= act(3) +
obs(6)), but downstream code that reads `obs_dim` for normalizer construction or
projection bounds would silently produce wrong shapes if set to `128`.

### `max_path_length` dual role

In the visual aligning config, `max_path_length` is a **loadpath key only** — it appears
in the checkpoint directory name as `steps{max_path_length}`. It does NOT cap rollout
steps (that is `max_episode_length=400`). The training config and plan config MUST have
matching `max_path_length` values, or `FileNotFoundError` at eval load time.

---

## 12. Summary: Visual vs Non-Visual vs D3IL

| Dimension | FM-PCC Visual | FM-PCC Non-Visual | D3IL Visual Agents |
|---|---|---|---|
| Images | 2 cameras, 96×96 | None | 2 cameras, 96×96 |
| State obs | 6D (des_c_pos + c_pos) | 20D (full state) | 6D |
| Obs window | 1 frame | — | 5 frames |
| Trajectory dim | 9D (act3 + obs6) | 22D (act2 + obs20) | varies per model |
| Action dim | 3D (Δx, Δy, Δz) | 2D (vx, vy) | 3D |
| MPC horizon | H=8, chunk=4 | H=8, chunk=4 | None (single-step) |
| Candidates | 4 per replan | 4 per replan | 1 |
| Constraint projection | Yes (SLSQP) | Yes (same projector) | No |
| Inference cost | High (2× ResNet/step) | Low (state only) | High (2× ResNet/step) |
| Temporal context | 1 frame (MPC compensates) | — | 5 frames |

The visual path is the primary evaluation mode. Use the non-visual path as an ablation
to isolate the contribution of image observations, and D3IL visual agents as the
reference academic baseline.
