# Non-Visual Aligning: Architecture, Math, and D3IL Comparison

**Scope**: Gen7 FM-PCC and Gen6V4 DPCC — non-visual (state-only) aligning pipeline  
**Date**: 2026-05-28

---

## 1. What "Non-Visual" Means in This Codebase

The aligning task has two distinct operating modes controlled by the `if_vision` flag:

| Mode | `if_vision` | Input to model | Cameras active |
|---|---|---|---|
| **Visual** (primary) | `True` | Bird's-eye image + in-hand image + state | Yes (EGL offscreen) |
| **Non-visual** (state-only) | `False` | Flat 20D state vector | No (BLIND render) |

In both modes the **environment dynamics and task are identical** — the robot must push a box to a target position and orientation. The only difference is what information the policy receives as input.

The non-visual mode exists for:
1. **Ablation**: quantify how much visual input contributes
2. **Baseline comparison**: against D3IL's state-only agents
3. **Compute efficiency**: no ResNet encoding overhead at inference

---

## 2. Observation Space

### 2.1 Training Dataset Observation — 20D

Both `Aligning_Dataset` (state-only) and `Aligning_Img_Dataset` (visual) construct the
same 20D state vector from the demonstration pickle files:

```
obs[0:3]   = robot_des_pos      (desired TCP command, x/y/z)
obs[3:6]   = robot_c_pos        (actual TCP position, x/y/z)
obs[6:9]   = push_box_pos       (box centre, x/y/z)
obs[9:13]  = push_box_quat      (box orientation, w/x/y/z)
obs[13:16] = target_box_pos     (target location, x/y/z)
obs[16:20] = target_box_quat    (target orientation, w/x/y/z)
─────────────────────────────────────────────────
Total: 20D
```

`robot_des_pos` is the *commanded* position sent to the low-level controller; it differs
from `robot_c_pos` (the actual achieved position) due to controller lag.

### 2.2 Simulation Runtime Observation — 17D

At evaluation time `Robot_Push_Env.get_observation(if_vision=False)` returns only:

```
env_state[0:3]   = robot_pos     (actual TCP, x/y/z)
env_state[3:6]   = box_pos       (x/y/z)
env_state[6:10]  = box_quat      (w/x/y/z)
env_state[10:13] = target_pos    (x/y/z)
env_state[13:17] = target_quat   (w/x/y/z)
─────────────────────────────────────────────────
Total: 17D
```

`robot_des_pos` does not exist in the live simulation — only the actual TCP is exposed.

### 2.3 The 17D → 20D Mismatch and the "Mental Robot Position" Bridge

The model was trained on 20D but the environment provides 17D.
The evaluation wrapper resolves this with a **mental robot position** (`mental_robot_pos`):

```python
# Initialised from the first environment observation
if self.mental_robot_pos is None:
    self.mental_robot_pos = env_state[:3].copy()   # start at actual TCP

# Build 20D observation for the policy
obs_20d = np.concatenate([self.mental_robot_pos, env_state])  # 3D + 17D = 20D

# After executing action, update the tracked desired position
self.mental_robot_pos += np.array([action[0], action[1], 0.0])  # integrate vx, vy
```

**Why this works:** In the training data, `robot_des_pos[t+1] ≈ robot_des_pos[t] + action[t]`
because the action is defined as `des_pos[t+1] − des_pos[t]`. At eval time the agent
integrates its own predictions into `mental_robot_pos`, replicating the role that the
logged desired position played in training. The agent effectively learns:
> "My next action should take `mental_robot_pos` closer to the target, given the current
> box and target state."

This is an **implicit closed-loop strategy** — the gap between `mental_robot_pos` and
`robot_c_pos` (first 3 vs next 3 dims of obs) encodes accumulated tracking error, which
the policy learns to reason about.

---

## 3. Action Space

### Training actions (both modes)

```
action[0] = vx = des_pos_x[t+1] − des_pos_x[t]    (x-velocity command)
action[1] = vy = des_pos_y[t+1] − des_pos_y[t]    (y-velocity command)
─────────────────────────────────────────────────
Training action_dim: 2D
```

Vertical motion (`vz`) is near-zero in the push task (robot slides horizontally), so it
is dropped for efficiency.

### Execution actions (7D, sent to MuJoCo controller)

The 2D policy output is promoted to a 7D pose command before being sent to the environment:

```python
desired_pos_3d = mental_robot_pos.copy()
desired_pos_3d[:2] += action_2d          # integrate vx, vy into x, y
gripper_quat   = [0, 1, 0, 0]           # fixed horizontal gripper
action_7d      = np.concatenate([desired_pos_3d, gripper_quat])
env.step(action_7d)
```

---

## 4. Model Architecture

### 4.1 Visual Path (Gen7 FM-PCC primary)

```
┌──────────────────────────────────────────────────────────────┐
│  Inputs                                                       │
│  bp_image_seq    (T, 3, 96, 96)  bird's-eye camera           │
│  inhand_img_seq  (T, 3, 96, 96)  in-hand camera              │
│  obs_seq         (T, 6)          des_c_pos(3) + c_pos(3)     │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  MultiImageObsEncoder (ResNet)   │
│  Each (T × 3, 96, 96) → (T, 64) │
│  Two cameras → concat → (T, 128) │
│  + obs linear proj → (T, 128)    │
│  Combined conditioning: (T, 256) │
└──────────┬───────────────────────┘
           │  conditioning context
           ▼
┌──────────────────────────────────────────────────────────────┐
│  VisualUNet (U-Net 1D, temporal)                             │
│  Trajectory input: x ∈ ℝ^{H×9}  (H=8, 3D action + 6D obs)  │
│  Down/Up blocks with cross-attention to conditioning          │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼  Flow Matching ODE (100 Euler steps, t: 0→1)
┌──────────────────────────┐
│  Output: trajectory      │
│  action = traj[:, :3]    │  3D position delta
└──────────────────────────┘
```

### 4.2 Non-Visual Path (Gen6V4 DPCC legacy; ablation for Gen7)

```
┌──────────────────────────────────────────────────────────────┐
│  Input                                                        │
│  obs_seq  (T, 20)    full 20D state (no images)              │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  Obs encoder: linear projection  │
│  (T, 20) → (T, 128) conditioning │
└──────────┬───────────────────────┘
           │  conditioning context
           ▼
┌──────────────────────────────────────────────────────────────┐
│  VisualUNet (same U-Net 1D, vision encoder bypassed)         │
│  Trajectory input: x ∈ ℝ^{H×22}  (H=8, 2D action + 20D obs)│
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼  DDPM (Gen6V4) or Flow Matching (Gen7 if enabled)
┌──────────────────────────────────┐
│  Output: trajectory              │
│  action = traj[:, :2]            │  2D velocity (vx, vy)
└──────────────────────────────────┘
```

**Key architectural differences:**

| Dimension | Visual | Non-Visual |
|---|---|---|
| Conditioning obs | 6D (des_c_pos + c_pos) | 20D (full state inc. box + target) |
| Conditioning vector | ~256D (ResNet + state) | ~128D (state linear only) |
| Trajectory shape | `[H, 9]` (3+6) | `[H, 22]` (2+20) |
| Action dim | 3D (Δx, Δy, Δz) | 2D (vx, vy) |
| Vision encoder | ResNet (2 × 64D → 128D) | None |
| Inference cost | High (image encoding per step) | Low (pure state) |

Note: in the visual path the obs conditioning is 6D because the task-relevant state
(`des_c_pos`, `c_pos`) is sufficient when paired with the visual context. In the
non-visual path all 20D must be passed explicitly because there are no images to
convey the box/target geometry.

---

## 5. The Generative Model Math

### 5.1 Gen7: Flow Matching (visual primary)

The policy learns a **vector field** `v_θ(x_t, t | cond)` that transports noise to
demonstrations. At training time:

```
x_0 ~ p_data(τ)          # sample a demonstration trajectory τ ∈ ℝ^{H×9}
x_1 ~ N(0, I)            # sample pure noise
x_t = (1-t)·x_0 + t·x_1  # linear interpolation  (t ∈ [0,1])

target velocity: u_t = x_1 - x_0   (constant along path)

Loss: L = E_{t,x_0,x_1} [ ‖ v_θ(x_t, t | cond) - u_t ‖² ]
```

Inference is ODE integration from `x_1` (noise) to `x_0` (trajectory):

```
dx/dt = v_θ(x_t, t | cond)
Euler: x_{t-Δt} = x_t - Δt · v_θ(x_t, t | cond)
```

100 Euler steps, `t: 1 → 0`. The time parameter `t` is sampled from a Beta distribution
`Beta(α=1.5, β=1.0)` during training (biases toward `t ≈ 0`, i.e. near-data), controlled
by `time_beta_alpha_v3` / `time_beta_beta_v3` in the config.

### 5.2 Gen6V4: DDPM (non-visual legacy)

The policy learns a **denoising network** `ε_θ(x_t, t | cond)` via:

```
x_0 ~ p_data(τ)          # sample trajectory τ ∈ ℝ^{H×22}
t   ~ Uniform{1, ..., T}  # discrete timestep (T=100 or 256)
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)   # forward noise

Loss: L = E_{t,x_0,ε} [ ‖ ε_θ(x_t, t | cond) - ε ‖² ]
```

Inference: iterative denoising `x_T → x_{T-1} → ... → x_0` (16–100 steps).

**Comparison:**

| | Flow Matching (Gen7) | DDPM (Gen6V4) |
|---|---|---|
| Training target | velocity field `u_t = x_1 - x_0` | noise `ε` |
| Forward process | linear OT interpolation | Markov Gaussian chain |
| Time schedule | continuous `t ∈ [0,1]`, Beta prior | discrete `t ∈ {1,…,T}`, cosine β |
| Inference | ODE integration (Euler) | Iterative denoising (DDPM/DDIM) |
| Steps (inference) | 100 | 16–100 |
| Mode coverage | Better (straight trajectories) | Good but curved |
| Training stability | High | Good |

Both condition on the same `cond` tensor — only the generative mechanism differs.

### 5.3 Trajectory Representation (shared)

The trajectory tensor `x ∈ ℝ^{H×d}` concatenates actions and observations along the
feature axis:

```
Visual:     x[h, :] = [Δpos_x, Δpos_y, Δpos_z,  des_c_x, des_c_y, des_c_z,  c_x, c_y, c_z]
                         action (3D)                    obs (6D)
Non-visual: x[h, :] = [v_x, v_y,  robot_des(3), robot_c(3), box(3), box_q(4), tgt(3), tgt_q(4)]
                         action (2D)                         obs (20D)
```

The U-Net processes this as a 1D temporal signal of length `H=8`, with feature channels
as the "spatial" dimension. The loss weights `action_weight` more heavily than obs
prediction to bias the model toward accurate action generation:

```
Loss = action_weight · ‖ v_action ‖² + obs_weight · ‖ v_obs ‖²
     = 10 · action_loss + 1 · obs_loss     (non-visual DDPM)
     = 1  · action_loss + 1 · obs_loss     (visual FM)
```

---

## 6. Training and Evaluation Data Flow

### 6.1 Training

```
Non-visual:
  Aligning_Dataset.__getitem__()
    → (obs[T,20], act[T,2], mask[T])
    → DataLoader batch: (B, T, 20), (B, T, 2), (B, T)
    → train_agent(): state, action, mask = data
    → model(state, action) → loss

Visual:
  Aligning_Img_Dataset.__getitem__()
    → (bp_imgs[T,3,96,96], inhand[T,3,96,96], obs[T,20], act[T,2], mask[T])
    → DataLoader batch
    → train_vision_agent(): bp_imgs, inhand_imgs, obs, action, mask = data
    → obs sliced to obs_seq_len=5; action sliced from obs_seq_len onward
    → state = (bp_imgs, inhand_imgs, obs)
    → model(state, action) → loss
```

### 6.2 Evaluation Rollout

```
Non-visual rollout:
  env.reset(if_vision=False) → obs_17d
  mental_robot_pos ← obs_17d[:3]

  loop (up to 400 steps):
    obs_20d = concat(mental_robot_pos, obs_17d)       # bridge 17→20D
    obs_norm = normaliser.normalise(obs_20d)
    cond = {0: obs_norm}                              # condition on current obs

    if replan:                                        # every action_seq_size steps
      traj = ODE_solve(x_T~N, cond) ∈ ℝ^{H×22}      # generate full horizon
      action_seq = traj[:, :2]                        # extract 2D actions
      action_seq = normaliser.unnorm_action(action_seq)

    action = action_seq[step_in_chunk]
    mental_robot_pos[:2] += action                    # integrate velocity
    env.step(concat(mental_robot_pos, [0,1,0,0]))    # 7D pose command
    obs_17d ← env output

Visual rollout:
  env.reset(if_vision=True) → (obs_6d, bp_img, inhand_img)
  context buffers ← sliding window of last obs_seq_len frames

  loop:
    append (bp_img, inhand_img, obs_6d) to context
    state = (bp_img_seq, inhand_seq, obs_seq)         # tuples of window

    if replan:
      traj = ODE_solve(x_T~N, encode(state)) ∈ ℝ^{H×9}
      action_seq = traj[:, :3]                        # extract 3D actions

    action = action_seq[step_in_chunk]
    env.step(action → 7D)
    obs ← env output
```

---

## 7. Comparison to D3IL State-Only Aligning

D3IL (ICLR 2024) provides its own state-only (`if_vision=False`) agents. These are the
direct academic baselines.

### 7.1 Observation Space

| | FM-PCC / DPCC Non-Visual | D3IL State-Only |
|---|---|---|
| Training obs | 20D (des_pos + c_pos + box + target) | 20D (same construction from same dataset) |
| Runtime obs | 17D from env + 3D mental bridge = 20D | 20D directly (D3IL eval uses different sim wrapper) |
| Obs dim in config | `obs_dim: 20` | `obs_dim: 20` |
| Action dim | 2D (vx, vy) | 3D (dx, dy, dz) in D3IL config (3rd dim ~0) |
| Window size | 8 (FM/DPCC sequence model) | 1 (D3IL baseline, single-step) |

D3IL's `aligning_config.yaml` specifies `action_dim: 3` for legacy compatibility, but
the third dimension (z) is effectively zero throughout the push task. FM-PCC trims
this to 2D explicitly.

### 7.2 Architecture Family

| | FM-PCC (Gen7) | DPCC (Gen6V4) | D3IL Baselines |
|---|---|---|---|
| Primary model | Flow Matching + UNet1D | DDPM-EncDec + UNet1D | Multiple (BC, DDPM, BESO, ACT, …) |
| Sequence length | `H=8` (plans full horizon) | `H=8` | `window_size=1` (single-step for most) |
| MPC planning | Yes (replan every 4 steps, `mpc_batch_size=4`) | Yes | No (open-loop rollout) |
| Constraint projection | Yes (YAML-configured geometry) | Yes | No |
| Vision encoder | ResNet (visual mode) | ResNet (visual mode) | ResNet (vision variants) |
| Action chunking | 4 steps (execute then replan) | 4 steps | 1 step (immediate) |

### 7.3 The MPC Advantage (Non-Visual Context)

Even in the non-visual case, FM-PCC and DPCC plan a full `H=8` step horizon and execute
a chunk of 4 actions before replanning. D3IL baselines (BC, DDPM-MLP, etc.) are typically
single-step: predict one action given current observation, execute, repeat.

```
D3IL single-step:   obs_t → policy → action_t → execute → obs_{t+1} → repeat

FM-PCC/DPCC MPC:    obs_t → diffusion/FM → τ_{t:t+H} (full trajectory)
                    execute τ_{t:t+4}
                    replan at t+4 → τ_{t+4:t+12} → execute τ_{t+4:t+8}
                    ...
```

Replanning with the actual observation (closed-loop MPC) allows the policy to recover
from drift, making the non-visual FM-PCC/DPCC fundamentally different from D3IL baselines
even at equal observation dimension.

### 7.4 Constraint Projection (FM-PCC Only)

In visual mode, FM-PCC applies geometric constraint projection during ODE integration:
each candidate trajectory is checked against YAML-defined boundaries (table edges, joint
limits, obstacle spheres). Non-visual mode applies the same projection since constraints
are defined in Cartesian robot space, not image space.

D3IL agents have no constraint mechanism — they run open-loop without any geometric
safety guarantees.

### 7.5 Success Metric

All modes share the same success criterion from D3IL:

```
mean_distance = 0.5 × (3D_position_error_metres + rotation_error / π)
success       = mean_distance < 0.033
```

This allows direct numerical comparison between:
- FM-PCC non-visual
- DPCC non-visual
- D3IL state-only agents (BC, DDPM, BESO, ACT, etc.)

---

## 8. Config Flags and SLURM Control

### Switching modes in config

**FM-PCC** (`config/aligning-d3il-visual.py`):
```python
base['plan_fm_visual_aligning']['if_vision'] = False   # eval: use state obs
base['fm_visual_aligning']['if_vision']      = False   # train: use state dataset
```

**DPCC** (`config/aligning-d3il-visual.py`):
```python
base['ddpm_encdec_vision_nonvisual'] = {
    **base['ddpm_encdec_vision'],
    'action_dim': 2,
    'obs_dim':    20,
    'if_vision':  False,
    'prefix':     'ddpm_encdec_vision_nonvisual/',
}
```

### GIF recording in non-visual eval

`if_vision=False` sets `RenderMode.BLIND` — no cameras, no frames. The eval script
auto-promotes to visual rendering when `--record` is requested:

```python
if not if_vision and record_mode != 'none':
    if_vision = True   # force camera rendering for diagnostics only
    # model still runs in non-visual inference mode
    # images are rendered offscreen purely for human review
```

The model itself never sees these images — they are only for the GIF output.

---

## 9. Summary: When to Use Non-Visual Mode

| Use case | Recommendation |
|---|---|
| Full FM-PCC evaluation | Use `if_vision=True` (primary mode) |
| Ablation: how much does vision help? | Run `if_vision=False` with same seed, compare `success_rate` |
| Baseline comparison vs D3IL | Use D3IL state-only agents (`bc`, `ddpm`, `beso`, `act`) via `d3il_visual_aligning_baseline_test/` |
| Debug MPC / constraint logic | Non-visual is faster (no ResNet encoding), good for iteration |
| Compute-constrained evaluation | Non-visual: ~3× faster inference, no GPU RAM for image encoding |

The D3IL state-only agents are the **purest** academic baseline — no MPC, no constraint
projection, single-step inference, same observation space. The FM-PCC non-visual mode
is the **fairest intra-system comparison** — same model family, same MPC loop, same
constraints, only the image stream removed.
