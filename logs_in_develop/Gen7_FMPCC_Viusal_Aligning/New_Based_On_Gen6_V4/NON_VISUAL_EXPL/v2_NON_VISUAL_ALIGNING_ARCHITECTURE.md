# Non-Visual Aligning: Architecture, Math, and D3IL Comparison (v2 — UF-17)

**Scope**: Gen7 FM-PCC and Gen6V4 DPCC — non-visual (state-only) aligning pipeline  
**Date**: 2026-05-29  
**Supersedes**: [`v1_NON_VISUAL_ALIGNING_ARCHITECTURE.md`](v1_NON_VISUAL_ALIGNING_ARCHITECTURE.md)  
**UF-17 fix**: [`../u_f_17_fix_non_visual/PLAN_NON_VISUAL_FIX.md`](../u_f_17_fix_non_visual/PLAN_NON_VISUAL_FIX.md)

---

## 1. What "Non-Visual" Means in This Codebase

The aligning task has two distinct operating modes controlled by the `if_vision` flag:

| Mode | `if_vision` | Input to model | Cameras active |
|---|---|---|---|
| **Visual** (primary) | `True` | Bird's-eye image + in-hand image + 6D robot state | Yes (EGL offscreen) |
| **Non-visual** (state-only) | `False` | 23D trajectory (full task state, no images) | No (BLIND render) |

In both modes the **environment dynamics and task are identical** — the robot must push a
box to a target position and orientation. The only difference is what information the
policy receives as input.

The non-visual mode exists for:
1. **Ablation**: quantify how much visual input contributes
2. **Baseline comparison**: against D3IL's state-only agents
3. **Compute efficiency**: no ResNet encoding overhead at inference

> **v1 → v2 key change**: v1 used a broken 22D trajectory (2D action, `mental_robot_pos`
> bridge, wrong projector dims). v2 follows pure original DPCC principle: 23D trajectory
> (3D action + 20D full state), `apply_conditioning` pins full obs at step 0, no FiLM,
> no bridge.

---

## 2. Observation Space

### 2.1 Training Dataset Observation — 20D

`StateOnlyAligningDataset` builds a 20D state vector directly from the demonstration
pickle files, identical to what `Aligning_Dataset` (D3IL) reads:

```
obs[0:3]   = robot_des_pos      (commanded TCP position, x/y/z)
obs[3:6]   = robot_c_pos        (actual TCP position, x/y/z)
obs[6:9]   = push_box_pos       (box centre, x/y/z)
obs[9:13]  = push_box_quat      (box orientation, w/x/y/z)
obs[13:16] = target_box_pos     (target location, x/y/z)
obs[16:20] = target_box_quat    (target orientation, w/x/y/z)
─────────────────────────────────────────────────
Total: 20D
```

Pickle keys: `robot['des_c_pos']`, `robot['c_pos']`, `push-box['pos']`,
`push-box['quat']`, `target-box['pos']`, `target-box['quat']`.

`robot_des_pos` is the *commanded* position sent to the low-level controller; it differs
from `robot_c_pos` (the actual achieved position) due to controller lag.

### 2.2 Simulation Runtime Observation — built as 20D directly

At evaluation time `aligning_sim.py` builds 20D obs by prepending the last commanded
position to the 17D environment observation:

```python
pred_action = env.robot_state()    # last commanded position (3D des_c_pos)
while not done:
    obs = np.concatenate((pred_action[:3], obs))   # 3D + 17D = 20D
    pred_action = agent.predict(obs)               # wrapper receives 20D
    pred_action = pred_action[0] + obs[:3]         # delta + des_c_pos = new abs pos
    obs, reward, done, info = env.step(...)        # env resets obs to 17D
```

This mirrors original DPCC (avoiding task) exactly: `obs = concat(action[:2], obs)`.
The 20D layout at runtime:

```
obs[0:3]   = des_c_pos    (last commanded pos, prepended by sim)
obs[3:6]   = c_pos        (actual TCP, from env.get_observation)
obs[6:9]   = box_pos
obs[9:13]  = box_quat
obs[13:16] = target_pos
obs[16:20] = target_quat
```

**No bridge needed.** The sim already provides `des_c_pos` at each step.
The `mental_robot_pos` integration from v1 is removed.

---

## 3. Action Space

### Training and inference actions — 3D

```
action[0] = dx = des_pos_x[t+1] − des_pos_x[t]    (x-velocity command)
action[1] = dy = des_pos_y[t+1] − des_pos_y[t]    (y-velocity command)
action[2] = dz = des_pos_z[t+1] − des_pos_z[t]    (z-velocity command)
─────────────────────────────────────────────────
action_dim: 3D   (matches visual path)
```

The z component is near-zero throughout the push task but is kept for dimensional
consistency with the visual path and the projector's dynamics constraint wiring
(`c_pos_z ← dz`).

### Execution (7D, sent to MuJoCo controller)

The sim handles position integration: `new_abs_pos = action_delta + obs[:3]` (des_c_pos
from the 20D obs). The 7D command is assembled by the sim:

```python
pred_action = pred_action[0] + obs[:3]              # absolute 3D position
pred_action = np.concatenate((pred_action, [0, 1, 0, 0]))  # 7D pose command
env.step(pred_action)
```

---

## 4. Model Architecture

### 4.1 Visual Path (Gen7 FM-PCC primary — unchanged)

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
│  FiLM conditioning to UNet       │
└──────────┬───────────────────────┘
           │  128D visual context → FiLM at every UNet block
           ▼
┌──────────────────────────────────────────────────────────────┐
│  VisualUNet (U-Net 1D)  cond_dim=128, use_cond_projection=T  │
│  Trajectory input: x ∈ ℝ^{H×9}  (H=8, 3D action + 6D obs)  │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼  Flow Matching ODE (100 Euler steps, t: 0→1)
┌──────────────────────────┐
│  action = traj[:, :3]    │  3D position delta
└──────────────────────────┘
```

### 4.2 Non-Visual Path (UF-17 — pure DPCC)

```
┌──────────────────────────────────────────────────────────────┐
│  Input (from sim at each step)                                │
│  obs_20d  = [des_c_pos(3) | c_pos(3) | box(3) | box_q(4)    │
│              | tgt(3) | tgt_q(4)]   — full task state        │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼  apply_conditioning: pins obs_20d at traj step 0
           │  NO FiLM, NO conditioning vector, cond ignored by UNet
           │
┌──────────────────────────────────────────────────────────────┐
│  VisualUNet (if_vision=False)                                 │
│    cond_dim=0, use_cond_projection=False                     │
│    → backbone receives cond=None, only time_mlp(t) used      │
│  Trajectory input: x ∈ ℝ^{H×23}  (H=8, 3D action + 20D obs)│
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼  Flow Matching ODE (Gen7) / DDPM (Gen6V4)
┌──────────────────────────┐
│  action = traj[:, :3]    │  3D position delta
└──────────────────────────┘
```

This is **mechanistically 1:1 with original DPCC** (`/workspaces/dpcc`), where the
`cond` parameter of `UNet1DTemporalCondModel` is accepted but completely ignored. The
only difference is 3D actions instead of 2D.

**Key architectural differences (visual vs non-visual):**

| Dimension | Visual | Non-Visual (UF-17) |
|---|---|---|
| Conditioning mechanism | FiLM (ResNet → 128D) | `apply_conditioning` only |
| Conditioning obs | 6D (des_c_pos + c_pos) | None (cond ignored by UNet) |
| Task info (box/target) | ResNet encodes from images | In trajectory obs dims 6-22 |
| Trajectory shape | `[H, 9]` (3+6) | `[H, 23]` (3+20) |
| Action dim | 3D | 3D |
| Vision encoder | ResNet (2 × 64D → 128D) | None |
| UNet cond_dim | 128 | 0 |
| Inference cost | High (ResNet per step) | Low (pure state) |

The 23D trajectory is necessary because the visual path uses FiLM to convey box/target
context, which the non-visual path has no equivalent for. Moving box/target into the
trajectory obs dims is the DPCC-correct solution.

---

## 5. The Generative Model Math

### 5.1 Gen7: Flow Matching (non-visual — same math, different trajectory dim)

The FM objective is identical to the visual path; only the trajectory dimension changes:

```
x_0 ~ p_data(τ)          # sample τ ∈ ℝ^{H×23}  ← 23D not 9D
x_1 ~ N(0, I)
x_t = (1-t)·x_0 + t·x_1

target: u_t = x_1 - x_0

Loss: L = E_{t,x_0,x_1} [ ‖ v_θ(x_t, t) - u_t ‖² ]
```

`cond` passed to the model is `{0: obs_20d_anchor}`. `apply_conditioning` pins
`x_t[:, 0, 3:]` = obs_20d at each ODE step. The network sees this only through
the frozen trajectory slice, not through a separate conditioning vector.

Time sampling: `t = 1 - Beta(α=1.5, β=1.0).sample()` — same Beta prior as visual.

### 5.2 Gen6V4: DDPM (non-visual)

```
x_0 ~ p_data(τ)          # sample τ ∈ ℝ^{H×23}
t   ~ Uniform{1, ..., T}  # discrete timestep (T=100)
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

Loss: L = E_{t,x_0,ε} [ ‖ ε_θ(x_t, t) - ε ‖² ]
```

### 5.3 Trajectory Representation

The trajectory tensor `x ∈ ℝ^{H×d}` layout:

```
Visual (9D):
x[h, :] = [ dx   dy   dz | des_x des_y des_z | c_x  c_y  c_z ]
             act(3)         des_c_pos(3)         c_pos(3)
             0-2             3-5                  6-8

Non-visual (23D):
x[h, :] = [ dx   dy   dz | des_x des_y des_z | c_x  c_y  c_z
             act(3)         des_c_pos(3)         c_pos(3)
             0-2             3-5                  6-8

           | bx   by   bz | bw   bx   by   bz | tx   ty   tz | tw   tx   ty   tz ]
             box_pos(3)     box_quat(4)           tgt_pos(3)    tgt_quat(4)
             9-11            12-15                16-18          19-22
```

The constraint-relevant dims (action 0-2, c_pos 6-8) are at **identical indices** in
both 9D and 23D. The projector's dynamics wiring `[6←0, 7←1, 8←2]` is unchanged.

`action_weight` in the loss:
```
Non-visual FM:   action_weight=10 (same as DPCC convention)
Visual FM:       action_weight=1
```

---

## 6. Training and Evaluation Data Flow

### 6.1 Training

```
Non-visual (UF-17):
  StateOnlyAligningDataset.__getitem__()
    → obs_raw  (H, 20)   # [des_c_pos|c_pos|box_pos|box_quat|tgt_pos|tgt_quat]
    → act_raw  (H, 3)    # des_c_pos[t+1] - des_c_pos[t]
    → obs_norm = obs_normalizer.normalize(obs_raw)   # LimitsNormalizer, 20D
    → act_norm = act_normalizer.normalize(act_raw)   # LimitsNormalizer, 3D
    → trajectory = concat([act_norm, obs_norm]) ∈ ℝ^{H×23}
    → conditions = {0: obs_norm[0]}                  # 20D anchor, no image keys
    → Batch(trajectory, conditions)

  VisualFlowMatching.loss(trajectories, conditions):
    if not self.model.if_vision:          # UF-17 G1 guard
        return self.p_losses(x=trajectories, cond=conditions, t=t)
    # visual path below — not reached

Visual (reference):
  ParityAligningDataset.__getitem__()
    → trajectory ∈ ℝ^{H×9}
    → conditions = {0: obs_6d_norm[0], 'primary_img': ..., 'wrist_img': ...}
```

### 6.2 Evaluation Rollout

```
Non-visual rollout (UF-17):
  aligning_sim.py builds 20D obs each step:
    obs = concat(last_commanded_pos[:3], env_obs_17d)   # 20D

  VisualAgentWrapper.predict(obs_20d):
    des_robot_pos_np = obs_20d[:3]
    robot_pos_np     = obs_20d[3:6]
    mental_robot_pos ← obs_20d[:3]   # init from actual des_c_pos

    obs_norm = obs_normalizer.normalize(obs_20d)   # 20D normalizer
    cond = {0: obs_norm.repeat(B)}                 # (B, 20) anchor

    if replan:
      trajectory ← model(cond)              # (B, H, 23)
      trajectory ← projector.project(traj)  # trajectory_dim=23
      action_seq = unnorm(traj[which, :4, :3])   # (4, 3) delta actions

    action = action_seq[step_in_chunk]   # (3,)
    mental_robot_pos += action
    return action   # sim adds to obs[:3] to get absolute pos

Visual rollout (reference):
  env → (obs_6d, bp_img, inhand_img)
  context deques ← sliding window (window_size=1)
  cond = {0: (bp_batch, inhand_batch, obs_batch)}   # visual tuple
  trajectory ← model(cond)   # (B, H, 9)
```

---

## 7. Constraint Projection (UF-17)

The projector operates on the 23D trajectory but only constrains dims 0-8:

```
setup_dpcc_projector(..., trajectory_dim=23):
    pad = 23 - 9 = 14
    lb = [-inf×6 | ws_lb(3) | -inf×14]   # 23D
    ub = [+inf×6 | ws_ub(3) | +inf×14]   # 23D

    Projector(transition_dim=23, ...)
```

Constraint dim map (unchanged from visual):

| Constraint | Dims | Notes |
|---|---|---|
| Bounds on c_pos | 6, 7, 8 | workspace lb/ub |
| Dynamics | `[6←0, 7←1, 8←2]` | Euler: `c_pos[t+1] = c_pos[t] + act[t]` |
| Halfspace (XY) | 6, 7 | EE horizontal position |
| Obstacles | 6, 7, 8 | sphere exclusion |

Dims 9-22 (box/target obs) carry ±inf bounds — the projector ignores them.

---

## 8. Comparison to D3IL State-Only Aligning

D3IL (ICLR 2024) provides its own state-only agents. These are the direct academic baselines.

### 8.1 Observation Space

| | FM-PCC / DPCC Non-Visual (UF-17) | D3IL State-Only |
|---|---|---|
| Training obs | 20D (same layout as D3IL) | 20D |
| Runtime obs | 20D (sim prepends des_c_pos, no bridge) | 20D directly |
| In trajectory | Yes — full obs in trajectory dims 3-22 | Yes (obs = trajectory) |
| Action dim | **3D** (dx, dy, dz) | 2D (vx, vy — D3IL drops dz) |
| Obs window | H=8 (full horizon in trajectory) | `window_size=1` (single-step) |

### 8.2 Architecture Family

| | FM-PCC non-visual (Gen7) | DPCC non-visual (Gen6V4) | D3IL State-Only |
|---|---|---|---|
| Primary model | Flow Matching + UNet1D | DDPM + UNet1D | BC, DDPM-MLP, BESO, ACT, … |
| Conditioning | `apply_conditioning` only | same | obs history window |
| FiLM | None ✓ | None ✓ | None |
| UNet cond | Ignored (cond_dim=0) | Ignored | N/A (no UNet in most) |
| Trajectory dim | 23D `[act(3)+obs(20)]` | 23D | obs only (no joint traj) |
| Sequence length | H=8 | H=8 | window=1 (single-step) |
| MPC planning | Yes (replan every 4) | Yes | No |
| Constraint projection | Yes (SLSQP) | Yes | No |
| Action chunking | 4 steps | 4 steps | 1 step |

### 8.3 The MPC Advantage

Even without images, FM-PCC and DPCC plan a full H=8 horizon and execute 4 actions
before replanning. D3IL baselines are typically single-step:

```
D3IL single-step:   obs_t → policy → action_t → execute → obs_{t+1} → repeat

FM-PCC/DPCC MPC:    obs_t → FM/DDPM → τ_{t:t+8} (full 23D trajectory)
                    execute τ_{t:t+4} (4 actions)
                    replan at t+4 using fresh obs
```

### 8.4 What Makes Them Comparable

All modes share the D3IL success criterion:

```
mean_distance = 0.5 × (3D_position_error_metres + rotation_error / π)
success       = mean_distance < 0.033
```

Use `no_constraint` or `dynamics_only` variant for the fairest D3IL comparison
(D3IL has no projection; these variants run the full MPC without SLSQP snapping).

---

## 9. Config Flags

### Non-visual training block (`config/aligning-d3il-visual.py`)

```python
base['ddpm_encdec_vision_nonvisual'] = {
    **base['ddpm_encdec_vision'],
    'action_dim': 3,    # UF-17: 3D velocity [dx, dy, dz]
    'obs_dim':    20,   # full state
    'if_vision':  False,
    'prefix':     'ddpm_encdec_vision_nonvisual/',
}
```

`VisualUNet.__init__` detects `if_vision=False` and sets:
- `transition_dim = action_dim + obs_dim = 23`
- `latent_dim = 0` (no ResNet)
- `use_cond_projection = False` (no FiLM)

The backbone UNet receives `cond=None` at every forward call — same as original DPCC.

### GIF recording in non-visual eval

`if_vision=False` sets `RenderMode.BLIND`. Passing `--record gif` auto-promotes to
visual rendering for the output only — the **model still runs in non-visual mode**:

```python
if not if_vision and record_mode != 'none':
    if_vision = True   # render for recording only; model prediction unchanged
```

---

## 10. Summary: When to Use Non-Visual Mode

| Use case | Recommendation |
|---|---|
| Full FM-PCC evaluation | Use `if_vision=True` (primary mode) |
| Ablation: how much does vision help? | Run `if_vision=False`, same seed, compare `success_rate` |
| Baseline comparison vs D3IL | Use `no_constraint` variant; D3IL state-only via `d3il_visual_aligning_baseline_test/` |
| Debug MPC / constraint logic | Non-visual is ~3× faster (no ResNet), good for rapid iteration |
| Compute-constrained evaluation | Non-visual: no GPU RAM for image encoding |

The D3IL state-only agents are the **purest** academic baseline — no MPC, no constraint
projection, single-step, same obs. The FM-PCC non-visual mode is the **fairest
intra-system comparison** — same model family, same MPC loop, same projector, only the
image stream removed.

**What changed from v1**: action 2D→3D, trajectory 22D→23D, `mental_robot_pos` bridge
removed, `apply_conditioning` now pins full 20D obs (not fake 6D), projector correctly
parameterised to `transition_dim=23`, training routes to base `p_losses()` without
image keys. Architecture principle is now 1:1 with original DPCC.
