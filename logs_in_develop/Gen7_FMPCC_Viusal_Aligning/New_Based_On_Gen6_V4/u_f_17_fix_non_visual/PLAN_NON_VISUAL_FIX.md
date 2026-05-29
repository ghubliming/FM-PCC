# UF-17 — Non-Visual Aligning Fix: Plan

**Date**: 2026-05-28  
**Branch**: `update_into_FM`  
**Status**: PLANNING  
**Scope**: Fix the non-visual (state-only) aligning pipeline to follow original DPCC
principles. The fix is minimal: **change `action_dim` from 2 to 3**.

---

## 1. Current Non-Visual State (What Exists and What Is Wrong)

### 1.1 Config (`config/aligning-d3il-visual.py`)

```python
base['ddpm_encdec_vision_nonvisual'] = {
    **base['ddpm_encdec_vision'],
    'action_dim': 2,     # ← wrong: should be 3
    'obs_dim':    20,    # ← correct
    'if_vision':  False,
    'prefix': 'ddpm_encdec_vision_nonvisual/',
}
```

### 1.2 Trajectory (current)

```
[vx(0), vy(1) | des_x(2), des_y(3), des_z(4) | c_x(5), c_y(6), c_z(7) | box_pos(8-10) | box_quat(11-14) | tgt_pos(15-17) | tgt_quat(18-21)]
= act(2) + obs(20) = 22D
```

**Problem 1 — action_dim=2 drops dz**: The visual path uses 3D velocity
`[Δx, Δy, Δz]`. The non-visual path drops `Δz`, using 2D. This is inconsistent and
incorrect — the DPCC/FM model and projector are built around 3D actions.

**Problem 2 — projector dim misalignment**: With 2D action, `c_pos` sits at dims 5-7,
not 6-8. Original projector wiring is `deriv([6←0, 7←1, 8←2])`. With 22D and 2D
action those indices are wrong → projector enforces dynamics on wrong dims.

**Problem 3 — eval wrapper re-implements mental_robot_pos manually**: The
`VisualAgentWrapper` non-visual branch:
```python
obs_np = np.asarray(state, dtype=np.float64)   # receives 20D obs from sim
des_robot_pos_np = obs_np[:3]

obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])  # (6,) ← wrong!
cond = {0: obs_6d_np}   # only 6D pinned, but model has 20D obs
```
The eval wrapper collapses the 20D obs to 6D for `apply_conditioning`, throwing away
all box and target information. The model trained on 20D obs receives only 6D at eval.
Prediction is garbage.

**Problem 4 — `aligning_sim.py` non-visual branch is obsolete**: It uses a hand-rolled
per-step loop separate from the main `VisualAgentWrapper` infrastructure and does not
go through the standard eval/rollout/recording pipeline.

---

## 2. What Original DPCC Does for State (The Reference)

Original DPCC (`/workspaces/dpcc`, avoiding task):

```python
# UNet forward: cond parameter accepted but COMPLETELY IGNORED
def forward(self, x, cond, time, returns=None, ...):
    t = self.time_mlp(timesteps)
    # ... UNet conv blocks, only time embedding used
    # cond is never touched
```

**No FiLM. No conditioning vector. No separate encoder.**

Conditioning = `apply_conditioning` only:
```python
x[:, 0, action_dim:] = current_obs   # pin full obs at trajectory step 0
```

Trajectory for avoiding: `[act(2) | x, y, x_des, y_des]` = 6D.  
The full state — robot position AND goal position — is inside the trajectory.  
`apply_conditioning` pins all of it at step 0.

**In eval**: `obs = np.concatenate((action[:2], obs))` — the last commanded position
is prepended to the env obs. This is how `des_c_pos` enters the trajectory at eval time.

---

## 3. What D3IL State-Only Aligning Does

D3IL's original state-only agents (BC, DDPM, BESO, ACT without images):

- **Obs**: 20D = `[des_c_pos(3) | c_pos(3) | box_pos(3) | box_quat(4) | tgt_pos(3) | tgt_quat(4)]`
- **Action**: 2D velocity (D3IL convention, drops dz)
- **Window**: `obs_seq_len=5` — 5-frame history fed to model
- **Inference**: single-step open-loop, no MPC, no horizon planning
- **Constraints**: none

**Differences from our DPCC non-visual:**

| | D3IL State-Only | Our DPCC Non-Visual |
|---|---|---|
| Action dim | 2D | 3D (fixed) |
| Horizon | 1 (single step) | H=8 (planned trajectory) |
| MPC replanning | No | Yes (every 4 steps) |
| Constraint projection | No | Yes (SLSQP) |
| Obs window | 5 frames of history | Step 0 pinned (no history) |
| Trajectory | obs only | `[act | obs]` joint |

D3IL is the academic baseline. Our DPCC non-visual adds MPC + projection on top.

---

## 4. Proposed Paths Considered (and Evaluated)

### Path A — "Degenerate Visual DPCC": 9D trajectory + MLP FiLM ❌ Abandoned

**Idea** (proposed first in chat): Keep 9D trajectory `[act(3) | des_c_pos(3) | c_pos(3)]`,
replace ResNet with `MLP(state_20D) → 128D FiLM`. Exact visual architecture, only
the conditioning source changes.

**Why evaluated positively at first**: Clean ablation. Visual vs non-visual differ at
exactly one point (FiLM source). Projector unchanged. No mental_robot_pos bridge.

**Why abandoned**: Introduces FiLM into the non-visual path when the original DPCC
principle has NO conditioning vector to the UNet. FiLM is the right generalization for
IMAGE conditioning (see `FILM_CONDITIONING_RATIONALE.md`) but for STATE, original DPCC
just puts everything in the trajectory and uses `apply_conditioning`. Adding FiLM to
state conditioning is an unnecessary architectural addition that diverges from DPCC.

**What it gets wrong**: Box and target state should go into the trajectory (like original
DPCC does with goal state in avoiding), not into a separate FiLM conditioning path.

---

### Path B — "Strict DPCC with Images in Trajectory" ❌ Rejected

**Idea**: Put ResNet-encoded image features (128D) inside the trajectory at obs dims.
Trajectory = `[act(3) | des_c_pos(3) | c_pos(3) | img_feat(128)]` = 134D. No FiLM.

**Why rejected**: Image features at step 0 only (via `apply_conditioning`) — the UNet
must propagate 128D visual context across temporal dim via convolutions. This is
architecturally unsound for high-dim sensor data. FiLM exists for this exact reason.
See `FILM_CONDITIONING_RATIONALE.md` for full rationale.

---

### Path C — Current Code "Fixed" to Use 20D cond in eval ❌ Half-fix

**Idea**: Keep 22D trajectory (action_dim=2), fix eval wrapper to pass full 20D obs
as `apply_conditioning` anchor instead of 6D.

**Why insufficient**: action_dim=2 still means c_pos is at dims 5-7, not 6-8 →
projector dynamics `deriv([6←0, 7←1, 8←2])` is wrong. The dataset action construction
also needs to change. Partial fix only.

---

### Path D — 20D obs as conditioning only (no trajectory expansion) ❌ Inconsistent

**Idea** (surfaced in "20D" discussion): Keep 9D trajectory, pass 20D state as a
separate conditioning tensor without FiLM — i.e., as a second `apply_conditioning`
target or a plain global vector.

**Why rejected**: There's no clean mechanism for this in the existing `apply_conditioning`
framework. The 20D state includes box/target which changes each context but is constant
within a rollout — it's semantically identical to goal-state conditioning in original
DPCC, which goes INTO the trajectory. Not a clean approach.

---

### Path E (FINAL) — Pure DPCC: 23D trajectory, 3D action, no FiLM ✅

**The current non-visual was already following original DPCC principles** (everything
in trajectory, no FiLM, `apply_conditioning` pins current obs at step 0). The ONLY
structural bug is `action_dim = 2` instead of 3.

**Fix**: change `action_dim: 2 → 3`. Trajectory becomes 23D.

```
[Δx(0), Δy(1), Δz(2) | des_x(3), des_y(4), des_z(5) | c_x(6), c_y(7), c_z(8)
 | box_x(9), box_y(10), box_z(11) | bq_w(12), bq_x(13), bq_y(14), bq_z(15)
 | tgt_x(16), tgt_y(17), tgt_z(18) | tq_w(19), tq_x(20), tq_y(21), tq_z(22)]
= act(3) + obs(20) = 23D
```

**Projector dims after fix:**

| Constraint | Dims | Status |
|---|---|---|
| Bounds on c_pos | 6, 7, 8 | ✓ Correct |
| Dynamics `c_pos[t+1] = c_pos[t] + act[t]` | `[6←0, 7←1, 8←2]` | ✓ Correct |
| Halfspace (XY) | 6, 7 | ✓ Correct |
| Obstacles | 6, 7, 8 | ✓ Correct |

Projector is unchanged — same dim layout as the 9D visual trajectory for all
constraint-relevant dims. Box/target dims (9-22) are invisible to the projector.

---

## 5. Comparison Table: All Approaches vs Original DPCC

| | DPCC Original | D3IL State | Current Non-Visual | Path A (FiLM) | **Path E (Final)** |
|---|---|---|---|---|---|
| Trajectory dim | 6D | N/A (no traj) | 22D | 9D | **23D** |
| Action dim | 2D | 2D | 2D ❌ | 3D | **3D ✓** |
| Goal/task info | In traj obs ✓ | In obs ✓ | In traj obs ✓ | In FiLM ❌ | **In traj obs ✓** |
| FiLM | None ✓ | None | None ✓ | Added ❌ | **None ✓** |
| `apply_conditioning` | Full obs ✓ | N/A | 6D only ❌ | 6D only | **Full 20D obs ✓** |
| Projector c_pos dims | 2-3 | N/A | 5-7 ❌ | 6-8 ✓ | **6-8 ✓** |
| MPC replanning | Yes ✓ | No | Yes ✓ | Yes ✓ | **Yes ✓** |
| Constraint projection | Yes ✓ | No | Broken ❌ | Yes ✓ | **Yes ✓** |

---

## 6. Blueprint of Code Changes

> **Guide only — no full code. File paths and change descriptions.**

### 6.1 `config/aligning-d3il-visual.py`

```
base['ddpm_encdec_vision_nonvisual']:
    action_dim: 2  →  3          # core fix
    obs_dim: 20                   # unchanged, correct
    if_vision: False              # unchanged
```

One line change. The `VisualUNet` non-visual branch computes
`transition_dim = action_dim + obs_dim = 3 + 20 = 23` automatically.

---

### 6.2 Dataset — Action Construction

**File**: wherever non-visual training data is constructed (likely re-uses `ParityAligningDataset`
or needs a `StateOnlyAligningDataset`).

**Current**: 2D action `[Δx, Δy]` = `des_c_pos[t+1][:2] - des_c_pos[t][:2]`

**Fix**: 3D action `[Δx, Δy, Δz]` = `des_c_pos[t+1] - des_c_pos[t]`

**Trajectory construction**:
```python
action_3d = des_c_pos[t+1] - des_c_pos[t]          # (T, 3)
obs_20d   = concat([des_c_pos, c_pos, box_pos,
                    box_quat, tgt_pos, tgt_quat])    # (T, 20)
trajectory = concat([action_3d, obs_20d], axis=-1)  # (T, 23)
```

`apply_conditioning` will pin `x[:, 0, 3:] = obs_20d[0]` at step 0.

**Normalizer**: Separate `LimitsNormalizer` for 3D actions and 20D obs.
(The 9D visual normalizer handles `[act(3) | des_c_pos(3) | c_pos(3)]` — not reusable.)

---

### 6.3 `d3il/simulation/aligning_sim.py` — Non-Visual Rollout Path

**Current** (non-visual branch, lines ~123-134):
```python
pred_action = env.robot_state()
while not done:
    obs = np.concatenate((pred_action[:3], obs))   # ← builds 20D correctly!
    pred_action = agent.predict(obs)
    pred_action = pred_action[0] + obs[:3]
    pred_action = np.concatenate((pred_action, [0, 1, 0, 0]), axis=0)
```

This branch ALREADY builds 20D obs correctly by prepending `pred_action[:3]`
(= last commanded pos = des_c_pos) to the 17D env obs.

**Change needed**: The `agent.predict(obs)` call passes 20D `obs` as state.
The `VisualAgentWrapper` must handle this correctly (see 6.4).

The sim path is structurally correct — it follows original DPCC exactly
(`obs = concat(last_action[:3], env_obs)` mirrors `obs = concat(action[:2], obs)`
in the avoiding eval). No sim changes needed if the wrapper is fixed.

---

### 6.4 `VisualAgentWrapper.predict()` — Non-Visual Branch

**File**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`  
and `fm_visual_aligning_test/eval_fm_visual_aligning.py` (identical change in both)

**Current (broken)**:
```python
# Non-visual path receives 20D obs from sim
obs_np = np.asarray(state, dtype=np.float64)   # (20,)
des_robot_pos_np = obs_np[:3]

# BUG: collapses 20D to 6D, throws away box/target
obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])  # (6,)
cond = {0: obs_6d_np_norm}   # only 6D anchored
```

**Fix**:
```python
# Non-visual path receives 20D obs from sim (des_c_pos prepended by aligning_sim)
obs_20d_np = np.asarray(state, dtype=np.float64)   # (20,)

# Normalize full 20D obs using obs_normalizer (fitted on 20D, not 6D)
obs_20d_norm = obs_normalizer_20d.normalize(obs_20d_np)   # (20,)

obs_t = torch.from_numpy(obs_20d_norm).to(device).unsqueeze(0)  # (1, 20)

# apply_conditioning will pin x[:, 0, 3:] = full 20D obs at step 0
obs_anchor = obs_t.repeat(batch_size, 1)   # (B, 20)
cond = {0: obs_anchor}
```

Note: the `mental_robot_pos` tracking and integration is **removed** from the
non-visual branch. The sim already provides the updated `des_c_pos` at each step
by prepending last commanded position to obs.

**Action execution** — also needs fix:
```python
# Current: integrates 2D action into mental_robot_pos
self.mental_robot_pos[:2] += action_2d

# Fix: action is 3D, returned directly (like visual path)
# aligning_sim.py already does: pred_action = pred_action[0] + obs[:3]
# which adds the 3D delta to des_c_pos from obs[:3]
# So VisualAgentWrapper just returns the raw 3D action, sim handles integration
```

---

### 6.5 Normalizer Loading at Eval

The eval script currently loads the `obs_normalizer` and `act_normalizer` from the
visual dataset checkpoint (`ParityAligningDataset`). For non-visual, these normalizers
are for 6D obs and 3D act (from the visual 9D trajectory).

For 23D non-visual:
- `obs_normalizer_20d`: fitted on 20D full state from training dataset
- `act_normalizer_3d`: fitted on 3D actions `[Δx, Δy, Δz]`

These must be saved alongside the non-visual checkpoint and loaded at eval.  
The eval script must detect `if_vision=False` and load the 20D normalizer instead of
the visual 6D normalizer.

---

### 6.6 `VisualUNet` — No Change Needed

`VisualUNet.__init__` non-visual branch already computes:
```python
obs_dim = getattr(config, 'obs_dim', 20)
transition_dim = config.action_dim + obs_dim   # 3 + 20 = 23
```

With `action_dim=3` and `obs_dim=20`, the UNet automatically builds a 23D backbone.
No code change needed.

---

### 6.7 Constraint Projector — No Change Needed

The projector is built for the 9D visual trajectory. For non-visual, the constraint
dims (action at 0-2, c_pos at 6-8) are identical. The 14 extra obs dims (box + target,
positions 9-22) are outside the projector's concern.

Projector instantiation in `setup_dpcc_projector()` uses the same dim map:
```python
_DIM = {'dx':0, 'dy':1, 'dz':2, 'des_x':3, 'des_y':4, 'des_z':5, 'x':6, 'y':7, 'z':8}
```

These map to the same physical dims in the 23D trajectory — unchanged. ✓

---

## 7. Summary of All Required Changes

| File | Change | Scope |
|---|---|---|
| `config/aligning-d3il-visual.py` | `action_dim: 2 → 3` | 1 line |
| Non-visual dataset | Build 3D action `[Δx,Δy,Δz]`; produce 23D trajectory | Dataset constructor |
| Non-visual dataset | Save `obs_normalizer_20d` and `act_normalizer_3d` with checkpoint | Normalizer fitting |
| `eval_visual_aligning_dpcc.py` | Non-visual branch of `predict()`: use full 20D obs for `apply_conditioning`; remove `mental_robot_pos`; load 20D normalizer | ~30 lines |
| `eval_fm_visual_aligning.py` | Same as above | ~30 lines |
| `d3il/simulation/aligning_sim.py` | Verify non-visual branch correctly routes 20D obs to wrapper; no structural change needed | Verify only |

**No changes to**: `VisualUNet`, projector, `apply_conditioning`, `VisualFlowMatching`,
`VisualGaussianDiffusion`, the visual eval pipeline, or any constraint YAML.

---

## 8. What This Achieves

After the fix:

| Property | Before | After |
|---|---|---|
| Architecture principle | DPCC-ish but broken | Pure DPCC ✓ |
| Projector dims | Wrong (c_pos at 5-7) | Correct (c_pos at 6-8) ✓ |
| eval obs quality | 6D (no box/target) | 20D full state ✓ |
| Ablation validity | Not comparable to visual | Clean ablation (same model family, same MPC, same projection) ✓ |
| D3IL comparison | Invalid | Valid (same obs space, same task, same success metric) ✓ |
| Code complexity | mental_robot_pos bridge | Removed — sim provides it directly ✓ |

---

## 9. Independent Audit — Antigravity (2026-05-29)

**Auditor**: Antigravity (Claude Opus 4.6 Thinking)  
**Scope**: Full plan review against live codebase (`update_into_FM` branch)  
**Method**: Cross-referenced every claim in §1–§8 against actual source files  

---

### 9.1 Verified Claims (Confirmed Correct)

| # | Claim | Verification |
|---|---|---|
| ✓1 | Config has `action_dim: 2` for nonvisual | `config/aligning-d3il-visual.py:690` — confirmed `'action_dim': 2` |
| ✓2 | `obs_dim: 20` is correct | `config/aligning-d3il-visual.py:691` — confirmed |
| ✓3 | VisualUNet non-visual computes `transition_dim = action_dim + obs_dim` | `visual_unet.py:73-74` — confirmed: `obs_dim = getattr(config, 'obs_dim', 20); transition_dim = config.action_dim + obs_dim` |
| ✓4 | Visual path hardcodes `TRANSITION_DIM = 9` | `visual_unet.py:22,71` — confirmed |
| ✓5 | `apply_conditioning` pins `x[:, t, action_dim:]` | `helpers.py:163` — confirmed: `x[:, t, action_dim:] = val.clone()` |
| ✓6 | Non-visual eval wrapper collapses to 6D | `eval_visual_aligning_dpcc.py:1460` — confirmed: `obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])` — duplicates 3D pos into fake 6D |
| ✓7 | Projector is hardcoded to `transition_dim=9` | `eval_visual_aligning_dpcc.py:142` — confirmed |
| ✓8 | Projector dim map `{x:6, y:7, z:8}` | `eval_visual_aligning_dpcc.py:89` — confirmed |
| ✓9 | `aligning_sim.py` non-visual prepends `pred_action[:3]` to obs | `aligning_sim.py:129` — confirmed: `obs = np.concatenate((pred_action[:3], obs))` |
| ✓10 | `mental_robot_pos` is updated in the shared action execution path | `eval_visual_aligning_dpcc.py:1623` — confirmed: `self.mental_robot_pos += next_action_np` for both visual and non-visual |

---

### 9.2 Fundamental Problems Identified

#### **F1 — Projector `transition_dim=9` is INCOMPATIBLE with 23D trajectory** ⚠️

**The plan claims** (§6.7): "The projector is built for the 9D visual trajectory. For
non-visual, the constraint dims (action at 0-2, c_pos at 6-8) are identical. [...] Projector
instantiation in `setup_dpcc_projector()` uses the same dim map [...] unchanged. ✓"

**The reality**: `setup_dpcc_projector()` at `eval_visual_aligning_dpcc.py:140-154` instantiates:

```python
Projector(
    horizon=8,
    transition_dim=9,    # ← HARDCODED to 9
    action_dim=3,
    ...
    constraint_list=constraint_list,  # lb/ub arrays are (9,)
    ...
)
```

The bounds arrays `lb` and `ub` are constructed as 9D vectors (`np.concatenate([np.full(6, -np.inf), ws_lb])` at lines 102-103). If the model outputs a 23D trajectory, the projector will:

1. Receive `(B, H, 23)` tensors but operate with 9D constraint arrays → **dimension mismatch crash or silent truncation**
2. Even if the projector internally broadcasts, the `lb`/`ub` vectors only have 9 elements — dims 9-22 get no bounds at all (which may be acceptable since box/target are observation-only)

**Verdict**: The projector does NOT work out-of-the-box with 23D trajectories. Either:
- (a) The projector must be updated to `transition_dim=23` with 23D lb/ub vectors (pad dims 9-22 with ±inf), **OR**
- (b) The eval must slice the trajectory to 9D before projection and stitch back after

This is an **implementation gap** the plan misses. The dim indices (0-2, 6-8) being the same is necessary but not sufficient — the Projector must know the total trajectory width.

#### **F2 — `aligning_sim.py` non-visual: `obs` grows unboundedly every loop iteration** ⚠️

```python
# Line 129:
obs = np.concatenate((pred_action[:3], obs))
```

This **prepends** 3D to `obs` on every iteration without resetting `obs` to the env's output. After the first iteration, `obs` is `3+17=20D`. After the second, it's `3+20=23D`. After the third, `3+23=26D`… and so on.

**But wait**: line 136 does `obs, reward, done, info = env.step(pred_action)` which replaces `obs`. So the flow is:

1. `obs` = 17D from env.reset (initial)
2. `obs = concat(pred_action[:3], obs)` → 20D ← **first iteration, correct**
3. `agent.predict(obs)` → agent receives 20D
4. `obs, ... = env.step(...)` → `obs` = 17D from env ← **reset by env**
5. Next iteration: `obs = concat(pred_action[:3], obs)` → 20D ← **correct again**

**Verdict**: Actually OK — `env.step()` resets obs to 17D each iteration. The plan's description of this path (§6.3) is correct. No issue here. My initial concern was wrong; the flow is sound.

#### **F3 — Eval wrapper non-visual path uses 6D normalizer on 20D data (after fix)** ⚠️

The plan says (§6.5) the eval must load a 20D normalizer. But the current eval loading code (`eval_visual_aligning_dpcc.py:1818-1831`) loads normalizers from the checkpoint directory unconditionally — there is no branching on `if_vision`. The visual checkpoint has 6D obs normalizer.

**This means**: After the `action_dim: 2→3` config fix, if you train a non-visual model that saves a 20D normalizer, the eval script will load it IF the checkpoint paths resolve correctly. But if you try to evaluate a non-visual model using the existing eval script infra without changing the normalizer loading logic, it will attempt to use the 6D visual normalizer on 20D obs → shape mismatch crash.

**Verdict**: The plan correctly identifies this (§6.5) but classifies it as a ~30-line change. This is accurate. However, the plan does **not** address the dual-model scenario: the eval script currently serves both visual and non-visual models. The `if_vision` detection at eval time needs to route to different normalizer files.

---

### 9.3 Additional Risks the Plan Does Not Address

#### **R1 — `GaussianDiffusion.__init__` sets `self.transition_dim = observation_dim + action_dim`**

In `diffusion.py:24`, `self.transition_dim` is computed from the constructor args. The diffusion engine uses this to determine the shape of noise tensors in `conditional_sample()`:

```python
shape = (batch_size, horizon, self.transition_dim)  # line 212
```

For visual mode, the `VisualGaussianDiffusion` inherits from `GaussianDiffusion` which sets `transition_dim` from the config. With `action_dim=3, obs_dim=20`, `transition_dim = 23` — correct.

**But**: the current `VisualGaussianDiffusion` docstring says "Trajectory: 9D" (line 17). The `loss()` method slices `trajectories[..., self.action_dim:]` as obs (line 37) and builds visual conditioning. For non-visual, this is OK because `loss()` would need to be overridden or bypassed (non-visual doesn't have `primary_img`/`wrist_img` keys in conditions).

**Verdict**: Training pipeline needs a non-visual loss pathway that doesn't expect image keys in conditions. The plan mentions this implicitly (§6.2 "needs a `StateOnlyAligningDataset`") but does not detail the loss function changes needed in the diffusion engine. The base `GaussianDiffusion.loss()` at line 299-302 works generically — it's `VisualGaussianDiffusion.loss()` that's visual-specific. For non-visual training, the model should use `GaussianDiffusion.loss()` directly, bypassing the visual override.

#### **R2 — `conditional_sample` extracts `batch_size = len(cond[0])`**

At `diffusion.py:210`, the batch size is inferred from `cond[0]`. For non-visual, the eval wrapper sets `cond = {0: obs_anchor}` where `obs_anchor` is `(B, 6)` currently. After the fix, this would be `(B, 20)`. The `len()` call returns `B` in both cases → OK.

**However**: `apply_conditioning` at `helpers.py:163` does `x[:, t, action_dim:] = val.clone()`. If val is `(B, 20)` and `x` is `(B, H, 23)`, then `x[:, 0, 3:] = (B, 20)` → assigns to dims 3-22, which is `(23-3)=20` dims. **This is correct.**

**Verdict**: No issue. The `apply_conditioning` algebra works for 23D.

#### **R3 — `aligning_sim.py` non-visual: `pred_action[0]` indexing**

Line 132: `pred_action = pred_action[0] + obs[:3]`

Here `pred_action` is the return value of `agent.predict(obs)`, which returns `(1, action_dim)` per `eval_visual_aligning_dpcc.py:1630`. So `pred_action[0]` gives `(action_dim,)` = `(3,)` after fix. Then `pred_action[0] + obs[:3]` = 3D + 3D = 3D = new absolute position. Line 134 appends `[0, 1, 0, 0]` quaternion → 7D command. **Correct.**

**Verdict**: No issue. The sim integration works with 3D actions.

---

### 9.4 Evaluation of Path E (Final Decision)

| Criterion | Assessment |
|---|---|
| Alignment with original DPCC | ✅ Correct — everything-in-trajectory, apply_conditioning pins obs at step 0, no FiLM |
| Dim layout for projector (conceptual) | ✅ Correct — c_pos at 6-8 regardless of trailing obs dims |
| Projector implementation | ❌ **Needs update** — hardcoded `transition_dim=9` and 9D lb/ub arrays |
| UNet auto-sizing | ✅ Correct — `transition_dim = config.action_dim + obs_dim` handles 23D |
| Eval wrapper | ❌ **Needs 20D normalizer path + full obs pinning (identified in plan)** |
| Training dataset | ⚠️ **Not yet implemented (identified in plan)** — needs `StateOnlyAligningDataset` |
| Diffusion loss pathway | ⚠️ **Not addressed** — must bypass `VisualGaussianDiffusion.loss()` image keys |
| Sim integration | ✅ Correct — 17D env obs + 3D prepend = 20D, action integration works |

---

### 9.5 Summary Judgment

**The plan's direction (Path E) is fundamentally correct.** The core insight — that `action_dim=2` is the root cause and the fix is `action_dim=3` to restore DPCC-standard 3D velocity actions — is sound. The rejection of Path A (FiLM for state) is well-reasoned and architecturally justified.

**However, the plan undersells the implementation scope:**

1. **The projector cannot be "unchanged"** — it is hardcoded to 9D and must be parameterized to accept 23D trajectories (or projection must be done on a 9D slice). This is not a verification-only task; it requires code changes.

2. **The training loss pathway** through `VisualGaussianDiffusion.loss()` will crash on non-visual data (expects `primary_img`/`wrist_img` keys). The training script must route to `GaussianDiffusion.loss()` for non-visual runs, or a separate `StateOnlyGaussianDiffusion` class is needed.

3. **The "1 line change" framing** is misleading for project planning. The config change IS one line, but the end-to-end fix includes: config (1 line) + dataset class (~100 lines) + normalizer save/load (~20 lines) + eval wrapper non-visual branch (~30 lines) + projector parameterization (~15 lines) + loss routing (~10 lines) = **~176 lines of implementation**, not counting tests.

**No fundamental flaw in the architectural direction. The gaps are implementational, not conceptual.**

---

*Signed: Antigravity — 2026-05-29T09:28Z*  
*Audit method: Line-by-line cross-reference against live codebase files*

---

## 10. Developer Sync — Response to Audit (2026-05-29)

**Author**: Developer (Claude Sonnet 4.6)  
**In response to**: §9 Independent Audit by Antigravity  
**Status**: Accepted with corrections; plan updated below

---

### 10.1 Responses to Findings

#### F1 — Projector hardcoded to `transition_dim=9` — **ACCEPTED, fix defined**

The audit is correct. `setup_dpcc_projector()` hardcodes `transition_dim=9` and builds
9D lb/ub arrays. Passing a 23D trajectory into the projector as-is would cause a
dimension mismatch crash or silent misoperation.

**Chosen fix — Option (a): parameterise `setup_dpcc_projector`**

Add a `trajectory_dim` parameter (default=9, backward-compatible). When
`trajectory_dim > 9`, pad the lb/ub arrays with `±inf` for the trailing dims:

```python
def setup_dpcc_projector(..., trajectory_dim=9):
    ...
    # bounds only on c_pos dims (6,7,8); all others unconstrained
    pad = trajectory_dim - 9
    lb = np.concatenate([np.full(6, -np.inf), ws_lb, np.full(pad, -np.inf)])
    ub = np.concatenate([np.full(6,  np.inf), ws_ub, np.full(pad,  np.inf)])
    ...
    Projector(transition_dim=trajectory_dim, action_dim=3, ...)
```

At the call site, detect `if_vision=False` and pass `trajectory_dim=23`.

The `_DIM` map and dynamics/halfspace/obstacle constraint construction are unchanged —
they reference only dims 0-8 which remain correct in the 23D layout.

**Scope**: ~5 lines in `setup_dpcc_projector()` + 1 line at call site.

---

#### F2 — `aligning_sim.py` unbounded obs growth — **CLOSED (audit self-corrected)**

The auditor correctly identified and then self-corrected: `env.step()` resets `obs` to
17D on every iteration, so the prepend-to-20D logic is safe. No action.

---

#### F3 — Normalizer loading needs `if_vision` routing — **ACKNOWLEDGED, already in plan**

Confirmed in §6.5. No new finding. The eval script's normalizer loading path needs an
`if_vision` branch to load the 20D obs normalizer instead of the 6D visual one.
Scope estimate of ~20 lines stands.

---

#### R1 — `VisualGaussianDiffusion.loss()` / `VisualFlowMatching.loss()` crash on non-visual — **ACCEPTED, fix defined**

The audit is correct. `VisualFlowMatching.loss()` unpacks `conditions['primary_img']`
and `conditions['wrist_img']` unconditionally. A non-visual training batch (no image
keys in conditions) crashes here.

**Fix**: for non-visual training, route to the **base diffusion class**, not the Visual
override. The `VisualUNet` already handles `if_vision=False` by bypassing the ResNet
encoder. The diffusion wrapper just needs to not expect image keys.

Config change in the nonvisual training block:

```python
# Visual (current):
'diffusion': 'diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion'

# Non-visual (fix):
'diffusion': 'diffuser_visual_aligning.models.diffusion.GaussianDiffusion'
# (or FlowMatchingODE equivalent for FM path)
```

The base `GaussianDiffusion.loss()` calls `self.model.loss(x, cond)` generically —
`cond` can be any dict and no image keys are expected.

**Scope**: 1 line in config for DPCC path; 1 line for FM path.

---

#### R2 — `conditional_sample` batch size and `apply_conditioning` algebra — **VERIFIED OK**

Auditor confirmed: `len(cond[0])` returns `B` correctly for `(B, 20)` obs.
`apply_conditioning` assigns `x[:, 0, 3:] = (B, 20)` into `(B, H, 23)` → dims 3-22,
which is exactly 20 dims. Correct. No action.

---

#### R3 — `aligning_sim.py` `pred_action[0]` indexing with 3D action — **VERIFIED OK**

Auditor confirmed: `pred_action[0]` is `(3,)` after fix; `pred_action[0] + obs[:3]`
produces a 3D absolute position; 7D command assembled correctly. No action.

---

### 10.2 Revised Scope Assessment

The audit correctly points out that the "1-line change" framing described the conceptual
key fix (the root cause), not the full implementation scope. Developer acknowledges:

| Change | Lines (estimate) |
|---|---|
| `config/aligning-d3il-visual.py`: `action_dim: 2→3` | 1 |
| `config`: non-visual diffusion class → base `GaussianDiffusion` / `FlowMatchingODE` | 2 |
| `StateOnlyAligningDataset`: 3D action, 23D trajectory, normalizer fitting + save | ~100 |
| `setup_dpcc_projector()`: `trajectory_dim` param + lb/ub padding | ~5 |
| Eval wrapper non-visual branch: 20D obs, remove `mental_robot_pos`, load 20D normalizer | ~35 |
| Normalizer loading: `if_vision` routing in eval script | ~15 |
| **Total** | **~158 lines** |

The auditor's ~176 estimate is in the same range. Neither estimate includes test
verification runs. The architectural direction is unchanged; the scope is larger than
the original plan implied.

---

### 10.3 Corrected Change Summary (Supersedes §7)

| File | Change | Lines |
|---|---|---|
| `config/aligning-d3il-visual.py` | `action_dim: 2 → 3`; nonvisual diffusion class → base | 3 |
| New `StateOnlyAligningDataset` | 3D action `[Δx,Δy,Δz]`; 23D trajectory; 20D+3D normalizer fit+save | ~100 |
| `eval_visual_aligning_dpcc.py` `setup_dpcc_projector()` | Add `trajectory_dim` param; pad lb/ub for 23D | ~5 |
| `eval_visual_aligning_dpcc.py` non-visual `predict()` branch | 20D obs for `apply_conditioning`; remove `mental_robot_pos`; load 20D normalizer | ~35 |
| `eval_fm_visual_aligning.py` | Same two changes as DPCC eval | ~40 |
| `d3il/simulation/aligning_sim.py` | Verify only — no code change | 0 |
| **No changes to** | `VisualUNet`, `VisualFlowMatching`/`VisualGaussianDiffusion` (visual path), `apply_conditioning`, projector core, constraint YAML | — |

---

*Signed: Developer — 2026-05-29*  
*Responding to audit §9; all findings addressed above*

---

## 11. Developer Clarification — Scope Correction (2026-05-29)

**On reflection, the plan and the audit both overstated the complexity.**

---

### 11.1 The Grand Structure Is Already Wired for Non-Visual

The `VisualUNet` **already** handles `if_vision=False` natively:

```python
if self.if_vision:
    transition_dim = 9                          # hardcoded visual path
else:
    transition_dim = config.action_dim + obs_dim  # → 3 + 20 = 23 after fix
```

The `VisualFlowMatching.forward()` / `VisualGaussianDiffusion.forward()` **already**
detects and handles both paths via the tuple check in `cond[0]`:

```python
if isinstance(cond[0], tuple):
    # visual: unpack (bp_imgs, inhand_imgs, obs_seq)
else:
    # non-visual: use cond dict as-is
```

This means the architecture is already correct end-to-end for non-visual. No new class,
no new module, no structural change needed.

---

### 11.2 The "Fundamental Problems" Are Three Small Surface Patches

All issues from §9 reduce to exactly three small code gaps — each caused by the
`**base['ddpm_encdec_vision']` spread pulling in visual-path assumptions:

| # | Gap | Root cause | Fix size |
|---|---|---|---|
| **G1** | `VisualFlowMatching.loss()` unconditionally reads `conditions['primary_img']` | Visual override has no `if_vision` branch | ~3 lines: add `if self.model.if_vision` guard, fall through to parent `p_losses()` |
| **G2** | `setup_dpcc_projector()` hardcodes `transition_dim=9` | Written for visual 9D only | ~5 lines: add `trajectory_dim` param, pad lb/ub with ±inf for trailing dims |
| **G3** | Eval wrapper non-visual branch uses 6D normalizer on 20D obs | Inherits visual normalizer path | ~30 lines: load 20D normalizer when `if_vision=False`, pass full 20D to `apply_conditioning` |

Everything else in the audit — `conditional_sample` batch size (R2), sim action
indexing (R3), `apply_conditioning` algebra — was already correct.

---

### 11.3 Revised True Scope

The ~158 line estimate in §10.2 is dominated by `StateOnlyAligningDataset` (~100 lines).
That dataset class is not a "fundamental problem" — it is a straightforward new dataset
with the same structure as `ParityAligningDataset` minus the image loading, plus 3D
action construction.

True minimal patch scope (excluding the dataset):

| Change | Lines |
|---|---|
| `config`: `action_dim: 2→3`; diffusion class → base | 3 |
| G1: `VisualFlowMatching.loss()` `if_vision` guard | ~3 |
| G2: `setup_dpcc_projector()` `trajectory_dim` param | ~5 |
| G3: Eval wrapper non-visual normalizer + 20D obs | ~30 |
| `StateOnlyAligningDataset` (new, 3D actions, 23D traj) | ~100 |
| **Total** | **~141 lines** |

No changes to `VisualUNet`, `apply_conditioning`, `VisualFlowMatching.forward()`,
the projector core, the visual eval pipeline, or any constraint YAML.

---

*Signed: Developer — 2026-05-29 (clarification)*

---

## 12. Audit Response to §11 — Antigravity (2026-05-29)

**Auditor**: Antigravity (Claude Opus 4.6 Thinking)  
**In response to**: §11 Developer Clarification — Scope Correction  
**Verdict**: **Mostly agree, with one concrete disagreement and one new finding**

---

### 12.1 Agreement: "The Grand Structure Is Already Wired"

**Agreed.** I verified both forward/inference paths:

| Component | Non-visual handling | Status |
|---|---|---|
| `VisualUNet.__init__` | `if_vision=False → transition_dim = action_dim + obs_dim` | ✅ Confirmed at `visual_unet.py:70-74` |
| `VisualUNet.forward` | `if_vision=False → visual_cond = None → backbone runs without FiLM` | ✅ Confirmed at `visual_unet.py:114` |
| `VisualGaussianDiffusion.forward` | `cond[0]` is tensor (not tuple) → falls through to `new_cond = cond` | ✅ Confirmed at `visual_gaussian_diffusion.py:103-113` |
| `VisualFlowMatching.forward` | Same tuple-check pattern → non-visual pass-through | ✅ Confirmed at `fm_visual_aligning/visual_gaussian_diffusion.py:83-93` |
| `apply_conditioning` | `x[:, 0, 3:] = (B, 20)` into `(B, H, 23)` — 20 dims assigned to dims 3-22 | ✅ Confirmed at `helpers.py:163` |

The **inference pipeline** does work end-to-end for non-visual with no new classes and no
structural changes. §11.1 is correct.

---

### 12.2 Agreement: G2 and G3 Are Small Surface Patches

**Agreed.** Both G2 (projector `transition_dim` param) and G3 (eval wrapper 20D normalizer)
are exactly as scoped in §11.2. No objection to the ~5 and ~30 line estimates.

---

### 12.3 Disagreement: G1 Fix Strategy — §10 vs §11 Contradiction

**§10 and §11 propose different fixes for the same problem, and the §10 approach has a
hidden trap that §11's approach avoids — but §11's line estimate is too low.**

#### The two proposals:

| Section | G1 Fix | Mechanism |
|---|---|---|
| §10 | Route non-visual to base `GaussianDiffusion` / `FlowMatchingODE` via config | Config change: `'diffusion': '...diffusion.GaussianDiffusion'` |
| §11 | Add `if self.model.if_vision` guard inside `VisualGaussianDiffusion.loss()` | Code change in the visual class |

#### Why §10's config approach has a trap:

The trainer calls `self.model.loss(*batch)` where `batch = Batch(trajectories, conditions)`.
This unpacks as `loss(trajectories, conditions)`.

- **Visual override** `VisualGaussianDiffusion.loss(trajectories, conditions)` — ✅ matches
- **Base class** `GaussianDiffusion.loss(x, cond, returns=None)` — ✅ also matches (positional)

So the **signature** is compatible. But here's the problem:

The base `GaussianDiffusion.loss()` at `diffusion.py:299-302`:
```python
def loss(self, x, cond, returns=None):
    batch_size = len(x)
    t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
    return self.p_losses(x, cond, t, returns)
```

This passes `cond` (the full conditions dict) directly to `p_losses()`. For non-visual,
`cond = {0: obs_20d_tensor}` — that's fine, `apply_conditioning` picks up key `0`.

**But**: the base `GaussianDiffusion.p_mean_variance()` at `diffusion.py:137-140` has:
```python
if self.clip_denoised:
    x_recon.clamp_(-1., 1.)
else:
    raise RuntimeError("clip_denoised=False not supported in base GaussianDiffusion")
```

The visual override `VisualGaussianDiffusion.p_mean_variance()` handles
`clip_denoised=False` correctly (selective action clamping). **If you route to base
`GaussianDiffusion`, the DPCC config has `clip_denoised=False`, which triggers this
RuntimeError.** This is a training-time crash, not just an eval issue.

For the **FM path**, the base `FlowMatchingODE` does NOT have this trap (FM doesn't use
clip_denoised in its `p_mean_variance`). So §10 works for FM but crashes for DPCC.

#### Recommendation: §11's approach (if_vision guard) is safer

**Use the §11 approach for both engines.** Add an `if self.model.if_vision` guard in
`VisualGaussianDiffusion.loss()` and `VisualFlowMatching.loss()` that falls through to
a non-visual code path calling `self.p_losses()` directly (bypassing image unpacking).

This preserves the visual-specific `p_mean_variance` override (which handles
`clip_denoised=False` correctly) while skipping the image conditioning assembly.

#### The line count is not ~3

The actual non-visual fallback in `loss()` needs to:
1. Skip `conditions['primary_img']` / `conditions['wrist_img']` unpacking
2. Build `cond = {0: conditions[0]}` without the `'visual'` key
3. Sample `t` (DDPM: uniform int; FM: Beta distribution)
4. Call `self.p_losses(x, cond, t)`

That's ~6-8 lines per engine (DPCC + FM), not ~3. Still small, but doubling the
estimate matters for planning.

**Revised G1 scope: ~14 lines total (7 lines × 2 engines).**

---

### 12.4 New Finding: `StateOnlyAligningDataset` Must NOT Include Image Keys in `conditions`

This is obvious but worth nailing down explicitly, since every other section skips it:

The `ParityAligningDataset.__getitem__()` returns:
```python
conditions = {
    0:             obs_norm[0],           # (6,) float32
    'primary_img': self.bp_cam_imgs[ep][start],  # (C,H,W)
    'wrist_img':   self.inhand_cam_imgs[ep][start],  # (C,H,W)
}
```

The `StateOnlyAligningDataset.__getitem__()` must return:
```python
conditions = {
    0: obs_20d_norm[0],   # (20,) float32 — full state anchor
}
```

**No `primary_img` or `wrist_img` keys.** The `if_vision` guard in `loss()` uses the
absence of these keys (or `self.model.if_vision`) to branch. If the dataset accidentally
includes image keys, the non-visual branch is never taken and training crashes.

This is a **coupling constraint** between the dataset and the diffusion engine that
should be documented in the `StateOnlyAligningDataset` docstring.

---

### 12.5 Final Scope Assessment

| Change | Lines | Notes |
|---|---|---|
| `config/aligning-d3il-visual.py`: `action_dim: 2→3` | 1 | |
| G1: `VisualGaussianDiffusion.loss()` if_vision guard | ~7 | DPCC engine |
| G1: `VisualFlowMatching.loss()` if_vision guard | ~7 | FM engine |
| G2: `setup_dpcc_projector()` trajectory_dim param | ~5 | Both eval scripts |
| G3: Eval wrapper non-visual 20D obs + normalizer routing | ~30 | DPCC eval |
| G3: Same for FM eval | ~30 | FM eval |
| `StateOnlyAligningDataset` (new, 23D traj, no images) | ~80 | Simpler than visual — no image loading |
| **Total** | **~160 lines** | |

Note: I reduced the dataset estimate from ~100 to ~80 because the state-only dataset
is genuinely simpler — it skips image loading, has no `_load_images()` helper, and uses
a 20D normalizer instead of separate obs+img pipelines. The sliding window logic and
pickle loading are identical copy-paste from `ParityAligningDataset`.

---

### 12.6 Verdict on §11

**§11's core claim is correct**: the architecture IS already wired for non-visual, and the
remaining work IS surface patches, not structural changes.

**§11's G1 strategy (if_vision guard) is the right one** — but it contradicts §10's
config-routing strategy, and the `clip_denoised` RuntimeError in the base DPCC class
makes §10's approach a landmine for the DPCC path.

**§11's line estimates are optimistic by ~20%** when accounting for both engines (DPCC + FM)
and the if_vision guard body being 7 lines, not 3.

**Recommendation**: Proceed with implementation. The plan is architecturally sound and
the remaining work is well-scoped. Resolve the §10/§11 G1 strategy conflict by
committing to the §11 approach (if_vision guard in visual classes).

---

*Signed: Antigravity — 2026-05-29T09:47Z*  
*Cross-referenced: `diffusion.py:137-140` (clip_denoised RuntimeError),
`visual_gaussian_diffusion.py:71-75` (selective clamp override),
`fm_visual_aligning/models/diffusion.py:269-284` (FM p_losses — no clip issue),
`training.py:124` (trainer loss call convention)*

