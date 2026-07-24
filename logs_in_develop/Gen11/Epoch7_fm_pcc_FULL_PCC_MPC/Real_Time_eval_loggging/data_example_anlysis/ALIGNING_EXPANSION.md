# Aligning Expansion: Who Designed What

**Verified 2026-06-26 against arXiv:2412.09342 full text.**

> DPCC paper evaluates on the **avoiding task only**, pure state (4D), no cameras.
> Everything "visual" and everything "aligning" in this repo is **FM-PCC's own design**.

---

## 1. What DPCC actually covers (paper-verified)

DPCC §6.1 (arXiv:2412.09342):

> "The task is for a robot manipulator to reach a line (green) with its end-effector without
> colliding with one of the six obstacles (red). The state s_t ∈ ℝ⁴ consists of the current
> and desired end-effector positions in the 2D plane, and the action a_t ∈ ℝ² contains the
> desired Cartesian velocities, which are sent to a low-level controller."
> "we sample a batch of B=4 trajectories with horizon length H+1=8"

**DPCC experimental scope:**

| Property | Value |
|---|---|
| Task | D3IL **avoiding** only |
| Input | Pure state — NO cameras |
| State dim | 4D [des_xy(2) \| c_xy(2)] |
| Action dim | 2D Δdes_xy |
| Trajectory | 6D [act(2) \| des_xy(2) \| c_xy(2)] |
| Horizon | H=8 |
| FiLM | NONE |

The `_dpcc` suffix in our eval file names (e.g. `eval_visual_aligning_dpcc.py`) means
"using the DPCC-style PCC projector architecture," NOT that it replicates the original
DPCC paper. Everything visual and everything aligning is FM-PCC's own work.

---

## 2. Full lineage — who owns what

```
D3IL (KIT)
├── Task: avoiding  — [des_xy(2)|c_xy(2)] = 4D state, 2D action
│     data, env, reward, IK controller
│     ├── non-visual DDPM: action-level, conditioned on 4D state
│     └── visual DDPM: bp_cam + inhand_cam → MultiImageObsEncoder → fused obs input (early fusion)
└── Task: aligning  — [des_c_pos(3)|c_pos(3)|push_box(3)|push_box_q(4)|tgt(3)|tgt_q(4)] = 20D state, 3D action
      data, env, reward, IK controller
      ├── non-visual DDPM: action-level, conditioned on 20D state
      └── visual DDPM: bp_cam + inhand_cam → MultiImageObsEncoder → fused obs input (early fusion)
          NOTE: D3IL visual uses EARLY FUSION — cameras encoded into obs, passed as state to DDPM.
                NOT FiLM into the denoiser. The denoiser MLP sees one fused vector.

DPCC (Römer et al. arXiv:2412.09342)
└── Takes D3IL AVOIDING (non-visual)
      grafts Janner TemporalUnet+GaussianDiffusion onto D3IL's 4D state format
      adds PCC projector (SLSQP + deriv constraint binding c_xy to act)
      trajectory = 6D [act(2)|des_xy(2)|c_xy(2)]
      NO visual, NO aligning, NO FiLM

FM-PCC (us) — extends DPCC principles to new domains
├── FM avoiding (non-visual):     6D traj,  same as DPCC but FlowMatching
├── FM visual avoiding:           6D traj + camera (FiLM into UNet), FM-PCC addition
│     NOTE: D3IL also has visual avoiding — we differ in HOW visual is used (FiLM vs early fusion)
├── FM visual aligning:           9D traj [act(3)|des_c_pos(3)|c_pos(3)] + camera (FiLM into UNet), FM-PCC addition
│     NOTE: D3IL also has visual aligning — same cameras, different conditioning method
├── FM non-visual aligning:       23D traj [act(3)|20D obs], FM-PCC addition (WRONG — see TODO)
└── FM UAV:                       12D traj [act(3)|p_des(3)|p(3)|v(3)], FM-PCC addition
```

---

## 3. H=8 horizon is Janner — D3IL has NO horizon concept

This is the clearest lineage question in the whole stack.

**D3IL DDPM** (`d3il/agents/models/diffusion/diffusion_policy.py`):
```python
shape = (batch_size, self.action_dim)   # single action — NO horizon dimension
action = self.p_sample_loop(state, goal, shape, ...)
```
D3IL diffuses a **single action** per call. No `horizon` parameter exists anywhere in
`ddpm_agent.py`. The eval loop calls `agent.predict(obs)` at every environment step and
gets one delta back. No planning window, no MPC, no trajectory.

**Janner diffuser** (`/workspaces/diffuser/diffuser/models/diffusion.py:45-50`):
```python
def __init__(self, model, horizon, observation_dim, action_dim, ...):
    self.horizon = horizon      # ← horizon is a FIRST-CLASS constructor argument

def conditional_sample(self, cond, horizon=None, ...):
    shape = (batch_size, horizon, self.transition_dim)   # (B, H, traj_dim)
```
The horizon `H` is baked into the UNet's temporal convolution architecture:
`TemporalUnet.__init__(..., horizon, ...)` — the conv layer sizes are computed from `H`.
You cannot change H at runtime without rebuilding the model.

**DPCC paper §6.1**: "we sample a batch of B=4 trajectories with horizon length H+1=8"
DPCC chose H=8 from Janner's framework and applied it to D3IL's data.

**Lineage of H=8**:
```
Janner (2022) — invented H-step trajectory diffusion, horizon as architecture param
    ↓
DPCC (2024) — used H=8 on D3IL avoiding data
    ↓
FM-PCC (us) — inherited H=8 across avoiding, aligning, UAV
```

D3IL contributed the data format and tasks. Janner contributed the H-step trajectory
planning (the "MPC" framing: plan H steps, execute step 0, replan). DPCC is the bridge
that merged them. FM-PCC extended that merged system to new domains.

---

## 4. D3IL original aligning — the data we adapt from

D3IL aligning dataset (`d3il/environments/dataset/aligning_dataset.py:62-84`):

```python
input_state = np.concatenate((robot_des_pos, robot_c_pos,          # (3+3)
                              push_box_pos, push_box_quat,          # (3+4)
                              target_box_pos, target_box_quat),     # (3+4)
                             axis=-1)                               # = 20D
vel_state   = robot_des_pos[1:] - robot_des_pos[:-1]               # 3D action (same vel_state pattern)
```

D3IL's own DDPM for aligning diffuses actions (3D delta), conditions on 20D obs at every
step, NO horizon, NO trajectory, NO projector. Clean but context-hungry: 14 of 20 dims are
box/target state that changes per episode.

**D3IL aligning eval (non-visual)** (`d3il/simulation/aligning_sim.py:123-136`):

```python
pred_action = env.robot_state()           # initial des_c_pos (3D)
while not done:
    obs = np.concatenate((pred_action[:3], obs))   # prepend accumulated des_pos → 3+20 = 23D
    pred_action = agent.predict(obs)               # DDPM outputs 3D delta
    pred_action = pred_action[0] + obs[:3]         # EXPLICIT integration: new_des = old_des + Δ
    obs, reward, done, info = env.step(...)
```

Box position is re-read fresh at every step from `env.step()`.

---

## 4. FM-PCC visual aligning — our own design

We chose to adapt DPCC's 6D avoiding approach to the 3D aligning task. Our design choices:

**Dataset** (`diffuser_visual_aligning/datasets/sequence.py:50-51`):
```python
OBS_DIM  = 6   # [des_c_pos(3), c_pos(3)] — dropped box/target dims
TRAJ_DIM = 9   # [act(3) | des_c_pos(3) | c_pos(3)]
```

We dropped the 14 box+target dims from the trajectory and replaced them with FiLM visual
conditioning. This was our design choice: cameras can see the box and target, so encode
them visually rather than as state dims in the Janner trajectory.

**FiLM** (`diffuser_visual_aligning/models/unet1d_temporal_cond.py:116-132`) — FM-PCC addition:
```python
self.cond_mlp = nn.Sequential(          # visual embedding → FiLM scale/shift
    nn.Linear(cond_dim, time_dim), nn.Mish(),
    nn.Linear(time_dim, time_dim), nn.Mish(),
    nn.Linear(time_dim, time_dim * 4),
)
```
FiLM modulates every UNet residual block. The cameras see where the box and target are;
the model learns to plan robot motion accordingly. Box/target never appear in the 9D tensor.

**Projector**: dynamics constraint binds `c_pos` (traj dims 6-8) ← `act` (dims 0-2):
```
('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])
```
Same structure as DPCC avoiding, just extended to 3D (from 2D).

**Integration** (`eval_visual_aligning_dpcc.py:1657`):
```python
self.mental_robot_pos += next_action_np   # same accumulator as DPCC avoiding
```

**Is this a faithful expansion of DPCC's avoiding principles?**

| Principle | DPCC avoiding (6D) | FM-PCC visual aligning (9D) | Faithful? |
|---|---|---|---|
| Janner [act\|obs] format | [act(2)\|des_xy(2)\|c_xy(2)] | [act(3)\|des_c_pos(3)\|c_pos(3)] | ✓ yes |
| action = Δdes (vel_state) | Δdes_xy (2D) | Δdes_c_pos (3D) | ✓ yes |
| integration: des += act | in eval loop | mental_robot_pos += act | ✓ yes |
| deriv constraint on c_pos | c_xy (dims 2-3 of 6D) | c_pos (dims 6-8 of 9D) | ✓ yes |
| obs at t=0 via apply_cond | {0: des_xy\|c_xy} (4D) | {0: des_c_pos\|c_pos} (6D) | ✓ yes |
| self-referential des in obs | yes (des_xy in state) | yes (des_c_pos in state) | ✓ same flaw |
| FiLM visual conditioning | NONE | added for box+target context | FM-PCC addition |
| extend to 3D space | no (2D) | yes (3D) | FM-PCC addition |

**Verdict**: The core DPCC principles (Janner format, delta integration, PCC deriv constraint,
apply_conditioning at t=0) are faithfully ported. FiLM and 3D extension are FM-PCC's own
design to handle the aligning context.

---

> ## ⚠ TODO / DESIGN BUG — Non-Visual Aligning Trajectory Dim is WRONG
>
> **Current code**: `StateOnlyAligningDataset` → 23D trajectory `[act(3)|20D obs]`
> (`diffuser_visual_aligning/datasets/sequence.py:183-253`,
> `train_visual_aligning_dpcc.py:169-171`, `eval_visual_aligning_dpcc.py:83`).
>
> **Why it's wrong**: the correct non-visual should mirror the visual path structurally —
> same 9D `[act(3)|des_c_pos(3)|c_pos(3)]` robot-only tensor, just **without FiLM**.
> The 20D expanded trajectory (including box_pos, box_quat, tgt_pos, tgt_quat in dims
> 9-22) is not a DPCC-faithful design: DPCC avoiding uses 6D (pure robot state, no goal
> dims). Non-visual aligning should follow suit: 9D (pure robot state), no goal dims.
>
> **The v2 architecture doc** (`Gen7_FMPCC_Viusal_Aligning/.../v2_NON_VISUAL_ALIGNING_ARCHITECTURE.md`)
> argued 23D was "necessary because FiLM has no equivalent." This reasoning is
> **rejected** — the correct DPCC-faithful non-visual is BLIND to goal: same tensor as
> visual, just without the visual conditioning pathway.
>
> **Recommended fix** — make trajectory dim switchable:
>
> ```python
> # In StateOnlyAligningDataset (or a new RobotOnlyAligningDataset):
> # Mode A (default): 9D robot-only — matches visual tensor, no goal info
> #   OBS_DIM=6, TRAJ_DIM=9, conditions={0: obs_6d}
> # Mode B (opt-in):  23D full-state — includes box/target, FM-PCC extension
> #   OBS_DIM=20, TRAJ_DIM=23, conditions={0: obs_20d}
> #
> # train_visual_aligning_dpcc.py: --non_visual_obs_dim 6 (default) or 20
> ```
>
> **Latest logs using 23D (now known wrong)**:
> `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_17_fix_non_visual/`
> and `NON_VISUAL_EXPL/v2_NON_VISUAL_ALIGNING_ARCHITECTURE.md` (2026-05-29).
> Any non-visual checkpoint trained with `StateOnlyAligningDataset` (23D) must be
> retrained to use the corrected 9D design.

---

## 5. FM-PCC non-visual aligning — current (wrong) 23D vs correct 9D

**Current (wrong) implementation** (`diffuser_visual_aligning/datasets/sequence.py:183-253`):
```python
OBS_DIM  = 20  # full [des_c_pos(3)|c_pos(3)|push_box(3)|push_box_q(4)|tgt_pos(3)|tgt_q(4)]
TRAJ_DIM = 23  # [act(3) | 20D obs]
conditions = {0: obs_20d_norm[0]}   # (20,) anchor
```

**Correct (target) design** — robot-only, same 9D as visual, no FiLM:
```python
OBS_DIM  = 6   # [des_c_pos(3)|c_pos(3)] — same as visual
TRAJ_DIM = 9   # [act(3) | 6D obs] — identical to visual, just no FiLM
conditions = {0: obs_6d_norm[0]}   # (6,) anchor — same as visual apply_conditioning
```

The correct non-visual is **goal-blind**: it knows robot kinematics (des_c_pos, c_pos)
but nothing about box or target. This makes it the clean ablation of the visual model —
same tensor, same projector, only the FiLM visual conditioning pathway removed.

**Comparison of all approaches:**

| Property | D3IL DDPM (non-visual) | FM-PCC 23D (WRONG) | FM-PCC 9D (CORRECT) |
|---|---|---|---|
| obs dim | 20D (goal always in obs) | 20D (goal in traj dims 9-22) | 6D (robot only, no goal) |
| goal info | yes, fresh every step | yes, stale after t=0 | NONE — blind |
| DPCC-faithful? | N/A (action-level) | NO (extra dims DPCC never had) | YES |
| projector ignores dims | N/A | yes (dims 9-22 float free) | no (all 9 dims constrained) |
| clean visual ablation? | no | no (different traj dim) | YES (same H×9 tensor) |
| box/target in projector | N/A | unconstrained | N/A (not in tensor) |

The 9D non-visual makes the visual↔non-visual comparison structurally clean: same model,
same trajectory tensor shape, same projector constraint wiring — only FiLM is switched off.

---

## 6. Why D3IL's own aligning differs from both our variants

D3IL aligning (20D, action-level DDPM) vs DPCC-style avoiding (4D/6D, trajectory diffuser):

The D3IL aligning task has per-episode CONTEXT (box and target position are random each
episode). D3IL handles this by keeping box+target in the full 20D obs at every step.

DPCC's approach for avoiding doesn't face this problem: the avoiding task has no per-episode
goal that needs to be communicated — the goal line is always in the same place. So DPCC
can use a short 4D state without needing explicit goal encoding.

When FM-PCC adapts DPCC to aligning, we face D3IL's original problem: the goal (target
position) varies per episode. Correct solutions:

| Solution | Variant | How goal is communicated | Status |
|---|---|---|---|
| FiLM | visual aligning (9D) | camera images encode box+target per episode | CORRECT |
| Robot-only, blind | non-visual aligning (9D, NO FiLM) | not communicated — pure ablation | CORRECT (target) |
| In trajectory | non-visual aligning (23D) | 20D obs includes goal, but projector ignores dims 9-22 | WRONG (current) |

The 23D approach is rejected because it introduces box/target dims that the projector
cannot constrain, breaks the clean visual/non-visual comparison, and is not DPCC-faithful.
If goal-aware non-visual is needed in future, the right mechanism is a separate goal
conditioning vector (FiLM with state input instead of camera), not expanding the trajectory.

Both solutions work but have different limitations:
- FiLM: goal is in camera; if FiLM doesn't extract goal position well → trajectory planning is blind to goal
- In-trajectory: goal is in dims 9-22 but projector ignores them; after t=0, box tracking is stale for H=8 steps

---

## 7. Correction to WHY_FM_KEEPS_PLANNING.md and prior session

The previous session incorrectly stated "DPCC is visual aligning ONLY." This is **false**:
- DPCC paper (arXiv:2412.09342) = non-visual avoiding, 4D state, 2D action
- Visual aligning = FM-PCC's own extension
- Non-visual aligning = FM-PCC's own extension
- Visual avoiding = FM-PCC's own extension

The `_dpcc` naming in our eval scripts refers to "using DPCC-style PCC projector," not
"implementing the original DPCC paper."

---

*Generated 2026-06-26. Verified against arXiv:2412.09342 full text. DPCC paper scope confirmed: avoiding only, pure state, 4D, no cameras.*
