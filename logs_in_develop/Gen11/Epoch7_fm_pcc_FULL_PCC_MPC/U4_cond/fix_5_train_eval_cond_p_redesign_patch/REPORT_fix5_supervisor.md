# Bug Report: DPCC Dynamics Constraint Mis-Binding and Free-Running Integration in UAV FM-PCC Eval

**Date**: 2026-06-28  
**Project**: FM-PCC — Gen11 Epoch 7 UAV Closed-Loop Evaluation  
**Author**: ghubliming  
**Fix Reference**: `fix_5_train_eval_cond_p_redesign_patch` (Gen11 E7 U4 F5)  
**Status**: Implemented and committed (2026-06-28)

---

## 1. Background

The FM-PCC pipeline trains a Flow Matching (FM) model to generate receding-horizon position
command trajectories for a UAV. During evaluation, a closed MuJoCo loop runs the trained
model in MPC style: at every FM step, the model receives a 9-D observation
`obs = [p_des | p | v]` and outputs a 3-D action `Δp_des`. A PID controller then tracks
the commanded position `p_des` across `decim` physics steps before the next FM query.

A **DPCC projector** (Diffusion Policy with Constraint-based Correction), inherited from
the D3IL robotics codebase, applies a convex QP correction to the FM's H-step trajectory
plan before execution. The correction enforces a **dynamics constraint** that ensures the
plan is self-consistent under first-order Euler integration.

This report documents two coupled bugs in how that constraint was applied to the UAV
setting, and the single-toggle fix that resolves both.

---

## 2. Code Provenance — How DPCC Entered FM-PCC

The `Projector` class and `DynamicConstraints` machinery were ported verbatim from the
original DPCC repo into the FM-PCC stack. The inheritance chain is:

```
/workspaces/dpcc/diffuser/sampling/projection.py         ← DPCC original
    └─ /workspaces/FM-PCC/flow_matcher_v3_uav/sampling/projection.py   ← copied verbatim
```

The constraint *selection* logic (what dims get a `deriv` pair) lives in a helper function
in the DPCC repo:

```
/workspaces/dpcc/diffuser/utils/constraints_helpers.py   ← DPCC original
    └─ inline in setup_dpcc_projector()                   ← inlined / adapted in FM-PCC
       /workspaces/FM-PCC/FM_v3_uav_test/eval_fm_uav.py
```

The eval loop (observation assembly + action integration) was written fresh for UAV, but
the DPCC `avoiding-d3il` eval (`/workspaces/dpcc/scripts/eval.py`) is the structural template.

---

## 3. Where the DPCC Code Is Wrong

### 3.1 Bug A — Constraint Binds to Commanded Position (`p_des`), Not Real Position (`p`)

#### Original DPCC code — `avoiding-d3il` domain

**File**: `/workspaces/dpcc/diffuser/utils/constraints_helpers.py`, L47–L53  
**Function**: `formulate_dynamics_constraints()`

```python
# DPCC original (avoiding-d3il branch):
if 'avoiding' in exp and action_dim > 0:
    dynamic_constraints = [
        ('deriv', np.array([act_obs_indices['x'],     act_obs_indices['vx']])),   # L49
        ('deriv', np.array([act_obs_indices['y'],     act_obs_indices['vy']])),   # L50
        ('deriv', np.array([act_obs_indices['x_des'], act_obs_indices['vx']])),   # L51  ← commanded x_des
        ('deriv', np.array([act_obs_indices['y_des'], act_obs_indices['vy']])),   # L52  ← commanded y_des
    ]
```

In the `avoiding-d3il` environment the robot has `x_des, y_des` (commanded position) and
`x, y` (real position). DPCC binds the `deriv` constraint to **both** — because the arm
controller tracks with negligible lag so `x_des ≈ x` always.

**How the `deriv` constraint is consumed** —  
**File**: `/workspaces/dpcc/diffuser/sampling/projection.py`, L337–L401  
**Class**: `DynamicConstraints.build_matrices()`

```python
# DPCC projection.py L365–L392:
if 'deriv' in type:
    x_idx  = int(vals[0])   # the position channel (e.g. x_des at index 3)
    dx_idx = int(vals[1])   # the velocity/action channel (e.g. vx at index 0)
    # builds equality: x[x_idx][t+1] = x[x_idx][t] + dt * x[dx_idx][t]
```

`x_idx` is whatever index is passed in — the projector does not know or care whether that
is real `x` or commanded `x_des`. The caller decides.

#### Our FM-PCC code — where the same pattern was adapted (incorrectly)

**File**: `/workspaces/FM-PCC/FM_v3_uav_test/eval_fm_uav.py`, L179–L187  
**Function**: `setup_dpcc_projector()`

```python
# FM-PCC BEFORE fix_5 (the buggy binding):
if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
    constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
    #                            ^^^  ← dim 3 = p_des_x, dim 4 = p_des_y, dim 5 = p_des_z
    #   enforces: p_des[t+1] = p_des[t] + dt * action[t]  — Euler in commanded space
```

UAV 12-D transition layout: `[action(0,1,2) | p_des(3,4,5) | p_real(6,7,8) | v(9,10,11)]`

Dims 3,4,5 are `p_des` (the **commanded** position). Dims 6,7,8 are `p` (the **real**
measured drone position from MuJoCo `data.qpos[:3]`). By binding to dims 3,4,5, the
projector enforces Euler-consistency for the commanded sequence only — it never touches
the real position.

**What this means mathematically:**  
The `DynamicConstraints.build_matrices` equality (`projection.py` L388–L392 — same code in
both DPCC `/workspaces/dpcc/diffuser/sampling/projection.py` and FM-PCC
`/workspaces/FM-PCC/flow_matcher_v3_uav/sampling/projection.py`) becomes:

```
p_des[t+1] = p_des[t] + dt * action[t]     (Euler in commanded space)
```

The projector corrects the FM's raw H-step output so that the **commanded position
sequence is internally consistent**. But it operates entirely in *commanded space* — it
never references `p` (dims 6,7,8), the **real measured drone position**.

**The problem:**  
When the drone lags the command (which is always the case — the PID has finite bandwidth),
`p_des ≠ p`. The projector plans forward from `p_des[0]`, a position the drone is **not
actually at**. The feasibility reasoning is performed from a fictional starting point.  
The projector is, in effect, *dreaming*: it corrects the trajectory to obey physics relative
to where the drone was *told* to be, not where it *actually is*.

This is correct in the D3IL arm domain (for which DPCC was designed), because robotic arms
track position commands with negligible lag — `p_des ≈ p` always. For a quadrotor under
PID control, the lag can be 0.1–0.5 m, and it grows unboundedly under the free-running
integration described next.

---

### 3.2 Bug B — Free-Running Integration Accumulates Unbounded OOD Error

#### Original DPCC code — `avoiding-d3il` eval loop

**File**: `/workspaces/dpcc/scripts/eval.py`, L234–L237  
**Location**: inner eval loop body

```python
# DPCC original avoiding-d3il eval loop:
action, samples = policy(conditions={0: obs}, ...)       # L231
if 'avoiding' in exp:
    next_pos_des = action + obs[:2]                      # L236  ← action added to obs (commanded pos)
    obs, rew, terminated, info = env.step(               # L237
        np.concatenate((next_pos_des, fixed_z, ...), axis=0))
    ...
    obs = np.concatenate((next_pos_des[:2], obs))        # L248  ← next obs seeds from next_pos_des
```

In the `avoiding-d3il` arm environment, `obs[:2]` is the arm's current commanded
`[x_des, y_des]`, and `next_pos_des = action + obs[:2]` advances the command by the action.
**Because the arm tracks perfectly**, `obs[:2]` ≈ real arm position at every step — the
accumulation does not cause OOD drift.

#### Our FM-PCC code — where the same pattern was adapted (incorrectly)

**File**: `/workspaces/FM-PCC/FM_v3_uav_test/eval_fm_uav.py`, L369–L374  
**Function**: `rollout_one()`

```python
# FM-PCC BEFORE fix_5 (the buggy integration):
# fix_5 anchor-p: p_des = p + action (grounded to real position every step).
# Default: free-running Euler (p_des += action, commanded space only).
if anchor_to_p:
    p_des = p + action
else:
    p_des = p_des + action    # ← L374: free-running — p_des drifts from p
```

`p_des` starts at `init_pos = p` (real position at episode start) and accumulates FM
actions indefinitely. After `k` steps:

```
p_des[k] = p_des[0] + Σ_{i=0}^{k-1} action[i]
         = p[0] + Σ actions
```

Nothing forces `p_des[k]` back toward `p[k]` (the actual drone position). The tracking
error `p_des - p` grows without bound.

**The cascade failure:**  
```
p_des grows unboundedly from p
→ PID chases a setpoint far outside the feasible envelope
→ UAV lags further, tracking error increases
→ next FM obs = [p_des_far | p_lagging | v] with |p_des − p| >> training distribution
→ FM produces garbage actions (OOD input → OOD output)
→ spiral failure: drone oscillates, crashes, or freezes
```

**Evidence from training data:**  
Training episodes record `obs = [p_des | p | v]` from the expert PID data collection.
In training, `p_des - p` reflects real PID tracking lag — bounded, typically < 0.3 m.
When free-running integration pushes `|p_des - p|` to several metres, the FM input is
completely out of distribution.

---

### 3.3 Why the Two Bugs Are Coupled

Bug A means the projector enforces feasibility from the wrong starting point.  
Bug B means the observation at the next FM step is out of distribution.

They interact:
- Bug B grows `p_des - p` unboundedly
- Bug A means the projector plans forward from the wrong (drifted) `p_des[0]`, making its
  corrections internally consistent but globally wrong
- With both bugs, the plan is self-consistent in commanded space but divorced from reality
  in physical space

Neither bug alone is the root cause — they are two faces of the same failure: **the
commanded position `p_des` is never re-anchored to the real drone position `p`**.

---

### 3.4 Origin of the Bugs — Domain Transfer from D3IL

| Aspect | D3IL arm (`avoiding-d3il`) | UAV quadrotor + PID |
|---|---|---|
| Controller | Direct position servo | PID tracking with finite bandwidth |
| Tracking lag | Negligible (`p_des ≈ p`) | 0.1–0.5 m, grows under free-running |
| `x_des ≈ x` assumption | ✅ Valid | ❌ Invalid |
| Free-running `p_des += action` | Safe (arm reaches p_des each step) | Unsafe (drone lags → OOD drift) |
| Constraint on `x_des` dims | Correct (x_des = x) | Wrong (x_des ≠ x under lag) |

DPCC did not invent this pattern — it is a correct design for servo-controlled arms.
**The mistake was applying it without modification to a system with significant tracking lag.**

---

## 4. The Fix — Anchor-P Integration (fix_5, no retrain)

### 4.1 Key Insight — DPCC Collapse

In the ideal DPCC case (perfect tracking, `p_des = p`):

```
OLD:  p_des_next = p_des + action = p + action   (since p_des = p)
NEW:  p_des_next = p_real + action = p + action   (identical)
```

The fix collapses to the **same behaviour** when tracking is perfect. It is a
**strict generalisation** of the correct DPCC formula — the difference appears only
when `p_des ≠ p` (real tracking lag), exactly the case being fixed.

### 4.2 Change 1 — Rebind the Dynamics Constraint to Real Position

**File changed**: `/workspaces/FM-PCC/FM_v3_uav_test/eval_fm_uav.py`  
**Function**: `setup_dpcc_projector()`, L179–L187

```python
# BEFORE (bug — dims 3,4,5 = p_des):
constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
#   enforces: p_des[t+1] = p_des[t] + action[t]  (Euler in commanded space)

# AFTER fix_5 (anchor_to_p=True — dims 6,7,8 = real p):
constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]
#   enforces: p[t+1] = p[t] + action[t]  (Euler in real space, geo-calibrated)
```

The projector now corrects the FM's action so the **real position sequence** is
Euler-consistent from the **actual drone starting position** `p[0]`. The action is
geo-calibrated: it represents how much the drone will actually move.

The underlying `DynamicConstraints.build_matrices` logic is **identical** in both repos:

| | DPCC original | FM-PCC (copied) |
|---|---|---|
| **File** | `/workspaces/dpcc/diffuser/sampling/projection.py` L344–L401 | `/workspaces/FM-PCC/flow_matcher_v3_uav/sampling/projection.py` L344–L401 |
| **Code** | Verbatim identical (copy-paste) | Verbatim identical |
| **Bug** | None — correct generic machinery | None — same correct machinery |
| **Bug location** | In caller: wrong `x_idx` chosen | In caller: wrong `x_idx` chosen |

The fix is in the **caller** (`setup_dpcc_projector()`) — the `x_idx` passed to `deriv`
is changed from dims 3,4,5 (`p_des`) to dims 6,7,8 (`p_real`). The `DynamicConstraints`
class itself is not touched.

### 4.3 Change 2 — Anchor Integration to Real Position

**File changed**: `/workspaces/FM-PCC/FM_v3_uav_test/eval_fm_uav.py`  
**Function**: `rollout_one()`, L371–L374

```python
# BEFORE (bug — mirrors dpcc/scripts/eval.py L236, but without perfect tracking):
p_des = p_des + action      # free-running: p_des drifts unboundedly from p

# AFTER fix_5 (anchor_to_p=True):
p_des = p + action          # anchored: p_des always within |action| of real p
```

**Comparison with original DPCC eval**:

| | DPCC original (`/workspaces/dpcc/scripts/eval.py` L236) | FM-PCC before fix (`eval_fm_uav.py` L374) | FM-PCC after fix (L372) |
|---|---|---|---|
| Formula | `next_pos_des = action + obs[:2]` | `p_des = p_des + action` | `p_des = p + action` |
| Anchor | `obs[:2]` = commanded x_des (≈ real, arm) | `p_des` = commanded (drifts, UAV) | `p` = real MuJoCo position |
| Valid? | ✅ arm tracks perfectly | ❌ UAV lags → OOD | ✅ always grounded |

### 4.4 Why Both Changes Are Required Together

| Change alone | Problem |
|---|---|
| Rebind constraint only (dims 6,7,8) | Projector plans from real `p`, but integration still drifts `p_des` → next step's obs still goes OOD |
| Anchor integration only (`p_des = p + action`) | `p_des` stays near `p`, but projector enforces Euler in commanded space from the wrong anchor → plan is inconsistent with where drone actually is |
| **Both together** | Projector plans from real `p`; integration keeps `p_des` near `p`; obs never goes OOD; plan and execution are mutually consistent |

### 4.5 Config Toggle

Both changes activate under one flag:

**File**: `/workspaces/FM-PCC/config/uav.py`, `plan_flow_matching_v3_uav` block, L160:
```python
'anchor_to_p': False,   # True: enable fix_5 (real-p constraint + anchor integration)
```

When `anchor_to_p=False` (default), the old behaviour is exactly preserved — no
behaviour change to the baseline. **No training changes. No checkpoint moves. No new data.**

---

## 5. Complete Cross-Reference Table

| Concern | DPCC original | FM-PCC (before fix) | FM-PCC (after fix_5) |
|---|---|---|---|
| **Constraint helper** | `dpcc/diffuser/utils/constraints_helpers.py` L47–53 | Inlined in `eval_fm_uav.py` L179–187 | Same, dims changed |
| **Constraint engine** | `dpcc/diffuser/sampling/projection.py` L337–401 `DynamicConstraints` | `flow_matcher_v3_uav/sampling/projection.py` L337–401 (verbatim copy) | Unchanged (correct) |
| **`deriv` dims** | `x_des` (correct for arm, `x_des≈x`) | `p_des` dims 3,4,5 (wrong for UAV) | real `p` dims 6,7,8 ✅ |
| **Integration formula** | `dpcc/scripts/eval.py` L236: `action + obs[:2]` (arm ≈ real) | `eval_fm_uav.py` L374: `p_des + action` (drifts) | L372: `p + action` ✅ |
| **`skip_initial_state` anchor** | `dpcc/diffuser/sampling/projection.py` L99–108: pins `b[0] = s_0[x_idx]` | Same (from real traj[0]) — was `x_idx` = `p_des` dim | Now `x_idx` = real `p` dim ✅ |

---

## 6. Mathematical Statement

Let `p_k` = real drone position at FM step `k`, `p_des_k` = commanded setpoint fed into
FM obs, `a_k` = FM action output at step `k`.

**Before fix:**
```
Projector:   p_des[t+1] = p_des[t] + a[t]  (constraint anchored at p_des[0])
Integration: p_des_{k+1} = p_des_k + a_k   (free-running)
Result:      |p_des_k − p_k| → ∞ as k → ∞  (OOD spiral)
```

**After fix:**
```
Projector:   p[t+1] = p[t] + a[t]          (constraint anchored at p[0] = p_k, real)
Integration: p_des_{k+1} = p_k + a_k       (anchored, |p_des − p| ≤ |a_k|)
Result:      |p_des_{k+1} − p_{k+1}| ≈ |a_k − Δp_k|  (bounded by PID lag, in-distribution)
```

When tracking is perfect (`p_{k+1} = p_des_{k+1}`): both formulas give identical results,
confirming the fix is a **generalisation**, not a deviation.

---

## 7. Summary Table

| Aspect | Before fix | After fix (`anchor_to_p=True`) |
|---|---|---|
| Dynamics constraint anchor | `p_des` (dims 3,4,5) — commanded, drifted | `p` (dims 6,7,8) — real measured position |
| Integration formula | `p_des = p_des + action` — free-running Euler | `p_des = p + action` — re-anchored each step |
| `p_des − p` in obs | Grows unboundedly → OOD after ~5–10 steps | Bounded to `|action|` → always in-distribution |
| Projector behaviour | Plans from fictional `p_des` starting point | Plans from real drone position |
| Training needed | — | None (eval-only change) |
| Domain of origin | D3IL arm (near-zero lag → correct there) | UAV PID (significant lag → was incorrect) |

---

## 8. Files Changed in fix_5

| File | Lines changed | Change |
|---|---|---|
| `config/uav.py` | L154–L160 | Added `'anchor_to_p': False`; removed superseded `reanchor_alpha`, `lead_gain`, `cond_mode` eval knobs |
| `FM_v3_uav_test/eval_fm_uav.py` | L129, L149, L179–187, L371–374, L481–482 | `load_pcc_config`, `setup_dpcc_projector` (sig + deriv dims), `rollout_one` (integration), `_run_variant` (read + tag + call sites) |

No training scripts, dataset files, model weights, or checkpoints were modified.

---

## 9. Reference Documents

- Design analysis: [`DESIGN_fix5_anchor_p_integration.md`](DESIGN_fix5_anchor_p_integration.md)
- Changelog: [`CHANGELOG_fix5_anchor_p.md`](CHANGELOG_fix5_anchor_p.md)
- DPCC constraint helper (original): `/workspaces/dpcc/diffuser/utils/constraints_helpers.py` L34–54
- DPCC projector (original): `/workspaces/dpcc/diffuser/sampling/projection.py` L337–401
- DPCC eval loop (template): `/workspaces/dpcc/scripts/eval.py` L112–121 (constraint assembly), L230–248 (action integration)
- FM-PCC projector (copy): `/workspaces/FM-PCC/flow_matcher_v3_uav/sampling/projection.py`
- FM-PCC eval (adapted): `/workspaces/FM-PCC/FM_v3_uav_test/eval_fm_uav.py`
- FM-PCC config: `/workspaces/FM-PCC/config/uav.py`
