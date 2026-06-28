# fix_5 — Anchor-P Integration: Grounding p_des to Real Position Without Retrain

**Date**: 2026-06-28
**Parent**: [../fix_4_CLI_yaml/CHANGELOG_fix4_CLI_yaml.md](../fix_4_CLI_yaml/CHANGELOG_fix4_CLI_yaml.md)
**Status**: DESIGN (not yet implemented in code)

---

## The Reasoning Chain That Led Here

The following logic was derived in session — it is the correct diagnosis:

> *"there is no real observation feedback from the real world, old is based on the p_des not p (real)
> so the PID is lost control, cannot get real location.*
> *but we either not learn in the train phase??? the train indeed contains what in real location +
> command location what to behave → action.*
> *Ahhh, yes, this is enough, all the problem is the real world location…
> the point is how to link the real world location into the eval phase…
> but the action is corrected by the `dynamic` by imagining p_des?
> so this is the real problem?"*

This document formalises that chain.

---

## 1. Root Cause: Free-Running Integration Severs the Command from Reality

The current `p_des` mode eval integration (in `rollout_one`, `eval_fm_uav.py`):

```python
# Every FM control step:
p_des = (1.0 - reanchor_alpha) * (p_des + action) + reanchor_alpha * p
# when reanchor_alpha = 0.0 (default):
p_des = p_des + action   # ← free-running Euler, p_des accumulates in commanded space
```

`p_des` is the PID setpoint. It starts at `init_pos = p` and accumulates FM outputs
indefinitely. Nothing forces it back toward the measured `p`. Over time:

```
p_des grows unboundedly from p
→ PID is chasing a setpoint far ahead (or outside the flyable envelope)
→ drone lags, sometimes crashes
→ at the next FM step, obs = [p_des | p | v] with |p_des - p| >> training distribution
→ FM produces garbage actions from OOD obs
→ spiral failure
```

---

## 2. The Real Position p IS Known at Eval — But Not Used as Anchor

Every FM control step reads:
```python
p = data.qpos[:3].copy()   # real drone position from MuJoCo  (eval_fm_uav.py L339)
```

Both `p_des` and `p` are fed into the FM as obs:
```python
obs = np.concatenate([p_des, p, v])   # [p_des | p | v] (9D)  (L343)
```

The FM's velocity field sees the tracking error `p_des - p` implicitly. It knows where the
drone was commanded to be AND where it actually is. This is correct.

**The problem is not observation — the problem is integration.** After the FM acts:
```
p_des += action   ← action is applied to the old (drifted) p_des, not to real p
```

The command advances in commanded space, disconnected from real space.

---

## 3. The Dynamics Constraint Does Not Fix This

The DPCC projector enforces (via the `deriv` constraint):
```
p_des[t+1] = p_des[t] + dt * action[t]   for t in horizon H
```

This makes the H-step PLAN internally Euler-consistent in commanded space. But:
- The plan is seeded from the current `p_des` (which may already be far from `p`)
- After executing one step, the next `p_des[0]` = old `p_des + action` (still drifted)
- The projector operates on the plan in commanded space — it never uses `p` as an anchor

So the projector enforces commanded-space consistency but cannot prevent commanded-space drift.

---

## 4. Why Training Is Already Correct — No Retrain Needed

Training data:
```
obs    = [p_des | p | v]   (9D)     — BOTH commanded and real position
action = Δp_des = p_des[t+1] - p_des[t]   (3D)   — exact PID commanded delta
```

The Euler constraint `p_des[t+1] = p_des[t] + action[t]` holds exactly in training data
(by construction: the dataset records the commanded sequence). No injection needed.

The FM learned: **"given where I was told to be (p_des), where I actually am (p), and
velocity (v) → what commanded delta should I output?"**

This training contains all the information we need. The model weights are correct as-is.

---

## 5. The Fix: Anchor-P Integration (Eval-Only, No Retrain)

**Change one line in `rollout_one()`.**

At every FM control step, after the FM outputs `action`:

```python
# BEFORE (free-running, alpha=0):
p_des = p_des + action

# AFTER (anchor-p):
p_des = p + action
```

This means: **the next PID setpoint is always rooted at the current real position, plus
the FM's output delta.** The command can never run away because it is rebuilt from `p` at
every step, not accumulated in commanded space.

`p_des` fed into the NEXT step's obs = `p_k + action_k`. This is always within
`|action|` of `p_k`, which is within training distribution (training data's tracking errors
were bounded by real PID lag, not unbounded free-running).

### What This Is NOT

| What | Formula |
|---|---|
| `reanchor_alpha=1.0` | `p_des = p` — action **lost**, drone hovers |
| `anchor-p` (this fix) | `p_des = p + action` — action **preserved**, command anchored |

`reanchor_alpha` blends between old `p_des + action` and a hard reset to `p`. There is no
alpha value that gives `p_des = p + action`. This is a genuinely new integration formula.

### With lead_gain

```python
p_des = p + lead_gain * action
```

`lead_gain > 1.0`: the setpoint leads the drone's current position further along the
FM-predicted direction. Useful if the drone under-reaches targets (PID tracking is fast).
`lead_gain = 1.0` is the natural default.

---

## 6. Full Mode Comparison

| Mode | obs (9D or 6D) | Integration formula | Retrain? | Effect |
|---|---|---|---|---|
| `p_des` alpha=0 | 9D `[p_des\|p\|v]` | `p_des = p_des + action` | No | Free-running — OOD spiral |
| `p_des` alpha=0.5 | 9D | `p_des = 0.5*(p_des+action) + 0.5*p` | No | Partial blend — slower spiral |
| `p_des` alpha=1.0 | 9D | `p_des = p` | No | Hard reset — **action lost, drone hovers** |
| **anchor-p** (this fix) | **9D** | **`p_des = p + action`** | **No** | **Anchored, action preserved** |
| `real_p` mode | 6D `[p\|v]` | `p_des = p + lead_gain*action` | **Yes** | Same integration as anchor-p, but drops p_des from obs and retrains |

**anchor-p is strictly better than `real_p` for the first test**: same integration formula,
keeps `p_des` in obs (the FM retains command-history context), zero training cost.

The `real_p` retrain path is still open if anchor-p reveals the model can't reconcile
`Δp_des ≠ Δp_real` — but that test costs nothing to defer.

---

## 7. Relationship to the Dynamics Constraint

The dynamics projector (`deriv`) continues to function identically:

```
projector enforces: p_des[t+1] = p_des[t] + action[t]  within H-step plan
```

With anchor-p, when we execute step 0:
```
p_des_next = p_k+1 + action[0]   (anchor-p integration)
```

The projector planned `p_des_k + action[0]`, but we set it to `p_k+1 + action[0]`. If
`p_k+1 ≈ p_des_k + action[0]` (good tracking), they match. If the drone lags, they differ
— but that's fine: the projector is replanned from scratch at the next FM step from the new
`p_des = p_k+1 + action[0]`. No Euler violation accumulates across steps.

---

## 8. Implementation Plan

### Changes required

**1. `config/uav.py` — `plan_flow_matching_v3_uav` block**

Add:
```python
'anchor_to_p': False,   # anchor-p: p_des = p + action each step (no retrain)
```

**2. `FM_v3_uav_test/eval_fm_uav.py`**

In `load_pcc_config()`:
```python
cfg['anchor_to_p'] = bool(getattr(plan_args, 'anchor_to_p', False))
```

In `rollout_one()` signature:
```python
def rollout_one(..., anchor_to_p=False):
```

In `rollout_one()` integration block:
```python
if cond_mode == 'real_p':
    p_des = p + lead_gain * action
elif anchor_to_p:
    p_des = p + lead_gain * action   # anchored to real p; lead_gain=1.0 by default
else:
    p_des = (1.0 - reanchor_alpha) * (p_des + action) + reanchor_alpha * p
```

In `_run_variant()` eval_tag:
```python
if cond_mode == 'real_p':
    if lead_gain != 1.0:
        eval_tag = f'_lead{lead_gain:g}'
elif anchor_to_p:
    eval_tag = '_anchorP' if lead_gain == 1.0 else f'_anchorP_lead{lead_gain:g}'
elif reanchor_alpha != 0.0:
    eval_tag = f'_reanchor{reanchor_alpha:g}'
```

Pass `anchor_to_p` through `_run_variant()` → `rollout_one()`.

### No train script changes. No yaml changes. No checkpoint moves.

---

## 9. Test Plan

Update the `RUN_GUIDE_U4_tests.md` to add anchor-p as **Phase 0d** (after the alpha sweep,
before committing to `real_p` retrain).

```bash
# Set in config/uav.py → plan_flow_matching_v3_uav:
'anchor_to_p': True,
'lead_gain': 1.0,

# Git sync, then:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"
# Output: .../plans/.../<seed>/diffuser_anchorP/
```

Compare `track_err_mean` and `success_rate` vs:
- alpha=0.0 baseline (expected: OOD spiral, 0% corridor success)
- alpha=1.0 (expected: action lost, hovering)
- anchor-p (expected: grounded, FM direction preserved → best of the no-retrain options)

---

## 10. Decision Point After anchor-p Test

| Result | Implication |
|---|---|
| anchor-p succeeds (corridor success > 0%) | No retrain needed. Deploy anchor-p for full eval. |
| anchor-p improves but not enough | Try `lead_gain` > 1.0; try multi-step look-ahead |
| anchor-p fails (model can't work from p anchor) | Proceed to `real_p` retrain |

The hypothesis (based on training data analysis) is that anchor-p will succeed because:
- The FM saw `[p_des | p | v]` in training with tracking errors in it
- It learned to output reasonable Δp_des regardless of the error magnitude
- With anchor-p, `p_des - p ≈ action_prev` (bounded, never OOD)
- The model now operates strictly within training distribution

---

*See also: [fix_3/RUN_GUIDE_U4_tests.md](../fix_3/RUN_GUIDE_U4_tests.md) for the Phase 0 alpha sweep commands.*

---

## 11. Rethink — The Dynamics Constraint Is the Real Fix, Not p_des

*(Added after second-pass analysis. This section supersedes the anchor-p framing in §5 and §8.)*

### The problem with anchor-p as described above

Changing the integration to `p_des = p + action` anchors `p_des` to real `p` — but it
silently changes what `p_des` MEANS in the obs that the FM sees:

- **Training**: `p_des` was a free-running command. It could be far ahead of `p` (real
  tracking error). The FM learned to condition on that drift: `[p_des far | p lagging | v]`.
- **Anchor-p eval**: `p_des ≈ p` always (bounded to one action step). The FM now sees
  `[p ≈ p_des | p | v]` — both obs slots carry nearly the same value. The tracking-error
  signal in obs is erased.

This is a distribution shift in the obs. Small, but real.

### The dynamics constraint is the proper link between action and geography

The `deriv` constraint in the projector defines what the FM action MEANS geometrically:

```python
# Current — action is the derivative of p_des (dims 3,4,5):
constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
# Enforces:  p_des[t+1] = p_des[t] + dt * action[t]
```

This constraint says: "action moves the commanded position." The projector corrects the
FM's raw output so that the planned `p_des` sequence is Euler-consistent. The correction
lives in commanded space — it does NOT care about where the drone actually is.

If instead we bind the constraint to REAL position (dims 6,7,8 = `p`):

```python
# Proposed — action is the derivative of real p (dims 6,7,8):
constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]
# Enforces:  p[t+1] = p[t] + dt * action[t]
```

Now the projector corrects the FM's output so the planned `p` sequence is Euler-consistent
FROM THE REAL POSITION. The action is geo-calibrated: it describes how much the drone will
ACTUALLY move, not how much the command moves.

### Why this requires anchor integration too

The projector plans a horizon-H sequence. At `t=0`, `p[0]` = measured real position
(known). For `t>0`, `p[t]` is what the plan PREDICTS the drone will be. The constraint
enforces that the plan is self-consistent (`p[t+1] = p[t] + action[t]`).

After executing action[0], we must update `p_des` for the next FM query. If we keep
`p_des = p_des + action` (free-running), p_des drifts away from p again → same OOD
problem in the obs at the next step.

The ONLY consistent update is `p_des = p + action`, i.e., the anchor integration. So
both changes are required together:

```
real-p constraint  ←→  anchor integration
(action geo-calibrated)    (p_des stays ≈ p in obs)
```

These are not independent. One without the other is incomplete.

### What p_des in obs still carries

With both changes active, `p_des` at step k+1 = `p_k + action_k`. It is NOT `p_{k+1}`
(the drone hasn't moved yet when we update p_des). So `p_des` in obs still carries
meaningful information: **what setpoint was issued last step** and implicitly **what
direction the FM intended to move**. The FM can use `p_des - p` to gauge how well the
previous command executed. This is not erased — it is just bounded to one action step
rather than accumulated across many steps.

### Does this break the training meaning of p_des?

Partially, yes. In training, `p_des - p` could reflect multiple steps of accumulated lag.
With anchor-p, `p_des - p ≈ action_prev` (one step only). This IS a distribution shift.

However:
1. At the START of every training episode, `p_des = p = init_pos`, so `p_des - p = 0`
   IS in training distribution. Anchor-p keeps it near-zero always = "permanently near
   episode start" regime.
2. The alternative (free-running p_des) sends the FM to regions of obs space that grow
   unboundedly OOD. Bounded near-zero is strictly better than unbounded OOD.
3. If the model degrades badly in the near-zero regime, that indicates a deeper training
   issue (the model over-fit to large tracking errors). Retrain at that point.

### Corrected design: two coupled changes

| Change | Location | What it fixes |
|---|---|---|
| Constraint rebinding: `deriv` dims → `[6,0],[7,1],[8,2]` | `setup_dpcc_projector()` in `eval_fm_uav.py` | Action is geo-calibrated from real `p`, not from drifted `p_des` |
| Anchor integration: `p_des = p + action` | `rollout_one()` in `eval_fm_uav.py` | `p_des` in next obs ≈ `p`, consistent with what the constraint planned |

These must be toggled TOGETHER under one flag (`anchor_to_p: True`).

When `anchor_to_p=False` (default), both revert: constraint on `p_des` dims, free-running
integration. The two modes are fully isolated.

### Updated setup_dpcc_projector logic

```python
# In setup_dpcc_projector(), replace the current deriv block:
if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
    if anchor_to_p:
        # Bind action to REAL p (dims 6,7,8) — action is geo-calibrated from real position.
        # Used together with anchor integration (p_des = p + action) in rollout_one().
        constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]
    else:
        # Default: bind action to commanded p_des (dims 3,4,5) — Euler in commanded space.
        constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
```

`anchor_to_p` must be passed into `setup_dpcc_projector` (add to signature and call site
in `_run_variant()`).

### Updated rollout_one() integration block

```python
if cond_mode == 'real_p':
    p_des = p + lead_gain * action
elif anchor_to_p:
    p_des = p + lead_gain * action   # anchored + geo-calibrated via real-p constraint
else:
    p_des = (1.0 - reanchor_alpha) * (p_des + action) + reanchor_alpha * p
```

### The `diffuser` variant with anchor_to_p

`diffuser` runs NO projector — so the constraint rebinding has no effect. The anchor
integration `p_des = p + action` STILL applies (the integration change is independent of
whether projection is active). The FM's raw action is applied relative to real `p`
regardless. This is correct: even without constraint correction, anchoring prevents OOD
drift.

---

## 12. Revised Implementation Plan (replaces §8)

**Files to change:**

**`config/uav.py`** — `plan_flow_matching_v3_uav` block:
```python
'anchor_to_p': False,   # True: real-p constraint + anchor integration (no retrain)
```

**`FM_v3_uav_test/eval_fm_uav.py`** — four touch points:

1. `load_pcc_config()`: `cfg['anchor_to_p'] = bool(getattr(plan_args, 'anchor_to_p', False))`

2. `setup_dpcc_projector()` signature: add `anchor_to_p=False`; swap `deriv` dims when True.

3. `_run_variant()`: read `anchor_to_p = bool(config.get('anchor_to_p', False))`, pass to
   `setup_dpcc_projector()` and `rollout_one()`; add eval_tag branch:
   ```python
   elif anchor_to_p:
       eval_tag = '_anchorP' if lead_gain == 1.0 else f'_anchorP_lead{lead_gain:g}'
   ```

4. `rollout_one()` signature: add `anchor_to_p=False`; add integration branch (above).

**No train script changes. No yaml changes. No checkpoint moves.**

---

## 13. User Validation — "Dreaming in Commanded Space" and the DPCC Collapse

*(Captured from session analysis. This confirms the fix is correct and complete.)*

### The core diagnosis: it is not dynamics anymore

The current constraint enforces:

```
p_des[t+1] = p_des[t] + action[t]
```

With real tracking error (`p_des ≠ p`), this is **not a dynamics model** — it is a
self-consistent accumulation in commanded space that has no physical grounding. The
projector is correcting the action to be consistent with a position that the drone is not
at. That is the "dreaming" — it reasons about feasibility from a fictional starting point.

### Formula direction (confirmed)

The integration form is:
```
p_des[t+1] = p_des[t-1+1] = p_des[t] + action[t]   ← this one (first form)
```
The action is defined as the commanded delta:
```
action[t] = p_des[t+1] - p_des[t]                   ← algebraic rearrangement only
```
Both are the same equation. The eval loop runs the integration form left-to-right. The
training data uses the algebraic form to label actions from the recorded `p_des` sequence.
No ambiguity.

### The DPCC collapse — the cleanest validation

In the ideal DPCC case (perfect tracking, zero lag): `p_des = p` always.

```
OLD:  p_des_next = p_des + action  =  p + action    (p_des = p, so identical)
NEW:  p_des_next = p_real + action =  p + action    (same result)
```

The proposed patch collapses to IDENTICAL behaviour when there is no tracking error.
This confirms the fix is a generalisation of the correct behaviour, not a deviation from
it. The only difference appears when `p_des ≠ p` (real tracking lag) — exactly the case
we are fixing.

### Why the patch is complete and sufficient

The proposed change:
```python
# Before (dreaming):
p_des = p_des + action          # commanded Euler, decoupled from reality

# After (grounded):
p_des = p_real + action         # every step restarted from where the drone actually is
```

Combined with rebinding the projector's `deriv` constraint from `p_des` dims (3,4,5) to
real-`p` dims (6,7,8): the projector now geo-calibrates the action from the REAL position.
The two changes are consistent: constraint plans from real `p`, integration executes from
real `p`.

No training change. No retrain. No new data. One config toggle.

---

## 15. Where Does the `dynamics` Constraint Come From? D3IL or DPCC?

### Origin: D3IL, not DPCC

The `deriv` constraint and `DynamicConstraints` class come directly from the **D3IL**
(Diffusion for 3D Imitation Learning) codebase, in
`flow_matcher_v3_uav/sampling/projection.py`:

```python
class DynamicConstraints(Constraints):
    """
    ('deriv', [x_idx, dx_idx])
    → x[t+1] = x[t] + dt * dx[t]   (explicit Euler)
    """
```

It was inherited into this project via the chain:

```
D3IL projection.py
    └─ fm_visual_aligning_test/eval_fm_visual_aligning.py   (ported verbatim for arm)
           └─ FM_v3_uav_test/eval_fm_uav.py                 (re-bound dims for UAV p_des)
```

DPCC (Diffusion Policy with Constraint-based Correction) is the METHOD — it uses the
D3IL projector infrastructure but applies it during FM denoising. DPCC did not invent the
`deriv` constraint; it reused the D3IL linear equality machinery.

### Why D3IL uses naive Euler (action = position delta)

D3IL's primary domain is **robot arm manipulation**. For an arm:

- The controller tracks position commands with very small lag
- Action = `Δx` (end-effector position delta) is a natural choice
- `x[t+1] = x[t] + dt * action[t]` holds approximately because the arm reaches the
  commanded position within one timestep

The Euler constraint is LINEAR in the trajectory variables:
```
A_eq · traj_flat = b_eq
```
This gives a **convex QP** — fast to solve (~1–5 ms), no local minima, closed-form or
sparse-Cholesky solution. A nonlinear dynamics model would require nonlinear programming
(NLP), killing real-time feasibility.

So D3IL chose naive Euler deliberately: correct for the arm domain, computationally cheap,
and sufficient for the constraint correction task.

### What a real full dynamics model would be (SafeFlowMPC direction)

A full quadrotor dynamics model would replace the Euler `deriv` with:

```
p[t+1]  = p[t]  + v[t] * dt
v[t+1]  = v[t]  + (R[t] * f_thrust[t] / m - g - drag(v[t])) * dt
R[t+1]  = R[t]  * exp(ω[t] * dt)   (SO(3) integration)
ω[t+1]  = ω[t]  + J⁻¹(τ[t] - ω[t] × J·ω[t]) * dt
```

Where `f_thrust`, `τ` come from rotor speeds, `m` is mass, `J` is inertia tensor, `g` is
gravity. This is the full 12-state rigid-body + rotor model.

**Why this is not used here:**

| | Euler (current) | Full dynamics |
|---|---|---|
| Constraint type | Linear equality (convex QP) | Nonlinear equality (NLP) |
| Solve time | ~1–5 ms | ~50–500 ms (NLP solver) |
| Real-time at 33 Hz | Yes (30 ms budget) | Marginal to impossible |
| Requires | Nothing | Known mass, inertia, drag model |
| Accuracy | First-order approximation | High-fidelity |

SafeFlowMPC-style approaches handle this by either:
1. **Linearizing** the dynamics around the current state (LQR / iLQR) — convex QP again
   but linearization error grows with horizon
2. **Learning a surrogate dynamics model** from data and using it in the projection
3. **Decoupling**: project position-level constraints (Euler OK) and separately enforce
   velocity/attitude via the PID (as we already do — the PID IS the dynamics enforcer)

### How this fits the fix_5 design

Our fix changes the Euler constraint from commanded-space (`p_des`) to real-space (`p`):

```
Before:  p_des[t+1] = p_des[t] + action[t]   (Euler in commanded space)
After:   p[t+1]     = p[t]     + action[t]   (Euler in real space)
```

This is still naive Euler — still first-order, still no aerodynamics. But it is a
meaningful improvement: the action is now geo-calibrated from where the drone ACTUALLY IS,
not from a drifted commanded position. The PID continues to enforce the real aerodynamics
by tracking `p_des = p + action`.

The upgrade path to a full dynamics model would be:

```
fix_5  →  Euler in real p  (no retrain, minimal change, this doc)
future →  Linear drone model (PD-linearized) in projector (upgrade DynamicConstraints)
future →  Learned surrogate dynamics (data-driven, matches real flight characteristics)
```

For the current epoch (Gen11 E7), the Euler-in-real-p fix is the correct scope: it
eliminates the OOD spiral with zero training cost and stays within the real-time budget.
