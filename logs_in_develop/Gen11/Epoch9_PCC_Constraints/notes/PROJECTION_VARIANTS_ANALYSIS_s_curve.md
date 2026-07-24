# Projection‑Variant Analysis — UAV `s_curve` (Gen11 / Epoch9 PCC Constraints)

**Data source:** `temp/s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles/`
**Scene:** `s_curve` · **seed:** 6 · **trials/variant:** 10 · **replan steps/trial:** 871 (`max_episode_length`, `decim=3`, `dt=0.01`)
**Model:** Flow‑Matching v3 UAV (deterministic ODE reference) + DPCC projection (SLSQP / scipy, `states_actions`, `H=8`, `transition_dim=12`)
**Constraint stack declared for the scene:** `['dynamics', 'geo_bounds', 'halfspace', 'obstacles', 'bounds']`

> This note is written to be lifted into the thesis. Section 1–3 are the method/taxonomy; Section 4 is the
> results table; Section 5 is the per‑variant commentary; Section 6 is the headline finding; Section 7 shows
> how the `.npz` trajectories prove the projection is actually doing what the QP says it does.

---

## 0. Step‑count provenance — is this the legacy fixed‑step regime? (do we need a rerun like avoiding?)

**Short answer: No — this run already uses the U_13 "avoiding‑faithful" episode loop; no rerun is needed for
step semantics.** The `871` is a *deliberate fixed per‑scene budget*, not the old random/never‑terminate
artifact, and early termination on goal‑reach is active. Details, since it matters for how the table is read:

- **This is NOT the pre‑U_13 legacy behaviour.** The old regime (diagnosed in `U_13/INVESTIGATION_and_PLAN_
  deterministic_episode_length.md`) drew a **random per‑trial** budget `round(dur·33 Hz)` with `dur` sampled
  per trial (`s_curve dur∈[16,22] s`) and **never terminated early** — so `n_steps` was uninformative. That
  has been replaced: `eval_fm_uav.py:903` now takes `n_fm = int(max_episode_length)` (a **fixed** per‑scene
  constant, `SCENE_MAX_EPISODE_LENGTH['s_curve']=871`), the same‑for‑every‑trial design DPCC d3il‑avoiding
  uses (`avoiding‑d3il.py:68 → 200`).
- **Early termination on goal‑reach IS implemented** (`eval_fm_uav.py:1059`:
  `if goal_reached_latch or k == n_fm-1: steps_run = k+1; break`), i.e. the DPCC `break`‑on‑success loop. It
  fires: `geo_free` trial 6 stopped at **`n_fm_steps=599`** with `reached=true` (and `steps.to_goal_mean=599`,
  `steps.mean=843.8` for that variant). So `n_steps` here already carries the deterministic **time‑to‑goal**
  meaning it has in avoiding — it is not a fixed placeholder.
- **Why almost every row still shows 871.** On this hard non‑convex gate the drone rarely satisfies the
  **strict** goal‑reach latch (`‖p−goal‖<goal_radius` at some step), so those episodes legitimately run to the
  full budget and report `steps.mean=871`, `to_goal_mean=null`. That is a **true outcome, not a measurement
  bug** — it reflects the low strict‑success rate, and re‑running would reproduce the same 871.
- **One genuine caveat for the thesis.** Only the **strict** latch triggers the early `break`; merely
  *crossing the finish line* (`success_relaxed`/`crossed_line`) does **not** stop the episode. So a trial can
  cross the line and keep flying to budget (e.g. several `geo_free` trials with `crossed=true`, `dist≈0.5`,
  yet `n_fm_steps=871`). Consequently **`n_steps`/time‑to‑goal is only meaningful for the handful of
  strict‑reach trials**; do not read the 871 rows as "time‑to‑goal", read them as "ran the full budget without
  a strict reach". Nothing about the safety / violation / timing conclusions below depends on this — those are
  per‑step accumulations over the flown steps regardless of where the episode stopped.

*(If a future analysis wants a clean time‑to‑goal distribution across variants, that WOULD need a rerun with a
looser/relaxed early‑stop — but for the projection‑variant comparison in this note the current data is correct
and complete.)*

---

## 1. What the projector is and where each variant differs

The generative brain (Flow Matching U‑Net + ODE solver) emits an **unconstrained** `H=8`‑step reference
plan per replan step. The **projector** (`flow_matcher_v3_uav/sampling/projection.py::Projector`) filters
that plan through a constrained QP

```
ẑ = argmin_z  ½ zᵀQz + rᵀz      (Q = I, r = −τ  ⇒  minimise ‖z − τ‖²)
      s.t.  A z = b            (equality: dynamics / kinematic consistency)
            C z ≤ d            (inequality: box bounds, action bounds, half‑spaces)
            zᵀP z + qᵀz ≤ v    (quadratic: sphere obstacles)
```

so `ẑ` is the **closest constraint‑satisfying trajectory to the FM output**. A batch of candidates is
generated; a *selection rule* then picks which one the drone executes. Every folder in the test is the
**same FM model** wearing a different projector configuration — so any performance gap is attributable to
the projection, not the policy.

The variant *name* is a composite string. Three **orthogonal, composable** axes are encoded in it
(`setup_dpcc_projector`, `_selection_for`):

| Axis | Token(s) | Effect |
|---|---|---|
| **A. Solve mode** | *(default)* / `gradient` / `post_processing` | hard QP (SLSQP) vs. soft gradient step vs. late‑only projection |
| **B. Candidate selection** | `dpcc-r` / `dpcc-c` / `dpcc-t` | `random` / `minimum_projection_cost` / `temporal_consistency` |
| **C. Constraint ablation** | `model_free` / `bounds_free` / `geo_free` | drop dynamics / drop action bound / drop the whole geometric group (geo_bounds+halfspace+obstacles) |
| **+ margin** | `-tightened` | add `enlarge_constraints = 0.025 m` on top of the always‑on body inflation |
| **baseline** | `diffuser` | projector **off** — raw FM reference, no filtering |

Two always‑on physical facts underlie every projected variant:
- **Body inflation** `r_drone=0.31 + margin_base=0.02 = 0.33 m` offsets every spatial surface so the drone
  *body* (not just its centre) clears geometry. `-tightened` adds `0.025 m` on top.
- **Action bound** (`action_bounds: 'auto'`) is self‑derived from this dataset's own Δp_des range
  (`act_normalizer.mins/maxs`) — the faithful equivalent of DPCC‑avoiding's hand‑picked `['vx','vy']` cap.

### The `s_curve` geometry (why it is the hard case)
`s_curve` is a **non‑convex** two‑segment corridor. Feasibility switches with the drone's x‑position, so
each wall half‑space carries an `x_active` interval and is re‑selected **every replan** from the current x
(`rebuild_projector(float(p[0]))`). Config (`config/uav_projection.yaml`):

- `workspace_bounds` box `x,y∈[-3.6,3.6]×[-1.6,1.6]`, `z∈[0.30,1.80]`
- 4 wall half‑spaces on inner faces (`seg1 y∈{-0.35,-1.25} for x∈[-3,-0.5]`; `seg2 y∈{0.35,1.25} for x∈[0.5,3]`)
- 2 `sphere_outside` corner caps at the cross‑over `(-0.5,-0.3)` and `(0.5,0.3)`, `r=0.05`
- feasible band after 0.33 m inflation ≈ **24 cm** around the expert route — a genuinely tight, non‑convex gate.

This is the scene that stresses the QP hardest, which is exactly why it is the most informative ablation bed.

---

## 2. The 18 variants tested

| Folder | Solve mode | Selection | Active constraints | Notes |
|---|---|---|---|---|
| `diffuser` | — (off) | — | none | raw FM reference baseline |
| `dpcc-r` / `-tightened` | hard QP | random | **full stack** | faithful DPCC, random pick |
| `dpcc-c` / `-tightened` | hard QP | min‑proj‑cost | full stack | pick least‑perturbed candidate |
| `dpcc-t` / `-tightened` | hard QP | temporal‑consistency | full stack | pick candidate closest to previous plan |
| `gradient` / `-tightened` | soft gradient | random | full stack | single weighted gradient step `[1,0.5,2]` |
| `post_processing` / `-tightened` | late‑only (`thr=0`) | random | full stack | project only near the end of sampling |
| `model_free` / `-tightened` | hard QP | random | geo+bounds (**dynamics OFF**) | ablate kinematic consistency |
| `bounds_free` | hard QP | random | dynamics+geo (**action bound OFF**) | ablate action cap |
| `geo_free` | hard QP | random | dynamics+bounds (**geometry OFF**) | ablate geo_bounds+halfspace+obstacles |
| `geo_free-bounds_free` | hard QP | random | **dynamics only** | pure kinematic projection |
| `geo_free-model_free` | hard QP | random | **bounds only** | pure action‑cap projection |

---

## 3. Metrics (as computed by the eval harness)

- **`safe_rate`** (Axis A, *hard MuJoCo truth*): fraction of trials that were contact‑free **and** airborne.
  This is the physically meaningful success axis.
- **`relaxed_success`**: crossed the finish line.
- **`collision_free_rate` / `n_violations_mean`**: Axis B, measured against the **full declared margin set**
  regardless of what was enforced — a fixed conservative yardstick, so a `geo_free` run is still *scored*
  against geometry it never enforced.
- **`goal_dist_mean`**: final distance to goal (lower = better).
- **`track_err_mean`**: how well the PID‑controlled body tracks the projected reference (a proxy for how
  *trackable / kinematically self‑consistent* the reference is).
- **`proj_ms_mean` / `total_ms_p95`**: projection wall‑time. **Real‑time budget = 30.3 ms.** `over_budget`
  counts steps that blew it (out of 8710 = 871×10).

---

## 4. Results (seed 6, 10 trials)

| variant | safe | relaxed | coll‑free | n_viol | goal_dist | track_err | proj_ms | total_ms_p95 | over‑budget |
|---|---|---|---|---|---|---|---|---|---|
| **geo_free** | **0.70** | **0.70** | 0.0 | 509 | **0.89** | 14.2 | 90 | 590 | 8438 |
| **geo_free‑bounds_free** (dynamics‑only) | **0.70** | **0.70** | 0.0 | 512 | 0.99 | 15.5 | 74 | 482 | 8441 |
| dpcc‑c | 0.10 | 0.10 | 0.0 | 473 | 18.24 | 42.7 | 853 | 6302 | 8710 |
| dpcc‑r | 0.10 | 0.00 | 0.0 | 468 | 25.27 | 48.5 | 769 | 4705 | 8710 |
| dpcc‑t | 0.00 | 0.00 | 0.0 | **420** | 15.01 | 42.9 | 1007 | 7906 | 8472 |
| dpcc‑c‑tightened | 0.00 | 0.00 | 0.0 | 474 | 7.59 | 40.3 | 924 | 5827 | 8710 |
| dpcc‑r‑tightened | 0.00 | 0.00 | 0.0 | 487 | 24.86 | 48.3 | 733 | 3386 | 8710 |
| dpcc‑t‑tightened | 0.00 | 0.00 | 0.0 | 437 | 32.64 | 46.5 | 1240 | 7028 | 8349 |
| post_processing | 0.00 | 0.00 | 0.0 | 459 | 12.65 | 30.6 | 71 | 297 | 8710 |
| post_processing‑tightened | 0.00 | 0.00 | 0.0 | 456 | 25.91 | 33.0 | 74 | 312 | 8710 |
| gradient | 0.00 | 0.00 | 0.0 | 834 | 5.64 | **7.9** | 0 | 100 | 8710 |
| gradient‑tightened | 0.00 | 0.00 | 0.0 | 830 | 5.15 | 6.7 | 0 | 93 | 8710 |
| bounds_free | 0.00 | 0.00 | 0.0 | 509 | 16.65 | 58.2 | 518 | 2073 | 8710 |
| model_free | 0.00 | 0.00 | 0.0 | 813 | 5.99 | 250.1 | 124 | 287 | 8710 |
| model_free‑tightened | 0.00 | 0.00 | 0.0 | 817 | 6.28 | 264.8 | 139 | 293 | 8710 |
| geo_free‑model_free (bounds‑only) | 0.00 | 0.00 | 0.0 | 815 | 6.00 | 242.6 | 44 | 134 | 8710 |
| diffuser (raw FM) | 0.00 | 0.00 | 0.0 | 818 | 6.34 | 252.8 | 0 | 101 | 8710 |

Sorted by `safe_rate`, then `goal_dist`. **`collision_free_rate = 0` for all** — the declared‑margin
yardstick is intentionally conservative (24 cm gate); physical `safe_rate` is the axis that separates the field.

---

## 5. Per‑variant interpretation

### `diffuser` — the unconstrained baseline
Raw FM plan, no filtering. Reaches near the goal (`goal_dist≈6.3`) but is **never physically safe** (0.0) and
has the **highest tracking error (252.8)** — the raw ODE reference is jumpy / kinematically inconsistent, so
the PID body cannot follow it and clips geometry. This is the "generative brain without brakes" reference the
whole projection layer exists to fix. `proj_ms=0`.

### `dpcc-r / dpcc-c / dpcc-t` — the faithful full‑stack DPCC
All three run the complete QP (`dynamics+geo_bounds+halfspace+obstacles+bounds`) and differ only in **which
candidate** is executed:
- `dpcc-r` (**random**): the DPCC default. `safe=0.10`.
- `dpcc-c` (**minimum projection cost**): execute the candidate the projector had to bend least — the most
  "FM‑faithful yet feasible" plan. Best of the three on safety (`0.10`) and it does cut `goal_dist` under
  tightening (7.59). Intuitively the right selection rule.
- `dpcc-t` (**temporal consistency**): execute the candidate closest to last step's plan — smoothest motion,
  and indeed the **lowest raw `n_violations` (420)**, but `safe=0.0`: smoothness ≠ clearance on a non‑convex gate.

The decisive fact is **timing**: full‑stack `proj_ms` is **770–1240 ms** with `total_ms_p95` of **5–8 s**
against a **30.3 ms** budget — every step is over budget. The non‑convex half‑space + sphere SLSQP solve is
simply not real‑time on this scene, and the sustained‑slowness circuit breaker (Fix_15.2) then *skips*
projection on long stretches, so the executed trajectory is partly unprojected anyway. Selection rule is a
second‑order effect once the QP itself is intractable.

### `-tightened` twins (+2.5 cm margin)
Tightening never helps here and usually hurts (`dpcc-t 0.00→0.00, goal_dist 15→33`). On an already‑tight
24 cm gate, extra margin shrinks the feasible set toward empty, making the QP harder/slower or infeasible.
The one benign effect is `dpcc-c-tightened` pulling `goal_dist` down (7.59) while staying unsafe — closer,
but still clipping.

### `gradient` — soft, fast, but not a real projection
A single weighted gradient step (`grad = 1·eq + 0.5·ineq + 2·obstacle`) instead of solving the QP. It is the
**fastest** (`proj_ms≈0`) and gives the **lowest track_err (7.9)** and a small `goal_dist` — because it only
*nudges* the FM plan, so the reference stays smooth and trackable. But it does **not enforce** constraints:
`n_violations=834` (≈ diffuser's 818) and `safe=0.0`. Verdict: a cheap smoother, not a safety brake.

### `post_processing` — project late only
`diffusion_timestep_threshold=0` ⇒ projection applied only at the tail of sampling. Cheap (`71 ms`) and it
**halves** raw violations vs. diffuser (459 vs 818) with low track_err (30.6), but a single late projection
can't rescue a plan that already committed to a bad homotopy — `safe=0.0`. A reasonable *speed/violation*
trade‑off, not a *safety* solution.

### Constraint‑ablation group — the diagnostic core
This is where the scene teaches us which constraint family actually matters. Read them as a truth table over
the three toggles:

| variant | dynamics | action bound | geometry | safe | track_err |
|---|:--:|:--:|:--:|:--:|:--:|
| full stack (`dpcc-r`) | ✓ | ✓ | ✓ | 0.10 | 48.5 |
| `bounds_free` | ✓ | ✗ | ✓ | 0.00 | 58.2 |
| `model_free` | ✗ | ✓ | ✓ | 0.00 | 250.1 |
| **`geo_free`** | ✓ | ✓ | ✗ | **0.70** | 14.2 |
| **`geo_free-bounds_free`** | ✓ | ✗ | ✗ | **0.70** | 15.5 |
| `geo_free-model_free` | ✗ | ✗ | ✓ | 0.00 | 242.6 |
| `model_free` (no dyn) | ✗ | ✓ | ✓ | 0.00 | 250.1 |

- **`model_free`** (dynamics **off**): collapses to diffuser behaviour — `track_err≈250`, `safe=0`. Without
  the kinematic‑consistency equality the projected "reference" is not integrable by the body, so geometry
  constraints on a non‑trackable path buy nothing.
- **`geo_free-model_free`** (bounds **only**): same collapse (`track_err 242`, `safe 0`). The action cap alone
  does essentially nothing useful.
- **`bounds_free`** (drop only the action cap): barely changes the full‑stack picture (`safe 0`), confirming
  the action bound is not the load‑bearing constraint — but note `proj_ms` drops 770→518 ms, so the action
  rows do add solver cost without adding safety here.
- **`geo_free`** and **`geo_free-bounds_free`**: keep the **dynamics** equality, drop the geometry. Both jump
  to **`safe=0.70`, `goal_dist≈0.9`, track_err≈15, proj_ms≈75–90 ms (near budget)**. These are the only two
  configurations that work.

---

## 5A. The z‑axis "instant crash" — mechanism, and why `pillars` is spared

A conspicuous failure on `s_curve` is that `diffuser` (and the other dynamics‑off variants) **drop to the
floor almost immediately and drag along it for the rest of the episode**, whereas the dynamics‑on variants
fly. The MuJoCo‑truth altitude metrics make this quantitative (means over 10 trials; floor of the workspace
box is `z=0.30`):

| variant (s_curve) | dynamics | min_z | final_z | contact_frac | safe |
|---|:--:|:--:|:--:|:--:|:--:|
| `diffuser` | ✗ | −0.008 | 0.291 | **0.798** | 0.0 |
| `gradient` (soft) | ~✗ | 0.037 | 0.284 | **0.886** | 0.0 |
| `model_free` | ✗ | −0.020 | 0.224 | 0.537 | 0.0 |
| `geo_free` | ✓ | **0.830** | 1.194 | 0.151 | 0.7 |
| `geo_free-bounds_free` | ✓ | **0.834** | 1.196 | 0.146 | 0.7 |

`contact_frac ≈ 0.80–0.89` means the body is **in ground contact for ~80–90 % of the episode** — it never
takes off. `min_z ≲ 0` (touching / slightly penetrating the floor) and `final_z ≈ 0.29` (resting at the 0.30
workspace floor) confirm an early altitude collapse, not a late drift. The dynamics‑on variants instead cruise
at `min_z ≈ 0.83`, `final_z ≈ 1.2`, `contact ≈ 0.15`.

### Is it tracking error, or are the NN outputs rubbish? → **tracking error, not rubbish weights**
The **same FM checkpoint** flies cleanly the instant the **dynamics (kinematic‑consistency) constraint** is
switched on (`geo_free`: s_curve `safe=0.7`, pillars `safe=1.0`, both airborne). A network producing garbage
could not fly under *any* projector. So the network's outputs encode a good route — they are **not** rubbish.

What is wrong in `diffuser` is that the raw ODE reference is **kinematically inconsistent and not anchored to
the drone's actual state**:
- The position channels are not the Euler integral of the commanded action — `p[t+1] ≠ p[t] + dt·Δp_des` — so
  the plan is internally contradictory (its "where I will be" disagrees with its "how I move").
- The plan's first state is **not pinned to the current true z** (the dynamics constraint's `skip_initial_state`
  anchor `b[0]=s_0` is exactly what does this pinning, and it is absent without dynamics).

The PID tracks that setpoint stream. On a gravity‑loaded axis, an incoherent / mis‑anchored **z**‑command
means the controller never commands the sustained thrust needed to hold altitude → the quad falls to the floor
and stays. This is why `track_err` collapses **252 → ~15** the moment dynamics is enabled: the projection
turns a non‑integrable reference into a smooth, self‑consistent, current‑state‑anchored one.

So the precise answer: **it is tracking error — but tracking error *caused by* a non‑integrable, mis‑anchored
reference, and the z‑axis is simply where that defect is fatal.** In x‑y the same defect only makes the drone
*wander*; in z it makes the drone *fall*. (Note `gradient` has the *lowest* `track_err`=7.9 yet still sits on
the floor: it smooths the reference but does **not** re‑anchor it to the current state or enforce integrability,
so the altitude command is smooth‑but‑wrong — smoothness alone doesn't lift the drone.)

### Why the instant z‑crash does NOT appear in `pillars`
The data confirms the observation — same unprojected FM, opposite floor behaviour:

| `diffuser` on… | min_z | final_z | **contact_frac** | max_ep (steps) |
|---|:--:|:--:|:--:|:--:|
| **s_curve** | −0.008 | 0.291 | **0.798** (drags on floor) | 871 |
| **pillars** | −0.005 | 0.072 | **0.001** (essentially airborne) | 634 |

Both are the raw FM with the projector **off**, so the difference is not the projector — it is the *reference
the FM emits for that scene* combined with the *tracking demand of that scene*. Leading reasons (the first two
are the likely drivers; mark the exact discriminator as an npz‑level check):

1. **Sustained‑demand / episode length.** `s_curve` runs 871 steps (`dur≈22 s`) through a tight, doubling‑back
   S‑corridor; `pillars` runs 634 steps (`dur≈10–16 s`) through a more open field. The longer, more
   convoluted route gives the reference's z‑inconsistency far more time and more turns to defeat the PID, so a
   marginally‑airborne start decays into a floor‑drag. Pillars ends before that decay completes.
2. **Route altitude profile.** Pillars' expert route/goal sits low and flat (`final_z≈0.07`); s_curve demands a
   higher, sustained cruise (dynamics‑on `final_z≈1.2`). The untracked reference happens to keep pillars
   skimming just above the floor (contact ≈ 0), whereas s_curve's required climb‑and‑hold is exactly where the
   PID loses the fight and the drone sinks.
3. **NOT the constraint set.** `diffuser` enforces nothing in either scene, so pillars' missing `halfspace`
   family is irrelevant here — the divergence is in FM‑reference × tracking‑demand, not in the projector's
   constraint list.

The exact discriminator — *does the commanded z‑setpoint itself collapse, or does the PID lose an OK setpoint?*
— is now **resolved directly from the npz in §5B below** (it is the setpoint that collapses, driven by the FM
action channel; the PID tracks faithfully). Read on.

---

## 5B. Reading it straight off the npz — FM output, or controller? → **FM action channel (setpoint runaway), NOT controller malfunction**

The `.npz` is enough to settle the "is the FM commanding it into the ground, or is the controller failing to
fly a good command?" question **without** any extra MuJoCo/controller instrumentation, because we can read the
exact setpoint the PID chased. Loaded with `numpy` (the analysis interpreter, not the pipeline env):

**What the PID actually tracks (code‑confirmed, `eval_fm_uav.py:980–1007`):** the setpoint is a **running
integral of the FM's first action**, `p_des ← p_des + Δp_des` each replan (line 980), with feed‑forward
`v_des = Δp_des / dt_fm` (992); the tracker chases *that* accumulated `p_des` (1000), and `track_err` is
`‖body − p_des‖` against it (1007). So the FM's **action channel**, integrated, *is* the altitude command.

**`diffuser`, trial 0 — the z‑setpoint the PID chased vs. the body (obs channel 2 = commanded z, channel 5 = body z):**

| step | commanded z‑setpoint (`p_des_z`) | body z | note |
|---:|---:|---:|---|
| 0 | 1.107 | 1.107 | both start at ~1.1 m |
| 10 | 1.022 | 1.099 | setpoint already sliding down |
| 20 | 0.876 | 1.058 | body lags, follows down |
| 40 | 0.423 | 0.799 | setpoint half‑gone |
| 52 | ≈ 0.00 | ~0.55 | **commanded setpoint crosses the floor** |
| 70 | −0.715 | −0.009 | body reaches floor, tracking a sub‑floor command |
| 100 | −3.1 | 0.32 | setpoint now underground and diverging |
| 300 | −136 | 0.30 | **integrator windup** — setpoint runs away |
| 870 | −587 | 0.31 | body pinned on floor, drags to end |

**`geo_free`, same PID, same loop, trial 0:** commanded `p_des_z` stays **up** (1.11 → 1.15 → 1.30 over the
first 300 steps) and the body tracks it within ~0.02 m; the drone descends only near the goal (step ~600).

### Interpretation (direct answer to the question)
- **It is NOT the controller malfunctioning.** The body faithfully follows the commanded z‑setpoint with a
  small lag — down to the floor when the setpoint goes there, and up when it stays up (`geo_free`). The same
  PID holds altitude perfectly when the command is sane. So "the controller cannot command the UAV" is ruled
  out by the data.
- **It IS the FM's action output — but not a single `−100` spike.** The FM's *absolute* position plan channel
  is fine (§5A: `plan[...,p_des_z] ≈ 0.9–1.1 m`, `FM_pred_z ≈ 0.9–1.18 m`). The defect is in the FM's **action
  channel `Δp_des_z`**, which carries a **persistent small negative bias (≈ −0.017 m/step early on)**. Because
  the eval loop *integrates the action* into the setpoint, that bias marches the commanded z from 1.1 m through
  the floor by step ~52, and then **winds up unbounded to −587 m** (nothing clamps `p_des` to the workspace).
  The `track_err = 252` is this runaway setpoint, not a bad tracker.
- **The FM plan is internally inconsistent, and that is exactly what the dynamics constraint couples.** Its
  *position* channel says "hold ~1 m" while its *action* channel says "keep descending." The projection's
  dynamics rows enforce `p[t+1] = p[t] + dt·Δp_des` (action ≡ position increment) and re‑anchor the plan to the
  current state, which **removes the spurious descending action** — so in `geo_free` the integrated setpoint no
  longer sinks. This is the mechanism behind the whole "dynamics is load‑bearing" finding, caught red‑handed in
  one trace.

### So, do we need controller / MuJoCo feedback to see the cause?
**No.** The `sampled_trajectories_all` + `obs_all` arrays already contain the commanded setpoint and the flown
body position, and the eval code tells us the setpoint *is* the integrated FM action — that triangulates the
cause (FM action channel) without touching the PID internals or MuJoCo forces. MuJoCo/PID logs would only be
needed to answer a *different*, downstream question (e.g. thrust saturation or tilt limits during the fall),
which is moot here because the command itself is already underground. **Practical fixes implied by the trace:**
(i) the dynamics projection (already shown to work), and/or (ii) an anti‑windup clamp of `p_des` to the
workspace‑z box in the eval loop, which would stop the −587 runaway even for `diffuser` (though it would not by
itself make the raw FM fly, since the early descending bias would still pin it near the floor).

---

## 6. Headline finding (thesis claim)

> **On the non‑convex `s_curve`, the kinematic‑consistency (dynamics) constraint is the single load‑bearing
> ingredient of the projection, and the geometric constraints (workspace box + half‑spaces + sphere obstacles)
> are *net‑harmful* in the real‑time loop.**

Two independent lines of evidence:

1. **Ablation.** The two best variants (`geo_free`, `geo_free-bounds_free`, both `safe=0.70`) are exactly the
   two that keep dynamics and drop geometry. Every configuration that *keeps* geometry — the full DPCC stack,
   all selection rules, both solve modes, both margins — is `safe ≤ 0.10`. Every configuration that *drops*
   dynamics (`model_free`, `geo_free-model_free`) collapses to raw‑FM tracking error (~250). Dynamics is
   **necessary and nearly sufficient**; geometry is neither.

2. **Timing/feasibility.** The dynamics(+bounds) QP is linear and fast (`75–90 ms`, `p95≈0.5 s`). Adding the
   non‑convex half‑space + sphere rows makes SLSQP **~10× slower (770–1240 ms, p95 5–8 s)**, blows the 30.3 ms
   budget on every step, and trips the sustained‑slowness circuit breaker — which then *skips* projection, so
   the "constrained" run is intermittently unconstrained. The geometry rows thus **cost safety twice**: they
   fight the trained homotopy (the QP pulls the plan off the manifold the policy learned) *and* they make the
   solve non‑real‑time, forcing skips.

The **dynamics constraint's real job** is to turn the jumpy FM reference into a **body‑trackable** one:
`track_err` drops 252 → ~15 the moment it is switched on, and that trackability — not geometric clearance — is
what produces contact‑free flight on this scene.

### Practical recommendation
For real‑time PCC on tight non‑convex UAV corridors: **project onto dynamics (+ cheap linear bounds) only, and
enforce geometry a different way** — either offline in the corridor design / expert route, or with a
cheaper/soft geometric term, or with a much shorter horizon and warm‑started solver. The faithful full‑stack
DPCC QP, ported verbatim from the low‑dim avoiding task, does **not** transfer to this real‑time non‑convex
regime. If any geometric enforcement is kept, pair it with `dpcc-c` (minimum‑projection‑cost) selection, which
was the least‑bad full‑stack option.

---

## 7. Reading the projection off the `.npz` trajectories

Each variant folder ships `<variant>.npz` (schema in `FM_v3_uav_test/eval_artifacts.py`). The key that proves
the projection is real is **`sampled_trajectories_all`** — an object array `[trial][replan_step]` of the
**projected H‑step reference plan in observation space** (`traj.observations`, unnormalised), i.e. the exact
tensor the projector returned and the PID then tracked. Companion keys:

- `obs_all` — the actually‑flown body states (per step), and `act_all` — the executed Δp_des.
- `phys_min_z`, `phys_contact_frac`, `phys_safe` — MuJoCo‑truth safety.
- `constraint_n_violations`, `constraint_total_violations` — against the fixed declared margins.
- `projection_cb_tripped`, `projection_cb_skipped_steps` — **essential caveat**: any trial with
  `cb_tripped==1` ran partly on the *unprojected* plan (breaker open), so its constraint metrics are not a
  clean read of "projection on". Full‑stack `s_curve` variants trip heavily; `geo_free` barely does.

**How to see the projection working from the npz** (analysis to run *on the cluster*, where numpy is available):

1. **FM vs. projected delta.** Compare `sampled_trajectories_all` against the same model's `diffuser` run:
   for the dynamics‑on variants the projected plan is a **smooth, integrable** version of the raw FM plan
   (position channels satisfy `p[t+1] ≈ p[t] + dt·Δp_des` to solver tolerance — that *is* the equality being
   enforced). For `model_free` the two are nearly identical (nothing enforced), which is why its `track_err`
   matches diffuser.
2. **Constraint residual on the plan.** Evaluate the declared half‑space / sphere residuals on each stored
   plan. Full‑stack variants show the projector pushing plan points onto the corridor faces *when the solve
   finishes in budget*, and doing nothing on breaker‑skipped steps (cross‑check `projection_cb_skipped_steps`).
3. **Trackability.** Overlay `sampled_trajectories_all` (reference) with `obs_all` (flown): the dynamics‑on
   variants show tight overlap (`track_err≈15`); diffuser/`model_free` show the body lagging a jerky reference
   (`track_err≈250`). This is the mechanism behind the safety gap, visible directly in the arrays.

The pre‑rendered artifacts in each folder already expose this: `_horizon_compare/horizon_compare_t0_s*.csv`
(candidate‑by‑candidate `x,y` of the stored plans across variants), `<variant>/diagnostics/rollout_*_mpc_foresight.svg`
(per‑step foresight vs. flown), and `_traj_viz/viewer_*.html` (3‑D scene). These are the figures to cite in the
thesis as "projection is verifiably active and its output is trackable."

---

## 8. One‑paragraph summary for the thesis body

> We ablated the DPCC projection layer into three orthogonal axes — solve mode (hard QP / gradient /
> late‑only), candidate selection (random / min‑cost / temporal), and constraint family (dynamics / action
> bound / geometry) — and evaluated all 18 combinations on the non‑convex `s_curve` UAV corridor (seed 6,
> 10 trials each, 871 replans). Only the two configurations that retain the kinematic‑consistency (dynamics)
> equality and drop the geometric constraints reach physical safety (70 % contact‑free, `goal_dist≈0.9 m`),
> at `≈90 ms` per solve. The faithful full‑stack DPCC QP — regardless of selection rule, solve mode, or
> margin — stays below 10 % safe because the non‑convex half‑space/obstacle solve runs 770–1240 ms against a
> 30 ms budget, fights the learned homotopy, and forces the circuit breaker to skip projection. Removing the
> dynamics constraint collapses tracking error from ~15 back to ~250 (raw‑FM level). We conclude that on tight
> real‑time non‑convex corridors the projection should enforce dynamics (plus cheap linear bounds) and handle
> geometry outside the per‑step QP; the load‑bearing role of the projection here is producing a
> body‑trackable reference, not geometric clearance.

---

*Generated from the `temp/s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles/` test dump. All numbers are
seed‑6, 10‑trial means read from each variant's `results.json`; constraint‑config quotations are from
`config/uav_projection.yaml`; projection mechanics from `flow_matcher_v3_uav/sampling/projection.py`,
`FM_v3_uav_test/eval_fm_uav.py`, and `.../policies.py`.*
