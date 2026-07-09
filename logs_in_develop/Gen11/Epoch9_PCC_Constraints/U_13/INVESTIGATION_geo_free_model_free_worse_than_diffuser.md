# U_13 — Investigation: the UAV corridor projection-variant ordering (why partial ablations score BELOW raw `diffuser`)

**Date:** 2026-07-09. Status: **INVESTIGATION ONLY — no code changed. No bug found in the
ablation logic.** (Supersedes the first draft, which over-attributed the effect to the action
clamp; corrected in §4 after the user supplied the full variant ordering.)

**Trigger (user's words):** *"`geo_free-model_free` outputs way worse results than pure
`diffuser` — makes no sense; I thought a projected variant should be at least as good."*
**Extended observation (the key new data):** on corridor, the trajectory quality orders as

> `model_free`  <  `geo_free-model_free`  <  `geo_free-bounds_free`  <  **`diffuser`**  <  `post_processing`  <  `dpcc-c/r/t`

*"the benign ordering really confuses me — are some new constraints DESTROYING the original
raw outputs???"*

**Verdict up front:** Yes — and it is expected, not a bug. Geometry/bounds constraints
applied **without the dynamics constraint** (or applied repeatedly mid-generation without it)
*do* corrupt the raw FM output. The dynamics constraint (DC_FIX) is the **load-bearing**
piece: it is the only thing that couples the *executed action* to the *constrained states*.
Strip it and projection stops being a coherent correction and becomes a decoupled
perturbation — and the MORE geometry you project without it, the worse. That single principle
reproduces the entire ordering above.

---

## 1. The variant map (verified against `setup_dpcc_projector`, `eval_fm_uav.py:674-749`)

Two axes matter: **which constraint families are active**, and **when/how often** the
projector runs inside the ODE loop.

| variant | dynamics (DC_FIX) | action bound | geometry (box+walls+pillars) | projection schedule |
|---|:--:|:--:|:--:|---|
| `model_free` | ❌ | ✅ | ✅ (**full**) | guided, last 50% |
| `geo_free-model_free` | ❌ | ✅ | ❌ | guided, last 50% |
| `geo_free-bounds_free` | ✅ | ❌ | ❌ | guided, last 50% |
| `diffuser` | — | — | — | **none** (`projector=None`) |
| `post_processing` | ✅ | ✅ | ✅ (full) | **once, final step** (threshold 0.0, `:749`) |
| `dpcc-c/r/t` | ✅ | ✅ | ✅ (full) | guided, last 50% + candidate selection |

Gate code confirms the family columns (`:674` geo_bounds, `:688` bounds, `:726` dynamics,
`:732` halfspace, `:742` obstacles). Schedule: `diffusion.py:209` computes `snapping_start_idx`
from the threshold; `post_processing` sets threshold 0.0 → only `loop_idx == flow_steps-1`
fires; `dpcc-*` keep 0.5 → last ~10 of 20 steps fire. **No dispatch bug** — the ordering is
produced by what these variants *are*, not by a defect.

---

## 2. The one mechanism that explains everything: dynamics is the action↔state coupling

**Where each family binds (UAV 12-D transition `[act(0,1,2) | p_des(3,4,5) | p(6,7,8) | v]`):**
- **Geometry** (box/walls/pillars) binds to **actual position `p` (dims 6,7,8)** only —
  DPCC-faithful (`setup_dpcc_projector` docstring; `eval_fm_uav.py:674-745`).
- **Action bound** binds to **`act` (dims 0,1,2)** only.
- **What the closed loop actually executes** is the **first action `act` (0,1,2)**
  (`eval_fm_uav.py:956, 971`).
- **Dynamics DC_FIX** is the ONLY family that ties them together:
  `p_des[t+1]=p_des[t]+act[t]` and `p[t+1]=p[t]+act[t]` (`eval_fm_uav.py:726-730`), and via
  `skip_initial_state` it also **anchors the plan to the current measured state**
  (`projection.py:99-108`, `b[0]=s_0`).

**With dynamics ON:** the projector solves for a *self-consistent, current-state-anchored*
(action, state) plan that also satisfies geometry/bounds. When geometry pushes the predicted
`p` off a wall, DC_FIX propagates that into a correspondingly adjusted `act` — the executed
command stays coherent with the collision-free states. Projection is a *meaningful* correction.

**With dynamics OFF (`model_free`):** action dims and state dims are **decoupled** in the QP.
Geometry yanks the state channels `p`; the action channels are not tied to them. But this is
**guided** generation: after each mid-ODE projection, integration continues and the learned
velocity field is re-evaluated on the geometry-shifted `x` (`diffusion.py:263-278`). So the
shifted states **feed back** into the next step's predicted action — but *incoherently*,
because no DC_FIX keeps them consistent. Over the last 10 steps the executed action is dragged
around by geometry corrections it is never reconciled with. **That is the corruption.**

**Corollary (reproduces the left half of the ordering):** the more geometry you project
without dynamics, the more yanking, the worse the executed action.
- `model_free` keeps the **full** geometry stack → maximum decoupled yanking → **worst**.
- `geo_free-model_free` keeps only the mild action clamp, no geometry → less corruption →
  above `model_free` but still below `diffuser`.
- `geo_free-bounds_free` keeps **dynamics alone**, nothing to enforce → the projection is a
  consistent, anchored re-projection of the FM plan (near no-op). It sits **just below**
  `diffuser`: the re-anchoring to the measured state + the projector-path overhead perturb the
  raw output slightly, but there is no decoupled family fighting it.

---

## 3. Why the right half (`post_processing`, `dpcc`) BEATS `diffuser`

Both keep the **full stack including dynamics**, so their projections are coherent (§2), AND
they add geometry that genuinely helps corridor (staying off the walls → higher `safe`/goal
rate). The difference from the corrupting variants is entirely the presence of dynamics + how
the projection is scheduled:

- **`post_processing`** projects **once, on the final, fully-formed FM sample** (threshold
  0.0). It takes a good trajectory and snaps it to feasibility a single time — no mid-ODE
  feedback loop to destabilize. Safest use of projection → modest gain over `diffuser`.
- **`dpcc-c/r/t`** project throughout the last 50% *with* dynamics keeping every step
  consistent, **plus candidate selection** (`dpcc-c` = min projection cost, `dpcc-t` =
  temporal consistency, `dpcc-r` = random; `_selection_for`, `eval_fm_uav.py:768-774`). Guided
  feasibility + best-candidate pick → best.

So the full ordering is three regimes, not a paradox:

| regime | variants | what projection does | vs. `diffuser` |
|---|---|---|---|
| **A. dynamics removed** | `model_free`, `geo_free-model_free` | decoupled geometry/bounds corrupt the executed action via the guided feedback loop; worse with more geometry | **worse** (much) |
| **B. dynamics only** | `geo_free-bounds_free` | consistent, anchored re-projection; nothing enforced | ~equal, **slightly worse** |
| **C. full stack, coherent** | `post_processing`, `dpcc-*` | consistent correction + real geometry help | **better** |

---

## 4. Correction to the first draft (the action-clamp claim)

The first version of this doc argued the *action clamp itself* actively degrades results.
That was over-stated. Worked out precisely: with dynamics OFF and only the `bounds` rows, the
QP (`Q=I`, axis-aligned box on decoupled action coords) **separates per-coordinate** and
reduces to an elementwise clamp `act ← clip(act, −1, +1)` in normalized space — idempotent, a
**no-op whenever the FM stays in the training action range**. In isolation it should be ≈
`diffuser`, not "way worse." The real driver of the left-half degradation is the **guided
mid-generation feedback without dynamics coupling** (§2), which is present for
`geo_free-model_free` too (it still runs the guided schedule), and is dominant for
`model_free` because full geometry is doing the yanking. The clamp is a minor contributor, not
the cause.

---

## 5. Direct answer to "are new constraints destroying the raw outputs?"

**Yes — but only geometry/bounds constraints applied WITHOUT the dynamics constraint, and
especially when applied repeatedly mid-generation.** The dynamics constraint is not optional
garnish; it is the mechanism that makes any projection coherent with what the controller
executes. The `*_free` ablations that remove it are **diagnostic probes, not deployable
configs** — their underperformance vs. `diffuser` is the *expected and correct* signal that
dynamics coupling is essential. The deployable variants (`dpcc-*`, and `post_processing`) keep
it and behave as intended (≥ `diffuser`). Nothing is destroying the pipeline; the ablations
are doing exactly what an ablation should: showing you which piece is load-bearing.

---

## 6. Diagnostic ladder to confirm the mechanism on the cluster

Run on corridor, same seed. Each pair isolates one claim in §2–3:

1. `diffuser` — raw FM baseline.
2. `geo_free-bounds_free` (**dynamics only**) — should be ≈ #1, marginally below. Confirms
   regime B (consistent re-projection is near-no-op).
3. `geo_free-model_free` (**bounds only, dynamics off**) — guided clamp without coupling.
4. `model_free` (**full geometry, dynamics off**) — should be the **worst**; confirms
   "more decoupled geometry = more corruption."
5. `post_processing` (full stack, once) and `dpcc-c` (full stack, guided+select) — should
   **beat** #1; confirms regime C.

Read per variant from `results.json` / npz:
- `success.strict` split into `goal.reached` vs `physical.safe` — is corruption causing
  crashes (`safe`↓) or goal-misses (`reached`↓)?
- `n_fm_steps` vs `max_episode_length` (396) — do the corrupting variants burn the full budget?
- **Executed-action drift:** log `‖act_projected − act_raw‖` for #3/#4 vs #1 at step 0 — direct
  evidence that geometry-without-dynamics moves the executed command (the smoking gun for §2).
- `timing.proj_cost` — magnitude of the projection each step.

Expected outcome if §2 is right: #4 ≫ #3 in executed-action drift (full geometry yanks harder),
#2 ≈ #1 (dynamics-only barely moves the action), #5 improves `reached`/`safe` over #1.

---

## 7. Files inspected (no edits)
- `FM_v3_uav_test/eval_fm_uav.py` — `setup_dpcc_projector` gates + schedule (`:674-749`),
  DC_FIX binding (`:726-730`), executed-action path (`:956,971`), `_selection_for` (`:768-774`),
  `projector=None` for diffuser (`:1204-1226`), fixed budget/early-exit (`:900,930-1052`).
- `flow_matcher_v3_uav/models/diffusion.py:200-278` — guided projection schedule + the
  project→continue-integrating feedback loop.
- `flow_matcher_v3_uav/sampling/projection.py:70-155` (QP, `skip_initial_state` anchoring
  `:99-108`), `:255-334` (`SafetyConstraints` bound normalization).
- `config/uav_projection.yaml` — variants, per-scene `constraint_types`, `action_bounds:'auto'`, inflation.
- `../U_8_new_projection_var_upgrade/CHANGELOG_U8_projection_variant_ablation.md` — the truth table this confirms against.
