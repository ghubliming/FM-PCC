# U16 fix — `-pdes`: bind the geometry to the setpoint, not the lagging drone

**Date:** 2026-09-13 · **Gen:** 15 · **Follows:** [`CHANGELOG_20260913_corridor_v2_wide_slide.md`](CHANGELOG_20260913_corridor_v2_wide_slide.md) §6.1
**Run:** `Slurm_Codes/temp_bash/eval_20260913_u16fix_pdes.sh` (same scene, slide, K, trials and FRAC as U16)

## 1. Diagnosis — from the U16 rollout step logs, no new run

U16 failed by **overshoot**: both projected arms ended at y ≈ −0.63 against a −0.37 slide limit and
missed the goal (0/2, diffuser 2/2). The per-step `OBS`/`FM` lines show why (dpcc-t-bounds_free, route C, step 219):

```
OBS   p_des=(1.263, -0.294)   p=(0.880, -0.009)   track_err=0.469m
FM    first action = (0.000, -0.022, ...)            <- full sideways push at the cap, zero forward
```

- The drone **lags its own setpoint by 0.37–0.50 m** on every rollout, the diffuser included.
  That is the normal tracking behaviour of this pipeline.
- Geometric constraints bind to the **actual** position `p` (dims 6–8). Each plan starts at the
  lagging drone, and the dynamics row `p ← act` models it as moving instantly. So while the real drone
  is still catching up, the projector sees a violation and pushes `act` again. The setpoint integrates
  those pushes and **winds up**.
- Measured: when HardFlow's drone first cleared the slide on C (x = 1.14, y −0.17), its setpoint was
  already at −0.60, **0.44 m deeper**. Both arms end with the setpoint at −0.65, the drone follows, and
  nothing brings it back. The corridor model never learned sideways motion.

The Gen11 design study (`logs_in_develop/Gen11/Epoch9_PCC_Constraints/Plan/STUDY_DPCC_constraint_dim_binding.md`
§4) justifies `p`-only binding by saying `p` and `p_des` are "the running Euler integral of the same
action **from the same initial state**". On the UAV the initial states differ by the tracking lag, and that
premise is what breaks. The study's own caveat anticipated lag, and proposed tightening. Tightening
answers a setpoint that is not pushed far enough; U16 shows the opposite, a setpoint pushed too far.

## 2. The fix — opt-in variant token `-pdes`

| file | change |
| :-- | :-- |
| `mix_uav_test/eval_mix_uav.py` `setup_dpcc_projector` | if `'pdes' in variant.split('-')`: halfspace, obstacle and workspace-box rows bind to `p_des` (dims 3–5) instead of `p` (6–8). Dynamics, action bounds and everything else unchanged. **No token → byte-identical to before** |
| `mix_uav_test/eval_mix_uav.py` `rollout_one` | for `-pdes` variants the per-replan `x_active` gate reads the setpoint's x (`p_des[0]`), consistent with the binding. Default variants still use the drone's x |
| `mix_uav_test/eval_mix_uav.py` `UAV_MIX_VARIANTS` check | `<existing variant>-pdes` is accepted without adding it to any shared list (it is a toggle, not a new variant). Other unknown names are still rejected |
| `mix_uav/sampling/hardflow_projection.py` | `'-pdes'` added to `_TOGGLE_SUFFIXES`, as that list's own comment requires ("keep in sync with the variant-name gates"). Otherwise `hardflow_new-t-pdes` would hide its `-t` selector and silently run at B = 1 |

HardFlow builds its NLP from the same `setup_dpcc_projector` constraint list, so `hardflow_new-pdes`
gets the same binding. **Violations are still scored on the actual drone position**
(`_exec_constraint_violations`), unchanged. The fix is judged against the real drone, not the setpoint.

Local checks (no torch/mujoco here): `py_compile` on both files; the box layout is finite on dims 6/8
by default and 3/5 with `-pdes`; selection parsing is unchanged; the allow-list accepts the `-pdes` names
and still rejects an unknown base.

### Why this is not a corridor-specific change

It is a variant-level toggle, like `geo_free`, `bounds_free` and `model_free`, and applies identically
on any scene. Existing variants, results and the paper's DPCC-faithful default are untouched. **If
`-pdes` is adopted for the paper, every scene and the diffusion baseline must be run with it**, since it is
a projector design choice.

## 3. Prediction (before the run)

- **No overshoot:** drone y at x = 2.0 near −0.37 (U16: −0.63).
- **Fewer or shallower slide violations** than U16 (30 steps, 8–9 cm): the setpoint leads by ~0.4 m, so it
  meets the slide before the drone does.
- **Horizon steps past x = 2** still see the infinite slide line, so the drone may end a few cm below −0.37.
- **Goal:** L (goal y −0.12) is likely reachable. **C (goal y 0.00) likely still misses by ~7 cm**,
  because nothing steers the drone back up after the slide.

## 4. Run record

| job | submitted | status |
| --: | :-- | :-- |
| (id not recorded) | 2026-09-13 | DONE. Results `temp/1309/test_Pde/` (merged remote folder: U16 arms byte-identical to U16 + the two `-pdes` arms), child log not downloaded |

### 4.1 Result of fix 1 (2 trials, K = 3, one seed: injection, not a DA)

| arm | route | success | goal dist | collision-free | slide viol. steps | max depth | y at x = 2.0 | y at x = 2.8 |
| :-- | :-- | :-: | --: | :-: | --: | --: | --: | --: |
| dpcc-t-bounds_free (U16) | L / C | 0 / 0 | 0.73 / 0.77 | 0 / 0 | 30 / 30 | 8 / 8 cm | −0.61 / −0.63 | −0.67 / −0.64 |
| **dpcc-t-bounds_free-pdes** | L / C | **1** / 0 | 0.30 / 0.42 | 0 / 0 | 37 / 49 | **2 / 2 cm** | **−0.35 / −0.35** | — / −0.39 |
| hardflow_new (U16) | L / C | 0 / 0 | 0.75 / 0.81 | 0 / 0 | 30 / 32 | 8 / 9 cm | −0.64 / −0.64 | −0.65 / −0.62 |
| **hardflow_new-pdes** | L / C | 0 / 0 | 0.55 / 0.65 | 0 / 0 | 36 / 45 | **1 / 1 cm** | **−0.37 / −0.37** | −0.57 / −0.57 |

All arms crossed the finish line (`relaxed_rate` 1.00). None had divergence aborts or circuit-breaker trips.

**Prediction check:**
- ✅ **No overshoot:** exit y −0.35/−0.37 vs the −0.37 limit (U16: −0.63).
- ✅ **Much shallower:** 8–9 cm → 1–2 cm.
- ❌ **Not fewer violation steps:** 30–33 → 36–49.
- ✅ DPCC reached the goal on L. ✅ C still misses (by 12 cm, predicted 7).

**Why more steps but shallow:** the projector now places the setpoint exactly on the limit, and the drone
rides it with small tracking error. For `-pdes`, ~80% of violating steps are < 5 mm and **none exceed
2.5 cm** (DPCC C: 38 < 5 mm, 5 at 5–10 mm, 5 at 1–2 cm, 1 at 2–2.5 cm; HF: all < 1 cm). For U16's
p-binding, 24/30 steps were > 2.5 cm. This is the zero-margin case that DPCC's `-tightened` exists for.

**HardFlow drift after the exit** (−0.57 at x = 2.8 vs DPCC's −0.39) exposed a pre-existing bug (§6).

## 6. 🔴 Pre-existing bug found: HardFlow ignored `x_active`

`HardFlowPolicy` builds its NLP once, in `_run_variant`, from `setup_dpcc_projector(..., return_constraint_list=True)`
with **no `current_x`**. By that function's own contract, that means **every x_active halfspace is
active everywhere**. The per-replan rebuild in `rollout_one` did `policy.projector = rebuild_projector(x)`.
`HardFlowPolicy.projector` is `None` by design and never used for constraints, so the rebuild was a
silent no-op for this arm.

**Affected:** every HardFlow result on a geometry with x_active halfspaces.
- **`s_curve` / `s_curve_hg`:** both segments' walls were enforced at once, i.e. seg1's y ∈ [−1.25, −0.35]
  AND seg2's y ∈ [0.35, 1.25]. That is an empty intersection, so the problem was infeasible at every x.
  **HardFlow rows in `Campaign_20260907_five_missions/DA_20260910_T4_s_curve_budget_explore.md` and
  `DA_20260909_T5_mjpc_vs_pid_s_curve.md` were produced under this bug and are not valid HardFlow
  measurements.** Both files now carry a warning pointing here.
- **`corridor_*`:** walls with window [−2, 2] stayed on at |x| > 2. That is harmless, because the routes
  satisfy them there. U16's slide, however, kept pushing past the exit.
- Pillars (no x_active halfspaces) and every DPCC arm are **not affected**.

**Fix (fix 2):**
- `HardFlowNLP.update_constraint_list` rebuilds the SLSQP projector from the new list. The ipopt backend
  builds its CasADi problem once, so it gets a one-time warning instead of a silent pretence.
- `HardFlowPolicy.update_constraint_list` forwards to it.
- In `_run_variant` the rebuild closure returns the constraint list for HardFlow variants.
- In `rollout_one` it is handed to `policy.update_constraint_list` when the policy has one; the DPCC path is unchanged.

## 7. Fix 2 — tightening + HardFlow gating

`Slurm_Codes/temp_bash/eval_20260913_u16fix2_tightened.sh`, same scene, slide, K, trials and FRAC:

| variant | tests |
| :-- | :-- |
| `dpcc-t-bounds_free-tightened-pdes` | DPCC's own margin (`enlarge_constraints` 0.025 m) > every `-pdes` penetration seen |
| `hardflow_new-pdes` | the gating fix alone. Overwrites the remote `hardflow_sls-pdes`; the buggy copy is kept locally in `test_Pde` |
| `hardflow_new-tightened-pdes` | gating fix + margin |

Also in this pull: the `UAV_MIX_VARIANTS` allow-list resolves composed toggles (`-pdes`, `-tightened`,
`-bounds_free`, `-geo_free`, `-model_free`) to their core name, and rejects toggles on `diffuser`.
`_TOGGLE_SUFFIXES` in `hardflow_projection.py` already strips them for HardFlow's batch-size rule.

**Demo criterion.** The user's own definition is "slide along the halfspace and cross the final line",
which maps to `relaxed_and_constraints` (crossed the line, zero violations). Strict success also needs the
goal point, which route C cannot reach after being pushed to −0.37: nothing steers the corridor model
back up. For the paper run, pick the metric explicitly before launching, and report both.

**Prediction (before the run):**
- `dpcc-…-tightened-pdes` is collision-free on L and C.
- `hardflow_new-pdes` no longer drifts past the exit (y at x = 2.8 near −0.39, was −0.57), with shallow violations remaining.
- `hardflow_new-tightened-pdes` is collision-free on L and C.

## 5. What comes after (user, 2026-09-13)

If the fix passes, the next job is the **full corridor paper evaluation**, whose goal is to show that
FM-PCC works on `corridor_v2`. An engine ranking of af > mf > fm > diffusion is a bonus, expected but
not required. Every arm, including the diffusion K20 baseline, must run under the same settings:
`corridor_v2` scene, `FMPCC_SAFE_EPS_FRAC=1.0`, and the same binding (`-pdes` or not).
