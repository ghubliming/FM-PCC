# U16 — Fix 1 + Fix 2: what they change on `corridor_v2` (full account for review)

**Date:** 2026-09-13 · **Gen:** 15 · **Commits:** fix 1 = `ca28806f` (U16 hotfix2), fix 2 = `07efb114` (U16 hotfix.3)
**Test bench:** geo entry `corridor_v2_slide` (U16, commit `074152e3`) · detailed working log: [`CHANGELOG_20260913_u16fix_pdes_binding.md`](CHANGELOG_20260913_u16fix_pdes_binding.md)

---

## 0. Summary

**Neither fix touches `corridor_v2` itself.** The scene XML, the geo entry, the config file, the model, the
dataset, the controller, the violation scorer and the metric definitions are byte-identical to the U16 commit.
`git diff 074152e3 --stat` over `config/`, `d3il/` and `eval_artifacts.py` is empty. The fixes change one thing:
**how the projector applies the constraints it is given**. Two parts are opt-in via the variant name; one part
repairs a HardFlow bug and applies to every HardFlow run.

| # | change | opt-in? | what it does on corridor_v2 |
| :-- | :-- | :-- | :-- |
| fix 1 | `-pdes` variant token | **yes** | the slide, walls and box constrain the **setpoint** `p_des` instead of the lagging drone `p` |
| fix 2a | HardFlow honours `x_active` | **no** (bug fix, all `hardflow*` variants) | HardFlow switches windowed halfspaces per replan, as DPCC always did |
| fix 2b | `-tightened` composed with `-pdes` | **yes** (existing DPCC token, **no new code**) | planning margin 0.310 → 0.335 m |
| fix 2c | variant allow-list resolves composed tokens | n/a (plumbing) | lets `hardflow_new-tightened-pdes` etc. be requested |

---

## 1. What was held fixed across U16, fix 1 and fix 2 (the test bench)

| item | value |
| :-- | :-- |
| MuJoCo scene | `scene_corridor_v2.xml`, walls at y = ±1.0 (inner faces ±0.95) |
| geo entry | `corridor_v2_slide`: slide `[[-2.0, 0.95], [2.0, -0.05]]`, side below, x_active [−2, 2]; walls ±0.95; caps at (±2, ±1.0) r 0.05 |
| model / data / routes / goals | trained `corridor` mf checkpoint (`H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet`, seed 6), `scene: corridor` |
| run settings | `FMPCC_SAFE_EPS_FRAC=1.0`, `FMPCC_SAFE_EPS_MODE=scaled`, K = 3, T = 0.5, mpc4, `pid_stopgo`, 2 trials (L, C), `record=gif` 320 px |
| planning inflation | `r_drone 0.31 + margin_base 0.0` (+ `enlarge_constraints 0.025` only for `-tightened`) |
| scorer | `_exec_constraint_violations` on the **actual** drone position, `inflation.r_drone = 0.31`, unchanged |
| metrics | strict success (goal within 0.30 m + safe), S&C = strict + zero violations, same as pillars / s_curve |

---

## 2. Fix 1 — `-pdes`: bind the geometry to the setpoint (`ca28806f`)

### 2.1 The problem it fixes (measured in U16)

The UAV state in each plan is `[act | p_des | p]`. `p_des` is the commanded setpoint, which the action integrates
exactly (`p_des ← act`). `p` is the real drone, which the PID drags toward `p_des` with lag. **On every corridor
rollout the drone trails its setpoint by 0.37–0.50 m in x**, the diffuser included.

The default projector binds halfspaces, obstacles and the box to `p` (dims 6–8). The Gen11 design study justified
this because `p` and `p_des` are "the running Euler integral of the same action **from the same initial state**".
On the UAV they do not start from the same state:
1. Each plan starts at the lagging drone.
2. The dynamics row `p ← act` then assumes the drone moves instantly.
3. So while the real drone is still catching up, the projector sees a violation and pushes `act` again.
4. The setpoint integrates every push and **winds up**.

U16 evidence (`dpcc-t-bounds_free`, route C, step 219):
```
OBS   p_des=(1.263, -0.294)   p=(0.880, -0.009)   track_err=0.469m
FM    first action = (0.000, -0.022, ...)        # full sideways push at the cap, zero forward
```
When HardFlow's drone first cleared the slide (x = 1.14, y −0.17), its setpoint was already at −0.60.
Both arms ended with the setpoint at −0.65 against a −0.37 limit. The drone followed, and success was 0/2.

### 2.2 The code

`mix_uav_test/eval_mix_uav.py`:
- **`setup_dpcc_projector` (L1268–1271):** `_geo_on_pdes = 'pdes' in variant.split('-')`. If set, the position
  entries of `_DIM` become `x:3, y:4, z:5` (`p_des`) instead of `x:6, y:7, z:8` (`p`). This moves:
  - halfspace rows: `_hs = {'x': _DIM['x'], 'y': _DIM['y']}`, L1345;
  - obstacle rows: `dims = [_DIM[d] ...]`, L1356;
  - workspace-box rows: offset `_goff` 3 vs 6, L1295.

  Dynamics rows (`p_des ← act`, `p ← act`), action bounds, margin and selection are **unchanged**.
- **`rollout_one` (L1677):** the per-replan `x_active` gate reads `p_des[0]` for `-pdes` variants, `p[0]` otherwise.
  The gate follows the binding: a constraint on the setpoint is switched by the setpoint's position.

`mix_uav/sampling/hardflow_projection.py`:
- **`_TOGGLE_SUFFIXES` (L825):** `'-pdes'` added. HardFlow's batch-size rule strips toggles before reading the
  `-r/-c/-t` selector, so `hardflow_new-t-pdes` keeps B = 4 instead of silently falling to B = 1.

### 2.3 What it does on corridor_v2, concretely

- **With `-pdes`:** the projector requires the setpoint to satisfy "y ≤ slide − margin". Because the setpoint is
  the exact integral of the action, it stops pushing once the setpoint is legal, so there is no windup. The drone
  follows the setpoint about 0.4 m behind in x, where the slide limit is higher. That is the safe side of a
  descending slide.
- **Without `-pdes`:** identical to U16. No variant, geometry or result changed meaning.
- **Scoring is not moved:** the result is still judged on where the real drone was.

### 2.4 Evidence (U16 vs fix 1, same bench)

| arm | y at x = 2.0 (limit −0.37) | max slide penetration | slide viol. steps L / C | success L / C | collision-free |
| :-- | --: | --: | --: | :-- | :-: |
| dpcc-t-bounds_free (U16) | −0.61 / −0.63 | 78 mm | 30 / 30 | 0 / 0 | 0/2 |
| dpcc-t-bounds_free-**pdes** | **−0.35 / −0.35** | **22 mm** | 37 / 49 | **1** / 0 | 0/2 |
| hardflow_new (U16) | −0.64 / −0.64 | 90 mm | 30 / 32 | 0 / 0 | 0/2 |
| hardflow_new-**pdes** (fix 1, before 2a) | −0.37 / −0.37 | 28 mm | 36 / 45 | 0 / 0 | 0/2 |

The overshoot is gone and penetration fell by ~75%. The remaining violations are the drone **riding the limit**:
for `-pdes`, ~80% of violating steps are < 5 mm and none exceed 25 mm.

### 2.5 Caveats

- **This is a projector design choice, not a bug fix.** It departs from the Gen11 `p`-only binding. **If used for
  the paper, every arm must use it**, including the diffusion K20 baseline and any other scene in the same table.
- Plans still model `p ← act` (instant drone). `-pdes` makes that mismatch harmless for the geometry but does not remove it.
- Horizon steps beyond the slide's end still see the infinite line until the setpoint's gate switches it off (see §3.5).

---

## 3. Fix 2 (`07efb114`)

### 3.1 Fix 2a — HardFlow honours `x_active` (bug fix, not opt-in)

**Bug.** `_run_variant` built `HardFlowPolicy` once from
`setup_dpcc_projector(..., return_constraint_list=True)` **without `current_x`**. By that function's contract,
that includes **every** x_active halfspace at every x. The per-replan rebuild in `rollout_one` assigned
`policy.projector`, but `HardFlowPolicy.projector` is `None` by design and its NLP never reads it. So the
rebuild was a silent no-op for HardFlow.

**Code:**
- `hardflow_projection.py` `HardFlowNLP.update_constraint_list` (L441): on the **slsqp** backend (the one these runs
  use), rebuilds the SLSQP projector from the new list. That is the same per-replan rebuild DPCC does. On **ipopt**,
  the CasADi problem is not rebuilt and a one-time warning is printed.
- `HardFlowPolicy.update_constraint_list` (L1242) forwards to it.
- `eval_mix_uav.py` `_run_variant` (L2099): for HardFlow variants the rebuild closure returns the constraint list.
- `rollout_one` (L1679): if the policy has `update_constraint_list`, the list goes there; DPCC keeps `policy.projector = …`.

**On corridor_v2:** the slide now switches off after x = 2 for HardFlow too. **Evidence:** `hardflow_new-pdes`
y at x = 2.8 went from −0.57 (bug) to −0.35 (fixed), and `proj_ms` from 36 to 10.

**Affects other results. This is the one non-opt-in change:**

| geometry | effect of the bug | status |
| :-- | :-- | :-- |
| `s_curve`, `s_curve_hg` | both segments' walls enforced at once (seg1 y ∈ [−1.25, −0.35] **and** seg2 y ∈ [0.35, 1.25]), infeasible everywhere | **HardFlow rows in T4, T5 and the campaign closure are invalid**; warnings added to those files |
| `corridor_hg`, `corridor_ball*`, `corridor_gate*` | walls with window [−2, 2] stayed on at \|x\| > 2 | harmless for routes; the slides kept pushing past the exit |
| pillars, every DPCC arm | none | unaffected |

### 3.2 Fix 2b — `-tightened` + `-pdes` (no new code)

`-tightened` is the existing DPCC variant token. `setup_dpcc_projector` L1272–1281 adds `enlarge_constraints`
(0.025 m, yaml) to the planning margin. That makes it **0.335 m** for the planner, while the scorer still uses **0.31 m**.
It is the margin the Gen11 study names as DPCC's native answer to plant lag.

**On corridor_v2:** the setpoint rides 25 mm inside the limit instead of on it, which is more than every `-pdes`
penetration measured in fix 1. **Violations are still scored against the physical 0.31 m, so the margin is not
subtracted from the yardstick.**

### 3.3 Fix 2c — plumbing

- `eval_mix_uav.py` L693: the `UAV_MIX_VARIANTS` allow-list strips `-pdes`, `-tightened`, `-bounds_free`,
  `-geo_free`, `-model_free` to the core name before checking it. Without this, `dpcc-t-bounds_free-tightened-pdes`
  was rejected because only its core `dpcc-t` is listed. Toggles on `diffuser` are rejected, since it has no
  projector. Unknown cores are still rejected.

### 3.4 Evidence (fix 2 run, same bench)

| arm | collision-free | violations L / C | closest slide clearance L / C | y at x = 2.0 | success L / C | S&C |
| :-- | :-: | --: | --: | --: | :-- | :-: |
| diffuser | 0/2 | 39 / 58 | −290 / −418 mm | −0.07 / +0.07 | 1 / 1 | 0.00 |
| dpcc-t-bounds_free-pdes | 0/2 | 37 / 49 | −22 / −21 mm | −0.35 / −0.35 | 1 / 0 | 0.00 |
| **dpcc-t-bounds_free-tightened-pdes** | **2/2** | **0 / 0** | **+3.1 / +5.5 mm** | −0.37 / −0.37 | **1** / 0 | **0.50** |
| hardflow_new-pdes (2a fixed) | 0/2 | 36 / 44 | −28 / −26 mm | −0.34 / −0.34 | 1 / 0 | 0.00 |
| **hardflow_new-tightened-pdes** | 0/2 | **1 / 1** | **−1.5 / −1.5 mm** | −0.37 / −0.37 | 1 / 0 | 0.00 |

No divergence aborts and no circuit-breaker trips in any arm.

### 3.5 What is left

- **The exit.** Every arm's closest approach is at x ≈ 1.98–2.00, where the slide ends. With `-pdes` the gate reads
  the setpoint, which reaches x = 2.0 about 0.4 m before the drone, so the setpoint is released while the drone is
  still on the last part of the slide. DPCC keeps +3–5 mm; HardFlow loses 1.5 mm on one step per route.
- **Goal on C and R (geometry, not the fixes).** The slide pushes the drone to −0.37 and the corridor model never
  steers back sideways. Strict success needs |y − route| ≤ 0.30 at x = 2.8, so C (needs ≥ −0.30) and R (needs
  ≥ −0.18) **cannot succeed on this slide however well the projector works**. L succeeds with goal distance
  0.30, right on the radius.

---

## 4. Consequence for the full run on this exact bench

It is valid to run the full evaluation on `corridor_v2_slide` with `-tightened-pdes` as it stands. Expect:
- collision-free near 1.0 for DPCC, and ≈ 1.0 minus the exit millimetres for HardFlow;
- **strict success and S&C capped at ≈ 1/3 for every projected arm**, because only L can reach its goal;
- the diffuser at success 1.0 / S&C 0.0.

The table would read "projection makes the drone collision-free but costs success". That cost comes from the
slide's depth, not from the projector.

A geometry with a shallower exit (drone pushed to ≈ −0.12) and a flat run-out past x = 2 would remove both the
success cap and the exit millimetres. It is a new geo entry and needs one confirming injection.

## 5. How to switch any of this off

- `-pdes`, `-tightened`: omit the token from the variant name. Default variants are byte-identical to before fix 1.
- HardFlow `x_active` fix: revert `07efb114` (not recommended, since it restores the infeasible s_curve behaviour).
- None of the fixes are gated by environment variables; nothing on disk has to be restored.

## 6. Verification status

- Only compile-checked locally (Docker has no torch, mujoco or casadi). The toggle's box layout, selection parsing
  and allow-list resolution were checked on the extracted logic.
- Behaviour is verified by the cluster injections above. The child logs were not downloaded, so the scene and eps
  setting are confirmed indirectly: the drone flew at y ≈ −0.37 to −0.63 inside x ≤ 2 contact-free, which is
  impossible in the original XML, and paths moved 0.3–0.7 m, which is impossible at FRAC 1e-3.
- n = 2 trials, one seed, K = 3: injections, not a DA.
