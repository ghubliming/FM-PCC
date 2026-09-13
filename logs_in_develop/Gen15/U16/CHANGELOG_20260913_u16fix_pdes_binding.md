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
| — | — | not yet submitted |
