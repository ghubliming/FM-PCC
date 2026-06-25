# Epoch 7 CHANGELOG — full PCC/MPC bone restored on the UAV eval (single pass)

**Date:** 2026-06-25. Implements `PLAN.md` in one pass. Dynamics constraint is **real**;
bounds/halfspace/obstacle are **empty placeholders** (wired, not run). Copied the bone from
`fm_visual_aligning_test/eval_fm_visual_aligning.py` rather than inventing, for debuggability.

## What changed

### `config/uav_eval.yaml` — NEW, the PCC eval config (mirrors `config/visual_aligning_eval.yaml`)
Loaded by the eval via `yaml.safe_load` (same pattern as the FMv3ODE/visual evals — those read
`config/projection_eval.yaml` / `config/visual_aligning_eval.yaml`). Contains
`projection_variants=['diffuser','dpcc-r','dpcc-c','dpcc-t']`, `constraint_types=['dynamics']`,
`batch_size=4`, `dt=1.0`, `diffusion_timestep_threshold`, `enlarge_constraints`, and **empty
placeholders** `workspace_bounds=null`, `halfspace_constraints=[]`, `obstacle_constraints=[]`.
`config/uav.py` only carries a NOTE pointing here (no PCC fields).

**`dt=1.0`** (corrected): the action IS the position delta `Δp_des`, so the Euler dynamics
constraint is `p_des[t+1]=p_des[t]+1.0·act` — NOT `act×(1/33)`. (Same convention the
visual-aligning yaml documents.)

### `FM_v3_uav_test/eval_fm_uav.py` — the bone
- **`ProjectorNormalizer` + `setup_dpcc_projector(...)`** — copied near-verbatim from the
  visual-aligning eval, adapted to the UAV **12-D transition**
  `[act(3) | p_des(3) | p(3) | v(3)]`. The bounds/halfspace/obstacle blocks are kept verbatim
  but gated on their (empty) config keys → never built. **Dynamics is the only active
  constraint**, and it binds **`p_des`** (`('deriv',[3,0]),[4,1],[5,2]`) — NOT the actual `p`
  — because `p_des` is the exact integrator of the action while the drone's `p` lags
  (visual-aligning binds `c_pos` only because its arm tracks perfectly).
- **`build_policy` → `build_experiment`** — loads model+dataset+args once; the Policy is now
  built **per variant**.
- **`_selection_for(variant)`** — FMv3ODE-exact: `dpcc-t`→`temporal_consistency`,
  `dpcc-c`→`minimum_projection_cost`, else `random`.
- **`_run_variant(...)` + variant loop in `eval_scene`** — for each variant: build projector
  (`None` for `diffuser`) + selection + Policy, run all trials, write `plans/<variant>/`.
- **`rollout_one(..., batch_size)`** — samples a **batch** (`pcc_batch_size`, default 4); the
  policy selects one candidate's first action; `plans` stores the whole **MPC candidate fan**
  (`sampled_trajectories_all[trial]` shape `(T, batch, horizon, obs)`).
- **Constraint-aware metrics** added to rollout/summary: `success_and_constraints`,
  `collision_free`, `n_violations`, `total_violations` (+ `*_rate`/`*_mean`). Dynamics-only
  ⇒ no safety/obstacle constraints to violate ⇒ trivially clean (`collision_free=True`,
  violations 0); `success_and_constraints == success`.

### `FM_v3_uav_test/eval_artifacts.py`
- `save_npz` now writes the **real** `n_success_and_constraints` / `n_violations` /
  `total_violations` from the rollouts (were zero-filled placeholders).
- `write_eval_log` reports `success_and_constraints`, `collision_free_rate`, violations.

## Output (per scene, per variant)
```
logs/UAV_FM/uav-<scene>/plans/flow_matching_v3_uav/<exp>/<seed>/
  diffuser/   dpcc-r/   dpcc-c/   dpcc-t/        ← one folder per variant
    <variant>.npz  results.json  <variant>.png  all.png  eval_<variant>.log  diagnostics/
```
`<variant>.npz` carries the full FMv3ODE schema incl. the batch candidate fan, so
`npz_analysis/analyze_npz.py` `plan_cand_spread` / `--replot-plans` are now meaningful
(batch>1) and `dpcc-c`/`dpcc-t` should show committed (lower-spread) plans vs `diffuser`.

## Faithful-reuse / verification
- Projector + policy engine reused **as-is** from the forked `flow_matcher_v3_uav/sampling/`
  (byte-identical to FMv3ODE). No new sampling/projection logic invented.
- Verified (Docker, no torch/MuJoCo): all files `py_compile` clean; config block resolves;
  variant→projector→selection mapping correct (diffuser→None/random; dpcc-r/-c/-t→dynamics-only
  projector with the right selection); constraint-list is dynamics-only with bounds/halfspace/
  obstacle correctly skipped; artifacts flow through with constraint metrics + the
  `(T,4,8,9)` candidate fan.
- The real Projector/Policy path (SLSQP, sampling) is cluster-only (needs torch+MuJoCo).

## Scope / notes
- **Not run this epoch:** bounds/halfspace/obstacle constraints (empty placeholders);
  `constraint_types=['dynamics']`. Per-scene geometry fills these later, no eval rewrite
  needed (the gated blocks already accept them).
- `eval_fm_uav.sh`'s `--projection` arg is now vestigial (variants come from config);
  harmless via `parse_known_args`. Output is per-variant folders, not the old single `fm_only/`.
- **Open design call resolved here:** dynamics binds `p_des`, per `PLAN.md §3`.
- Risk to watch on the cluster: SLSQP projection × batch × ~hundreds of FM steps may be slow —
  profile; the `diffusion_timestep_threshold` knob is available to bound cost.
- Working-tree only — sync to the cluster to run.
