# U3 PLAN — Restore legacy eval artifacts for UAV FM (npz / json / log / 2D overview / GIF)

## Objective
Upgrade `FM_v3_uav_test/eval_fm_uav.py` so each eval emits the **same artifact set
as the legacy FMv3-ODE / visual evals**, adapted for the 3-D UAV: per-rollout
`npz` + `json` + `log`, a **2-D path-overview PNG (primary visual, replaces GIF)**,
and an optional **GIF**. PCC/MPC-constraint visuals are **placeholders** (PCC lands
Epoch 7). Today's U2 eval only writes a single `results.json` summary — U3 brings
back the rich, analysis-script-compatible outputs.

## Guiding principle — build only the `diffuser` (FM-only) variant; placeholder the rest
The legacy eval is multi-**variant** (`diffuser`, `dpcc-c`, `halfspace`, …) with
constraint metrics per variant. **U3 fully implements only the `diffuser` / `fm_only`
baseline** — the pure-FM path with no constraint projection. Every constraint/PCC
variant and its metrics (`n_success_and_constraints`, `n_violations`,
`total_violations`, the foresight SVG, the per-variant loops) are **scaffolded as
typed placeholders now** (zero-filled npz keys, stub files, commented hooks) so that
Epoch 7's DPCC work is a clean drop-in with **no schema or layout change**. Prefer a
placeholder over omission wherever it makes E7 cheaper — that convenience is the
explicit design goal here, not full coverage in E6.

## Key insight — reuse, don't reinvent
The UAV expert-collect module **already ships the renderers** we need; they know the
scenes, overhead camera, and pillar/obstacle geometry:
- `uav_expert_data_collect/generate_overview_plots.py` — top-down XY plot,
  `_draw_obstacles(ax, obstacles)` (pillar circles + 0.31 m rotor safety ring),
  `_homotopy_color`. → reuse for the **2-D overview**.
- `uav_expert_data_collect/generate_trajectory_gifs.py` — `_render_overhead(model,
  data, renderer)` via `mujoco.Renderer` + overhead cam, `imageio.mimsave`. → reuse
  for the **GIF**.

U3 = refactor those two into importable helpers and feed them **FM-rollout** data
(currently they read expert `.pkl` episodes), then match the legacy `npz` schema.

## Legacy schema to match (source of truth)
From `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` + the Fix-9
upgrade (`logs_in_develop/gen5_.../fix_9_eval_outptut_upgrade/`):

| Artifact | Legacy | U3 (UAV) |
|----------|--------|----------|
| `{variant}.npz` | `n_success`, `n_steps`, `obs_all`, `act_all`, `sampled_trajectories_all`, `args` | same keys; `variant` = projection (`fm_only`) |
| `eval_{variant}.log` | captured stdout | same |
| `{variant}.png` | 6-panel top-down (x,y) traj + sampled plans + constraints | top-down (x,y) + FM 8-step plans + pillars |
| `all.png` | aggregate across trials/seeds | aggregate across trials |
| `diagnostics/rollout_<r>.gif` | MuJoCo render | overhead GIF (opt-in) |
| `diagnostics/rollout_<r>_stats.json` | per-rollout metrics | per-rollout dict (already built in `rollout_one`) |
| `*_mpc_foresight.{svg,png}` | MPC/PCC plan viz | **PLACEHOLDER** (Epoch 7 PCC) |

`obs_all` / `act_all` / `sampled_trajectories_all` are `np.array(..., dtype=object)`
(ragged per-trial) so existing analysis scripts read them unmodified.

## Integration points in `eval_fm_uav.py`
- `rollout_one()` already steps MuJoCo with `model`/`data` and tracks `p`, `v`,
  `p_des`, `action` per step → **add buffers** `obs_buffer` (=[p_des|p|v]),
  `action_buffer`, and `plan_buffer` (the FM's H=8 horizon prediction, the
  `sampled_trajectories` analog), and **optionally append a rendered frame**.
- `eval_scene()` already aggregates rollouts → **add the npz/log/png writes** beside
  the existing `results.json` (keep `results.json` for back-compat).

## Output layout (under the existing per-model eval dir)
```
logs/UAV_FM/uav-<scene>/flow_matching_v3_uav/.../<seed>/eval/<projection>/
  results.json                      # keep (U2 summary)
  fm_only.npz                       # NEW legacy-schema archive
  eval_fm_only.log                  # NEW captured stdout
  fm_only.png                       # NEW 2-D overview (primary visual)
  all.png                           # NEW aggregate
  diagnostics/
    rollout_<r>_stats.json          # NEW per-rollout
    rollout_<r>.gif                 # NEW (opt-in --record gif|all)
    rollout_<r>_mpc_foresight.svg   # NEW PLACEHOLDER (empty/stub, Epoch 7)
```

## Phased steps
- **P1 — npz + log + per-rollout json (no rendering).** Add buffers to
  `rollout_one`, write `fm_only.npz` (legacy keys) + `eval_fm_only.log` +
  `diagnostics/rollout_<r>_stats.json`. Lowest-risk, unblocks analysis scripts.
- **P2 — 2-D overview PNG.** Refactor `generate_overview_plots._draw_obstacles` +
  XY plotting into an importable helper; draw real path (black), FM 8-step plans
  (blue), start/goal markers, pillars. Add a **side view (x,z or z-vs-t)** panel —
  altitude is the headline UAV failure mode (U2: drone never left z≈0.08), top-down
  alone hides it. Write `fm_only.png` + `all.png`.
- **P3 — GIF (opt-in).** Reuse `generate_trajectory_gifs._render_overhead`; gate
  behind a `--record {none,gif,all}` arg (default `none`) so default runs stay fast.
- **P4 — MPC/PCC placeholder.** Emit a stub `rollout_<r>_mpc_foresight.svg` with a
  `TODO: PCC projection (Epoch 7)` marker, and reserve npz keys
  (`n_success_and_constraints`, `total_violations`) zero-filled so the Epoch-7 DPCC
  drop-in needs no schema change.

## PCC / constraint hooks (placeholders only — Epoch 7)
- npz: include `n_success_and_constraints`, `n_violations`, `total_violations`
  zero-filled now.
- Plot: leave a commented `# plot_pcc_constraints(...)` hook beside the pillar
  drawing, mirroring legacy `utils.plot_environment_constraints`.
- SVG: stub file only. No projection math in Epoch 6.

## Success criteria
- [ ] `fm_only.npz` loads and contains `obs_all`, `act_all`, `sampled_trajectories_all`
      with correct per-trial shapes; existing analysis scripts read it unmodified.
- [ ] `fm_only.png` shows the real path + FM plans + pillars, **plus** an altitude
      view that makes the "never took off" failure visible at a glance.
- [ ] `eval_fm_only.log` + `rollout_<r>_stats.json` written every run.
- [ ] `--record gif` produces overhead GIFs; default `none` adds ~0 overhead.
- [ ] PCC placeholders present (stub SVG + zeroed npz keys) with no Epoch-6 logic.

## Out of scope (Epoch 7)
Real PCC/DPCC constraint projection + foresight rendering. U3 only restores the
artifact *plumbing* and leaves typed placeholders.

## Risks / notes
- Headless render needs `MUJOCO_GL=egl` (eval already sets it + the GPU-leak guard).
- `dtype=object` ragged arrays must match legacy exactly or analysis scripts break —
  validate against an old `.npz` before closing U3.
- This is artifact plumbing; it does **not** fix the U2 policy failure (0% success).
  Keep that as a separate modeling unit — U3 just makes that failure *visible*.
