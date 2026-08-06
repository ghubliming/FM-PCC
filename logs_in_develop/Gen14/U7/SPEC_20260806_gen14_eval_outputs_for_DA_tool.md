# Gen14 (Visual-Mix-ML) eval outputs — spec for the new DA tool

**Date:** 2026-08-06 · **Epoch:** U7 · **Producer:** `mix_visual_aligning_test/eval_mix_visual_aligning.py`
**Audience:** the agent writing the replacement visual-aligning DA tool.
**Reference target:** `Data_Analysis/DA_Code_v3/` + `Slurm_Codes/sbatch/DA/run_da_batch_avoiding_combined.sh`.
**Out of scope:** the existing VA DA code — assume it is being rewritten.

**Verdict up front:** Gen14 can feed a DA tool of the `DA_Code_v3` shape, but **not
unmodified**. Four directory-layout mismatches and four missing metric keys. All are small.
Details in §4–§5.

---

## §1 — On-disk layout Gen14 emits

```
logs/aligning-d3il-visual/plans/mix_visual_aligning_{engine}/
  H8_K{K}_M{solver}_T{thr}_D{diffusion}_V{if_vision}_mpc{B}_film{v}_E{engine}/   ← "candidate"
    {seed}/
      config_snapshot_aligning-d3il-visual/        aligning-d3il-visual.py + visual_aligning_eval.yaml
      results/                                     ← `results_train_set/` when --eval-on-train
        expert_references/                         expert_rollout_<i>.{mp4,gif}
        {geo_name}/                                e.g. combined_5, combined_5-tightened
          constraint_overview.png
          {variant}/                               ← 🔴 EXTRA LEVEL vs DA_Code_v3
            {variant}.npz                          ← the single source of truth
            {variant}.png
            constraint_metrics.json
            eval_{variant}.log
            realtime_{variant}_rollout{r}.log       (one per rollout)
            diag_first_replan.txt
            diagnostics/
              rollout_{r}_stats.json
              rollout_{r}_report.png
              rollout_{r}_mpc_foresight.svg
              rollout_{r}.gif                       (only when --record != none)
```

`{engine}` ∈ `diffusion | fm | mf | af`. Each engine is a **separate candidate tree** — a
four-arm comparison is a four-candidate batch run, exactly the shape
`multi_candidate_discovery.discover_candidates()` already expects at the `plans/` level.

**Variant set (19 + HardFlow), read from `config/visual_aligning_eval.yaml`:**
`diffuser`, `dpcc-r`, `dpcc-c`, `dpcc-t`, `dpcc-c-dt{0p25,0p5,2p0,4p0}`, `gradient`,
`post_processing`, `model_free`, `geo_free`, `bounds_free`, `geo_free-bounds_free`,
`geo_free-model_free`, `model_free-bounds_free`, `hardflow_new-{r,c,t}`.

**Geometry:** `{geo_name}` and its auto-generated `{geo_name}-tightened` twin. Note this is a
**directory level**, not a variant-name suffix — the opposite of the state-only convention
where tightening is baked into the filename (`dpcc-c-tightened.npz`).

---

## §2 — The `.npz` schema (49 keys)

One file per (candidate, seed, geo, variant). `N` = `n_contexts` (30 in the U7 runs).

**Scalars:** `success_rate`, `entropy`, `elapsed_seconds`, `seed`, `complete`, `args` (object —
the full resolved config).

**Per-rollout, shape (N,):**

| group | keys |
|---|---|
| outcome | `n_success`, `success_strict`, `success_relaxed`, `mean_distance`, `mean_dist_per_rollout`, `n_steps`, `avg_time`, `mode_encoding` (N,1) |
| tracking | `max_phys_error_per_rollout`, `outcome_max_physical_tracking_error` |
| context | `context_box_init_xy` (N,2), `context_target_xy` (N,2), `context_box_angle_deg`, `context_target_angle_deg`, `context_init_xy_dist` |
| contact | `contact_first_step`, `contact_last_step`, `contact_first_pos_xy` (N,2), `contact_last_pos_xy` (N,2) |
| constraint — executed | `constraint_exec_n_violated_steps`, `_sat_rate`, `_zero_violation`, `_bounds_viol_count`, `_halfspace_viol_count`, `_obstacle_viol_count`, `_max_bounds_viol_m`, `_max_halfspace_viol_m`, `_max_obstacle_penetration_m`, `_margin_mean_m`, `_first_violation_step`, `_longest_safe_streak`, `_dyn_err_mean`, `_dyn_err_max` |
| constraint — planned | `constraint_plan_post_viol_rate_mean`, `_max`, `_n_replan_steps` |
| projector health | `projection_cb_tripped`, `projection_cb_skipped_steps` |

**Raw traces (object arrays, large):** `obs_all` (N,T,6), `act_all` (N,T,3),
`sampled_trajectories_all` (N,T,B,H,3), `selected_idx_all` (N,T),
`physical_tracking_errors` (N,T). `sampled_trajectories_all` is ~90% of the file size —
the DA tool should `mmap`/lazy-load and only touch it when it needs candidate-level analysis.

**Sidecars.** `constraint_metrics.json` carries `{variant, geo_name, seed, n_rollouts, exec,
plan, per_rollout}` with mean/std already reduced — cheaper than opening the npz for
aggregate-only work. `diagnostics/rollout_{r}_stats.json` holds the per-rollout record
including `context.box_obstacle_conflict` (see §5.3).

---

## §3 — What `DA_Code_v3` consumes today

Discovery (`data_loader.py:58,67`) hard-codes:

```
{candidate}/{seed}/results/halfspace_{hs_variant}/{variant}.npz
```

with `hs_variant ∈ {top-right-hard, top-left-hard, both-hard}` (`config.py:62`). The metric
list it reports on (`config.py:69`) is seven names, and the reference state-only npz contains
exactly those seven plus `args`/`obs_all`/`act_all`/`sampled_trajectories_all`:

```
n_success  n_success_and_constraints  n_steps  n_violations
total_violations  avg_time  collision_free_completed
```

`_load_result_file()` (`data_loader.py:145`) is otherwise **generic** — it ingests every key in
the npz and auto-derives `{key}`, `{key}_array`, `{key}_std`. So extra Gen14 keys cost nothing;
only *missing* keys and the *path shape* matter.

---

## §4 — Pre-check: layout

| # | `DA_Code_v3` expects | Gen14 emits | fix |
|---|---|---|---|
| L1 | `…/{seed}/results/` | `results_train_set/` under `--eval-on-train` | glob both |
| L2 | `halfspace_{variant}/` | `{geo_name}/` — `combined_5`, `combined_5-tightened` | replace the `halfspace_` prefix rule with a directory scan; treat geo as a free-text dimension |
| L3 | `{variant}.npz` directly in the geo dir | `{variant}/{variant}.npz` — one extra level | one extra `join` |
| L4 | variant name carries `-tightened` | tightening is the **geo directory**, and `--eval-on-train` appends `_train_set` to the variant name | strip the `_train_set` suffix; key results on `(geo, variant)` not on variant alone |

None of these are blockers — L2 is the only one that changes the tool's data model, and it
changes it in the right direction (geo becomes a first-class axis instead of a name prefix).

**Batch entry point.** `Slurm_Codes/sbatch/DA/run_da_batch_avoiding_combined.sh` is directly
reusable: swap `--parent-path` to `logs/aligning-d3il-visual/plans/mix_visual_aligning_diffusion,…_fm,…_mf,…_af`.
The conda/env/`MPLBACKEND=agg`/`PYTHONPATH` preamble needs no change.

---

## §5 — Pre-check: metrics

### 5.1 Present, same name, same semantics

`n_success`, `n_steps`, `avg_time`, plus `args`, `obs_all`, `act_all`,
`sampled_trajectories_all`. **3 of 7 core metrics.**

### 5.2 Missing — but derivable in the DA tool, no eval change needed

| `DA_Code_v3` metric | Gen14 equivalent |
|---|---|
| `collision_free_completed` | `constraint_exec_zero_violation` (already 0/1 per rollout) |
| `n_violations` | `constraint_exec_n_violated_steps` |
| `n_success_and_constraints` | `n_success * constraint_exec_zero_violation` |
| `total_violations` | **🔴 no direct equivalent** — Gen14 records per-family *maxima* (`max_bounds_viol_m`, `max_halfspace_viol_m`, `max_obstacle_penetration_m`) and `margin_mean_m`, never a cumulative sum over steps |

`total_violations` is the one real gap. Two options: (a) have the DA tool report the maxima
instead and drop the cumulative metric, or (b) add a running sum to the constraint accumulator
in `eval_mix_visual_aligning.py` — an eval-side change, so it would only apply to future runs
and every existing `temp/0508` result would stay `NaN`. **Recommend (a)** unless the cumulative
number is specifically wanted for the thesis.

### 5.3 Two Gen14-specific things the tool must handle

**Frozen rollouts.** Under a tightened geometry the D1 box-obstacle guard can declare a context
unusable and hold position for the whole episode without ever calling the model
(`eval_mix_visual_aligning.py:1427`, `:2072-2079`). Those rollouts land in the arrays with
`sat_rate = 1.0` and `zero_violation = 1`, which inflates any constraint aggregate. They are
identifiable only via `diagnostics/rollout_{r}_stats.json → context.box_obstacle_conflict`.
**The DA tool should read that flag and expose a masked/unmasked toggle.** Better still, ask for
a `context_box_obstacle_conflict` (N,) array to be added to the npz so the JSONs need not be
opened — currently it is not exported.

**Projector circuit breaker.** `projection_cb_tripped` / `projection_cb_skipped_steps` mark
rollouts where the projector stopped projecting under load. A variant with a nonzero trip count
did not run the policy it claims to. Surface it as a data-quality column, not a metric.

### 5.4 Aggregation

Gen14 writes **no `all_seeds/` aggregate directory** (the state-only pipeline does). Multi-seed
reduction is entirely the DA tool's job. Note that `seeds` and `n_contexts` live in
`config/visual_aligning_eval.yaml`, which is **shared with Gen6V4 and Gen7** — the Gen14 sbatch
overrides seeds via `--seeds`, so the authoritative record of what actually ran is
`{seed}/config_snapshot_aligning-d3il-visual/`, not the repo copy of the yaml.

---

## §6 — Summary for the implementing agent

**Can Gen14 feed a `DA_Code_v3`-shaped tool?** Yes.

Work required, all on the DA side, none in the eval:

1. Rewrite discovery for `{seed}/results[_train_set]/{geo}/{variant}/{variant}.npz`; make geo a
   first-class axis and strip the `_train_set` suffix.
2. Derive `collision_free_completed`, `n_violations`, `n_success_and_constraints` from the
   `constraint_exec_*` keys.
3. Drop `total_violations` or redefine it against the per-family maxima.
4. Mask D1-frozen rollouts (read `rollout_{r}_stats.json`) and expose the circuit-breaker
   counters as data quality.
5. Lazy-load `sampled_trajectories_all`.

**One eval-side change worth requesting** (small, additive, improves every future run):
export `context_box_obstacle_conflict` as an (N,) array in the npz so #4 needs no JSON parsing.
