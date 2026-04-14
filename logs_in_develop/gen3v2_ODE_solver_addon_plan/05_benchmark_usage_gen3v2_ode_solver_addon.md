# 05 Benchmark Usage: Gen3v2 ODE Solver Addon

Date: 2026-04-14
Status: Ready to use
Depends on: 03_coding_execution_record_gen3v2_ode_solver_addon.md, 04_validation_record_gen3v2_ode_solver_addon.md

---

## 1) Purpose

Use this standalone benchmark before full FM-PCC evaluation to compare ODE options quickly and consistently.

Scope boundary:
1. This benchmark focuses on ODE behavior in FM vector-field sampling.
2. Projection behavior is intentionally out-of-scope and disabled in benchmark sampling.

Benchmark script:
- `FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py`

What it tests:
1. VF-only ODE-in-VF sampling benchmark by default (scope-safe mode).
2. Legacy and torchdiffeq solver variants under identical trial settings.
3. Per-trial and aggregated metrics for speed and trajectory-quality trade-offs in VF-only mode.
4. Optional env-policy mode for broader end-to-end behavior when explicitly requested.

---

## 2) Quick Start

Run from project root (`FM-PCC/`):

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py
```

Default sweep:
1. `legacy_euler:euler`
2. `torchdiffeq:dopri5`
3. `torchdiffeq:rk4`
4. `torchdiffeq:midpoint`

Default seed behavior:
1. If `--seed` is not provided, benchmark uses the first seed in `config/projection_eval.yaml`.
2. This removes hardcoded `seed=5` behavior from the benchmark script.

---

## 3) Common Commands

### 3.0 VF-only ODE benchmark (recommended for solver comparison)

This mode targets ODE-in-VF sampling directly and avoids full env-step confounds.

Default mode note:
1. `vf_only` is now the default benchmark mode.

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --benchmark-mode vf_only \
  --n-trials 50 \
  --vf-batch-size 16 \
  --flow-steps 10 \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint \
  --plot
```

### 3.1 More trials for a stronger comparison

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --n-trials 50 --max-episode-length 200
```

For full env+policy benchmark explicitly:

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --benchmark-mode env_policy --n-trials 50 --max-episode-length 200
```

Use `env_policy` only when you explicitly want broader end-to-end behavior; it is outside strict VF-only scope.

### 3.2 Custom solver list

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4
```

### 3.3 Per-solver tolerances / step size

Format:
- `backend:method:rtol:atol:step_size`
- Use `none` for optional values.

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --solver-spec torchdiffeq:dopri5:1e-5:1e-6:none,torchdiffeq:rk4:none:none:0.1
```

### 3.4 Enable comparison plots

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --plot
```

### 3.5 Select environment layout variant

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --halfspace-variant both-hard
```

### 3.6 One-Time Full Test (recommended before full eval)

This runs a stronger single benchmark pass with plots and a fixed output directory.

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --n-trials 50 \
  --max-episode-length 200 \
  --halfspace-variant both-hard \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint \
  --plot \
  --output-dir FM_v3_ode_selectable_test/benchmark_outputs/fulltest_auto-seed_both-hard
```

If you want a specific checkpoint seed:

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --seed 7 --plot
```

After this one-time full test, verify:
1. `summary.csv` exists and contains one row per solver option.
2. `summary.json` exists and matches the CSV values.
3. Plot files exist (`benchmark_summary_plots.png`, `benchmark_tradeoff_scatter.png`, `benchmark_inference_per_trial.png`).
4. `trials_<backend>_<method>.json` exists for each solver option.
5. `run_meta.json` confirms `benchmark_mode` and `flow_steps_v3` used for this run.

---

## 4) Output Location and Naming

Outputs are written to:
- `FM_v3_ode_selectable_test/benchmark_outputs/<timestamp>_seed<seed>_<halfspace_variant>/`

Saved files:
1. `run_meta.json` (run configuration snapshot)
2. `summary.json` (aggregate metrics per solver)
3. `summary.csv` (same summary as table)
4. `trials_<backend>_<method>.json` (per-trial records)
5. Optional plots when `--plot` is enabled:
   - `benchmark_summary_plots.png`
   - `benchmark_tradeoff_scatter.png`
   - `benchmark_inference_per_trial.png`

---

## 5) Metrics to Compare First

Primary decision metrics:
1. For `vf_only`: `avg_inference_ms`, `avg_final_goal_dist`, `avg_traj_smoothness`
2. For `env_policy`: `success_and_constraints_rate`, `avg_inference_ms`, `avg_n_violations`, `avg_total_violation`

Recommended selection rule:
1. Keep a legacy reference run (`legacy_euler:euler`).
2. Prefer a torchdiffeq method only if safe-success stays equal or better with acceptable runtime.
3. Reject options with unstable safety metrics even if success-only looks good.

---

## 6) Runtime Notes

1. Solver settings are applied at runtime from benchmark options (plan-time style override behavior).
2. If `torchdiffeq` backend is requested but package is unavailable, run fails fast by design.
3. `step_size` only applies to fixed-step methods in diffusion implementation.
4. This benchmark is a pre-eval filter, not a replacement for full projection-variant evaluation.
5. `benchmark_mode=vf_only` is the closest check for "ODE on VF" behavior.
