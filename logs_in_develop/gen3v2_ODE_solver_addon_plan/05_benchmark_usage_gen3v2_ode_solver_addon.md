# 05 Benchmark Usage: Gen3v2 ODE Solver Addon

Date: 2026-04-14
Status: Ready to use
Depends on: 03_coding_execution_record_gen3v2_ode_solver_addon.md, 04_validation_record_gen3v2_ode_solver_addon.md

---

## 1) Purpose

Use this standalone benchmark to test only FM vector-field (VF) ODE integration behavior.

Scope boundary:
1. Only VF ODE integration is benchmarked.
2. No projection benchmarking.
3. No env-policy evaluation mode.

Benchmark script:
1. `FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py`

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
2. No hardcoded `seed=5`.

---

## 3) Common Commands

### 3.1 One-Time Full VF ODE Test (recommended)

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --n-trials 50 \
  --vf-batch-size 16 \
  --flow-steps 10 \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint \
  --plot \
  --output-dir FM_v3_ode_selectable_test/benchmark_outputs/fulltest_vf_only
```

### 3.2 More trials

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --n-trials 100 --vf-batch-size 16 --flow-steps 10
```

### 3.3 Custom solver list

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4
```

### 3.4 Per-solver tolerances / step size

Format:
- `backend:method:rtol:atol:step_size`
- Use `none` for optional values.

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --solver-spec torchdiffeq:dopri5:1e-5:1e-6:none,torchdiffeq:rk4:none:none:0.1
```

### 3.5 Enable comparison plots

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --plot
```

If you want a specific checkpoint seed:

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --seed 7 --plot
```

After one-time full test, verify:
1. `summary.csv` exists and contains one row per solver option.
2. `summary.json` exists and matches the CSV values.
3. Plot files exist (`benchmark_summary_plots.png`, `benchmark_tradeoff_scatter.png`, `benchmark_inference_per_trial.png`).
4. `trials_<backend>_<method>.json` exists for each solver option.
5. `run_meta.json` confirms `flow_steps_v3` used for this run.

---

## 4) Output Location and Naming

Outputs are written to:
- `FM_v3_ode_selectable_test/benchmark_outputs/<timestamp>_seed<seed>_vf_only/`

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
1. `avg_inference_ms`
2. `avg_final_goal_dist`
3. `avg_traj_smoothness`
4. `avg_batch_final_xy_std`

Recommended selection rule:
1. Keep a legacy reference run (`legacy_euler:euler`).
2. Prefer a torchdiffeq method if it reduces `avg_final_goal_dist` with acceptable `avg_inference_ms`.
3. Use `avg_traj_smoothness` and `avg_batch_final_xy_std` as stability tie-breakers.

---

## 6) Runtime Notes

1. Solver settings are applied at runtime from benchmark options (plan-time style override behavior).
2. If `torchdiffeq` backend is requested but package is unavailable, run fails fast by design.
3. `step_size` only applies to fixed-step methods in diffusion implementation.
4. Benchmark conditions come from real dataset observations; no projection or env-policy branch is used.
