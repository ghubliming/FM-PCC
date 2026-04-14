# Benchmark ODE Solvers v3 — Usage Guide

> Based on local WSL, but the Colab version is similar (just adjust paths and env setup).

Date: 2026-04-14
Script: `FM_v3_ode_selectable_test/benchmark_ode_solvers.py`

---

## 1) What This Does

Compares wall-clock inference time of different ODE solvers on a **synthetic vector field** (stable spiral + nonlinear damping). Every solver integrates the *exact same dynamics* so the only variable is the solver itself.

| In Scope | Out of Scope |
|---|---|
| ODE solver speed comparison | FM model loading |
| Identical synthetic VF for all methods | Dataset loading |
| Timing statistics + optional plots | Env rollout metrics |
| | Projection metrics |

---

## 2) Run

From the project root (`dpcc/`):

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --seed 0 \
  --n-trials 50 \
  --batch-size 128 \
  --state-dim 8 \
  --t0 0.0 \
  --t1 1.0 \
  --steps 20 \
  --solver-spec legacy_euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint \
  --plot \
  --output-dir FM_v3_ode_selectable_test/benchmark_outputs/synthetic_vf_demo
```

### Jupyter / WSL cell example

```bash
%%bash
DPCC="$HOME/DPCC/dpcc"
D3IL_ROOT="$DPCC/d3il"
GYM_AV="$D3IL_ROOT/environments/d3il/envs/gym_avoiding_env"

cd $DPCC

export MPLBACKEND=agg
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$DPCC:$D3IL_ROOT:$GYM_AV"

~/miniconda3/envs/dpcc/bin/python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --n-trials 50 \
  --batch-size 16 \
  --steps 10 \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint \
  --plot \
  --output-dir FM_v3_ode_selectable_test/benchmark_outputs/fulltest_vf_only
```

---

## 3) CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--seed` | int | `0` | Random seed for reproducibility |
| `--n-trials` | int | `100` | Number of timing repetitions per solver |
| `--batch-size` | int | `128` | Batch of random initial states per trial |
| `--state-dim` | int | `8` | State dimension (must be **even** ≥ 2) |
| `--t0` | float | `0.0` | Integration start time |
| `--t1` | float | `1.0` | Integration end time |
| `--steps` | int | `20` | Number of steps for fixed-step methods |
| `--rtol` | float | `1e-5` | Relative tolerance (adaptive methods) |
| `--atol` | float | `1e-6` | Absolute tolerance (adaptive methods) |
| `--solver-spec` | str | see below | Comma-separated solver list |
| `--output-dir` | str | auto-timestamped | Output directory |
| `--plot` | flag | off | Generate bar-chart PNGs |

### Solver spec format

Comma-separated entries. Each entry is one of:

| Format | Example | Notes |
|---|---|---|
| `legacy_euler` | `legacy_euler` | Numpy forward-Euler |
| `legacy_euler:euler` | `legacy_euler:euler` | Same thing, also accepted |
| `torchdiffeq:<method>` | `torchdiffeq:dopri5` | Any torchdiffeq method |

**Fixed-step methods** (use `--steps`): `euler`, `midpoint`, `rk4`, `heun2`, `heun3`, `explicit_adams`, `implicit_adams`, `fixed_adams`

**Adaptive methods** (use `--rtol`/`--atol`): `dopri5`, `dopri8`, `bosh3`, `adaptive_heun`

Default spec:
```
legacy_euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint
```

---

## 4) Output Files

All outputs land in `--output-dir`:

| File | Content |
|---|---|
| `run_meta.json` | Full run configuration (seed, n_trials, solvers, etc.) |
| `summary.json` | Per-solver aggregated stats |
| `summary.csv` | Same data in CSV format |
| `trials_<backend>_<method>.json` | Per-trial raw timing for each solver |
| `plot_<metric>.png` | Individual bar chart per metric (if `--plot`) |
| `plot_overview.png` | Combined 2×3 chart of all metrics (if `--plot`) |

---

## 5) Metrics

From `summary.json` / `summary.csv`, each solver row contains:

| Metric | Meaning |
|---|---|
| `avg_ms` | Mean inference time in milliseconds |
| `std_ms` | Standard deviation |
| `p50_ms` | Median (50th percentile) |
| `p95_ms` | 95th percentile |
| `min_ms` | Fastest trial |
| `max_ms` | Slowest trial |

### How to compare

1. Use `legacy_euler` as the **baseline**.
2. Prefer solvers with lower `avg_ms` **and** stable `p95_ms`.
3. Check `std_ms` — high variance means unreliable timing.

---

## 6) Synthetic Vector Field

The VF is a stable spiral with nonlinear damping, applied pair-wise across the state dimensions:

$$
\dot{u}_k = -(\alpha + \beta r_k^2)\, u_k - \omega\, v_k
$$

$$
\dot{v}_k = \omega\, u_k - (\alpha + \beta r_k^2)\, v_k
$$

where $r_k^2 = u_k^2 + v_k^2$ and the default parameters are $\alpha=0.35$, $\omega=1.25$, $\beta=0.12$.

This is deterministic, smooth, and identical for every solver — making timing comparisons fair.

---

## 7) Code Structure

```
benchmark_ode_solvers_v3.py
│
├── spiral_vf()              # Synthetic vector field (numpy)
├── euler_integrate()         # Forward-Euler integrator (numpy)
├── torchdiffeq_integrate()   # torchdiffeq wrapper
├── parse_solvers()           # CLI solver-spec parser
├── compute_stats()           # Timing statistics
├── make_plots()              # matplotlib bar charts
└── main()                    # CLI entry point
```
