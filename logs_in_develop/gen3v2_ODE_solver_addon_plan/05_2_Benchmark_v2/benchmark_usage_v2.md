# Benchmark ODE Solvers v2 — Usage Guide

Date: 2026-04-15
Script: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v2.py`

---

## 1) Purpose
V2 addresses the architectural flaws discovered in the V1 audit. It acts as a highly accurate proxy for your real evaluation pipeline.

### Main Improvements:
1. **Real Model Loading & True Pipeline**: `--vf-mode flow_matcher` loads exact serialized U-Net checkpoints directly from `/logs/`. It strictly uses `GaussianDiffusion.p_sample_loop()` for 100% pure representation of the production execution overhead.
2. **Synthetic V1 Fallbacks**: Supports V1's synthetic `--vf-mode neural` (a 1.5M PyTorch MLP) alongside the analytic `--vf-mode spiral` for granular isolated baseline testing.
3. **No Grad Safety**: Correctly protects tracking engines with `@torch.no_grad()` to ensure PyTorch doesn't construct invisible graphs that unnecessarily bottleneck `torchdiffeq` speeds.

| In Scope | Out of Scope |
|---|---|
| ODE solver speed comparison matching actual eval | Env rollout metrics |
| True 1D Temporal U-Net compute benchmarking | Loading non-diffusion model weights |
| Chunked integration penalty isolation | Dataset/RL trajectories visual verification | 

---

## 2) How to Run

### Example: Running True U-Net from Checkpoint
The script now mirrors `eval_flow_matching` exactly. You build the path just like you would in your config or evaluation script: using `loadbase`, `dataset`, `diffusion-loadpath`, and `diffusion-seed`.

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v2.py \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --n-trials 10 \
  --batch-size 64 \
  --horizon 128 \
  --device cuda \
  --solver-spec legacy_euler,torchdiffeq:euler,torchdiffeq:dopri5 \
  --plot
```

### Example: Running the Baseline PyTorch "Bridge Tax" Benchmark
If you just want to measure pure `numPy` vs `PyTorch` wrapper overhead WITHOUT the model (simulating a nearly instant mathematical vector field like a spiral):

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v2.py \
  --vf-mode spiral \
  --n-trials 50 \
  --batch-size 128 \
  --state-dim 8 \
  --solver-spec legacy_euler,torchdiffeq:euler
```

---

## 3) CLI Arguments Reference

| Argument | Description |
|---|---|
| `--vf-mode` | `flow_matcher` (Real U-Net) or `spiral` (Analytic Math). |
| `--loadbase` | Usually `logs`. The root directory where model checkpoints are saved. |
| `--dataset` | E.g., `avoiding-d3il`. The name of the environment/dataset inside the logbase. |
| `--diffusion-loadpath` | **Required if `flow_matcher`**. Experiment string, e.g. `flow_matching/H8_K20_...` |
| `--diffusion-seed` | Generally `0` or `1`. The run seed identifier sub-folder. |
| `--horizon` | Usually `128` or `256`. The sequence length for the U-Net inputs. |
| `--t0`, `--t1` | `0.0` and `1.0`. The overall timeframe for integration bounds. |
| `--plot` | Generates 6 precise timing characteristic bar charts in `benchmark_outputs_v2`. |

---

## 4) Troubleshooting & Understanding Chunked Overheads

If you test `flow_matcher` mode with `torchdiffeq:dopri5`, you will likely notice the inference time skyrockets. 
This is because `p_sample_loop` breaks integration recursively into chunks to support constraints later on:
1. The solver must perform initial error derivations from scratch on the very first sub-step.
2. Breaking a $1.0$-length journey into 20 small $0.05$ increments eliminates the step-size growth advantage Dopri5 is known for.

**Takeaway:** If you plan on interrupting the solver 20 times to project safe-states against constraints, you should almost certainly stick with `legacy_euler` or a lightweight fixed step like `torchdiffeq:euler` or `rk4`. Adaptive solvers simply don't have enough geometric leeway to accelerate inside tiny evaluation chunks.
