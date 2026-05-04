# Benchmark ODE Solvers v2 — Usage Guide

Date: 2026-04-15
Script: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v2.py`

---

## 1) Purpose
V2 addresses the architectural flaws discovered in the V1 audit. It acts as a highly accurate proxy for your real evaluation pipeline.

### Main Improvements:
1. **Real Model Loading & True Pipeline**: `--vf-mode flow_matcher` loads exact serialized U-Net checkpoints directly from `/logs/`. It strictly uses `GaussianDiffusion.p_sample_loop()` for 100% pure representation of the production execution overhead.
2. **Synthetic V1 Fallbacks**: Supports V1's synthetic `--vf-mode neural` (a 1.5M PyTorch MLP) alongside the analytic `--vf-mode spiral` for granular isolated baseline testing.
3. **Rigorous GPU Timing**: Implements explicit `torch.cuda.synchronize()` points to ensure CPU timers don't stop before the GPU finish its work (fixing a major source of bias in raw PyTorch benchmarks).
4. **Bridge Tax Simulation**: Adds `--include-bridge-tax` to emulate the NumPy <-> Torch conversion costs inherent in the production `eval.py` loop.
5. **Exact Condition Matching**: Fixed the `dummy_obs` shape ([batch, obs_dim]) to match the single-step conditioning logic used in production, preventing shape-mismatch errors during model initialization.
6. **Dependency Isolation (Mocking)**: Implemented an internal mock for `minari` and dataset dependencies. This ensures the benchmark can load and run trained models via `pickle` even in environments where dataset libraries are missing (fixing the typical `ModuleNotFoundError`).

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
  --device cuda \
  --batch-size 4 \
  --horizon 8 \
  --steps 10 \
  --solver-spec legacy_euler,torchdiffeq:euler,torchdiffeq:dopri5 \
  --plot
```

### The "Real Task" Parity Configuration
If you want the benchmark to exactly mirror the timing of a single step in `eval.py` for `avoiding-d3il`, you **must** use these parameters:
- `--batch-size 4`: The `plan` dict in `config/avoiding-d3il.py` allocates exactly 4 parallel rollouts during evaluation.
- `--horizon 8`: The U-Net output length for the sequence.
- `--steps 10` (or `20`): Set this exactly to the `flow_steps_v3` or `ode_inference_steps` value in your config model block.

**Why does Batch Size matter so much for GPU?**
If you casually set `--batch-size 64` to "stress test" the hardware, you will get significantly better `avg_ms` per-sample. GPUs are designed to process massive blocks of data all at once.
However, in your *actual* eval script, the batch size is "choked" at `4`. This means the GPU spends most of its time waiting for Python commands ("Dispatch Overhead") rather than doing math.
**Rule of thumb**: To see how the code will perform in production, test at `--batch-size 4`. To test the theoretical limit of your GPU, test at `--batch-size 128`.

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

| Argument | Default | Description |
|---|---|---|
| `--vf-mode` | `spiral` | `flow_matcher` (Real U-Net), `neural` (1.5M MLP), or `spiral` (Analytic Math). |
| `--loadbase` | `logs` | The root directory where model checkpoints are saved. |
| `--dataset` | `avoiding-d3il` | The name of the environment/dataset inside the logbase. |
| `--diffusion-loadpath` | `""` | **Required if `flow_matcher`**. Experiment string, e.g. `flow_matching/H8_K20_...` |
| `--diffusion-seed` | `0` | The run seed identifier sub-folder. |
| `--horizon` | `128` | The sequence length for the U-Net inputs. |
| `--batch-size` | `128` | Number of samples processed in parallel during execution. |
| `--steps` | `20` | Fixed steps taken in a single iteration. |
| `--n-trials` | `50` | Number of solver repetitions for the final metric aggregation. |
| `--t0`, `--t1` | `0.0`, `1.0` | The overall timeframe for integration bounds. |
| `--device` | `cpu` | Hardware context variable (`cpu` or `cuda`). |
| `--include-bridge-tax` | `False` | Moves data between NumPy<->Torch inside the timer to match `eval.py` latencies exactly. |
| `--plot` | `False` | Generates 6 precise timing characteristic bar charts in `benchmark_outputs_v2`. |

---

## 4) Troubleshooting & Understanding Chunked Overheads

If you test `flow_matcher` mode with `torchdiffeq:dopri5`, you will likely notice the inference time skyrockets. 
This is because `p_sample_loop` breaks integration recursively into chunks to support constraints later on:
1. The solver must perform initial error derivations from scratch on the very first sub-step.
2. Breaking a $1.0$-length journey into 20 small $0.05$ increments eliminates the step-size growth advantage Dopri5 is known for.

**Takeaway:** If you plan on interrupting the solver 20 times to project safe-states against constraints, you should almost certainly stick with `legacy_euler` or a lightweight fixed step like `torchdiffeq:euler` or `rk4`. Adaptive solvers simply don't have enough geometric leeway to accelerate inside tiny evaluation chunks.

---

## 5) Understanding the Metric: Environment Steps vs ODE Steps

When comparing results with `eval.py`, remember:

- **`eval.py` Metric**: `Average computation time per step` = The time for **one full Robot step**. This is what the benchmark's `avg_ms` or `p50_ms` measures.
- **`benchmark_v2.py` Metric**: `p50_ms` = The time for **one full integration journey** (e.g. 20 Euler steps). 
- **Individual ODE Step**: To calculate the cost of a single Euler transition, simply divide the benchmark's `p50_ms` by 20.

> [!TIP]
> Use `--include-bridge-tax` on a GPU to see exactly how much of your "step time" is actually just moving data across the PCIe bus!
