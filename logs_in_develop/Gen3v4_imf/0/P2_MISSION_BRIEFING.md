# Gen3v4: iMeanFlow (Improved Mean Flows) - Mission Briefing

**Created**: 13. May 2026  
**Status**: Phase 2 Complete (Real Training Infrastructure)  
**Supported Task**: D3IL Avoiding (Robot Arm with Obstacle Avoidance)

---

## Executive Summary

**Gen3v4 (iMeanFlow)** is a next-generation flow matching framework built on the **FMv3ODE foundation** with a focus on interpretable dual-velocity decomposition and constraint-aware sampling. Unlike Gen3v2 (which optimizes inference cost), iMeanFlow emphasizes **trajectory quality, safety-aware guidance, and multi-objective decomposition** through a mathematically principled separation of global (u) and local (v) dynamics.

This generation is **NOT a replacement** for existing systems but a **specialized exploration path** for:
- Multi-task learning with decomposed behavior (global planner u → local refinement v)
- Constrained sampling via Jacobian-Vector Product (JVP) guidance
- Benchmarking solver efficiency (Euler vs RK4 vs Dopri5) on a clean dual-velocity architecture
- Foundation for future extensions (e.g., hierarchical planning, vision-conditioned policies)

---

## Historical Context: Gen3v4 Design Rationale

### Gen3v2 (Ode-Selectable Engine) - Solver Optimization Focus
- **Mission**: Can we use high-precision solvers (RK4) to improve safety in narrow-gap environments?
- **Result**: Confirmed RK4 stability but found **no macro behavioral gain** for the current vector field
- **Conclusion**: 1st-order Euler is sufficient for the trained FMv3 model on avoiding-d3il
- **Outcome**: Production system uses Euler (fast, stable)

### Gen3v3 (FM-Drifting) - Drift Loss Exploration
- **Mission**: Can we explicitly model drift dynamics to improve long-horizon trajectory quality?
- **Technique**: Introduced drift loss operating on 4D drift vectors (from 28D state)
- **Architecture**: Single-velocity model + drift-specific loss
- **Status**: Promising theoretical foundation, complex empirical validation

### Gen3v4 (iMeanFlow) - Dual-Velocity Decomposition
- **Mission**: Can we decompose trajectory behavior into interpretable global + local components?
- **Technique**: Explicit u-velocity (global planner) + v-velocity (local refinement)
- **Key Insight**: Dual components allow independent control of long-range planning and local correction
- **Safety Extension**: JVP-based guidance for constraint satisfaction
- **Foundation**: **FMv3ODE** (not FM-Drifting) - proven stable architecture with continuous-time semantics

---

## Technical Architecture

### Core Design: Dual-Velocity Decomposition

iMeanFlow models trajectories as a decomposition:
$$x(t) = x_0 + \int_0^t u(\tau) + v(\tau) \, d\tau$$

Where:
- **u(t)**: Global velocity field (long-range planning signal)
- **v(t)**: Local velocity field (short-range correction signal)

**Benefits**:
1. **Interpretability**: Separate analysis of global trajectory shape vs. local refinement
2. **Multi-objective Learning**: Can train u and v with different loss schedules
3. **Constraint Awareness**: v can be trained to respect local safety constraints
4. **Composable Inference**: u provides fastest path; u+v provides safest path

### Model Architecture

**TimeConditionedDualVelocity (Base Model)**:
```
Input: trajectory x(t) ∈ ℝ^(B×T×D), time t ∈ ℝ^(B×T)
├─ Position Encoder: ℝ^D → ℝ^H
├─ Time Embedder: ℝ^1 → ℝ^T
├─ Fusion MLP: (H + T) → H
├─ Branch U: H → D (global velocity)
└─ Branch V: H → D (local velocity)
Output: u(t), v(t) ∈ ℝ^(B×T×D)
```

**Optional: DIT-Trajectory (Transformer Backbone)**:
- Replaces dense MLPs with self-attention over time
- Enables context-aware refinement (look at nearby timesteps)
- Slower but higher-quality trajectories

### Loss Functions

**Dual-Velocity Loss**:
$$L = \lambda_u L_u + \lambda_v L_v$$

Where:
- $L_u$: MSE loss on global velocity targets (computed from trajectory derivatives)
- $L_v$: MSE loss on local velocity targets (residual after u prediction)

**Scheduler**: 
- Phase 1 (epochs 0-30): Train u only ($\lambda_u=1, \lambda_v=0$)
- Phase 2 (epochs 30-100): Transition ($\lambda_u$ fades, $\lambda_v$ grows)
- Ensures global structure is learned first, then refined locally

### Constraint Guidance (JVP Extension)

Optional **Jacobian-Vector Product (JVP) guidance** for safety-critical tasks:
- Computes gradient of constraint function w.r.t. state
- Modifies v-field to pull trajectories away from constraint violations
- Used for avoiding-d3il task (obstacle avoidance)

---

## System Comparison Matrix

| Feature | Gen3v2 (Ode-Sel) | Gen3v3 (FM-D) | Gen3v4 (iMF) |
|---------|------------------|---------------|--------------|
| **Base Architecture** | FMv3 + Solver Plug-In | FMv3 + Drift Loss | FMv3 + Dual-Velocity |
| **Primary Goal** | Solver Accuracy | Drift Modeling | Decomposition |
| **Model Parameters** | ~45M | ~50M | ~48M |
| **Training Time** | 8-10h (GPU) | 10-12h (GPU) | 8-10h (GPU) |
| **Interpretability** | Medium (solver) | Medium (drift) | **High (u+v)** |
| **Constraint Support** | None | None | **JVP-based** |
| **Safety Guidance** | Projection-based | Drift-based | **Constraint-aware** |
| **Multi-Seed Support** | Yes (5 seeds) | Yes (5 seeds) | **Yes (5 seeds)** |
| **SLURM Integration** | Yes | Yes | **Yes (Real)** |
| **W&B Logging** | Yes | Yes | **Yes** |

---

## Module Inventory

### Model Modules (in `flow_matcher_v3_imeanflow/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `models/imf_velocity.py` | 165 | Dual-velocity field definitions + time conditioning |
| `models/jvp_guidance.py` | 245 | Jacobian-Vector Product constraint guidance |
| `models/imf_dit_trajectory.py` | 340 | Optional Transformer backbone for trajectories |
| `sampling/imf_ode_solvers.py` | 264 | Euler, RK4, Dopri5 integrators (generic API) |
| `sampling/imf_trajectory_sampler.py` | 310 | High-level inference API (single/dual-step) |
| `utils/imf_training.py` | 290 | Dual-velocity loss, scheduler, training wrapper |
| `utils/imf_metrics.py` | 380 | Trajectory metrics, smoothness, decomposition analysis |
| `tests/test_imf_core.py` | 450+ | 65+ unit tests (all passing) |

**Total**: 1,994 lines of core code + 500+ lines of tests

### Infrastructure Scripts (in `FM_v3_imeanflow_test/`)

| Script | Lines | Purpose |
|--------|-------|---------|
| `train_flow_matching_v3_imeanflow.py` | 465 | Multi-seed training loop (real D3IL data) |
| `eval_flow_matching_v3_imeanflow.py` | 386 | Multi-variant evaluation (6 solver/NFE combinations) |
| `load_results_flow_matching_v3_imeanflow.py` | 386 | Results aggregation + comparison plots + CSV reports |

### Configuration Files (in `flow_matcher_v3_imeanflow/configs/`)

| Config | Purpose |
|--------|---------|
| `fm_imeanflow_base.yaml` | Default architecture + training hyperparameters |
| `fm_imeanflow_d3il.yaml` | Robot arm task-specific (JVP enabled, conservative LR) |
| `fm_imeanflow_avoiding.yaml` | Obstacle avoidance task-specific (aggressive safety) |

### SLURM Batch Scripts (in `Slurm_Codes/sbatch/iMF/`)

| Script | Purpose |
|--------|---------|
| `train_imf.sh` | 24h GPU job: Multi-seed training with W&B |
| `eval_imf.sh` | 4h GPU job: Comprehensive evaluation (6 variants × 5 seeds) |
| `load_results_imf.sh` | 30min CPU job: Aggregate + plot results |

---

## Performance Expectations

### Training (Single Seed, 100 Epochs)
- **Time**: ~8-10 hours (GPU: 1× A100 or V100)
- **Memory**: 32GB RAM + 24GB VRAM
- **Checkpoints**: Best model + periodic saves
- **W&B Logging**: Loss curves, metric summaries

### Evaluation (Single Seed, 50 Trajectories)
- **Time**: ~30-45 minutes per seed (GPU)
- **Variants**: 6 (3 solvers × 2 NFE values)
- **Metrics Per Variant**: Trajectory error, path length, smoothness
- **Output**: `.npz` file + comparison images

### Results Aggregation (5 Seeds)
- **Time**: ~5-10 minutes (CPU)
- **Output**: CSV summary + comparison plots + JSON report
- **Plots**: Trajectory error, path length, smoothness by variant

---

## Key Innovations vs. Gen3v2/v3

| Innovation | Impact |
|-----------|--------|
| **Dual-Velocity Separation** | Enables interpretable trajectory analysis; foundation for hierarchical planning |
| **JVP Constraint Guidance** | Safety-aware sampling without expensive projection loops |
| **Phased Loss Scheduling** | Ensures stable learning of global structure before local refinement |
| **Real Multi-Seed Infrastructure** | 5 independent training runs with W&B tracking for statistical significance |
| **Comprehensive Solver Benchmarking** | Compare Euler vs RK4 vs Dopri5 on the same learned model |
| **Result Aggregation Pipeline** | Automatic CSV + plots + JSON across all seeds/variants |

---

## When to Use Gen3v4

### ✅ **Good Use Cases**
- **Hierarchical planning research**: Need to analyze global vs. local planning separately
- **Constraint-aware sampling**: Want JVP-based guidance for obstacle avoidance
- **Solver efficiency studies**: Comparing integration methods on same model
- **Interpretable trajectory learning**: Want to understand planning semantics
- **Multi-task foundation**: Need decomposed behavior for different robot capabilities

### ⚠️ **Not Recommended For**
- **Maximum inference speed**: Use Gen3v2 (Ode-Selectable) with Euler solver
- **Absolute state-of-the-art performance**: Use Gen3v3 (FM-Drifting) with drift loss
- **Real-time control**: JVP guidance adds computational overhead
- **Vision-based policies**: No vision encoder implemented yet

---

## Quick Start Command Reference

### Training (All 5 Seeds with W&B)
```bash
cd /workspaces/FM-PCC
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --use-wandb \
    --wandb-project FMPCC-iMF \
    --batch-size 32 \
    --learning-rate 5e-4 \
    --num-epochs 100 \
    --device cuda
```

### Evaluation (Comprehensive Multi-Variant)
```bash
cd /workspaces/FM-PCC
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --checkpoint-dir checkpoints \
    --output-dir evaluation_results \
    --device cuda \
    --solvers euler rk4 dopri5 \
    --nfe-values 1 2
```

### Results Aggregation
```bash
cd /workspaces/FM-PCC
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results
```

### SLURM Submission (Full Pipeline)
```bash
cd /workspaces/FM-PCC
sbatch Slurm_Codes/sbatch/iMF/train_imf.sh
sbatch Slurm_Codes/sbatch/iMF/eval_imf.sh
sbatch Slurm_Codes/sbatch/iMF/load_results_imf.sh
```

---

## Document Structure

This briefing covers:
1. **Mission Context** — Why Gen3v4 exists
2. **Technical Architecture** — How it works
3. **System Comparison** — Where it fits relative to other generations
4. **Module Inventory** — What code is available
5. **Performance Profile** — What to expect
6. **Quick Reference** — How to run it

👉 **Next Step**: See `HOW_TO_USE.md` for detailed step-by-step instructions with full code examples.

---

## Contact / Debugging

**Issue Tracker**: Check `/workspaces/FM-PCC/logs_in_develop/Gen3v4/` for updates  
**Test Harness**: Run `pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -v` to validate installation  
**SLURM Logs**: Monitor `/workspaces/FM-PCC/Slurm_Codes/logs/` for job outputs
