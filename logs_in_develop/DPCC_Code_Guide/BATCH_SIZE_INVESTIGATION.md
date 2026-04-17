# Investigation Note: The `batch_size` Parameter in DPCC / FMPCC

This document clarifies the role, default values, and architectural impact of the `batch_size` parameter within the Flow Matching and Diffusion planning pipelines.

## 1. What is the `batch_size`?

In terms of **Inference / Deployment** (the `plan` stage), the `batch_size` does NOT refer to training data blocks. Instead, it represents the **Number of Parallel Candidate Trajectories**.

At every single environment step (the robot's "thought" cycle):
1. The **Policy** initializes $N$ parallel noise tensors (where $N = batch\_size$).
2. The **Flow Matcher / ODE Solver** integrates all $N$ candidates simultaneously on the GPU.
3. The **Selection Logic** evaluates all $N$ trajectories and chooses the "best" one based on the selection strategy (e.g., `temporal_consistency` or `minimum_projection_cost`).
4. The robot executes the first action of the **winning trajectory**.

> [!NOTE]
> Increasing the `batch_size` improves the safety and optimality of the robot (more candidates to choose from) but increases the GPU calculation load.

---

## 2. Default Configuration Logic

The value currently set in `config/avoiding-d3il.py` is **`4`**, but a **Deep Analysis of the Evaluation Phase and Gen1 documentation** confirms the intended standard is **`20`**.

### Where is it defined?
The primary source of truth is the configuration file:
*   **Location**: [config/avoiding-d3il.py](file:///workspaces/FM-PCC/config/avoiding-d3il.py)
*   **Lines**: 
    - `plan` configuration block (line ~335)
    - `plan_fm` configuration block (line ~368)

```python
    'plan': {
        'policy': 'sampling.Policy',
        'max_episode_length': 200,
        'batch_size': 4,        # <--- The default set point for planning
        ...
    },
```

### How is it passed to the code?
The evaluation script [scripts/eval.py](file:///workspaces/FM-PCC/scripts/eval.py) parses this config and passes it directly to the policy's call method:

```python
# scripts/eval.py: Line 231
action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, ...)
```

The policy then mirrors this batch size to the model and the reward/return tensors in [diffuser/sampling/policies.py](file:///workspaces/FM-PCC/diffuser/sampling/policies.py).

---

## 3. The "Deep Analysis" Correction: Why 20?

There is a documented discrepancy between the current code (`batch_size: 4`) and the intended design documented in [Gen1_Debugging.md](file:///workspaces/FM-PCC/logs_in_develop/gen1_output_and_debugging/Gen1_Debugging.md).

### 3.1 The "K20" Terminology Fusion
In early versions (Gen1), the parameter **`K20`** (which now refers to 20 ODE steps) was explicitly defined as **20 Trajectory Samples** (candidates).
- **Intended Search Breadth**: Robust MPC requires 20+ candidates to find clear paths in "hard" obstacle scenarios.
- **Modern Interpretation**: $K=20$ is now used for math precision (Number of Function Evaluations), while `batch_size=4` is a restricted setting likely used for debug speed.

### 3.2 Recommendation for Performance Audits
For a high-fidelity audit of ODE solvers, you should set `batch_size: 20` to mirror the original DPCC search breadth. This ensures the solver is being tested under the "production-grade" workload where selecting the best of 20 paths is the priority.

---

## 4. Why is it *currently* set to 4?

The choice of `4` in the current config file is a compromise for **Debug Latency**:
1. **Diversity**: While 4 is better than 1, it only provides a fraction of the search coverage of 20.
2. **GPU Efficiency**: On many GPUs, batches of 4-16 are "free," but 20 starts to push the hardware closer to the compute-bound regime.

> [!IMPORTANT]
> If you are experiencing high collision rates on "hard" tasks, the first fix should be reverting to the **20 candidate standard**.
